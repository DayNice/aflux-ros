from collections.abc import Iterable
from pathlib import Path
from typing import Any, assert_never, cast

import polars as pl
from rosbags.interfaces.typing import Typesdict
from rosbags.typesys import get_types_from_msg
from rosbags.typesys.store import Typestore

from ._message_node import ArrayNode, LeafNode, ListNode, MessageNode, StructNode


def read_message_schema_dir(message_schema_dir: str | Path) -> dict[str, str]:
    """Read all `.msg` files under a directory into a message schema map.

    Each key is the file's relative path with the `.msg` suffix removed.
    For example, `my_msgs/msg/Point.msg` becomes `my_msgs/msg/Point`.
    """
    message_schema_dir = Path(message_schema_dir)

    message_schema_map: dict[str, str] = {}
    for message_schema_file in message_schema_dir.rglob("*.msg"):
        message_type = message_schema_file.relative_to(message_schema_dir).as_posix()
        message_type = message_type.removesuffix(".msg")
        message_schema = message_schema_file.read_text(encoding="utf-8")
        message_schema_map[message_type] = message_schema

    return message_schema_map


def register_message_schema_map(
    typestore: Typestore,
    message_schema_map: dict[str, str],
) -> None:
    """Register a message schema map to a typestore."""
    add_types: Typesdict = {}
    for message_type, message_schema in message_schema_map.items():
        add_types.update(get_types_from_msg(message_schema, message_type))
    typestore.register(add_types)


def register_message_schema_dir(
    typestore: Typestore,
    message_schema_dir: str | Path,
) -> None:
    """Read and register all `.msg` files under a directory."""
    message_schema_map = read_message_schema_dir(message_schema_dir)
    register_message_schema_map(typestore, message_schema_map)


def convert_messages_into_series(
    message_node: MessageNode,
    messages: Iterable[Any],
    *,
    name: str = "",
) -> pl.Series:
    """Convert messages into a Polars series."""
    dumped_messages = []
    for message in messages:
        msgtype = getattr(message, "__msgtype__", None)

        if msgtype is None:
            dumped_message = message
        else:
            assert isinstance(msgtype, str), "msgtype should be an instance of str"
            if msgtype != str(message_node):
                msg = f"Message type should match given node: {message_node=!r}, {msgtype=!r}"
                raise ValueError(msg)

            dumped_message = message_node.dump_message(message)

        dumped_messages.append(dumped_message)

    if len(dumped_messages) == 0:
        return pl.Series(name, [], dtype=message_node.to_dataframe_dtype())

    match message_node:
        case LeafNode():
            return pl.Series(name, dumped_messages, dtype=message_node.to_dataframe_dtype())
        case StructNode():
            return _convert_struct_messages_into_series(message_node, dumped_messages, name=name)
        case ArrayNode():
            return _convert_array_messages_into_series(message_node, dumped_messages, name=name)
        case ListNode():
            return _convert_list_messages_into_series(message_node, dumped_messages, name=name)
        case _:
            assert_never(message_node)


def _convert_struct_messages_into_series(
    struct_node: StructNode,
    messages: list[Any],
    *,
    name: str = "",
) -> pl.Series:
    """Convert struct messages into a Polars struct series."""
    polars_dtype = struct_node.to_dataframe_dtype()

    for message in messages:
        if isinstance(message, dict):
            continue
        msg = f"Message should be an instance of dict: {struct_node=!r}, {message=!r}"
        raise ValueError(msg)
    messages = cast(list[dict[str, Any]], messages)

    field_series_map: dict[str, pl.Series] = {}
    for field_name, field_node in struct_node.field_node_map.items():
        field_messages = [message[field_name] for message in messages]

        field_series = convert_messages_into_series(
            field_node,
            field_messages,
            name=field_name,
        )
        field_series_map[field_name] = field_series

    ser = pl.DataFrame(field_series_map).to_struct(name)
    return ser.cast(polars_dtype)


def _convert_array_messages_into_series(
    array_node: ArrayNode,
    messages: list[Any],
    *,
    name: str = "",
) -> pl.Series:
    """Convert array messages into a Polars array series."""
    polars_dtype = array_node.to_dataframe_dtype()

    if isinstance(array_node.item_node, LeafNode):
        ser = pl.Series(name, messages, dtype=polars_dtype)
        # cast again due to polars ignoring explicit dtype in favor of numpy derived value
        # e.g. `messages = [np.array([1, 2]), np.array([3, 4])]`
        return ser.cast(polars_dtype)

    for message in messages:
        if isinstance(message, list):
            continue
        msg = f"Message should be an instance of list: {array_node=!r}, {message=!r}"
        raise ValueError(msg)
    messages = cast(list[list[Any]], messages)

    items = [item for message in messages for item in message]
    item_ser = convert_messages_into_series(array_node.item_node, items, name=name)
    ser = item_ser.reshape((-1, array_node.size)).alias(name)
    return ser.cast(polars_dtype)


def _convert_list_messages_into_series(
    list_node: ListNode,
    messages: list[Any],
    *,
    name: str = "",
) -> pl.Series:
    """Convert list messages into a Polars list series."""
    polars_dtype = list_node.to_dataframe_dtype()

    if isinstance(list_node.item_node, LeafNode):
        ser = pl.Series(name, messages, dtype=polars_dtype)
        # cast again due to polars ignoring explicit dtype in favor of numpy derived values
        # e.g. `messages = [np.array([1, 2]), np.array([3, 4])]`
        return ser.cast(polars_dtype)

    for message in messages:
        if isinstance(message, list):
            continue
        msg = f"Message should be an instance of list: {list_node=!r}, {message!r}"
        raise ValueError(msg)
    messages = cast(list[list[Any]], messages)

    items = [item for message in messages for item in message]
    item_series = _convert_struct_messages_into_series(list_node.item_node, items, name=name)

    num_rows = len(messages)
    df_row = pl.int_range(num_rows, eager=True).alias("row_index").to_frame()

    row_indices = []
    for i, message in enumerate(messages):
        for _ in range(len(message)):
            row_indices.append(i)
    df_item = pl.DataFrame({"item": item_series, "row_index": row_indices})
    df_item = df_item.group_by("row_index").agg("item")

    df = df_row.join(df_item, on="row_index", how="left")
    ser = df["item"].fill_null(pl.lit([]))
    return ser.cast(polars_dtype)
