from pathlib import Path

from rosbags.interfaces.typing import Typesdict
from rosbags.typesys import get_types_from_msg
from rosbags.typesys.store import Typestore


def read_message_schema_dir(message_schema_dir: str | Path) -> dict[str, str]:
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
    add_types: Typesdict = {}
    for message_type, message_schema in message_schema_map.items():
        add_types.update(get_types_from_msg(message_schema, message_type))
    typestore.register(add_types)


def register_message_schema_dir(
    typestore: Typestore,
    message_schema_dir: str | Path,
) -> None:
    message_schema_map = read_message_schema_dir(message_schema_dir)
    register_message_schema_map(typestore, message_schema_map)
