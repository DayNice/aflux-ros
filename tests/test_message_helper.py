from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from rosbags.typesys.store import Typestore

import aflux_ros
from aflux_ros import ArrayNode, LeafNode, ListNode, StructNode


class TestMessageHelper:
    def test_read_message_schema_dir(self, tmp_path: Path) -> None:
        msg_dir = tmp_path / "my_msgs" / "msg"
        msg_dir.mkdir(parents=True)
        (msg_dir / "Custom.msg").write_text("float64 data\n", encoding="utf-8")
        (msg_dir / "Another.msg").write_text("int32 id\n", encoding="utf-8")

        schema_map = aflux_ros.read_message_schema_dir(tmp_path)
        assert schema_map == {
            "my_msgs/msg/Custom": "float64 data\n",
            "my_msgs/msg/Another": "int32 id\n",
        }

    def test_register_message_schema_map(self) -> None:
        typestore = Typestore()
        schema_map = {"my_msgs/msg/Point": "float64 x\nfloat64 y\n"}

        assert "my_msgs/msg/Point" not in typestore.fielddefs

        aflux_ros.register_message_schema_map(typestore, schema_map)

        assert "my_msgs/msg/Point" in typestore.fielddefs
        fields = typestore.fielddefs["my_msgs/msg/Point"][1]
        assert len(fields) == 2
        assert fields[0][0] == "x"
        assert fields[1][0] == "y"


class TestConvertMessagesIntoSeries:
    def test_empty_messages_preserve_the_node_dtype(self) -> None:
        node = ListNode(LeafNode("float64"))

        ser = aflux_ros.convert_messages_into_series(node, [])

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == []

    def test_leaf(self) -> None:
        node = LeafNode("float64")

        ser = aflux_ros.convert_messages_into_series(node, [1.0, 2.0])

        assert ser.dtype == pl.Float64
        assert ser.to_list() == [1.0, 2.0]

    def test_struct(self) -> None:
        node = StructNode(
            "test_msgs/msg/Point",
            {"x": LeafNode("float64"), "y": LeafNode("float64")},
        )
        messages = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]

        ser = aflux_ros.convert_messages_into_series(node, messages)

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == [
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0},
        ]

    def test_array(self) -> None:
        node = ArrayNode(LeafNode("float64"), 2)
        messages = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        ser = aflux_ros.convert_messages_into_series(node, messages)

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_list(self) -> None:
        node = ListNode(LeafNode("float64"))
        messages = [np.array([1.0]), np.array([2.0, 3.0])]

        ser = aflux_ros.convert_messages_into_series(node, messages)

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == [[1.0], [2.0, 3.0]]

    def test_array_of_structs_with_nested_lists(self) -> None:
        inner_node = StructNode(
            "test_msgs/msg/Inner",
            {"value": LeafNode("float64"), "values": ListNode(LeafNode("float64"))},
        )
        node = ArrayNode(inner_node, 2)
        messages = [
            [
                {"value": 1.0, "values": np.array([1.1])},
                {"value": 2.0, "values": np.array([2.1, 2.2])},
            ],
            [
                {"value": 3.0, "values": np.array([3.1, 3.2])},
                {"value": 4.0, "values": np.array([4.1])},
            ],
        ]

        ser = aflux_ros.convert_messages_into_series(node, messages)

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == [
            [
                {"value": 1.0, "values": [1.1]},
                {"value": 2.0, "values": [2.1, 2.2]},
            ],
            [
                {"value": 3.0, "values": [3.1, 3.2]},
                {"value": 4.0, "values": [4.1]},
            ],
        ]

    def test_list_of_structs_with_variable_lengths_and_empty_rows(self) -> None:
        inner_node = StructNode(
            "test_msgs/msg/Inner",
            {"value": LeafNode("float64"), "values": ListNode(LeafNode("float64"))},
        )
        node = ListNode(inner_node)
        messages = [
            [],
            [{"value": 1.0, "values": np.array([1.1])}],
            [
                {"value": 2.0, "values": np.array([2.1, 2.2])},
                {"value": 3.0, "values": np.array([3.1])},
            ],
        ]

        ser = aflux_ros.convert_messages_into_series(node, messages)

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == [
            [],
            [{"value": 1.0, "values": [1.1]}],
            [
                {"value": 2.0, "values": [2.1, 2.2]},
                {"value": 3.0, "values": [3.1]},
            ],
        ]

    def test_raw_message_is_dumped_before_nested_conversion(self) -> None:
        inner_node = StructNode(
            "test_msgs/msg/Inner",
            {"value": LeafNode("float64"), "values": ListNode(LeafNode("float64"))},
        )
        node = StructNode("test_msgs/msg/Outer", {"inners": ListNode(inner_node)})
        messages = [
            SimpleNamespace(
                __msgtype__="test_msgs/msg/Outer",
                inners=[
                    SimpleNamespace(
                        __msgtype__="test_msgs/msg/Inner",
                        value=1.0,
                        values=np.array([1.1, 1.2]),
                    )
                ],
            )
        ]

        ser = aflux_ros.convert_messages_into_series(node, messages)

        assert ser.dtype == node.to_dataframe_dtype()
        assert ser.to_list() == [
            {"inners": [{"value": 1.0, "values": [1.1, 1.2]}]},
        ]

    def test_raw_message_type_must_match_node(self) -> None:
        node = StructNode("test_msgs/msg/Point", {"x": LeafNode("float64")})
        message = SimpleNamespace(__msgtype__="test_msgs/msg/Other", x=1.0)

        with pytest.raises(ValueError, match="Message type should match given node"):
            aflux_ros.convert_messages_into_series(node, [message])
