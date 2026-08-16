from pathlib import Path

import numpy as np
import polars as pl
from rosbags.typesys.store import Typestore

import aflux_ros
from aflux_ros import ArrayNode, LeafNode, ListNode


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
    def test_polars_ignores_explicit_dtype_in_favor_of_numpy_derived_value(self):
        messages = [np.array([1, 2]), np.array([3, 4])]

        explicit_dtype = pl.Array(pl.Float64, 2)
        ser = pl.Series(messages, dtype=explicit_dtype)
        assert ser.dtype != explicit_dtype, "Polars has fixed issue."

        explicit_dtype = pl.List(pl.Int64)
        ser = pl.Series(messages, dtype=explicit_dtype)
        assert ser.dtype != explicit_dtype, "Polars has fixed issue."

    def test_array(self):
        messages = [np.array([1, 2]), np.array([3, 4])]

        array_node = ArrayNode(LeafNode("float64"), 2)
        ser = aflux_ros.convert_messages_into_series(array_node, messages)
        assert ser.dtype == array_node.to_dataframe_dtype()

    def test_list(self):
        messages = [np.array([1, 2]), np.array([3, 4])]

        list_node = ListNode(LeafNode("float64"))
        ser = aflux_ros.convert_messages_into_series(list_node, messages)
        assert ser.dtype == list_node.to_dataframe_dtype()
