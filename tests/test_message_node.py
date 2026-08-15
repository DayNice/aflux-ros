import polars as pl
import pytest
from pytest_mock import MockerFixture
from rosbags.interfaces import Nodetype

import aflux_ros
from aflux_ros import (
    ArrayNode,
    LeafNode,
    ListNode,
    StructNode,
)


class TestNodes:
    class TestLeafNode:
        def test_str(self) -> None:
            assert str(LeafNode("float64", 0)) == "float64"
            assert str(LeafNode("string", 10)) == "string<=10"

    class TestStructNode:
        def test_str(self) -> None:
            node = StructNode("geometry_msgs/msg/Point", {"x": LeafNode("float64")})
            assert str(node) == "geometry_msgs/msg/Point"

    class TestArrayNode:
        def test_str(self) -> None:
            node = ArrayNode(LeafNode("float64"), 3)
            assert str(node) == "float64[3]"

    class TestListNode:
        def test_str(self) -> None:
            assert str(ListNode(LeafNode("float64"))) == "float64[]"
            assert str(ListNode(LeafNode("float64"), 10)) == "float64[<=10]"


class TestParsing:
    @pytest.fixture
    def mock_typestore(self, mocker: MockerFixture):
        typestore = mocker.MagicMock()

        def get_msgdef(msgtype: str):
            msgdef = mocker.MagicMock()
            if msgtype == "geometry_msgs/msg/Point":
                msgdef.fields = [
                    ("x", (Nodetype.BASE, ("float64", 0))),
                    ("y", (Nodetype.BASE, ("float64", 0))),
                    ("z", (Nodetype.BASE, ("float64", 0))),
                ]
            else:
                msgdef.fields = []
            return msgdef

        typestore.get_msgdef.side_effect = get_msgdef
        return typestore

    class TestParseFieldValue:
        def test_base_node(self, mock_typestore) -> None:
            node = aflux_ros.parse_field_value_into_node(mock_typestore, (Nodetype.BASE, ("float64", 0)))
            assert isinstance(node, LeafNode)
            assert node.dtype == "float64"
            assert node.max_length == 0

        def test_name_node(self, mock_typestore) -> None:
            node = aflux_ros.parse_field_value_into_node(mock_typestore, (Nodetype.NAME, "geometry_msgs/msg/Point"))
            assert isinstance(node, StructNode)
            assert node.dtype == "geometry_msgs/msg/Point"
            assert "x" in node.field_node_map

        def test_array_node(self, mock_typestore) -> None:
            node = aflux_ros.parse_field_value_into_node(
                mock_typestore, (Nodetype.ARRAY, ((Nodetype.BASE, ("float64", 0)), 3))
            )
            assert isinstance(node, ArrayNode)
            assert node.item_node.dtype == "float64"
            assert node.size == 3

        def test_sequence_node(self, mock_typestore) -> None:
            node = aflux_ros.parse_field_value_into_node(
                mock_typestore, (Nodetype.SEQUENCE, ((Nodetype.BASE, ("uint8", 0)), 0))
            )
            assert isinstance(node, ListNode)
            assert node.item_node.dtype == "uint8"
            assert node.max_size == 0

    class TestParseMsgtype:
        def test_struct_node_creation(self, mock_typestore) -> None:
            node = aflux_ros.parse_message_type_into_node(mock_typestore, "geometry_msgs/msg/Point")
            assert isinstance(node, StructNode)
            assert node.dtype == "geometry_msgs/msg/Point"
            assert set(node.field_node_map.keys()) == {"x", "y", "z"}


class TestDumpMessage:
    def test_leaf_node(self) -> None:
        node = LeafNode("float64")
        assert node.dump_message(42.0) == 42.0

    def test_struct_node(self) -> None:
        from types import SimpleNamespace

        node = StructNode("geometry_msgs/msg/Point", {"x": LeafNode("float64"), "y": LeafNode("float64")})
        msg = SimpleNamespace(x=1.0, y=2.0)
        assert node.dump_message(msg) == {"x": 1.0, "y": 2.0}

    def test_array_node(self) -> None:
        import numpy as np

        node = ArrayNode(LeafNode("float64"), 3)
        assert node.dump_message([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

        arr = np.array([1.0, 2.0, 3.0])
        assert node.dump_message(arr) is arr

    def test_list_node(self) -> None:
        import numpy as np

        node = ListNode(LeafNode("float64"))
        assert node.dump_message([1.0, 2.0]) == [1.0, 2.0]

        arr = np.array([1.0, 2.0])
        assert node.dump_message(arr) is arr


class TestConvertIntoDataframeDtype:
    def test_leaf_node(self) -> None:
        leaf_node = LeafNode("float64")
        assert leaf_node.to_dataframe_dtype() == pl.Float64()

    def test_struct_node(self) -> None:
        leaf_node = LeafNode("float64")
        struct_node = StructNode("dummy", {"x": leaf_node, "y": leaf_node})
        assert struct_node.to_dataframe_dtype() == pl.Struct({"x": pl.Float64(), "y": pl.Float64()})

    def test_array_node(self) -> None:
        leaf_node = LeafNode("float64")
        array_node = ArrayNode(leaf_node, 5)
        assert array_node.to_dataframe_dtype() == pl.Array(pl.Float64(), 5)

    def test_list_node(self) -> None:
        leaf_node = LeafNode("float64")
        list_node = ListNode(leaf_node)
        assert list_node.to_dataframe_dtype() == pl.List(pl.Float64())
