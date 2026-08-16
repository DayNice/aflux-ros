from pathlib import Path

import numpy as np
import polars as pl
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from aflux_ros import BagReader, register_message_schema_map


@pytest.fixture
def tmp_rosbag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bag_dir = tmp_path_factory.mktemp("rosbag") / "test_bag"

    typestore = get_typestore(Stores.LATEST)
    register_message_schema_map(
        typestore,
        {
            "test_msgs/msg/Inner": "float64 value\nfloat64[] values\n",
            "test_msgs/msg/Outer": (
                "float64[] scalars\n"
                "float64[2] fixed_scalars\n"
                "test_msgs/msg/Inner[] inners\n"
                "test_msgs/msg/Inner[2] fixed_inners\n"
            ),
        },
    )
    inner_type = typestore.types["test_msgs/msg/Inner"]
    outer_type = typestore.types["test_msgs/msg/Outer"]

    nested_message_1 = outer_type(
        scalars=np.array([1.0, 2.0]),
        fixed_scalars=np.array([3.0, 4.0]),
        inners=[
            inner_type(value=5.0, values=np.array([5.1, 5.2])),
        ],
        fixed_inners=[
            inner_type(value=5.0, values=np.array([5.1, 5.2])),
            inner_type(value=6.0, values=np.array([6.1, 6.2])),
        ],
    )
    nested_message_2 = outer_type(
        scalars=np.array([11.0, 12.0]),
        fixed_scalars=np.array([13.0, 14.0]),
        inners=[
            inner_type(value=15.0, values=np.array([15.1, 15.2])),
            inner_type(value=16.0, values=np.array([16.1, 16.2])),
        ],
        fixed_inners=[
            inner_type(value=15.0, values=np.array([15.1, 15.2])),
            inner_type(value=16.0, values=np.array([16.1, 16.2])),
        ],
    )

    with Writer(bag_dir, version=8) as writer:
        conn = writer.add_connection("/simple_topic", "std_msgs/msg/Float64", typestore=typestore)
        for i in range(10):
            msg = typestore.types["std_msgs/msg/Float64"](data=float(i))
            rawdata = typestore.serialize_cdr(msg, "std_msgs/msg/Float64")
            writer.write(conn, i * 1_000_000_000, rawdata)

        nested_conn = writer.add_connection("/nested_topic", "test_msgs/msg/Outer", typestore=typestore)
        rawdata = typestore.serialize_cdr(nested_message_1, "test_msgs/msg/Outer")
        writer.write(nested_conn, 0, rawdata)
        rawdata = typestore.serialize_cdr(nested_message_2, "test_msgs/msg/Outer")
        writer.write(nested_conn, 1, rawdata)

    return bag_dir


class TestBagReader:
    def test_topic_info_map(self, tmp_rosbag: Path) -> None:
        with BagReader(tmp_rosbag) as reader:
            info = reader.topic_info_map["/simple_topic"]

            assert info.topic == "/simple_topic"
            assert info.message_type == "std_msgs/msg/Float64"
            assert info.num_messages == 10

    def test_get_messages(self, tmp_rosbag: Path) -> None:
        with BagReader(tmp_rosbag) as reader:
            message_tuples = list(reader.get_messages("/simple_topic"))

            assert len(message_tuples) == 10
            for i, (timestamp, message) in enumerate(message_tuples):
                assert timestamp == i * 1_000_000_000
                assert message.data == float(i)

    def test_get_message_dataframe(self, tmp_rosbag: Path) -> None:
        with BagReader(tmp_rosbag) as reader:
            df_simple = reader.get_message_dataframe("/simple_topic")

            assert len(df_simple) == 10
            assert df_simple.schema["timestamp"] == pl.Int64
            assert df_simple.schema["/simple_topic"] == pl.Struct({"data": pl.Float64})

            timestamps = df_simple["timestamp"].to_list()
            values = df_simple["/simple_topic"].struct.field("data").to_list()

            for i in range(10):
                assert timestamps[i] == i * 1_000_000_000
                assert values[i] == float(i)

            df_nested = reader.get_message_dataframe("/nested_topic")

        assert df_nested.schema["/nested_topic"] == pl.Struct(
            {
                "scalars": pl.List(pl.Float64),
                "fixed_scalars": pl.Array(pl.Float64, 2),
                "inners": pl.List(
                    pl.Struct(
                        {
                            "value": pl.Float64,
                            "values": pl.List(pl.Float64),
                        }
                    )
                ),
                "fixed_inners": pl.Array(
                    pl.Struct(
                        {
                            "value": pl.Float64,
                            "values": pl.List(pl.Float64),
                        }
                    ),
                    2,
                ),
            }
        )
        assert df_nested["/nested_topic"].to_list() == [
            {
                "scalars": [1.0, 2.0],
                "fixed_scalars": [3.0, 4.0],
                "inners": [
                    {"value": 5.0, "values": [5.1, 5.2]},
                ],
                "fixed_inners": [
                    {"value": 5.0, "values": [5.1, 5.2]},
                    {"value": 6.0, "values": [6.1, 6.2]},
                ],
            },
            {
                "scalars": [11.0, 12.0],
                "fixed_scalars": [13.0, 14.0],
                "inners": [
                    {"value": 15.0, "values": [15.1, 15.2]},
                    {"value": 16.0, "values": [16.1, 16.2]},
                ],
                "fixed_inners": [
                    {"value": 15.0, "values": [15.1, 15.2]},
                    {"value": 16.0, "values": [16.1, 16.2]},
                ],
            },
        ]

    def test_multiple_bag_dirs(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        typestore = get_typestore(Stores.LATEST)
        bag1 = tmp_path_factory.mktemp("rosbag1") / "bag1"
        bag2 = tmp_path_factory.mktemp("rosbag2") / "bag2"

        with Writer(bag1, version=8) as writer:
            conn = writer.add_connection("/topic1", "std_msgs/msg/Float64", typestore=typestore)
            msg = typestore.types["std_msgs/msg/Float64"](data=1.0)
            writer.write(conn, 100, typestore.serialize_cdr(msg, "std_msgs/msg/Float64"))

        with Writer(bag2, version=8) as writer:
            conn = writer.add_connection("/topic2", "std_msgs/msg/Float64", typestore=typestore)
            msg = typestore.types["std_msgs/msg/Float64"](data=2.0)
            writer.write(conn, 200, typestore.serialize_cdr(msg, "std_msgs/msg/Float64"))

        with BagReader([bag1, bag2]) as reader:
            assert "/topic1" in reader.topic_info_map
            assert "/topic2" in reader.topic_info_map
            assert reader.topic_info_map["/topic1"].num_messages == 1
            assert reader.topic_info_map["/topic2"].num_messages == 1
