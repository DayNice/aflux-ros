from pathlib import Path

import polars as pl
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from aflux_ros import BagReader


@pytest.fixture
def tmp_rosbag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bag_path = tmp_path_factory.mktemp("rosbag") / "test_bag"

    typestore = get_typestore(Stores.LATEST)

    with Writer(bag_path, version=8) as writer:
        conn = writer.add_connection("/test_topic", "std_msgs/msg/Float64", typestore=typestore)
        for i in range(10):
            msg = typestore.types["std_msgs/msg/Float64"](data=float(i))
            rawdata = typestore.serialize_cdr(msg, "std_msgs/msg/Float64")
            writer.write(conn, i * 1_000_000_000, rawdata)

    return bag_path


class TestBagReader:
    def test_topic_info_map(self, tmp_rosbag: Path) -> None:
        with BagReader(tmp_rosbag) as reader:
            info = reader.topic_info_map["/test_topic"]

            assert info.topic == "/test_topic"
            assert info.message_type == "std_msgs/msg/Float64"
            assert info.num_messages == 10

    def test_get_messages(self, tmp_rosbag: Path) -> None:
        with BagReader(tmp_rosbag) as reader:
            message_tuples = list(reader.get_messages("/test_topic"))

            assert len(message_tuples) == 10
            for i, (timestamp, message) in enumerate(message_tuples):
                assert timestamp == i * 1_000_000_000
                assert message.data == float(i)

    def test_get_message_dataframe(self, tmp_rosbag: Path) -> None:
        with BagReader(tmp_rosbag) as reader:
            df = reader.get_message_dataframe("/test_topic")

            assert len(df) == 10
            assert df.schema["timestamp"] == pl.Int64
            assert df.schema["/test_topic"] == pl.Struct({"data": pl.Float64})

            timestamps = df["timestamp"].to_list()
            values = df["/test_topic"].struct.field("data").to_list()

            for i in range(10):
                assert timestamps[i] == i * 1_000_000_000
                assert values[i] == float(i)
