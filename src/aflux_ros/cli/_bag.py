from cyclopts import App
from rosbags.typesys import Stores, get_typestore

import aflux_ros
from aflux_ros import BagReader

from ._parameters import InputDir, OutputFile

app = App(help="Inspect a bag.")


def _get_bag_reader(
    bag_parent_dir: InputDir,
    message_schema_dir: InputDir | None = None,
) -> BagReader:
    typestore = None
    if message_schema_dir is not None:
        typestore = get_typestore(Stores.LATEST)
        aflux_ros.register_message_schema_dir(typestore, message_schema_dir)
    return BagReader(bag_parent_dir, typestore)


@app.command
def topics(
    bag_parent_dir: InputDir,
    *,
    message_schema_dir: InputDir | None = None,
):
    with _get_bag_reader(bag_parent_dir, message_schema_dir) as bag_reader:
        topic_info_map = bag_reader.topic_info_map.copy()

    for topic_info in topic_info_map.values():
        print(topic_info.model_dump_json())


@app.command
def message_dataframe(
    bag_parent_dir: InputDir,
    topic: str,
    output_file: OutputFile,
    *,
    message_schema_dir: InputDir | None = None,
):
    with _get_bag_reader(bag_parent_dir, message_schema_dir) as bag_reader:
        df = bag_reader.get_message_dataframe(topic)
        df.write_parquet(output_file)
