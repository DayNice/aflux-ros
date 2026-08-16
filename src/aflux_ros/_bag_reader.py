from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import polars as pl
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore

from ._message_node import (
    StructNode,
    parse_message_type_into_node,
)
from ._types import TopicInfo


class BagReader:
    def __init__(
        self,
        bag_dir: str | Path | Iterable[str | Path],
        *,
        fallback_typestore: Typestore | None = None,
    ):
        if isinstance(bag_dir, str | Path):
            self._bag_dirs = [Path(bag_dir)]
        else:
            self._bag_dirs = [Path(p) for p in bag_dir]

        if fallback_typestore is None:
            self._fallback_typestore = get_typestore(Stores.LATEST)
        else:
            self._fallback_typestore = fallback_typestore

        self._reader = AnyReader(self._bag_dirs, default_typestore=self._fallback_typestore)
        self._reader.open()

    @cached_property
    def topic_info_map(self) -> dict[str, TopicInfo]:
        topic_info_map: dict[str, TopicInfo] = {}
        for topic, raw_topic_info in self._reader.topics.items():
            if raw_topic_info.msgtype is None:
                msg = f"Topic with multiple message types is unsupported: {topic!r}"
                raise ValueError(msg)
            topic_info = TopicInfo(
                topic=topic,
                message_type=raw_topic_info.msgtype,
                num_messages=raw_topic_info.msgcount,
            )
            topic_info_map[topic] = topic_info
        return topic_info_map

    def get_message_node(self, topic: str) -> StructNode:
        topic_info = self.topic_info_map[topic]
        return parse_message_type_into_node(self._reader.typestore, topic_info.message_type)

    def get_raw_bytes(self, topic: str) -> Iterator[tuple[int, str, bytes]]:
        connections = self._reader.topics[topic].connections
        for connection, timestamp, rawdata in self._reader.messages(connections):
            yield timestamp, connection.msgtype, rawdata

    def get_messages(self, topic: str) -> Iterator[tuple[int, Any]]:
        for timestamp, msgtype, rawdata in self.get_raw_bytes(topic):
            message = self._reader.deserialize(rawdata, msgtype)
            yield timestamp, message

    def dump_messages(self, topic: str) -> Iterator[tuple[int, dict[str, Any]]]:
        node = self.get_message_node(topic)
        for timestamp, message in self.get_messages(topic):
            yield timestamp, node.dump_message(message)

    def get_message_dataframe(self, topic: str) -> pl.DataFrame:
        node = self.get_message_node(topic)
        schema = pl.Schema(
            {
                "timestamp": pl.Int64,
                topic: node.to_dataframe_dtype(),
            }
        )

        records: list[dict[str, Any]] = []
        for timestamp, message in self.dump_messages(topic):
            record = {"timestamp": timestamp, topic: message}
            records.append(record)

        return pl.from_dicts(records, schema=schema)

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
