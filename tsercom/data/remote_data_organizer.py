import datetime
import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy  # Used in get_interpolated_at
from functools import partial
from typing import (
    Generic,
    Optional,
    TypeVar,
)  # List is used in get_new_data

from sortedcontainers import SortedList

from tsercom.caller_id.caller_identifier import CallerIdentifier

# from tsercom.data.annotated_instance import AnnotatedInstance # Not directly used by RDO
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.util.is_running_tracker import IsRunningTracker

logger = logging.getLogger(__name__)

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class RemoteDataOrganizer(
    Generic[DataTypeT],
    RemoteDataReader[DataTypeT],
    DataTimeoutTracker.Tracked,
):
    class Client(ABC):
        @abstractmethod
        def _on_data_available(
            self,
            data_organizer: "RemoteDataOrganizer[DataTypeT]",
        ) -> None:
            raise NotImplementedError

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        caller_id: CallerIdentifier,
        client: Optional["RemoteDataOrganizer.Client"] = None,
    ) -> None:
        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__caller_id: CallerIdentifier = caller_id
        self.__client: RemoteDataOrganizer.Client | None = client
        self.__data_lock: threading.Lock = threading.Lock()
        self.__data: SortedList[DataTypeT] = SortedList(
            key=lambda item: item.timestamp,
        )
        self.__last_access: datetime.datetime = datetime.datetime.min.replace(
            tzinfo=datetime.timezone.utc,
        )
        self.__is_running: IsRunningTracker = IsRunningTracker()
        super().__init__()

    @property
    def caller_id(self) -> CallerIdentifier:
        return self.__caller_id

    def start(self) -> None:
        self.__is_running.start()

    def stop(self) -> None:
        self.__is_running.stop()

    def has_new_data(self) -> bool:
        with self.__data_lock:
            if not self.__data:
                return False
            most_recent_timestamp = self.__data[-1].timestamp
            last_access_timestamp = self.__last_access
            return most_recent_timestamp > last_access_timestamp

    def get_new_data(self) -> list[DataTypeT]:
        with self.__data_lock:
            results: list[DataTypeT] = []
            if not self.__data:
                return results

            for item in reversed(self.__data):
                if item.timestamp > self.__last_access:
                    results.append(item)
                else:
                    break
            if results:
                self.__last_access = results[0].timestamp
            return results

    def get_most_recent_data(self) -> DataTypeT | None:
        with self.__data_lock:
            if not self.__data:
                return None
            return self.__data[-1]

    def get_data_for_timestamp(
        self,
        timestamp: datetime.datetime,
    ) -> DataTypeT | None:
        with self.__data_lock:
            if not self.__data:
                return None

            if (
                timestamp.tzinfo is None
                and self.__data[0].timestamp.tzinfo is not None
            ):
                timestamp = timestamp.replace(
                    tzinfo=self.__data[0].timestamp.tzinfo,
                )

            class DummyItemForBisectSearch:
                def __init__(self, ts: datetime.datetime) -> None:
                    self.timestamp: datetime.datetime = ts

            idx: int = self.__data.bisect_right(
                DummyItemForBisectSearch(timestamp),
            )
            if idx == 0:
                return None
            return self.__data[idx - 1]

    def _on_data_ready(self, new_data: DataTypeT) -> None:
        if not isinstance(new_data, ExposedData):
            raise TypeError(
                f"Expected new_data to be an instance of ExposedData, but got {type(new_data).__name__}.",
            )
        assert (
            new_data.caller_id == self.caller_id
        ), f"Data's caller_id '{new_data.caller_id}' does not match organizer's '{self.caller_id}'"
        self.__thread_pool.submit(self.__on_data_ready_impl, new_data)

    def __on_data_ready_impl(self, new_data: DataTypeT) -> None:
        if not self.__is_running.get():
            return
        data_inserted_or_updated = False
        with self.__data_lock:
            if (
                self.__data
                and self.__data[0].timestamp.tzinfo is not None
                and new_data.timestamp.tzinfo is None
            ):
                new_data.timestamp = new_data.timestamp.replace(
                    tzinfo=self.__data[0].timestamp.tzinfo,
                )

            class DummyItemForBisectSearch:
                def __init__(self, ts: datetime.datetime) -> None:
                    self.timestamp: datetime.datetime = ts

            item_to_find = DummyItemForBisectSearch(new_data.timestamp)
            idx: int = self.__data.bisect_left(item_to_find)

            item_is_latest = (
                not self.__data
                or new_data.timestamp >= self.__data[-1].timestamp
            )

            if (
                idx < len(self.__data)
                and self.__data[idx].timestamp == new_data.timestamp
            ):
                self.__data.pop(idx)
                self.__data.add(new_data)
                data_inserted_or_updated = True
            else:
                self.__data.add(new_data)
                if item_is_latest:
                    data_inserted_or_updated = True
        if data_inserted_or_updated and self.__client is not None:
            self.__client._on_data_available(self)

    def _on_triggered(self, timeout_seconds: int) -> None:
        self.__thread_pool.submit(
            partial(self.__timeout_old_data, timeout_seconds),
        )

    def __timeout_old_data(self, timeout_seconds: int) -> None:
        if not self.__is_running.get():
            return
        reference_tz = None
        if self.__data and self.__data[0].timestamp.tzinfo is not None:
            reference_tz = self.__data[0].timestamp.tzinfo
        current_time = datetime.datetime.now(reference_tz)
        timeout_delta = datetime.timedelta(seconds=timeout_seconds)
        oldest_allowed_timestamp = current_time - timeout_delta
        with self.__data_lock:
            while (
                self.__data
                and self.__data[0].timestamp < oldest_allowed_timestamp
            ):
                self.__data.pop(0)

    def get_interpolated_at(
        self,
        timestamp: datetime.datetime,
    ) -> DataTypeT | None:
        with self.__data_lock:
            if not self.__data:
                return None
            ref_tz = self.__data[0].timestamp.tzinfo
            if ref_tz is not None and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ref_tz)
            elif ref_tz is None and timestamp.tzinfo is not None:
                logger.warning(
                    "Query timestamp timezone awareness mismatch with stored data (naive).",
                )
                timestamp = timestamp.replace(tzinfo=None)

            if timestamp <= self.__data[0].timestamp:
                return deepcopy(self.__data[0])
            if timestamp >= self.__data[-1].timestamp:
                return deepcopy(self.__data[-1])

            class DummyItemForBisectSearch:
                def __init__(self, ts: datetime.datetime) -> None:
                    self.timestamp: datetime.datetime = ts

            item_for_bisect = DummyItemForBisectSearch(timestamp)
            idx_right: int = self.__data.bisect_left(item_for_bisect)

            if idx_right == 0:
                logger.error(
                    "get_interpolated_at: idx_right is 0 after boundary checks.",
                )
                return deepcopy(self.__data[0])
            if self.__data[idx_right].timestamp == timestamp:
                return deepcopy(self.__data[idx_right])

            item_left: DataTypeT = self.__data[idx_right - 1]
            item_right: DataTypeT = self.__data[idx_right]
            t_diff_secs: float = (
                timestamp - item_left.timestamp
            ).total_seconds()
            total_t_interval_secs: float = (
                item_right.timestamp - item_left.timestamp
            ).total_seconds()

            if total_t_interval_secs == 0:
                return deepcopy(item_left)
            ratio: float = t_diff_secs / total_t_interval_secs

            try:
                payload_left_attr = getattr(item_left, "data", None)
                payload_right_attr = getattr(item_right, "data", None)

                data_left = deepcopy(payload_left_attr)
                data_right = deepcopy(payload_right_attr)

                if not isinstance(data_left, (int, float)) or not isinstance(
                    data_right,
                    (int, float),
                ):
                    logger.error(
                        "Data payloads for interpolation are not numeric after deepcopy. Left: %s (%s), Right: %s (%s)",
                        data_left,
                        type(data_left).__name__,
                        data_right,
                        type(data_right).__name__,
                    )
                    return None
                interpolated_inner_data = data_left + ratio * (
                    data_right - data_left
                )
            except TypeError as e:
                logger.error(
                    "Type error during arithmetic operations for interpolation. Error: %s",
                    e,
                )
                return None

            try:
                interpolated_item = deepcopy(item_left)
                interpolated_item.timestamp = timestamp # pyright: ignore [reportGeneralTypeIssues]
                setattr(interpolated_item, 'data', interpolated_inner_data)  # type: ignore[attr-defined]
                return interpolated_item
            except AttributeError as e:
                logger.error(
                    "Failed to set attributes on new instance of type %s for interpolated data: %s",
                    type(item_left).__name__,
                    e,
                )
                return None
