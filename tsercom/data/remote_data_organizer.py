"""Defines RemoteDataOrganizer for managing time-ordered data from a single remote source, including timeout logic."""

import logging  # Add logging import
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import datetime
from functools import partial
import threading
from typing import Deque, Generic, List, Optional, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.util.is_running_tracker import IsRunningTracker

# Generic type for the data being organized, bound by ExposedData.
TDataType = TypeVar("TDataType", bound=ExposedData)

logger = logging.getLogger(__name__)  # Initialize logger


class RemoteDataOrganizer(
    Generic[TDataType], RemoteDataReader[TDataType], DataTimeoutTracker.Tracked
):
    """Organizes and provides access to data received from a specific remote endpoint.

    This class is responsible for managing a time-ordered collection of data
    (of type `TDataType`) associated with a single `CallerIdentifier`.
    It ensures thread-safe access to this data, handles data input via the
    `RemoteDataReader` interface, and implements the `DataTimeoutTracker.Tracked`
    interface to facilitate data timeout logic. It can notify a `Client` when
    new data becomes available.
    """

    class Client(ABC):
        """Interface for clients that need to be notified by `RemoteDataOrganizer`."""

        @abstractmethod
        def _on_data_available(
            self, data_organizer: "RemoteDataOrganizer[TDataType]"
        ) -> None:
            """Callback invoked when new data is processed and available in the organizer.

            Args:
                data_organizer: The `RemoteDataOrganizer` instance that has new data.
            """
            pass

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        caller_id: CallerIdentifier,
        client: Optional["RemoteDataOrganizer.Client[TDataType]"] = None,
    ) -> None:
        """Initializes a RemoteDataOrganizer.

        Args:
            thread_pool: A `ThreadPoolExecutor` used for submitting data
                         processing tasks asynchronously.
            caller_id: The `CallerIdentifier` for the remote endpoint whose
                       data this organizer will manage.
            client: An optional client implementing `RemoteDataOrganizer.Client`
                    to receive callbacks when new data is available.
        """
        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__caller_id: CallerIdentifier = caller_id
        self.__client: Optional[RemoteDataOrganizer.Client[TDataType]] = client

        # Thread lock to protect access to __data and __last_access.
        self.__data_lock: threading.Lock = threading.Lock()

        # Deque to store received data, ordered by timestamp (most recent first).
        self.__data: Deque[TDataType] = Deque[TDataType]()

        # Timestamp of the most recent data item retrieved via get_new_data().
        self.__last_access: datetime.datetime = datetime.datetime.min

        self.__is_running: IsRunningTracker = IsRunningTracker()

        super().__init__()

    @property
    def caller_id(self) -> CallerIdentifier:
        """Gets the `CallerIdentifier` associated with this data organizer.

        Returns:
            The `CallerIdentifier` instance.
        """
        return self.__caller_id

    def start(self) -> None:
        """Starts this data organizer, allowing it to process incoming data.

        Raises:
            RuntimeError: If the organizer is already running (typically by `IsRunningTracker.start()`).
        """
        self.__is_running.start()

    def stop(self) -> None:
        """Stops this data organizer from processing new data.

        Once stopped, no new data will be added, and data timeout mechanisms
        (if part of a `DataTimeoutTracker`) might cease or behave differently
        based on the tracker's implementation.

        Raises:
            RuntimeError: If the organizer is not running or has already been stopped (typically by `IsRunningTracker.stop()`).
        """
        self.__is_running.stop()

    def has_new_data(self) -> bool:
        """Checks if new data has been received since the last call to `get_new_data`.

        "New data" is defined as data items with a timestamp more recent than
        the timestamp of the last item retrieved by `get_new_data`.

        Returns:
            True if new data is available, False otherwise.
        """
        with self.__data_lock:
            if not self.__data:
                return False

            most_recent_timestamp = self.__data[0].timestamp
            last_access_timestamp = self.__last_access
            result = most_recent_timestamp > last_access_timestamp

            return result

    def get_new_data(self) -> List[TDataType]:
        """Retrieves all data items received since the last call to this method.

        Updates the internal "last access" timestamp to the timestamp of the
        most recent item retrieved in this call.

        Returns:
            A list of new `TDataType` items, ordered from most recent to oldest.
            Returns an empty list if no new data is available.
        """
        with self.__data_lock:
            results: List[TDataType] = []
            if not self.__data:
                return results

            for item_idx, item in enumerate(self.__data):
                if item.timestamp > self.__last_access:
                    results.append(item)
                else:
                    break

            if results:
                new_last_access = results[0].timestamp
                self.__last_access = new_last_access

            return results

    def get_most_recent_data(self) -> Optional[TDataType]:
        """Returns the most recently received data item, regardless of last access time.

        Returns:
            The most recent `TDataType` item, or `None` if no data has been received.
        """
        with self.__data_lock:
            if not self.__data:
                return None
            # The leftmost item in the deque is the most recent.
            return self.__data[0]

    def get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[TDataType]:
        """Returns the most recent data item received at or before the given timestamp.

        Args:
            timestamp: The specific `datetime` to find data for.

        Returns:
            The `TDataType` item whose timestamp is the latest at or before the
            specified `timestamp`, or `None` if no such data exists (e.g., all
            data is newer, or no data at all).
        """
        with self.__data_lock:
            if not self.__data:
                return None

            # If the requested timestamp is older than the oldest data we have,
            # then no data at or before that timestamp exists.
            if timestamp < self.__data[-1].timestamp:
                return None

            for item in self.__data:
                if (
                    item.timestamp <= timestamp
                ):  # Found the most recent item at or before the timestamp
                    return item
        # Should not be reached if timestamp >= __data[-1].timestamp and __data is not empty,
        # but as a fallback or if logic changes, return None.
        return None

    def _on_data_ready(self, new_data: TDataType) -> None:
        """Handles an incoming data item.

        Validates the data, ensures it matches the organizer's `caller_id`,
        and submits it for asynchronous processing via `__on_data_ready_impl`.

        Args:
            new_data: The new data item to process.

        Raises:
            TypeError: If `new_data` is not an instance of `ExposedData`.
            AssertionError: If `new_data.caller_id` does not match this
                            organizer's `caller_id`.
        """
        if not isinstance(new_data, ExposedData):
            raise TypeError(
                f"Expected new_data to be an instance of ExposedData, but got {type(new_data).__name__}."
            )

        # Ensure the data belongs to this organizer.
        assert (
            new_data.caller_id == self.caller_id
        ), f"Data's caller_id '{new_data.caller_id}' does not match organizer's '{self.caller_id}'"
        self.__thread_pool.submit(self.__on_data_ready_impl, new_data)

    def __on_data_ready_impl(self, new_data: TDataType) -> None:
        """Internal implementation to process and store new data.

        This method is executed by the thread pool. It adds the new data to the
        internal deque in chronological order (most recent first) and notifies
        the client if data was inserted or updated.

        Args:
            new_data: The `TDataType` item to process.
        """
        # Do not process data if the organizer is not running.
        if not self.__is_running.get():
            return

        data_inserted_or_updated = False
        with self.__data_lock:
            if not self.__data:  # No data yet, just append.
                self.__data.append(new_data)
                data_inserted_or_updated = True
            else:
                current_most_recent_time = self.__data[0].timestamp
                new_data_time = new_data.timestamp

                if new_data_time < current_most_recent_time:
                    logger.debug(
                        f"CallerID {self.caller_id}: Discarding out-of-order older data "
                        f"(ts: {new_data_time}, newest_is: {current_most_recent_time})."
                    )
                    # DESIGN NOTE: This implementation handles new data as follows:
                    # - If the deque is empty, the new data is added.
                    # - If the new data's timestamp is older than the current newest data,
                    #   it is currently discarded (see log message). For a more robust history
                    #   that includes all out-of-order data, one might insert it in sorted order.
                    # - If timestamps match, the existing newest item is updated.
                    # - If the new data is strictly newer, it's added to the front.
                    # The current approach prioritizes simplicity and focuses on the latest state
                    # or strictly sequential data.
                    # To implement full sorted insertion for a complete history:
                    #   # idx = 0
                    #   # while idx < len(self.__data) and new_data_time < self.__data[idx].timestamp:
                    #   #     idx += 1
                    #   # self.__data.insert(idx, new_data)
                    #   # data_inserted_or_updated = True
                    pass
                elif new_data_time == current_most_recent_time:
                    # Data with the same timestamp as the newest; update the newest.
                    self.__data[0] = new_data
                    data_inserted_or_updated = True
                else:  # new_data_time > current_most_recent_time
                    # New data is the absolute newest; add to the front.
                    self.__data.appendleft(new_data)
                    data_inserted_or_updated = True

        if data_inserted_or_updated and self.__client is not None:
            self.__client._on_data_available(self)

    def _on_triggered(self, timeout_seconds: int) -> None:
        """Callback from `DataTimeoutTracker` when a timeout period elapses.

        Submits the `__timeout_old_data` method to the thread pool to remove
        stale data. This is part of the `DataTimeoutTracker.Tracked` interface.

        Args:
            timeout_seconds: The duration of the timeout that triggered this callback.
        """
        self.__thread_pool.submit(
            partial(self.__timeout_old_data, timeout_seconds)
        )

    def __timeout_old_data(self, timeout_seconds: int) -> None:
        """Removes data older than the specified timeout period.

        This method is executed by the thread pool. It calculates the oldest
        allowed timestamp and removes all data items from the end of the deque
        (oldest items) that are older than this threshold.

        Args:
            timeout_seconds: The timeout duration in seconds. Data older than
                             `now - timeout_seconds` will be removed.
        """
        # Do not timeout data if the organizer is not running.
        if not self.__is_running.get():
            return

        current_time = datetime.datetime.now()
        timeout_delta = datetime.timedelta(seconds=timeout_seconds)
        oldest_allowed_timestamp = current_time - timeout_delta

        with self.__data_lock:
            while (
                self.__data
                and self.__data[-1].timestamp < oldest_allowed_timestamp
            ):
                self.__data.pop()
