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

TDataType = TypeVar("TDataType", bound=ExposedData)


class RemoteDataOrganizer(
    Generic[TDataType], RemoteDataReader[TDataType], DataTimeoutTracker.Tracked
):
    """
    This class is responsible for organizing data received from a remote
    endpoint and exposing that to external viewers in a simple, thread-safe
    manner.
    """

    class Client(ABC):
        @abstractmethod
        def _on_data_available(
            self, data_organizer: "RemoteDataOrganizer[TDataType]"
        ) -> None:
            pass

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        caller_id: CallerIdentifier,
        client: Optional["RemoteDataOrganizer.Client"] = None,
    ):
        self.__thread_pool = thread_pool
        self.__caller_id = caller_id
        self.__client = client

        # Stored data.
        self.__data_lock = threading.Lock()
        self.__data = Deque[TDataType]()
        self.__last_access = datetime.datetime.min # Initialize to a very old timestamp

        # Schedule timeout.
        self.__is_running = IsRunningTracker()

        super().__init__()

    @property
    def caller_id(self) -> CallerIdentifier:
        return self.__caller_id

    def start(self) -> None:
        """
        Starts this instance.
        """
        assert not self.__is_running.get()
        self.__is_running.set(True)

    def stop(self) -> None:
        """
        Stops this instance from running. After this call, no new data is added
        and no data times out.
        """
        assert self.__is_running.get()
        self.__is_running.set(False)

    def has_new_data(self) -> bool:
        """
        Returns wheter the caller assocaited with this instance has sent more
        data since the last call to get_new_data().
        """
        with self.__data_lock:
            if len(self.__data) == 0:
                # print(f"DEBUG: [RemoteDataOrganizer.has_new_data] id(self): {id(self)} - No data, returning False. self.__last_access: {self.__last_access}")
                return False
            
            is_new = self.__data[0].timestamp > self.__last_access
            # print(f"DEBUG: [RemoteDataOrganizer.has_new_data] id(self): {id(self)} - Data present. Newest timestamp: {self.__data[0].timestamp}, self.__last_access: {self.__last_access}. Returning: {is_new}")
            return is_new

    def get_new_data(self) -> List[TDataType]:
        """
        Returns all data received since the last call to get_new_data() for the
        assocaited caller.
        """
        with self.__data_lock:
            # print(f"DEBUG: [RemoteDataOrganizer.get_new_data] id(self): {id(self)} - Entered. Current self.__last_access: {self.__last_access}")
            results = []
            if not self.__data: # Handle empty deque case early
                # print(f"DEBUG: [RemoteDataOrganizer.get_new_data] id(self): {id(self)} - No data in deque. Returning empty list.")
                return results

            for i in range(len(self.__data)):
                if self.__data[i].timestamp > self.__last_access:
                    results.append(self.__data[i])
                else:
                    # Data is sorted by timestamp (newest first), so we can break
                    break
            
            # Update __last_access if new data was actually retrieved and data is present
            if results and self.__data: # self.__data check is redundant if results is not empty, but good for safety
                new_last_access = self.__data[0].timestamp # The timestamp of the newest item overall
                # print(f"DEBUG: [RemoteDataOrganizer.get_new_data] id(self): {id(self)} - New data found. Updating self.__last_access from {self.__last_access} to {new_last_access}")
                self.__last_access = new_last_access
            # else:
                # print(f"DEBUG: [RemoteDataOrganizer.get_new_data] id(self): {id(self)} - No new data found or deque is empty. self.__last_access remains {self.__last_access}")

            # print(f"DEBUG: [RemoteDataOrganizer.get_new_data] id(self): {id(self)} - Returning {len(results)} items. New self.__last_access: {self.__last_access}")
            return results

    def get_most_recent_data(self) -> TDataType | None:
        """
        Returns the most recently received data for the associated caller.
        """
        with self.__data_lock:
            if len(self.__data) == 0:
                return None

            return self.__data[0]

    def get_data_for_timestamp(
        self, timestamp: datetime.datetime
    ) -> TDataType | None:
        """
        Returns the data most recently received before |timestamp|, or None if
        that data either has not yet been received or has timed out.
        """
        with self.__data_lock:
            if len(self.__data) == 0:
                return None

            if timestamp < self.__data[-1].timestamp:
                return None

            for i in range(len(self.__data)):
                if timestamp > self.__data[i].timestamp:
                    return self.__data[i]

        return None

    def _on_data_ready(self, new_data: TDataType) -> None:
        # Validate the data.
        assert issubclass(type(new_data), ExposedData), type(new_data)
        assert new_data.caller_id == self.caller_id, (
            new_data.caller_id,
            self.caller_id,
        )

        # Do real processing.
        # print(f"DEBUG: [RemoteDataOrganizer._on_data_ready] id(self): {id(self)} - Submitting __on_data_ready_impl for data with timestamp {new_data.timestamp}")
        self.__thread_pool.submit(self.__on_data_ready_impl, new_data)

    def __on_data_ready_impl(self, new_data: TDataType) -> None:
        # Exit early if not running.
        if not self.__is_running.get():
            # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Not running, exiting.")
            return

        data_inserted_or_updated = False
        # Try to insert the data.
        with self.__data_lock:
            # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Processing data with timestamp {new_data.timestamp}. Current deque size: {len(self.__data)}")
            if len(self.__data) == 0:
                self.__data.append(new_data)
                data_inserted_or_updated = True
                # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Appended data, deque size: {len(self.__data)}")
            else:
                first_time = self.__data[0].timestamp
                other_time = new_data.timestamp

                if other_time < first_time:
                    # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Data is older, not adding. Data timestamp: {other_time}, Newest in deque: {first_time}")
                    return # Data is older than what we have, do nothing or handle out-of-order if necessary

                elif other_time == first_time:
                    self.__data[0] = new_data # Update if same timestamp
                    data_inserted_or_updated = True
                    # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Updated data with same timestamp.")
                    return

                else: # new_data.timestamp > self.__data[0].timestamp (newest)
                    self.__data.appendleft(new_data)
                    data_inserted_or_updated = True
                    # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Appended data to left (newest), deque size: {len(self.__data)}")


        # If new data was added, inform the user.
        if data_inserted_or_updated and self.__client is not None:
            # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Data was inserted/updated, informing client.")
            self.__client._on_data_available(self)
        # else:
            # print(f"DEBUG: [RemoteDataOrganizer.__on_data_ready_impl] id(self): {id(self)} - Data not new or client is None, not informing client.")


    def _on_triggered(self, timeout_seconds: int) -> None:
        self.__thread_pool.submit(
            partial(self.__timeout_old_data, timeout_seconds)
        )

    def __timeout_old_data(self, timeout_seconds: int) -> None:
        # Get the timeout.
        current_time = datetime.datetime.now()
        timeout = datetime.timedelta(seconds=timeout_seconds)
        oldest_allowed = current_time - timeout

        # Eliminate old data.
        with self.__data_lock:
            removed_count = 0
            while (
                len(self.__data) > 0
                and self.__data[-1].timestamp < oldest_allowed
            ):
                self.__data.pop()
                removed_count +=1
            # if removed_count > 0:
                # print(f"DEBUG: [RemoteDataOrganizer.__timeout_old_data] id(self): {id(self)} - Removed {removed_count} old items. Remaining size: {len(self.__data)}")
