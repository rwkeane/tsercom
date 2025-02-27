from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import datetime
from functools import partial
import threading
from typing import Deque, Generic, List, Optional, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TDataType = TypeVar("TDataType", bound = ExposedData)
class RemoteDataOrganizer(Generic[TDataType], RemoteDataReader[TDataType]):
    """
    This class is responsible for organizing data received from a remote
    endpoint and exposing that to external viewers in a simple, thread-safe
    manner.
    """
    class Client(ABC):
        @abstractmethod
        def _on_data_available(self, data_organizer : 'RemoteDataOrganizer'):
            pass
        
    def __init__(self,
                 thread_pool : ThreadPoolExecutor,
                 caller_id : CallerIdentifier,
                 client : Optional['RemoteDataOrganizer.Client'] = None,
                 timeout_seconds : int = 60):
        self.__thread_pool = thread_pool
        self.__caller_id = caller_id
        self.__client = client
        self.__timeout_seconds = timeout_seconds

        # Stored data.
        self.__data_lock = threading.Lock()
        self.__data = Deque[TDataType]()
        self.__last_access = datetime.datetime.min

        # Schedule timeout.
        self.__is_running = IsRunningTracker()
        if timeout_seconds <= 0:
            self.__thread_pool.submit(self.__timeout_old_data,
                                      1000 * timeout_seconds / 2)

        super().__init__()

    @property
    def caller_id(self):
        return self.__caller_id
    
    def start(self):
        """
        Starts this instance.
        """
        assert not self.__is_running.get()
        self.__is_running.set(True)

    def stop(self):
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
                return False
            
            return self.__data[0].timestamp > self.__last_access
        
    def get_new_data(self) -> List[TDataType]:
        """
        Returns all data received since the last call to get_new_data() for the
        assocaited caller.
        """
        with self.__data_lock:
            results = []
            for i in range(len(self.__data)):
                if self.__data[i].timestamp > self.__last_access:
                    results.append(self.__data[i])
                else:
                    break
                
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
            self, timestamp : datetime.datetime) -> TDataType | None:
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
    
    def __timeout_old_data(self):
        # Get the timeout.
        current_time = datetime.datetime.now()
        timeout =  datetime.timedelta(seconds = self.__timeout_seconds)
        oldest_allowed = current_time - timeout

        # Eliminate old data.
        with self.__data_lock:
            while len(self.__data) > 0 and \
                  self.__data[-1].timestamp < oldest_allowed:
                self.__data.pop()

        # Schedule next timeout task.
        if self.__is_running.get():
            self.__thread_pool.submit(self.__timeout_old_data,
                                      1000 * self.__timeout_seconds / 2)
    
    def _on_data_ready(self, new_data : TDataType):
        # Validate the data.
        assert issubclass(type(new_data), ExposedData), type(new_data)
        assert new_data.caller_id == self.caller_id, \
                (new_data.caller_id, self.caller_id)
        
        # Do real processing.
        self.__thread_pool.submit(self.__on_data_ready_impl, new_data)

    def __on_data_ready_impl(self, new_data : TDataType):
        # Exit early if not running.
        if not self.__is_running.get():
            return
        
        # Try to insert the data.
        with self.__data_lock:
            if len(self.__data) == 0:
                self.__data.append(new_data)
            else:
                first_time = self.__data[0].timestamp
                other_time = new_data.timestamp

                if other_time < first_time:
                    return
                
                elif other_time == first_time:
                    self.__data[0] = new_data
                    return

                else:
                    self.__data.appendleft(new_data)

        # If new data was added, inform the user.
        if not self.__client is None:
            self.__client._on_data_available(self)