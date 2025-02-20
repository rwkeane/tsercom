from abc import ABC, abstractmethod
import datetime
from functools import partial
import threading
from typing import Deque, Generic, List, Optional, TypeVar

from util.caller_id.caller_identifier import CallerIdentifier
from util.data.exposed_data import ExposedData
from util.data.remote_data_reader import RemoteDataReader
from util.is_running_tracker import IsRunningTracker
from util.threading.task_runner import TaskRunner


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
                 task_runner : TaskRunner,
                 caller_id : CallerIdentifier,
                 client : Optional['RemoteDataOrganizer.Client'] = None,
                 timeout_seconds : int = 60):
        self.__task_runner = task_runner
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
            self.__task_runner.post_task_with_delay(self.__timeout_old_data,
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
        assert self.__task_runner.is_running_on_task_runner()

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
            self.__task_runner.post_task_with_delay(
                    self.__timeout_old_data, 1000 * self.__timeout_seconds / 2)
    
    def _on_data_ready(self, new_data : TDataType):
        if not self.__task_runner.is_running_on_task_runner():
            self.__task_runner.post_task(partial(self._on_data_ready, new_data))
            return
        
        # Validate the data.
        assert issubclass(type(new_data), ExposedData), type(new_data)
        assert new_data.caller_id == self.caller_id, \
                (new_data.caller_id, self.caller_id)

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