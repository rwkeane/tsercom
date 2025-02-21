from abc import ABC, abstractmethod
import datetime

from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import SynchronizedTimestamp


class ClientSynchronizedClock(SynchronizedClock):
    """
    This class defines a clock that is synchronized with the server-side clock
    as defined by |client|.
    """
    class Client(ABC):
        @abstractmethod
        async def get_offset_seconds(self) -> float:
            pass

    def __init__(self, client : 'ClientSynchronizedClock.Client'):
        self.__client = client
        super().__init__()
    
    def desync(self, time : SynchronizedTimestamp) -> datetime.datetime:
        offset = self.__client.get_offset_seconds()
        timestamp = time.as_datetime()
        offset = datetime.timedelta(seconds = offset)

        return timestamp - offset
    
    def sync(self, timestamp : datetime.datetime) -> SynchronizedTimestamp:
        delta_future = self.__client.get_offset_seconds()
        delta = datetime.timedelta(seconds = delta_future)

        timestamp = timestamp + delta
        return SynchronizedTimestamp(timestamp)
