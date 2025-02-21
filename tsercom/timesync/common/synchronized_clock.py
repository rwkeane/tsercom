from abc import ABC, abstractmethod
import datetime

from tsercom.timesync.common.synchronized_timestamp import SynchronizedTimestamp


class SynchronizedClock(ABC):
    @property
    def now(self) -> SynchronizedTimestamp:
        return self.sync(datetime.datetime.now())

    @abstractmethod
    def desync(self, time : SynchronizedTimestamp) -> datetime.datetime:
        pass

    @abstractmethod
    def sync(self, timestamp : datetime.datetime) -> SynchronizedTimestamp:
        pass