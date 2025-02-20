import datetime

from timesync.common.synchronized_clock import SynchronizedClock
from timesync.common.synchronized_timestamp import SynchronizedTimestamp


class ServerSynchronizedClock(SynchronizedClock):
    def desync(self, time : SynchronizedTimestamp) -> datetime.datetime:
        return time.as_datetime()
    
    def sync(self, timestamp : datetime.datetime) -> SynchronizedTimestamp:
        return SynchronizedTimestamp(timestamp)