import datetime

from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


class FakeSynchronizedClock(SynchronizedClock):
    """
    A mock or pass-through implementation of the SynchronizedClock.

    This clock does not perform any actual time manipulation. The `sync` and
    `desync` methods return the input timestamp or its direct representation
    without any modification or offset adjustments. It is typically used in
    testing or scenarios where time synchronization is not required or is
    handled externally.
    """

    def desync(self, time: SynchronizedTimestamp) -> datetime.datetime:
        """
        "Desynchronizes" a SynchronizedTimestamp back to a naive datetime object.

        In this fake implementation, this method directly returns the datetime
        object contained within the `SynchronizedTimestamp` without any
        modification, as no actual synchronization offset is applied by this clock.

        Args:
            time: The SynchronizedTimestamp to desynchronize.

        Returns:
            The naive datetime.datetime object extracted from the input.
        """
        return time.as_datetime()

    def sync(self, timestamp: datetime.datetime) -> SynchronizedTimestamp:
        """
        "Synchronizes" a naive datetime object into a SynchronizedTimestamp.

        In this fake implementation, this method directly wraps the given naive
        datetime.datetime object into a `SynchronizedTimestamp` without any
        modification, as no actual synchronization offset is applied by this clock.

        Args:
            timestamp: The naive datetime.datetime object to synchronize.

        Returns:
            A SynchronizedTimestamp created directly from the input datetime.
        """
        return SynchronizedTimestamp(timestamp)
