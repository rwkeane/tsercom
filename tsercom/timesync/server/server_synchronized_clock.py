"""Server-side synchronized clock implementation."""

import datetime

from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


class ServerSynchronizedClock(SynchronizedClock):
    """
    A synchronized clock implementation representing the server's perspective.

    On the server, its local time is the authoritative synchronized
    time. Thus, `sync`/`desync` operations are effectively pass-through:
    - `sync` wraps a naive server local `datetime` into a
      `SynchronizedTimestamp` without modification.
    - `desync` unwraps a `SynchronizedTimestamp` back to a naive server
      local `datetime` without modification.
    This clock assumes `datetime.datetime.now()` is source of truth.
    """

    def desync(self, time: SynchronizedTimestamp) -> datetime.datetime:
        """
        Converts a SynchronizedTimestamp to a naive server local datetime.

        Server's local time is authoritative synchronized time; this method
        returns the datetime from provided SynchronizedTimestamp directly.

        Args:
            time: The SynchronizedTimestamp to desynchronize.

        Returns:
            The naive datetime.datetime object representing server local time.
        """
        return time.as_datetime()

    def sync(self, timestamp: datetime.datetime) -> SynchronizedTimestamp:
        """
        Converts a naive server local datetime to a SynchronizedTimestamp.

        Server's local time is authoritative; this method creates a
        SynchronizedTimestamp directly from the given naive datetime object
        without any modification.

        Args:
            timestamp: The naive datetime.datetime object (server's local time)
                       to synchronize.

        Returns:
            A SynchronizedTimestamp representing the server's local time.
        """
        return SynchronizedTimestamp(timestamp)
