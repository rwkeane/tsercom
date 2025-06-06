"""Client-side synchronized clock, uses a TimeSyncClient for offsets."""

import datetime
from abc import ABC, abstractmethod

from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


class ClientSynchronizedClock(SynchronizedClock):
    """
    This class defines a clock that is synchronized with the server-side clock
    as defined by |client|.
    """

    class Client(ABC):
        """
        An abstract interface for a client that can provide the time offset
        between this client and a server. This offset is used by the
        ClientSynchronizedClock to adjust timestamps.
        """

        @abstractmethod
        def get_offset_seconds(self) -> float:
            """
            Retrieves the time offset in seconds between this client and the
            server.

            A positive value indicates that the client's clock is ahead of the
            server's clock. A negative value indicates that the client's clock
            is behind the server's clock.

            Returns:
                The time offset in seconds as a float.
            """

        @abstractmethod
        def get_synchronized_clock(self) -> SynchronizedClock:
            """
            Returns a SynchronizedClock instance using this client for offsets.
            """

        @abstractmethod
        def start_async(self) -> None:
            """
            Starts the time synchronization client asynchronously.
            """

        @abstractmethod
        def stop(self) -> None:
            """
            Stops the time synchronization client.
            """

    def __init__(self, client: "ClientSynchronizedClock.Client") -> None:
        """
        Initializes the ClientSynchronizedClock.

        Args:
            client: ClientSynchronizedClock.Client instance for time offset.
        """  # Shortened line
        self.__client = client
        super().__init__()

    def desync(self, time: SynchronizedTimestamp) -> datetime.datetime:
        # A positive offset means the client is ahead of the server.
        # A negative offset means the client is behind the server.
        offset_seconds = self.__client.get_offset_seconds()
        timestamp_dt = time.as_datetime()
        offset_timedelta = datetime.timedelta(seconds=offset_seconds)

        # To desynchronize a timestamp (i.e., convert it from server time
        # back to client's local time), subtract the offset.
        # For example:
        # If client is 5s ahead (offset_seconds = 5):
        #   Server time (timestamp_dt) = 12:00:05
        #   Client local time = 12:00:05 - 5s = 12:00:00
        # If client is 5s behind (offset_seconds = -5):
        #   Server time (timestamp_dt) = 11:59:55
        #   Client local time = 11:59:55 - (-5s) = 12:00:00
        return timestamp_dt - offset_timedelta

    def sync(self, timestamp: datetime.datetime) -> SynchronizedTimestamp:
        # A positive offset means the client is ahead of the server.
        # A negative offset means the client is behind the server.
        offset_seconds = self.__client.get_offset_seconds()
        offset_timedelta = datetime.timedelta(seconds=offset_seconds)

        # To synchronize a timestamp (i.e., convert it from client's local
        # time to server time), we need to add the offset.
        # For example:
        # If client is 5s ahead (offset_seconds = 5):
        #   Client local time (timestamp) = 12:00:00
        #   Server time = 12:00:00 + 5s = 12:00:05
        # If client is 5s behind (offset_seconds = -5):
        #   Client local time (timestamp) = 12:00:00
        #   Server time = 12:00:00 + (-5s) = 11:59:55
        #
        # Note: The variable name `delta_future` in the original code was
        # a bit misleading. It's simply the offset. The operation here
        # adjusts the local timestamp to the server's perspective.
        synchronized_dt = timestamp + offset_timedelta
        return SynchronizedTimestamp(synchronized_dt)
