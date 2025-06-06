import datetime
from abc import ABC, abstractmethod

from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


class SynchronizedClock(ABC):
    """
    An abstract base class for clocks that provide time synchronization.

    This class defines a common interface for obtaining the current synchronized
    time (`now` property) and for converting timestamps between the local system's
    naive datetime representation and a `SynchronizedTimestamp` (which represents
    time in a synchronized domain, typically aligned with a server or a common
    reference time).

    Subclasses must implement the `sync` and `desync` methods to define the
    specific logic for how local time is mapped to synchronized time and vice-versa.
    """

    @property
    def now(self) -> SynchronizedTimestamp:
        """
        Gets the current time as a SynchronizedTimestamp.

        This property provides the current time, adjusted by the clock's
        synchronization logic (as implemented in the `sync` method).

        Returns:
            A SynchronizedTimestamp representing the current synchronized time.
        """
        return self.sync(datetime.datetime.now())

    @abstractmethod
    def desync(self, time: SynchronizedTimestamp) -> datetime.datetime:
        """
        Converts a SynchronizedTimestamp to a local naive datetime.datetime object.

        This method takes a timestamp that is considered to be in the
        synchronized time domain and converts it back to the local system's
        naive datetime representation, effectively removing any synchronization
        offset or adjustment applied by this clock.

        Args:
            time: The SynchronizedTimestamp to desynchronize.

        Returns:
            A naive datetime.datetime object representing the timestamp in local time.
        """
        pass

    @abstractmethod
    def sync(self, timestamp: datetime.datetime) -> SynchronizedTimestamp:
        """
        Converts a local naive datetime.datetime object to a SynchronizedTimestamp.

        This method takes a naive datetime object from the local system and
        applies the clock's synchronization logic (e.g., adding an offset
        obtained from a time server) to convert it into a SynchronizedTimestamp,
        representing the time in the synchronized domain.

        Args:
            timestamp: The naive datetime.datetime object to synchronize.

        Returns:
            A SynchronizedTimestamp representing the input timestamp in the
            synchronized time domain.
        """
        pass
