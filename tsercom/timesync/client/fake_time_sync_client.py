"""Fake TimeSyncClient implementation for testing purposes."""

import threading
from collections import deque

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
from tsercom.timesync.common.constants import kNtpPort
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.util.is_running_tracker import IsRunningTracker


class FakeTimeSyncClient(ClientSynchronizedClock.Client):
    """A fake client for time synchronization.

    Used for testing or when NTP is unavailable. This class defines the
    client-side of an NTP client-server handshake but does not perform
    any actual network synchronization. The `start_async` method simply
    initializes the time offset to zero, and `get_offset_seconds`
    returns this offset.
    """

    def __init__(
        self, watcher: ThreadWatcher, server_ip: str, ntp_port: int = kNtpPort
    ) -> None:
        """Initialize the FakeTimeSyncClient.

        Args:
            watcher: A ThreadWatcher instance.
            server_ip: IP address of the time sync server (unused in fake).
            ntp_port: Port for NTP communication (unused in fake).

        """
        self.__watcher = watcher
        self.__server_ip = server_ip
        self.__ntp_port = ntp_port

        # __sync_loop_thread: Intended for actual NTP sync loop. Remains None.

        self.__sync_loop_thread: threading.Thread | None = None

        # __time_offset_lock: Lock for thread-safe access to __time_offsets.
        self.__time_offset_lock = threading.Lock()

        # __time_offsets: Deque for offset samples. Fake client uses [0.0].
        self.__time_offsets = deque[float]()

        # __is_running: Manages running state of this client.
        self.__is_running = IsRunningTracker()

        # __start_barrier: Event to block get_offset_seconds until start_async.
        self.__start_barrier = threading.Event()

    def get_synchronized_clock(self) -> SynchronizedClock:
        """Return a SynchronizedClock instance using this client.

        Returns:
            A ClientSynchronizedClock configured with this FakeTimeSyncClient.

        """
        return ClientSynchronizedClock(self)

    def get_offset_seconds(self) -> float:
        """Retrieve the current time offset in seconds.

        Waits until client started (via `start_async`). Fake client
        typically returns an average of `__time_offsets` (usually 0.0).

        Returns:
            The time offset in seconds (average of pre-defined values).

        """
        self.__start_barrier.wait()

        with self.__time_offset_lock:
            count = len(self.__time_offsets)
            assert count > 0, "Time offsets deque should not be empty after start."
            return sum(self.__time_offsets) / count

    def is_running(self) -> bool:
        """Check if the time synchronization client is marked as running.

        Returns:
            True if the client is running, False otherwise.

        """
        return self.__is_running.get()

    def stop(self) -> None:
        """Stop the NTP synchronization client.

        Sets running state to False and clears start barrier.
        """
        self.__is_running.set(False)
        self.__start_barrier.clear()

    def start_async(self) -> None:
        """Start the "fake" NTP synchronization client.

        Marks client as running, initializes time offset to 0.0.
        No actual network synchronization is performed.
        """
        assert not self.__is_running.get(), "Client is already running."
        self.__is_running.set(True)
        self.__start_barrier.set()
        with self.__time_offset_lock:
            if not self.__time_offsets:  # Initialize only if empty
                self.__time_offsets.append(0.0)
