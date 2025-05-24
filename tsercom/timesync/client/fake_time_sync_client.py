from typing import Deque
import threading

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
from tsercom.timesync.common.constants import kNtpPort
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.util.is_running_tracker import IsRunningTracker


class FakeTimeSyncClient(ClientSynchronizedClock.Client):
    """
    This class is used to synchronize clocks with an endpoint running an
    TimeSyncServer. Specifically, this class defines the client-side of an NTP
    client-server handshake, receiving the offset from a call to the server.
    """

    def __init__(
        self, watcher: ThreadWatcher, server_ip: str, ntp_port: int = kNtpPort
    ):
        self.__watcher = watcher
        self.__server_ip = server_ip
        self.__ntp_port = ntp_port

        # The running task for the sync loop.
        self.__sync_loop_thread: threading.Thread | None = None

        # To keep time_offset safe.
        self.__time_offset_lock = threading.Lock()
        self.__time_offsets = Deque[float]()

        # To keep is_running safe.
        self.__is_running = IsRunningTracker()

        # To avoid getting an offset before its been determined.
        self.__start_barrier = threading.Event()

    def get_synchronized_clock(self) -> SynchronizedClock:
        return ClientSynchronizedClock(self)

    def get_offset_seconds(self) -> float:
        self.__start_barrier.wait()

        # Average the currently stored values of
        with self.__time_offset_lock:
            count = len(self.__time_offsets)
            assert count > 0
            return sum(self.__time_offsets) / count

    def is_running(self) -> bool:
        return self.__is_running.get()

    def stop(self) -> None:
        """Stops the NTP synchronization thread."""
        self.__is_running.set(False)

    def start_async(self) -> None:
        """Starts the NTP synchronization thread."""
        # Set the service to started.
        assert not self.__is_running.get()
        self.__is_running.set(True)
