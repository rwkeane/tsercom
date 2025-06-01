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

    NOTE: Despite the name and the original intention described above, this
    current implementation is a "fake" client. It does not perform any actual
    NTP network synchronization. The `start_async` method simply initializes
    the time offset to zero, and `get_offset_seconds` will return this initial
    (or averaged) offset.
    """

    def __init__(
        self, watcher: ThreadWatcher, server_ip: str, ntp_port: int = kNtpPort
    ) -> None:
        """
        Initializes the FakeTimeSyncClient.

        Args:
            watcher: A ThreadWatcher instance to monitor threads created by this client.
            server_ip: The IP address of the time synchronization server.
                       (Note: Not used in the current fake implementation).
            ntp_port: The port for NTP communication.
                      (Note: Not used in the current fake implementation).
        """
        self.__watcher = watcher
        self.__server_ip = (
            server_ip  # IP of the server to sync with (unused in fake).
        )
        self.__ntp_port = ntp_port  # Port for NTP (unused in fake).

        # __sync_loop_thread: Intended to be the thread that would run an actual
        # NTP synchronization loop. In this fake implementation, it remains None
        # as no actual sync loop is started.
        self.__sync_loop_thread: threading.Thread | None = None

        # __time_offset_lock: A lock to protect concurrent access to the
        # __time_offsets deque, ensuring thread safety when reading or
        # modifying the stored time offsets.
        self.__time_offset_lock = threading.Lock()

        # __time_offsets: A deque to store time offset samples. In a real client,
        # this would hold multiple measurements from an NTP server. In this fake
        # version, it's initialized with a single 0.0 offset by start_async.
        self.__time_offsets = Deque[float]()

        # __is_running: An IsRunningTracker instance to manage and signal the
        # running state of this client (e.g., started/stopped).
        self.__is_running = IsRunningTracker()

        # __start_barrier: A threading.Event used to block calls to
        # `get_offset_seconds` until `start_async` has been called and an
        # initial offset is available (or assumed to be available). This prevents
        # attempts to access an offset before the client is "started".
        self.__start_barrier = threading.Event()

    def get_synchronized_clock(self) -> SynchronizedClock:
        """
        Returns a SynchronizedClock instance that uses this client for offset
        information.

        Returns:
            A ClientSynchronizedClock configured with this FakeTimeSyncClient.
        """
        return ClientSynchronizedClock(self)

    def get_offset_seconds(self) -> float:
        """
        Retrieves the current time offset in seconds.

        This method waits until the client is started (via `start_async`)
        before returning an offset. In this fake implementation, the offset
        is typically an average of values in `__time_offsets`, which
        `start_async` initializes to contain a single 0.0.

        Returns:
            The time offset in seconds. Currently, this is an average of
            pre-defined values (typically just 0.0).
        """
        self.__start_barrier.wait()

        with self.__time_offset_lock:
            count = len(self.__time_offsets)
            assert (
                count > 0
            ), "Time offsets deque should not be empty after start."
            return sum(self.__time_offsets) / count

    def is_running(self) -> bool:
        """
        Checks if the time synchronization client is currently marked as running.

        Returns:
            True if the client is running, False otherwise.
        """
        return self.__is_running.get()

    def stop(self) -> None:
        """
        Stops the NTP synchronization client.

        In this fake implementation, it sets the running state to False and
        clears the start barrier, effectively resetting the client.
        """
        self.__is_running.set(False)
        self.__start_barrier.clear()  # Ensure get_offset_seconds blocks until next start.

    def start_async(self) -> None:
        """
        Starts the "fake" NTP synchronization client.

        This method marks the client as running and initializes the time offset.
        In this fake implementation, no actual network synchronization is performed.
        Instead, it sets a default time offset of 0.0.
        """
        assert not self.__is_running.get(), "Client is already running."
        self.__is_running.set(True)
        self.__start_barrier.set()
        with self.__time_offset_lock:
            if not self.__time_offsets:
                self.__time_offsets.append(0.0)
