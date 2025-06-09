"""TimeSyncClient implementation using NTP."""

import logging
import threading
import time
from typing import Deque

import ntplib  # type: ignore[import-untyped]

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
from tsercom.timesync.common.constants import kNtpPort, kNtpVersion
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.util.is_running_tracker import IsRunningTracker

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes # State for time sync client.
class TimeSyncClient(ClientSynchronizedClock.Client):
    """
    Synchronizes clocks with an NTP server.

    Defines client-side of NTP handshake.
    NOTE: This is a real NTP client, unlike FakeTimeSyncClient.
    """

    def __init__(
        self, watcher: ThreadWatcher, server_ip: str, ntp_port: int = kNtpPort
    ) -> None:
        """
        Initializes the TimeSyncClient.

        Args:
            watcher: ThreadWatcher to monitor the synchronization thread.
            server_ip: IP address of the NTP server.
            ntp_port: Port of the NTP server.
        """
        self.__watcher = watcher
        self.__server_ip = server_ip
        self.__ntp_port = ntp_port
        self.__sync_loop_thread: threading.Thread | None = None
        self.__time_offset_lock = threading.Lock()
        self.__time_offsets = Deque[float]()
        self.__is_running = IsRunningTracker()
        self.__start_barrier = threading.Event()

    def get_synchronized_clock(self) -> SynchronizedClock:
        """Returns a SynchronizedClock using this client for offsets."""
        return ClientSynchronizedClock(self)

    def get_offset_seconds(self) -> float:
        """Retrieves current averaged time offset in seconds from NTP server.

        Waits for first successful sync. Returns average of recent offsets.

        Returns:
            Averaged time offset in seconds.

        Raises:
            AssertionError: If called after barrier set but offsets empty.
        """
        self.__start_barrier.wait()
        with self.__time_offset_lock:
            count = len(self.__time_offsets)
            assert count > 0, "Offsets deque empty after start barrier."
            return sum(self.__time_offsets) / count

    def is_running(self) -> bool:
        """Checks if NTP client is running (sync loop active/starting)."""
        return self.__is_running.get()

    def stop(self) -> None:
        """Stops NTP sync thread, causing __run_sync_loop to terminate."""
        self.__is_running.set(False)

    def start_async(self) -> None:
        """Starts NTP sync thread. Does nothing if already running."""
        assert not self.__is_running.get(), "TimeSyncClient already running."
        self.__is_running.set(True)
        self.__sync_loop_thread = self.__watcher.create_tracked_thread(
            self.__run_sync_loop
        )
        self.__sync_loop_thread.start()

    def __run_sync_loop(self) -> None:
        """Main loop for periodically synchronizing time with NTP server.

        Runs in `self.__sync_loop_thread`. Queries NTP server, stores
        rolling average of offsets, handles exceptions, and manages
        `__start_barrier` for initial offset availability.
        """
        max_offset_count = 10  # Max samples for rolling average
        offset_frequency_seconds = 3  # Sync frequency

        ntp_client = ntplib.NTPClient()
        while self.__is_running.get():
            try:
                response = ntp_client.request(
                    self.__server_ip, port=self.__ntp_port, version=kNtpVersion
                )
                with self.__time_offset_lock:
                    self.__time_offsets.append(response.offset)
                    if len(self.__time_offsets) > max_offset_count:
                        self.__time_offsets.popleft()
                logging.info("New NTP Offset: %.6f seconds", response.offset)
            except ntplib.NTPException as e:
                logging.error("NTP error: %s", e)
            # pylint: disable=broad-exception-caught # Ensures loop continues
            except Exception as e:
                if isinstance(e, AssertionError):
                    logging.error("AssertionError during NTP sync: %s", e)
                    self.__watcher.on_exception_seen(e)
                    return  # Terminate loop after reporting.
                logging.error("Error during NTP sync: %s", e)

            if not self.__start_barrier.is_set():
                with self.__time_offset_lock:
                    if len(self.__time_offsets) > 0:
                        self.__start_barrier.set()
                        logging.info(
                            "TimeSyncClient: First valid NTP offset received."
                        )
            time.sleep(offset_frequency_seconds)
