from typing import Deque
import ntplib  # type: ignore
import time
import threading
import logging

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
from tsercom.timesync.common.constants import kNtpPort, kNtpVersion
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.util.is_running_tracker import IsRunningTracker


class TimeSyncClient(ClientSynchronizedClock.Client):
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

        # Run the request loop.
        self.__sync_loop_thread = self.__watcher.create_tracked_thread(
            self.__run_sync_loop
        )
        self.__sync_loop_thread.start()

    def __run_sync_loop(self) -> None:
        """Periodically synchronizes with the NTP server."""
        # Make the "real" offset the average of the returned values over the
        # last 30 seconds.
        kMaxOffsetCount = 10
        kOffsetFrequencySeconds = 3

        # Run the loop.
        ntp_client = ntplib.NTPClient()
        while self.__is_running.get():
            try:
                # NOTE: Blocking call.
                response = ntp_client.request(
                    self.__server_ip, port=self.__ntp_port, version=kNtpVersion
                )
                with self.__time_offset_lock:
                    self.__time_offsets.append(response.offset)
                    if len(self.__time_offsets) > kMaxOffsetCount:
                        self.__time_offsets.popleft()
                # Successfully received and processed an offset
                logging.info(f"New NTP Offset: {response.offset:.6f} seconds")
            except ntplib.NTPException as e:
                logging.error(f"NTP error: {e}")
            except Exception as e:
                # Maintain the original behavior for AssertionError regarding the watcher
                if isinstance(e, AssertionError):
                    logging.error(f"AssertionError during NTP sync: {e}") # Log it as an error
                    self.__watcher.on_exception_seen(e) # Call watcher as originally intended
                    raise # Re-raise AssertionError to halt as before
                logging.error(f"Error during NTP sync: {e}")

            # Set the startup barrier only if it hasn't been set yet AND we have valid offsets.
            if not self.__start_barrier.is_set():
                with self.__time_offset_lock:
                    if len(self.__time_offsets) > 0:
                        self.__start_barrier.set()
                        logging.info("TimeSyncClient initialized with first valid offset.")
                    # If len(self.__time_offsets) is still 0, the barrier remains unset.
                    # The warning about fake data and appending 0 is removed.

            time.sleep(kOffsetFrequencySeconds)  # Resynchronize periodically.
