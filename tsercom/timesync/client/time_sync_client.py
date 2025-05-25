from typing import Deque
import ntplib  # type: ignore  # ntplib may not have type stubs available
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
    ) -> None:
        """
        Initializes the TimeSyncClient.

        Args:
            watcher: A ThreadWatcher instance to monitor the synchronization thread.
            server_ip: The IP address of the NTP server.
            ntp_port: The port of the NTP server.
        """
        self.__watcher = watcher  # ThreadWatcher to manage the lifecycle of the sync thread.
        self.__server_ip = server_ip  # IP address of the NTP server to synchronize with.
        self.__ntp_port = ntp_port  # Port number for the NTP server.

        # __sync_loop_thread: Holds the reference to the thread that runs the
        # NTP synchronization loop (__run_sync_loop). Initialized when
        # start_async is called.
        self.__sync_loop_thread: threading.Thread | None = None

        # __time_offset_lock: A threading.Lock to ensure thread-safe access
        # to the __time_offsets deque, which is shared between the sync loop
        # thread and any thread calling get_offset_seconds.
        self.__time_offset_lock = threading.Lock()

        # __time_offsets: A deque to store a running list of recent time offset
        # values obtained from the NTP server. Used to calculate an averaged offset.
        self.__time_offsets = Deque[float]()

        # __is_running: An IsRunningTracker instance to manage the running state
        # of the client. Controls the synchronization loop and signals stopping.
        self.__is_running = IsRunningTracker()

        # __start_barrier: A threading.Event that blocks calls to
        # get_offset_seconds until at least one valid time offset has been
        # received from the NTP server. This ensures the client doesn't return
        # an offset before it's properly initialized.
        self.__start_barrier = threading.Event()

    def get_synchronized_clock(self) -> SynchronizedClock:
        """
        Returns a SynchronizedClock instance that uses this client for time offset
        information.

        Returns:
            A ClientSynchronizedClock configured with this TimeSyncClient.
        """
        return ClientSynchronizedClock(self)

    def get_offset_seconds(self) -> float:
        """
        Retrieves the current averaged time offset in seconds from the NTP server.

        This method waits until the `__start_barrier` is set, which occurs after
        the first successful NTP synchronization. It then returns an average of
        the most recently collected time offsets.

        Returns:
            The averaged time offset in seconds.


        Raises:
            AssertionError: If called after the barrier is set but the offsets deque
                            is unexpectedly empty.
        """
        self.__start_barrier.wait()

        with self.__time_offset_lock:
            count = len(self.__time_offsets)
            assert count > 0, "Time offsets deque should not be empty after start barrier."
            return sum(self.__time_offsets) / count

    def is_running(self) -> bool:
        """
        Checks if the NTP time synchronization client is currently running.

        Returns:
            True if the client is running (i.e., the sync loop is active or starting),
            False otherwise.
        """
        return self.__is_running.get()

    def stop(self) -> None:
        """
        Stops the NTP synchronization thread.

        Sets the internal running state to False, which will cause the
        `__run_sync_loop` to terminate after its current iteration.
        """
        self.__is_running.set(False)

    def start_async(self) -> None:
        """
        Starts the NTP synchronization thread.

        Sets the client to a running state and starts a new thread dedicated to
        periodically synchronizing time with the NTP server via `__run_sync_loop`.
        Does nothing if the client is already running.
        """
        assert not self.__is_running.get(), "TimeSyncClient is already running."
        self.__is_running.set(True)

        self.__sync_loop_thread = self.__watcher.create_tracked_thread(
            self.__run_sync_loop
        )
        self.__sync_loop_thread.start()

    def __run_sync_loop(self) -> None:
        """
        The main loop for periodically synchronizing time with the NTP server.

        This method runs in a separate thread (`self.__sync_loop_thread`). It
        continuously queries the NTP server for the time offset, stores a
        rolling average of these offsets, and handles exceptions during the
        process. It also manages the `__start_barrier` to signal when the
        first valid offset has been obtained.
        """
        # Make the "real" offset the average of the returned values over the
        # last 30 seconds.
        kMaxOffsetCount = 10
        kOffsetFrequencySeconds = 3

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
                logging.info(f"New NTP Offset: {response.offset:.6f} seconds")
            except ntplib.NTPException as e:
                logging.error(f"NTP error: {e}")
            except Exception as e:
                # Special handling for AssertionError:
                # This is intended to catch critical assertion failures within the sync loop.
                # The behavior is to log the error, notify the ThreadWatcher
                # (which might have specific logic for such failures, e.g., system health),
                # and then re-raise the AssertionError to halt the sync loop thread,
                # as assertion failures typically indicate an unrecoverable state or bug.
                if isinstance(e, AssertionError):
                    logging.error(f"AssertionError during NTP sync: {e}")
                    self.__watcher.on_exception_seen(e)
                    raise # Re-raise AssertionError to halt the sync loop.
                logging.error(f"Error during NTP sync: {e}")

            # The __start_barrier is used to signal that the TimeSyncClient has
            # successfully obtained at least one time offset from the server and
            # is ready to provide synchronization information.
            # It should only be set once. Once set, get_offset_seconds() can proceed.
            if not self.__start_barrier.is_set():
                with self.__time_offset_lock:
                    # Check if we have successfully populated __time_offsets.
                    if len(self.__time_offsets) > 0:
                        self.__start_barrier.set()
                        logging.info(
                            "TimeSyncClient initialized: First valid NTP offset received."
                        )
            # If len(self.__time_offsets) is still 0 (e.g., due to continuous NTP errors
            # since startup), the barrier remains unset, and get_offset_seconds()
            # will continue to block.

            time.sleep(kOffsetFrequencySeconds)  # Resynchronize periodically.
