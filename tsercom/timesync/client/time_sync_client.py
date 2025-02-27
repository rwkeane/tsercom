from typing import Deque
import ntplib
import time
import threading

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import ClientSynchronizedClock
from tsercom.timesync.common.constants import kNtpPort, kNtpVersion
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.util.is_running_tracker import IsRunningTracker

class TimeSyncClient(ClientSynchronizedClock.Client):
    """
    This class is used to synchronize clocks with an endpoint running an
    TimeSyncServer. Specifically, this class defines the client-side of an NTP
    client-server handshake, receiving the offset from a call to the server.
    """
    def __init__(self,
                 watcher : ThreadWatcher,
                 server_ip : str,
                 ntp_port : int = kNtpPort):
        self.__watcher = watcher
        self.__server_ip = server_ip
        self.__ntp_port = ntp_port

        # The running task for the sync loop.
        self.__sync_loop_thread : threading.Thread = None

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

    def is_running(self):
        return self.__is_running.get()

    def stop(self):
        """Stops the NTP synchronization thread."""
        self.__is_running.set(False)
        
    def start_async(self):
        """Starts the NTP synchronization thread."""
        # Set the service to started.
        assert not self.__is_running.get()
        self.__is_running.set(True)

        # Run the request loop.
        self.__sync_loop_thread = self.__watcher.create_tracked_thread(
                self.__run_sync_loop)
        self.__sync_loop_thread.start()
    
    def __run_sync_loop(self):
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
                response = ntp_client.request(self.__server_ip,
                                              port = self.__ntp_port,
                                              version = kNtpVersion)
                with self.__time_offset_lock:
                    self.__time_offsets.append(response.offset)
                    if len(self.__time_offsets) > kMaxOffsetCount:
                        self.__time_offsets.popleft()

                print(f"New NTP Offset: {response.offset:.6f} seconds")
            except ntplib.NTPException as e:
                print(f"NTP error: {e}")
            except Exception as e:
                print(f"Error during NTP sync: {e}")
                if isinstance(e, AssertionError):
                    self.__watcher.on_exception_seen(e)
                    raise e

            # If this is the first call, set the startup barrier.
            if not self.__start_barrier.is_set():
                with self.__time_offset_lock:
                    if len(self.__time_offsets) == 0:
                        print("WARNING: Fake time sync data used.")
                        self.__time_offsets.append(0)
                self.__start_barrier.set()
            
            time.sleep(kOffsetFrequencySeconds)  # Resynchronize periodically.
