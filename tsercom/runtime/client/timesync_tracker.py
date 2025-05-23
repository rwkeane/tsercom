from typing import Dict
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.fake_time_sync_client import FakeTimeSyncClient
from tsercom.timesync.client.time_sync_client import TimeSyncClient
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


class TimeSyncTracker:
    def __init__(
        self, thread_watcher: ThreadWatcher, *, is_testing: bool = False
    ):
        self.__thread_watcher = thread_watcher

        self.__map: Dict[str, tuple[int, TimeSyncClient]] = {}
        self.__is_test_run = is_testing

    def on_connect(self, ip: str) -> SynchronizedClock:
        if ip not in self.__map:
            if self.__is_test_run:
                new_client = FakeTimeSyncClient(self.__thread_watcher, ip)
            else:
                new_client = TimeSyncClient(self.__thread_watcher, ip)

            new_client.start_async()

            self.__map[ip] = (1, new_client)
            return new_client.get_synchronized_clock()

        else:
            # Retrieve and update the count.
            # Note: Tuples are immutable, so we have to create a new one.
            current_count, client_instance = self.__map[ip]
            current_count += 1
            self.__map[ip] = (current_count, client_instance)
            return client_instance.get_synchronized_clock()

    def on_disconnect(self, ip: str):
        if ip not in self.__map:
            raise KeyError(
                f"IP address '{ip}' not found in timesync tracker during disconnect. It may have already been disconnected or was never tracked."
            )

        # Retrieve and update the count.
        # Note: Tuples are immutable, so we have to create a new one.
        current_count, client_instance = self.__map[ip]
        current_count -= 1

        if current_count == 0:
            del self.__map[ip]
            client_instance.stop()
        else:
            self.__map[ip] = (current_count, client_instance)
