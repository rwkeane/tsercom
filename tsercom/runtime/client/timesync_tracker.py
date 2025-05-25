"""Provides TimeSyncTracker for managing time synchronization with multiple endpoints."""

from typing import Dict
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.fake_time_sync_client import FakeTimeSyncClient
from tsercom.timesync.client.time_sync_client import TimeSyncClient
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


class TimeSyncTracker:
    """Manages TimeSyncClient instances for various IP endpoints.

    This tracker ensures that only one `TimeSyncClient` is active per IP
    address. It reference counts connections to an IP and starts/stops
    the `TimeSyncClient` accordingly. In testing mode, it can use
    `FakeTimeSyncClient`.
    """

    def __init__(
        self, thread_watcher: ThreadWatcher, *, is_testing: bool = False
    ):
        """Initializes the TimeSyncTracker.

        Args:
            thread_watcher: For monitoring threads created by TimeSyncClient.
            is_testing: If True, uses `FakeTimeSyncClient` instead of real `TimeSyncClient`.
        """
        self.__thread_watcher = thread_watcher

        self.__map: Dict[str, tuple[int, TimeSyncClient]] = {}
        self.__is_test_run = is_testing

    def on_connect(self, ip: str) -> SynchronizedClock:
        """Handles a new connection to an IP endpoint for time synchronization.

        If a `TimeSyncClient` for the given IP doesn't exist, it creates and
        starts one. It increments a reference count for the IP.

        Args:
            ip: The IP address of the endpoint to connect to.

        Returns:
            A `SynchronizedClock` instance associated with the endpoint.
        """
        if ip not in self.__map:
            # If this is the first connection to this IP, create a new TimeSyncClient.
            if self.__is_test_run:
                new_client = FakeTimeSyncClient(self.__thread_watcher, ip)
            else:
                new_client = TimeSyncClient(self.__thread_watcher, ip)

            new_client.start_async()

            self.__map[ip] = (1, new_client)
            return new_client.get_synchronized_clock()

        else:
            # Note: Tuples are immutable, so we have to create a new one.
            current_count, client_instance = self.__map[ip]
            current_count += 1
            self.__map[ip] = (current_count, client_instance)
            return client_instance.get_synchronized_clock()

    def on_disconnect(self, ip: str):
        """Handles a disconnection from an IP endpoint.

        Decrements the reference count for the IP. If the count reaches zero,
        stops and removes the `TimeSyncClient` for that IP.

        Args:
            ip: The IP address of the disconnected endpoint.

        Raises:
            KeyError: If the IP was not previously tracked.
        """
        if ip not in self.__map:
            raise KeyError(
                f"IP address '{ip}' not found in timesync tracker during disconnect. It may have already been disconnected or was never tracked."
            )

        current_count, client_instance = self.__map[ip]
        current_count -= 1

        if current_count == 0:
            # If this was the last connection, stop and remove the client.
            del self.__map[ip]
            client_instance.stop()
        else:
            # Otherwise, just decrement the reference count.
            self.__map[ip] = (current_count, client_instance)
