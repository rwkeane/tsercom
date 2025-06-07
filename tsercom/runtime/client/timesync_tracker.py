"""Manages time synchronization with multiple IP endpoints."""

import logging
from typing import Dict

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
from tsercom.timesync.client.fake_time_sync_client import FakeTimeSyncClient
from tsercom.timesync.client.time_sync_client import TimeSyncClient
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

logger = logging.getLogger(__name__)


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
            thread_watcher: For monitoring threads by TimeSyncClient.
            is_testing: If True, uses `FakeTimeSyncClient`.
        """
        self.__thread_watcher = thread_watcher
        self.__map: Dict[str, tuple[int, ClientSynchronizedClock.Client]] = {}
        self.__is_test_run = is_testing

    def on_connect(self, ip: str) -> SynchronizedClock:
        """Handles a new connection to an IP endpoint for time synchronization.

        If no `TimeSyncClient` for IP exists, creates and starts one.
        It increments a reference count for the IP.

        Args:
            ip: The IP address of the endpoint to connect to.

        Returns:
            A `SynchronizedClock` instance associated with the endpoint.
        """
        new_client: ClientSynchronizedClock.Client
        if ip not in self.__map:
            if self.__is_test_run:
                new_client = FakeTimeSyncClient(self.__thread_watcher, ip)
            else:
                new_client = TimeSyncClient(self.__thread_watcher, ip)
            new_client.start_async()
            self.__map[ip] = (1, new_client)
            return new_client.get_synchronized_clock()

        current_count, client_instance = self.__map[ip]
        current_count += 1
        self.__map[ip] = (current_count, client_instance)
        return client_instance.get_synchronized_clock()

    def on_disconnect(self, ip: str) -> None:
        """Handles a disconnection from an IP endpoint.

        Decrements the reference count for the IP. If count reaches zero,
        stops and removes the `TimeSyncClient` for that IP.

        Args:
            ip: The IP address of the disconnected endpoint.

        Raises:
            KeyError: If the IP was not previously tracked.
        """
        if ip not in self.__map:
            # pylint: disable=consider-using-f-string
            msg = (
                "IP address '%s' not found in timesync tracker during "
                "disconnect. May have already been disconnected or "
                "never tracked." % ip
            )
            raise KeyError(msg)

        current_count, client_instance = self.__map[ip]
        current_count -= 1

        if current_count == 0:
            del self.__map[ip]
            client_instance.stop()
        else:
            self.__map[ip] = (current_count, client_instance)
