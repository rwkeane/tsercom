"""Provides TimeSyncTracker for managing time synchronization with multiple endpoints."""

from typing import Dict, Optional  # Added Optional
from tsercom.caller_id.caller_identifier import CallerIdentifier  # Added
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
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
        self.__thread_watcher = thread_watcher  # Standard private
        self.__ip_to_client_map: Dict[
            str, tuple[int, ClientSynchronizedClock.Client]
        ] = {}  # Standard private
        self.__is_test_run = is_testing  # Standard private
        self.__endpoint_to_clock_obj: Dict[str, SynchronizedClock] = (
            {}
        )  # Standard private
        self.__caller_id_to_endpoint: Dict[CallerIdentifier, str] = (
            {}
        )  # Standard private

    def on_connect(
        self, ip: str, caller_id: CallerIdentifier
    ) -> SynchronizedClock:
        """Handles a new connection to an IP endpoint for time synchronization.

        If a `TimeSyncClient` for the given IP doesn't exist, it creates and
        starts one. It increments a reference count for the IP.

        Args:
            ip: The IP address of the endpoint to connect to.

        Returns:
            A `SynchronizedClock` instance associated with the endpoint.
        """
        client_instance: ClientSynchronizedClock.Client
        if ip not in self.__ip_to_client_map:  # Use standard private
            # If this is the first connection to this IP, create a new TimeSyncClient.
            if self.__is_test_run:  # Use standard private
                client_instance = FakeTimeSyncClient(
                    self.__thread_watcher, ip
                )  # Use standard private
            else:
                client_instance = TimeSyncClient(
                    self.__thread_watcher, ip
                )  # Use standard private

            client_instance.start_async()
            self.__ip_to_client_map[ip] = (
                1,
                client_instance,
            )  # Use standard private
        else:
            # Note: Tuples are immutable, so we have to create a new one.
            current_count, existing_client_instance = self.__ip_to_client_map[
                ip
            ]  # Use standard private
            current_count += 1
            client_instance = existing_client_instance  # Use existing client
            self.__ip_to_client_map[ip] = (
                current_count,
                client_instance,
            )  # Use standard private

        clock = client_instance.get_synchronized_clock()
        self.__endpoint_to_clock_obj[ip] = clock  # Use standard private
        self.__caller_id_to_endpoint[caller_id] = ip  # Use standard private
        return clock

    def on_disconnect(self, ip: str) -> None:
        """Handles a disconnection from an IP endpoint.

        Decrements the reference count for the IP. If the count reaches zero,
        stops and removes the `TimeSyncClient` for that IP.

        Args:
            ip: The IP address of the disconnected endpoint.

        Raises:
            KeyError: If the IP was not previously tracked.
        """
        if ip not in self.__ip_to_client_map:  # Use standard private
            raise KeyError(
                f"IP address '{ip}' not found in timesync tracker during disconnect. It may have already been disconnected or was never tracked."
            )

        current_count, client_instance = self.__ip_to_client_map[
            ip
        ]  # Use standard private
        current_count -= 1

        if current_count == 0:
            # If this was the last connection, stop and remove the client.
            del self.__ip_to_client_map[ip]  # Use standard private
            client_instance.stop()
            self.__endpoint_to_clock_obj.pop(ip, None)  # Use standard private
            # Find and remove caller_ids associated with this ip
            callers_to_remove = [
                cid
                for cid, e_ip in self.__caller_id_to_endpoint.items()
                if e_ip == ip  # Use standard private
            ]
            for cid in callers_to_remove:
                del self.__caller_id_to_endpoint[cid]  # Use standard private
        else:
            # Otherwise, just decrement the reference count.
            self.__ip_to_client_map[ip] = (
                current_count,
                client_instance,
            )  # Use standard private

    def get_clock_for_endpoint(
        self, endpoint: str
    ) -> Optional[SynchronizedClock]:
        """Retrieves the SynchronizedClock for a given endpoint if one is active."""
        return self.__endpoint_to_clock_obj.get(
            endpoint
        )  # Use standard private

    def get_clock_for_caller_id(
        self, caller_id: CallerIdentifier
    ) -> Optional[SynchronizedClock]:
        """Retrieves the SynchronizedClock for a given CallerIdentifier if one is active."""
        endpoint = self.__caller_id_to_endpoint.get(
            caller_id
        )  # Use standard private
        if endpoint:
            return self.__endpoint_to_clock_obj.get(
                endpoint
            )  # Use standard private
        return None
