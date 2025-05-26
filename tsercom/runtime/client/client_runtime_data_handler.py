"""Implements the client-side runtime data handling logic.

This module defines `ClientRuntimeDataHandler` which is responsible for
managing data flow, caller registration, and time synchronization aspects
for Tsercom runtimes operating in a client role.
"""

from typing import Generic, TypeVar
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
import logging
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class ClientRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    """Handles data, events, and caller management for client runtimes.

    It integrates with a `TimeSyncTracker` for clock synchronization and
    an `IdTracker` to manage associations between caller IDs and their
    network endpoints. It processes incoming events and makes data
    available via a `RemoteDataReader`.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
        *,
        is_testing: bool = False,
    ):
        """Initializes the ClientRuntimeDataHandler.

        Args:
            thread_watcher: For monitoring internal threads.
            data_reader: The reader for incoming data instances.
            event_source: The poller for incoming event instances.
            is_testing: If True, enables testing-specific behaviors (e.g., fake time sync).
        """
        super().__init__(data_reader, event_source)

        self.__clock_tracker = TimeSyncTracker(
            thread_watcher, is_testing=is_testing
        )
        self.__id_tracker = IdTracker()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
        """Registers a new caller and its endpoint, returning a data processor.

        Adds the caller to the ID tracker and initializes time synchronization
        for the endpoint.

        Args:
            caller_id: The `CallerIdentifier` of the new caller.
            endpoint: The network endpoint (e.g., IP address) of the caller.
            port: The port number of the caller.

        Returns:
            An `EndpointDataProcessor` configured for this caller.
        """
        self.__id_tracker.add(caller_id, endpoint, port)
        clock = self.__clock_tracker.on_connect(endpoint)
        return self._create_data_processor(caller_id, clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """
        Unregisters a caller.

        Args:
            caller_id: The ID of the caller to unregister.

        Returns:
            True if the caller was found and unregistered, False otherwise.
        """
        address_port_tuple = self.__id_tracker.try_get(caller_id)

        if address_port_tuple is not None:
            address, _ = (
                address_port_tuple  # port is not needed for on_disconnect
            )
            self.__id_tracker.remove(caller_id)
            self.__clock_tracker.on_disconnect(address)
            return True
        else:
            logging.warning(
                f"Attempted to unregister non-existent caller_id: {caller_id}"
            )
            return False

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Tries to retrieve the CallerIdentifier for a given endpoint and port.

        Args:
            endpoint: The network endpoint of the caller.
            port: The port number of the caller.

        Returns:
            The `CallerIdentifier` if found, otherwise `None`.
        """
        return self.__id_tracker.try_get(endpoint, port)
