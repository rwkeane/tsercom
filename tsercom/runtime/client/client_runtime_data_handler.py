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
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
        *,
        is_testing: bool = False,
    ):
        super().__init__(data_reader, event_source)

        self.__clock_tracker = TimeSyncTracker(
            thread_watcher, is_testing=is_testing
        )
        self.__id_tracker = IdTracker()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
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
            address, _ = address_port_tuple  # port is not needed for on_disconnect
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
        return self.__id_tracker.try_get(endpoint, port)
