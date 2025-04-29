from typing import Generic, TypeVar
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
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
    ):
        super().__init__(data_reader, event_source)

        self.__clock_tracker = TimeSyncTracker(thread_watcher)
        self.__id_tracker = IdTracker()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
        self.__id_tracker.add(caller_id, endpoint, port)
        clock = self.__clock_tracker.on_connect(endpoint)
        return self._create_data_processor(caller_id, clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> None:
        address, port = self.__id_tracker.get(caller_id)
        assert False, "Find out if I should be keeping or deleting these?"
        self.__clock_tracker.on_disconnect(address)

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self.__id_tracker.try_get(endpoint, port)
