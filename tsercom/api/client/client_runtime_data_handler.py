from typing import Generic, TypeVar
from tsercom.api.client.timesync_tracker import TimeSyncTracker
from tsercom.api.endpoint_data_processor import EndpointDataProcessor
from tsercom.api.id_tracker import IdTracker
from tsercom.api.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.threading.thread_watcher import ThreadWatcher


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class ClientRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    def __init__(self, thread_watcher: ThreadWatcher):
        super().__init__()

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
