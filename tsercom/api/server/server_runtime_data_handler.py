from typing import Generic, TypeVar
from tsercom.api.endpoint_data_processor import EndpointDataProcessor
from tsercom.api.id_tracker import IdTracker
from tsercom.api.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.server.time_sync_server import TimeSyncServer


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class ServerRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    def __init__(self):
        super().__init__()

        self.__server = TimeSyncServer()
        self.__server.start_async()

        self.__clock = self.__server.get_synchronized_clock()

        self.__id_tracker = IdTracker()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
        self.__id_tracker.add(caller_id, endpoint, port)
        return self._create_data_processor(caller_id, self.__clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        # Keep all CallerID instances around, so a connection can be
        # re-established if possible.
        pass

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self.__id_tracker.try_get(endpoint, port)
