from typing import Generic, TypeVar
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)
from tsercom.timesync.server.time_sync_server import TimeSyncServer
# datetime was imported but not used after removing print statements
# from datetime import datetime 


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class ServerRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    # The 'process_data' method was added for debugging and is not part of the original class structure.
    # It is being removed as per the cleanup task.
    # async def process_data(self, caller_id: CallerIdentifier, data: TDataType, timestamp: datetime):
    #     data_value = getattr(data, 'value', str(data))
    #     # print(f"DEBUG: [ServerRuntimeDataHandler.process_data] Caller ID: {caller_id}, Data: {data_value}")
    #     # print(f"DEBUG: [ServerRuntimeDataHandler.process_data] Before calling self.__data_source.send_data for Caller ID: {caller_id}, Data: {data_value}")
    #     pass

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
        *,
        is_testing: bool = False,
    ):
        super().__init__(data_reader, event_source)

        self.__id_tracker = IdTracker()

        if is_testing:
            self.__clock = FakeSynchronizedClock()
            return

        self.__server = TimeSyncServer()
        self.__server.start_async()

        self.__clock = self.__server.get_synchronized_clock()

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
        self.__id_tracker.add(caller_id, endpoint, port)
        return self._create_data_processor(caller_id, self.__clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        # Keep all CallerID instances around, so a connection can be
        # re-established with the same id (if reconnection possible).
        pass

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self.__id_tracker.try_get(endpoint, port)
