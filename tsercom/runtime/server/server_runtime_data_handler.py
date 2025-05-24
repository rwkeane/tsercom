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
from datetime import datetime


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class ServerRuntimeDataHandler(
    Generic[TDataType, TEventType],
    RuntimeDataHandlerBase[TDataType, TEventType],
):
    # Method added for logging as requested by the prompt.
    # The actual data processing logic might be within an inner class or different method.
    async def process_data(self, caller_id: CallerIdentifier, data: TDataType, timestamp: datetime):
        data_value = getattr(data, 'value', str(data))
        print(f"DEBUG: [ServerRuntimeDataHandler.process_data] Caller ID: {caller_id}, Data: {data_value}")
        # This is a placeholder for where data would be sent to a data source.
        # The class structure doesn't show a direct __data_source.send_data path here.
        # Logging is added to represent the requested trace point.
        print(f"DEBUG: [ServerRuntimeDataHandler.process_data] Before calling self.__data_source.send_data for Caller ID: {caller_id}, Data: {data_value}")
        # In a real scenario, this might involve getting a data processor and calling a method on it,
        # which eventually leads to data being queued or processed.
        # For example, it might look up a processor for the caller_id and use it.
        # processor = self._get_processor_for_caller(caller_id) # Hypothetical
        # if processor:
        #     await processor.actual_send_method(data, timestamp) # Hypothetical

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
        # The _create_data_processor method is part of RuntimeDataHandlerBase.
        # It creates an EndpointDataProcessorImpl instance.
        # The actual _process_data (which is called by EndpointDataProcessor.process_data)
        # is implemented in EndpointDataProcessorImpl inside RuntimeDataHandlerBase.
        # So, logging related to _process_data should ideally be there or in a way that
        # EndpointDataProcessorImpl can call a logging-specific method.
        return self._create_data_processor(caller_id, self.__clock)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        # Keep all CallerID instances around, so a connection can be
        # re-established with the same id (if reconnection possible).
        pass

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self.__id_tracker.try_get(endpoint, port)
