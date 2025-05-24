from abc import abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Dict, Generic, List, Optional, TypeVar

import grpc

from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.rpc.grpc.addressing import get_client_ip, get_client_port
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.proto import ServerTimestamp
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class RuntimeDataHandlerBase(
    Generic[TDataType, TEventType], RuntimeDataHandler[TDataType, TEventType]
):
    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
    ):
        super().__init__()
        print(f"DEBUG: [RuntimeDataHandlerBase.__init__] Received data_reader arg: {data_reader}, id(data_reader arg): {id(data_reader)}")
        print(f"DEBUG: [RuntimeDataHandlerBase.__init__] Received event_source arg: {event_source}, id(event_source arg): {id(event_source)}")

        self.__data_reader = data_reader
        self.__event_source = event_source
        print(f"DEBUG: [RuntimeDataHandlerBase.__init__] After assignment - self.__data_reader: {self.__data_reader}, id(self.__data_reader): {id(self.__data_reader)}")
        print(f"DEBUG: [RuntimeDataHandlerBase.__init__] After assignment - self.__event_source: {self.__event_source}, id(self.__event_source): {id(self.__event_source)}")


    def register_caller(
        self,
        caller_id: CallerIdentifier,
        endpoint: Optional[str] = None,
        port: Optional[int] = None,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> EndpointDataProcessor | None:
        assert (endpoint is None) != (context is None)
        assert (port is None) == (endpoint is None)

        if context is not None:
            assert isinstance(context, grpc.aio.ServicerContext)
            endpoint = get_client_ip(caller_id) 
            port = get_client_port(caller_id) 
            if endpoint is None:
                return None
            assert port is not None

        return self._register_caller(caller_id, endpoint, port)

    def get_data_iterator(
        self,
    ) -> AsyncIterator[
        Dict[CallerIdentifier, List[SerializableAnnotatedInstance]]
    ]:
        return self

    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self._try_get_caller_id(endpoint, port)

    async def _on_data_ready(self, data: AnnotatedInstance[TDataType]): # Changed to async def
        data_value = getattr(getattr(data, 'data', data), 'value', str(data))
        print(f"DEBUG: [RuntimeDataHandlerBase._on_data_ready] id(self): {id(self)}. self.__data_reader: {self.__data_reader}, id(self.__data_reader): {id(self.__data_reader)}. Forwarding data: {data_value}, Caller ID: {data.caller_id}")
        if self.__data_reader is None:
            print(f"ERROR: [RuntimeDataHandlerBase._on_data_ready] self.__data_reader is None. Cannot call _on_data_ready on it. id(self): {id(self)}")
            # If it's None, and this method is awaited, it must return something awaitable or raise.
            # However, the original error was that self.__data_reader itself was None, so the call failed.
            # Now that self.__data_reader should be valid, this path might not be hit for that reason.
            # If it were to be hit, simply returning None (implicitly) from an async def is fine.
        else:
            # Assuming self.__data_reader._on_data_ready is NOT async, so no await here.
            # DataReaderSink._on_data_ready uses put_nowait, which is synchronous.
            self.__data_reader._on_data_ready(data)

    @abstractmethod
    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor:
        pass

    @abstractmethod
    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        pass

    @abstractmethod
    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        pass

    async def __anext__(
        self,
    ) -> Dict[CallerIdentifier, List[SerializableAnnotatedInstance]]:
        return await self.__event_source.__anext__()

    async def __aiter__(
        self,
    ) -> AsyncIterator[
        Dict[CallerIdentifier, List[SerializableAnnotatedInstance]]
    ]:
        return self

    def _create_data_processor(
        self, caller_id: CallerIdentifier, clock: SynchronizedClock
    ):
        print(f"DEBUG: [RuntimeDataHandlerBase._create_data_processor] id(self): {id(self)}. Creating __DataProcessorImpl for caller_id: {caller_id}")
        return RuntimeDataHandlerBase.__DataProcessorImpl(self, caller_id, clock)


    class __DataProcessorImpl(EndpointDataProcessor):
        def __init__(
            self,
            data_handler: "RuntimeDataHandlerBase", 
            caller_id: CallerIdentifier,
            clock: SynchronizedClock, 
        ):
            super().__init__(caller_id)
            self.__data_handler = data_handler
            self.__clock = clock
            print(f"DEBUG: [__DataProcessorImpl.__init__] Initialized. id(self): {id(self)}, data_handler id: {id(self.__data_handler)}")


        async def desynchronize(self, timestamp: ServerTimestamp) -> datetime:
            return self.__clock.desync(timestamp)

        async def deregister_caller(self):
            self.__data_handler._unregister_caller(self.caller_id)

        async def _process_data(self, data: TDataType, timestamp: datetime):
            data_value = getattr(data, 'value', str(data))
            print(f"DEBUG: [EndpointDataProcessorImpl._process_data] id(self): {id(self)}. Wrapping data. Data: {data_value}, Caller ID: {self.caller_id}, Timestamp: {timestamp}")
            wrapped_data = AnnotatedInstance(data, self.caller_id, timestamp)
            print(f"DEBUG: [EndpointDataProcessorImpl._process_data] id(self): {id(self)}. Calling self.__data_handler._on_data_ready (id: {id(self.__data_handler)}). Wrapped Data: {data_value}, Caller ID: {self.caller_id}")
            await self.__data_handler._on_data_ready(wrapped_data)
