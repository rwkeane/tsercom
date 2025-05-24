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
        self.__data_reader = data_reader
        self.__event_source = event_source

    def register_caller(
        self,
        caller_id: CallerIdentifier,
        endpoint: Optional[str] = None,
        port: Optional[int] = None,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> EndpointDataProcessor | None:
        if (endpoint is None) == (context is None):
            raise ValueError(
                "Exactly one of 'endpoint'/'port' combination or 'context' must be provided to register_caller. "
                f"Got endpoint={endpoint}, context={'<Provided>' if context is not None else None}."
            )
        # This check implies that if endpoint is not None, port must not be None.
        # And if endpoint is None, port must be None.
        if (port is None) != (endpoint is None):
            raise ValueError(
                "If 'endpoint' is provided, 'port' must also be provided. If 'endpoint' is None, 'port' must also be None. "
                f"Got endpoint={endpoint}, port={port}."
            )

        if context is not None:
            if not isinstance(context, grpc.aio.ServicerContext):
                raise TypeError(
                    f"Expected context to be an instance of grpc.aio.ServicerContext, but got {type(context).__name__}."
                )
            endpoint = get_client_ip(context)
            port = get_client_port(context)

            if endpoint is None:
                # If endpoint is None, we cannot register the caller.
                # This case is already handled by returning None.
                return None
            if port is None:
                # If port is None, but endpoint was determined, this indicates an unexpected issue.
                raise ValueError(
                    f"Could not determine client port from context for endpoint {endpoint}."
                )

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

    async def _on_data_ready(self, data: AnnotatedInstance[TDataType]):
        if self.__data_reader is None:
            # This case should ideally not be hit if initialization is correct
            # Consider logging an error or raising if it's an invalid state
            return
        # DataReaderSink._on_data_ready is synchronous
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

        async def desynchronize(self, timestamp: ServerTimestamp) -> datetime:
            return self.__clock.desync(timestamp)

        async def deregister_caller(self):
            self.__data_handler._unregister_caller(self.caller_id)

        async def _process_data(self, data: TDataType, timestamp: datetime):
            wrapped_data = AnnotatedInstance(data, self.caller_id, timestamp)
            await self.__data_handler._on_data_ready(wrapped_data)
