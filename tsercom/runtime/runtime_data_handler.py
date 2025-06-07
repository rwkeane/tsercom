"""Abstract interface for runtime data handlers in Tsercom."""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)

import grpc

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor

DataTypeT = TypeVar("DataTypeT")
EventTypeT = TypeVar("EventTypeT")


class RuntimeDataHandler(ABC, Generic[DataTypeT, EventTypeT]):
    """Abstract contract for data handling and caller registration within a runtime.

    This interface defines the essential operations for managing data flow
    and caller interactions in a Tsercom runtime. Key responsibilities include
    registering new callers (endpoints), providing mechanisms to check for
    existing caller IDs, and potentially offering an iterator for event data.

    Type Args:
        DataTypeT: The generic type of data objects that this handler processes.
        EventTypeT: The generic type of event objects that this handler processes.
    """

    @overload
    async def register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]:
        pass

    @overload
    async def register_caller(
        self, caller_id: CallerIdentifier, context: grpc.aio.ServicerContext
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT] | None:
        pass

    @abstractmethod
    async def register_caller(
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT] | None:
        """Registers a caller with the runtime.

        This method associates a `CallerIdentifier` with its network endpoint
        (via `endpoint` and `port` args), or extracted from a
        `grpc.aio.ServicerContext`. Implementations handle creating or
        retrieving an `EndpointDataProcessor` for the caller.

        Args:
            caller_id: The unique identifier of the caller.
            *args: Can be `(endpoint_str, port_int)` or `(grpc_context)`.
                See `RuntimeDataHandlerBase` for detailed argument processing.
            **kwargs: Can be `endpoint="...", port=123` or `context=grpc_context`.
                See `RuntimeDataHandlerBase` for detailed argument processing.

        Returns:
            An `EndpointDataProcessor` instance configured for the registered
            caller if successful. Returns `None` if registration fails (e.g.,
            if IP address cannot be extracted from the gRPC context).
        """
