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
        """Register a caller using an explicit endpoint address and port.

        This overload is used when the network location (IP address or hostname
        and port) of the caller is known directly.

        Args:
            caller_id: The unique identifier for the caller.
                (Type: CallerIdentifier)
            endpoint: The network endpoint string (e.g., IP address or hostname)
                of the caller. (Type: str)
            port: The network port number of the caller. (Type: int)

        Returns:
            EndpointDataProcessor[DataTypeT, EventTypeT]: An endpoint processor
            instance for the registered caller.

        """
        pass

    @overload
    async def register_caller(
        self, caller_id: CallerIdentifier, context: grpc.aio.ServicerContext
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT] | None:
        """Register a caller using a gRPC service context.

        This overload is typically used on the server side of a gRPC call,
        where the caller's network information can be extracted from the
        provided `ServicerContext`.

        Args:
            caller_id: The unique identifier for the caller.
                (Type: CallerIdentifier)
            context: The gRPC asynchronous servicer context from which the
                caller's endpoint information (IP and port) will be extracted.
                (Type: grpc.aio.ServicerContext)

        Returns:
            Optional[EndpointDataProcessor[DataTypeT, EventTypeT]]: An endpoint
            processor instance for the registered caller if its network address
            can be successfully extracted from the context. Returns `None` if
            the address cannot be determined (e.g., for non-IP based transports
            or if context does not provide peer info).

        """
        pass

    @abstractmethod
    async def register_caller(
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT] | None:
        """Register a caller with the runtime.

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
