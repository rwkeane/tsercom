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
    """Contract for data handling and caller registration for a runtime.

    Includes providing event data iterator, checking caller IDs,
    and registering new callers.
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
            endpoint: Network endpoint (e.g., IP address) of the caller.
                      Required if `context` is not provided.
            port: Port number of the caller. Required if `context` not provided.
            context: gRPC servicer context, from which endpoint information
                     extracted if `endpoint` and `port` are not given.

        Returns:
            An `EndpointDataProcessor` for the registered caller,
            or `None` if registration fails (e.g. context has no address).
        """
