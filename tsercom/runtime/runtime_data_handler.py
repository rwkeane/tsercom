"""Abstract interface for runtime data handlers in Tsercom."""

from abc import ABC, abstractmethod
from typing import (
    AsyncIterator,
    Generic,
    List,
    TypeVar,
    overload,
    Any,
)

import grpc

from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeDataHandler(ABC, Generic[TDataType, TEventType]):
    """Defines the contract for handling data and caller registration for a runtime.

    This includes providing an iterator for event data, checking for caller IDs,
    and registering new callers.
    """

    @abstractmethod
    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Checks if a CallerIdentifier is registered for the given endpoint and port.

        Args:
            endpoint: The IP address or hostname of the endpoint.
            port: The port number of the endpoint.

        Returns:
            The `CallerIdentifier` if found, otherwise `None`.
        """
        pass

    @overload
    def register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[TDataType, TEventType]:  # Added TEventType
        pass

    @overload
    def register_caller(
        self, caller_id: CallerIdentifier, context: grpc.aio.ServicerContext
    ) -> (
        EndpointDataProcessor[TDataType, TEventType] | None
    ):  # Added TEventType
        pass

    @abstractmethod
    def register_caller(
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> (
        EndpointDataProcessor[TDataType, TEventType] | None
    ):  # Added TEventType
        """Registers a caller with the runtime.

        This method is responsible for associating a `CallerIdentifier` with
        its network endpoint, which can be provided directly via `endpoint` and
        `port` arguments, or extracted from a `grpc.aio.ServicerContext`.
        Implementations should handle the logic for creating or retrieving an
        `EndpointDataProcessor` for the caller.

        Args:
            caller_id: The unique identifier of the caller.
            endpoint: The network endpoint (e.g., IP address) of the caller.
                      Required if `context` is not provided.
            port: The port number of the caller. Required if `context` is not provided.
            context: The gRPC servicer context, from which endpoint information
                     can be extracted if `endpoint` and `port` are not given.

        Returns:
            An `EndpointDataProcessor` instance for the registered caller,
            or `None` if registration fails (e.g., context does not yield an address).
        """
        pass
