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


DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class RuntimeDataHandler(ABC, Generic[DataTypeT, EventTypeT]):
    """Contract for data handling and caller registration for a runtime.

    Includes providing event data iterator, checking caller IDs,
    and registering new callers.
    """

    @abstractmethod
    def get_data_iterator(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
        """Returns async iterator for lists of serializable event instances.

        Yields:
            Lists of `SerializableAnnotatedInstance[EventTypeT]`.
        """

    @abstractmethod
    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Checks if CallerID is registered for a given endpoint and port.

        Args:
            endpoint: The IP address or hostname of the endpoint.
            port: The port number of the endpoint.

        Returns:
            The `CallerIdentifier` if found, otherwise `None`.
        """

    @overload
    def register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT]:
        pass

    @overload
    def register_caller(
        self, caller_id: CallerIdentifier, context: grpc.aio.ServicerContext
    ) -> EndpointDataProcessor[DataTypeT] | None:
        pass

    @abstractmethod
    def register_caller(
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> EndpointDataProcessor[DataTypeT] | None:
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
