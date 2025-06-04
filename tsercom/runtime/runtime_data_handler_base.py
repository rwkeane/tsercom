"""Base implementation for `RuntimeDataHandler`."""

# pylint: disable=W0221 # Allow arguments-differ for register_caller flexibility

from abc import abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Generic, List, TypeVar, Any, overload

import grpc  # Standard import

# Local application imports
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.rpc.grpc_util.addressing import get_client_ip, get_client_port
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.proto import ServerTimestamp
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class RuntimeDataHandlerBase(
    Generic[DataTypeT, EventTypeT], RuntimeDataHandler[DataTypeT, EventTypeT]
):
    """Provides common functionality for runtime data handlers.

    Manages data reading via `RemoteDataReader` and event polling via
    `AsyncPoller`. Handles caller registration by parsing context or direct
    endpoint info. Implements async iterator for event data.
    """

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
    ):
        """Initializes RuntimeDataHandlerBase.

        Args:
            data_reader: The `RemoteDataReader` to sink data into.
            event_source: The `AsyncPoller` to source event data from.
        """
        super().__init__()
        self.__data_reader = data_reader
        self.__event_source = event_source

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

    # pylint: disable=too-many-branches # Complex argument parsing logic
    def register_caller(  # pylint: disable=arguments-differ # Actual signature uses *args, **kwargs
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> EndpointDataProcessor[DataTypeT] | None:
        """Registers a caller using either endpoint/port or gRPC context.

        This impl of `RuntimeDataHandler.register_caller` validates inputs,
        extracts endpoint/port from context if provided, then delegates
        to `_register_caller`.

        Args:
            caller_id: The `CallerIdentifier` of the caller.
            *args: Can contain (endpoint, port) or (context,).
            **kwargs: Can contain endpoint="...", port=123 or context=ctx.

        Returns:
            An `EndpointDataProcessor` for the caller, or `None` if
            registration fails.

        Raises:
            ValueError: If arguments are inconsistent.
            TypeError: If context is not of the expected type.
        """
        _endpoint: str | None = None
        _port: int | None = None
        _context: grpc.aio.ServicerContext | None = None

        if len(args) == 1 and isinstance(args[0], grpc.aio.ServicerContext):
            _context = args[0]
        elif (
            len(args) == 2
            and isinstance(args[0], str)
            and isinstance(args[1], int)
        ):
            _endpoint = args[0]
            _port = args[1]
        elif args:
            # pylint: disable=consider-using-f-string
            msg = (
                "Unexpected positional args: %s. Provide (endpoint, port) "
                "or (context,)." % (args,)
            )
            raise ValueError(msg)

        if "endpoint" in kwargs:
            if _endpoint is not None or _context is not None:
                raise ValueError(
                    "Cannot use endpoint via both args & kwargs, or with context."
                )
            _endpoint = kwargs.pop("endpoint")
        if "port" in kwargs:
            if _port is not None or _context is not None:
                raise ValueError(
                    "Cannot use port via both args & kwargs, or with context."
                )
            _port = kwargs.pop("port")
        if "context" in kwargs:
            if (
                _context is not None
                or _endpoint is not None
                or _port is not None
            ):
                raise ValueError(
                    "Cannot use context via args & kwargs, or with endpoint/port."
                )
            _context = kwargs.pop("context")

        if kwargs:
            # pylint: disable=consider-using-f-string
            msg = "Unexpected keyword arguments: %s" % list(kwargs.keys())
            raise ValueError(msg)

        if (_endpoint is None and _port is None) == (_context is None):
            raise ValueError(
                "Provide (endpoint/port) or context, not both/neither."
            )
        if (_port is None) != (_endpoint is None):
            raise ValueError(
                "If 'endpoint' provided, 'port' must be too, and vice-versa."
            )

        actual_endpoint: str
        actual_port: int

        if _context is not None:
            if not isinstance(_context, grpc.aio.ServicerContext):
                # pylint: disable=consider-using-f-string
                msg = (
                    "Expected context: grpc.aio.ServicerContext, got %s."
                    % type(_context).__name__
                )
                raise TypeError(msg)
            extracted_endpoint = get_client_ip(_context)
            extracted_port = get_client_port(_context)

            if extracted_endpoint is None:
                return None
            if extracted_port is None:
                # pylint: disable=consider-using-f-string
                msg = (
                    "Could not get client port from context for endpoint: %s."  # Shortened
                    % extracted_endpoint
                )
                raise ValueError(msg)
            actual_endpoint = extracted_endpoint
            actual_port = extracted_port
        elif _endpoint is not None and _port is not None:
            actual_endpoint = _endpoint
            actual_port = _port
        else:
            raise ValueError(  # Line 147 (Pylint) - Shortened
                "Internal error: Inconsistent endpoint/port/context state."
            )

        return self._register_caller(caller_id, actual_endpoint, actual_port)

    def get_data_iterator(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
        """Returns async iterator for event data, grouped by CallerIdentifier.

        This impl returns `self` as it implements `__aiter__` / `__anext__`.
        """  # Line 167 (Pylint) - Shortened
        return self

    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Checks for existing caller ID using `_try_get_caller_id`.

        Args:
            endpoint: IP address or hostname.
            port: Port number.

        Returns:
            `CallerIdentifier` if found, `None` otherwise.
        """
        return self._try_get_caller_id(endpoint, port)

    async def _on_data_ready(self, data: AnnotatedInstance[DataTypeT]) -> None:
        """Callback for new data to be processed by the data reader.

        Args:
            data: The `AnnotatedInstance` with data and metadata.
        """
        if self.__data_reader is None:
            raise ValueError("Data reader instance is None.")
        # pylint: disable=protected-access # Internal component call
        self.__data_reader._on_data_ready(data)

    @abstractmethod
    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT]:
        """Template for subclass-specific caller registration logic.

        Args:
            caller_id: `CallerIdentifier` of the caller.
            endpoint: IP address or hostname of the caller.
            port: Port number of the caller.

        Returns:
            An `EndpointDataProcessor` for the caller.
        """

    @abstractmethod
    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Template for subclass-specific caller deregistration.

        Args:
            caller_id: `CallerIdentifier` of the caller to unregister.

        Returns:
            True if caller was found and unregistered, False otherwise.
        """

    @abstractmethod
    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Template for subclass logic to find caller ID by endpoint.

        Args:
            endpoint: IP address or hostname.
            port: Port number.

        Returns:
            `CallerIdentifier` if found, `None` otherwise.
        """

    async def __anext__(
        self,
    ) -> List[SerializableAnnotatedInstance[EventTypeT]]:
        """Retrieves next batch of events from the event source."""
        return await self.__event_source.__anext__()

    def __aiter__(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
        """Returns self as the async iterator for events."""
        return self

    def _create_data_processor(
        self, caller_id: CallerIdentifier, clock: SynchronizedClock
    ) -> EndpointDataProcessor[DataTypeT]:
        """Factory method to create a concrete `EndpointDataProcessor`.

        Args:
            caller_id: `CallerIdentifier` for the processor.
            clock: `SynchronizedClock` to be used by the processor.

        Returns:
            An instance of `_DataProcessorImpl`.
        """
        return RuntimeDataHandlerBase._DataProcessorImpl(
            self, caller_id, clock
        )

    class _DataProcessorImpl(EndpointDataProcessor[DataTypeT]):
        """Concrete `EndpointDataProcessor` for `RuntimeDataHandlerBase`.

        Handles data desync, caller deregistration, and data processing
        by delegating to the outer `RuntimeDataHandlerBase` instance.
        """

        def __init__(
            self,
            data_handler: "RuntimeDataHandlerBase[DataTypeT, EventTypeT]",
            caller_id: CallerIdentifier,
            clock: SynchronizedClock,
        ):
            """Initializes the _DataProcessorImpl.

            Args:
                data_handler: Parent `RuntimeDataHandlerBase` instance.
                caller_id: `CallerIdentifier` for this processor.
                clock: `SynchronizedClock` for timestamp desynchronization.
            """
            super().__init__(caller_id)
            self.__data_handler = data_handler
            self.__clock = clock

        async def desynchronize(self, timestamp: ServerTimestamp) -> datetime:
            """Desynchronizes a server timestamp using the provided clock."""
            return self.__clock.desync(timestamp)

        async def deregister_caller(self) -> None:
            """Deregisters caller via the parent data handler."""
            # Return value of _unregister_caller (bool) is ignored
            # to match supertype's None return.
            # pylint: disable=protected-access # Outer class call
            self.__data_handler._unregister_caller(self.caller_id)
            return None

        async def _process_data(
            self, data: DataTypeT, timestamp: datetime
        ) -> None:
            """Processes data by wrapping it and passing to parent."""
            wrapped_data = AnnotatedInstance(
                caller_id=self.caller_id, timestamp=timestamp, data=data
            )
            # pylint: disable=protected-access # Outer class call
            await self.__data_handler._on_data_ready(wrapped_data)
