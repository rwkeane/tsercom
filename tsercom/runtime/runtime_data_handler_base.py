"""Provides the abstract base for runtime data handlers in Tsercom.

This module defines `RuntimeDataHandlerBase`, an abstract class that establishes
common functionality for handling data and events within Tsercom runtimes.
It integrates an `IdTracker` for managing caller identities and their associated
event pollers, a `RemoteDataReader` for sinking processed data, and an
`AsyncPoller` for sourcing incoming events.

Concrete implementations, such as `ClientRuntimeDataHandler` and
`ServerRuntimeDataHandler`, inherit from this base to provide specific logic
for client and server-side operations, respectively.
"""

from abc import abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import (
    Any,
    Generic,
    List,
    Optional,
    TypeVar,
    overload,
)

import grpc

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.rpc.grpc_util.addressing import get_client_ip, get_client_port
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.common.proto import ServerTimestamp
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT")


# pylint: disable=arguments-differ # Handled by *args, **kwargs in actual implementation matching abstract
class RuntimeDataHandlerBase(
    Generic[DataTypeT, EventTypeT], RuntimeDataHandler[DataTypeT, EventTypeT]
):
    """Abstract base class providing common functionality for runtime data handlers.

    This class manages the core components for data and event flow within a
    Tsercom runtime. It uses a `RemoteDataReader` to output data, an `AsyncPoller`
    as the primary source for incoming events, and an `IdTracker` to manage
    `CallerIdentifier` associations and per-caller `AsyncPoller` instances for
    event dispatching.

    Key responsibilities include:
    - Initializing and holding references to data reader, main event source,
      and ID tracker.
    - Providing a mechanism to register callers (via endpoint/port or gRPC context)
      and delegate to subclass-specific registration logic.
    - Implementing the async iterator protocol (`__aiter__`, `__anext__`) to stream
      events from the main event source.
    - Offering a factory method (`_create_data_processor`) for creating
      concrete `EndpointDataProcessor` instances.
    - Dispatching incoming events from the main event source to per-caller pollers.

    Subclasses must implement `_register_caller` and `_unregister_caller` to
    define specific behaviors for client or server roles.

    Type Args:
        DataTypeT: The generic type of data objects that this handler deals with.
        EventTypeT: The type of event objects that this handler processes.
    """

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
        min_send_frequency_seconds: float | None = None,
    ):
        """Initializes the RuntimeDataHandlerBase.

        Args:
            data_reader: The `RemoteDataReader` instance where processed data
                (as `AnnotatedInstance[DataTypeT]`) will be sent.
            event_source: The primary `AsyncPoller` instance that sources
                incoming events (as `SerializableAnnotatedInstance[EventTypeT]`).
            min_send_frequency_seconds: Optional minimum time interval, in seconds,
                to be used for the per-caller `AsyncPoller` instances created
                by the internal `IdTracker`. This controls how frequently events
                are polled for each registered caller. If `None`, the default
                polling frequency of `AsyncPoller` is used.
        """
        super().__init__()
        self.__data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]] = (
            data_reader
        )
        self.__event_source: AsyncPoller[
            SerializableAnnotatedInstance[EventTypeT]
        ] = event_source

        # Define a properly typed factory for the IdTracker
        def _poller_factory() -> (
            AsyncPoller[SerializableAnnotatedInstance[EventTypeT]]
        ):
            return AsyncPoller(
                min_poll_frequency_seconds=min_send_frequency_seconds
            )

        self.__id_tracker = IdTracker[
            AsyncPoller[SerializableAnnotatedInstance[EventTypeT]]
        ](_poller_factory)

        # Start the background task for dispatching events from the main event_source
        # to individual caller-specific pollers managed by the IdTracker.
        run_on_event_loop(self.__dispatch_poller_data_loop)

    @property
    def _id_tracker(
        self,
    ) -> IdTracker[AsyncPoller[SerializableAnnotatedInstance[EventTypeT]]]:
        """Provides access to the internal `IdTracker` instance.

        The `IdTracker` manages mappings between `CallerIdentifier` objects,
        their network addresses, and their dedicated `AsyncPoller` instances for
        receiving events.
        """
        return self.__id_tracker

    @overload
    async def register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]: ...

    @overload
    async def register_caller(
        self, caller_id: CallerIdentifier, context: grpc.aio.ServicerContext
    ) -> Optional[
        EndpointDataProcessor[DataTypeT, EventTypeT]
    ]:  # Can return None if context parsing fails
        ...

    async def register_caller(
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[EndpointDataProcessor[DataTypeT, EventTypeT]]:
        """Registers a caller and returns an `EndpointDataProcessor` for it.

        This method handles different ways of identifying a caller:
        1. By explicit `endpoint` (e.g., IP address) and `port` number.
        2. By a `grpc.aio.ServicerContext` object, from which the endpoint and
           port are extracted.

        After determining the endpoint and port, it delegates to the abstract
        `_register_caller` method, which must be implemented by subclasses to
        perform specific registration logic (e.g., setting up time synchronization)
        and to return the appropriate `EndpointDataProcessor`.

        Args:
            caller_id: The `CallerIdentifier` for the caller to be registered.
            *args: Can be `(endpoint_str, port_int)` or `(grpc_context)`.
            **kwargs: Can be `endpoint="...", port=123` or `context=grpc_context`.

        Returns:
            An `EndpointDataProcessor` instance configured for the registered
            caller if successful. Returns `None` if registration fails (e.g.,
            if IP address cannot be extracted from the gRPC context).

        Raises:
            ValueError: If the provided arguments are inconsistent (e.g., mixing
                endpoint/port with context, or providing partial endpoint/port).
            TypeError: If `context` is provided but is not of the expected
                `grpc.aio.ServicerContext` type.
        """
        _endpoint: Optional[str] = None
        _port: Optional[int] = None
        _context: Optional[grpc.aio.ServicerContext] = None

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
            raise ValueError(
                "Unexpected positional args. Provide (endpoint, port) "
                f"or (context,). Got: {args}"
            )

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
            raise ValueError(
                f"Unexpected keyword arguments: {list(kwargs.keys())}"
            )

        if (_endpoint is None and _port is None) == (
            _context is None
        ):  # Both provided or neither provided
            raise ValueError(
                "Provide (endpoint and port) or context, but not both or neither."
            )
        if (_port is None) != (
            _endpoint is None
        ):  # One provided without the other
            raise ValueError(
                "If 'endpoint' is provided, 'port' must also be, and vice-versa."
            )

        actual_endpoint: str
        actual_port: int

        if _context is not None:
            if not isinstance(_context, grpc.aio.ServicerContext):
                raise TypeError(
                    "Expected context: grpc.aio.ServicerContext, "
                    f"got {type(_context).__name__}."
                )
            extracted_endpoint = get_client_ip(_context)
            extracted_port = get_client_port(_context)

            if extracted_endpoint is None:
                # Not necessarily an error, could be a non-IP context (e.g., UDS)
                # or context where peer is not available.
                return None
            if extracted_port is None:
                raise ValueError(
                    f"Could not get client port from context for endpoint: {extracted_endpoint}."
                )
            actual_endpoint = extracted_endpoint
            actual_port = extracted_port
        elif (
            _endpoint is not None and _port is not None
        ):  # This implies _context is None
            actual_endpoint = _endpoint
            actual_port = _port
        else:
            # This state should be unreachable due to prior validation.
            raise ValueError(
                "Internal error: Inconsistent endpoint/port/context state after validation."
            )

        return await self._register_caller(
            caller_id, actual_endpoint, actual_port
        )

    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> Optional[CallerIdentifier]:
        """Checks if a `CallerIdentifier` exists for the given network address.

        Args:
            endpoint: The IP address or hostname of the caller.
            port: The port number of the caller.

        Returns:
            The `CallerIdentifier` if a mapping exists for the given
            endpoint and port, otherwise `None`.
        """
        return self._try_get_caller_id(endpoint, port)

    async def _on_data_ready(self, data: AnnotatedInstance[DataTypeT]) -> None:
        """Callback invoked by `_DataProcessorImpl` when new data is processed.

        This method forwards the annotated data instance to the underlying
        `RemoteDataReader`.

        Args:
            data: The `AnnotatedInstance` containing the processed data and
                its associated metadata (caller ID, timestamp).

        Raises:
            ValueError: If the internal data reader (`self.__data_reader`) is `None`.
        """
        if self.__data_reader is None:  # Should not happen with proper init
            raise ValueError("Data reader instance is None.")
        # pylint: disable=protected-access # Calling protected method of a collaborator
        self.__data_reader._on_data_ready(data)

    @abstractmethod
    # pylint: disable=arguments-differ
    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]:
        """Performs subclass-specific logic for registering a new caller.

        This method is called by the public `register_caller` after arguments
        are validated and the caller\'s endpoint and port are determined.
        Implementations should handle tasks such as setting up time synchronization
        for the caller and then creating and returning an appropriate
        `EndpointDataProcessor` instance using `_create_data_processor`.

        Args:
            caller_id: The `CallerIdentifier` of the caller to register.
            endpoint: The resolved IP address or hostname of the caller.
            port: The resolved port number of the caller.

        Returns:
            An `EndpointDataProcessor` instance configured for the new caller.
        """

    @abstractmethod
    async def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Performs subclass-specific logic for unregistering a caller.

        This method is called when a caller associated with the given
        `CallerIdentifier` needs to be removed or cleaned up. Implementations
        should handle tasks such as stopping time synchronization, removing
        the caller from the `IdTracker`, and releasing any other associated
        resources.

        Args:
            caller_id: The `CallerIdentifier` of the caller to unregister.

        Returns:
            True if the caller was found and successfully unregistered,
            False otherwise (e.g., if the caller ID was not found).
        """

    async def __anext__(
        self,
    ) -> List[SerializableAnnotatedInstance[EventTypeT]]:
        """Retrieves the next batch of events from the main event source.

        This method makes `RuntimeDataHandlerBase` an asynchronous iterator,
        allowing consumers to iterate over events polled by `self.__event_source`.

        Returns:
            A list of `SerializableAnnotatedInstance[EventTypeT]` objects.

        Raises:
            StopAsyncIteration: When the event source is exhausted.
        """
        return await self.__event_source.__anext__()

    def __aiter__(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
        """Returns self as the asynchronous iterator for events.

        This allows `RuntimeDataHandlerBase` instances to be used directly in
        `async for` loops to consume events from the main event source.
        """
        return self

    def _create_data_processor(
        self, caller_id: CallerIdentifier, clock: SynchronizedClock
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT]:
        """Factory method to create a `_DataProcessorImpl` instance.

        This method retrieves the dedicated `AsyncPoller` for the given `caller_id`
        from the `IdTracker` and uses it to instantiate the
        `_DataProcessorImpl`.

        Args:
            caller_id: The `CallerIdentifier` for which to create the data processor.
            clock: The `SynchronizedClock` to be used by the data processor for
                timestamp desynchronization.

        Returns:
            A new `_DataProcessorImpl` instance configured for the specified
            caller and clock.

        Raises:
            KeyError: If the `caller_id` is not found in the `IdTracker` or if
                it does not have an associated data poller (which would indicate
                an internal state inconsistency).
        """
        # The IdTracker is configured to store AsyncPoller instances as TrackedDataT.
        # get() by ID returns (address, port, data_poller_instance)
        _address, _port, data_poller = self.__id_tracker.get(caller_id)
        if data_poller is None:
            # This should ideally not happen if add() correctly uses the data_factory
            raise ValueError(
                f"No data poller found in IdTracker for {caller_id}"
            )

        return RuntimeDataHandlerBase._DataProcessorImpl(
            self, caller_id, clock, data_poller
        )

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> Optional[CallerIdentifier]:
        """Tries to retrieve a `CallerIdentifier` for a given network address.

        Args:
            endpoint: The IP address or hostname of the caller.
            port: The port number of the caller.

        Returns:
            The `CallerIdentifier` if a mapping exists for the given endpoint
            and port, otherwise `None`.
        """
        # try_get(address, port) returns (CallerIdentifier, TrackedDataT) or None
        result_tuple = self.__id_tracker.try_get(endpoint, port)
        if result_tuple is None:
            return None
        # We only need the CallerIdentifier from the (CallerIdentifier, TrackedDataT) tuple
        return result_tuple[0]

    async def __dispatch_poller_data_loop(self) -> None:
        """Continuously polls the main event source and dispatches events.

        This background task runs in the event loop. It asynchronously iterates
        over events from `self.__event_source`. For each event, it looks up the
        `CallerIdentifier` in the `IdTracker` to find the corresponding
        per-caller `AsyncPoller`. If found, the event is put onto that
        dedicated poller.
        """
        try:
            async for events_batch in self.__event_source:
                for event_item in events_batch:
                    if event_item.caller_id is None:
                        # Dispatch to all known pollers if caller_id is None
                        all_pollers = self._id_tracker.get_all_tracked_data()
                        for poller in all_pollers:
                            if poller is not None:
                                poller.on_available(event_item)
                    else:
                        # try_get by caller_id returns (address, port, data_poller) or None
                        id_tracker_entry = self._id_tracker.try_get(
                            event_item.caller_id
                        )
                        if id_tracker_entry is None:
                            # Potentially log this? Caller might have deregistered.
                            continue

                        _address, _port, per_caller_poller = id_tracker_entry
                        if per_caller_poller is not None:
                            per_caller_poller.on_available(event_item)
                        # else: Potentially log if poller is None but entry existed?
        except Exception as e:
            # This generic catch might hide issues during testing if not re-raised.
            # However, ThreadWatcher is supposed to catch and report these.
            # For now, ensure it's visible during tests.
            print(
                f"CRITICAL ERROR in __dispatch_poller_data_loop: {type(e).__name__}: {e}"
            )
            import traceback

            traceback.print_exc()
            raise

    class _DataProcessorImpl(EndpointDataProcessor[DataTypeT, EventTypeT]):
        """Concrete `EndpointDataProcessor` for `RuntimeDataHandlerBase`.

        This inner class implements the `EndpointDataProcessor` interface. It
        handles data desynchronization using the provided `SynchronizedClock`,
        delegates caller deregistration to the parent `RuntimeDataHandlerBase`,
        processes data by wrapping it into an `AnnotatedInstance` and passing
        it to the parent\'s `_on_data_ready` method, and provides an async
        iterator for events from its dedicated per-caller `AsyncPoller`.
        """

        def __init__(
            self,
            data_handler: "RuntimeDataHandlerBase[DataTypeT, EventTypeT]",
            caller_id: CallerIdentifier,
            clock: SynchronizedClock,
            data_poller: AsyncPoller[
                SerializableAnnotatedInstance[EventTypeT]
            ],
        ):
            """Initializes the `_DataProcessorImpl`.

            Args:
                data_handler: The parent `RuntimeDataHandlerBase` instance to
                    which operations like data submission and deregistration
                    are delegated.
                caller_id: The `CallerIdentifier` this processor is for.
                clock: The `SynchronizedClock` used for desynchronizing
                    `ServerTimestamp` objects.
                data_poller: The dedicated `AsyncPoller` instance from which this
                    processor will source events for the specific caller.
            """
            super().__init__(caller_id)
            self.__data_handler: (
                "RuntimeDataHandlerBase[DataTypeT, EventTypeT]"
            ) = data_handler
            self.__clock: SynchronizedClock = clock
            self.__data_poller: AsyncPoller[
                SerializableAnnotatedInstance[EventTypeT]
            ] = data_poller

        async def desynchronize(
            self,
            timestamp: ServerTimestamp,
            context: Optional[grpc.aio.ServicerContext] = None,
        ) -> Optional[datetime]:
            """Desynchronizes a `ServerTimestamp` to a local `datetime` object.

            Uses the `SynchronizedClock` provided during initialization.

            Args:
                timestamp: The `ServerTimestamp` to desynchronize.
                context: Optional. The `grpc.aio.ServicerContext` for a gRPC call.
                    If provided and `timestamp` is invalid (cannot be parsed by
                    `SynchronizedTimestamp.try_parse`), the call will be aborted.

            Returns:
                The desynchronized `datetime` object in UTC. Returns `None` if
                `timestamp` is invalid. If `context` was provided and the
                timestamp was invalid, the gRPC call would have been aborted
                before returning `None`.
            """
            synchronized_ts_obj = SynchronizedTimestamp.try_parse(timestamp)
            if synchronized_ts_obj is None:
                if context is not None:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Invalid timestamp provided",
                    )
                return None

            return self.__clock.desync(synchronized_ts_obj)

        async def deregister_caller(self) -> None:
            """Deregisters the caller by delegating to the parent data handler.

            This calls the `_unregister_caller` method of the parent
            `RuntimeDataHandlerBase`.
            """
            # pylint: disable=protected-access # Calling protected method of parent/owner
            await self.__data_handler._unregister_caller(self.caller_id)
            # The return value (bool) of _unregister_caller is ignored here,
            # as EndpointDataProcessor.deregister_caller returns None.

        async def _process_data(
            self, data: DataTypeT, timestamp: datetime
        ) -> None:
            """Processes data by creating an `AnnotatedInstance` and passing it to the parent.

            The data is wrapped with its `CallerIdentifier` and the provided
            `datetime` timestamp, then submitted via the parent data handler\'s
            `_on_data_ready` method.

            Args:
                data: The data item to process.
                timestamp: The synchronized `datetime` (UTC) of the data.
            """
            wrapped_data = AnnotatedInstance(
                caller_id=self.caller_id, timestamp=timestamp, data=data
            )
            # pylint: disable=protected-access # Calling protected method of parent/owner
            await self.__data_handler._on_data_ready(wrapped_data)

        def __aiter__(
            self,
        ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
            """Returns an asynchronous iterator for events from the dedicated per-caller poller."""
            return self.__data_poller.__aiter__()
