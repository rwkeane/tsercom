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

import asyncio
import logging  # Added import
from abc import abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)

import grpc

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.rpc.grpc_util.addressing import get_client_ip, get_client_port
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.common.proto import ServerTimestamp
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.util.is_running_tracker import IsRunningTracker  # Added import

EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT")

_logger = logging.getLogger(__name__)  # Added logger


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
        event_source: AsyncPoller[EventInstance[EventTypeT]],
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
        self.__data_reader: RemoteDataReader[AnnotatedInstance[DataTypeT]] = data_reader
        self.__event_source: AsyncPoller[EventInstance[EventTypeT]] = event_source

        def _poller_factory() -> AsyncPoller[EventInstance[EventTypeT]]:
            # Each poller created by the factory needs its own IsRunningTracker.
            # This tracker should be started as these pollers are expected to be active.
            tracker = IsRunningTracker()
            tracker.start()
            return AsyncPoller(
                is_running_tracker=tracker,
                min_poll_frequency_seconds=min_send_frequency_seconds,
            )

        self.__id_tracker = IdTracker[AsyncPoller[EventInstance[EventTypeT]]](
            _poller_factory
        )

        self.__dispatch_task: asyncio.Task[None] | None = None
        # Import get_global_event_loop and is_global_event_loop_set locally to avoid circular dependency issues at module level
        from tsercom.threading.aio.global_event_loop import (
            get_global_event_loop,
            is_global_event_loop_set,
        )

        self._loop_on_init: asyncio.AbstractEventLoop | None = None  # Added type hint
        if is_global_event_loop_set():
            self._loop_on_init = get_global_event_loop()  # Store loop used at init
            self.__dispatch_task = self._loop_on_init.create_task(
                self.__dispatch_poller_data_loop()
            )
            _logger.debug(
                "__dispatch_task %s created on loop %s.",
                self.__dispatch_task,
                id(self._loop_on_init),
            )
        else:
            # self._loop_on_init is already None due to the type hint and default initialization
            _logger.warning(
                "No global event loop set during RuntimeDataHandlerBase init. "
                "__dispatch_poller_data_loop will not start."
            )

    async def async_close(self) -> None:
        """Cancels and awaits the background dispatch task."""
        current_loop = asyncio.get_running_loop()
        _logger.info(
            "RuntimeDataHandlerBase async_close called. Current loop: %s",
            id(current_loop),
        )

        if (
            hasattr(self, "_RuntimeDataHandlerBase__dispatch_task")
            and self.__dispatch_task
        ):
            task = self.__dispatch_task
            # Ensure task loop retrieval is safe
            task_loop = None
            try:
                task_loop = task.get_loop()
            except RuntimeError:  # Can happen if task is done and loop is closed
                _logger.warning(
                    "Could not get loop for task %s during async_close, it might be done and its loop closed.",
                    task,
                )

            task_loop_id = id(task_loop) if task_loop else "N/A"
            _logger.debug(
                "Attempting to close __dispatch_task: %s (created on loop: %s, current task loop: %s)",
                task,
                (id(self._loop_on_init) if self._loop_on_init else "N/A"),
                task_loop_id,
            )

            if not task.done():
                # It's crucial that task.cancel() and await task happen on the loop the task is running on.
                # If self._loop_on_init is different from current_loop, this might be an issue.
                # However, pytest-asyncio and conftest should ensure fixture teardown runs on the same loop as test.
                if self._loop_on_init and self._loop_on_init is not current_loop:
                    _logger.warning(
                        "Potential loop mismatch in async_close: task loop %s (init_loop %s) vs current_loop %s. This might cause issues.",
                        task_loop_id,
                        id(self._loop_on_init),
                        id(current_loop),
                    )

                _logger.debug("Cancelling dispatch_task: %s", task)
                task.cancel()
                try:
                    await task
                    _logger.debug(
                        "Dispatch_task %s awaited after cancellation (processed CancelledError).",
                        task,
                    )
                except asyncio.CancelledError:
                    _logger.info(
                        "Dispatch_task %s successfully cancelled and handled CancelledError.",
                        task,
                    )
                except Exception as e:
                    _logger.error(
                        "Exception while awaiting cancelled dispatch_task %s: %s",
                        task,
                        e,
                        exc_info=True,
                    )
            else:
                _logger.debug("Dispatch_task %s was already done.", task)
            self.__dispatch_task = None
        else:
            _logger.debug("No __dispatch_task found or already cleared in async_close.")

    @property
    def _id_tracker(
        self,
    ) -> IdTracker[AsyncPoller[EventInstance[EventTypeT]]]:
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
    ) -> (
        EndpointDataProcessor[DataTypeT, EventTypeT] | None
    ):  # Can return None if context parsing fails
        ...

    async def register_caller(
        self,
        caller_id: CallerIdentifier,
        *args: Any,
        **kwargs: Any,
    ) -> EndpointDataProcessor[DataTypeT, EventTypeT] | None:
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
        _endpoint: str | None = None
        _port: int | None = None
        _context: grpc.aio.ServicerContext | None = None

        if len(args) == 1 and isinstance(args[0], grpc.aio.ServicerContext):
            _context = args[0]
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
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
            if _context is not None or _endpoint is not None or _port is not None:
                raise ValueError(
                    "Cannot use context via args & kwargs, or with endpoint/port."
                )
            _context = kwargs.pop("context")

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        if (_endpoint is None and _port is None) == (
            _context is None
        ):  # Both provided or neither provided
            raise ValueError(
                "Provide (endpoint and port) or context, but not both or neither."
            )
        if (_port is None) != (_endpoint is None):  # One provided without the other
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

        return await self._register_caller(caller_id, actual_endpoint, actual_port)

    def check_for_caller_id(self, endpoint: str, port: int) -> CallerIdentifier | None:
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

        self.__data_reader._on_data_ready(data)

    @abstractmethod
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
    ) -> list[EventInstance[EventTypeT]]:
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
    ) -> AsyncIterator[list[EventInstance[EventTypeT]]]:
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
        _address, _port, data_poller = self.__id_tracker.get(caller_id)
        if data_poller is None:
            raise ValueError(f"No data poller found in IdTracker for {caller_id}")

        return RuntimeDataHandlerBase._DataProcessorImpl(
            self, caller_id, clock, data_poller
        )

    def _try_get_caller_id(self, endpoint: str, port: int) -> CallerIdentifier | None:
        """Tries to retrieve a `CallerIdentifier` for a given network address.

        Args:
            endpoint: The IP address or hostname of the caller.
            port: The port number of the caller.

        Returns:
            The `CallerIdentifier` if a mapping exists for the given endpoint
            and port, otherwise `None`.
        """
        result_tuple = self.__id_tracker.try_get(endpoint, port)
        if result_tuple is None:
            return None
        return result_tuple[0]

    async def __dispatch_poller_data_loop(self) -> None:
        """Continuously polls the main event source and dispatches events.

        This background task runs in the event loop. It asynchronously iterates
        over events from `self.__event_source`. For each event, it looks up the
        `CallerIdentifier` in the `IdTracker` to find the corresponding
        per-caller `AsyncPoller`. If found, the event is put onto that
        dedicated poller.
        """
        _logger.info("Starting __dispatch_poller_data_loop.")
        try:
            async for events_batch in self.__event_source:
                for event_item in events_batch:
                    if event_item.caller_id is None:
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
                await asyncio.sleep(0)  # Yield control occasionally
        except asyncio.CancelledError:
            _logger.info("__dispatch_poller_data_loop received CancelledError.")
            raise  # Important to propagate for the awaiter
        except Exception as e:
            # This logging was originally just print(), changed to _logger.critical
            _logger.critical(
                "CRITICAL ERROR in __dispatch_poller_data_loop: %s: %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            # Consider how to report this to ThreadWatcher if applicable
            raise
        finally:
            _logger.info("__dispatch_poller_data_loop finished.")

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
            data_poller: AsyncPoller[EventInstance[EventTypeT]],
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
            self.__data_handler: RuntimeDataHandlerBase[DataTypeT, EventTypeT] = (
                data_handler
            )
            self.__clock: SynchronizedClock = clock
            self.__data_poller: AsyncPoller[EventInstance[EventTypeT]] = data_poller

        async def desynchronize(
            self,
            timestamp: ServerTimestamp,
            context: grpc.aio.ServicerContext | None = None,
        ) -> datetime | None:
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

            await self.__data_handler._unregister_caller(self.caller_id)
            # The return value (bool) of _unregister_caller is ignored here,
            # as EndpointDataProcessor.deregister_caller returns None.

        async def _process_data(self, data: DataTypeT, timestamp: datetime) -> None:
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

            await self.__data_handler._on_data_ready(wrapped_data)

        async def __aiter__(
            self,
        ) -> AsyncIterator[list[SerializableAnnotatedInstance[EventTypeT]]]:
            """Returns an asynchronous iterator for events from the dedicated per-caller poller."""
            async for event_instance_batch in self.__data_poller:
                processed_batch: list[SerializableAnnotatedInstance[EventTypeT]] = []
                for event_instance in event_instance_batch:
                    synchronized_ts = self.__clock.sync(event_instance.timestamp)
                    serializable_event = SerializableAnnotatedInstance(
                        data=event_instance.data,
                        caller_id=event_instance.caller_id,
                        timestamp=synchronized_ts,
                    )
                    processed_batch.append(serializable_event)
                yield processed_batch
