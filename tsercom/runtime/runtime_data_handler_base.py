"""Base implementation for `RuntimeDataHandler`."""

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
from tsercom.rpc.grpc_util.addressing import get_client_ip, get_client_port
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.proto import ServerTimestamp
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class RuntimeDataHandlerBase(
    Generic[TDataType, TEventType], RuntimeDataHandler[TDataType, TEventType]
):
    """Provides common functionality for runtime data handlers.

    Manages data reading via `RemoteDataReader` and event polling via `AsyncPoller`.
    Handles caller registration by parsing context or direct endpoint info.
    Implements the async iterator protocol for event data.
    """

    def __init__(
        self,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
    ):
        """Initializes RuntimeDataHandlerBase.

        Args:
            data_reader: The `RemoteDataReader` to sink data into.
            event_source: The `AsyncPoller` to source event data from.
        """
        super().__init__()
        self.__data_reader = data_reader
        self.__event_source = event_source

    def register_caller(
        self,
        caller_id: CallerIdentifier,
        endpoint: Optional[str] = None,
        port: Optional[int] = None,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> EndpointDataProcessor[TDataType] | None:
        """Registers a caller using either endpoint/port or gRPC context.

        This concrete implementation of `RuntimeDataHandler.register_caller`
        validates inputs and extracts endpoint/port from context if provided,
        then delegates to the abstract `_register_caller` method.

        Args:
            caller_id: The `CallerIdentifier` of the caller.
            endpoint: The IP address or hostname.
            port: The port number.
            context: The gRPC servicer context.

        Returns:
            An `EndpointDataProcessor` for the caller, or `None` if registration fails.

        Raises:
            ValueError: If arguments are inconsistent (e.g., endpoint without port,
                        or both/neither endpoint and context are provided).
            TypeError: If context is not of the expected type.
        """
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

        actual_endpoint: str
        actual_port: int

        if context is not None:
            if not isinstance(context, grpc.aio.ServicerContext):
                raise TypeError(
                    f"Expected context to be an instance of grpc.aio.ServicerContext, but got {type(context).__name__}."
                )
            extracted_endpoint = get_client_ip(context)
            extracted_port = get_client_port(context)

            if extracted_endpoint is None:
                return None  # Cannot register if IP cannot be determined from context
            if extracted_port is None:
                raise ValueError(
                    f"Could not determine client port from context for endpoint {extracted_endpoint}."
                )
            actual_endpoint = extracted_endpoint
            actual_port = extracted_port
        elif (
            endpoint is not None and port is not None
        ):  # Explicitly check for type safety
            actual_endpoint = endpoint
            actual_port = port
        else:
            # This case should be theoretically unreachable due to the initial XOR check on endpoint and context,
            # and the subsequent check on endpoint/port pairing.
            # However, as a safeguard:
            raise ValueError(
                "Internal error: Inconsistent endpoint/port/context state."
            )

        return self._register_caller(caller_id, actual_endpoint, actual_port)

    def get_data_iterator(
        self,
    ) -> AsyncIterator[
        Dict[CallerIdentifier, List[SerializableAnnotatedInstance[TEventType]]]
    ]:
        """Returns an async iterator for event data, grouped by CallerIdentifier.

        This implementation returns `self` as it implements `__aiter__` and `__anext__`.
        """
        return self

    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Checks for an existing caller ID using the `_try_get_caller_id` template method.

        Args:
            endpoint: The IP address or hostname.
            port: The port number.

        Returns:
            The `CallerIdentifier` if found, `None` otherwise.
        """
        return self._try_get_caller_id(endpoint, port)

    async def _on_data_ready(self, data: AnnotatedInstance[TDataType]) -> None:
        """Callback invoked when new data is ready to be processed by the data reader.

        Args:
            data: The `AnnotatedInstance` containing the data and metadata.
        """
        # Define data_val_for_log and caller_id_for_log at the top
        data_val_for_log = ""
        if hasattr(data, 'data') and hasattr(data.data, 'value'):
            data_val_for_log = data.data.value
        elif hasattr(data, 'data'):
            data_val_for_log = str(data.data)
        else:
            data_val_for_log = str(data)
        
        caller_id_for_log = data.caller_id if hasattr(data, 'caller_id') else 'N/A'

        print(f"RuntimeDataHandlerBase._on_data_ready (id={id(self)}): ENTERED. Data val='{data_val_for_log}', caller_id='{caller_id_for_log}', data_id={id(data)}.", flush=True)
        
        data_reader_instance = self.__data_reader 

        if data_reader_instance is None:
            print(f"RuntimeDataHandlerBase._on_data_ready (id={id(self)}): self.__data_reader is None. Cannot process data '{data_val}'.", flush=True)
            return

        # --- New Detailed Method Inspection ---
        print(f"RuntimeDataHandlerBase._on_data_ready (id={id(self)}): Inspecting method before call for data_val='{data_val_for_log}', data_id={id(data)}:", flush=True)
        print(f"  - data_reader_instance is: (id={id(data_reader_instance)}, type={type(data_reader_instance).__name__})", flush=True)
        
        actual_method_obj = None
        is_callable_direct = False
        method_id_direct = "N/A"
        try:
            actual_method_obj = data_reader_instance._on_data_ready # Access the method
            method_id_direct = id(actual_method_obj)
            is_callable_direct = callable(actual_method_obj)
            print(f"  - Direct access data_reader_instance._on_data_ready is: {actual_method_obj} (id={method_id_direct})", flush=True)
            print(f"  - Is callable (direct access)?: {is_callable_direct}", flush=True)
        except Exception as e_direct_access:
            print(f"  - EXCEPTION during direct access to _on_data_ready: {type(e_direct_access).__name__} - {e_direct_access}", flush=True)

        getattr_method_obj = "NOT_FOUND"
        is_callable_getattr = False
        method_id_getattr = "N/A"
        try:
            getattr_method_obj = getattr(data_reader_instance, '_on_data_ready', "NOT_FOUND")
            if getattr_method_obj != "NOT_FOUND":
                method_id_getattr = id(getattr_method_obj)
                is_callable_getattr = callable(getattr_method_obj)
            print(f"  - getattr(data_reader_instance, '_on_data_ready') is: {getattr_method_obj} (id={method_id_getattr})", flush=True)
            print(f"  - Is callable (getattr)?: {is_callable_getattr}", flush=True)
        except Exception as e_getattr_access:
            print(f"  - EXCEPTION during getattr access or callable check: {type(e_getattr_access).__name__} - {e_getattr_access}", flush=True)
        # --- End New Detailed Method Inspection ---

        print(f"RuntimeDataHandlerBase._on_data_ready (id={id(self)}): About to call data_reader_instance._on_data_ready for data_val='{data_val_for_log}', data_id={id(data)}.", flush=True)
        try:
            data_reader_instance._on_data_ready(data)
            print(f"RuntimeDataHandlerBase._on_data_ready (id={id(self)}): SUCCESSFULLY RETURNED from data_reader_instance._on_data_ready for data_val='{data_val_for_log}', data_id={id(data)}.", flush=True)
        except Exception as e:
            print(f"RuntimeDataHandlerBase._on_data_ready (id={id(self)}): EXCEPTION during data_reader_instance._on_data_ready for data_val='{data_val_for_log}', data_id={id(data)}: {type(e).__name__} - {e}", flush=True)
            # To understand if this exception is then handled or re-raised by caller,
            # we might need to re-raise, but for now, just log.
            # raise 
        # --- End New Detailed Logging ---

    @abstractmethod
    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[TDataType]:
        """Template method for subclass-specific caller registration logic.

        Args:
            caller_id: The `CallerIdentifier` of the caller.
            endpoint: The IP address or hostname of the caller.
            port: The port number of the caller.

        Returns:
            An `EndpointDataProcessor` for the caller.
        """
        pass

    @abstractmethod
    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        """Template method for subclass-specific caller deregistration logic.

        Args:
            caller_id: The `CallerIdentifier` of the caller to unregister.

        Returns:
            True if the caller was found and unregistered, False otherwise.
        """
        pass

    @abstractmethod
    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        """Template method for subclass-specific logic to find a caller ID by endpoint.

        Args:
            endpoint: The IP address or hostname.
            port: The port number.

        Returns:
            The `CallerIdentifier` if found, `None` otherwise.
        """
        pass

    async def __anext__(
        self,
    ) -> Dict[
        CallerIdentifier, List[SerializableAnnotatedInstance[TEventType]]
    ]:
        """Retrieves the next batch of events from the event source.

        Implements the async iterator protocol.
        """
        return await self.__event_source.__anext__()

    def __aiter__(  # Changed to a regular method
        self,
    ) -> AsyncIterator[
        Dict[CallerIdentifier, List[SerializableAnnotatedInstance[TEventType]]]
    ]:
        """Returns self as the async iterator for events.

        Implements the async iterator protocol.
        """
        return self

    def _create_data_processor(
        self, caller_id: CallerIdentifier, clock: SynchronizedClock
    ) -> EndpointDataProcessor[TDataType]:  # Corrected return type
        """Factory method to create a concrete `EndpointDataProcessor`.

        Args:
            caller_id: The `CallerIdentifier` for the processor.
            clock: The `SynchronizedClock` to be used by the processor.

        Returns:
            An instance of `__DataProcessorImpl`.
        """
        return RuntimeDataHandlerBase.__DataProcessorImpl(
            self, caller_id, clock
        )

    class __DataProcessorImpl(EndpointDataProcessor[TDataType]):
        """Concrete implementation of `EndpointDataProcessor` for `RuntimeDataHandlerBase`.

        This nested class handles data desynchronization, caller deregistration,
        and data processing by delegating to the outer `RuntimeDataHandlerBase` instance.
        """

        def __init__(
            self,
            data_handler: "RuntimeDataHandlerBase[TDataType, TEventType]",
            caller_id: CallerIdentifier,
            clock: SynchronizedClock,
        ):
            """Initializes the __DataProcessorImpl.

            Args:
                data_handler: The parent `RuntimeDataHandlerBase` instance.
                caller_id: The `CallerIdentifier` for this processor.
                clock: The `SynchronizedClock` for timestamp desynchronization.
            """
            super().__init__(caller_id)
            self.__data_handler = data_handler
            self.__clock = clock

        async def desynchronize(self, timestamp: ServerTimestamp) -> datetime:
            """Desynchronizes a server timestamp using the provided clock."""
            return await self.__clock.desync(timestamp)  # Added await

        async def deregister_caller(self) -> None:
            """Deregisters the caller via the parent data handler."""
            await self.__data_handler._unregister_caller(
                self.caller_id
            )  # Added await

        async def _process_data(
            self, data: TDataType, timestamp: datetime
        ) -> None:
            """Processes data by wrapping it in an `AnnotatedInstance` and passing to the parent."""
            print(f"DataProcessorImpl._process_data: Received data='{data.value if hasattr(data, 'value') else data}', timestamp='{timestamp}', caller_id='{self.caller_id}'. Self id={id(self)}", flush=True)
            wrapped_data = AnnotatedInstance(data, self.caller_id, timestamp)
            print(f"DataProcessorImpl._process_data: Wrapped data='{wrapped_data.data.value if hasattr(wrapped_data.data, 'value') else wrapped_data.data}', caller_id='{wrapped_data.caller_id}'. Self id={id(self)}", flush=True)
            await self.__data_handler._on_data_ready(wrapped_data)
