"""Wraps a local runtime, providing a handle for interaction and data flow."""

from datetime import datetime
from typing import Generic, Optional, TypeVar

from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.threading.async_poller import AsyncPoller

# Type variable for data, bound by ExposedData.
TDataType = TypeVar("TDataType", bound=ExposedData)
# Type variable for events.
TEventType = TypeVar("TEventType")


class RuntimeWrapper(
    Generic[TDataType, TEventType],
    RuntimeHandle[TDataType, TEventType],
    RemoteDataReader[AnnotatedInstance[TDataType]],  # Changed TDataType
):
    """A wrapper that acts as a RuntimeHandle for local process runtimes.

    It bridges commands (start/stop) to the runtime and manages the flow of
    events and data to and from the runtime. It uses an `AsyncPoller` for
    events and a `RemoteDataAggregator` for data.
    """

    def __init__(
        self,
        event_poller: AsyncPoller[EventInstance[TEventType]],
        data_aggregator: RemoteDataAggregatorImpl[
            AnnotatedInstance[TDataType]
        ],  # Changed TDataType
        bridge: RuntimeCommandBridge,
    ) -> None:
        """Initializes the RuntimeWrapper.

        Args:
            event_poller: An AsyncPoller to handle incoming events.
            data_aggregator: A RemoteDataAggregatorImpl to manage data.
            bridge: A RuntimeCommandBridge to send commands to the runtime.
        """
        self.__event_poller: AsyncPoller[EventInstance[TEventType]] = (
            event_poller
        )
        self.__aggregator: RemoteDataAggregatorImpl[
            AnnotatedInstance[TDataType]
        ] = data_aggregator  # Changed TDataType
        self.__bridge: RuntimeCommandBridge = bridge

    def start(self) -> None:
        """Starts the underlying runtime via the command bridge."""
        self.__bridge.start()

    def stop(self) -> None:
        """Stops the underlying runtime via the command bridge."""
        self.__bridge.stop()

    def on_event(
        self,
        event: TEventType,
        caller_id: Optional[CallerIdentifier] = None,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Handles an incoming event by wrapping it and passing it to the poller.

        Args:
            event: The event data to process.
            caller_id: Optional identifier of the caller that generated the event.
            timestamp: Optional timestamp for the event. If None, defaults to now.
        """
        # Ensure a timestamp for the event.
        if timestamp is None:
            timestamp = datetime.now()

        wrapped_event = EventInstance(event, caller_id, timestamp)
        self.__event_poller.on_available(wrapped_event)

    def _on_data_ready(
        self, new_data: AnnotatedInstance[TDataType]
    ) -> None:  # Changed TDataType
        """Callback method invoked when new data is ready from the runtime.

        This method is part of the `RemoteDataReader` interface.

        Args:
            new_data: The new data instance that has become available.
        """
        self.__aggregator._on_data_ready(
            new_data
        )  # Aggregator now expects AnnotatedInstance

    def _get_remote_data_aggregator(
        self,
    ) -> RemoteDataAggregator[
        AnnotatedInstance[TDataType]
    ]:  # Changed TDataType
        """Provides access to the remote data aggregator.

        This method is part of the `RemoteDataReader` interface.

        Returns:
            The `RemoteDataAggregator` instance used by this wrapper.
        """
        return self.__aggregator

    @property
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[TDataType]]:
        # TODO: Address potential type mismatch if _get_remote_data_aggregator
        # returns RemoteDataAggregator[TDataType] instead of AnnotatedInstance[TDataType].
        # This should now be fixed.
        return self._get_remote_data_aggregator()
