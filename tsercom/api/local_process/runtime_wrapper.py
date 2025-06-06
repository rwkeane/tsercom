"""Wraps a local runtime, providing a handle for interaction and data flow."""

from datetime import datetime
from typing import Generic, Optional, TypeVar

from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.aio.async_poller import AsyncPoller

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class RuntimeWrapper(
    Generic[DataTypeT, EventTypeT],
    RuntimeHandle[DataTypeT, EventTypeT],
    RemoteDataReader[AnnotatedInstance[DataTypeT]],
):
    """A wrapper that acts as a RuntimeHandle for local process runtimes.

    It bridges commands (start/stop) to the runtime and manages the flow of
    events and data to and from the runtime. It uses an `AsyncPoller` for
    events and a `RemoteDataAggregator` for data.
    """

    def __init__(
        self,
        event_poller: AsyncPoller[EventInstance[EventTypeT]],
        data_aggregator: RemoteDataAggregatorImpl[
            AnnotatedInstance[DataTypeT]
        ],  # Changed DataTypeT
        bridge: RuntimeCommandBridge,
    ) -> None:
        """Initializes the RuntimeWrapper.

        Args:
            event_poller: An AsyncPoller to handle incoming events.
            data_aggregator: A RemoteDataAggregatorImpl to manage data.
            bridge: A RuntimeCommandBridge to send commands to the runtime.
        """
        self.__event_poller: AsyncPoller[EventInstance[EventTypeT]] = (
            event_poller
        )
        self.__aggregator: RemoteDataAggregatorImpl[
            AnnotatedInstance[DataTypeT]
        ] = data_aggregator
        self.__bridge: RuntimeCommandBridge = bridge

    def start(self) -> None:
        """Starts the underlying runtime via the command bridge."""
        self.__bridge.start()

    def stop(self) -> None:
        """Stops the underlying runtime via the command bridge."""
        self.__bridge.stop()

    def on_event(
        self,
        event: EventTypeT,
        caller_id: Optional[CallerIdentifier] = None,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Wraps an incoming event and passes it to the event poller.

        Args:
            event: The event data to process.
            caller_id: Optional ID of the caller generating the event.
            timestamp: Optional event timestamp. Defaults to `datetime.now()`.
        """
        if timestamp is None:
            timestamp = datetime.now()

        wrapped_event = EventInstance(event, caller_id, timestamp)
        self.__event_poller.on_available(wrapped_event)

    def _on_data_ready(self, new_data: AnnotatedInstance[DataTypeT]) -> None:
        """Callback method invoked when new data is ready from the runtime.

        This method is part of the `RemoteDataReader` interface.

        Args:
            new_data: The new data instance that has become available.
        """
        # pylint: disable=W0212 # Internal callback for client data readiness
        self.__aggregator._on_data_ready(new_data)

    def _get_remote_data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[DataTypeT]]:
        """Provides access to the remote data aggregator.

        This method is part of the `RemoteDataReader` interface.

        Returns:
            The `RemoteDataAggregator` instance used by this wrapper.
        """
        return self.__aggregator

    @property
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[DataTypeT]]:
        """Provides the remote data aggregator."""
        return self._get_remote_data_aggregator()
