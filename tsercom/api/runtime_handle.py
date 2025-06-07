"""Defines the abstract base class for a runtime handle."""

import datetime  # Keep 'import datetime' as 'datetime.datetime' is used
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator

# Type variable for data, bound by ExposedData.
DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
# Type variable for events.
EventTypeT = TypeVar("EventTypeT")


class RuntimeHandle(ABC, Generic[DataTypeT, EventTypeT]):
    """Abstract handle for a running Runtime instance.

    This class provides an interface to control (start/stop) and send events
    to a Runtime, as well as a simplified way to receive data from it
    via a data aggregator.
    """

    @property
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[DataTypeT]]:
        """RemoteDataAggregator for retrieving data from this runtime.

        Returns:
            A RemoteDataAggregator for this handle.
        """
        # This property relies on _get_remote_data_aggregator being implemented.
        # For an ABC, it's better to make the property itself abstract
        # or implement it using an abstract method if there's common logic.
        aggregator = self._get_remote_data_aggregator()
        return aggregator

    @abstractmethod
    def start(self) -> None:
        """Starts the associated runtime or service."""
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> None:
        """Stops the associated runtime or service."""
        raise NotImplementedError()

    # Overloads for the on_event method, defining different ways to call it.
    # These guide type checkers and IDEs for better developer experience.
    @overload
    def on_event(self, event: EventTypeT) -> None:
        """Sends an event to the runtime with only the event data.

        Args:
            event: The event data to send.
        """
        ...

    @overload
    def on_event(self, event: EventTypeT, caller_id: CallerIdentifier) -> None:
        """Sends an event to the runtime with event data and a caller ID.

        Args:
            event: The event data to send.
            caller_id: ID of the caller originating the event.
        """
        ...

    @overload
    def on_event(
        self, event: EventTypeT, *, timestamp: datetime.datetime
    ) -> None:
        """Sends an event to the runtime with event data and a specific timestamp.

        Args:
            event: The event data to send.
            timestamp: Timestamp for the event.
        """
        ...

    @overload
    def on_event(
        self,
        event: EventTypeT,
        caller_id: CallerIdentifier,
        *,
        timestamp: datetime.datetime,
    ) -> None:
        """Sends an event with event data, caller ID, and a specific timestamp.

        Args:
            event: The event data to send.
            caller_id: ID of the caller originating the event.
            timestamp: Timestamp for the event.
        """
        ...

    @abstractmethod
    def on_event(
        self,
        event: EventTypeT,
        caller_id: Optional[CallerIdentifier] = None,
        *,
        timestamp: Optional[datetime.datetime] = None,
    ) -> None:
        """Sends an event to the runtime.

        The event can be sent with or without a specific caller ID and
        with an optional custom timestamp. If no caller ID is provided,
        the event may be broadcast or handled by a default mechanism.
        If no timestamp is provided, the current time is typically used
        by the implementation.

        Args:
            event: The event data to send.
            caller_id: Optional. ID of the caller originating the event.
                       If specified, event might be targeted or filtered.
            timestamp: Optional. Timestamp for the event.
                       If None, implementations default to `datetime.now()`.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_remote_data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[DataTypeT]]:
        """Abstract method for subclasses to provide their RemoteDataAggregator.

        Returns:
            The RemoteDataAggregator instance for this runtime.
        """
        raise NotImplementedError()
