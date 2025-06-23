"""Defines the abstract base class for a runtime handle."""

import datetime
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

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
        """Start the associated runtime or service."""
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> None:
        """Stop the associated runtime or service."""
        raise NotImplementedError()

    # Overloads for the on_event method, defining different ways to call it.
    # These guide type checkers and IDEs for better developer experience.
    @overload
    def on_event(self, event: EventTypeT) -> None:
        """Send an event to the associated runtime.

        This overload is used when only the event data is provided. The system
        will typically assign a default caller ID and use the current time as
        the timestamp.

        Args:
            event: The event data to send. (Type: EventTypeT)

        """
        ...

    @overload
    def on_event(self, event: EventTypeT, caller_id: CallerIdentifier) -> None:
        """Send an event to the associated runtime with a specific caller ID.

        This overload is used when the event data and a specific caller ID are
        provided. The system will typically use the current time as the timestamp.

        Args:
            event: The event data to send. (Type: EventTypeT)
            caller_id: The identifier of the entity originating this event.
                (Type: CallerIdentifier)

        """
        ...

    @overload
    def on_event(self, event: EventTypeT, *, timestamp: datetime.datetime) -> None:
        """Send an event to the associated runtime with a specific timestamp.

        This overload is used when the event data and a specific timestamp are
        provided. The system will typically assign a default caller ID.
        The timestamp must be provided as a keyword argument.

        Args:
            event: The event data to send. (Type: EventTypeT)
            timestamp: The explicit timestamp for this event.
                (Type: datetime.datetime)

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
        """Send an event with event data, caller ID, and a specific timestamp.

        This overload is used when event data, a specific caller ID, and an
        explicit timestamp are all provided. The timestamp must be provided as
        a keyword argument.

        Args:
            event: The event data to send. (Type: EventTypeT)
            caller_id: The identifier of the entity originating this event.
                (Type: CallerIdentifier)
            timestamp: The explicit timestamp for this event.
                (Type: datetime.datetime)

        """
        ...

    @abstractmethod
    def on_event(
        self,
        event: EventTypeT,
        caller_id: CallerIdentifier | None = None,
        *,
        timestamp: datetime.datetime | None = None,
    ) -> None:
        """Send an event to the runtime.

        This is the main implementation for sending an event. It can be called
        with various combinations of event data, caller ID, and timestamp.
        Refer to the specific @overload signatures for detailed argument
        combinations and their descriptions. If caller_id or timestamp are
        not provided, system defaults may be used.
        """
        # Main impl docstring is minimal per prompt (overloads documented).
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
