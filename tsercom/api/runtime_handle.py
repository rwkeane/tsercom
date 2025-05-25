"""Defines the abstract base class for a runtime handle."""

from abc import ABC, abstractmethod
import datetime # Keep 'import datetime' as 'datetime.datetime' is used
from typing import Generic, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator

# Type variable for data, bound by ExposedData.
TDataType = TypeVar("TDataType", bound=ExposedData)
# Type variable for events.
TEventType = TypeVar("TEventType")


class RuntimeHandle(ABC, Generic[TDataType, TEventType]):
    """Abstract handle for a running Runtime instance.

    This class provides an interface to control (start/stop) and send events
    to a Runtime, as well as a simplified way to receive data from it
    via a data aggregator.
    """

    @property
    @abstractmethod
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[TDataType]]:
        """The RemoteDataAggregator for retrieving data from this runtime.

        Returns:
            A RemoteDataAggregator instance associated with this runtime handle.
        """
        # This property relies on _get_remote_data_aggregator being implemented.
        # For an ABC, it's better to make the property itself abstract
        # or implement it using an abstract method if there's common logic.
        raise NotImplementedError()


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
    def on_event(self, event: TEventType) -> None: ...

    @overload
    def on_event(
        self, event: TEventType, caller_id: CallerIdentifier
    ) -> None: ...

    @overload
    def on_event(
        self, event: TEventType, *, timestamp: datetime.datetime
    ) -> None: ...

    @overload
    def on_event(
        self,
        event: TEventType,
        caller_id: CallerIdentifier,
        *,
        timestamp: datetime.datetime,
    ) -> None: ...

    @abstractmethod
    def on_event(
        self,
        event: TEventType,
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
            caller_id: Optional. The identifier of the caller originating the event.
                       If specified, the event might be targeted or filtered based
                       on this ID.
            timestamp: Optional. The timestamp to associate with the event.
                       If None, implementations usually default to `datetime.now()`.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_remote_data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[TDataType]]:
        """Abstract method for subclasses to provide their RemoteDataAggregator.

        Returns:
            The RemoteDataAggregator instance associated with this runtime.
        """
        raise NotImplementedError()
