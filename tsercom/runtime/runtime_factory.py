"""Abstract base class for Tsercom runtime factories."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance

# ExposedData import removed as DataTypeT is no longer bound to it here.
# SerializableAnnotatedInstance might become unused in this file
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.async_poller import AsyncPoller

EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT")  # No longer constrained by ExposedData


class RuntimeFactory(
    Generic[DataTypeT, EventTypeT],
    RuntimeInitializer[DataTypeT, EventTypeT],
    ABC,
):
    """Defines the contract for factories that create Runtime instances.

    Extends `RuntimeInitializer` and requires implementations to provide
    a `RemoteDataReader` and an `AsyncPoller` for event handling.

    Type Args:
        DataTypeT: The generic type of data objects the runtime processes.
        EventTypeT: The generic type of event objects the runtime processes.
    """

    @property
    @abstractmethod
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[DataTypeT]]:
        """Provides a `RemoteDataReader` for accessing annotated data instances.

        Subclasses must implement this property.
        """

    @property
    @abstractmethod
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[EventTypeT]]:
        """Provides an `AsyncPoller` for receiving event instances.

        Subclasses must implement this property.
        """

    @abstractmethod
    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[DataTypeT]]:
        """Internal abstract method by subclasses to provide the data reader.

        This method is typically called by the `remote_data_reader` property.

        Returns:
            A `RemoteDataReader` instance.
        """

    @abstractmethod
    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[EventTypeT]]:
        """Internal abstract method by subclasses to provide the event poller.

        This method is typically called by the `event_poller` property.

        Returns:
            An `AsyncPoller` instance.
        """

    def _stop(self) -> None:
        """
        Stops any underlying calls and executions associated with this instance.
        """
