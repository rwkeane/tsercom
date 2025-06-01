"""Abstract base class for Tsercom runtime factories."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance

# SerializableAnnotatedInstance might become unused in this file
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller
from tsercom.data.exposed_data import ExposedData  # Import ExposedData


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType", bound=ExposedData)  # Constrain TDataType


class RuntimeFactory(
    Generic[TDataType, TEventType],
    RuntimeInitializer[TDataType, TEventType],
    ABC,
):
    """Defines the contract for factories that create Runtime instances.

    Extends `RuntimeInitializer` and requires implementations to provide
    a `RemoteDataReader` and an `AsyncPoller` for event handling.
    """

    @property
    @abstractmethod
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        """Provides a `RemoteDataReader` for accessing annotated data instances.

        Subclasses must implement this property.
        """
        pass

    @property
    @abstractmethod
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:  # Reverted to EventInstance
        """Provides an `AsyncPoller` for receiving event instances.

        Subclasses must implement this property.
        """
        pass

    @abstractmethod
    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        """Internal abstract method to be implemented by subclasses to provide the data reader.

        This method is typically called by the `remote_data_reader` property.

        Returns:
            A `RemoteDataReader` instance.
        """
        pass

    @abstractmethod
    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:  # Reverted to EventInstance
        """Internal abstract method to be implemented by subclasses to provide the event poller

        This method is typically called by the `event_poller` property.

        Returns:
            An `AsyncPoller` instance.
        """
        pass

    def _stop(self) -> None:
        """
        Stops any underlying calls and executions associated with this instance.
        """
        pass
