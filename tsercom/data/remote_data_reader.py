"""Defines the RemoteDataReader abstract base class, an interface for components that process incoming remote data."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData

# Generic type for the data that the reader will handle, bound by ExposedData.
TDataType = TypeVar("TDataType", bound=ExposedData)


class RemoteDataReader(ABC, Generic[TDataType]):
    """Abstract interface for classes that process incoming remote data.

    This interface defines a single method, `_on_data_ready`, which should be
    implemented by concrete classes to handle new data items as they arrive.
    It's often used as a light-weight stand-in or a component of more complex
    data handling systems like `RemoteDataOrganizer` or `RemoteDataAggregator`.
    """

    @abstractmethod
    def _on_data_ready(self, new_data: TDataType) -> None:
        """Callback method to process a new data item.

        Implementers should define the logic to handle the `new_data`
        when this method is invoked.

        Args:
            new_data: The new data item of type `TDataType` that has been received.
        """
        pass
