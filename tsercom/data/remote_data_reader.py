"""RemoteDataReader ABC: interface for components processing remote data."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


# pylint: disable=R0903 # Abstract data reading interface
class RemoteDataReader(ABC, Generic[DataTypeT]):
    """Abstract interface for classes that process incoming remote data.

    This interface defines a single method, `_on_data_ready`, which should be
    implemented by concrete classes to handle new data items as they arrive.
    It's often used as a light-weight stand-in or a component of more complex
    data handling systems like `RemoteDataOrganizer` or `RemoteDataAggregator`.
    """

    @abstractmethod
    def _on_data_ready(self, new_data: DataTypeT) -> None:
        """Callback method to process a new data item.

        Implementers should define the logic to handle the `new_data`
        when this method is invoked.

        Args:
            new_data: The new data item of type `DataTypeT` received.
        """
