"""DataHost ABC for hosts that expose a RemoteDataAggregator."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


# pylint: disable=R0903 # Abstract data hosting interface
class DataHost(ABC, Generic[DataTypeT]):
    """Abstract base class for data hosts that expose a RemoteDataAggregator.

    Subclasses (like ClientHost and ServerHost) should inherit from DataHost
    to provide a standardized way of accessing a `RemoteDataAggregator`.
    This pattern simplifies data aggregation logic and avoids duplication.
    """

    @property
    def remote_data_aggregator(self) -> RemoteDataAggregator[DataTypeT]:
        """The `RemoteDataAggregator` instance for this data host.

        This property provides access to the aggregator, which is responsible
        for collecting and providing data from a remote source.

        Returns:
            The `RemoteDataAggregator[TDataType]` instance.
        """
        # Delegates to the abstract method that subclasses must implement.
        return self._remote_data_aggregator()

    @abstractmethod
    def _remote_data_aggregator(self) -> RemoteDataAggregator[DataTypeT]:
        """Provides the `RemoteDataAggregator` instance.

        This abstract method must be implemented by subclasses to return their
        specific instance of `RemoteDataAggregator`. This aggregator is then
        exposed publicly via the `remote_data_aggregator` property.

        Returns:
            An instance of `RemoteDataAggregator[DataTypeT]`.
        """
        # Abstract method: subclasses must provide the implementation.
