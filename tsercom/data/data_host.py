from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator


TDataType = TypeVar("TDataType", bound = ExposedData)
class DataHost(ABC, Generic[TDataType]):
    """
    This is the base class that all ClientHost and ServerHost classes should
    inheret from, to allow for simple exposure of a RemoteDataAggregator in a
    user-friendly way, while still taking care of the maintenance that would
    otherwise need to be duplicated between all such servers.
    """

    @property
    def remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        return self._remote_data_aggregator()
    
    @abstractmethod
    def _remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        pass