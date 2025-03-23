from abc import abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.runtime.runtime import Runtime


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
class RunningRuntime(Generic[TDataType, TEventType], Runtime[TEventType]):
    def __init__(self):
        super().__init__()

    @property
    def data_aggregator(self):
        return self._get_remote_data_aggregator()

    @abstractmethod
    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        raise NotImplementedError()