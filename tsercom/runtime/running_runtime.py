from abc import abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_initializer import RuntimeInitializer


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RunningRuntime(Generic[TDataType, TEventType], Runtime[TEventType]):
    def __init__(self):
        super().__init__()

    @property
    def data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        return self._get_remote_data_aggregator()

    @property
    def initializer(self) -> RuntimeInitializer[TDataType, TEventType]:
        return self._get_initializer()

    @abstractmethod
    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        raise NotImplementedError()

    @abstractmethod
    def _get_initializer(self) -> RuntimeInitializer[TDataType, TEventType]:
        raise NotImplementedError()
