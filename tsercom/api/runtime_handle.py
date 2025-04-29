from abc import abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.runtime.runtime_initializer import RuntimeInitializer


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
TInitializerType = TypeVar(
    "TInitializerType", bound=RuntimeInitializer[TDataType, TEventType]
)


class RuntimeHandle(Generic[TDataType, TEventType, TInitializerType]):
    @property
    def data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        """
        The RemoteDataAggregator that can be used to retrieve data from this
        runtime.
        """
        return self._get_remote_data_aggregator()

    @property
    def initializer(self) -> TInitializerType:
        """
        The initializer with which this runtime is associated.
        """
        return self._get_initializer()

    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        raise NotImplementedError()

    @abstractmethod
    def on_event(self, event: TEventType):
        raise NotImplementedError()

    @abstractmethod
    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        raise NotImplementedError()

    @abstractmethod
    def _get_initializer(self) -> TInitializerType:
        raise NotImplementedError()
