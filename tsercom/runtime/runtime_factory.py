from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class RuntimeFactory(
    ABC,
    Generic[TDataType, TEventType],
    RuntimeInitializer[TDataType, TEventType],
):
    @property
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        pass

    @property
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        pass

    @abstractmethod
    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        pass

    @abstractmethod
    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        pass
