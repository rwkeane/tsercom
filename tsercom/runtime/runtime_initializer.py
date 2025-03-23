from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.runtime.runtime import Runtime


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeInitializer(ABC, Generic[TDataType, TEventType]):
    # TODO: Split into client and server so that the |clock| only used in create
    # when appropriate.
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(
        self,
        clock: SynchronizedClock,
        data_reader: RemoteDataReader[TDataType],
    ) -> Runtime[TEventType]:
        pass
