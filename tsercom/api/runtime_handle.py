from abc import abstractmethod
import datetime
from typing import Generic, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeHandle(Generic[TDataType, TEventType]):
    @property
    def data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        """
        The RemoteDataAggregator that can be used to retrieve data from this
        runtime.
        """
        return self._get_remote_data_aggregator()
    
    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        raise NotImplementedError()

    @overload
    def on_event(self, event: TEventType):
        ...

    @overload
    def on_event(self, event: TEventType, caller_id : CallerIdentifier):
        ...

    @overload
    def on_event(self, event: TEventType, *, timestamp : datetime.datetime):
        ...

    @overload
    def on_event(self, event: TEventType, caller_id : CallerIdentifier, *, timestamp : datetime.datetime):
        ...

    @abstractmethod
    def on_event(self, event: TEventType, caller_id : Optional[CallerIdentifier] = None, *, timestamp : Optional[datetime.datetime] = None):
        raise NotImplementedError()

    @abstractmethod
    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        raise NotImplementedError()
