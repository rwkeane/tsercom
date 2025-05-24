from abc import abstractmethod
import datetime
from typing import Generic, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeHandle(Generic[TDataType, TEventType]):
    """
    This class provides a "Handle" for a currently running Runtime. In other
    words, it provides simple APIs to control and send local events to the
    Runtime, as well as a simplified way to receive data back from it.
    """

    @property
    def data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[TDataType]]:
        """
        The RemoteDataAggregator that can be used to retrieve data from this
        runtime.
        """
        return self._get_remote_data_aggregator()

    @abstractmethod
    def start(self):
        """
        Starts the service.
        """
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        """
        Stops the service.
        """
        raise NotImplementedError()

    @overload
    def on_event(self, event: TEventType) -> None: ...

    """
    Sends the |event| to the runtime to be sent out to connected endpoints with
    ANY CallerId.
    """

    @overload
    def on_event(
        self, event: TEventType, caller_id: CallerIdentifier
    ) -> None: ...

    """
    Sends the |event| to the runtime with |caller_id| as specified, or does
    nothing if no endpoint with that ID is connected.
    """

    @overload
    def on_event(
        self, event: TEventType, *, timestamp: datetime.datetime
    ) -> None: ...

    """
    Sends the |event| to the runtime to be sent out to connected endpoints with
    ANY CallerId. Uses |timestamp| as the time associated with |event| instead
    of the current time.
    """

    @overload
    def on_event(
        self,
        event: TEventType,
        caller_id: CallerIdentifier,
        *,
        timestamp: datetime.datetime,
    ) -> None: ...

    """
    Sends the |event| to the runtime with |caller_id| as specified, or does
    nothing if no endpoint with that ID is connected. Uses |timestamp| as the
    time associated with |event| instead of the current time.
    """

    @abstractmethod
    def on_event(
        self,
        event: TEventType,
        caller_id: Optional[CallerIdentifier] = None,
        *,
        timestamp: Optional[datetime.datetime] = None,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _get_remote_data_aggregator(
        self,
    ) -> RemoteDataAggregator[AnnotatedInstance[TDataType]]:
        raise NotImplementedError()
