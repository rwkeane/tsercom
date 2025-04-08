from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
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
        # TODO: Add a parameter for the GrpcChannel
        data_reader: RemoteDataReader[TDataType],
    ) -> Runtime[TEventType]:
        """
        Creates a new Runtime instance. This method will only be called once
        per instance.

        |clock| is the Clock used for sycnhronizing with remote clients.
        |data_reader| is the endpoint to which received data should be passed.
        """
        pass

    def client(self) -> RemoteDataAggregator[TDataType].Client | None:
        """
        Returns the client that should be informed when new data is provided to
        the RemoteDataAggregator instance created for the runtime created from
        this initializer, or None if no such instance should be used.
        """
        return None

    def timeout(self) -> int | None:
        """
        Returns the timeout (in seconds) that should be used for data received
        by the runtime created from this initializer, or None if data should not
        time out.
        """
        return 60
