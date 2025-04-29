from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class RuntimeInitializer(ABC, Generic[TDataType, TEventType]):
    """
    A base class for server and client runtime initializer instances. Mainly
    used to simplify sharing of code between client and server.
    """
    @abstractmethod
    def create(
        self,
        thread_watcher : ThreadWatcher,
        data_handler : RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime[TEventType]:
        """
        Creates a new Runtime instance. This method will only be called once
        per instance.

        |data_handler| is the object responsible for providing Event data from
        the RuntimeHandle, as well as providing a path to send data back to that
        instance.
        |grpc_channel_factory| is a factory used to create gRPC Channels, per
        user specification.
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
