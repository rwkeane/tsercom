from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.runtime.runtime import Runtime


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class ServerRuntimeInitializer(
    ABC, Generic[TDataType, TEventType], RuntimeInitializer[TDataType]
):

    @abstractmethod
    def create(
        self,
        data_reader: RemoteDataReader[TDataType],
        grpc_channel_factory : GrpcChannelFactory,
        clock: SynchronizedClock,
    ) -> Runtime[TEventType]:
        """
        Creates a new Runtime instance. This method will only be called once
        per instance.

        |data_reader| is the endpoint to which received data should be passed.
        |grpc_channel_factory| is a factory used to create gRPC Channels, per
        user specification.
        |clock| is the Clock used for sycnhronizing with remote clients.
        """
        pass
