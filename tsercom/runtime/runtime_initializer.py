"""Defines the abstract base class for Tsercom runtime initializers."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_config import RuntimeConfig
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.data.exposed_data import ExposedData  # Import ExposedData


TDataType = TypeVar("TDataType", bound=ExposedData)  # Constrain TDataType
TEventType = TypeVar("TEventType")


class RuntimeInitializer(
    ABC, Generic[TDataType, TEventType], RuntimeConfig[TDataType]
):
    """
    This class is to be implemented to specify creation of user-defined
    Runtime instances.
    """

    @abstractmethod
    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        """
        Creates a new Runtime instance. This method will only be called once
        per instance.

        |thread_watcher| provides APIs for error handling, and is required for
        calling many Tsercom APIs.
        |data_handler| is the object responsible for providing Event data from
        the RuntimeHandle, as well as providing a path to send data back to that
        instance.
        |grpc_channel_factory| is a factory used to create gRPC Channels, per
        user specification.
        """
        pass
