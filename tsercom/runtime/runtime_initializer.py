"""Defines the abstract base class for Tsercom runtime initializers."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_config import RuntimeConfig
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT")  # No longer constrained by ExposedData
EventTypeT = TypeVar("EventTypeT")


class RuntimeInitializer(
    ABC,
    Generic[DataTypeT, EventTypeT],
    RuntimeConfig[DataTypeT],
):
    """Abstract base class for Tsercom runtime initializers.

    This class provides the blueprint for creating user-defined `Runtime`
    instances. Subclasses must implement the `create` method to define
    the specific instantiation logic for their runtime.

    The `RuntimeConfig` part of its inheritance provides configuration access,
    while `Generic[DataTypeT, EventTypeT]` defines the data and event types
    the runtime will handle.

    Type Args:
        DataTypeT: The generic type of data objects the runtime processes.
        EventTypeT: The generic type of event objects the runtime processes.
    """

    @abstractmethod
    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[DataTypeT, EventTypeT],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        """
        Creates a new Runtime instance. This method will only be called once
        per instance.

        |thread_watcher| provides APIs for error handling, and is required for
        calling many Tsercom APIs.
        data_handler: The `RuntimeDataHandler` responsible for providing event
            data from the `RuntimeHandle` and for sending data back to that instance.
        grpc_channel_factory: A factory used to create gRPC channels as per
            user specification. This is a required dependency.
        """
