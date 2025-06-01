"""Defines LocalRuntimeFactory, responsible for creating and configuring Runtime instances that operate within the same process as the caller."""

from typing import Generic, TypeVar

from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher

TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType", bound=ExposedData)


class LocalRuntimeFactory(
    Generic[TDataType, TEventType], RuntimeFactory[TDataType, TEventType]
):
    """Factory for creating Runtime instances that operate in the local process.

    This factory utilizes a `RuntimeInitializer` to construct the core runtime
    and then links it with a `RuntimeCommandBridge` for communication.
    It also manages data reading and event polling mechanisms specific to
    local process operations.
    """

    def __init__(
        self,
        initializer: RuntimeInitializer[TDataType, TEventType],
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_poller: AsyncPoller[EventInstance[TEventType]],
        bridge: RuntimeCommandBridge,
    ) -> None:
        """Initializes a LocalRuntimeFactory.

        Args:
            initializer: The initializer responsible for creating the runtime core.
            data_reader: The reader for incoming data instances.
            event_poller: The poller for incoming event instances.
            bridge: The bridge for command communication with the runtime.
        """
        self.__initializer = initializer
        self.__data_reader = data_reader
        self.__event_poller = event_poller
        self.__bridge = bridge

        super().__init__(other_config=self.__initializer)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory | None,
    ) -> Runtime:
        """Creates a new Runtime instance.

        This method uses the stored `RuntimeInitializer` to construct the
        runtime, then sets up the command bridge for this specific runtime instance.

        Args:
            thread_watcher: A ThreadWatcher instance to monitor runtime threads.
            data_handler: A handler for processing data and events within the runtime.
            grpc_channel_factory: A factory for creating gRPC channels if needed by the runtime.

        Returns:
            The newly created and configured Runtime instance.
        """
        runtime = self.__initializer.create(
            thread_watcher, data_handler, grpc_channel_factory
        )
        self.__bridge.set_runtime(runtime)
        return runtime

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        """Provides the remote data reader for the runtime.

        This method is part of the `RuntimeFactory`'s contract to make
        the data reader available.

        Returns:
            The `RemoteDataReader` instance configured for this factory.
        """
        return self.__data_reader

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        """Provides the event poller for the runtime.

        This method is part of the `RuntimeFactory`'s contract to make
        the event poller available.

        Returns:
            The `AsyncPoller` instance for events, configured for this factory.
        """
        return self.__event_poller

    @property
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        """Gets the RemoteDataReader used by runtimes created by this factory. This reader handles incoming data instances."""
        return self._remote_data_reader()

    @property
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        """Gets the AsyncPoller used by runtimes created by this factory. This poller handles incoming event instances."""
        return self._event_poller()
