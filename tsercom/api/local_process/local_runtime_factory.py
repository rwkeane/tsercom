"""LocalRuntimeFactory: creates/configures local Runtime instances."""

from typing import Generic, TypeVar

# pylint: disable=C0301 # Black-formatted import
from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader

# pylint: disable=C0301 # Black-formatted import
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime

# pylint: disable=C0301 # Black-formatted import
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_factory import RuntimeFactory

# pylint: disable=C0301 # Black-formatted import
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher

EventTypeT = TypeVar("EventTypeT")
DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


class LocalRuntimeFactory(
    Generic[DataTypeT, EventTypeT], RuntimeFactory[DataTypeT, EventTypeT]
):
    """Factory for local-process Runtime instances.

    Uses RuntimeInitializer for core runtime, links with RuntimeCommandBridge.
    Manages local data reading and event polling.
    """

    def __init__(
        self,
        initializer: RuntimeInitializer[
            DataTypeT, EventTypeT
        ],  # pylint: disable=C0301
        data_reader: RemoteDataReader[
            AnnotatedInstance[DataTypeT]
        ],  # pylint: disable=C0301
        event_poller: AsyncPoller[
            EventInstance[EventTypeT]
        ],  # pylint: disable=C0301
        bridge: RuntimeCommandBridge,
    ) -> None:
        """Initializes a LocalRuntimeFactory.

        Args:
            initializer: Creates the runtime core.
            data_reader: Reader for incoming data.
            event_poller: Poller for incoming events.
            bridge: Bridge for command communication.
        """
        self.__initializer = initializer
        self.__data_reader = data_reader
        self.__event_poller = event_poller
        self.__bridge = bridge

        super().__init__(other_config=self.__initializer)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[
            DataTypeT, EventTypeT
        ],  # pylint: disable=C0301
        grpc_channel_factory: GrpcChannelFactory | None,
    ) -> Runtime:
        """Creates a new Runtime instance.

        Uses stored `RuntimeInitializer` to construct runtime, then sets up
        the command bridge for this runtime instance.

        Args:
            thread_watcher: ThreadWatcher to monitor runtime threads.
            data_handler: Handler for data/events within runtime.
            grpc_channel_factory: Factory for gRPC channels if runtime needs them.

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
    ) -> RemoteDataReader[AnnotatedInstance[DataTypeT]]:
        """Provides the remote data reader for the runtime.

        Part of `RuntimeFactory` contract to make data reader available.

        Returns:
            The `RemoteDataReader` for this factory.
        """
        return self.__data_reader

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[EventTypeT]]:
        """Provides the event poller for the runtime.

        Part of `RuntimeFactory` contract to make event poller available.

        Returns:
            The `AsyncPoller` for events, configured for this factory.
        """
        return self.__event_poller

    @property
    def remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[DataTypeT]]:
        """Gets RemoteDataReader for runtimes by this factory (handles data)."""  # pylint: disable=C0301
        return self._remote_data_reader()

    @property
    def event_poller(
        self,
    ) -> AsyncPoller[EventInstance[EventTypeT]]:
        """Gets AsyncPoller for runtimes by this factory (handles events)."""  # pylint: disable=C0301
        return self._event_poller()
