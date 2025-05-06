from typing import Generic, TypeVar

from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher

TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class LocalRuntimeFactory(
    Generic[TDataType, TEventType], RuntimeFactory[TDataType, TEventType]
):
    def __init__(
        self,
        initializer: RuntimeInitializer,
        data_reader: RemoteDataReader[AnnotatedInstance[TDataType]],
        event_poller: AsyncPoller[EventInstance[TEventType]],
        bridge: RuntimeCommandBridge,
    ):
        self.__initializer = initializer
        self.__data_reader = data_reader
        self.__event_poller = event_poller
        self.__bridge = bridge

        super().__init__(other_config=self.__initializer)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        runtime = self.__initializer.create(
            thread_watcher, data_handler, grpc_channel_factory
        )
        self.__bridge.set_runtime(runtime)
        return runtime

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        return self.__data_reader

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        return self.__event_poller
