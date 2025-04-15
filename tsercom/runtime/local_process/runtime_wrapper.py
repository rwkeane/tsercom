from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.running_runtime import RunningRuntime
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.server.server_runtime_initializer import (
    ServerRuntimeInitializer,
)

TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeWrapper(
    Generic[TDataType, TEventType],
    RunningRuntime[TDataType, TEventType],
    RemoteDataReader[TDataType],
):
    def __init__(
        self,
        runtime: Runtime,
        data_aggregator: RemoteDataAggregatorImpl[TDataType],
        initializer: ServerRuntimeInitializer[TDataType, TEventType],
    ):
        self.__runtime = runtime
        self.__aggregator = data_aggregator
        self.__initializer = initializer

    def start_async(self):
        self.__runtime.start_async()

    def on_event(self, event: TEventType):
        self.__runtime.on_event(event)

    async def stop(self) -> None:
        self.__runtime.stop()

    def _on_data_ready(self, new_data: TDataType) -> None:
        self.__aggregator._on_data_ready(new_data)

    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        return self.__aggregator

    def _get_initializer(
        self,
    ) -> ServerRuntimeInitializer[TDataType, TEventType]:
        return self.__initializer
