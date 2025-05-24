from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.remote_runtime_factory import (
    RemoteRuntimeFactory,
)
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class SplitRuntimeFactoryFactory(RuntimeFactoryFactory):
    def __init__(
        self, thread_pool: ThreadPoolExecutor, thread_watcher: ThreadWatcher
    ):
        super().__init__()

        self.__thread_pool = thread_pool
        self.__thread_watcher = thread_watcher

    def _create_pair(
        self, initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> tuple[
        RuntimeHandle[TDataType, TEventType],
        RuntimeFactory[TDataType, TEventType],
    ]:
        event_sink, event_source = create_multiprocess_queues()
        data_sink, data_source = create_multiprocess_queues()
        runtime_command_sink, runtime_command_source = (
            create_multiprocess_queues()
        )

        factory = RemoteRuntimeFactory[TDataType, TEventType](
            initializer, event_source, data_sink, runtime_command_source
        )

        aggregator = RemoteDataAggregatorImpl[TDataType](
            self.__thread_pool,
            client=initializer.data_aggregator_client,
            timeout=initializer.timeout_seconds,
        )

        runtime = ShimRuntimeHandle[TDataType, TEventType](
            self.__thread_watcher,
            event_sink,
            data_source, 
            runtime_command_sink,
            aggregator, 
        )

        return runtime, factory
