"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from multiprocessing.context import BaseContext
from typing import Any, TypeVar

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.remote_runtime_factory import (
    RemoteRuntimeFactory,
)
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.data.annotated_instance import (
    AnnotatedInstance,
)
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.multiprocess.multiprocessing_context_provider import (
    MultiprocessingContextProvider,
)
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class SplitRuntimeFactoryFactory(RuntimeFactoryFactory[DataTypeT, EventTypeT]):
    """Creates factories and handles for split-process runtimes.

    Sets up IPC queues and instantiates `RemoteRuntimeFactory` and
    `ShimRuntimeHandle` for managing a runtime in a child process.
    It uses MultiprocessingContextProvider to determine the appropriate
    multiprocessing context and queue factories.
    """

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        thread_watcher: ThreadWatcher,
    ) -> None:
        """Initializes the SplitRuntimeFactoryFactory.

        Args:
            thread_pool: ThreadPoolExecutor for async tasks.
            thread_watcher: ThreadWatcher to monitor threads.
        """
        super().__init__()

        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__thread_watcher: ThreadWatcher = thread_watcher
        # Default to "spawn" context method as it is generally safer and
        # widely compatible. MultiprocessingContextProvider will determine
        # internally whether to use torch or standard context.
        # Since this provider instance's factory is used for different queue
        # types (events, data), we use Any here.
        self.__mp_context_provider: MultiprocessingContextProvider[Any] = (
            MultiprocessingContextProvider[Any](context_method="spawn")
        )

    @property
    def multiprocessing_context(self) -> BaseContext:
        return self.__mp_context_provider.context

    def _create_pair(
        self, initializer: RuntimeInitializer[DataTypeT, EventTypeT]
    ) -> tuple[
        RuntimeHandle[DataTypeT, EventTypeT],
        RuntimeFactory[DataTypeT, EventTypeT],
    ]:
        """Creates a handle and factory for a split-process runtime.

        Sets up IPC queues using the MultiprocessingContextProvider.
        Creates `RemoteRuntimeFactory` (for child process) and
        `ShimRuntimeHandle` (for parent) using these queues.

        Args:
            initializer: Configuration for the runtime.

        Returns:
            A tuple: (ShimRuntimeHandle, RemoteRuntimeFactory).
        """
        mp_context = self.__mp_context_provider.context
        queue_factory_instance = self.__mp_context_provider.queue_factory

        max_ipc_q_size = initializer.max_ipc_queue_size
        is_ipc_block = initializer.is_ipc_blocking
        event_sink, event_source = queue_factory_instance.create_queues(
            max_ipc_queue_size=max_ipc_q_size,
            is_ipc_blocking=is_ipc_block,
        )
        data_sink, data_source = queue_factory_instance.create_queues(
            max_ipc_queue_size=max_ipc_q_size,
            is_ipc_blocking=is_ipc_block,
        )

        # Command queues use a Default factory but with the context derived
        # from the provider, ensuring consistency if the main context is,
        # for example, PyTorch-specific.
        command_queue_factory = DefaultMultiprocessQueueFactory[RuntimeCommand](
            context=mp_context
        )
        runtime_command_sink: MultiprocessQueueSink[RuntimeCommand]
        runtime_command_source: MultiprocessQueueSource[RuntimeCommand]
        runtime_command_sink, runtime_command_source = (
            command_queue_factory.create_queues()
        )

        factory = RemoteRuntimeFactory[DataTypeT, EventTypeT](
            initializer,
            event_source,
            data_sink,
            runtime_command_source,
        )

        if initializer.timeout_seconds is not None:
            aggregator = RemoteDataAggregatorImpl[AnnotatedInstance[DataTypeT]](
                self.__thread_pool,
                client=initializer.data_aggregator_client,
                timeout=initializer.timeout_seconds,
            )
        else:
            aggregator = RemoteDataAggregatorImpl[AnnotatedInstance[DataTypeT]](
                self.__thread_pool,
                client=initializer.data_aggregator_client,
            )

        runtime_handle = ShimRuntimeHandle[DataTypeT, EventTypeT](
            self.__thread_watcher,
            event_sink,
            data_source,
            runtime_command_sink,
            aggregator,
        )

        return runtime_handle, factory
