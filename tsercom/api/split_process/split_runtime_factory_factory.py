"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from typing import (
    Tuple,
    TypeVar,
)

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
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)

# Removed: TorchMultiprocessQueueFactory (Delegating factory handles it internally if needed,
# but SplitFactoryFactory doesn't choose it directly anymore)
from tsercom.threading.multiprocess.delegating_multiprocess_queue_factory import (
    DelegatingMultiprocessQueueFactory,
    _TORCH_AVAILABLE,  # Changed from is_torch_available
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


# pylint: disable=R0903 # Concrete factory implementation
class SplitRuntimeFactoryFactory(RuntimeFactoryFactory[DataTypeT, EventTypeT]):
    """Creates factories and handles for split-process runtimes.

    Sets up IPC queues and instantiates `RemoteRuntimeFactory` and
    `ShimRuntimeHandle` for managing a runtime in a child process.
    Uses `DelegatingMultiprocessQueueFactory` if PyTorch is available,
    allowing dynamic selection of queue transport based on first data item.
    """

    def __init__(
        self, thread_pool: ThreadPoolExecutor, thread_watcher: ThreadWatcher
    ) -> None:
        """Initializes the SplitRuntimeFactoryFactory.

        Args:
            thread_pool: ThreadPoolExecutor for async tasks.
            thread_watcher: ThreadWatcher to monitor threads.
        """
        super().__init__()

        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__thread_watcher: ThreadWatcher = thread_watcher

    def _create_pair(
        self, initializer: RuntimeInitializer[DataTypeT, EventTypeT]
    ) -> Tuple[
        RuntimeHandle[DataTypeT, EventTypeT],
        RuntimeFactory[DataTypeT, EventTypeT],
    ]:
        """Creates a handle and factory for a split-process runtime."""

        event_queue_factory: MultiprocessQueueFactory[
            EventInstance[EventTypeT]
        ]
        data_queue_factory: MultiprocessQueueFactory[
            AnnotatedInstance[DataTypeT]
        ]

        # Top-level check for PyTorch availability.
        # If available, Delegating factory will decide specific queue type (torch/default)
        # based on the first item sent.
        # If not available, always use default queues.
        if _TORCH_AVAILABLE:  # Changed from is_torch_available()
            event_queue_factory = DelegatingMultiprocessQueueFactory[
                EventInstance[EventTypeT]
            ]()
            data_queue_factory = DelegatingMultiprocessQueueFactory[
                AnnotatedInstance[DataTypeT]
            ]()
        else:
            event_queue_factory = DefaultMultiprocessQueueFactory[
                EventInstance[EventTypeT]
            ]()
            data_queue_factory = DefaultMultiprocessQueueFactory[
                AnnotatedInstance[DataTypeT]
            ]()

        # Command queues always use the default factory.
        command_queue_factory: MultiprocessQueueFactory[RuntimeCommand]
        command_queue_factory = DefaultMultiprocessQueueFactory[
            RuntimeCommand
        ]()

        event_sink: MultiprocessQueueSink[EventInstance[EventTypeT]]
        event_source: MultiprocessQueueSource[EventInstance[EventTypeT]]
        event_sink, event_source = event_queue_factory.create_queues()

        data_sink: MultiprocessQueueSink[AnnotatedInstance[DataTypeT]]
        data_source: MultiprocessQueueSource[AnnotatedInstance[DataTypeT]]
        data_sink, data_source = data_queue_factory.create_queues()

        runtime_command_sink: MultiprocessQueueSink[RuntimeCommand]
        runtime_command_source: MultiprocessQueueSource[RuntimeCommand]
        runtime_command_sink, runtime_command_source = (
            command_queue_factory.create_queues()
        )

        factory = RemoteRuntimeFactory[DataTypeT, EventTypeT](
            initializer, event_source, data_sink, runtime_command_source
        )

        # Aggregator setup remains the same
        if initializer.timeout_seconds is not None:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                client=initializer.data_aggregator_client,
                timeout=initializer.timeout_seconds,
            )
        else:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
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
