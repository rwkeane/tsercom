"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, TypeVar

from tsercom.common.system.torch_utils import TORCH_IS_AVAILABLE
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
from tsercom.threading.multiprocess.delegating_queue_factory import (
    DelegatingMultiprocessQueueFactory,
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
    """

    def __init__(
        self, thread_pool: ThreadPoolExecutor, thread_watcher: ThreadWatcher
    ) -> None:
        """Initializes the SplitRuntimeFactoryFactory.

        Args:
            thread_pool: ThreadPoolExecutor for async tasks (e.g. data aggregator).
            thread_watcher: ThreadWatcher to monitor threads from components
                            like ShimRuntimeHandle.
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
        """Creates a handle and factory for a split-process runtime.

        Sets up 3 pairs of multiprocess queues (events, data, commands).
        Creates `RemoteRuntimeFactory` (for child process) and
        `ShimRuntimeHandle` (for parent) using these queues.

        Args:
            initializer: Configuration for the runtime.

        Returns:
            A tuple: (ShimRuntimeHandle, RemoteRuntimeFactory).
        """
        # --- Dynamic queue factory selection ---
        # Command queues always use the default factory as they don't transport tensors.
        command_queue_factory: MultiprocessQueueFactory[RuntimeCommand] = (
            DefaultMultiprocessQueueFactory[RuntimeCommand]()
        )

        event_queue_factory: MultiprocessQueueFactory[
            EventInstance[EventTypeT]
        ]
        data_queue_factory: MultiprocessQueueFactory[
            AnnotatedInstance[DataTypeT]
        ]

        if TORCH_IS_AVAILABLE:
            # If PyTorch is available, use the Delegating factory which will then
            # decide at runtime (first put()) whether to use Torch or Default queues.
            event_queue_factory = DelegatingMultiprocessQueueFactory[
                EventInstance[EventTypeT]
            ]()
            data_queue_factory = DelegatingMultiprocessQueueFactory[
                AnnotatedInstance[DataTypeT]
            ]()
        else:
            # If PyTorch is not available, fall back to default queues directly.
            event_queue_factory = DefaultMultiprocessQueueFactory[
                EventInstance[EventTypeT]
            ]()
            data_queue_factory = DefaultMultiprocessQueueFactory[
                AnnotatedInstance[DataTypeT]
            ]()
        # --- End dynamic queue factory selection ---

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

        if initializer.timeout_seconds is not None:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                # pylint: disable=W0511 # type: ignore [arg-type] # TODO: Client expects RemoteDataAggregator[DataTypeT], gets [AnnotatedInstance[DataTypeT]]
                client=initializer.data_aggregator_client,
                timeout=initializer.timeout_seconds,
            )
        else:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                # pylint: disable=W0511 # type: ignore [arg-type] # TODO: Client expects RemoteDataAggregator[DataTypeT], gets [AnnotatedInstance[DataTypeT]]
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
