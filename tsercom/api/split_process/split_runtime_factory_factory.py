"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, TypeVar

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.remote_runtime_factory import (
    RemoteRuntimeFactory,
)
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.data.annotated_instance import (
    AnnotatedInstance,
)  # Import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
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
        event_sink: MultiprocessQueueSink[EventInstance[EventTypeT]]
        event_source: MultiprocessQueueSource[EventInstance[EventTypeT]]
        event_sink, event_source = create_multiprocess_queues()

        data_sink: MultiprocessQueueSink[AnnotatedInstance[DataTypeT]]
        data_source: MultiprocessQueueSource[AnnotatedInstance[DataTypeT]]
        data_sink, data_source = create_multiprocess_queues()

        runtime_command_sink: MultiprocessQueueSink[RuntimeCommand]
        runtime_command_source: MultiprocessQueueSource[RuntimeCommand]
        runtime_command_sink, runtime_command_source = (
            create_multiprocess_queues()
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
