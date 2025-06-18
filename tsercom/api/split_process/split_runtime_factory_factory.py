"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, TypeVar

# Removed get_args as it's no longer needed for dynamic type inspection
# import torch # No longer needed directly in this file for queue selection

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
# Removed DefaultMultiprocessQueueFactory and TorchMultiprocessQueueFactory
# Removed MultiprocessQueueFactory as Delegating... is now used directly
from tsercom.threading.multiprocess.delegating_queue_factory import (
    DelegatingMultiprocessQueueFactory,
)
# DefaultMultiprocessQueueFactory might still be needed if command queues are truly always default
# For now, assuming Delegating for all based on prompt. If command queues have special needs,
# this might need to be DefaultMultiprocessQueueFactory.
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory, # Keeping for command queue as per original logic, can be changed if subtask implies all.
)                                     # Re-evaluating: prompt says "always use DelegatingMultiprocessQueueFactory"
                                      # This implies command_queue_factory should also be Delegating.

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

    Sets up IPC queues using DelegatingMultiprocessQueueFactory and
    instantiates `RemoteRuntimeFactory` and `ShimRuntimeHandle`
    for managing a runtime in a child process.
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

        Sets up 3 pairs of multiprocess queues (events, data, commands)
        using DelegatingMultiprocessQueueFactory.
        Creates `RemoteRuntimeFactory` (for child process) and
        `ShimRuntimeHandle` (for parent) using these queues.

        Args:
            initializer: Configuration for the runtime.

        Returns:
            A tuple: (ShimRuntimeHandle, RemoteRuntimeFactory).
        """
        # --- Simplified queue factory instantiation ---
        # Always use DelegatingMultiprocessQueueFactory for data and events.
        # The type parameter T for DelegatingMultiprocessQueueFactory should match the item in the queue.
        event_queue_factory = DelegatingMultiprocessQueueFactory[
            EventInstance[EventTypeT]
        ]()
        data_queue_factory = DelegatingMultiprocessQueueFactory[
            AnnotatedInstance[DataTypeT]
        ]()

        # Per prompt "always use DelegatingMultiprocessQueueFactory", command queues also use it.
        command_queue_factory = DelegatingMultiprocessQueueFactory[
            RuntimeCommand
        ]()
        # --- End simplified queue factory instantiation ---

        event_sink: MultiprocessQueueSink[EventInstance[EventTypeT]]
        event_source: MultiprocessQueueSource[EventInstance[EventTypeT]]

        # The most logical interpretation for IPC is that each channel needs ONE queue.
        # So we need 3 queues:
        event_channel_queue = event_queue_factory.create_queue()
        data_channel_queue = data_queue_factory.create_queue()
        command_channel_queue = command_queue_factory.create_queue()

        # Now, Sinks and Sources wrap these individual queues.
        # Events: Parent (Shim) writes, Child (Remote) reads
        shim_event_sink = MultiprocessQueueSink[EventInstance[EventTypeT]](event_channel_queue)
        remote_event_source = MultiprocessQueueSource[EventInstance[EventTypeT]](event_channel_queue)

        # Data: Child (Remote) writes, Parent (Shim) reads
        remote_data_sink = MultiprocessQueueSink[AnnotatedInstance[DataTypeT]](data_channel_queue)
        shim_data_source = MultiprocessQueueSource[AnnotatedInstance[DataTypeT]](data_channel_queue)

        # Commands: Parent (Shim) writes, Child (Remote) reads
        shim_command_sink = MultiprocessQueueSink[RuntimeCommand](command_channel_queue)
        remote_command_source = MultiprocessQueueSource[RuntimeCommand](command_channel_queue)

        # This setup makes sense for IPC.

        factory = RemoteRuntimeFactory[DataTypeT, EventTypeT](
            initializer, remote_event_source, remote_data_sink, remote_command_source
        )

        if initializer.timeout_seconds is not None:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                client=initializer.data_aggregator_client, # type: ignore[arg-type]
                timeout=initializer.timeout_seconds,
            )
        else:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                client=initializer.data_aggregator_client, # type: ignore[arg-type]
            )

        runtime_handle = ShimRuntimeHandle[DataTypeT, EventTypeT](
            self.__thread_watcher,
            shim_event_sink,
            shim_data_source,
            shim_command_sink,
            aggregator,
        )

        return runtime_handle, factory
