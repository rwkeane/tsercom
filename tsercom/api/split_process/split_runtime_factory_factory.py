"""Defines a factory for creating runtime factories and handles for split-process runtimes."""

from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Tuple

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


TDataType = TypeVar("TDataType")  # Generic type for data.
TEventType = TypeVar("TEventType")  # Generic type for events.


class SplitRuntimeFactoryFactory(RuntimeFactoryFactory[TDataType, TEventType]):
    """Creates factories and handles for runtimes operating in separate processes.

    This factory specializes in setting up the necessary inter-process communication
    mechanisms (queues) and instantiating `RemoteRuntimeFactory` and
    `ShimRuntimeHandle` to manage a runtime in a child process.
    """

    def __init__(
        self, thread_pool: ThreadPoolExecutor, thread_watcher: ThreadWatcher
    ) -> None:
        """Initializes the SplitRuntimeFactoryFactory.

        Args:
            thread_pool: A ThreadPoolExecutor for asynchronous tasks, primarily
                         used by the data aggregator.
            thread_watcher: A ThreadWatcher to monitor threads created by
                            components like ShimRuntimeHandle.
        """
        super().__init__()

        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__thread_watcher: ThreadWatcher = thread_watcher

    def _create_pair(
        self, initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> Tuple[
        RuntimeHandle[TDataType, TEventType],
        RuntimeFactory[TDataType, TEventType],
    ]:
        """Creates a handle and factory for a split-process runtime.

        This method sets up three pairs of multiprocess queues for events,
        data, and runtime commands. It then creates a `RemoteRuntimeFactory`
        (intended for the child process) and a `ShimRuntimeHandle` (for the
        parent process) that use these queues to communicate.

        Args:
            initializer: The RuntimeInitializer containing configuration for
                         the runtime to be created.

        Returns:
            A tuple containing:
                - ShimRuntimeHandle: The handle to interact with the remote runtime.
                - RemoteRuntimeFactory: The factory to create the runtime in the
                                        remote process.
        """
        # Each returns a (sink, source) pair.
        event_sink, event_source = create_multiprocess_queues()
        data_sink, data_source = create_multiprocess_queues()
        runtime_command_sink, runtime_command_source = (
            create_multiprocess_queues()
        )

        # It gets the source end of event/command queues and sink end of data queue.
        factory = RemoteRuntimeFactory[TDataType, TEventType](
            initializer, event_source, data_sink, runtime_command_source
        )

        aggregator = RemoteDataAggregatorImpl[
            TDataType
        ](
            self.__thread_pool,
            client=initializer.data_aggregator_client,  # Client to consume aggregated data.
            timeout=initializer.timeout_seconds,
        )

        # It gets the sink end of event/command queues and source end of data queue.
        runtime_handle = ShimRuntimeHandle[TDataType, TEventType](
            self.__thread_watcher,
            event_sink,  # Sink for events to the remote runtime.
            data_source,  # Source for data from the remote runtime.
            runtime_command_sink,  # Sink for commands to the remote runtime.
            aggregator,  # The local aggregator for data.
        )

        return runtime_handle, factory
