"""Defines SplitRuntimeFactoryFactory for creating factories for remote, split-process runtimes."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, TypeVar, Any # Added Any, Tuple

from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.remote_runtime_factory import RemoteRuntimeFactory
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)
from tsercom.threading.thread_watcher import ThreadWatcher

# Type variable for the raw data type handled by the runtime.
TDataType = TypeVar("TDataType")
# Type variable for the event type handled by the runtime.
TEventType = TypeVar("TEventType")


class SplitRuntimeFactoryFactory(RuntimeFactoryFactory[
    AnnotatedInstance[TDataType], # RuntimeHandle data type will be AnnotatedInstance of raw TDataType
    TEventType
]):
    """A factory for creating RemoteRuntimeFactory instances and their ShimRuntimeHandles.

    This factory specializes in setting up runtimes that operate in a separate
    process. It orchestrates the creation of multiprocess queues for inter-process
    communication, linking a `ShimRuntimeHandle` (in the main process) with a
    `RemoteRuntimeFactory` (which will create the runtime in the spawned process).
    """
    def __init__(
        self, thread_pool: ThreadPoolExecutor, thread_watcher: ThreadWatcher
    ) -> None:
        """Initializes the SplitRuntimeFactoryFactory.

        Args:
            thread_pool: A ThreadPoolExecutor for managing asynchronous tasks,
                         such as those within the data aggregator.
            thread_watcher: A ThreadWatcher for monitoring threads, for instance,
                            those started by DataReaderSource within ShimRuntimeHandle.
        """
        super().__init__()
        # Store the thread pool and watcher for component instantiation.
        self.__thread_pool: ThreadPoolExecutor = thread_pool
        self.__thread_watcher: ThreadWatcher = thread_watcher

    def _create_pair(
        self, initializer: RuntimeInitializer[TDataType, TEventType] # Initializer uses raw TDataType
    ) -> Tuple[
        RuntimeHandle[AnnotatedInstance[TDataType], TEventType], # Handle uses AnnotatedInstance[TDataType]
        RuntimeFactory[TDataType, TEventType], # Remote factory uses raw TDataType for its generic signature
    ]:
        """Creates a ShimRuntimeHandle and its corresponding RemoteRuntimeFactory.

        This method is the core of the factory. It sets up three sets of
        multiprocess queues for:
        1. Events: From main process (ShimRuntimeHandle) to remote process (Runtime).
        2. Data: From remote process (Runtime) to main process (ShimRuntimeHandle).
        3. Commands: From main process (ShimRuntimeHandle) to remote process (Runtime).

        Args:
            initializer: The RuntimeInitializer. Its `TDataType` is the raw data type.
                         It provides configuration like the data aggregator client and timeouts.

        Returns:
            A tuple containing:
                - ShimRuntimeHandle: The handle for the main process to interact with
                                     the remote runtime. Its TDataType is
                                     `AnnotatedInstance[TDataType_from_initializer]`.
                - RemoteRuntimeFactory: The factory to be used in the remote process.
                                        Its TDataType is `TDataType_from_initializer`.
        """
        # Create queue for events (main process -> remote process)
        # ShimRuntimeHandle sends events, RemoteRuntimeFactory's EventSource receives them.
        # EventSource expects EventInstance[TEventType].
        event_to_runtime_sink, event_to_runtime_source = create_multiprocess_queues[
            EventInstance[TEventType]
        ]()

        # Create queue for data (remote process -> main process)
        # RemoteRuntimeFactory's DataReaderSink sends data, ShimRuntimeHandle's DataReaderSource receives it.
        # DataReaderSink in RemoteRuntimeFactory sends AnnotatedInstance[TDataType].
        data_from_runtime_sink, data_from_runtime_source = create_multiprocess_queues[
            AnnotatedInstance[TDataType] # TDataType here is the raw type from initializer
        ]()

        # Create queue for runtime commands (main process -> remote process)
        # ShimRuntimeHandle sends commands, RemoteRuntimeFactory's RuntimeCommandSource receives them.
        command_to_runtime_sink, command_to_runtime_source = create_multiprocess_queues[
            RuntimeCommand
        ]()

        # Data aggregator for the ShimRuntimeHandle in the main process.
        # It receives AnnotatedInstance[TDataType] from the data_from_runtime_source queue.
        data_aggregator = RemoteDataAggregatorImpl[AnnotatedInstance[TDataType]](
            self.__thread_pool,
            client=initializer.data_aggregator_client, # This client operates on raw TDataType
            timeout=initializer.timeout_seconds,
        )

        # Create the ShimRuntimeHandle for the main process.
        # Its TDataType is AnnotatedInstance[TDataType_from_initializer].
        handle = ShimRuntimeHandle[AnnotatedInstance[TDataType], TEventType](
            self.__thread_watcher,
            event_queue=event_to_runtime_sink,      # Sends EventInstance[TEventType]
            data_queue=data_from_runtime_source,    # Receives AnnotatedInstance[TDataType]
            runtime_command_queue=command_to_runtime_sink, # Sends RuntimeCommand
            data_aggregator=data_aggregator,        # Aggregates AnnotatedInstance[TDataType]
        )

        # Create the RemoteRuntimeFactory for the remote process.
        # Its TDataType is TDataType_from_initializer (raw data type).
        # It will produce/consume raw TDataType internally but send AnnotatedInstance.
        factory = RemoteRuntimeFactory[TDataType, TEventType](
            initializer,
            event_source_queue=event_to_runtime_source,    # Receives EventInstance[TEventType]
            data_reader_queue=data_from_runtime_sink,      # Sends AnnotatedInstance[TDataType]
            command_source_queue=command_to_runtime_source # Receives RuntimeCommand
        )

        return handle, factory
