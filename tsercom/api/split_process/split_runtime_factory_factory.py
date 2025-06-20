"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, get_args

import torch

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
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.thread_watcher import ThreadWatcher

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


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
    ) -> tuple[
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
        resolved_data_type = None
        resolved_event_type = None

        # Prioritize inspecting the initializer's direct __orig_class__ (e.g., for MyInitializer[torch.Tensor, str])
        if hasattr(initializer, "__orig_class__"):
            generic_args = get_args(initializer.__orig_class__)
            if generic_args and len(generic_args) == 2:
                if not isinstance(generic_args[0], TypeVar):
                    resolved_data_type = generic_args[0]
                if not isinstance(generic_args[1], TypeVar):
                    resolved_event_type = generic_args[1]

        # Fallback: Iterate __orig_bases__ to find the RuntimeInitializer[SpecificA, SpecificB]
        if resolved_data_type is None or resolved_event_type is None:
            for base in getattr(initializer, "__orig_bases__", []):
                if (
                    hasattr(base, "__origin__")
                    and base.__origin__ is RuntimeInitializer
                ):
                    base_generic_args = get_args(base)
                    if base_generic_args and len(base_generic_args) == 2:
                        if resolved_data_type is None and not isinstance(
                            base_generic_args[0], TypeVar
                        ):
                            resolved_data_type = base_generic_args[0]
                        if resolved_event_type is None and not isinstance(
                            base_generic_args[1], TypeVar
                        ):
                            resolved_event_type = base_generic_args[1]
                        if (
                            resolved_data_type is not None
                            and resolved_event_type is not None
                        ):
                            break

        # Declare data_event_queue_factory with the base type for mypy
        event_queue_factory: MultiprocessQueueFactory[EventInstance[EventTypeT]]
        data_queue_factory: MultiprocessQueueFactory[AnnotatedInstance[DataTypeT]]
        command_queue_factory: MultiprocessQueueFactory[RuntimeCommand]

        uses_torch_tensor = False
        if resolved_data_type is torch.Tensor or resolved_event_type is torch.Tensor:
            uses_torch_tensor = True

        if uses_torch_tensor:
            # Assuming EventInstance and AnnotatedInstance generics are compatible with Torch queues
            event_queue_factory = TorchMultiprocessQueueFactory[
                EventInstance[EventTypeT]
            ]()
            data_queue_factory = TorchMultiprocessQueueFactory[
                AnnotatedInstance[DataTypeT]
            ]()
        else:
            event_queue_factory = DefaultMultiprocessQueueFactory[
                EventInstance[EventTypeT]
            ]()
            data_queue_factory = DefaultMultiprocessQueueFactory[
                AnnotatedInstance[DataTypeT]
            ]()

        # Command queues always use the default factory
        command_queue_factory = DefaultMultiprocessQueueFactory[RuntimeCommand]()
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
