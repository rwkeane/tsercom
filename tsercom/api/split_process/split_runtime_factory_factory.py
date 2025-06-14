"""Factory for creating split-process runtime factories and handles."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, TypeVar, Any, cast
import torch # Add torch import for type checking

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
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
    QueueTypeT as DefaultQueueTypeT # Alias to avoid conflict
)
# Import the new TorchMultiprocessQueueFactory
from tsercom.threading.multiprocess.torch_queue_factory import (
    TorchMultiprocessQueueFactory,
    QueueTypeT as TorchQueueTypeT # Alias to avoid conflict
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
        Dynamically selects queue type based on DataTypeT and EventTypeT.
        """
        use_torch_queues = False

        # Attempt to determine if torch.Tensor is used for DataTypeT or EventTypeT
        # This relies on how the class is parameterized at instantiation.
        # e.g., factory = SplitRuntimeFactoryFactory[torch.Tensor, OtherType](...)
        # type(self) gives the concrete class, possibly with generics resolved.
        # We need to inspect the generic arguments provided to RuntimeFactoryFactory.

        # __orig_class__ usually holds the generic class before substitution.
        # If self is an instance of SplitRuntimeFactoryFactory[SpecificType1, SpecificType2],
        # then __orig_class__ might be SplitRuntimeFactoryFactory[DataTypeT, EventTypeT] (the definition)
        # or it could be SplitRuntimeFactoryFactory[SpecificType1, SpecificType2] if accessed on a
        # fully resolved type object.
        # The type arguments are typically found in type(self).__parameters__ for the TypeVars
        # or in type(self).__args__ if it's a concrete generic alias.

        # More reliable for instances: check __orig_class__ if it exists and has __args__
        # This is for when SplitRuntimeFactoryFactory is instantiated with concrete types.
        # e.g. MyFactory = SplitRuntimeFactoryFactory[torch.Tensor, int]
        #      instance = MyFactory() -> type(instance) is MyFactory
        #      We need to trace back to SplitRuntimeFactoryFactory's parameterization for MyFactory.

        # Start with type(self) and walk __orig_bases__ to find RuntimeFactoryFactory
        # This is complex. A simpler, though less direct, approach if specific initializers
        # are used for tensor types would be to check `isinstance(initializer, TorchTensorInitializer)`.
        # However, the prompt requires checking DataTypeT/EventTypeT.

        actual_data_type: Any = None
        actual_event_type: Any = None

        # Try to get actual types from generic instantiation if possible
        # This is highly dependent on Python version and how generics are used.
        if hasattr(self, "__orig_class__"): # Python 3.7+ for instances of generic classes
            # self.__orig_class__ is SplitRuntimeFactoryFactory[DataTypeT, EventTypeT] with resolved types
            type_args = getattr(self.__orig_class__, "__args__", None)
            if type_args and len(type_args) == 2:
                actual_data_type = type_args[0]
                actual_event_type = type_args[1]

        # If DataTypeT or EventTypeT are themselves TypeVars bound to torch.Tensor,
        # this check will see the TypeVar, not the bound.
        # Example: T = TypeVar('T', bound=torch.Tensor)
        # MyFactory = SplitRuntimeFactoryFactory[T, int]
        # Here, actual_data_type would be T. We'd need to check T.__bound__.

        if isinstance(actual_data_type, TypeVar):
            actual_data_type = getattr(actual_data_type, '__bound__', actual_data_type)
        if isinstance(actual_event_type, TypeVar):
            actual_event_type = getattr(actual_event_type, '__bound__', actual_event_type)

        if actual_data_type == torch.Tensor or actual_event_type == torch.Tensor:
            use_torch_queues = True

        # Ensure torch is imported if we are going to reference torch.Tensor
        # It's already imported at the top.

        event_sink: MultiprocessQueueSink[EventInstance[EventTypeT]]
        event_source: MultiprocessQueueSource[EventInstance[EventTypeT]]
        data_sink: MultiprocessQueueSink[AnnotatedInstance[DataTypeT]]
        data_source: MultiprocessQueueSource[AnnotatedInstance[DataTypeT]]
        runtime_command_sink: MultiprocessQueueSink[RuntimeCommand]
        runtime_command_source: MultiprocessQueueSource[RuntimeCommand]

        if use_torch_queues:
            # The create_queues methods are generic, type inference should work.
            event_sink, event_source = TorchMultiprocessQueueFactory.create_queues()
            data_sink, data_source = TorchMultiprocessQueueFactory.create_queues()
            # RuntimeCommand queue typically does not handle tensors.
            runtime_command_sink, runtime_command_source = create_multiprocess_queues()
        else:
            event_sink, event_source = create_multiprocess_queues()
            data_sink, data_source = create_multiprocess_queues()
            runtime_command_sink, runtime_command_source = create_multiprocess_queues()

        factory = RemoteRuntimeFactory[DataTypeT, EventTypeT](
            initializer, event_source, data_sink, runtime_command_source
        )

        # Cast is necessary here because RemoteDataAggregatorImpl is generic on InstanceTypeT,
        # and initializer.data_aggregator_client expects RemoteDataAggregatorClient[DataTypeT],
        # but RemoteDataAggregatorImpl provides RemoteDataAggregator[AnnotatedInstance[DataTypeT]].
        # This discrepancy needs careful handling or adjustment of generic constraints.
        # For now, we assume the existing type: ignore or a cast is acceptable.

        aggregator_client_casted = cast(Any, initializer.data_aggregator_client)

        if initializer.timeout_seconds is not None:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                client=aggregator_client_casted,
                timeout=initializer.timeout_seconds,
            )
        else:
            aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                client=aggregator_client_casted,
            )

        runtime_handle = ShimRuntimeHandle[DataTypeT, EventTypeT](
            self.__thread_watcher,
            event_sink,
            data_source,
            runtime_command_sink,
            aggregator,
        )

        return runtime_handle, factory
