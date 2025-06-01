"""Defines the LocalRuntimeFactoryFactory for creating pairs of RuntimeHandles and LocalRuntimeFactory instances."""

from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Tuple
from tsercom.data.annotated_instance import (
    AnnotatedInstance,
)  # Import AnnotatedInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.api.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.local_process.local_runtime_factory import LocalRuntimeFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller

TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class LocalRuntimeFactoryFactory(
    RuntimeFactoryFactory[TDataType, TEventType]
):  # Added type parameters
    """Factory class responsible for creating LocalRuntimeFactory instances
    and their associated RuntimeHandles for local process runtimes.
    """

    def __init__(self, thread_pool: ThreadPoolExecutor) -> None:
        """Initializes the LocalRuntimeFactoryFactory.

        Args:
            thread_pool: A ThreadPoolExecutor for managing asynchronous tasks.
        """
        super().__init__()
        self.__thread_pool = thread_pool

    def _create_pair(
        self, initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> Tuple[
        RuntimeHandle[TDataType, TEventType],
        RuntimeFactory[TDataType, TEventType],
    ]:
        """Creates a RuntimeHandle and its corresponding LocalRuntimeFactory.

        This method sets up the necessary components for a local runtime,
        including data aggregation, event polling, and command bridging.

        Args:
            initializer: The RuntimeInitializer containing configuration for the runtime.

        Returns:
            A tuple containing the created RuntimeHandle and the LocalRuntimeFactory.
        """
        if initializer.timeout_seconds is not None:
            data_aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[TDataType]
            ](
                self.__thread_pool,
                client=initializer.data_aggregator_client,  # type: ignore [arg-type] # TODO: Client expects RemoteDataAggregator[TDataType], gets [AnnotatedInstance[TDataType]]
                timeout=initializer.timeout_seconds,
            )
        else:
            data_aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[TDataType]
            ](
                self.__thread_pool,
                client=initializer.data_aggregator_client,  # type: ignore [arg-type] # TODO: Client expects RemoteDataAggregator[TDataType], gets [AnnotatedInstance[TDataType]]
            )
        event_poller = AsyncPoller[EventInstance[TEventType]]()
        bridge = RuntimeCommandBridge()

        factory = LocalRuntimeFactory[TDataType, TEventType](
            initializer, data_aggregator, event_poller, bridge
        )
        handle = RuntimeWrapper(event_poller, data_aggregator, bridge)

        return handle, factory
