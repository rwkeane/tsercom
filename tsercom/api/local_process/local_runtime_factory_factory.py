"""Defines the LocalRuntimeFactoryFactory for creating pairs of RuntimeHandles and LocalRuntimeFactory instances."""

from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Tuple

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

# Type variables for generic typing
TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class LocalRuntimeFactoryFactory(RuntimeFactoryFactory):
    """Factory class responsible for creating LocalRuntimeFactory instances
    and their associated RuntimeHandles for local process runtimes.
    """
    def __init__(self, thread_pool: ThreadPoolExecutor) -> None:
        """Initializes the LocalRuntimeFactoryFactory.

        Args:
            thread_pool: A ThreadPoolExecutor for managing asynchronous tasks.
        """
        super().__init__()
        # Store the thread pool for use in creating data aggregators.
        self.__thread_pool: ThreadPoolExecutor = thread_pool

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
        data_aggregator = RemoteDataAggregatorImpl[TDataType](
            self.__thread_pool,
            client=initializer.data_aggregator_client,
            timeout=initializer.timeout_seconds,
        )
        event_poller = AsyncPoller[EventInstance[TEventType]]()
        bridge = RuntimeCommandBridge()

        factory = LocalRuntimeFactory[TDataType, TEventType](
            initializer, data_aggregator, event_poller, bridge
        )
        handle = RuntimeWrapper(event_poller, data_aggregator, bridge)

        return handle, factory
