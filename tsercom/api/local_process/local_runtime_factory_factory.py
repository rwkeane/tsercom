"""LocalRuntimeFactoryFactory for local RuntimeHandle/Factory pairs."""

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, TypeVar

from tsercom.api.local_process.local_runtime_factory import LocalRuntimeFactory
from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.api.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.data.annotated_instance import (
    AnnotatedInstance,
)  # Import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.async_poller import AsyncPoller

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


# pylint: disable=R0903 # Concrete factory implementation
class LocalRuntimeFactoryFactory(
    RuntimeFactoryFactory[DataTypeT, EventTypeT]
):  # Added type parameters
    """Creates LocalRuntimeFactory instances and associated RuntimeHandles."""

    def __init__(self, thread_pool: ThreadPoolExecutor) -> None:
        """Initializes the LocalRuntimeFactoryFactory.

        Args:
            thread_pool: ThreadPoolExecutor for managing asynchronous tasks.
        """
        super().__init__()
        self.__thread_pool = thread_pool

    def _create_pair(
        self, initializer: RuntimeInitializer[DataTypeT, EventTypeT]
    ) -> Tuple[
        RuntimeHandle[DataTypeT, EventTypeT],
        RuntimeFactory[DataTypeT, EventTypeT],
    ]:
        """Creates a RuntimeHandle and its corresponding LocalRuntimeFactory.

        Sets up components for a local runtime: data aggregation, event
        polling, and command bridging.

        Args:
            initializer: Configuration for the runtime.

        Returns:
            A tuple (RuntimeHandle, LocalRuntimeFactory).
        """
        if initializer.timeout_seconds is not None:
            data_aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                # pylint: disable=W0511,C0301 # type: ignore [arg-type] # TODO: Client expects RemoteDataAggregator[DataTypeT], gets [AnnotatedInstance[DataTypeT]]
                client=initializer.data_aggregator_client,
                timeout=initializer.timeout_seconds,
            )
        else:
            data_aggregator = RemoteDataAggregatorImpl[
                AnnotatedInstance[DataTypeT]
            ](
                self.__thread_pool,
                # pylint: disable=W0511,C0301 # type: ignore [arg-type] # TODO: Client expects RemoteDataAggregator[DataTypeT], gets [AnnotatedInstance[DataTypeT]]
                client=initializer.data_aggregator_client,
            )
        event_poller = AsyncPoller[EventInstance[EventTypeT]]()
        bridge = RuntimeCommandBridge()
        factory = LocalRuntimeFactory[DataTypeT, EventTypeT](
            initializer, data_aggregator, event_poller, bridge
        )
        handle = RuntimeWrapper(event_poller, data_aggregator, bridge)

        return handle, factory
