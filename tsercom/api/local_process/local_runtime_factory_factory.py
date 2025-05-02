from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar
from tsercom.api.local_process.runtime_command_bridge import RuntimeCommandBridge
from tsercom.api.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.local_process.local_runtime_factory import LocalRuntimeFactory
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class LocalRuntimeFactoryFactory(RuntimeFactoryFactory[TDataType, TEventType]):
    def __init__(self, thread_pool : ThreadPoolExecutor):
        super().__init__()

        self.__thread_pool = thread_pool

    def _create_pair(self, initializer : RuntimeInitializer[TDataType, TEventType]) -> tuple[RuntimeHandle[TDataType, TEventType], RuntimeFactory[TDataType, TEventType]]:
        data_aggregator = RemoteDataAggregatorImpl[TDataType](self.__thread_pool, initializer.client(), initializer.timeout())
        event_poller = AsyncPoller[EventInstance[TEventType]]()
        bridge = RuntimeCommandBridge()
        
        factory = LocalRuntimeFactory[TDataType, TEventType](initializer, data_aggregator, event_poller, bridge)
        handle = RuntimeWrapper(event_poller, data_aggregator, bridge)

        return handle, factory