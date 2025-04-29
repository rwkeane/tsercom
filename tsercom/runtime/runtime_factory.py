from typing import TypeVar
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.threading.thread_watcher import ThreadWatcher


TEventType = TypeVar("TEventType")
class RuntimeFactory:
    def __init__(self):
        pass
    
    def create(
        self,
        thread_watcher : ThreadWatcher,
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime[TEventType]:
        pass