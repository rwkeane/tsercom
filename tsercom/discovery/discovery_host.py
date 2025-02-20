from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Generic, Optional, TypeVar, overload

from discovery.mdns.instance_listener import InstanceListener
from discovery.service_info import ServiceInfo
from util.caller_id.caller_identifier import CallerIdentifier
from util.threading.task_runner import TaskRunner


TServiceInfo = TypeVar('TServiceInfo', bound = ServiceInfo)
class DiscoveryHost(Generic[TServiceInfo], InstanceListener.Client):
    """
    Helper object, to wrap CallerId management (for reconnection) with service
    discovery.
    """
    class Client(ABC):
        @abstractmethod
        async def _on_service_added(self,
                                    connection_info : TServiceInfo,
                                    caller_id : CallerIdentifier):
            pass
        
    @overload
    def __init__(self,
                 task_runner : TaskRunner,
                 *,
                 service_type : str):
        pass
    
    @overload
    def __init__(self,
                 task_runner : TaskRunner,
                 *,
                 instance_listener_factory : \
                        Callable[[TaskRunner], InstanceListener]):
        pass

    def __init__(self,
                 task_runner : TaskRunner,
                 *,
                 service_type : Optional[str] = None,
                 instance_listener_factory : Optional[
                        Callable[[TaskRunner], InstanceListener]] = None):
        assert (not service_type is None) != \
               (not instance_listener_factory is None)

        self.__task_runner = task_runner
        self.__service_type = service_type
        self.__instance_listener_factory = instance_listener_factory

        self.__discoverer : InstanceListener = None
        self.__client : DiscoveryHost.Client = None

        self.__caller_id_map : Dict[str, CallerIdentifier] = {}
    
    def start_discovery(self, client : 'DiscoveryHost.Client'):
        """
        Starts discovery. Results are returned by a call to the |client|.
        """
        if not self.__task_runner.is_running_on_task_runner():
            self.__task_runner.post_task(partial(self.start_discovery, client))
            return
        
        assert not client is None
        assert self.__discoverer is None

        self.__client = client
        if not self.__instance_listener_factory is None:
            self.__discoverer = self.__instance_listener_factory(
                    self.__task_runner)
        else:
            self.__discoverer = InstanceListener[TServiceInfo](
                    self, self.__task_runner, self.__service_type)
        
    async def _on_service_added(self, connection_info : TServiceInfo):
        print("ENDPOINT FOUND")
        assert not self.__client is None
        caller_id : CallerIdentifier = None
        if connection_info.mdns_name in self.__caller_id_map:
            caller_id = self.__caller_id_map[connection_info.mdns_name]
        else:
            caller_id = CallerIdentifier()
            self.__caller_id_map[connection_info.mdns_name] = caller_id

        await self.__client._on_service_added(connection_info, caller_id)
        

        
    