from abc import ABC, abstractmethod
from functools import partial
import socket
from typing import Dict, Generic, List, Optional, TypeVar

from discovery.service_info import ServiceInfo
from discovery.mdns.record_listener import RecordListener
from util.threading.task_runner import TaskRunner

TServiceInfo = TypeVar('TServiceInfo', bound = ServiceInfo)
class InstanceListener(Generic[TServiceInfo], RecordListener.Client):
    """
    Searches for instances of the specified type.
    """
    class Client(ABC):
        @abstractmethod
        async def _on_service_added(self, connection_info : TServiceInfo):
            raise NotImplementedError("This method must be implemented!")
        
    def __init__(self,
                 client : 'InstanceListener.Client',
                 task_runner : TaskRunner,
                 service_type : str):
        assert not client is None and issubclass(
                type(client), InstanceListener.Client)
        assert issubclass(type(task_runner), TaskRunner)
        assert isinstance(service_type, str)
        
        self.__client = client
        self.__task_runner = task_runner

        self.__listener = RecordListener(self, service_type)

    def __populate_service_info(self,
                               record_name : str,
                               port : int,
                               addresses : List[bytes],
                               txt_record : Dict[bytes, Optional[str]]):
        if len(addresses) == 0:
            print(f"Failed to connect to service at UNKNOWN:{port}")
            return None
        
        addresses_out = []
        for i in range(len(addresses)):
            try:
                address = socket.inet_ntoa(addresses[i])
                addresses_out.append(address)
            except:
                continue

        if (len(addresses_out) == 0):
            print(f"Failed to connect to service at {addresses[0]}:{port}")
        
        name_encoded = "name".encode('utf-8') 
        if name_encoded in txt_record:  # b'str' is byte casted 'str'
            readable_name : bytes = txt_record[name_encoded]
            readable_name = readable_name.decode('utf-8')
        else:
            readable_name = record_name

        return ServiceInfo(readable_name, port, addresses_out, record_name)
    
    def _convert_service_info(
            self,
            service_info : ServiceInfo,
            txt_record : Dict[bytes, Optional[str]]) -> TServiceInfo:
        return service_info
    
    def _on_service_added(self,
                          record_name : str,
                          port : int,
                          addresses : List[bytes],
                          txt_record : Dict[bytes, Optional[str]]):
        service_info = self.__populate_service_info(
                record_name, port, addresses, txt_record)
        service_info = self._convert_service_info(service_info, txt_record)
        self.__task_runner.post_task(
                partial(self.__client._on_service_added, service_info))
        
        