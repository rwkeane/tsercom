from typing import Dict, Optional
from zeroconf import IPVersion, ServiceInfo, Zeroconf

from tsercom.util.ip import get_all_addresses

class RecordPublisher:
    """
    Low-level object to publish a service to various mDNS Records.
    """
    def __init__(self,
                 name : str,
                 type_ : str,
                 port : int,
                 properties : Optional[Dict[str, Optional[str]]] = None):
        assert not type_ is None and type_[0] == "_", type_

        if properties is None:
            properties = {}

        self.__ptr = "{0}._tcp.local.".format(type_)
        self.__srv = "{0}.{1}".format(name, self.__ptr)
        self.__port = port
        self.__txt = properties
        # print("publishing txt:",self.__txt)
        print("Publishing service:", self.__srv)

    def publish(self):
        info = ServiceInfo(
            self.__ptr,
            self.__srv,
            addresses=get_all_addresses(),
            port=self.__port,
            properties=self.__txt
        )

        zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        zeroconf.register_service(info)
        
        # print("Published mDNS Record")