from typing import List


class ServiceInfo:
    """
    Data-holder object for the results of an mDNS Query.
    """

    def __init__(
        self, name: str, port: int, addresses: List[str], mdns_name: str
    ):
        self.__mdns_name = mdns_name
        self.__name = name
        self.__port = port
        self.__addresses = addresses

    @property
    def mdns_name(self) -> str:
        return self.__mdns_name

    @property
    def name(self) -> str:
        return self.__name

    @property
    def port(self) -> int:
        return self.__port

    @property
    def addresses(self) -> List[str]:
        return self.__addresses

