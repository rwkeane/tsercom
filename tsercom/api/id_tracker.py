import threading
from typing import Dict, Optional, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier


class IdTracker:
    """
    Helper object for creating and managing a thread-safe bidirectional
    dictionary, to map between the local CallerID and the address / port.
    """

    def __init__(self):
        self.__lock = threading.Lock()
        self.__address_to_id: Dict[tuple[str, int], CallerIdentifier] = {}
        self.__id_to_address: Dict[CallerIdentifier, tuple[str, int]] = {}

    @overload
    def try_get(self, id: CallerIdentifier) -> str | None:
        pass

    @overload
    def try_get(self, address: str, port: int) -> CallerIdentifier | None:
        pass

    def try_get(
        self,
        id: Optional[CallerIdentifier] = None,
        address: Optional[str] = None,
        port: Optional[int] = None,
    ):
        assert (id is None) != (address is None), (id, address, port)
        assert (address is None) == (port is None), (id, address, port)

        if id is not None:
            with self.__lock:
                if id not in self.__id_to_address:
                    return None
                return self.__id_to_address[id]

        else:
            with self.__lock:
                if id not in self.__address_to_id:
                    return None
                return self.__address_to_id[(address, port)]

    @overload
    def get(self, id: CallerIdentifier) -> str:
        pass

    @overload
    def get(self, address: str, port: int) -> CallerIdentifier:
        pass

    def get(
        self,
        id: Optional[CallerIdentifier] = None,
        address: Optional[str] = None,
        port: Optional[int] = None,
    ):
        result = self.try_get(id, address, port)
        if result is None:
            raise KeyError()

        return result

    def add(self, id: CallerIdentifier, address: str, port: int):
        with self.__lock:
            if id in self.__id_to_address:
                raise KeyError()
            if (address, port) in self.__address_to_id:
                raise KeyError()

            self.__address_to_id[(address, port)] = id
            self.__id_to_address[id] = (address, port)

    def has_id(self, id: CallerIdentifier):
        with self.__lock:
            return id in self.__id_to_address

    def has_address(self, address: str, port: int):
        with self.__lock:
            return (address, port) in self.__address_to_id

    def __len__(self):
        with self.__lock:
            assert len(self.__id_to_address) == len(self.__address_to_id)
            return len(self.__id_to_address)

    def __iter__(self):
        with self.__lock:
            return self.__id_to_address.__iter__()
