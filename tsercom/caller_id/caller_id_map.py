import threading
from typing import Callable, Dict, Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier

TType = TypeVar("TType")


class CallerIdMap(Generic[TType]):

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__map: Dict[CallerIdentifier, TType] = {}

    def find_instance(
        self, caller_id: CallerIdentifier, factory: Callable[[], TType]
    ) -> TType:
        with self.__lock:
            if caller_id not in self.__map:
                self.__map[caller_id] = factory()

            return self.__map[caller_id]

    def for_all_items(self, function: Callable[[TType], None]) -> None:
        items = []
        with self.__lock:
            items = list(self.__map.values())

        for val in items:
            function(val)

    def count(self) -> int:
        with self.__lock:
            return len(self.__map)

    def __len__(self) -> int:
        return self.count()
