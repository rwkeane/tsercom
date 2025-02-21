import threading
from typing import Callable, Dict, Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier


TType = TypeVar("TType")
class CallerIdMap(Generic[TType]):
    def __init__(self):
        self.__lock = threading.Lock()
        self.__map : Dict[CallerIdentifier, TType] = {}
    
    def find_instance(self,
                      caller_id : CallerIdentifier,
                      factory : Callable[[], TType]):
        with self.__lock:
            if not caller_id in self.__map:
                self.__map[caller_id] = factory()
            
            return self.__map[caller_id]
    
    def for_all_items(self, function : Callable[[TType], None]):
        items = []
        with self.__lock:
            items = list(self.__map.values())

        for val in items:
            function(val)
    
    def count(self):
        with self.__lock:
            return len(self.__map)
        
    def __len__(self):
        return self.count()