import threading
from typing import Generic, TypeVar


TType = TypeVar('TType')
class Atomic(Generic[TType]):
    """
    This class provides atomic access (via locks) to an underlying type, both
    synchronously and asynchronously.
    """
    def __init__(self, value : TType):
        self.__value : TType = value
        self.__lock = threading.Lock()

    def set(self, value):
        with self.__lock:
            self.__value = value

    def get(self):
        with self.__lock:
            return self.__value
        