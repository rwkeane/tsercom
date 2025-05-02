from concurrent.futures import Future
from typing import Generic, TypeVar
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class InitializationPair(Generic[TDataType, TEventType]):
    def __init__(
        self,
        handle_future: Future[RuntimeHandle[TDataType, TEventType]],
        initializer: RuntimeInitializer[TDataType, TEventType],
    ):
        self.__handle_future = handle_future
        self.__initializer = initializer

    @property
    def handle_future(self):
        return self.__handle_future

    @property
    def initializer(self):
        return self.__initializer
