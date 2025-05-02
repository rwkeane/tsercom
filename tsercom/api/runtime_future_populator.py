from concurrent.futures import Future
from typing import Generic, TypeVar

from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.runtime_handle import RuntimeHandle


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")

class RuntimeFuturePopulator(
        Generic[TDataType, TEventType],
        RuntimeFactoryFactory[TDataType, TEventType].Client):
    def __init__(self, future : Future[RuntimeHandle[TDataType, TEventType]]):
        self.__future = future

    def _on_handle_ready(self, handle: RuntimeHandle[TDataType, TEventType]):
        assert not self.__future.running()
        self.__future.set_result(handle)