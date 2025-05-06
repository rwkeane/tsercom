from abc import ABC, abstractmethod
from typing import TypeVar
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class RuntimeFactoryFactory(ABC):
    class Client:
        @abstractmethod
        def _on_handle_ready(
            self, handle: RuntimeHandle[TDataType, TEventType]
        ):
            pass

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def _create_pair(
        self, initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> tuple[
        RuntimeHandle[TDataType, TEventType],
        RuntimeFactory[TDataType, TEventType],
    ]:
        pass

    def create_factory(
        self,
        client: "RuntimeFactoryFactory[TDataType, TEventType].Client",
        initializer: RuntimeInitializer[TDataType, TEventType],
    ) -> RuntimeFactory[TDataType, TEventType]:
        assert client is not None
        assert isinstance(client, RuntimeFactoryFactory.Client), type(client)

        handle, factory = self._create_pair(initializer)
        client._on_handle_ready(handle)
        return factory
