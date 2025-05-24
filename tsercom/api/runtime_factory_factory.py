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
        if client is None:
            raise ValueError("Client argument cannot be None for create_factory.")
        if not isinstance(client, RuntimeFactoryFactory.Client):
            raise TypeError(f"Client must be an instance of RuntimeFactoryFactory.Client, got {type(client).__name__}.")

        handle, factory = self._create_pair(initializer)
        client._on_handle_ready(handle)
        return factory
