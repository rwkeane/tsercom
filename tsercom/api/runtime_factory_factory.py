"""Defines the abstract base class for runtime factory creators."""

from abc import ABC, abstractmethod
from typing import TypeVar, Tuple, Generic

from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer

# Type variable for data, typically bound by some base data class.
TDataType = TypeVar("TDataType")
# Type variable for events.
TEventType = TypeVar("TEventType")


class RuntimeFactoryFactory(ABC, Generic[TDataType, TEventType]):
    """Abstract base class for factories that create RuntimeFactory instances.

    This class provides a common interface for creating different types of
    runtime factories (e.g., local, remote) and managing their associated handles.
    It uses a client callback mechanism to notify when a handle is ready.
    """

    class Client(Generic[TDataType, TEventType]):
        """Interface for clients of RuntimeFactoryFactory.

        Clients implement this interface to receive notifications when a
        RuntimeHandle is ready after a factory creation.
        """
        @abstractmethod
        def _on_handle_ready(
            self, handle: RuntimeHandle[TDataType, TEventType]
        ) -> None:
            """Callback invoked when a RuntimeHandle has been successfully created.

            Args:
                handle: The newly created RuntimeHandle.
            """
            pass

    def __init__(self) -> None:
        """Initializes the RuntimeFactoryFactory."""
        # The ABCMeta.__init__ is called implicitly.
        super().__init__()

    @abstractmethod
    def _create_pair(
        self, initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> Tuple[
        RuntimeHandle[TDataType, TEventType],
        RuntimeFactory[TDataType, TEventType],
    ]:
        """Abstract method to create a RuntimeHandle and its corresponding RuntimeFactory.

        Subclasses must implement this method to provide the specific logic for
        instantiating a handle and a factory based on the provided initializer.

        Args:
            initializer: The RuntimeInitializer containing configuration for the runtime.

        Returns:
            A tuple containing the created RuntimeHandle and the RuntimeFactory.
        """
        pass

    def create_factory(
        self,
        client: "RuntimeFactoryFactory.Client[TDataType, TEventType]",
        initializer: RuntimeInitializer[TDataType, TEventType],
    ) -> RuntimeFactory[TDataType, TEventType]:
        """Creates a RuntimeFactory and notifies the client when its handle is ready.

        This method orchestrates the creation of a runtime handle and factory pair
        using the `_create_pair` method, then informs the client via the
        `_on_handle_ready` callback.

        Args:
            client: The client instance that will be notified when the handle is ready.
            initializer: The RuntimeInitializer to configure the new factory and handle.

        Returns:
            The created RuntimeFactory instance.

        Raises:
            ValueError: If the client argument is None.
            TypeError: If the client is not an instance of RuntimeFactoryFactory.Client.
        """
        # Ensure the client is valid before proceeding.
        if client is None:
            raise ValueError("Client argument cannot be None for create_factory.")
        if not isinstance(client, RuntimeFactoryFactory.Client):
            raise TypeError(f"Client must be an instance of RuntimeFactoryFactory.Client, got {type(client).__name__}.")

        handle, factory = self._create_pair(initializer)
        
        client._on_handle_ready(handle)
        
        return factory
