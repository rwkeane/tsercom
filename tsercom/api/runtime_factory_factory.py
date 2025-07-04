"""Defines the abstract base class for runtime factory creators."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.data.exposed_data import ExposedData
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class RuntimeFactoryFactory(ABC, Generic[DataTypeT, EventTypeT]):
    """Abstract base class for factories that create RuntimeFactory instances.

    This class provides a common interface for creating different types of
    runtime factories (local, remote) and managing their associated handles.
    It uses a client callback mechanism to notify when a handle is ready.
    """

    class Client(ABC):
        """Interface for clients of `RuntimeFactoryFactory`.

        Clients implement this interface to receive notifications when a
        `RuntimeHandle` is ready after a factory creation process.
        """

        @abstractmethod
        def _on_handle_ready(
            self, handle: RuntimeHandle[DataTypeT, EventTypeT]
        ) -> None:
            """Invoke callback when a RuntimeHandle has been successfully created.

            Args:
                handle: The newly created RuntimeHandle.

            """

    def __init__(self) -> None:
        """Initialize the RuntimeFactoryFactory."""

    @abstractmethod
    def _create_pair(
        self, initializer: RuntimeInitializer[DataTypeT, EventTypeT]
    ) -> tuple[
        RuntimeHandle[DataTypeT, EventTypeT],
        RuntimeFactory[DataTypeT, EventTypeT],
    ]:
        """Create a RuntimeHandle and its corresponding RuntimeFactory.

        Subclasses must implement this method. It should provide the
        specific logic for instantiating a handle and a factory
        based on the provided `initializer`.

        Args:
            initializer: Configures the runtime.

        Returns:
            A tuple: (created RuntimeHandle, RuntimeFactory).

        """

    def create_factory(
        self,
        client: "RuntimeFactoryFactory.Client",
        initializer: RuntimeInitializer[DataTypeT, EventTypeT],
    ) -> RuntimeFactory[DataTypeT, EventTypeT]:
        """Create a RuntimeFactory and notify the client when its handle is ready.

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
        if client is None:
            raise ValueError("Client argument cannot be None for create_factory.")
        if not isinstance(client, RuntimeFactoryFactory.Client):
            # Long but readable error message string
            raise TypeError(
                f"Client must be an instance of RuntimeFactoryFactory.Client, "
                f"got {type(client).__name__}."
            )

        handle, factory = self._create_pair(initializer)

        client._on_handle_ready(handle)

        return factory
