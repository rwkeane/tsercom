"""Defines InitializationPair, a utility class for managing asynchronous initialization of tsercom runtimes."""

from concurrent.futures import Future
from typing import Generic, TypeVar
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer

# Type variables for generic typing
TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class InitializationPair(Generic[TDataType, TEventType]):
    """A container holding a future for a RuntimeHandle and its initializer.

    This class pairs a `RuntimeInitializer` with the `Future` that will eventually
    hold the `RuntimeHandle` created by that initializer. This is useful for
    managing asynchronous initialization of runtimes.
    """

    def __init__(
        self,
        handle_future: Future[RuntimeHandle[TDataType, TEventType]],
        initializer: RuntimeInitializer[TDataType, TEventType],
    ) -> None:
        """Initializes an InitializationPair instance.

        Args:
            handle_future: A Future that will resolve to a RuntimeHandle.
            initializer: The RuntimeInitializer used to create the RuntimeHandle.
        """
        self.__handle_future: Future[RuntimeHandle[TDataType, TEventType]] = (
            handle_future
        )
        self.__initializer: RuntimeInitializer[TDataType, TEventType] = (
            initializer
        )

    @property
    def handle_future(self) -> Future[RuntimeHandle[TDataType, TEventType]]:
        """Gets the Future for the RuntimeHandle.

        Returns:
            The Future object that will contain the RuntimeHandle once
            initialization is complete.
        """
        return self.__handle_future

    @property
    def initializer(self) -> RuntimeInitializer[TDataType, TEventType]:
        """Gets the RuntimeInitializer.

        Returns:
            The RuntimeInitializer instance associated with this pair.
        """
        return self.__initializer
