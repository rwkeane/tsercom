"""Defines Runtime ABC, a service that can be started and stopped."""

from abc import ABC, abstractmethod
from typing import Optional


class Runtime(ABC):
    """Abstract base class for a runtime service.

    A Runtime represents a service that has a lifecycle involving starting
    asynchronously and can be stopped (as defined by the `Stopable` interface).
    """

    @abstractmethod
    async def start_async(self) -> None:
        """Starts the runtime service asynchronously.

        Subclasses must implement this method to define the startup logic
        for the service.
        """

    @abstractmethod
    async def stop(self, exception: Optional[Exception] = None) -> None:
        """Asynchronously stops the object.

        Subclasses must implement this method to define how the runtime should
        stop. |exception| is the exception, if any, that caused the runtime to
        stop.
        """
