"""Defines the Runtime abstract base class, representing a service that can be started and stopped."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.util.stopable import Stopable

TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class Runtime(Stopable, ABC, Generic[TDataType, TEventType]):
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
        pass
