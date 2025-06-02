"""Defines Stopable ABC, an interface for objects that can be stopped."""

from abc import ABC, abstractmethod


# pylint: disable=R0903 # Abstract interface for stopable components
class Stopable(ABC):
    """Represents an object that has a defined stopping mechanism.

    This abstract base class provides a common interface for components
    or services that need to be explicitly stopped to release resources
    or terminate operations.
    """

    @abstractmethod
    async def stop(self) -> None:
        """Asynchronously stops the object.

        Subclasses must implement this method to define the specific actions
        required to stop the object's operation and clean up resources.
        """
