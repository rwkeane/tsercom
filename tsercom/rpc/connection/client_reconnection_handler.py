"""Defines the interface for client reconnection management."""

from abc import ABC, abstractmethod
from typing import Optional


class ClientReconnectionManager(ABC):
    """Abstract base class for managers that handle client reconnections.

    This class defines an interface for components that need to react to
    disconnection events, potentially with custom logic for attempting
    to reconnect or clean up resources.
    """

    @abstractmethod
    async def _on_disconnect(self, error: Optional[Exception] = None) -> None:
        """Callback method invoked when a client disconnection occurs.

        Subclasses should implement this method to define their specific
        behavior in response to a disconnection.

        Args:
            error: The exception that caused or accompanied the disconnection,
                   if any. Can be None if the disconnection was clean or the
                   cause is unknown.
        """
        pass
