"""Defines an abstract factory for creating connections."""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar, Union

ConnectionTypeT = TypeVar("ConnectionTypeT")


# pylint: disable=R0903 # Abstract factory interface
class ConnectionFactory(Generic[ConnectionTypeT], ABC):
    """An abstract factory for creating connections of a specific type."""

    @abstractmethod
    async def connect(
        self, addresses: Union[List[str], str], port: int
    ) -> Optional[ConnectionTypeT]:
        """Establishes a connection to a service.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            An instance of ConnectionTypeT if successful, otherwise None.
        """
