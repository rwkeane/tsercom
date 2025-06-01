from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Optional

TConnectionType = TypeVar("TConnectionType")


class ConnectionFactory(Generic[TConnectionType], ABC):
    """An abstract factory for creating connections of a specific type."""

    @abstractmethod
    async def connect(
        self, addresses: Union[List[str], str], port: int
    ) -> Optional[TConnectionType]:
        """Establishes a connection to a service.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            An instance of TConnectionType if connection is successful,
            otherwise None.
        """
        pass
