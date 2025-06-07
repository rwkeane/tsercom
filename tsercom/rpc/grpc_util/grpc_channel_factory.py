"""Abstract base class defining the interface for gRPC channel factories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import grpc

from tsercom.util.connection_factory import ConnectionFactory


class GrpcChannelFactory(ConnectionFactory[grpc.Channel], ABC):
    """
    Abstract factory for creating gRPC channels.
    It implements the `ConnectionFactory` interface for `grpc.Channel` type.
    Concrete implementations are responsible for the actual channel creation logic.
    """

    @abstractmethod
    async def find_async_channel(
        self,
        addresses: Union[List[str], str],
        port: int,
    ) -> Optional[grpc.Channel]:
        """Finds an asynchronous gRPC channel to the specified address(es) and port.

        Implementations should attempt to establish a connection and return
        the gRPC channel if successful.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            A `grpc.Channel` object if a channel is successfully established,
            otherwise `None`.
        """
        pass

    async def connect(
        self, addresses: Union[List[str], str], port: int
    ) -> Optional[grpc.Channel]:
        """Establishes a gRPC channel to the specified address(es) and port.

        This method implements the `ConnectionFactory.connect` interface by
        delegating to the `find_async_channel` method.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            An instance of `grpc.Channel` if connection is successful,
            otherwise `None`.
        """
        return await self.find_async_channel(addresses, port)
