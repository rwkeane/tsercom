"""Abstract base class defining the interface for gRPC channel factories."""

from __future__ import annotations  # Added this line
from abc import ABC, abstractmethod
import typing  # Ensure typing is imported

if typing.TYPE_CHECKING:
    from tsercom.rpc.common.channel_info import (
        ChannelInfo,
    )  # Moved under TYPE_CHECKING

# typing.Optional is removed as per task to use 'ChannelInfo' | None


class GrpcChannelFactory(ABC):
    """
    This class is responsible for finding channels to use for a gRPC Stub
    definition, by testing against various |addresses| and a given |port|.
    """

    @abstractmethod
    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> ChannelInfo | None:  # Changed to 'ChannelInfo' | None
        """Finds an asynchronous gRPC channel to the specified address(es) and port.

        Implementations should attempt to establish a connection and return
        information about the channel if successful.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            A `ChannelInfo` object if a channel is successfully established,
            otherwise `None`.
        """
        pass
