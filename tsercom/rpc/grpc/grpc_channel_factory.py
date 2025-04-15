from abc import ABC, abstractmethod

from tsercom.rpc.connection.channel_info import ChannelInfo


class GrpcChannelFactory(ABC):
    """
    This class is responsible for finding channels to use for a gRPC Stub
    definition, by testing against various |addresses| and a given |port|.
    """

    @abstractmethod
    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> ChannelInfo | None:
        """
        Finds an asyncronous channel, for asynchronous stub use. Returns the
        channel on success and None on failure.
        """
        pass
