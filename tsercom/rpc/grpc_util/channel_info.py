import dataclasses

"""Provides ChannelInfo dataclass for gRPC channel details and health check."""
import grpc

from tsercom.rpc.grpc_util.grpc_service_publisher import check_grpc_channel_health


@dataclasses.dataclass
class ChannelInfo:
    """Encapsulates information about a gRPC channel and its connection endpoint."""

    channel: grpc.aio.Channel
    address: str
    port: int

    async def is_healthy(self) -> bool:
        """Checks if the underlying gRPC channel is healthy.

        Returns:
            True if the channel is healthy (serving), False otherwise.

        """
        return await check_grpc_channel_health(self.channel)
