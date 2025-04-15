import asyncio
import grpc

from tsercom.rpc.connection.channel_info import ChannelInfo
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory


class InsecureGrpcChannelFactory(GrpcChannelFactory):
    """
    This class is responsible for finding channels to use for a gRPC Stub
    definition, by testing against various |addresses| and a given |port|.
    """

    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> ChannelInfo | None:
        """
        Finds an asyncronous channel, for asynchronous stub use.
        """
        # Parse the address
        assert addresses is not None
        if isinstance(addresses, str):
            addresses = [addresses]
        else:
            addresses = list(addresses)
        print("Connecting to", addresses)

        # Connect.
        print(
            f"Connecting to gRPC ({len(addresses)}",
        )
        address: str | None = None
        for address in addresses:
            try:
                channel = grpc.aio.insecure_channel(f"{address}:{port}")
                await asyncio.wait_for(channel.channel_ready(), timeout=5)
                return ChannelInfo(channel, address, port)

            except Exception as e:
                print(f"\t\tAddress Unreachable! Error '{e}'")
                if isinstance(e, AssertionError):
                    raise e

        return None
