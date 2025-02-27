from typing import Callable
import grpc

from tsercom.rpc.connection.channel_info import ChannelInfo
from tsercom.util.ip import get_all_address_strings


class GrpcChannelFactory:
    """
    This class is responsible for finding channels to use for a gRPC Stub
    definition, by testing against various |addresses| and a given |port|.
    """
    def __init__(self):
        pass
        
    async def find_async_channel(
            self, addresses :  list[str] | str, port : int) -> ChannelInfo:
        """
        Finds an asyncronous channel, for asynchronous stub use.
        """
        print("FINDING CHANNEL")
        return self.__find_channel_impl(
                grpc.aio.insecure_channel, addresses, port)

    def find_channel(self,
                     addresses :  list[str] | str,
                     port : int) -> ChannelInfo:
        """
        Finds a synchronous channel, for synchonous stub use.
        """
        return self.__find_channel_impl(grpc.insecure_channel, addresses, port)
        
    def __find_channel_impl(
            self,
            channel_factory : Callable[[str], grpc.Channel],
            addresses :  list[str] | str,
            port : int) -> ChannelInfo:
        # Parse the address
        print("Started!")
        if addresses is None:
            addresses = get_all_address_strings()
        elif isinstance(addresses, str):
            addresses = [ addresses ]
        else:
            addresses = list(addresses)
        print("Connecting to", addresses)

        # Get local addresses exposed to network.
        my_addresses = get_all_address_strings()

        # Connect.
        print(f"Connecting to gRPC ({len(addresses)}",
              f"IPs on port {port} for my {len(my_addresses)} addresses)....")
        address: str|None = None
        for address in addresses:
            for my_address in my_addresses:
                try:
                    channel = channel_factory(f'{my_address}:{port}')
                    return ChannelInfo(channel, address, port)
                
                except Exception as e:
                    print(f"\t\tAddress Unreachable! Error '{e}'")
                    if isinstance(e, AssertionError):
                        raise e

        return None