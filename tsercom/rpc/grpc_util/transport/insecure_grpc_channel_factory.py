"""Provides InsecureGrpcChannelFactory for creating insecure gRPC channels."""

import asyncio
import grpc # Keep for potential grpc.aio.AioRpcError, though not explicitly used in new code
import grpc.aio # Added for type hinting grpc.aio.Channel
import logging

from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from ..grpc_channel_credentials_provider import GrpcChannelCredentialsProvider


class InsecureGrpcChannelFactory(GrpcChannelFactory):
    """Creates insecure gRPC channels to specified addresses and port.

    This factory attempts to connect to a list of addresses sequentially
    and returns a ChannelInfo object for the first successful connection.
    It does not use any transport security (e.g., TLS).
    """

    def __init__(self, credentials_provider: GrpcChannelCredentialsProvider):
        """Initializes InsecureGrpcChannelFactory.

        Args:
            credentials_provider: The provider for gRPC channel credentials.
        """
        super().__init__()
        self._credentials_provider = credentials_provider

    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> ChannelInfo | None:
        """Attempts to establish an insecure gRPC channel.

        Iterates through the provided addresses, attempting to connect to each
        at the specified port. Returns a `ChannelInfo` object for the first
        successful connection. Logs connection attempts and errors.

        Args:
            addresses: A list of IP addresses or a single IP address string.
            port: The target port number.

        Returns:
            A `ChannelInfo` instance if a connection is successful,
            otherwise `None`.
        """
        # Parse the address
        assert addresses is not None
        address_list: list[str]
        if isinstance(addresses, str):
            address_list = [addresses]
        else:
            address_list = list(addresses)

        logging.info(
            f"Attempting to connect to addresses: {address_list} on port {port}"
        )

        # Connect.
        logging.info(
            f"Connecting to gRPC (trying {len(address_list)} address(es))..."
        )

        for current_address in address_list:
            target = f"{current_address}:{port}" 
            # Initialize channel to None before the try block, ensure it has type hint e.g. grpc.aio.Channel | None
            channel: grpc.aio.Channel | None = None 
            try:
                logging.info(f"Attempting connection to {target}") 
                channel = self._credentials_provider.create_insecure_channel(target=target)

                if not channel:
                    logging.warning(f"Channel creation returned None for {target}") 
                    continue # Try next address

                await asyncio.wait_for(channel.channel_ready(), timeout=5.0) 
                
                logging.info(f"Successfully connected to {target}") 
                return ChannelInfo(channel, current_address, port)

            except grpc.aio.AioRpcError as e:
                logging.warning(
                    f"Address {target} unreachable. gRPC AioRpcError: {e.code()} - {e.details()}" 
                )
                if channel:
                    await channel.close() 
            except asyncio.TimeoutError:
                logging.warning(
                    f"Timeout waiting for channel to be ready for {target}." 
                )
                if channel:
                    await channel.close() 
            except Exception as e:
                logging.warning(
                    f"Address {target} unreachable. Error: {e}" 
                )
                if channel:
                    await channel.close() 
                if isinstance(e, AssertionError):
                    raise
        
        logging.warning(
            f"Failed to connect to any of the provided addresses: {address_list} on port {port}"
        )
        return None
