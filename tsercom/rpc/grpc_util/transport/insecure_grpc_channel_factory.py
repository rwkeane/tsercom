"""Provides InsecureGrpcChannelFactory for creating insecure gRPC channels."""

import asyncio
import grpc # Keep for potential grpc.aio.AioRpcError, though not explicitly used in new code
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
            try:
                target = f"{current_address}:{port}"
                logging.info(
                    f"Attempting connection to {target}"
                )
                # Use the credentials provider to create the insecure channel
                channel = self._credentials_provider.create_insecure_channel(
                    target=target
                )
                if channel:
                    # Wait for the channel to be ready, with a timeout.
                    await asyncio.wait_for(
                        channel.channel_ready(), timeout=5.0
                    )
                    logging.info(
                        f"Successfully connected to {target}"
                    )
                    return ChannelInfo(channel, current_address, port)
                else:
                    logging.warning(f"Channel creation returned None for {target}")
                    # Continue to next address if channel is None

            except grpc.aio.AioRpcError as e: # More specific error handling
                logging.warning(
                    f"Address {target} unreachable. gRPC AioRpcError: {e.code()} - {e.details()}"
                )
            except asyncio.TimeoutError:
                logging.warning(
                    f"Timeout waiting for channel to be ready for {target}."
                )
            except Exception as e:
                logging.warning(
                    f"Address {target} unreachable. Error: {e}"
                )
                if isinstance(e, AssertionError):  # Re-raise assertion errors
                    raise
                # For other exceptions, continue to the next address.

        logging.warning(
            f"Failed to connect to any of the provided addresses: {address_list} on port {port}"
        )
        return None
