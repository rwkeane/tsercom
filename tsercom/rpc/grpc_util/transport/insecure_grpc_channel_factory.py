"""Provides InsecureGrpcChannelFactory for creating insecure gRPC channels."""

import asyncio
import logging

import grpc

from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory


class InsecureGrpcChannelFactory(GrpcChannelFactory):
    """Creates insecure gRPC channels to specified addresses and port.

    This factory attempts to connect to a list of addresses sequentially
    and returns a ChannelInfo object for the first successful connection.
    It does not use any transport security (e.g., TLS).
    """

    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> grpc.Channel | None:
        """Attempt to establish an insecure gRPC channel.

        Iterate through the provided addresses, attempting to connect to each
        at the specified port. Return a `ChannelInfo` object for the first
        successful connection. Log connection attempts and errors.

        Args:
            addresses: A list of IP addresses or a single IP address string.
            port: The target port number.

        Returns:
            A `grpc.Channel` instance if a connection is successful,
            otherwise `None`.

        """
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
        logging.info(f"Connecting to gRPC (trying {len(address_list)} address(es))...")

        for current_address in address_list:
            try:
                logging.info(f"Attempting connection to {current_address}:{port}")
                channel = grpc.aio.insecure_channel(f"{current_address}:{port}")
                # Wait for the channel to be ready, with a timeout.
                await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
                logging.info(f"Successfully connected to {current_address}:{port}")
                return channel  # Return the channel directly

            except Exception as e:
                logging.warning(
                    f"Address {current_address}:{port} unreachable. Error: {e}"
                )
                if isinstance(e, AssertionError):
                    raise
                # For other exceptions, continue to the next address.

        logging.warning(
            f"Failed to connect to any of the provided addresses: {address_list} "
            f"on port {port}"
        )
        return None
