from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Union

import grpc

# ChannelInfo is no longer returned by find_async_channel
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory

logger = logging.getLogger(__name__)


class ServerAuthGrpcChannelFactory(GrpcChannelFactory):
    """
    Creates a gRPC channel where the client authenticates the server
    using a provided root CA certificate.
    """

    def __init__(
        self,
        root_ca_cert_pem: bytes | str,
        server_hostname_override: Optional[str] = None,
    ):
        """
        Initializes the factory with the root CA certificate.

        Args:
            root_ca_cert_pem: PEM-encoded root CA certificate (bytes or string).
            server_hostname_override: If provided, this hostname will be used
                                      for SSL target name override, which is
                                      useful if the server's certificate CN
                                      does not match the target address (e.g., for IP addresses
                                      or localhost testing).
        """
        self.root_ca_cert_pem: bytes
        if isinstance(root_ca_cert_pem, str):
            self.root_ca_cert_pem = root_ca_cert_pem.encode("utf-8")
        else:
            self.root_ca_cert_pem = root_ca_cert_pem

        self.server_hostname_override: Optional[str] = server_hostname_override
        super().__init__()

    async def find_async_channel(
        self, addresses: Union[List[str], str], port: int
    ) -> Optional[grpc.Channel]:
        """
        Attempts to establish a secure gRPC channel to the specified address(es)
        and port, authenticating the server using the root CA.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            A `grpc.Channel` object if a channel is successfully established,
            otherwise `None`.
        """
        address_list: List[str]
        if isinstance(addresses, str):
            address_list = [addresses]
        else:
            address_list = list(addresses)

        logger.info(
            f"Attempting secure connection (Server Auth) to addresses: {address_list} on port {port}"
        )

        credentials = grpc.ssl_channel_credentials(
            root_certificates=self.root_ca_cert_pem
        )

        options: list[tuple[str, Any]] = []
        if self.server_hostname_override:
            options.append(
                (
                    "grpc.ssl_target_name_override",
                    self.server_hostname_override,
                )
            )

        channel: Optional[grpc.aio.Channel] = None
        for current_address in address_list:
            target = f"{current_address}:{port}"
            try:
                logger.info(
                    f"Attempting secure connection to {target} with server auth."
                )
                # Create a secure channel
                channel = grpc.aio.secure_channel(
                    target, credentials, options=options if options else None
                )

                # Wait for the channel to be ready, with a timeout.
                # Consider making the timeout configurable or part of class constants.
                await asyncio.wait_for(channel.channel_ready(), timeout=5.0)

                logger.info(
                    f"Successfully connected securely to {target} (Server Auth)."
                )
                return channel

            except grpc.aio.AioRpcError as e:
                # This specifically catches gRPC errors, e.g., connection failure, handshake failure
                logger.warning(
                    f"Secure connection to {target} (Server Auth) failed: gRPC Error {e.code()} - {e.details()}"
                )
                if channel:
                    await channel.close()
            except asyncio.TimeoutError:
                logger.warning(
                    f"Secure connection to {target} (Server Auth) timed out."
                )
                if channel:
                    await channel.close()
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while trying to connect to {target} (Server Auth): {e}"
                )
                if channel:  # Ensure channel is closed
                    await channel.close()
                if isinstance(e, AssertionError):
                    raise

        logger.warning(
            f"Failed to establish secure connection (Server Auth) to any of the provided addresses: {address_list} on port {port}"
        )
        return None
