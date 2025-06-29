"""Factory for gRPC channels with server certificate pinning."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import grpc

from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory

logger = logging.getLogger(__name__)


class PinnedServerAuthGrpcChannelFactory(GrpcChannelFactory):
    """Client authenticates server by pinning against a specific server certificate."""

    def __init__(
        self,
        expected_server_cert_pem: bytes | str,
        server_hostname_override: str | None = None,
    ):
        """Initialize the factory with the expected server certificate.

        Args:
            expected_server_cert_pem: PEM-encoded server certificate to pin
                                      against (bytes or string).
            server_hostname_override: If provided, this hostname will be used
                                      for SSL target name override. This is
                                      crucial if the target address (e.g. IP
                                      address) doesn't match any name in the
                                      server certificate's SANs or CN, but you
                                      still want to validate the certificate
                                      content.

        """
        self.expected_server_cert_pem: bytes
        if isinstance(expected_server_cert_pem, str):
            self.expected_server_cert_pem = expected_server_cert_pem.encode("utf-8")
        else:
            self.expected_server_cert_pem = expected_server_cert_pem

        self.server_hostname_override: str | None = server_hostname_override
        super().__init__()

    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> grpc.Channel | None:
        """Establish a secure gRPC channel to the specified address(es) and port.

        Authenticates the server by pinning its certificate.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            A `grpc.Channel` object if a channel is successfully established,
            otherwise `None`.

        """
        address_list: list[str]
        if isinstance(addresses, str):
            address_list = [addresses]
        else:
            address_list = list(addresses)

        logger.info(
            f"Attempting secure connection (Pinned Server Auth) to addresses: "
            f"{address_list} on port {port}"
        )

        # For pinning, the expected server certificate itself is provided as
        # the 'root_certificates'. gRPC will then ensure that the certificate
        # presented by the server matches this one.
        credentials = grpc.ssl_channel_credentials(
            root_certificates=self.expected_server_cert_pem
        )

        options: list[tuple[str, Any]] = []
        if self.server_hostname_override:
            options.append(
                (
                    "grpc.ssl_target_name_override",
                    self.server_hostname_override,
                )
            )
        # Without hostname override, gRPC would also try to validate the
        # hostname in the cert against the target address. If target is an IP,
        # this usually fails unless IP is in SANs. For pinning, you might
        # primarily care about the cert content, and override ensures hostname
        # validation doesn't fail separately if the pinned cert is correct.

        active_channel: grpc.aio.Channel | None = None

        for current_address in address_list:
            target = f"{current_address}:{port}"
            try:
                logger.info(
                    f"Attempting secure connection to {target} (Pinned Server Auth)."
                )
                active_channel = grpc.aio.secure_channel(
                    target, credentials, options=options if options else None
                )

                # Wait for the channel to be ready, with a timeout.
                # Consider making the timeout configurable or part of class constants.
                await asyncio.wait_for(active_channel.channel_ready(), timeout=5.0)

                logger.info(
                    f"Successfully connected securely to {target} (Pinned Server Auth)."
                )
                # Detach active_channel from the variable so it's not closed
                # in a finally block if successful
                channel_to_return = active_channel
                active_channel = None
                return channel_to_return

            except grpc.aio.AioRpcError as e:
                logger.warning(
                    f"Secure connection to {target} (Pinned Server Auth) failed: "
                    f"gRPC Error {e.code()} - {e.details()}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Secure connection to {target} (Pinned Server Auth) timed out."
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while trying to connect to "
                    f"{target} (Pinned Server Auth): {e}"
                )
                if isinstance(e, AssertionError):
                    raise
            finally:
                if active_channel:
                    await active_channel.close()
                    active_channel = None

        logger.warning(
            f"Failed to establish secure connection (Pinned Server Auth) to any of "
            f"the provided addresses: {address_list} on port {port}"
        )
        return None
