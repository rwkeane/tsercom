# tsercom/rpc/grpc_util/transport/pinned_server_auth_grpc_channel_factory.py
from __future__ import annotations

import asyncio
import grpc
import logging
from typing import Any, Optional  # Use modern Optional

from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory

logger = logging.getLogger(__name__)


class PinnedServerAuthGrpcChannelFactory(GrpcChannelFactory):
    """
    Creates a gRPC channel where the client authenticates the server
    by matching its certificate against an expected server certificate (pinning).
    """

    def __init__(
        self,
        expected_server_cert_pem: bytes | str,
        server_hostname_override: Optional[str] = None,
    ):
        """
        Initializes the factory with the expected server certificate.

        Args:
            expected_server_cert_pem: PEM-encoded server certificate to pin against (bytes or string).
            server_hostname_override: If provided, this hostname will be used
                                      for SSL target name override. This is crucial
                                      if the target address (e.g. IP address) doesn't match
                                      any name in the server certificate's SANs or CN,
                                      but you still want to validate the certificate content.
        """
        if isinstance(expected_server_cert_pem, str):
            self.expected_server_cert_pem: bytes = (
                expected_server_cert_pem.encode("utf-8")
            )
        else:
            self.expected_server_cert_pem: bytes = expected_server_cert_pem

        self.server_hostname_override: Optional[str] = server_hostname_override
        super().__init__()  # GrpcChannelFactory has no __init__, but good practice

    async def find_async_channel(
        self, addresses: list[str] | str, port: int
    ) -> ChannelInfo | None:
        """
        Attempts to establish a secure gRPC channel to the specified address(es)
        and port, authenticating the server by pinning its certificate.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            A `ChannelInfo` object if a channel is successfully established,
            otherwise `None`.
        """
        address_list: list[str]
        if isinstance(addresses, str):
            address_list = [addresses]
        else:
            address_list = list(addresses)  # Ensure it's a list copy

        logger.info(
            f"Attempting secure connection (Pinned Server Auth) to addresses: {address_list} on port {port}"
        )

        # For pinning, the expected server certificate itself is provided as the 'root_certificates'.
        # gRPC will then ensure that the certificate presented by the server matches this one.
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
        # Without hostname override, gRPC would also try to validate the hostname in the cert against the target address.
        # If target is an IP, this usually fails unless IP is in SANs.
        # For pinning, you might primarily care about the cert content, and override ensures hostname validation doesn't fail separately
        # if the pinned cert is correct.

        active_channel: Optional[grpc.aio.Channel] = (
            None  # To ensure closure on failure
        )

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
                await asyncio.wait_for(
                    active_channel.channel_ready(), timeout=5.0
                )

                logger.info(
                    f"Successfully connected securely to {target} (Pinned Server Auth)."
                )
                # Detach active_channel from the variable so it's not closed in a finally block if successful
                channel_to_return = active_channel
                active_channel = None
                return ChannelInfo(channel_to_return, current_address, port)

            except grpc.aio.AioRpcError as e:
                logger.warning(
                    f"Secure connection to {target} (Pinned Server Auth) failed: gRPC Error {e.code()} - {e.details()}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Secure connection to {target} (Pinned Server Auth) timed out."
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while trying to connect to {target} (Pinned Server Auth): {e}"
                )
                if isinstance(e, AssertionError):  # Re-raise assertion errors
                    raise
            finally:
                if (
                    active_channel
                ):  # If loop breaks or error occurs, close the partially opened channel
                    await active_channel.close()
                    active_channel = None  # Reset to prevent re-closing

        logger.warning(
            f"Failed to establish secure connection (Pinned Server Auth) to any of the provided addresses: {address_list} on port {port}"
        )
        return None
