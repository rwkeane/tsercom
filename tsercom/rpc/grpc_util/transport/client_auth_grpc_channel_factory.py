from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    List,
    Optional,
    Union,
)

import grpc

from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory

logger = logging.getLogger(__name__)


class ClientAuthGrpcChannelFactory(GrpcChannelFactory):
    """
    Creates a gRPC channel where the client uses a client certificate and key.
    Optionally, it can also validate the server's certificate using a root CA.
    """

    def __init__(
        self,
        client_cert_pem: bytes | str,
        client_key_pem: bytes | str,
        root_ca_cert_pem: Optional[bytes | str] = None,
        server_hostname_override: Optional[str] = None,
    ):
        """
        Initializes the factory with client credentials and optional CA for server validation.

        Args:
            client_cert_pem: PEM-encoded client certificate (bytes or string).
            client_key_pem: PEM-encoded client private key (bytes or string).
            root_ca_cert_pem: Optional PEM-encoded root CA certificate for server validation.
                              If None, server certificate is not validated by the client.
            server_hostname_override: If provided, this hostname will be used
                                      for SSL target name override.
        """
        self.client_cert_pem_bytes: bytes
        if isinstance(client_cert_pem, str):
            self.client_cert_pem_bytes = client_cert_pem.encode("utf-8")
        else:
            self.client_cert_pem_bytes = client_cert_pem

        self.client_key_pem_bytes: bytes
        if isinstance(client_key_pem, str):
            self.client_key_pem_bytes = client_key_pem.encode("utf-8")
        else:
            self.client_key_pem_bytes = client_key_pem

        self.root_ca_cert_pem_bytes: Optional[bytes]  # Declare type once
        if root_ca_cert_pem:
            if isinstance(root_ca_cert_pem, str):
                self.root_ca_cert_pem_bytes = root_ca_cert_pem.encode("utf-8")
            else:
                self.root_ca_cert_pem_bytes = root_ca_cert_pem
        else:
            self.root_ca_cert_pem_bytes = None

        self.server_hostname_override: Optional[str] = server_hostname_override
        super().__init__()

    async def find_async_channel(
        self, addresses: Union[List[str], str], port: int
    ) -> Optional[grpc.Channel]:
        """
        Attempts to establish a secure gRPC channel using client credentials.

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

        auth_type = "Client Auth"
        if self.root_ca_cert_pem_bytes:
            auth_type += " with Server Validation"
        else:
            auth_type += " (No Server Validation by Client)"

        logger.info(
            f"Attempting secure connection ({auth_type}) to addresses: {address_list} on port {port}"
        )

        credentials = grpc.ssl_channel_credentials(
            certificate_chain=self.client_cert_pem_bytes,
            private_key=self.client_key_pem_bytes,
            root_certificates=self.root_ca_cert_pem_bytes,
        )

        options: list[tuple[str, Any]] = []
        if self.server_hostname_override:
            options.append(
                (
                    "grpc.ssl_target_name_override",
                    self.server_hostname_override,
                )
            )

        active_channel: Optional[grpc.aio.Channel] = None

        for current_address in address_list:
            target = f"{current_address}:{port}"
            try:
                logger.info(
                    f"Attempting secure connection to {target} ({auth_type})."
                )
                active_channel = grpc.aio.secure_channel(
                    target, credentials, options=options if options else None
                )

                # Wait for the channel to be ready, with a timeout.
                await asyncio.wait_for(
                    active_channel.channel_ready(), timeout=5.0
                )

                logger.info(
                    f"Successfully connected securely to {target} ({auth_type})."
                )
                channel_to_return = active_channel
                active_channel = (
                    None  # Detach from variable so it's not closed in finally
                )
                return channel_to_return  # Return the channel directly

            except grpc.aio.AioRpcError as e:
                logger.warning(
                    f"Secure connection to {target} ({auth_type}) failed: gRPC Error {e.code()} - {e.details()}"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Secure connection to {target} ({auth_type}) timed out."
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while trying to connect to {target} ({auth_type}): {e}"
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
            f"Failed to establish secure connection ({auth_type}) to any of the provided addresses: {address_list} on port {port}"
        )
        return None
