"""Provides ChannelFactorySelector for obtaining gRPC channel factory instances.

This module now offers a selector that can create various types of gRPC channel
factories based on a GrpcChannelFactoryConfig.
"""

import os
import logging
from typing import Optional

from tsercom.config.grpc_channel_config import GrpcChannelFactoryConfig
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)

logger = logging.getLogger(__name__)


class ChannelFactorySelector:
    """Selects and provides instances of gRPC channel factories based on configuration."""

    def _load_credential(
        self, credential_pem_or_path: Optional[str]
    ) -> Optional[str | bytes]:
        """
        Loads credential. If it's an existing file path, reads content as bytes.
        Otherwise, returns the string as is (assuming it's direct PEM content).
        Returns None if input is None.
        """
        if credential_pem_or_path is None:
            return None

        is_likely_pem_content = (
            "\n" in credential_pem_or_path
            or "-----BEGIN" in credential_pem_or_path
        )

        if not is_likely_pem_content and os.path.exists(
            credential_pem_or_path
        ):
            logger.info(
                f"Loading credential from path: {credential_pem_or_path}"
            )
            try:
                with open(credential_pem_or_path, "rb") as f:
                    return f.read()
            except IOError as e:
                logger.error(
                    f"Failed to read credential file {credential_pem_or_path}: {e}"
                )
                raise ValueError(
                    f"Could not read credential file: {credential_pem_or_path}"
                ) from e
        else:
            # Assumed to be direct PEM string content (or was a non-existent path)
            return credential_pem_or_path  # Return as string, factory handles .encode() or bytes

    def create_factory_from_config(
        self, config: Optional[GrpcChannelFactoryConfig]
    ) -> GrpcChannelFactory:
        """
        Creates a GrpcChannelFactory based on the provided configuration.

        Args:
            config: The GrpcChannelFactoryConfig object. If None, defaults to
                    an InsecureGrpcChannelFactory.

        Returns:
            An instance of a GrpcChannelFactory.

        Raises:
            ValueError: If the configuration is invalid or incomplete for the
                        specified factory type.
        """
        if not config:
            logger.info(
                "No GrpcChannelFactoryConfig provided, defaulting to InsecureGrpcChannelFactory."
            )
            return InsecureGrpcChannelFactory()

        factory_type = config.factory_type
        server_hostname_override = config.server_hostname_override

        if factory_type == "insecure":
            return InsecureGrpcChannelFactory()

        elif factory_type == "client_auth":
            client_cert = self._load_credential(config.client_cert_pem_or_path)
            client_key = self._load_credential(config.client_key_pem_or_path)
            root_ca_cert = self._load_credential(
                config.root_ca_cert_pem_or_path
            )

            if not client_cert or not client_key:
                raise ValueError(
                    "ClientAuth factory requires client_cert_pem_or_path and client_key_pem_or_path."
                )

            return ClientAuthGrpcChannelFactory(
                client_cert_pem=client_cert,
                client_key_pem=client_key,
                root_ca_cert_pem=root_ca_cert,
                server_hostname_override=server_hostname_override,
            )

        elif factory_type == "pinned_server_auth":
            expected_server_cert = self._load_credential(
                config.expected_server_cert_pem_or_path
            )

            if not expected_server_cert:
                raise ValueError(
                    "PinnedServerAuth factory requires expected_server_cert_pem_or_path."
                )

            return PinnedServerAuthGrpcChannelFactory(
                expected_server_cert_pem=expected_server_cert,
                server_hostname_override=server_hostname_override,
            )

        elif factory_type == "server_auth":
            root_ca_cert = self._load_credential(
                config.root_ca_cert_pem_or_path
            )

            if not root_ca_cert:
                raise ValueError(
                    "ServerAuth factory requires root_ca_cert_pem_or_path."
                )

            return ServerAuthGrpcChannelFactory(
                root_ca_cert_pem=root_ca_cert,
                server_hostname_override=server_hostname_override,
            )

        else:
            # Should be caught by Literal type hinting, but good for runtime safety
            raise ValueError(f"Unknown GrpcChannelFactoryType: {factory_type}")
