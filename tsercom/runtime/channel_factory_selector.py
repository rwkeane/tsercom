"""Selects and manages gRPC channel factories based on configuration."""

import logging
from typing import Optional

from tsercom.rpc.grpc_util.channel_auth_config import (
    BaseChannelAuthConfig,
    ClientAuthChannelConfig,
    InsecureChannelConfig,
    PinnedServerChannelConfig,
    ServerCAChannelConfig,
)
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods # Factory selector pattern
class ChannelFactorySelector:
    """Selects gRPC channel factories based on ChannelAuthConfig."""

    def _read_file_content(self, file_path: str | None) -> bytes | None:
        """Reads the content of a file if the path is provided."""
        if file_path is None:
            return None
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except IOError as e:
            logger.error("Error reading file %s: %s", file_path, e)
            raise  # Re-raise the exception to be handled by the caller

    def create_factory(
        self, auth_config: Optional[BaseChannelAuthConfig]
    ) -> GrpcChannelFactory:
        """
        Creates an instance of a GrpcChannelFactory based on the provided
        ChannelAuthConfig.

        Args:
            auth_config: The channel authentication configuration object,
                         or None for an insecure channel.

        Returns:
            An instance of a GrpcChannelFactory subclass.

        Raises:
            ValueError: If a BaseChannelAuthConfig subclass is provided but
                        is not recognized, or files not accessible.
        """
        if auth_config is None or isinstance(
            auth_config, InsecureChannelConfig
        ):
            logger.info(
                "Creating GrpcChannelFactory for insecure configuration."
            )
            return InsecureGrpcChannelFactory()

        if isinstance(auth_config, ServerCAChannelConfig):
            logger.info(
                "Creating GrpcChannelFactory for Server CA configuration."
            )
            ca_cert_pem = self._read_file_content(
                auth_config.server_ca_cert_path
            )
            if not ca_cert_pem:
                # pylint: disable=consider-using-f-string
                raise ValueError(
                    "Failed to read server_ca_cert_path: %s"
                    % auth_config.server_ca_cert_path
                )
            return ServerAuthGrpcChannelFactory(
                root_ca_cert_pem=ca_cert_pem,
                server_hostname_override=auth_config.server_hostname_override,
            )

        if isinstance(auth_config, PinnedServerChannelConfig):
            logger.info(
                "Creating GrpcChannelFactory for Pinned Server configuration."
            )
            pinned_cert_pem = self._read_file_content(
                auth_config.pinned_server_cert_path
            )
            if not pinned_cert_pem:
                # pylint: disable=consider-using-f-string
                raise ValueError(
                    "Failed to read pinned_server_cert_path: %s"
                    % auth_config.pinned_server_cert_path
                )
            return PinnedServerAuthGrpcChannelFactory(
                expected_server_cert_pem=pinned_cert_pem,
                server_hostname_override=auth_config.server_hostname_override,
            )

        if isinstance(auth_config, ClientAuthChannelConfig):
            logger.info(
                "Creating GrpcChannelFactory for Client Auth configuration."
            )
            client_cert_pem = self._read_file_content(
                auth_config.client_cert_path
            )
            client_key_pem = self._read_file_content(
                auth_config.client_key_path
            )
            if not client_cert_pem or not client_key_pem:
                # This ValueError does not use string formatting, so no C0209
                raise ValueError(
                    "Failed to read client_cert_path or client_key_path "
                    "for tls_client_auth"
                )

            # root_ca_cert_pem for ServerAuthGrpcChannelFactory is None.
            return ClientAuthGrpcChannelFactory(
                client_cert_pem=client_cert_pem,
                client_key_pem=client_key_pem,
                root_ca_cert_pem=None,
                server_hostname_override=auth_config.server_hostname_override,
            )

        # This case handles unknown subclasses of BaseChannelAuthConfig
        # (implicit else after all returns from ifs)
        # pylint: disable=consider-using-f-string
        raise ValueError(
            "Unknown or unsupported ChannelAuthConfig type: %s"
            % type(auth_config)
        )
