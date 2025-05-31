# tsercom/runtime/channel_factory_selector.py
import logging
from tsercom.rpc.grpc_util.channel_auth_config import ChannelAuthConfig
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)

logger = logging.getLogger(__name__)


class ChannelFactorySelector:
    """Selects and provides instances of gRPC channel factories based on ChannelAuthConfig."""

    def _read_file_content(self, file_path: str | None) -> bytes | None:
        """Reads the content of a file if the path is provided."""
        if file_path is None:
            return None
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise  # Re-raise the exception to be handled by the caller

    def create_factory(
        self, auth_config: ChannelAuthConfig
    ) -> GrpcChannelFactory:
        """
        Creates an instance of a GrpcChannelFactory based on the provided
        ChannelAuthConfig.

        Args:
            auth_config: The ChannelAuthConfig object specifying the desired
                         channel security and parameters.

        Returns:
            An instance of a GrpcChannelFactory subclass.

        Raises:
            ValueError: If an unknown security_type is provided in auth_config
                        or if required file paths are not accessible.
        """
        logger.info(
            f"Creating GrpcChannelFactory for security_type: {auth_config.security_type}"
        )

        if auth_config.security_type == "insecure":
            return InsecureGrpcChannelFactory()

        elif auth_config.security_type == "tls_server_ca":
            if not auth_config.server_ca_cert_path:
                # This should ideally be caught by ChannelAuthConfig's __post_init__
                # but defensive check here is good.
                raise ValueError(
                    "server_ca_cert_path is required for tls_server_ca"
                )
            ca_cert_pem = self._read_file_content(
                auth_config.server_ca_cert_path
            )
            if not ca_cert_pem:
                raise ValueError(
                    f"Failed to read server_ca_cert_path: {auth_config.server_ca_cert_path}"
                )
            return ServerAuthGrpcChannelFactory(
                root_ca_cert_pem=ca_cert_pem,
                server_hostname_override=auth_config.server_hostname_override,
            )

        elif auth_config.security_type == "tls_pinned_server":
            if not auth_config.pinned_server_cert_path:
                raise ValueError(
                    "pinned_server_cert_path is required for tls_pinned_server"
                )
            pinned_cert_pem = self._read_file_content(
                auth_config.pinned_server_cert_path
            )
            if not pinned_cert_pem:
                raise ValueError(
                    f"Failed to read pinned_server_cert_path: {auth_config.pinned_server_cert_path}"
                )
            return PinnedServerAuthGrpcChannelFactory(
                expected_server_cert_pem=pinned_cert_pem,
                server_hostname_override=auth_config.server_hostname_override,
            )

        elif auth_config.security_type == "tls_client_auth":
            if (
                not auth_config.client_cert_path
                or not auth_config.client_key_path
            ):
                raise ValueError(
                    "client_cert_path and client_key_path are required for tls_client_auth"
                )
            client_cert_pem = self._read_file_content(
                auth_config.client_cert_path
            )
            client_key_pem = self._read_file_content(
                auth_config.client_key_path
            )
            if not client_cert_pem or not client_key_pem:
                raise ValueError(
                    "Failed to read client_cert_path or client_key_path for tls_client_auth"
                )

            # Per issue: "client cert for encryption with no server validation by client"
            # So, root_ca_cert_pem is explicitly None for ClientAuthGrpcChannelFactory.
            # ChannelAuthConfig's __post_init__ should also enforce that
            # server_ca_cert_path and pinned_server_cert_path are None for this type.
            return ClientAuthGrpcChannelFactory(
                client_cert_pem=client_cert_pem,
                client_key_pem=client_key_pem,
                root_ca_cert_pem=None, # Explicitly None for this security type
                server_hostname_override=auth_config.server_hostname_override,
            )
        else:
            # This case should ideally not be reached if ChannelSecurityType is used correctly
            raise ValueError(
                f"Unknown security_type: {auth_config.security_type}"
            )
