# tsercom/rpc/grpc_util/channel_auth_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# Define a type alias for the security types for clarity and potential future validation
ChannelSecurityType = Literal[
    "insecure",
    "tls_server_ca",
    "tls_pinned_server",
    "tls_client_auth",
]


@dataclass(frozen=True)
class ChannelAuthConfig:
    """
    Configuration for creating gRPC channels, specifying security parameters.

    Attributes:
        security_type: The type of security to apply to the channel.
        server_ca_cert_path: Path to the root CA certificate file for server
                             authentication. Required if security_type is
                             'tls_server_ca'.
        pinned_server_cert_path: Path to the server's certificate file to pin
                                 against. Required if security_type is
                                 'tls_pinned_server'.
        client_cert_path: Path to the client's certificate file. Required if
                          security_type is 'tls_client_auth'.
        client_key_path: Path to the client's private key file. Required if
                         security_type is 'tls_client_auth'.
        server_hostname_override: Optional hostname to use for SSL target name
                                  override. Useful if the server's certificate
                                  CN does not match the target address.
    """

    security_type: ChannelSecurityType
    server_ca_cert_path: Optional[str] = field(default=None)
    pinned_server_cert_path: Optional[str] = field(default=None)
    client_cert_path: Optional[str] = field(default=None)
    client_key_path: Optional[str] = field(default=None)
    server_hostname_override: Optional[str] = field(default=None)

    def __post_init__(self):
        # Basic validation to ensure required paths are provided for the security type
        if self.security_type == "tls_server_ca":
            if not self.server_ca_cert_path:
                raise ValueError(
                    "server_ca_cert_path is required for 'tls_server_ca' security type."
                )
        elif self.security_type == "tls_pinned_server":
            if not self.pinned_server_cert_path:
                raise ValueError(
                    "pinned_server_cert_path is required for 'tls_pinned_server' security type."
                )
        elif self.security_type == "tls_client_auth":
            if not self.client_cert_path or not self.client_key_path:
                raise ValueError(
                    "client_cert_path and client_key_path are required for 'tls_client_auth' security type."
                )

        # Ensure that only relevant paths are provided for a given security type to avoid confusion
        # For 'insecure', all cert paths should be None
        if self.security_type == "insecure":
            if self.server_ca_cert_path or \
               self.pinned_server_cert_path or \
               self.client_cert_path or \
               self.client_key_path:
                raise ValueError(
                    "Certificate paths should not be provided for 'insecure' security type."
                )
        # For 'tls_server_ca', other secure paths should be None
        elif self.security_type == "tls_server_ca":
            if self.pinned_server_cert_path or \
               self.client_cert_path or \
               self.client_key_path:
                raise ValueError(
                    "Only server_ca_cert_path should be provided for 'tls_server_ca'."
                )
        # For 'tls_pinned_server', other secure paths should be None
        elif self.security_type == "tls_pinned_server":
            if self.server_ca_cert_path or \
               self.client_cert_path or \
               self.client_key_path:
                raise ValueError(
                    "Only pinned_server_cert_path should be provided for 'tls_pinned_server'."
                )
        # For 'tls_client_auth', other server auth paths should be None
        elif self.security_type == "tls_client_auth":
            if self.server_ca_cert_path or \
               self.pinned_server_cert_path:
                # Note: ClientAuthGrpcChannelFactory *can* take a root_ca_cert_pem for server validation.
                # The issue statement: "client cert for encryption with no server validation by client"
                # implies we want to enforce this constraint here for this specific type.
                # If combined client+server auth (client validates server, server validates client)
                # becomes a desired distinct type, a new ChannelSecurityType could be added.
                # For now, 'tls_client_auth' means only client provides certs, and client does not validate server.
                raise ValueError(
                    "server_ca_cert_path and pinned_server_cert_path should not be provided for 'tls_client_auth' if client is not validating the server."
                )
