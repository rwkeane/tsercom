"""Defines dataclasses for configuring gRPC channel authentication."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseChannelAuthConfig:
    """Base class for gRPC channel authentication configurations."""

    pass


@dataclass(frozen=True)
class InsecureChannelConfig(BaseChannelAuthConfig):
    """Configuration for an insecure gRPC channel."""

    pass


@dataclass(frozen=True)
class ServerCAChannelConfig(BaseChannelAuthConfig):
    """Configuration for a gRPC channel.

    Client authenticates the server using a root CA certificate.
    """

    server_ca_cert_path: str
    server_hostname_override: str | None = field(default=None)


@dataclass(frozen=True)
class PinnedServerChannelConfig(BaseChannelAuthConfig):
    """Configuration for a gRPC channel.

    Client authenticates the server by pinning against a specific server certificate.
    """

    pinned_server_cert_path: str
    server_hostname_override: str | None = field(default=None)


@dataclass(frozen=True)
class ClientAuthChannelConfig(BaseChannelAuthConfig):
    """Configuration for a gRPC channel.

    Client presents its certificate, and the client does not validate the
    server's certificate.
    """

    client_cert_path: str
    client_key_path: str
    server_hostname_override: str | None = field(default=None)


@dataclass(frozen=True)
class MutualTLSChannelConfig(BaseChannelAuthConfig):
    """Configuration for a gRPC channel with mutual TLS authentication.

    Client authenticates server using root CA, server authenticates client
    using its certificate and key.
    """

    root_ca_cert_path: str  # Path to the root CA certificate for server validation
    client_cert_path: str  # Path to the client's certificate
    client_key_path: str  # Path to the client's private key
    server_hostname_override: str | None = field(default=None)
