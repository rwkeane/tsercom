"""Defines dataclasses for gRPC channel authentication configurations."""

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
    """Client authenticates server using a root CA certificate."""

    server_ca_cert_path: str
    server_hostname_override: str | None = field(default=None)


@dataclass(frozen=True)
class PinnedServerChannelConfig(BaseChannelAuthConfig):
    """Client authenticates server by pinning against a specific server certificate."""

    pinned_server_cert_path: str
    server_hostname_override: str | None = field(default=None)


@dataclass(frozen=True)
class ClientAuthChannelConfig(BaseChannelAuthConfig):
    """Client presents its certificate; client does not validate server's cert."""

    client_cert_path: str
    client_key_path: str
    server_hostname_override: str | None = field(default=None)
