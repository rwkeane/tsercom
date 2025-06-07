from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    """
    Configuration for a gRPC channel where the client authenticates the server
    using a root CA certificate.
    """

    server_ca_cert_path: str
    server_hostname_override: Optional[str] = field(default=None)


@dataclass(frozen=True)
class PinnedServerChannelConfig(BaseChannelAuthConfig):
    """
    Configuration for a gRPC channel where the client authenticates the server
    by pinning against a specific server certificate.
    """

    pinned_server_cert_path: str
    server_hostname_override: Optional[str] = field(default=None)


@dataclass(frozen=True)
class ClientAuthChannelConfig(BaseChannelAuthConfig):
    """
    Configuration for a gRPC channel where the client presents its certificate,
    and the client does not validate the server's certificate.
    """

    client_cert_path: str
    client_key_path: str
    server_hostname_override: Optional[str] = field(default=None)
