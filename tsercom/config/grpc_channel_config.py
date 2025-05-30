# tsercom/config/grpc_channel_config.py
from dataclasses import dataclass
from typing import Optional, Literal

GrpcChannelFactoryType = Literal[
    "insecure",
    "client_auth",
    "pinned_server_auth",
    "server_auth",
]


@dataclass(frozen=True)
class GrpcChannelFactoryConfig:
    """Configuration for creating GrpcChannelFactory instances."""

    factory_type: GrpcChannelFactoryType

    # Common for secure channels
    server_hostname_override: Optional[str] = None

    # For ServerAuth and optionally ClientAuth (for server validation)
    # These can be direct PEM strings or paths to PEM files.
    # The selector will handle loading if they are paths.
    root_ca_cert_pem_or_path: Optional[str] = None

    # For ClientAuth
    client_cert_pem_or_path: Optional[str] = None
    client_key_pem_or_path: Optional[str] = None

    # For PinnedServerAuth
    expected_server_cert_pem_or_path: Optional[str] = None
