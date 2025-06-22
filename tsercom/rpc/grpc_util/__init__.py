"""Utilities and helpers for working with gRPC in Tsercom.

This package provides tools for:
- Channel authentication configuration.
- gRPC channel creation and health checking.
- Server-side exception handling via interceptors.
- Extracting client information from gRPC contexts.
- Publishing gRPC services.
"""
from tsercom.rpc.grpc_util.addressing import get_client_ip, get_client_port
from tsercom.rpc.grpc_util.async_grpc_exception_interceptor import (
    AsyncGrpcExceptionInterceptor,
)
from tsercom.rpc.grpc_util.channel_auth_config import (
    BaseChannelAuthConfig,
    ClientAuthChannelConfig,
    InsecureChannelConfig,
    PinnedServerChannelConfig,
    ServerCAChannelConfig,
)
from tsercom.rpc.grpc_util.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.grpc_caller import (
    is_grpc_error,
    is_server_unavailable_error,
)
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher

__all__ = [
    "BaseChannelAuthConfig",
    "InsecureChannelConfig",
    "ServerCAChannelConfig",
    "PinnedServerChannelConfig",
    "ClientAuthChannelConfig",
    "GrpcChannelFactory",
    "get_client_ip",
    "get_client_port",
    "AsyncGrpcExceptionInterceptor",
    "is_server_unavailable_error",
    "is_grpc_error",
    "GrpcServicePublisher",
    "ChannelInfo",
]
