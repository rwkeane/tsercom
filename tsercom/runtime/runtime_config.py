"""Configuration for Tsercom runtimes, specifying service type and data handling."""

from enum import Enum
from typing import Literal, Optional, TypeVar, overload

from tsercom.config.grpc_channel_config import GrpcChannelFactoryConfig
from tsercom.data.remote_data_aggregator import RemoteDataAggregator

# TDataType = TypeVar("TDataType") # Removed
# TEventType = TypeVar("TEventType") # Removed


class ServiceType(Enum):
    """Enumerates the operational roles for a Tsercom runtime."""

    kClient = 1
    kServer = 2


class RuntimeConfig:
    """Holds configuration parameters for a Tsercom runtime.

    This includes the service type (client or server), data aggregator client,
    and data timeout settings. It supports initialization by direct parameters
    or by copying from another `RuntimeConfig` instance.
    """

    # Server-side TLS configuration paths
    server_tls_key_path: Optional[str]
    server_tls_cert_path: Optional[str]
    server_tls_client_ca_path: Optional[str]

    @overload
    def __init__(
        self,
        service_type: "ServiceType",
        *,
        data_aggregator_client: Optional[RemoteDataAggregator] = None,
        timeout_seconds: Optional[int] = 60,
        grpc_channel_factory_config: Optional[GrpcChannelFactoryConfig] = None,
        server_tls_key_path: Optional[str] = None,
        server_tls_cert_path: Optional[str] = None,
        server_tls_client_ca_path: Optional[str] = None,
    ): ...

    @overload
    def __init__(
        self,
        service_type: Literal["Client", "Server"],
        *,
        data_aggregator_client: Optional[RemoteDataAggregator] = None,
        timeout_seconds: Optional[int] = 60,
        grpc_channel_factory_config: Optional[GrpcChannelFactoryConfig] = None,
        server_tls_key_path: Optional[str] = None,
        server_tls_cert_path: Optional[str] = None,
        server_tls_client_ca_path: Optional[str] = None,
    ): ...

    @overload
    def __init__(self, *, other_config: "RuntimeConfig"): ...

    def __init__(
        self,
        service_type: Optional[
            Literal["Client", "Server"] | ServiceType
        ] = None,
        *,
        other_config: Optional["RuntimeConfig"] = None,
        data_aggregator_client: Optional[RemoteDataAggregator] = None,
        timeout_seconds: Optional[int] = 60,
        grpc_channel_factory_config: Optional[GrpcChannelFactoryConfig] = None,
        server_tls_key_path: Optional[str] = None,  # New server TLS field
        server_tls_cert_path: Optional[str] = None,  # New server TLS field
        server_tls_client_ca_path: Optional[
            str
        ] = None,  # New server TLS field
    ):
        """Initializes the RuntimeConfig.

        Can be initialized either by specifying `service_type` and other parameters,
        or by providing an `other_config` instance to copy from.

        Args:
            service_type: The type of service ('Client', 'Server', or ServiceType enum).
            other_config: An existing `RuntimeConfig` to clone.
            data_aggregator_client: Optional client for data aggregation.
            timeout_seconds: Optional data timeout in seconds.

        Raises:
            ValueError: If both or neither of `service_type` and `other_config`
                        are provided, or if `service_type` string is invalid.
        """
        if (service_type is None) == (other_config is None):
            raise ValueError(
                "Exactly one of 'service_type' or 'other_config' must be provided to RuntimeConfig. "
                f"Got service_type={service_type}, other_config={'<Provided>' if other_config is not None else None}."
            )

        if other_config is not None:
            RuntimeConfig.__init__(
                self,
                service_type=other_config.__service_type,
                data_aggregator_client=other_config.data_aggregator_client,
                timeout_seconds=other_config.timeout_seconds,
                grpc_channel_factory_config=other_config.grpc_channel_factory_config,
                server_tls_key_path=other_config.server_tls_key_path,
                server_tls_cert_path=other_config.server_tls_cert_path,
                server_tls_client_ca_path=other_config.server_tls_client_ca_path,
            )
            return

        if isinstance(service_type, str):
            if service_type == "Client":
                self.__service_type = ServiceType.kClient
            elif service_type == "Server":
                self.__service_type = ServiceType.kServer
            else:
                raise ValueError(f"Invalid service type: {service_type}")
        else:
            self.__service_type = service_type

        self.__data_aggregator_client: Optional[RemoteDataAggregator] = (
            data_aggregator_client
        )
        self.__timeout_seconds: Optional[int] = timeout_seconds
        self.__grpc_channel_factory_config: Optional[
            GrpcChannelFactoryConfig
        ] = grpc_channel_factory_config
        self.__server_tls_key_path: Optional[str] = server_tls_key_path
        self.__server_tls_cert_path: Optional[str] = server_tls_cert_path
        self.__server_tls_client_ca_path: Optional[str] = (
            server_tls_client_ca_path
        )

    def is_client(self) -> bool:
        """Checks if the runtime is configured as a client.

        See `is_server()` for more details on client/server distinction in Tsercom.

        Returns:
            True if the service type is kClient, False otherwise.
        """
        return self.__service_type == ServiceType.kClient

    def is_server(self) -> bool:
        """
        Returns whether this instance corresponds to a Server-side runtime or a
        Client side runtime. The difference being that a server instance assigns
        CallerIds and synchronizes timestamps to the local time, as opposed to
        the client that receives its id from the server (rather than picking one
        itself) and queries time synchronization offsets with any server it
        connects to.

        NOTE: This is entirely separate from the gRPC definition of "client" and
        "server" instances. It is perfectly possible (and often happens) that
        the gRPC Server is the tsercom client.
        """
        return self.__service_type == ServiceType.kServer

    @property
    def data_aggregator_client(self) -> RemoteDataAggregator:
        """
        Returns the client that should be informed when new data is provided to
        the RemoteDataAggregator instance created for the runtime created from
        this initializer, or None if no such instance should be used.
        """
        return self.__data_aggregator_client

    @property
    def timeout_seconds(self) -> int:
        """
        Returns the timeout (in seconds) that should be used for data received
        by the runtime created from this initializer, or None if data should not
        time out.
        """
        return self.__timeout_seconds

    @property
    def grpc_channel_factory_config(
        self,
    ) -> Optional[GrpcChannelFactoryConfig]:
        """
        Returns the configuration for creating GrpcChannelFactory instances,
        or None if no specific configuration is set.
        """
        return self.__grpc_channel_factory_config

    @property
    def server_tls_key_path(self) -> Optional[str]:
        return self.__server_tls_key_path

    @property
    def server_tls_cert_path(self) -> Optional[str]:
        return self.__server_tls_cert_path

    @property
    def server_tls_client_ca_path(self) -> Optional[str]:
        return self.__server_tls_client_ca_path


class ServiceType(Enum):
    """Enumerates the operational roles for a Tsercom runtime."""

    kClient = 1
    kServer = 2
