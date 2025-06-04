"""Config for Tsercom runtimes: service type, data handling."""

from enum import Enum
from typing import Literal, Optional, TypeVar, overload, Generic
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.exposed_data import ExposedData
from tsercom.rpc.grpc_util.channel_auth_config import BaseChannelAuthConfig

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


class ServiceType(Enum):
    """Defines the type of service for a Tsercom runtime."""

    CLIENT = 0
    SERVER = 1


class RuntimeConfig(Generic[DataTypeT]):
    """Holds configuration parameters for a Tsercom runtime.

    Includes service type, data aggregator client, timeout, and auth.
    Supports init by direct params or by copying another RuntimeConfig.
    """

    @overload
    def __init__(
        self,
        service_type: "ServiceType",
        *,
        data_aggregator_client: Optional[
            RemoteDataAggregator[DataTypeT]
        ] = None,
        timeout_seconds: Optional[int] = 60,
        auth_config: Optional[BaseChannelAuthConfig] = None,
    ): ...

    @overload
    def __init__(
        self,
        service_type: Literal["Client", "Server"],
        *,
        data_aggregator_client: Optional[
            RemoteDataAggregator[DataTypeT]
        ] = None,
        timeout_seconds: Optional[int] = 60,
        auth_config: Optional[BaseChannelAuthConfig] = None,
    ): ...

    @overload
    def __init__(self, *, other_config: "RuntimeConfig[DataTypeT]"): ...

    # pylint: disable=too-many-arguments # Config object needs many options
    def __init__(
        self,
        service_type: Optional[
            Literal["Client", "Server"] | ServiceType
        ] = None,
        *,
        other_config: Optional["RuntimeConfig[DataTypeT]"] = None,
        data_aggregator_client: Optional[
            RemoteDataAggregator[DataTypeT]
        ] = None,
        timeout_seconds: Optional[int] = 60,
        auth_config: Optional[BaseChannelAuthConfig] = None,
    ):
        """Initializes the RuntimeConfig.

        Can be initialized by `service_type` and other parameters,
        or by providing an `other_config` instance to copy from.

        Args:
            service_type: 'Client', 'Server', or ServiceType enum.
            other_config: An existing `RuntimeConfig` to clone.
            data_aggregator_client: Optional client for data aggregation.
            timeout_seconds: Optional data timeout in seconds.
            auth_config: Optional channel authentication configuration.

        Raises:
            ValueError: If `service_type` and `other_config` are not mutually
                        exclusive, or `service_type` string is invalid.
        """
        if (service_type is None) == (other_config is None):
            # Using f-string for ValueError as Pylint prefers it over %-style here
            other_config_str = (
                "<Provided>" if other_config is not None else None
            )
            raise ValueError(
                "Exactly one of 'service_type' or 'other_config' must be "
                f"provided. Got service_type={service_type}, "
                f"other_config={other_config_str}."
            )

        if other_config is not None:
            # Call __init__ again without 'other_config', using primary
            # constructor logic with values from other_config.
            # pylint: disable=non-parent-init-called # Recursive call for cloning
            RuntimeConfig.__init__(
                self,
                service_type=other_config.service_type_enum,
                data_aggregator_client=other_config.data_aggregator_client,
                timeout_seconds=other_config.timeout_seconds,
                auth_config=other_config.auth_config,
            )
            return

        if isinstance(service_type, str):
            if service_type == "Client":
                self.__service_type = ServiceType.CLIENT
            elif service_type == "Server":
                self.__service_type = ServiceType.SERVER
            else:
                raise ValueError(f"Invalid service type: {service_type}")
        elif isinstance(service_type, ServiceType):
            self.__service_type = service_type
        else:
            # This case should ideally not be reached with type hints.
            raise TypeError(f"Unsupported service_type: {type(service_type)}")

        self.__data_aggregator_client: Optional[
            RemoteDataAggregator[DataTypeT]
        ] = data_aggregator_client
        self.__timeout_seconds: Optional[int] = timeout_seconds
        self.__auth_config: Optional[BaseChannelAuthConfig] = auth_config

    def is_client(self) -> bool:
        """Checks if the runtime is configured as a client.

        See `is_server()` for Tsercom client/server distinction details.

        Returns:
            True if the service type is CLIENT, False otherwise.
        """
        return self.__service_type == ServiceType.CLIENT

    def is_server(self) -> bool:
        """Checks if runtime is Server-side or Client-side.

        Server assigns CallerIds and syncs time locally. Client receives ID
        from server and syncs time offsets with connected servers.

        NOTE: Distinct from gRPC client/server. gRPC Server can be tsercom client.
        """
        return self.__service_type == ServiceType.SERVER

    @property
    def service_type_enum(self) -> "ServiceType":
        """Returns the raw ServiceType enum value."""
        if isinstance(self.__service_type, str):
            if self.__service_type == "Client":
                return ServiceType.CLIENT
            if self.__service_type == "Server":
                return ServiceType.SERVER
            raise ValueError(
                f"Invalid string for service_type: {self.__service_type}"
            )
        return self.__service_type

    @property
    def data_aggregator_client(
        self,
    ) -> Optional[RemoteDataAggregator[DataTypeT]]:
        """Client for new data notifications. None if not set."""
        return self.__data_aggregator_client

    @property
    def timeout_seconds(self) -> Optional[int]:
        """Timeout in seconds for data received by the runtime.

        Returns None if data should not time out.
        """
        return self.__timeout_seconds

    @property
    def auth_config(self) -> Optional[BaseChannelAuthConfig]:
        """Channel auth config, or None for insecure channel."""
        return self.__auth_config
