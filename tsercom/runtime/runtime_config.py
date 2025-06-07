"""Configuration parameters for Tsercom runtimes.

This module defines `ServiceType` to distinguish between client and server
runtimes, and `RuntimeConfig` to encapsulate various settings used during
runtime initialization and operation. These settings include the service type,
data aggregation mechanisms, timeouts, and security configurations.
"""

from enum import Enum
from typing import Generic, Literal, Optional, TypeVar, overload

from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.rpc.grpc_util.channel_auth_config import BaseChannelAuthConfig

# ExposedData import removed as DataTypeT is no longer bound to it here.
DataTypeT = TypeVar("DataTypeT")
# EventTypeT was defined but not used in this file, so removing unless needed elsewhere.


class ServiceType(Enum):
    """Defines the operational mode of a Tsercom runtime.

    Attributes:
        CLIENT: Indicates the runtime operates as a client, typically connecting
            to a remote server or service, receiving a `CallerIdentifier` from it,
            and synchronizing its clock with the remote.
        SERVER: Indicates the runtime operates as a server, typically accepting
            connections, assigning `CallerIdentifier`s to clients, and acting as
            the authoritative time source.
    """

    CLIENT = 0
    SERVER = 1


class RuntimeConfig(Generic[DataTypeT]):
    """Holds configuration parameters for initializing a Tsercom runtime.

    This class encapsulates settings such as the service type (client or server),
    an optional client for a remote data aggregator, data timeout values,
    minimum send frequency for events, and channel authentication configurations.

    Instances can be created either by providing individual configuration
    parameters or by cloning an existing `RuntimeConfig` object.

    Type Args:
        DataTypeT: The generic type of data objects that runtimes configured
            with this object will handle.
    """

    @overload
    def __init__(
        self,
        service_type: ServiceType,
        *,
        data_aggregator_client: Optional[RemoteDataAggregator.Client] = None,
        timeout_seconds: Optional[int] = 60,
        min_send_frequency_seconds: Optional[float] = None,
        auth_config: Optional[BaseChannelAuthConfig] = None,
    ):
        """Initializes with ServiceType enum and optional configurations.

        Args:
            service_type: The operational mode as a `ServiceType` enum.
            data_aggregator_client: Optional client for data aggregation.
            timeout_seconds: Data timeout in seconds. Defaults to 60.
            min_send_frequency_seconds: Minimum event send interval.
            auth_config: Optional channel authentication configuration.
        """
        ...

    @overload
    def __init__(
        self,
        service_type: Literal["Client", "Server"],
        *,
        data_aggregator_client: Optional[RemoteDataAggregator.Client] = None,
        timeout_seconds: Optional[int] = 60,
        min_send_frequency_seconds: Optional[float] = None,
        auth_config: Optional[BaseChannelAuthConfig] = None,
    ):
        """Initializes with service type as string and optional configurations.

        Args:
            service_type: The operational mode as "Client" or "Server".
            data_aggregator_client: Optional client for data aggregation.
            timeout_seconds: Data timeout in seconds. Defaults to 60.
            min_send_frequency_seconds: Minimum event send interval.
            auth_config: Optional channel authentication configuration.
        """
        ...

    @overload
    def __init__(self, *, other_config: "RuntimeConfig[DataTypeT]"):
        """Initializes by cloning settings from another RuntimeConfig instance.

        Args:
            other_config: An existing `RuntimeConfig` instance to clone.
                All other parameters will be ignored if this is provided.
        """
        ...

    def __init__(
        self,
        service_type: Optional[
            Literal["Client", "Server"] | ServiceType
        ] = None,
        *,
        other_config: Optional["RuntimeConfig[DataTypeT]"] = None,
        data_aggregator_client: Optional[RemoteDataAggregator.Client] = None,
        timeout_seconds: Optional[int] = 60,
        min_send_frequency_seconds: Optional[float] = None,
        auth_config: Optional[BaseChannelAuthConfig] = None,
    ):
        """Initializes the RuntimeConfig.

        This constructor is overloaded. You can initialize either by specifying
        `service_type` along with other optional parameters, or by providing
        an `other_config` instance to clone its settings.

        Args:
            service_type: The operational mode of the runtime. Can be specified
                as a `ServiceType` enum member (e.g., `ServiceType.CLIENT`) or
                as a string literal ("Client" or "Server"). Must be provided if
                `other_config` is not.
            other_config: An existing `RuntimeConfig` instance from which to
                copy all configuration settings. If provided, `service_type` and
                other direct configuration arguments must not be set.
            data_aggregator_client: An optional client for a
                `RemoteDataAggregator`. This is used if the runtime needs to
                interact with a remote data aggregation service. The client
                should be parameterized with `DataTypeT`.
            timeout_seconds: Optional. The timeout duration in seconds for data
                items. If `None`, data does not time out. Defaults to 60 seconds.
            min_send_frequency_seconds: Optional. The minimum time interval,
                in seconds, between the dispatch of event batches. This can be
                used to control the rate of event processing or transmission.
                If `None`, there is no minimum frequency enforced at this level.
            auth_config: Optional. A `BaseChannelAuthConfig` instance defining
                the authentication and encryption settings for gRPC channels
                created by the runtime. If `None`, insecure channels may be used.

        Raises:
            ValueError: If `service_type` and `other_config` are not mutually
                exclusive (i.e., both are provided or neither is provided).
                Also raised if `service_type` is a string and not "Client"
                or "Server".
            TypeError: If `service_type` is provided but is not a `ServiceType`
                enum member or a valid string literal.
        """
        if (service_type is None) == (other_config is None):
            other_config_str = (
                "<Provided>" if other_config is not None else None
            )
            raise ValueError(
                "Exactly one of 'service_type' or 'other_config' must be "
                f"provided. Got service_type={service_type}, "
                f"other_config={other_config_str}."
            )

        if other_config is not None:
            # pylint: disable=non-parent-init-called # Standard cloning pattern
            RuntimeConfig.__init__(
                self,
                service_type=other_config.service_type_enum,  # Use enum for internal consistency
                data_aggregator_client=other_config.data_aggregator_client,
                timeout_seconds=other_config.timeout_seconds,
                min_send_frequency_seconds=other_config.min_send_frequency_seconds,
                auth_config=other_config.auth_config,
            )
            return

        # Ensure service_type is not None due to the initial check, then validate and assign.
        assert service_type is not None
        if isinstance(service_type, str):
            if (
                service_type.lower() == "client"
            ):  # Case-insensitive for string convenience
                self.__service_type: ServiceType = ServiceType.CLIENT
            elif service_type.lower() == "server":
                self.__service_type = ServiceType.SERVER
            else:
                raise ValueError(
                    f"Invalid service_type string: '{service_type}'. "
                    "Must be 'Client' or 'Server'."
                )
        elif isinstance(service_type, ServiceType):
            self.__service_type = service_type
        else:
            # This path should ideally not be hit if type hints are respected,
            # but provides a runtime safeguard.
            raise TypeError(
                f"Unsupported service_type: {type(service_type)}. "
                "Must be ServiceType enum or string ('Client'/'Server')."
            )

        self.__data_aggregator_client: Optional[
            RemoteDataAggregator.Client  # Removed [DataTypeT]
        ] = data_aggregator_client
        self.__timeout_seconds: Optional[int] = timeout_seconds
        self.__auth_config: Optional[BaseChannelAuthConfig] = auth_config
        self.__min_send_frequency_seconds: Optional[float] = (
            min_send_frequency_seconds
        )

    def is_client(self) -> bool:
        """Checks if the runtime is configured to operate as a client.

        A client runtime typically initiates connections, may receive a
        `CallerIdentifier` from a server, and synchronizes its time with servers.

        Returns:
            True if the `service_type` is `ServiceType.CLIENT`, False otherwise.
        """
        return self.__service_type == ServiceType.CLIENT

    def is_server(self) -> bool:
        """Checks if the runtime is configured to operate as a server.

        A server runtime typically accepts connections, assigns `CallerIdentifier`s
        to connecting clients, and acts as the authoritative source for time
        synchronization.

        Note:
            This Tsercom "server" role is distinct from a gRPC server. For example,
            a gRPC server process might host a Tsercom client runtime if it needs
            to connect to another Tsercom service.

        Returns:
            True if the `service_type` is `ServiceType.SERVER`, False otherwise.
        """
        return self.__service_type == ServiceType.SERVER

    @property
    def service_type_enum(self) -> ServiceType:
        """The configured `ServiceType` (CLIENT or SERVER) for the runtime."""
        # __service_type is guaranteed to be ServiceType enum after __init__
        return self.__service_type

    @property
    def data_aggregator_client(
        self,
    ) -> Optional[RemoteDataAggregator.Client]:
        """The configured client for a `RemoteDataAggregator`, if any.

        Returns:
            The `RemoteDataAggregator.Client` instance, or `None` if no data
            aggregator client is configured.
        """
        return self.__data_aggregator_client

    @property
    def timeout_seconds(self) -> Optional[int]:
        """The timeout duration in seconds for data items.

        Returns:
            The timeout in seconds, or `None` if data items should not time out.
        """
        return self.__timeout_seconds

    @property
    def min_send_frequency_seconds(self) -> Optional[float]:
        """The minimum configured interval for sending/processing event batches.

        Returns:
            The minimum send frequency in seconds, or `None` if no such
            minimum is set at this configuration level.
        """
        return self.__min_send_frequency_seconds

    @property
    def auth_config(self) -> Optional[BaseChannelAuthConfig]:
        """The channel authentication configuration.

        Returns:
            A `BaseChannelAuthConfig` instance defining security settings for
            gRPC channels, or `None` if insecure channels are to be used or
            no specific auth configuration is provided.
        """
        return self.__auth_config
