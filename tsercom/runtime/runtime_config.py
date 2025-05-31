"""Configuration for Tsercom runtimes, specifying service type and data handling."""

from enum import Enum
from typing import Literal, Optional, TypeVar, overload, Generic
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.exposed_data import ExposedData  # Import ExposedData

TDataType = TypeVar("TDataType", bound=ExposedData)  # Constrain TDataType
TEventType = TypeVar("TEventType")


class RuntimeConfig(Generic[TDataType]):
    """Holds configuration parameters for a Tsercom runtime.

    This includes the service type (client or server), data aggregator client,
    and data timeout settings. It supports initialization by direct parameters
    or by copying from another `RuntimeConfig` instance.
    """

    @overload
    def __init__(
        self,
        service_type: "ServiceType",
        *,
        data_aggregator_client: Optional[
            RemoteDataAggregator[TDataType]
        ] = None,
        timeout_seconds: Optional[int] = 60,
    ): ...

    @overload
    def __init__(
        self,
        service_type: Literal["Client", "Server"],
        *,
        data_aggregator_client: Optional[
            RemoteDataAggregator[TDataType]
        ] = None,
        timeout_seconds: Optional[int] = 60,
    ): ...

    @overload
    def __init__(self, *, other_config: "RuntimeConfig[TDataType]"): ...

    def __init__(
        self,
        service_type: Optional[
            Literal["Client", "Server"] | ServiceType
        ] = None,
        *,
        other_config: Optional["RuntimeConfig[TDataType]"] = None,
        data_aggregator_client: Optional[
            RemoteDataAggregator[TDataType]
        ] = None,
        timeout_seconds: Optional[int] = 60,
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
            )
            return

        if isinstance(service_type, str):
            if service_type == "Client":
                self.__service_type = ServiceType.kClient
            elif service_type == "Server":
                self.__service_type = ServiceType.kServer
            else:
                raise ValueError(f"Invalid service type: {service_type}")
            self.__service_type = service_type

        self.__data_aggregator_client: Optional[
            RemoteDataAggregator[TDataType]
        ] = data_aggregator_client
        self.__timeout_seconds: Optional[int] = timeout_seconds

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
    def service_type_enum(self) -> "ServiceType":
        """Returns the raw ServiceType enum value."""
        return (
            self.__service_type
        )  # __service_type is ServiceType | str, but should be ServiceType after __init__ logic

    @property
    def data_aggregator_client(
        self,
    ) -> Optional[RemoteDataAggregator[TDataType]]:
        """
        Returns the client that should be informed when new data is provided to
        the RemoteDataAggregator instance created for the runtime created from
        this initializer, or None if no such instance should be used.
        """
        return self.__data_aggregator_client

    @property
    def timeout_seconds(self) -> Optional[int]:
        """
        Returns the timeout (in seconds) that should be used for data received
        by the runtime created from this initializer, or None if data should not
        time out.
        """
        return self.__timeout_seconds
