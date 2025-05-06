from enum import Enum
from typing import Literal, Optional, TypeVar, overload
from tsercom.data.remote_data_aggregator import RemoteDataAggregator

TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class RuntimeConfig:
    @overload
    def __init__(
        self,
        service_type: "ServiceType",
        *,
        data_aggregator_client: Optional[RemoteDataAggregator] = None,
        timeout_seconds: Optional[int] = 60,
    ): ...

    @overload
    def __init__(
        self,
        service_type: Literal["Client", "Server"],
        *,
        data_aggregator_client: Optional[RemoteDataAggregator] = None,
        timeout_seconds: Optional[int] = 60,
    ): ...

    @overload
    def __init__(self, *, other_config: "RuntimeConfig"): ...

    def __init__(
        self,
        service_type: Optional[
            Literal["Client", "Server"] | "ServiceType"
        ] = None,
        *,
        other_config: Optional["RuntimeConfig"] = None,
        data_aggregator_client: Optional[RemoteDataAggregator] = None,
        timeout_seconds: Optional[int] = 60,
    ):
        assert (service_type is None) != (other_config is None), (
            service_type,
            other_config,
        )

        # Handle the delegating option.
        if other_config is not None:
            RuntimeConfig.__init__(
                self,
                service_type=other_config.__service_type,
                data_aggregator_client=other_config.data_aggregator_client,
                timeout_seconds=other_config.timeout_seconds,
            )
            return

        # Handle the default case.
        if isinstance(service_type, str):
            if service_type == "Client":
                self.__service_type = ServiceType.kClient
            elif service_type == "Server":
                self.__service_type = ServiceType.kServer
            else:
                raise ValueError(f"Invalid service type: {service_type}")
        else:
            self.__service_type = service_type

        self.__data_aggregator_client = data_aggregator_client
        self.__timeout_seconds = timeout_seconds

    def is_client(self) -> bool:
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


class ServiceType(Enum):
    kClient = 1
    kServer = 2
