from abc import ABC, abstractmethod
import asyncio
from functools import partial
from typing import Generic, List, Set, TypeVar
import typing

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.discovery.service_info import ServiceInfo
from tsercom.rpc.connection.channel_info import ChannelInfo
from tsercom.threading.aio.aio_utils import get_running_loop_or_none, is_running_on_event_loop, run_on_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher

if typing.TYPE_CHECKING:
    from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory


TServiceInfo = TypeVar('TServiceInfo', bound = ServiceInfo)
class DiscoverableGrpcEndpointConnector(Generic[TServiceInfo],
                                        DiscoveryHost.Client):
    """
    This class provides a simple wrapper around a DiscoveryHost, extending its
    functionality such that it will additionally try to connect a channel when
    an endpoint is found.
    """
    class Client(ABC):
        @abstractmethod
        async def _on_channel_connected(self,
                                        connection_info : TServiceInfo,
                                        caller_id : CallerIdentifier,
                                        channel_info : ChannelInfo):
            pass
        
    def __init__(self,
                 client : 'DiscoverableGrpcEndpointConnector.Client',
                 channel_factory : "GrpcChannelFactory",
                 discovery_host : DiscoveryHost[TServiceInfo]):
        self.__client = client
        self.__discovery_host = discovery_host
        self.__channel_factory = channel_factory

        self.__callers = set[CallerIdentifier]()

        self.__event_loop = None

        super().__init__()

    def start(self):
        """
        Starts service discovery.
        """
        self.__discovery_host.start_discovery(self)

    async def mark_client_failed(self, caller_id : CallerIdentifier):
        """
        Marks that the client associated with |client_id| is unhealthy and
        can be replaced.
        """
        if not is_running_on_event_loop(self.__event_loop):
            run_on_event_loop(partial(self.mark_client_failed, caller_id),
                              self.__event_loop)
            return

        assert caller_id in self.__callers
        self.__callers.remove(caller_id)

    async def _on_service_added(
            self, connection_info : TServiceInfo, caller_id : CallerIdentifier):
        if self.__event_loop is None:
            self.__event_loop = get_running_loop_or_none()
            assert not self.__event_loop is None
        else:
            assert is_running_on_event_loop(self.__event_loop)

        # Check if a connection already exists.
        if caller_id in self.__callers:
            print("DROPPING CALL: Already in use!")
            return
        
        # Try and create the gRPC connection.
        channel = await self.__channel_factory.find_async_channel(
                connection_info.addresses, connection_info.port)
        if channel is None:
            print("Invalid endpoint found!")
            return
        
        print("Endpoint connected!")
        
        # Pass it along to the next layer up.
        self.__callers.add(caller_id)
        await self.__client._on_channel_connected(
                connection_info, caller_id, channel)