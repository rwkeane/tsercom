import asyncio
import threading
import time
from typing import Optional
import grpc
import pytest
import torch

from tsercom.api import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.discovery.mdns.instance_listener import MdnsListenerFactory
from tsercom.discovery.mdns.instance_publisher import InstancePublisher
from tsercom.discovery.service_connector import ServiceConnector
from tsercom.discovery.service_info import ServiceInfo
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_config import ServiceType
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.global_event_loop import clear_tsercom_event_loop
from tsercom.threading.atomic import Atomic
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


has_been_hit = Atomic[bool](False)


class GenericServerRuntime(
    Runtime,
    ServiceConnector[ServiceInfo, grpc.Channel].Client,
):
    """
    Top-level class for handling connections with a client instance.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        channel_factory: GrpcChannelFactory,
        *,
        mdns_listener_factory: Optional[MdnsListenerFactory] = None,
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler

        # Handle service discovery.
        discoverer: DiscoveryHost
        if mdns_listener_factory is None:
            discoverer = DiscoveryHost(service_type="_foo._tcp")
        else:
            discoverer = DiscoveryHost(
                mdns_listener_factory=mdns_listener_factory
            )
        self.__connector = ServiceConnector[ServiceInfo, grpc.Channel](
            self, channel_factory, discoverer
        )

        super().__init__()

    async def start_async(self):
        """
        Allow for connections with clients to start.
        """
        await self.__connector.start()

    async def stop(self, exception: Optional[Exception] = None):
        pass

    async def _on_channel_connected(
        self,
        connection_info: ServiceInfo,
        caller_id: CallerIdentifier,
        channel_info: ChannelInfo,
    ):
        has_been_hit.set(True)


class GenericClientRuntime(
    Runtime,
):
    """
    This is the top-level class for the client-side of the Service.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        readable_name: str,
        port: int,
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler

        self.__is_running = IsRunningTracker()
        self.__mdns_publiser = InstancePublisher(
            port, "_foo._tcp", readable_name
        )
        self.__grpc_publisher = GrpcServicePublisher(self.__watcher, port)

        super().__init__()

    async def start_async(self):
        """
        Starts connecting with server instances, as well as advertising the
        connect-ability over mDNS.
        """
        assert not self.__is_running.get()
        self.__is_running.start()

        # def __connect(server: grpc.Server):
        #     add_CovarianceAnomolyServiceServicer_to_server(self, server)
        # await self.__grpc_publisher.start_async(__connect)

        await self.__mdns_publiser.publish()

    async def stop(self, exception: Optional[Exception] = None):
        self.__is_running.stop()
        await self.__grpc_publisher.stop()


class GenericServerRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        *,
        listener_factory: Optional[MdnsListenerFactory] = None,
    ):
        self.__listener_factory = listener_factory

        super().__init__(service_type=ServiceType.SERVER)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return GenericServerRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            mdns_listener_factory=self.__listener_factory,
        )


class GenericClientRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(self, host_port: int, name: str):
        self.__host_port = host_port
        self.__name = name

        super().__init__(service_type=ServiceType.CLIENT)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return GenericClientRuntime(
            thread_watcher, data_handler, self.__name, self.__host_port
        )


@pytest.fixture
def clear_loop_fixture():
    clear_tsercom_event_loop()
    yield
    # clear_tsercom_event_loop()


def test_anomoly_service(clear_loop_fixture):
    # Create runtimes and register them.
    client_initializer = GenericClientRuntimeInitializer(2024, "Client")
    server_initializer = GenericServerRuntimeInitializer()

    runtime_manager = RuntimeManager(is_testing=True)
    client_handle_f = runtime_manager.register_runtime_initializer(
        client_initializer
    )
    server_handle_f = runtime_manager.register_runtime_initializer(
        server_initializer
    )

    # Create EventLoop.
    def run_loop_in_thread(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=run_loop_in_thread, args=(new_loop,), daemon=True
    )
    thread.start()

    # Start it all.
    runtime_manager.start_in_process(new_loop)
    runtime_manager.check_for_exception()

    assert client_handle_f.done()
    assert server_handle_f.done()

    client_handle = client_handle_f.result()
    server_handle = server_handle_f.result()

    # Pass data.
    assert not has_been_hit.get()

    client_handle.start()
    runtime_manager.check_for_exception()
    server_handle.start()
    runtime_manager.check_for_exception()

    # Client -> Server.
    client_handle.on_event(torch.zeros(5))
    time.sleep(0.1)
    runtime_manager.check_for_exception()

    assert has_been_hit.get()

    # Stop and cleanup
    runtime_manager.check_for_exception()
    client_handle.stop()
    server_handle.stop()
    time.sleep(0.5)
    runtime_manager.check_for_exception()
    runtime_manager.shutdown()
