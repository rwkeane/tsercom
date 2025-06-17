import asyncio
import logging
from typing import Optional, TYPE_CHECKING

import grpc
import pytest
import pytest_asyncio
import socket
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
from tsercom.discovery.mdns.mdns_listener import MdnsListener
from zeroconf.asyncio import AsyncZeroconf  # Added import

if TYPE_CHECKING:
    pass  # This can remain for type checking if it's used elsewhere, or be removed if AsyncZeroconf replaces all uses.

    # For FakeMdnsListener methods, we'll use AsyncZeroconf.


has_been_hit = Atomic[bool](False)


class FakeMdnsListener(MdnsListener):
    __test__ = False  # To prevent pytest from collecting it as a test

    def __init__(
        self,
        client: MdnsListener.Client,
        service_type: str,
        port: int,
        zc_instance: Optional[AsyncZeroconf] = None,  # Added zc_instance
    ):
        # zc_instance is accepted for signature compatibility but not used by this Fake
        # super().__init__() # MdnsListener's parent (ServiceListener) has no __init__
        self.__client = client
        self.__service_type = service_type
        self.__port = port
        logging.info(
            f"FakeMdnsListener initialized for service type '{self.__service_type}' on port {self.__port}"
        )

    async def start(self) -> None:  # Changed to async def
        logging.info(
            f"FakeMdnsListener: Faking service addition for service type '{self.__service_type}' on port {self.__port}"
        )
        fake_mdns_instance_name = f"FakedServiceInstance.{self.__service_type}"
        if not fake_mdns_instance_name.endswith(".local."):
            if self.__service_type.count(
                "."
            ) == 2 and self.__service_type.endswith("."):
                fake_mdns_instance_name = (
                    f"FakedServiceInstance.{self.__service_type}local."
                )
            elif self.__service_type.count(".") == 1:
                fake_mdns_instance_name = (
                    f"FakedServiceInstance.{self.__service_type}.local."
                )

        fake_ip_address_bytes = socket.inet_aton("127.0.0.1")

        await self.__client._on_service_added(  # Changed to await
            name=fake_mdns_instance_name,
            port=self.__port,
            addresses=[fake_ip_address_bytes],
            txt_record={},
        )
        logging.info(
            f"FakeMdnsListener: _on_service_added called for {fake_mdns_instance_name}"
        )

    async def add_service(
        self, zc: AsyncZeroconf, type_: str, name: str
    ) -> None:  # Changed to async, type hint updated
        logging.debug(
            f"FakeMdnsListener: add_service called for {name} type {type_}, no action."
        )
        pass

    async def update_service(
        self, zc: AsyncZeroconf, type_: str, name: str
    ) -> None:  # Changed to async, type hint updated
        logging.debug(
            f"FakeMdnsListener: update_service called for {name}, no action."
        )
        pass

    async def remove_service(
        self, zc: AsyncZeroconf, type_: str, name: str
    ) -> None:  # Changed to async, type hint updated
        logging.debug(
            f"FakeMdnsListener: remove_service called for {name}, no action."
        )
        pass

    async def close(self) -> None:
        logging.info(
            f"FakeMdnsListener: close() called for service type '{self.__service_type}'."
        )
        pass


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
        logging.info("GenericServerRuntime stopping...")
        if self.__connector:
            await self.__connector.stop()
        # super().stop(exception) # Runtime.stop is synchronous.
        logging.info("GenericServerRuntime stopped.")

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

        def __connect(server: grpc.Server):
            # Add health servicer
            from grpc_health.v1 import health
            from grpc_health.v1 import health_pb2
            from grpc_health.v1 import health_pb2_grpc

            health_servicer = health.HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(
                health_servicer, server
            )
            # Mark the service as serving. Adjust service name as needed.
            health_servicer.set(
                "tsercom.GenericClientRuntime",
                health_pb2.HealthCheckResponse.SERVING,
            )

        await self.__grpc_publisher.start_async(__connect)

        await self.__mdns_publiser.publish()

    async def stop(self, exception: Optional[Exception] = None):
        logging.info("GenericClientRuntime stopping...")
        self.__is_running.stop()
        if self.__mdns_publiser:
            # Assuming InstancePublisher might need an async unpublish or close
            if hasattr(self.__mdns_publiser, "close") and callable(
                getattr(self.__mdns_publiser, "close")
            ):
                await self.__mdns_publiser.close()  # type: ignore
            elif hasattr(self.__mdns_publiser, "unpublish") and callable(
                getattr(self.__mdns_publiser, "unpublish")
            ):
                # If unpublish is not async, and close is preferred for async cleanup
                # This branch might indicate a sync unpublish. For now, assume close is the async one if present.
                pass  # Or call self.__mdns_publiser.unpublish() if it's okay to be sync

        if self.__grpc_publisher:
            await self.__grpc_publisher.stop_async()
        logging.info("GenericClientRuntime stopped.")


class GenericServerRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        *,
        listener_factory: Optional[MdnsListenerFactory] = None,
        fake_service_port: Optional[int] = None,
    ):
        self.__listener_factory = listener_factory
        self.__fake_service_port = fake_service_port
        super().__init__(service_type=ServiceType.SERVER)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        actual_mdns_listener_factory: Optional[MdnsListenerFactory] = None
        if self.__fake_service_port is not None:
            logging.info(
                f"Using FakeMdnsListener for port {self.__fake_service_port}"
            )

            # The service_type_arg in the lambda is what DiscoveryHost would pass to the factory.
            # FakeMdnsListener will use this service_type_arg.
            def fake_factory(
                client: MdnsListener.Client,
                service_type_arg: str,
                zc_instance_arg: Optional[
                    AsyncZeroconf
                ] = None,  # Added zc_instance_arg
            ) -> FakeMdnsListener:
                return FakeMdnsListener(
                    client,
                    service_type_arg,
                    self.__fake_service_port,
                    zc_instance=zc_instance_arg,  # Pass it through
                )

            actual_mdns_listener_factory = fake_factory  # type: ignore[assignment]
        else:
            actual_mdns_listener_factory = self.__listener_factory

        return GenericServerRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            mdns_listener_factory=actual_mdns_listener_factory,
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


@pytest_asyncio.fixture
async def clear_loop_fixture():
    # Ensure tsercom's global loop is managed.
    clear_tsercom_event_loop()
    yield
    clear_tsercom_event_loop()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_anomoly_service(clear_loop_fixture):
    loggers_to_modify = {
        "tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory": logging.INFO,
        "tsercom.discovery.service_connector": logging.INFO,
        "tsercom.discovery.mdns.record_listener": logging.INFO,
        "zeroconf": logging.DEBUG,
    }
    original_levels = {}
    for logger_name, temp_level in loggers_to_modify.items():
        logger_instance = logging.getLogger(logger_name)
        original_levels[logger_name] = logger_instance.level
        logger_instance.setLevel(temp_level)

    client_initializer = GenericClientRuntimeInitializer(2024, "Client")
    server_initializer = GenericServerRuntimeInitializer(fake_service_port=2024)

    runtime_manager = RuntimeManager(is_testing=True)
    client_handle_f = runtime_manager.register_runtime_initializer(
        client_initializer
    )
    server_handle_f = runtime_manager.register_runtime_initializer(
        server_initializer
    )

    # Create EventLoop. (Removed custom threaded loop)

    await runtime_manager.start_in_process_async()
    runtime_manager.check_for_exception()

    assert client_handle_f.done()
    assert server_handle_f.done()

    client_handle = client_handle_f.result()
    server_handle = server_handle_f.result()

    assert not has_been_hit.get()

    client_handle.start()
    runtime_manager.check_for_exception()
    server_handle.start()
    runtime_manager.check_for_exception()

    client_handle.on_event(torch.zeros(5))  # This is likely synchronous
    await asyncio.sleep(2.0)
    runtime_manager.check_for_exception()

    assert has_been_hit.get()

    runtime_manager.check_for_exception()
    client_handle.stop()
    server_handle.stop()
    await asyncio.sleep(0.5)
    runtime_manager.check_for_exception()
    runtime_manager.shutdown()

    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)
