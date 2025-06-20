from tsercom.test.proto import (
    E2ETestServiceStub,
    add_E2ETestServiceServicer_to_server,
    EchoRequest,
    EchoResponse,
    StreamDataRequest,
    E2ETestServiceServicer,
)

import asyncio
import logging
from typing import Optional, TYPE_CHECKING, AsyncIterator, cast, Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore
import pytest
import pytest_asyncio
import socket
import torch

from tsercom.api import RuntimeManager
from tsercom.api.local_process.runtime_wrapper import RuntimeWrapper
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
from zeroconf.asyncio import AsyncZeroconf

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
        zc_instance: Optional[AsyncZeroconf] = None,
    ):
        # zc_instance is accepted for signature compatibility but not used by this Fake
        # super().__init__() # MdnsListener's parent (ServiceListener) has no __init__
        self.__client = client
        self.__service_type = service_type
        self.__port = port
        logging.info(
            f"FakeMdnsListener initialized for service type '{self.__service_type}' on port {self.__port}"
        )

    async def start(self) -> None:
        logging.info(
            f"FakeMdnsListener: Faking service addition for service type '{self.__service_type}' on port {self.__port}"
        )
        fake_mdns_instance_name = f"FakedServiceInstance.{self.__service_type}"
        if not fake_mdns_instance_name.endswith(".local."):
            if self.__service_type.count(".") == 2 and self.__service_type.endswith(
                "."
            ):
                fake_mdns_instance_name = (
                    f"FakedServiceInstance.{self.__service_type}local."
                )
            elif self.__service_type.count(".") == 1:
                fake_mdns_instance_name = (
                    f"FakedServiceInstance.{self.__service_type}.local."
                )

        fake_ip_address_bytes = socket.inet_aton("127.0.0.1")

        await self.__client._on_service_added(
            name=fake_mdns_instance_name,
            port=self.__port,
            addresses=[fake_ip_address_bytes],
            txt_record={},
        )
        logging.info(
            f"FakeMdnsListener: _on_service_added called for {fake_mdns_instance_name}"
        )

    async def add_service(self, zc: AsyncZeroconf, type_: str, name: str) -> None:
        logging.debug(
            f"FakeMdnsListener: add_service called for {name} type {type_}, no action."
        )
        pass

    async def update_service(self, zc: AsyncZeroconf, type_: str, name: str) -> None:
        logging.debug(f"FakeMdnsListener: update_service called for {name}, no action.")
        pass

    async def remove_service(self, zc: AsyncZeroconf, type_: str, name: str) -> None:
        logging.debug(f"FakeMdnsListener: remove_service called for {name}, no action.")
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
        server_grpc_port: Optional[int] = None,
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler
        self.__server_grpc_port = server_grpc_port
        self.__grpc_publisher: Optional[GrpcServicePublisher] = None
        if self.__server_grpc_port is not None:
            logging.info(
                f"GenericServerRuntime will publish its own gRPC services on port {self.__server_grpc_port}"
            )
            self.__grpc_publisher = GrpcServicePublisher(
                self.__watcher, self.__server_grpc_port
            )
        else:
            logging.info(
                "GenericServerRuntime will NOT publish its own gRPC services (no port provided)"
            )

        # Handle service discovery.
        discoverer: DiscoveryHost
        if mdns_listener_factory is None:
            discoverer = DiscoveryHost(service_type="_foo._tcp")
        else:
            discoverer = DiscoveryHost(mdns_listener_factory=mdns_listener_factory)
        self.__connector = ServiceConnector[ServiceInfo, grpc.Channel](
            self, channel_factory, discoverer
        )

        super().__init__()

    async def start_async(self):
        """
        Allow for connections with clients to start.
        """
        if self.__grpc_publisher:

            def __connect_e2e_servicer(server: grpc.Server):
                logging.info(
                    "Adding E2eTestServicer to GenericServerRuntime's gRPC server."
                )
                add_E2ETestServiceServicer_to_server(E2eTestServicer(), server)
                health_servicer = health.HealthServicer()
                health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
                health_servicer.set(
                    "tsercom.GenericServerRuntime.E2ETestService",
                    health_pb2.HealthCheckResponse.SERVING,
                )

            await self.__grpc_publisher.start_async(__connect_e2e_servicer)
        await self.__connector.start()

    async def stop(self, exception: Optional[Exception] = None):
        if self.__grpc_publisher:
            await self.__grpc_publisher.stop_async()
        logging.info("GenericServerRuntime stopping...")
        if self.__connector:
            await self.__connector.stop()
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
        grpc_channel_factory: GrpcChannelFactory,
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory

        self.__is_running = IsRunningTracker()
        self.__mdns_publiser = InstancePublisher(port, "_foo._tcp", readable_name)
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
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
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
                await self.__mdns_publiser.close()
            elif hasattr(self.__mdns_publiser, "unpublish") and callable(
                getattr(self.__mdns_publiser, "unpublish")
            ):
                # If unpublish is not async, and close is preferred for async cleanup
                # This branch might indicate a sync unpublish. For now, assume close is the async one if present.
                pass  # Or call self.__mdns_publiser.unpublish() if it's okay to be sync

        if self.__grpc_publisher:
            await self.__grpc_publisher.stop_async()
        logging.info("GenericClientRuntime stopped.")

    async def create_e2e_test_stub(self, target_address: str) -> E2ETestServiceStub:
        """Creates a gRPC stub for the E2ETestService.

        Args:
            target_address: The address (e.g., 'localhost:port') of the server.

        Returns:
            An E2ETestServiceStub instance.
        """
        logging.info(
            f"GenericClientRuntime creating E2ETestServiceStub for target: {target_address}"
        )
        if (
            not hasattr(self, "_GenericClientRuntime__grpc_channel_factory")
            or not self.__grpc_channel_factory
        ):
            raise RuntimeError(
                "GrpcChannelFactory not available in GenericClientRuntime"
            )

        try:
            host, port_str = target_address.rsplit(":", 1)
            port = int(port_str)
        except ValueError:
            # Re-raise with more context if parsing fails, or handle as appropriate
            raise ValueError(
                f"Invalid target_address format for stub creation: {target_address}. Expected 'host:port'."
            )

        channel = await self.__grpc_channel_factory.connect(host, port)
        if not channel:
            raise RuntimeError(
                f"Failed to create channel to {target_address} using host='{host}', port={port}"
            )

        stub = E2ETestServiceStub(channel)
        return stub


class GenericServerRuntimeInitializer(RuntimeInitializer[torch.Tensor, torch.Tensor]):
    def __init__(
        self,
        *,
        listener_factory: Optional[MdnsListenerFactory] = None,
        fake_service_port: Optional[int] = None,
        server_grpc_port: Optional[int] = None,
    ):
        self.__listener_factory = listener_factory
        self.__fake_service_port = fake_service_port
        self.__server_grpc_port = server_grpc_port
        super().__init__(service_type=ServiceType.SERVER)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        actual_mdns_listener_factory: Optional[MdnsListenerFactory] = None
        if self.__fake_service_port is not None:
            logging.info(f"Using FakeMdnsListener for port {self.__fake_service_port}")

            # The service_type_arg in the lambda is what DiscoveryHost would pass to the factory.
            # FakeMdnsListener will use this service_type_arg.
            def fake_factory(
                client: MdnsListener.Client,
                service_type_arg: str,
                zc_instance_arg: Optional[AsyncZeroconf] = None,
            ) -> FakeMdnsListener:
                return FakeMdnsListener(
                    client,
                    service_type_arg,
                    self.__fake_service_port,
                    zc_instance=zc_instance_arg,
                )

            actual_mdns_listener_factory = fake_factory
        else:
            actual_mdns_listener_factory = self.__listener_factory

        return GenericServerRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            mdns_listener_factory=actual_mdns_listener_factory,
            server_grpc_port=self.__server_grpc_port,
        )


class GenericClientRuntimeInitializer(RuntimeInitializer[torch.Tensor, torch.Tensor]):
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
            thread_watcher,
            data_handler,
            self.__name,
            self.__host_port,
            grpc_channel_factory,
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
    client_handle_f = runtime_manager.register_runtime_initializer(client_initializer)
    server_handle_f = runtime_manager.register_runtime_initializer(server_initializer)

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

    client_handle.on_event(torch.zeros(5))
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


# BEGIN E2eTestServicer code block
e2e_servicer_received_messages = []


class E2eTestServicer(E2ETestServiceServicer):
    __test__ = False

    async def Echo(self, request: EchoRequest, context) -> EchoResponse:
        logging.info(f"E2eTestServicer received Echo request: {request.message}")
        e2e_servicer_received_messages.append(request.message)
        return EchoResponse(response=f"Server echoes: {request.message}")

    async def ServerStreamData(self, request: StreamDataRequest, context):
        logging.info(
            f"E2eTestServicer ServerStreamData called with id: {request.data_id}"
        )
        raise grpc.aio.RpcError(
            grpc.StatusCode.UNIMPLEMENTED,
            "ServerStreamData not fully implemented",
        )

    async def ClientStreamData(self, request_iterator, context) -> EchoResponse:
        messages_received_count = 0
        async for req in request_iterator:
            logging.info(
                f"E2eTestServicer ClientStreamData received data_id: {req.data_id}"
            )
            messages_received_count += 1
        return EchoResponse(
            response=f"ClientStreamData received {messages_received_count} messages."
        )

    async def BidirectionalStreamData(self, request_iterator, context):
        logging.info("E2eTestServicer BidirectionalStreamData called")
        async for req in request_iterator:
            logging.info(f"E2eTestServicer (Bidi) consumed data_id: {req.data_id}")
        raise grpc.aio.RpcError(
            grpc.StatusCode.UNIMPLEMENTED,
            "BidirectionalStreamData response generation not implemented",
        )


# END E2eTestServicer code block


@pytest.mark.asyncio
async def test_full_app_with_grpc_transport(clear_loop_fixture, caplog):
    """
    Tests the full application setup using direct gRPC communication between
    GenericClientRuntime and GenericServerRuntime for the E2ETestService.
    """
    caplog.set_level(logging.INFO)
    logging.info("Starting test_full_app_with_grpc_transport")

    SERVER_GRPC_PORT = 50051

    # Initialize Server
    server_initializer = GenericServerRuntimeInitializer(
        fake_service_port=2025,
        server_grpc_port=SERVER_GRPC_PORT,
    )

    # Initialize Client
    client_initializer = GenericClientRuntimeInitializer(
        host_port=2024, name="E2EClient"
    )

    runtime_manager = RuntimeManager(is_testing=True)
    server_handle_f = runtime_manager.register_runtime_initializer(server_initializer)
    client_handle_f = runtime_manager.register_runtime_initializer(client_initializer)

    await runtime_manager.start_in_process_async()
    assert server_handle_f.done() and client_handle_f.done()
    server_runtime_handle = server_handle_f.result()
    client_runtime_handle = client_handle_f.result()

    server_runtime_handle.start()
    client_runtime_handle.start()

    client_runtime_maybe = cast(
        RuntimeWrapper[Any, Any], client_runtime_handle
    )._get_runtime_for_test()
    assert (
        client_runtime_maybe is not None
    ), "Failed to get actual client runtime from handle"
    client_runtime: GenericClientRuntime = client_runtime_maybe  # type: ignore[assignment]
    stub = await client_runtime.create_e2e_test_stub(f"localhost:{SERVER_GRPC_PORT}")

    test_message = "Hello GRPC E2E"
    try:
        response = await stub.Echo(EchoRequest(message=test_message), timeout=5.0)
        logging.info(f"Echo response received: {response.response}")
        assert f"Server echoes: {test_message}" in response.response
        assert test_message in e2e_servicer_received_messages
    except grpc.aio.AioRpcError as e:
        logging.error(f"gRPC call failed: {e.details()} (status: {e.code()})")
        pytest.fail(f"gRPC Echo call failed: {e.details()}")

    async def stream_requests() -> AsyncIterator[StreamDataRequest]:
        for i in range(3):
            yield StreamDataRequest(data_id=i)
            await asyncio.sleep(0.1)

    try:
        response = await stub.ClientStreamData(stream_requests(), timeout=5.0)
        logging.info(f"ClientStreamData response: {response.response}")
        assert "ClientStreamData received 3 messages" in response.response
    except grpc.aio.AioRpcError as e:
        logging.error(
            f"ClientStreamData call failed: {e.details()} (status: {e.code()})"
        )
        pytest.fail(f"ClientStreamData gRPC call failed: {e.details()}")

    # Shutdown
    client_runtime_handle.stop()
    server_runtime_handle.stop()
    await asyncio.sleep(0.5)
    runtime_manager.shutdown()
    logging.info("test_full_app_with_grpc_transport completed.")
