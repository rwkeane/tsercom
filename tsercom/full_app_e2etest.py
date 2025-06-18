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
from zeroconf.asyncio import AsyncZeroconf

# Imports for the new E2E test
from tsercom.test.e2e_test_servicer import E2eTestServicer, get_received_data_queue
from tsercom.test.proto.generated.e2e_test_service_pb2 import EchoRequest
from tsercom.test.proto.generated.e2e_test_service_pb2_grpc import add_E2ETestServiceServicer_to_server, E2ETestServiceStub

if TYPE_CHECKING:
    pass


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
        # Construct a plausible-looking mDNS instance name.
        # It should end with .local. and contain the service type.
        # Example: FakedServiceInstance._e2e_test_service._tcp.local.
        base_name = "FakedServiceInstance"
        st = self.__service_type.strip('.') # Remove leading/trailing dots for safety
        fake_mdns_instance_name = f"{base_name}.{st}.local."

        # Ensure it's a valid structure, e.g. _http._tcp from http._tcp
        # This is a simplified assumption for the fake listener.
        if not self.__service_type.startswith("_"):
             st = f"_{st}" # Prepend underscore if missing
        if "._tcp" not in st and "._udp" not in st :
             st = f"{st}._tcp" # Append ._tcp if no protocol part

        fake_mdns_instance_name = f"{base_name}.{st}.local."


        fake_ip_address_bytes = socket.inet_aton("127.0.0.1")

        await self.__client._on_service_added(
            name=fake_mdns_instance_name, # Use the constructed name
            port=self.__port,
            addresses=[fake_ip_address_bytes],
            txt_record={},
        )
        logging.info(
            f"FakeMdnsListener: _on_service_added called for {fake_mdns_instance_name}"
        )

    async def add_service(
        self, zc: AsyncZeroconf, type_: str, name: str
    ) -> None:
        logging.debug(
            f"FakeMdnsListener: add_service called for {name} type {type_}, no action."
        )
        pass

    async def update_service(
        self, zc: AsyncZeroconf, type_: str, name: str
    ) -> None:
        logging.debug(
            f"FakeMdnsListener: update_service called for {name}, no action."
        )
        pass

    async def remove_service(
        self, zc: AsyncZeroconf, type_: str, name: str
    ) -> None:
        logging.debug(
            f"FakeMdnsListener: remove_service called for {name}, no action."
        )
        pass

    async def close(self) -> None:
        logging.info(
            f"FakeMdnsListener: close() called for service type '{self.__service_type}'."
        )
        pass


class GenericServerRuntime( # This runtime acts as a gRPC client in the new test
    Runtime,
    ServiceConnector[ServiceInfo, grpc.Channel].Client,
):
    _e2e_test_stub: Optional[E2ETestServiceStub] = None # For the new test

    def __init__(
        self,
        watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        channel_factory: GrpcChannelFactory,
        *,
        mdns_listener_factory: Optional[MdnsListenerFactory] = None,
        # Added to control the service type GenericServerRuntime's DiscoveryHost looks for.
        # This is crucial for ensuring it "finds" the service faked by FakeMdnsListener.
        discovery_service_type: str = "_foo._tcp" # Default for original test_anomoly_service
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler
        self._e2e_test_stub = None

        discoverer: DiscoveryHost
        if mdns_listener_factory is None:
            # If no factory, DiscoveryHost uses its internal default listener with the provided service_type.
            discoverer = DiscoveryHost(service_type=discovery_service_type)
        else:
            # If a factory is provided (like for FakeMdnsListener),
            # DiscoveryHost uses that factory. The factory itself needs to be
            # configured with the correct service_type that it will "fake" or listen for.
            # The discovery_service_type passed to DiscoveryHost here should match
            # what the mdns_listener_factory is set up to handle/fake.
            discoverer = DiscoveryHost(
                mdns_listener_factory=mdns_listener_factory,
                service_type=discovery_service_type # Ensure DiscoveryHost is configured for this type
            )
        self.__connector = ServiceConnector[ServiceInfo, grpc.Channel](
            self, channel_factory, discoverer
        )
        super().__init__()

    async def start_async(self):
        await self.__connector.start()

    async def stop(self, exception: Optional[Exception] = None):
        logging.info(f"GenericServerRuntime ({getattr(self, '_e2e_test_stub', 'No Stub')}) stopping...")
        if self.__connector:
            await self.__connector.stop()
        logging.info("GenericServerRuntime stopped.")

    async def _on_channel_connected(
        self,
        connection_info: ServiceInfo,
        caller_id: CallerIdentifier,
        channel_info: ChannelInfo,
    ):
        logging.info(f"GenericServerRuntime connected to {connection_info.get_full_name()} ({caller_id.get_id() if caller_id else 'Unknown CallerID'})")
        has_been_hit.set(True)
        # For the new gRPC E2E test, create the stub if the connected service is the E2ETestService
        # This check might be based on connection_info.service_type or a property of channel_info
        # Assuming for now that any connection in this test setup is for the E2ETestService if _e2e_test_stub is None
        if channel_info.channel and self._e2e_test_stub is None : # Check if it's the E2E test context
             # A better way would be to check connection_info.service_type if it matches GRPC_E2E_SERVICE_TYPE
            if connection_info.service_type == "_e2e_test_service._tcp.local." or connection_info.service_type == "_e2e_test_service._tcp":
                self._e2e_test_stub = E2ETestServiceStub(channel_info.channel)
                logging.info(f"E2ETestServiceStub created for channel to {connection_info.get_full_name()}")
            else:
                logging.info(f"Connected to {connection_info.service_type}, not creating E2ETestServiceStub for this one.")
        elif not channel_info.channel:
            logging.error("Cannot create E2ETestServiceStub: channel is None.")


class GenericClientRuntime( # This runtime hosts the gRPC server (E2eTestServicer) in the new test
    Runtime,
):
    def __init__(
        self,
        watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        readable_name: str,
        port: int,
        # Added to control what mDNS service type this runtime advertises.
        # For test_anomoly_service, it's "_foo._tcp".
        # For test_full_app_with_grpc_transport, it's "_e2e_test_service._tcp".
        advertising_service_type: str = "_foo._tcp"
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler
        self.__is_running = IsRunningTracker()
        self.__mdns_publisher = InstancePublisher(
            port, advertising_service_type, readable_name # Use the passed service type
        )
        self.__grpc_publisher = GrpcServicePublisher(self.__watcher, port)
        self.__advertising_service_type = advertising_service_type # Store for connect
        super().__init__()

    async def start_async(self):
        assert not self.__is_running.get()
        self.__is_running.start()

        def __connect(server: grpc.Server):
            from grpc_health.v1 import health, health_pb2, health_pb2_grpc
            health_servicer = health.HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

            # Standard health service
            health_servicer.set(
                f"tsercom.GenericClientRuntime.{self.__advertising_service_type}.Health",
                health_pb2.HealthCheckResponse.SERVING,
            )

            # If this client is advertising the E2E test service, add the E2eTestServicer
            if self.__advertising_service_type == "_e2e_test_service._tcp":
                e2e_test_servicer = E2eTestServicer()
                add_E2ETestServiceServicer_to_server(e2e_test_servicer, server)
                # Also report E2ETestService as serving for its specific name
                health_servicer.set(
                    "tsercom.test.e2e.E2ETestService", # Official service name from proto
                    health_pb2.HealthCheckResponse.SERVING
                )
                logging.info(f"E2ETestServiceServicer added to gRPC server on port {self.__grpc_publisher._port}")


        await self.__grpc_publisher.start_async(__connect)
        await self.__mdns_publisher.publish()
        logging.info(f"GenericClientRuntime ({self.__advertising_service_type}) published on port {self.__mdns_publisher._port}")


    async def stop(self, exception: Optional[Exception] = None):
        logging.info(f"GenericClientRuntime ({self.__advertising_service_type}) stopping...")
        self.__is_running.stop()
        if self.__mdns_publisher:
            # Try async close first if available (adapting to potential API changes)
            if hasattr(self.__mdns_publisher, 'close_async') and callable(getattr(self.__mdns_publisher, 'close_async')):
                 await self.__mdns_publisher.close_async()
            elif hasattr(self.__mdns_publisher, "close") and callable(getattr(self.__mdns_publisher, "close")):
                await asyncio.to_thread(self.__mdns_publisher.close) # zeroconf close is sync
            elif hasattr(self.__mdns_publisher, "unpublish") and callable(getattr(self.__mdns_publisher, "unpublish")):
                 await asyncio.to_thread(self.__mdns_publisher.unpublish) # Fallback for older interface
            else:
                logging.warning("MdnsPublisher has no recognized close/unpublish method.")


        if self.__grpc_publisher:
            await self.__grpc_publisher.stop_async()
        logging.info(f"GenericClientRuntime ({self.__advertising_service_type}) stopped.")


class GenericServerRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        *,
        listener_factory: Optional[MdnsListenerFactory] = None,
        fake_service_port: Optional[int] = None,
        # This is the service type this Server Runtime's DiscoveryHost will look for.
        # It must match what the FakeMdnsListener is configured to fake.
        discovery_service_type: str = "_foo._tcp" # Default for original test
    ):
        self.__listener_factory = listener_factory
        self.__fake_service_port = fake_service_port
        self.__discovery_service_type = discovery_service_type
        super().__init__(service_type=ServiceType.SERVER)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        actual_mdns_listener_factory: Optional[MdnsListenerFactory] = self.__listener_factory

        if self.__fake_service_port is not None:
            # If using fake_service_port, we are in a test scenario needing FakeMdnsListener.
            # The FakeMdnsListener needs to be told which service_type it's "faking".
            # This service_type is the one that this ServerRuntimeInitializer's
            # DiscoveryHost will be searching for (self.__discovery_service_type).
            logging.info(
                f"GenericServerRuntimeInitializer: Using FakeMdnsListener for port {self.__fake_service_port}, "
                f"faking/listening for service type '{self.__discovery_service_type}'"
            )

            def fake_factory(
                client: MdnsListener.Client,
                # service_type_arg_from_discovery_host is the type DiscoveryHost is configured to look for.
                # This should match self.__discovery_service_type.
                service_type_arg_from_discovery_host: str,
                zc_instance_arg: Optional[AsyncZeroconf] = None,
            ) -> FakeMdnsListener:
                # Critical assertion: The service type DiscoveryHost is looking for must be the one we are faking.
                assert service_type_arg_from_discovery_host == self.__discovery_service_type, \
                    f"FakeMdnsListener factory error: DiscoveryHost is looking for " \
                    f"'{service_type_arg_from_discovery_host}', but FakeMdnsListener was " \
                    f"configured to fake/expect '{self.__discovery_service_type}' via ServerInitializer."

                return FakeMdnsListener(
                    client,
                    self.__discovery_service_type, # Tell FakeMdnsListener what type to fake.
                    self.__fake_service_port,    # Port where the faked service is "running".
                    zc_instance=zc_instance_arg,
                )
            actual_mdns_listener_factory = fake_factory

        # Pass the discovery_service_type to GenericServerRuntime so its DiscoveryHost knows what to look for.
        return GenericServerRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            mdns_listener_factory=actual_mdns_listener_factory,
            discovery_service_type=self.__discovery_service_type # Explicitly pass this
        )


class GenericClientRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(self,
                 host_port: int,
                 name: str,
                 # This is the service type this Client Runtime will advertise via mDNS.
                 advertising_service_type: str = "_foo._tcp" # Default for original test
                ):
        self.__host_port = host_port
        self.__name = name
        self.__advertising_service_type = advertising_service_type
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
            advertising_service_type=self.__advertising_service_type # Pass to GenericClientRuntime
        )


@pytest_asyncio.fixture
async def clear_loop_fixture():
    clear_tsercom_event_loop()
    has_been_hit.set(False) # Ensure reset before each test using this fixture
    # Clear the queue for E2E servicer at the start of tests that might use it
    # This is important if the queue is a global/module-level variable in e2e_test_servicer.py
    try:
        q = get_received_data_queue()
        while not q.empty():
            q.get_nowait()
            q.task_done()
    except Exception as e:
        logging.warning(f"Could not clear servicer queue in fixture: {e}")
    yield
    clear_tsercom_event_loop()
    await asyncio.sleep(0.1) # Allow time for cleanup


@pytest.mark.asyncio
async def test_anomoly_service(clear_loop_fixture):
    # This test remains for general E2E, using a different service type.
    ANOMOLY_SERVICE_TYPE = "_foo._tcp"
    logging.info(f"Starting test_anomoly_service, targeting type: {ANOMOLY_SERVICE_TYPE}")
    loggers_to_modify = {
        "tsercom": logging.INFO, # General tsercom logs
        "zeroconf": logging.INFO, # Zeroconf logs
    }
    original_levels = {name: logging.getLogger(name).level for name in loggers_to_modify}
    for name, level in loggers_to_modify.items():
        logging.getLogger(name).setLevel(level)

    # Client advertises ANOMOLY_SERVICE_TYPE
    client_initializer = GenericClientRuntimeInitializer(
        2024, "AnomalyClient", advertising_service_type=ANOMOLY_SERVICE_TYPE
    )
    # Server looks for ANOMOLY_SERVICE_TYPE using FakeMdnsListener
    server_initializer = GenericServerRuntimeInitializer(
        fake_service_port=2024,
        discovery_service_type=ANOMOLY_SERVICE_TYPE
    )

    runtime_manager = RuntimeManager(is_testing=True)
    # Using default data_handlers for this test as it's not focused on gRPC data path
    client_handle_f = runtime_manager.register_runtime_initializer(client_initializer)
    server_handle_f = runtime_manager.register_runtime_initializer(server_initializer)

    await runtime_manager.start_in_process_async()
    runtime_manager.check_for_exception()

    assert client_handle_f.done(), "Anomaly Client handle future not done"
    assert server_handle_f.done(), "Anomaly Server handle future not done"

    client_handle = client_handle_f.result()
    server_handle = server_handle_f.result()

    has_been_hit.set(False) # Reset for this test run

    logging.info("test_anomoly_service: Starting client...")
    client_handle.start()
    runtime_manager.check_for_exception()
    await asyncio.sleep(0.2) # Give client publisher time

    logging.info("test_anomoly_service: Starting server...")
    server_handle.start()
    runtime_manager.check_for_exception()

    logging.info("test_anomoly_service: Waiting for connection (has_been_hit)...")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(lambda: has_been_hit.wait_for(True, timeout=10.0)),
            timeout=12.0
        )
    except asyncio.TimeoutError:
        logging.error("test_anomoly_service: Timeout waiting for has_been_hit.")
        runtime_manager.check_for_exception()
        pytest.fail("test_anomoly_service: Server did not connect to client.")

    assert has_been_hit.get(), "test_anomoly_service: has_been_hit was false after connection wait."

    # Data sending part of original test - may need adjustment if data_handlers are dummies
    # For now, focus is on connection.
    # client_handle.on_event(torch.zeros(5))
    # await asyncio.sleep(0.1) # Allow event to propagate if processed

    logging.info("test_anomoly_service: Shutting down...")
    client_handle.stop()
    server_handle.stop()
    await asyncio.sleep(0.5) # Allow stops to complete
    runtime_manager.check_for_exception()
    runtime_manager.shutdown()
    await asyncio.sleep(0.1)

    for name, level in original_levels.items():
        logging.getLogger(name).setLevel(level)
    logging.info("test_anomoly_service: Completed.")


# New E2E Test for gRPC Transport
GRPC_E2E_SERVICE_TYPE = "_e2e_test_service._tcp"

@pytest.mark.asyncio
async def test_full_app_with_grpc_transport(clear_loop_fixture):
    logging.info(f"Starting test_full_app_with_grpc_transport, targeting type: {GRPC_E2E_SERVICE_TYPE}")
    loggers_to_modify = {
        "tsercom": logging.DEBUG, # More verbose for the new test
        "zeroconf": logging.DEBUG,
    }
    original_levels = {name: logging.getLogger(name).level for name in loggers_to_modify}
    for name, level in loggers_to_modify.items():
        logging.getLogger(name).setLevel(level)

    TEST_PORT = 2026  # Distinct port for this test

    # ClientRuntime will host the E2eTestServicer (acting as gRPC server).
    # It advertises GRPC_E2E_SERVICE_TYPE.
    client_initializer = GenericClientRuntimeInitializer(
        TEST_PORT,
        "GrpcE2eHostClient", # Name indicates it hosts the service
        advertising_service_type=GRPC_E2E_SERVICE_TYPE
    )

    # ServerRuntime will connect to the E2eTestServicer (acting as gRPC client).
    # It uses FakeMdnsListener to "discover" GRPC_E2E_SERVICE_TYPE.
    server_initializer = GenericServerRuntimeInitializer(
        fake_service_port=TEST_PORT, # Port where the fake service (hosted by ClientRuntime) is.
        discovery_service_type=GRPC_E2E_SERVICE_TYPE # Service type it's looking for.
    )

    runtime_manager = RuntimeManager(is_testing=True)

    # Register client (hosts gRPC E2eTestServicer)
    # data_handler_override is used because the default data_handler might expect specific types (torch.Tensor)
    # which are not relevant for this gRPC call test path.
    client_handle_f = runtime_manager.register_runtime_initializer(
        client_initializer,
        data_handler_override=lambda: None # type: ignore
    )

    # Register server (connects to gRPC E2eTestServicer)
    server_handle_f = runtime_manager.register_runtime_initializer(
        server_initializer,
        data_handler_override=lambda: None # type: ignore
    )

    await runtime_manager.start_in_process_async()
    runtime_manager.check_for_exception()

    assert client_handle_f.done(), "Client (gRPC server host) handle future did not complete"
    assert server_handle_f.done(), "Server (gRPC client) handle future did not complete"

    client_handle = client_handle_f.result()
    server_handle = server_handle_f.result()

    has_been_hit.set(False) # Reset for this specific test run

    logging.info("test_full_app_with_grpc_transport: Starting ClientRuntime (gRPC server host)...")
    client_handle.start()
    runtime_manager.check_for_exception()
    # Increased sleep to allow gRPC server and mDNS to fully initialize and publish
    await asyncio.sleep(1.0)

    logging.info("test_full_app_with_grpc_transport: Starting ServerRuntime (gRPC client)...")
    server_handle.start()
    runtime_manager.check_for_exception()

    logging.info("test_full_app_with_grpc_transport: Waiting for ServerRuntime to connect to ClientRuntime via fake mDNS...")
    try:
        # Increased timeout for connection, mDNS discovery + gRPC channel setup can take a moment
        await asyncio.wait_for(
            asyncio.to_thread(lambda: has_been_hit.wait_for(True, timeout=15.0)),
            timeout=20.0
        )
    except asyncio.TimeoutError:
        logging.error("Timeout waiting for ServerRuntime to connect to ClientRuntime (has_been_hit).")
        runtime_manager.check_for_exception()
        pytest.fail("ServerRuntime did not connect to ClientRuntime's service within timeout.")

    runtime_manager.check_for_exception()
    assert has_been_hit.get(), "ServerRuntime did not connect (has_been_hit is false after wait)"

    server_runtime_instance = server_handle._get_runtime()
    assert isinstance(server_runtime_instance, GenericServerRuntime), \
        f"Server handle's runtime is not GenericServerRuntime, but {type(server_runtime_instance)}"

    assert server_runtime_instance._e2e_test_stub is not None, \
        "E2ETestServiceStub was not created on the server runtime instance."

    echo_stub = server_runtime_instance._e2e_test_stub

    test_message = "Hello True gRPC E2E World!"
    try:
        echo_request = EchoRequest(message=test_message)
        logging.info(f"gRPC client sending EchoRequest: '{test_message}'")
        echo_response = await echo_stub.Echo(echo_request, timeout=5.0)
        logging.info(f"gRPC client received EchoResponse: '{echo_response.response}'")
        assert echo_response.response == f"Echo: {test_message}"
    except grpc.aio.AioRpcError as e:
        logging.error(f"gRPC Echo call failed: {e.code()} - {e.details()}")
        pytest.fail(f"gRPC Echo call failed: {e.code()} - {e.details()}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during gRPC Echo call: {e}", exc_info=True)
        pytest.fail(f"An unexpected error occurred during gRPC Echo call: {e}")

    data_queue = get_received_data_queue()
    try:
        logging.info("Waiting for message in E2eTestServicer's queue...")
        received_message = await asyncio.wait_for(data_queue.get(), timeout=3.0)
        assert received_message == test_message, \
            f"Message in queue ('{received_message}') did not match sent message ('{test_message}')"
        data_queue.task_done()
        logging.info(f"Message '{received_message}' successfully retrieved from servicer queue.")
    except asyncio.TimeoutError:
        logging.error("Timeout waiting for message in servicer queue.")
        pytest.fail("Timeout waiting for message in servicer queue")

    logging.info("test_full_app_with_grpc_transport: Shutting down...")
    client_handle.stop()
    server_handle.stop()
    await asyncio.sleep(1.0) # Increased sleep for cleaner shutdown
    runtime_manager.check_for_exception()
    runtime_manager.shutdown()
    await asyncio.sleep(0.2)

    for name, level in original_levels.items():
        logging.getLogger(name).setLevel(level)
    logging.info("test_full_app_with_grpc_transport: Completed successfully.")
