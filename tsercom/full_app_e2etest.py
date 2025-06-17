import asyncio
import logging
from typing import Optional, TYPE_CHECKING, Union, Any, AsyncIterator
from asyncio import Queue

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
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_config import ServiceType
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.global_event_loop import clear_tsercom_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker
from tsercom.discovery.mdns.mdns_listener import MdnsListener
from zeroconf import Zeroconf
from zeroconf.asyncio import AsyncZeroconf

from tsercom.test.proto.generated.v1_73 import e2e_test_service_pb2
from tsercom.test.proto.generated.v1_73 import e2e_test_service_pb2_grpc

E2E_TEST_SERVICE_TYPE = "_e2etest._tcp.local."


if TYPE_CHECKING:
    pass


# has_been_hit = Atomic[bool](False) # This was replaced by queue logic


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

        await self.__client._on_service_added(
            name=fake_mdns_instance_name,
            port=self.__port,
            addresses=[fake_ip_address_bytes],
            txt_record={},
        )
        logging.info(
            f"FakeMdnsListener: _on_service_added called for {fake_mdns_instance_name}"
        )

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        # This FakeMdnsListener does not interact with a live Zeroconf instance for these.
        logging.debug(
            f"FakeMdnsListener: add_service called for {name} type {type_}, no action."
        )
        pass

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        logging.debug(
            f"FakeMdnsListener: update_service called for {name}, no action."
        )
        pass

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
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
    ServiceConnector.Client,
):
    """
    Test Server Runtime that acts as an RPC client to the E2ETestService.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        channel_factory: GrpcChannelFactory,
        rpc_result_queue: Queue[
            Union[e2e_test_service_pb2.EchoResponse, Exception]
        ],
        *,
        mdns_listener_factory: Optional[MdnsListenerFactory] = None,
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler
        self.__rpc_result_queue = rpc_result_queue

        # This service_type and mdns_listener_factory are part of the original test structure
        # that uses FakeMdnsListener. The E2E_TEST_SERVICE_TYPE is used by the actual
        # client runtime (GenericClientRuntime) for publishing.
        # The FakeMdnsListener setup in GenericServerRuntimeInitializer ensures this server
        # runtime connects to the correct port where GenericClientRuntime is serving.
        _service_type_for_discovery = (
            "_foo._tcp"  # Default, overridden by fake listener if used
        )
        if mdns_listener_factory is None:
            discoverer: DiscoveryHost[ServiceInfo] = DiscoveryHost[
                ServiceInfo
            ](service_type=_service_type_for_discovery)
        else:
            discoverer = DiscoveryHost[ServiceInfo](
                mdns_listener_factory=mdns_listener_factory
            )

        self.__connector = ServiceConnector[ServiceInfo, grpc.aio.Channel](
            self, channel_factory, discoverer
        )

        super().__init__()

    async def start_async(self) -> None:
        """
        Allow for connections with clients to start.
        """
        await self.__connector.start()

    async def stop(self, exception: Optional[Exception] = None) -> None:
        logging.info("GenericServerRuntime stopping...")
        if self.__connector:
            await self.__connector.stop()
        # Runtime.stop is synchronous, so no super().stop(exception) if it were async.
        logging.info("GenericServerRuntime stopped.")

    async def _on_channel_connected(
        self,
        connection_info: ServiceInfo,
        caller_id: CallerIdentifier,
        channel: grpc.aio.Channel,
    ) -> None:
        echo_message = "Hello from E2E test"
        request = e2e_test_service_pb2.EchoRequest(message=echo_message)

        stub = e2e_test_service_pb2_grpc.E2ETestServiceStub(channel)  # type: ignore[no-untyped-call]

        # Make the RPC call. Exceptions will propagate to the caller (ServiceConnector).
        # If ServiceConnector doesn't handle them and put them on the queue,
        # the test's queue.get() might timeout or get an unexpected item if an
        # error handler in ServiceConnector puts something else.
        # Current test logic assumes successful response or timeout on queue.get().
        response = await stub.Echo(request, timeout=10)

        await self.__rpc_result_queue.put(response)


class E2ETestServiceServicer(e2e_test_service_pb2_grpc.E2ETestServiceServicer):
    """
    Servicer for the E2E test service.
    """

    async def Echo(  # Changed to async def
        self,
        request: e2e_test_service_pb2.EchoRequest,
        context: grpc.aio.ServicerContext,  # Changed to grpc.aio.ServicerContext
    ) -> e2e_test_service_pb2.EchoResponse:
        """
        Handles the Echo RPC.
        """
        logging.info(
            f"E2ETestServiceServicer.Echo called with message: {request.message}"
        )
        # No actual async operations needed for simple echo, but signature matches aio server expectations
        return e2e_test_service_pb2.EchoResponse(response=request.message)


class GenericClientRuntime(
    Runtime,
):
    """
    Test Client Runtime that hosts the E2ETestService.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        readable_name: str,
        port: int,
        e2e_test_servicer: "E2ETestServiceServicer",
    ):
        self.__watcher = watcher
        self.__data_handler = data_handler
        self.__e2e_test_servicer = e2e_test_servicer

        self.__is_running = IsRunningTracker()
        self.__mdns_publiser = InstancePublisher(
            port, E2E_TEST_SERVICE_TYPE, readable_name
        )
        self.__grpc_publisher = GrpcServicePublisher(self.__watcher, port)

        super().__init__()

    async def start_async(self) -> None:
        """
        Starts advertising the service and the gRPC server.
        """
        assert not self.__is_running.get()
        self.__is_running.start()

        def __connect(server: grpc.aio.Server) -> None:
            # Commented out health servicer as it's not core to this E2E test's primary validation path.
            # from grpc_health.v1 import health # type: ignore[import-untyped]
            # from grpc_health.v1 import health_pb2
            # from grpc_health.v1 import health_pb2_grpc
            # health_servicer = health.HealthServicer()
            # health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
            # health_servicer.set("tsercom.GenericClientRuntime", health_pb2.HealthCheckResponse.SERVING)

            logging.info(
                f"GenericClientRuntime.__connect: Adding servicer {type(self.__e2e_test_servicer)} to server {server}"
            )
            e2e_test_service_pb2_grpc.add_E2ETestServiceServicer_to_server(  # type: ignore[no-untyped-call]
                self.__e2e_test_servicer, server
            )
            logging.info("GenericClientRuntime.__connect: Servicer added.")

        await self.__grpc_publisher.start_async(__connect)
        await self.__mdns_publiser.publish()

    async def stop(self, exception: Optional[Exception] = None) -> None:
        logging.info("GenericClientRuntime stopping...")
        self.__is_running.stop()
        if self.__mdns_publiser:
            if hasattr(self.__mdns_publiser, "close") and callable(
                getattr(self.__mdns_publiser, "close")
            ):
                await self.__mdns_publiser.close()
            elif hasattr(self.__mdns_publiser, "unpublish") and callable(
                getattr(self.__mdns_publiser, "unpublish")
            ):
                pass

        if self.__grpc_publisher:
            await self.__grpc_publisher.stop_async()
        logging.info("GenericClientRuntime stopped.")


class GenericServerRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        rpc_result_queue: Queue[
            Union[e2e_test_service_pb2.EchoResponse, Exception]
        ],
        *,
        listener_factory: Optional[MdnsListenerFactory] = None,
        fake_service_port: Optional[int] = None,
    ):
        self.__rpc_result_queue = rpc_result_queue
        self.__listener_factory = listener_factory
        self.__fake_service_port = fake_service_port
        super().__init__(service_type=ServiceType.SERVER)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        # This initializer uses FakeMdnsListener to ensure the GenericServerRuntime
        # connects to a known port (where GenericClientRuntime is serving).
        # The mdns_listener_factory argument will be removed in a later refactor
        # once service type matching (e.g., E2E_TEST_SERVICE_TYPE) is fully used
        # for discovery by GenericServerRuntime as well.
        actual_mdns_listener_factory: Optional[MdnsListenerFactory] = (
            self.__listener_factory
        )
        if self.__fake_service_port is not None:
            logging.info(
                f"Using FakeMdnsListener for port {self.__fake_service_port}"
            )
            actual_fake_port: int = self.__fake_service_port

            def fake_factory(
                client: MdnsListener.Client,
                service_type_arg: str,  # Provided by DiscoveryHost
                zc_instance_arg: Optional[AsyncZeroconf] = None,
            ) -> FakeMdnsListener:
                return FakeMdnsListener(
                    client,
                    service_type_arg,  # InstanceListener will use its default here
                    actual_fake_port,
                    zc_instance=zc_instance_arg,
                )

            actual_mdns_listener_factory = fake_factory

        return GenericServerRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self.__rpc_result_queue,
            mdns_listener_factory=actual_mdns_listener_factory,
        )


class GenericClientRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        host_port: int,
        name: str,
        e2e_test_servicer: "E2ETestServiceServicer",
    ):
        self.__host_port = host_port
        self.__name = name
        self.__e2e_test_servicer = e2e_test_servicer
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
            e2e_test_servicer=self.__e2e_test_servicer,  # Pass the servicer
        )


@pytest_asyncio.fixture
async def clear_loop_fixture() -> AsyncIterator[None]:
    clear_tsercom_event_loop()
    yield None
    clear_tsercom_event_loop()
    await asyncio.sleep(0.1)  # Short delay to allow loop cleanup tasks


@pytest.mark.asyncio
async def test_anomoly_service(clear_loop_fixture: Any) -> None:
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

    rpc_result_queue: Queue[
        Union[e2e_test_service_pb2.EchoResponse, Exception]
    ] = Queue()
    expected_sent_message = "Hello from E2E test"

    e2e_servicer = E2ETestServiceServicer()
    client_initializer = GenericClientRuntimeInitializer(
        2024, "E2EClient", e2e_test_servicer=e2e_servicer
    )

    server_initializer = GenericServerRuntimeInitializer(
        rpc_result_queue=rpc_result_queue,
        fake_service_port=2024,
        listener_factory=None,
    )

    runtime_manager: RuntimeManager[Any, Any] = RuntimeManager(is_testing=True)
    client_handle_f = runtime_manager.register_runtime_initializer(
        client_initializer
    )
    server_handle_f = runtime_manager.register_runtime_initializer(
        server_initializer
    )

    # No custom event loop management needed here; RuntimeManager handles it.

    await runtime_manager.start_in_process_async()
    runtime_manager.check_for_exception()

    assert client_handle_f.done()
    assert server_handle_f.done()

    client_handle = client_handle_f.result()
    server_handle = server_handle_f.result()

    client_handle.start()
    runtime_manager.check_for_exception()
    server_handle.start()
    runtime_manager.check_for_exception()

    rpc_outcome = await asyncio.wait_for(rpc_result_queue.get(), timeout=15)

    assert isinstance(
        rpc_outcome, e2e_test_service_pb2.EchoResponse
    ), f"Expected EchoResponse from queue, got {type(rpc_outcome)}. Outcome: {rpc_outcome}"

    rpc_response = rpc_outcome

    assert (
        rpc_response.response == expected_sent_message
    ), f"Echo response '{rpc_response.response}' does not match sent message '{expected_sent_message}'"

    runtime_manager.check_for_exception()
    client_handle.stop()
    server_handle.stop()
    await asyncio.sleep(0.5)
    runtime_manager.check_for_exception()
    runtime_manager.shutdown()

    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)
