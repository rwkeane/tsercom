import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

import grpc
import pytest
import pytest_asyncio
import socket

from tsercom.api import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.discovery_host import DiscoveryHost
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
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)

if TYPE_CHECKING:
    pass  # This can remain for type checking if it's used elsewhere, or be removed if AsyncZeroconf replaces all uses.

    # For FakeMdnsListener methods, we'll use AsyncZeroconf.

from tsercom.test.proto.generated.v1_73 import e2e_test_service_pb2
from tsercom.test.proto.generated.v1_73 import e2e_test_service_pb2_grpc


class E2ETestServicer(e2e_test_service_pb2_grpc.E2ETestServiceServicer):
    async def Echo(
        self,
        request: e2e_test_service_pb2.EchoRequest,
        context: grpc.aio.ServicerContext,
    ) -> e2e_test_service_pb2.EchoResponse:
        logging.info(
            f"E2ETestServicer received Echo request: {request.message}"
        )
        return e2e_test_service_pb2.EchoResponse(response=request.message)


class E2ETestClientRuntime(
    Runtime,
    ServiceConnector[ServiceInfo, grpc.aio.Channel].Client,
):
    """
    Top-level class for the E2E test gRPC client.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        channel_factory: GrpcChannelFactory,
    ):
        self.__watcher = watcher
        self.__connected_channel: Optional[grpc.aio.Channel] = None
        self.__channel_connected_event = asyncio.Event()

        discoverer: DiscoveryHost = DiscoveryHost(
            service_type="_e2e-test._tcp.local."
        )
        self.__connector = ServiceConnector[ServiceInfo, grpc.aio.Channel](
            self, channel_factory, discoverer
        )
        super().__init__()

    async def start_async(self):
        """
        Allow for connections with servers to start.
        """
        await self.__connector.start()

    async def stop(self, exception: Optional[Exception] = None):
        logging.info("E2ETestClientRuntime stopping...")
        if self.__connector:
            await self.__connector.stop()
        if self.__connected_channel:
            await self.__connected_channel.close()
        logging.info("E2ETestClientRuntime stopped.")

    async def _on_channel_connected(
        self,
        connection_info: ServiceInfo,
        caller_id: CallerIdentifier,
        channel: grpc.aio.Channel,
    ):
        logging.info(
            f"E2ETestClientRuntime: Channel connected to {caller_id} ({connection_info.name})"
        )
        self.__connected_channel = channel
        self.__channel_connected_event.set()

    async def wait_for_channel_ready(
        self, timeout: float = 10.0
    ) -> grpc.aio.Channel:
        await asyncio.wait_for(
            self.__channel_connected_event.wait(), timeout=timeout
        )
        if not self.__connected_channel:
            raise RuntimeError(
                "Channel connected event was set, but channel is None."
            )
        return self.__connected_channel

    def get_connected_channel(self) -> Optional[grpc.aio.Channel]:
        return self.__connected_channel


class E2ETestServerRuntime(
    Runtime,
):
    """
    This is the top-level class for the E2E test gRPC server.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        readable_name: str,
    ):
        self.__watcher = watcher

        self.__is_running = IsRunningTracker()
        self.__mdns_publiser = InstancePublisher(
            port,
            "_e2e-test._tcp.local.",
            readable_name,
        )
        self.__grpc_publisher = GrpcServicePublisher(self.__watcher, port)

        super().__init__()

    async def start_async(self):
        """
        Starts the gRPC server and advertises it over mDNS.
        """
        assert not self.__is_running.get()
        self.__is_running.start()

        def __connect(server: grpc.Server):
            e2e_test_service_pb2_grpc.add_E2ETestServiceServicer_to_server(
                E2ETestServicer(), server
            )

            # Add health servicer
            from grpc_health.v1 import health
            from grpc_health.v1 import health_pb2
            from grpc_health.v1 import health_pb2_grpc

            health_servicer = health.HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(
                health_servicer, server
            )
            health_servicer.set(
                "E2ETestService",
                health_pb2.HealthCheckResponse.SERVING,
            )

        await self.__grpc_publisher.start_async(__connect)

        await self.__mdns_publiser.publish()

    async def stop(self, exception: Optional[Exception] = None):
        logging.info("E2ETestServerRuntime stopping...")
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
        logging.info("E2ETestServerRuntime stopped.")


class E2ETestClientRuntimeInitializer(RuntimeInitializer[Any, Any]):
    def __init__(self):
        super().__init__(service_type=ServiceType.CLIENT)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[Any, Any],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        # data_handler is accepted to match base signature but not used.
        actual_channel_factory = InsecureGrpcChannelFactory()

        return E2ETestClientRuntime(
            thread_watcher,
            actual_channel_factory,
        )


class E2ETestServerRuntimeInitializer(RuntimeInitializer[Any, Any]):
    def __init__(self, host_port: int, service_name: str):
        self.__host_port = host_port
        self.__service_name = service_name

        super().__init__(service_type=ServiceType.CLIENT)

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[Any, Any],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        # data_handler is accepted to match base signature but not used.
        return E2ETestServerRuntime(
            thread_watcher,
            self.__host_port,
            self.__service_name,
        )


@pytest_asyncio.fixture
async def clear_loop_fixture():
    clear_tsercom_event_loop()
    yield
    clear_tsercom_event_loop()
    await asyncio.sleep(0.1)


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


@pytest.mark.asyncio
async def test_e2e_echo_service(clear_loop_fixture):
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

    server_port = get_free_port()
    logging.info(f"E2E Test Server will use port: {server_port}")

    e2e_test_server_initializer = E2ETestServerRuntimeInitializer(
        server_port, "TestServerInstance"
    )
    e2e_test_client_initializer = E2ETestClientRuntimeInitializer()

    runtime_manager = RuntimeManager(is_testing=True)
    e2e_test_server_handle_f = runtime_manager.register_runtime_initializer(
        e2e_test_server_initializer
    )
    e2e_test_client_handle_f = runtime_manager.register_runtime_initializer(
        e2e_test_client_initializer
    )

    await runtime_manager.start_in_process_async()
    runtime_manager.check_for_exception()

    assert e2e_test_server_handle_f.done()
    assert e2e_test_client_handle_f.done()

    e2e_test_server_handle = e2e_test_server_handle_f.result()
    e2e_test_client_handle = e2e_test_client_handle_f.result()

    e2e_test_server_handle.start()
    runtime_manager.check_for_exception()

    await asyncio.sleep(1.0)

    e2e_test_client_handle.start()
    runtime_manager.check_for_exception()

    e2e_client_runtime = e2e_test_client_handle.get_runtime()
    assert isinstance(e2e_client_runtime, E2ETestClientRuntime)

    try:
        logging.info("Waiting for channel to be ready...")
        channel = await e2e_client_runtime.wait_for_channel_ready(
            timeout=15.0
        )  # Timeout reduced
        assert channel is not None
        logging.info(
            f"E2E test client successfully connected to the server. Channel: {channel}"
        )  # Simplified logging
    except asyncio.TimeoutError:
        logging.error(
            "E2E test client timed out waiting to connect to the server."
        )
        # Accessing private members like this is for debugging only.
        discoverer_instance = getattr(
            e2e_client_runtime, "_E2ETestClientRuntime__connector", None
        )
        if discoverer_instance:
            discoverer_instance = getattr(
                discoverer_instance, "_ServiceConnector__discoverer", None
            )
        if (
            discoverer_instance
            and hasattr(discoverer_instance, "_zeroconf_instance")
            and getattr(discoverer_instance, "_zeroconf_instance", None)
            is not None
        ):
            zc = getattr(discoverer_instance, "_zeroconf_instance")
            if (
                zc
                and hasattr(zc, "async_protocol_handler")
                and zc.async_protocol_handler
            ):
                logging.error(
                    f"Zeroconf browser cache: {zc.async_protocol_handler.cache}"
                )
            else:
                logging.error(
                    "Zeroconf instance or protocol handler not available for detailed cache inspection."
                )
        else:
            logging.error(
                "Could not retrieve Zeroconf instance for cache inspection."
            )

        pytest.fail(
            "E2E test client timed out waiting to connect to the server."
        )

    stub = e2e_test_service_pb2_grpc.E2ETestServiceStub(channel)
    request_message = "Hello, gRPC E2E!"
    request = e2e_test_service_pb2.EchoRequest(message=request_message)

    logging.info(f"Sending Echo request: {request_message}")
    response = await stub.Echo(request)
    logging.info(f"Received Echo response: {response.response}")

    assert response.response == request_message

    runtime_manager.check_for_exception()
    e2e_test_server_handle.stop()
    e2e_test_client_handle.stop()
    await asyncio.sleep(0.5)
    runtime_manager.check_for_exception()
    runtime_manager.shutdown()

    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)
