from tsercom.test.proto import (
    E2ETestServiceStub,
    add_E2ETestServiceServicer_to_server,
    EchoRequest,
    EchoResponse,
    StreamDataRequest,
    StreamDataResponse,  # Added back / ensured
    E2ETestServiceServicer,
    E2EStreamRequest,
    E2EStreamResponse,
)
from tsercom.caller_id.proto import CallerId
from tsercom.tensor.proto import TensorChunk


import asyncio
import logging
from typing import (
    Optional,
    TYPE_CHECKING,
    AsyncIterator,
    cast,
    Any,
    List,
)  # Added List, Dict

# For queues and events
from asyncio import Queue
import uuid  # For unique IDs

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
from tsercom.rpc.grpc_util.channel_info import ChannelInfo  # Renamed
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.runtime.runtime import Runtime

# from tsercom.runtime.runtime_identity import RuntimeIdentity # Removed due to missing file
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
        channel_info: ChannelInfo,  # Renamed
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
e2e_servicer_received_messages: List[str] = []  # For existing Echo test

# For the new ExchangeData RPC test communication
# These queues will be instantiated by the test and passed to the servicer.
# This allows the test to retrieve data received/sent by the servicer.
# Note: Optional typing here because they are module-level; actual instances will be Queues.
server_received_caller_id_q: Optional[Queue[CallerId]] = None
server_received_data_q: Optional[Queue[TensorChunk]] = None
server_sent_data_q: Optional[Queue[TensorChunk]] = None


class E2eTestServicer(E2ETestServiceServicer):
    __test__ = False  # Prevent pytest from collecting this class

    def __init__(
        self,
        received_caller_id_q: Optional[Queue[CallerId]] = None,
        received_data_q: Optional[Queue[TensorChunk]] = None,
        sent_data_q: Optional[Queue[TensorChunk]] = None,
        server_id_str: Optional[str] = None,
    ):
        self._received_caller_id_q = received_caller_id_q
        self._received_data_q = received_data_q
        self._sent_data_q = sent_data_q
        self._server_id_full = server_id_str or str(uuid.uuid4())
        self._server_id_short = self._server_id_full[:8]  # Consistent short ID
        logging.info(
            f"E2eTestServicer instance created with ID: {self._server_id_short} (full: {self._server_id_full})"
        )

    async def Echo(
        self, request: EchoRequest, context: grpc.aio.ServicerContext
    ) -> EchoResponse:
        logging.info(
            f"E2eTestServicer ({self._server_id_short}) received Echo request: {request.message}"
        )
        e2e_servicer_received_messages.append(request.message)
        # Reverted to original Echo response format for compatibility with existing tests
        return EchoResponse(response=f"Server echoes: {request.message}")

    async def ServerStreamData(
        self, request: StreamDataRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[StreamDataResponse]:
        logging.info(
            f"E2eTestServicer ({self._server_id_short}) ServerStreamData called with id: {request.data_id}"
        )
        # Example implementation: stream a few responses
        for i in range(3):
            yield StreamDataResponse(
                data_chunk=f"Chunk {i} for id {request.data_id}", sequence_number=i
            )
            await asyncio.sleep(0.1)

    async def ClientStreamData(
        self,
        request_iterator: AsyncIterator[StreamDataRequest],
        context: grpc.aio.ServicerContext,
    ) -> EchoResponse:
        messages_received_count = 0
        async for req in request_iterator:
            logging.info(
                f"E2eTestServicer ({self._server_id_short}) ClientStreamData received data_id: {req.data_id}"
            )
            messages_received_count += 1
        # Reverted to original ClientStreamData response format for compatibility
        return EchoResponse(
            response=f"ClientStreamData received {messages_received_count} messages."
        )

    async def BidirectionalStreamData(
        self,
        request_iterator: AsyncIterator[StreamDataRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[StreamDataResponse]:
        logging.info(
            f"E2eTestServicer ({self._server_id_short}) BidirectionalStreamData called by {context.peer()}"
        )
        async for req in request_iterator:
            logging.info(
                f"E2eTestServicer ({self._server_id_short}) (Bidi) consumed data_id: {req.data_id} from {context.peer()}"
            )
            yield StreamDataResponse(
                data_chunk=f"Server ack for {req.data_id}", sequence_number=req.data_id
            )
        # Original code raised UNIMPLEMENTED. This is a basic implementation.
        # If this RPC is not part of the current task, it can remain as raise Unimplemented.
        # For now, providing a minimal implementation.
        # raise grpc.aio.RpcError(
        #     grpc.StatusCode.UNIMPLEMENTED,
        #     "BidirectionalStreamData response generation not implemented",
        # )

    async def ExchangeData(
        self,
        request_iterator: AsyncIterator[E2EStreamRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[E2EStreamResponse]:
        client_peer = context.peer()
        logging.info(
            f"E2eTestServicer ({self._server_id_short}): ExchangeData called by client: {client_peer}"
        )
        client_caller_id_obj: Optional[CallerId] = None

        try:
            async for request in request_iterator:
                if request.HasField("caller_id"):
                    client_caller_id_obj = request.caller_id
                    logging.info(
                        f"E2eTestServicer ({self._server_id_short}) received CallerId: {client_caller_id_obj.id} from {client_peer}"
                    )
                    if self._received_caller_id_q:
                        await self._received_caller_id_q.put(client_caller_id_obj)

                    yield E2EStreamResponse(
                        ack_message=f"CallerId {client_caller_id_obj.id} received by server {self._server_id_short}"
                    )

                elif request.HasField("data_chunk"):
                    chunk = request.data_chunk
                    logging.info(
                        f"E2eTestServicer ({self._server_id_short}) received data_chunk (idx: {chunk.starting_index}) from {client_peer}"
                    )
                    if self._received_data_q:
                        await self._received_data_q.put(chunk)

                    yield E2EStreamResponse(
                        ack_message=f"Data chunk {chunk.starting_index} processed by server {self._server_id_short}"
                    )

                    # Server sends its own data chunk back for bidirectional test
                    server_response_chunk = TensorChunk(
                        starting_index=chunk.starting_index + 1000,
                        data_bytes=f"server_data_for_{chunk.starting_index}".encode(
                            "utf-8"
                        ),
                    )
                    logging.info(
                        f"E2eTestServicer ({self._server_id_short}) sending data_chunk (idx: {server_response_chunk.starting_index}) to {client_peer}"
                    )
                    if self._sent_data_q:
                        await self._sent_data_q.put(server_response_chunk)
                    yield E2EStreamResponse(data_chunk=server_response_chunk)
                else:
                    logging.warning(
                        f"E2eTestServicer ({self._server_id_short}) received an empty E2EStreamRequest payload from {client_peer}"
                    )
                    yield E2EStreamResponse(ack_message="Empty payload received")

        except grpc.aio.AioCancelledError:
            logging.info(
                f"E2eTestServicer ({self._server_id_short}) ExchangeData stream cancelled by client: {client_peer}"
            )
        except Exception as e:
            logging.error(
                f"Error in E2eTestServicer ({self._server_id_short}) ExchangeData for {client_peer}: {e}",
                exc_info=True,
            )
            raise
        finally:
            logging.info(
                f"E2eTestServicer ({self._server_id_short}) ExchangeData stream with {client_peer} finished."
            )


# END E2eTestServicer code block


class TestDataSourceRuntime(Runtime):
    __test__ = False  # Prevent pytest collection
    """
    Hosts the E2ETestService for other runtimes to connect to.
    Simulates a data source advertising itself.
    """

    SERVICE_NAME = "_e2etestsource._tcp"

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        servicer_received_caller_id_q: Queue[CallerId],
        servicer_received_data_q: Queue[TensorChunk],
        servicer_sent_data_q: Queue[TensorChunk],
    ):
        super().__init__()
        self._watcher = watcher
        self._port = port
        self._identity_full = "test-data-source-" + str(uuid.uuid4())
        self._identity_short = self._identity_full[:8]
        logging.info(
            f"TestDataSourceRuntime initializing with ID {self._identity_short} on port {self._port}"
        )

        self._grpc_publisher = GrpcServicePublisher(self._watcher, self._port)
        self._servicer = E2eTestServicer(
            received_caller_id_q=servicer_received_caller_id_q,
            received_data_q=servicer_received_data_q,
            sent_data_q=servicer_sent_data_q,
            server_id_str=self._identity_full,
        )
        self._instance_publisher = InstancePublisher(
            port=self._port,
            service_type=self.SERVICE_NAME,
            instance_name=f"TestSourceInstance-{self._identity_short}",
        )
        self._is_running = IsRunningTracker()

    async def start_async(self):
        if self._is_running.get():
            logging.warning(
                f"TestDataSourceRuntime {self._identity_short} already started."
            )
            return
        self._is_running.start()
        logging.info(f"TestDataSourceRuntime {self._identity_short} starting...")

        def configure_server(server: grpc.aio.Server):
            add_E2ETestServiceServicer_to_server(self._servicer, server)
            health_servicer = health.HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
            health_servicer.set(
                "tsercom.test.E2ETestService",
                health_pb2.HealthCheckResponse.SERVING,
            )

        await self._grpc_publisher.start_async(configure_server)
        logging.info(
            f"TestDataSourceRuntime {self._identity_short} gRPC server started on port {self._port}."
        )

        await self._instance_publisher.publish()
        logging.info(
            f"TestDataSourceRuntime {self._identity_short} published service {self.SERVICE_NAME}."
        )

    async def stop(self, exception: Optional[Exception] = None):
        if not self._is_running.get_and_set(False):
            logging.warning(
                f"TestDataSourceRuntime {self._identity_short} already stopped or not started."
            )
            return

        logging.info(f"TestDataSourceRuntime {self._identity_short} stopping...")
        if self._instance_publisher:
            if hasattr(
                self._instance_publisher, "unpublish"
            ) and asyncio.iscoroutinefunction(self._instance_publisher.unpublish):
                await self._instance_publisher.unpublish()
            elif hasattr(
                self._instance_publisher, "close"
            ) and asyncio.iscoroutinefunction(self._instance_publisher.close):
                await self._instance_publisher.close()
            else:
                if hasattr(self._instance_publisher, "unpublish"):
                    self._instance_publisher.unpublish()

            logging.info(
                f"TestDataSourceRuntime {self._identity_short} service unpublished."
            )
        if self._grpc_publisher:
            await self._grpc_publisher.stop_async()
            logging.info(
                f"TestDataSourceRuntime {self._identity_short} gRPC server stopped."
            )
        logging.info(f"TestDataSourceRuntime {self._identity_short} stopped.")

    @property
    def identity_str(self) -> str:  # Changed from RuntimeIdentity to str
        return self._identity_full

    @property
    def port(self) -> int:
        return self._port


class AggregatorStreamHandler:
    """
    Manages the client-side of the ExchangeData bidirectional stream.
    """

    def __init__(
        self,
        stub: E2ETestServiceStub,
        client_id_str: str,  # Changed from RuntimeIdentity
        received_ack_q: Queue[str],
        received_data_q: Queue[TensorChunk],
        send_data_q: Queue[Optional[TensorChunk]],
    ):
        self._stub = stub
        self._client_id_full = client_id_str
        self._client_id_short = self._client_id_full[:8]
        self._received_ack_q = received_ack_q
        self._received_data_q = received_data_q
        self._send_data_q = send_data_q

        self._stream_task: Optional[asyncio.Task] = None
        self._is_active = False
        self._sending_active = True
        self._handshake_done = asyncio.Event()

    async def _request_generator(self) -> AsyncIterator[E2EStreamRequest]:
        caller_id_msg = CallerId(id=self._client_id_full)
        logging.info(
            f"AggregatorStreamHandler ({self._client_id_short}): Sending CallerId."
        )
        yield E2EStreamRequest(caller_id=caller_id_msg)

        while self._sending_active:
            try:
                data_to_send = await asyncio.wait_for(
                    self._send_data_q.get(), timeout=0.1
                )
                if data_to_send is None:
                    logging.info(
                        f"AggregatorStreamHandler ({self._client_id_short}): Sentinel received, stopping sender loop."
                    )
                    self._sending_active = False
                    break
                logging.info(
                    f"AggregatorStreamHandler ({self._client_id_short}): Sending data chunk idx {data_to_send.starting_index}."
                )
                yield E2EStreamRequest(data_chunk=data_to_send)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(
                    f"AggregatorStreamHandler ({self._client_id_short}): Error in request generator: {e}",
                    exc_info=True,
                )
                self._sending_active = False
                break
        logging.info(
            f"AggregatorStreamHandler ({self._client_id_short}): Request generator finished."
        )

    async def _run_stream(self):
        logging.info(
            f"AggregatorStreamHandler ({self._client_id_short}): Starting stream processing task."
        )
        self._is_active = True
        self._sending_active = True
        try:
            response_iterator = self._stub.ExchangeData(self._request_generator())
            async for response in response_iterator:
                if not self._is_active:
                    break
                if response.HasField("ack_message"):
                    ack = response.ack_message
                    logging.info(
                        f"AggregatorStreamHandler ({self._client_id_short}): Received ack: {ack}"
                    )
                    await self._received_ack_q.put(ack)
                    if (
                        "CallerId" in ack
                        and self._client_id_full in ack  # Check full ID in ACK
                        and "received" in ack
                    ):
                        self._handshake_done.set()
                elif response.HasField("data_chunk"):
                    chunk = response.data_chunk
                    logging.info(
                        f"AggregatorStreamHandler ({self._client_id_short}): Received data chunk idx {chunk.starting_index}"
                    )
                    await self._received_data_q.put(chunk)
                else:
                    logging.warning(
                        f"AggregatorStreamHandler ({self._client_id_short}): Received empty E2EStreamResponse payload."
                    )
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logging.info(
                    f"AggregatorStreamHandler ({self._client_id_short}): Stream cancelled (gRPC). Code: {e.code()}"
                )
            else:
                logging.error(
                    f"AggregatorStreamHandler ({self._client_id_short}): gRPC error in stream: {e.code()} - {e.details()}",
                    exc_info=True,
                )
        except Exception as e:
            logging.error(
                f"AggregatorStreamHandler ({self._client_id_short}): Exception in stream processing: {e}",
                exc_info=True,
            )
        finally:
            self._is_active = False
            self._sending_active = False
            if not self._handshake_done.is_set():
                logging.warning(
                    f"AggregatorStreamHandler ({self._client_id_short}): Stream ended before handshake completed."
                )
                self._handshake_done.set()
            logging.info(
                f"AggregatorStreamHandler ({self._client_id_short}): Stream processing task finished."
            )

    async def start(self):
        if self._stream_task and not self._stream_task.done():
            logging.warning(
                f"AggregatorStreamHandler ({self._client_id_short}): Stream task already running or not properly cleaned up."
            )
            return

        self._handshake_done.clear()
        self._is_active = True
        self._sending_active = True

        self._stream_task = asyncio.create_task(self._run_stream())
        logging.info(
            f"AggregatorStreamHandler ({self._client_id_short}): Stream processing initiated via task."
        )

    async def stop(self):
        logging.info(
            f"AggregatorStreamHandler ({self._client_id_short}): Attempting to stop stream."
        )
        self._is_active = False
        self._sending_active = False
        try:
            self._send_data_q.put_nowait(None)
        except asyncio.QueueFull:
            logging.warning(
                f"AggregatorStreamHandler ({self._client_id_short}): Send data queue full during stop, generator might be stuck."
            )

        if self._stream_task and not self._stream_task.done():
            logging.info(
                f"AggregatorStreamHandler ({self._client_id_short}): Cancelling stream task."
            )
            self._stream_task.cancel()
            try:
                await self._stream_task
                logging.info(
                    f"AggregatorStreamHandler ({self._client_id_short}): Stream task awaited after cancellation."
                )
            except asyncio.CancelledError:
                logging.info(
                    f"AggregatorStreamHandler ({self._client_id_short}): Stream task successfully cancelled."
                )
            except Exception as e:
                logging.error(
                    f"AggregatorStreamHandler ({self._client_id_short}): Error during stream task cancellation/await: {e}",
                    exc_info=True,
                )
        else:
            logging.info(
                f"AggregatorStreamHandler ({self._client_id_short}): Stream task already done or not started."
            )
        logging.info(
            f"AggregatorStreamHandler ({self._client_id_short}): Stop sequence finished."
        )

    async def wait_for_handshake(self, timeout: float = 5.0) -> bool:
        try:
            logging.info(
                f"AggregatorStreamHandler ({self._client_id_short}): Waiting for handshake event."
            )
            await asyncio.wait_for(self._handshake_done.wait(), timeout=timeout)
            logging.info(
                f"AggregatorStreamHandler ({self._client_id_short}): Handshake event received (is_set: {self._handshake_done.is_set()})."
            )
            return self._handshake_done.is_set()
        except asyncio.TimeoutError:
            logging.error(
                f"AggregatorStreamHandler ({self._client_id_short}): Timeout waiting for handshake signal."
            )
            return False

    def is_stream_active(self) -> bool:
        return (
            self._is_active
            and self._stream_task is not None
            and not self._stream_task.done()
        )


class TestDataAggregatorRuntime(
    Runtime, ServiceConnector[ServiceInfo, grpc.aio.Channel].Client
):
    __test__ = False  # Prevent pytest collection
    """
    Discovers TestDataSourceRuntime and initiates a gRPC stream for data exchange.
    Simulates a data aggregator.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        grpc_channel_factory: GrpcChannelFactory,
        client_received_ack_q: Queue[str],
        client_received_data_q: Queue[TensorChunk],
        client_send_data_q: Queue[Optional[TensorChunk]],
        mdns_listener_factory: Optional[MdnsListenerFactory] = None,
    ):
        super().__init__()
        self._watcher = watcher
        self._grpc_channel_factory = grpc_channel_factory
        self._identity_full = "test-data-aggregator-" + str(uuid.uuid4())
        self._identity_short = self._identity_full[:8]
        logging.info(
            f"TestDataAggregatorRuntime initializing with ID {self._identity_short}"
        )

        self._client_received_ack_q = client_received_ack_q
        self._client_received_data_q = client_received_data_q
        self._client_send_data_q = client_send_data_q

        discoverer = DiscoveryHost(
            service_type=TestDataSourceRuntime.SERVICE_NAME,
            mdns_listener_factory=mdns_listener_factory,
        )
        self._connector = ServiceConnector[ServiceInfo, grpc.aio.Channel](
            client=self,
            channel_factory=self._grpc_channel_factory,
            discovery_host=discoverer,
        )
        self._stream_handler: Optional[AggregatorStreamHandler] = None
        self._is_running = IsRunningTracker()
        self._connected_event = asyncio.Event()

    async def start_async(self):
        if self._is_running.get():
            logging.warning(
                f"TestDataAggregatorRuntime {self._identity_short} already started."
            )
            return
        self._is_running.start()
        logging.info(f"TestDataAggregatorRuntime {self._identity_short} starting...")
        await self._connector.start()
        logging.info(
            f"TestDataAggregatorRuntime {self._identity_short} connector started, discovering services."
        )

    async def stop(self, exception: Optional[Exception] = None):
        if not self._is_running.get_and_set(False):
            logging.warning(
                f"TestDataAggregatorRuntime {self._identity_short} already stopped or not started."
            )
            return

        logging.info(f"TestDataAggregatorRuntime {self._identity_short} stopping...")
        if self._stream_handler:
            await self._stream_handler.stop()
            logging.info(
                f"TestDataAggregatorRuntime {self._identity_short} stream handler stopped."
            )
        if self._connector:
            await self._connector.stop()
            logging.info(
                f"TestDataAggregatorRuntime {self._identity_short} connector stopped."
            )
        logging.info(f"TestDataAggregatorRuntime {self._identity_short} stopped.")

    async def _on_channel_connected(
        self,
        connection_info: ServiceInfo,
        caller_id: CallerIdentifier,  # This is the ID of the ServiceConnector itself, not the runtime
        channel_info: ChannelInfo,
    ) -> None:
        logging.info(
            f"TestDataAggregatorRuntime ({self._identity_short}): Connected to service {connection_info.name} "
            f"at {channel_info.target_address}."
        )
        stub = E2ETestServiceStub(channel_info.channel)
        self._stream_handler = AggregatorStreamHandler(
            stub=stub,
            client_id_str=self._identity_full,  # Pass own full ID to handler
            received_ack_q=self._client_received_ack_q,
            received_data_q=self._client_received_data_q,
            send_data_q=self._client_send_data_q,
        )
        await self._stream_handler.start()
        self._connected_event.set()
        logging.info(
            f"TestDataAggregatorRuntime ({self._identity_short}): Stream handler started for {connection_info.name}."
        )

    async def _on_channel_connect_failed(
        self, connection_info: ServiceInfo, exception: Exception
    ) -> None:
        logging.error(
            f"TestDataAggregatorRuntime ({self._identity_short}): Failed to connect to {connection_info.name}: {exception}",
            exc_info=True,
        )
        self._connected_event.set()

    async def _on_channel_disconnected(
        self, connection_info: ServiceInfo, exception: Optional[Exception]
    ) -> None:
        logging.info(
            f"TestDataAggregatorRuntime ({self._identity_short}): Disconnected from {connection_info.name}. "
            f"Exception: {exception if exception else 'N/A'}"
        )
        if self._stream_handler:
            await self._stream_handler.stop()
        self._connected_event.clear()

    async def send_data_chunk(self, data_chunk: TensorChunk) -> None:
        if not self._stream_handler or not self._stream_handler.is_stream_active():
            logging.error(
                f"TestDataAggregatorRuntime ({self._identity_short}): Stream not active, cannot send data."
            )
            raise RuntimeError("Stream not active for sending data.")
        await self._client_send_data_q.put(data_chunk)
        logging.info(
            f"TestDataAggregatorRuntime ({self._identity_short}): Queued data chunk idx {data_chunk.starting_index} for sending."
        )

    async def wait_for_connection_establishment(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
            return self._connected_event.is_set()
        except asyncio.TimeoutError:
            logging.error(
                f"TestDataAggregatorRuntime ({self._identity_short}): Timeout waiting for connection establishment."
            )
            return False

    async def wait_for_rpc_handshake(self, timeout: float = 5.0) -> bool:
        if not self._stream_handler:
            logging.warning(
                f"TestDataAggregatorRuntime ({self._identity_short}): No stream handler to wait for handshake."
            )
            return False
        return await self._stream_handler.wait_for_handshake(timeout)

    @property
    def identity_str(self) -> str:  # Changed from RuntimeIdentity to str
        return self._identity_full

    def is_connected_and_handshake_done(self) -> bool:
        return self._connected_event.is_set() and (
            self._stream_handler is not None
            and self._stream_handler._handshake_done.is_set()
        )


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
