from tsercom.test.proto.generated.v1_73 import (  # Updated import path
    e2e_test_service_pb2_grpc,
    e2e_test_service_pb2,
)
from tsercom.caller_id.proto.generated.v1_73 import caller_id_pb2
from tsercom.tensor.proto.generated.v1_73 import tensor_pb2
from tsercom.timesync.common.proto.generated.v1_73 import time_pb2


import asyncio
import logging
from typing import Optional, TYPE_CHECKING, AsyncIterator, Any, AsyncGenerator, Callable # Added Callable

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore
import pytest
import pytest_asyncio
import socket
import uuid # Added import for uuid

from tsercom.api import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier # Re-added
# Removed RuntimeOutOfProcessConfigLoader import
from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.discovery.mdns.instance_listener import MdnsListenerFactory
from tsercom.discovery.mdns.instance_publisher import InstancePublisher
from tsercom.discovery.service_connector import ServiceConnector
from tsercom.discovery.service_info import ServiceInfo
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_config import ServiceType
from tsercom.runtime.runtime_data_handler import (
    RuntimeDataHandler,
)  # Keep if used by existing code
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.global_event_loop import clear_tsercom_event_loop
# Atomic import will be removed by ruff if not used after deletions.
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import (
    IsRunningTracker,
) # IsRunningTracker import reinstated
from tsercom.discovery.mdns.mdns_listener import MdnsListener
from zeroconf.asyncio import AsyncZeroconf


if TYPE_CHECKING:
    pass


# Removed has_been_hit global variable, as test_anomoly_service is being removed.

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

        # Construct ServiceInfo and CallerIdentifier
        # For ServiceInfo, we need a way to create one. Assuming a simple constructor or factory.
        # If ServiceInfo is complex, this might need adjustment.
        # Let's assume ServiceInfo can be created with name, addresses, port, txt_record.
        # The actual ServiceInfo might have a more specific type if ServiceInfoT is constrained.
        # For this test, a generic ServiceInfo should be fine if DiscoveryHost can handle it.
        # DiscoveryHost is Generic[ServiceInfoT], ServiceConnector is Generic[SourceServiceInfoT, ChannelTypeT]
        # TestDataAggregatorRuntime uses ServiceInfo for SourceServiceInfoT.

        service_info = ServiceInfo(
            name=fake_mdns_instance_name, # This can be simplified, often mdns_name is used as 'name' too
            addresses=[socket.inet_ntoa(fake_ip_address_bytes)],
            port=self.__port,
            mdns_name=fake_mdns_instance_name # mdns_name is typically the full instance name
        )

        # CallerIdentifier needs a unique ID. We can base it on the instance name.
        # Generate a UUID based on the instance name for deterministic IDs in tests.
        # Using NAMESPACE_DNS as an example; any consistent namespace UUID would work.
        # DiscoveryHost._on_service_added will internally create a CallerIdentifier from service_info.mdns_name.
        # So, fake_mdns_instance_name (which becomes service_info.mdns_name) must be a UUID string.
        mdns_compatible_name_for_uuid_basis = self.__service_type.replace('.', '-') + "-fake-instance"
        generated_mdns_uuid_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, mdns_compatible_name_for_uuid_basis))

        # Ensure fake_mdns_instance_name used for ServiceInfo.mdns_name is the UUID string
        service_info = ServiceInfo(
            name=fake_mdns_instance_name, # User-friendly name can remain as is
            addresses=['::1', socket.inet_ntoa(fake_ip_address_bytes)], # Try IPv6 loopback first, then IPv4
            port=self.__port,
            mdns_name=generated_mdns_uuid_str # This must be the UUID string
        )

        # DiscoveryHost (self.__client) implements InstanceListener.Client,
        # whose _on_service_added takes only service_info.
        await self.__client._on_service_added(service_info)
        logging.info(
            f"FakeMdnsListener: _on_service_added called for {fake_mdns_instance_name} (mdns_name: {generated_mdns_uuid_str})"
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


# Removed GenericServerRuntime
# Removed GenericClientRuntime
# Removed GenericServerRuntimeInitializer
# Removed GenericClientRuntimeInitializer


class TestDataSourceRuntime(Runtime):
    """Runtime that hosts the E2ETestStreamServicer (server-side)."""

    __test__ = False  # Prevent pytest collection

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        grpc_port: int,
        service_name: str,  # e.g., "e2e_data_source"
        client_caller_id_event: asyncio.Event,
        server_received_data_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        server_data_to_send_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        server_caller_id_storage: list[caller_id_pb2.CallerId],
    ):
        super().__init__()
        self._thread_watcher = thread_watcher
        self._grpc_port = grpc_port
        self._service_name = service_name  # Used for mDNS advertising
        self._mdns_service_type = "_e2e_stream_data_source._tcp"  # Example service type

        # For E2ETestStreamServicer
        self._client_caller_id_event = client_caller_id_event
        self._server_received_data_queue = server_received_data_queue
        self._server_data_to_send_queue = server_data_to_send_queue
        self._server_caller_id_storage = server_caller_id_storage
        self._server_ready_event = asyncio.Event() # Event to signal gRPC server is up

        self._grpc_publisher: Optional[GrpcServicePublisher] = None
        self._instance_publisher: Optional[InstancePublisher] = None
        self._is_running = IsRunningTracker()

    @property
    def server_ready_event(self) -> asyncio.Event:
        return self._server_ready_event

    async def start_async(self) -> None:
        if self._is_running.get():
            logging.warning("TestDataSourceRuntime already running.")
            return
        self._is_running.start()
        logging.info(f"TestDataSourceRuntime starting on port {self._grpc_port}...")

        self._grpc_publisher = GrpcServicePublisher(
            self._thread_watcher, self._grpc_port
        )

        def _add_servicers(server: grpc.aio.Server) -> None:
            servicer = E2ETestStreamServicer(
                client_caller_id_event=self._client_caller_id_event,
                server_received_data_queue=self._server_received_data_queue,
                server_data_to_send_queue=self._server_data_to_send_queue,
                server_caller_id_storage=self._server_caller_id_storage,
            )
            e2e_test_service_pb2_grpc.add_E2ETestServiceServicer_to_server(
                servicer, server
            )

            # Add health servicer
            health_servicer = health.HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
            # Use the fully qualified service name for the specific service health
            service_health_name = e2e_test_service_pb2_grpc.E2ETestServiceServicer.SERVICE_NAME  # type: ignore
            health_servicer.set(
                service_health_name,
                health_pb2.HealthCheckResponse.SERVING,
            )
            health_servicer.set(
                "",  # Overall health for the server
                health_pb2.HealthCheckResponse.SERVING,
            )
            logging.info(
                f"E2ETestStreamServicer (service name: {service_health_name}) and HealthServicer added to gRPC server for TestDataSourceRuntime."
            )

        await self._grpc_publisher.start_async(_add_servicers)
        self._server_ready_event.set() # Signal that server is ready
        logging.info(f"TestDataSourceRuntime gRPC server started and ready_event set for port {self._grpc_port}.")

        self._instance_publisher = InstancePublisher(
            port=self._grpc_port,
            service_type=self._mdns_service_type,
            instance_name=self._service_name,
            # txt_records can be added if needed
        )
        await self._instance_publisher.publish()
        logging.info(
            f"TestDataSourceRuntime {self._service_name} published via mDNS on type {self._mdns_service_type} and port {self._grpc_port}."
        )

    async def stop(self, exception: Optional[Exception] = None) -> None: # Renamed from stop_async
        if not self._is_running.get_and_set(
            False
        ):  # Ensure it was running and set to not running
            logging.warning("TestDataSourceRuntime already stopped or not started.")
            return
        logging.info(f"TestDataSourceRuntime {self._service_name} stopping...")
        if self._instance_publisher:
            await self._instance_publisher.unpublish()
            self._instance_publisher = None
            logging.info(
                f"TestDataSourceRuntime {self._service_name} unpublished from mDNS."
            )
        if self._grpc_publisher:
            await self._grpc_publisher.stop_async()
            self._grpc_publisher = None
            logging.info(
                f"TestDataSourceRuntime {self._service_name} gRPC publisher stopped."
            )
        logging.info(f"TestDataSourceRuntime {self._service_name} stopped.")


class TestDataAggregatorRuntime(Runtime, ServiceConnector.Client):
    """Runtime that discovers TestDataSourceRuntime and acts as a gRPC client."""

    __test__ = False  # Prevent pytest collection

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        instance_listener_factory: Callable[[MdnsListener.Client], FakeMdnsListener],  # Corrected
        grpc_channel_factory: GrpcChannelFactory,
        own_caller_id: caller_id_pb2.CallerId,
        handshake_done_event: asyncio.Event,
        aggregator_received_data_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        aggregator_data_to_send_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        connection_established_event: asyncio.Event,
    ):
        super().__init__()
        self._thread_watcher = thread_watcher
        self._grpc_channel_factory = grpc_channel_factory
        self._own_caller_id = own_caller_id

        # Events and Queues for stream management and test synchronization
        self._handshake_done_event = handshake_done_event
        self._aggregator_received_data_queue = aggregator_received_data_queue
        self._aggregator_data_to_send_queue = aggregator_data_to_send_queue
        self._connection_established_event = connection_established_event

        self._target_service_type = "_e2e_stream_data_source._tcp"  # Must match what TestDataSourceRuntime publishes

        self._discovery_host = DiscoveryHost(
            instance_listener_factory=instance_listener_factory, # Corrected: Use instance_listener_factory
            # No service_type needed here for DiscoveryHost when instance_listener_factory is used
        )
        self._service_connector = ServiceConnector[ServiceInfo, grpc.aio.Channel](
            client=self,
            connection_factory=self._grpc_channel_factory, # Corrected keyword
            service_source=self._discovery_host,         # Corrected keyword
        )
        self._stub: Optional[e2e_test_service_pb2_grpc.E2ETestServiceStub] = None
        self._stream_active_event = (
            asyncio.Event()
        )  # To signal the stream management task to stop
        self._is_running = IsRunningTracker()
        self._active_calls: list[asyncio.Task] = []

    async def start_async(self) -> None:
        if self._is_running.get():
            logging.warning("TestDataAggregatorRuntime already running.")
            return
        self._is_running.start()
        self._stream_active_event.set()  # Mark stream as potentially active
        logging.info("TestDataAggregatorRuntime starting...")
        await self._service_connector.start()
        logging.info(
            f"TestDataAggregatorRuntime started, discovering {self._target_service_type}."
        )

    async def stop(self, exception: Optional[Exception] = None) -> None: # Renamed from stop_async
        if not self._is_running.get_and_set(False):
            logging.warning("TestDataAggregatorRuntime already stopped or not started.")
            return

        logging.info("TestDataAggregatorRuntime stopping...")
        self._stream_active_event.clear()  # Signal stream management to stop

        if self._service_connector:
            await self._service_connector.stop()

        # Cancel any active gRPC stream tasks
        for task in self._active_calls:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._active_calls, return_exceptions=True)
        self._active_calls.clear()

        logging.info("TestDataAggregatorRuntime stopped.")

    async def _on_channel_connected(
        self,
        connection_info: ServiceInfo, # Corrected: Matches ServiceConnector.Client ABC
        caller_id: CallerIdentifier,   # Corrected: Matches ServiceConnector.Client ABC
        channel: grpc.aio.Channel,   # This was correct
    ) -> None:
        logging.info(
            f"TestDataAggregatorRuntime: Channel connected to service '{connection_info.name}' (CallerID: {caller_id.id})."
        )
        self._stub = e2e_test_service_pb2_grpc.E2ETestServiceStub(channel)
        self._connection_established_event.set()

        # Start the stream management task
        stream_task = asyncio.create_task(self._manage_grpc_stream())
        self._active_calls.append(stream_task)
        logging.info("TestDataAggregatorRuntime: gRPC stream management task started.")

    async def _on_channel_disconnected( # This signature is not defined by ServiceConnector.Client ABC, but by ServiceSource.Client
        self, service_name: str, caller_id: CallerIdentifier # Corrected to match ServiceSource.Client
    ) -> None:
        logging.info(
            f"TestDataAggregatorRuntime: Channel disconnected from service '{service_name}' (CallerID: {caller_id.id})."
        )
        self._stub = None
        # self._stream_active_event.clear() # Might be too aggressive if reconnections are possible

    async def _manage_grpc_stream(self) -> None:
        if not self._stub:
            logging.error(
                "TestDataAggregatorRuntime: Stub not available for managing stream."
            )
            return

        logging.info("TestDataAggregatorRuntime: _manage_grpc_stream started.")

        request_iterator_closed = asyncio.Event()

        async def request_generator() -> (
            AsyncGenerator[e2e_test_service_pb2.E2EStreamRequest, None]
        ):
            try:
                # Send initial CallerId
                logging.info(
                    f"TestDataAggregatorRuntime: Sending CallerId: {self._own_caller_id.id}"
                )
                yield e2e_test_service_pb2.E2EStreamRequest(
                    caller_id=self._own_caller_id
                )

                # Then send data chunks from the queue
                while self._stream_active_event.is_set():
                    try:
                        chunk_to_send = await asyncio.wait_for(
                            self._aggregator_data_to_send_queue.get(), timeout=0.1
                        )
                        if chunk_to_send:
                            logging.info(
                                "TestDataAggregatorRuntime: Sending data chunk from aggregator."
                            )
                            yield e2e_test_service_pb2.E2EStreamRequest(
                                data_chunk=chunk_to_send
                            )
                            self._aggregator_data_to_send_queue.task_done()
                        else:  # Should not happen with valid chunks
                            logging.warning(
                                "TestDataAggregatorRuntime: Got None from send queue."
                            )
                            break
                    except asyncio.TimeoutError:
                        # This is normal, just means the queue was empty for a bit.
                        # Continue loop to check _stream_active_event.
                        pass
                    except Exception as e:
                        logging.error(
                            f"TestDataAggregatorRuntime: Error in request_generator sending data: {e}"
                        )
                        break
            except Exception as e:
                logging.error(
                    f"TestDataAggregatorRuntime: Error in request_generator: {e}",
                    exc_info=True,
                )
            finally:
                logging.info("TestDataAggregatorRuntime: Request generator finished.")
                request_iterator_closed.set()

        try:
            response_stream = self._stub.ExchangeData(request_generator())
            handshake_acked = False
            async for response in response_stream:
                if (
                    not self._stream_active_event.is_set()
                ):  # Check if runtime is stopping
                    logging.info(
                        "TestDataAggregatorRuntime: Stream processing stopped by runtime state."
                    )
                    break

                logging.info(
                    f"TestDataAggregatorRuntime: Received response: {response.ack_message}"
                )
                if (
                    not handshake_acked
                    and self._own_caller_id.id in response.ack_message
                ):
                    logging.info(
                        "TestDataAggregatorRuntime: Handshake with server successful."
                    )
                    self._handshake_done_event.set()
                    handshake_acked = True

                if response.HasField("data_chunk"):
                    logging.info(
                        "TestDataAggregatorRuntime: Received data chunk from server."
                    )
                    await self._aggregator_received_data_queue.put(response.data_chunk)

            # Wait for request generator to finish if it hasn't (e.g. if server closes stream first)
            await asyncio.wait_for(request_iterator_closed.wait(), timeout=5.0)

        except grpc.aio.AioRpcError as e:
            logging.error(
                f"TestDataAggregatorRuntime: gRPC error in stream: {e.code()} - {e.details()}"
            )
        except Exception as e:
            logging.error(
                f"TestDataAggregatorRuntime: Exception in _manage_grpc_stream: {e}",
                exc_info=True,
            )
        finally:
            logging.info("TestDataAggregatorRuntime: _manage_grpc_stream finished.")
            if (
                self._stream_active_event.is_set()
            ):  # If not stopped externally, mark as inactive
                # This event is more about the runtime stopping than the single stream.
                # If the stream ends but runtime is still up, it might try to reconnect.
                pass


@pytest_asyncio.fixture
async def clear_loop_fixture():
    # Ensure tsercom's global loop is managed.
    clear_tsercom_event_loop()
    yield
    clear_tsercom_event_loop()
    await asyncio.sleep(0.1)


# Removed test_anomoly_service


# BEGIN E2ETestStreamServicer
class E2ETestStreamServicer(e2e_test_service_pb2_grpc.E2ETestServiceServicer):
    """Server-side implementation for the E2E streaming test."""

    __test__ = False  # Prevent pytest collection

    def __init__(
        self,
        client_caller_id_event: asyncio.Event,
        server_received_data_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        server_data_to_send_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        server_caller_id_storage: list[caller_id_pb2.CallerId],
    ):
        self._client_caller_id_event = client_caller_id_event
        self._server_received_data_queue = server_received_data_queue
        self._server_data_to_send_queue = server_data_to_send_queue
        self._server_caller_id_storage = (
            server_caller_id_storage  # To store client's CallerId
        )
        self._client_id_str: Optional[str] = None

    async def Echo(
        self,
        request: e2e_test_service_pb2.EchoRequest,
        context: grpc.aio.ServicerContext,
    ) -> e2e_test_service_pb2.EchoResponse:
        # Keep Echo for compatibility or other tests, but not used by this stream test.
        logging.info(f"E2ETestStreamServicer received Echo request: {request.message}")
        return e2e_test_service_pb2.EchoResponse(
            response=f"Server echoes: {request.message}"
        )

    async def ExchangeData(
        self,
        request_iterator: AsyncIterator[e2e_test_service_pb2.E2EStreamRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[e2e_test_service_pb2.E2EStreamResponse, None]:
        logging.info("E2ETestStreamServicer: ExchangeData stream started.")
        client_identified = False

        # Task for sending data from server_data_to_send_queue
        send_task = None
        try:
            # First, handle incoming requests from the client
            async for request in request_iterator:
                if request.HasField("caller_id"):
                    self._client_id_str = request.caller_id.id
                    self._server_caller_id_storage.append(request.caller_id)
                    logging.info(
                        f"E2ETestStreamServicer: Received CallerId: {self._client_id_str}"
                    )
                    client_identified = True
                    self._client_caller_id_event.set()
                    yield e2e_test_service_pb2.E2EStreamResponse(
                        ack_message=f"CallerId {self._client_id_str} received"
                    )

                    # Start the sending task only after client is identified
                    if send_task is None:

                        async def _send_data_loop():
                            while True:
                                try:
                                    chunk_to_send = await asyncio.wait_for(
                                        self._server_data_to_send_queue.get(),
                                        timeout=0.1,
                                    )
                                    if chunk_to_send:
                                        logging.info(
                                            "E2ETestStreamServicer: Sending data chunk from server."
                                        )
                                        yield e2e_test_service_pb2.E2EStreamResponse(
                                            ack_message="Server data push",
                                            data_chunk=chunk_to_send,
                                        )
                                        self._server_data_to_send_queue.task_done()
                                    else:  # Should not happen with valid chunks
                                        break
                                except asyncio.TimeoutError:
                                    pass  # Normal, just check if client is still connected
                                except Exception as e:
                                    logging.error(
                                        f"E2ETestStreamServicer: Error in send_data_loop: {e}"
                                    )
                                    break
                                if (
                                    context.is_active()
                                ):  # Check if client is still connected
                                    await asyncio.sleep(
                                        0.01
                                    )  # Small sleep to prevent busy loop if queue is empty
                                else:
                                    logging.info(
                                        "E2ETestStreamServicer: Client disconnected, stopping send_data_loop."
                                    )
                                    break

                        # This is tricky because `yield` is not allowed in the task.
                        # The sending logic needs to be integrated into the main async generator.
                        # For now, let's simplify: server sends data primarily upon receiving data.
                        # A more robust solution would involve `asyncio.Queue` and `context.write` directly.
                        # The current structure `yield` from the main loop.

                elif request.HasField("data_chunk"):
                    if not client_identified:
                        logging.warning(
                            "E2ETestStreamServicer: Received data chunk before CallerId. Ignoring."
                        )
                        # Optionally, could terminate the stream or send an error.
                        # For this test, we'll be strict and expect CallerId first.
                        # await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "CallerId must be sent first.")
                        # return
                        continue  # Ignore data if client not identified

                    logging.info(
                        f"E2ETestStreamServicer: Received data chunk from {self._client_id_str}"
                    )
                    await self._server_received_data_queue.put(request.data_chunk)
                    yield e2e_test_service_pb2.E2EStreamResponse(
                        ack_message=f"Data chunk received by server from {self._client_id_str}"
                    )

                    # Example: Server immediately tries to send something from its queue if available
                    if not self._server_data_to_send_queue.empty():
                        try:
                            chunk_to_send = self._server_data_to_send_queue.get_nowait()
                            logging.info(
                                "E2ETestStreamServicer: Sending data chunk from server in response to client data."
                            )
                            yield e2e_test_service_pb2.E2EStreamResponse(
                                ack_message="Server data push (reactive)",
                                data_chunk=chunk_to_send,
                            )
                            self._server_data_to_send_queue.task_done()
                        except asyncio.QueueEmpty:
                            pass  # No data to send back immediately
                else:
                    logging.warning(
                        "E2ETestStreamServicer: Received empty payload in E2EStreamRequest."
                    )

            # After client closes its sending stream, keep sending if there's data
            logging.info(
                "E2ETestStreamServicer: Client has finished sending. Server checking its send queue."
            )
            while not self._server_data_to_send_queue.empty() and context.is_active():
                try:
                    chunk_to_send = self._server_data_to_send_queue.get_nowait()
                    logging.info(
                        "E2ETestStreamServicer: Sending remaining data chunk from server."
                    )
                    yield e2e_test_service_pb2.E2EStreamResponse(
                        ack_message="Server data push (remaining)",
                        data_chunk=chunk_to_send,
                    )
                    self._server_data_to_send_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                await asyncio.sleep(0.01)

        except grpc.aio.AioRpcError as e:
            logging.error(f"E2ETestStreamServicer: gRPC Error in ExchangeData: {e}")
        except Exception as e:
            logging.error(
                f"E2ETestStreamServicer: Error in ExchangeData: {e}", exc_info=True
            )
        finally:
            logging.info(
                f"E2ETestStreamServicer: ExchangeData stream for {self._client_id_str or 'unknown client'} ended."
            )
            # Signal that server-side processing for this client is done if needed by specific tests.


# END E2ETestStreamServicer


# Removed test_full_app_with_grpc_transport


# Initializers for the new E2E stream test
class TestDataSourceRuntimeInitializer(RuntimeInitializer[Any, Any]):
    def __init__(
        self,
        grpc_port: int,
        service_name: str,
        client_caller_id_event: asyncio.Event,
        server_received_data_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        server_data_to_send_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        server_caller_id_storage: list[caller_id_pb2.CallerId],
        # server_ready_event: asyncio.Event, # Removed from initializer
    ):
        super().__init__(service_type=ServiceType.SERVER)  # Or a custom type if needed
        self._grpc_port = grpc_port
        self._service_name = service_name
        self._client_caller_id_event = client_caller_id_event
        self._server_received_data_queue = server_received_data_queue
        self._server_data_to_send_queue = server_data_to_send_queue
        self._server_caller_id_storage = server_caller_id_storage
        # self._server_ready_event = server_ready_event # Removed from initializer

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[
            Any, Any
        ],  # Type params not critical for this test structure
        grpc_channel_factory: GrpcChannelFactory,  # Not directly used by DataSource, but part of signature
    ) -> Runtime:
        return TestDataSourceRuntime(
            thread_watcher=thread_watcher,
            grpc_port=self._grpc_port,
            service_name=self._service_name,
            client_caller_id_event=self._client_caller_id_event,
            server_received_data_queue=self._server_received_data_queue,
            server_data_to_send_queue=self._server_data_to_send_queue,
            server_caller_id_storage=self._server_caller_id_storage,
        )


class TestDataAggregatorRuntimeInitializer(RuntimeInitializer[Any, Any]):
    def __init__(
        self,
        instance_listener_factory: Callable[[MdnsListener.Client], FakeMdnsListener], # Corrected type
        own_caller_id: caller_id_pb2.CallerId,
        handshake_done_event: asyncio.Event,
        aggregator_received_data_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        aggregator_data_to_send_queue: asyncio.Queue[tensor_pb2.TensorChunk],
        connection_established_event: asyncio.Event,
    ):
        super().__init__(service_type=ServiceType.CLIENT)  # Or a custom type
        self._instance_listener_factory = instance_listener_factory # Corrected assignment
        self._own_caller_id = own_caller_id
        self._handshake_done_event = handshake_done_event
        self._aggregator_received_data_queue = aggregator_received_data_queue
        self._aggregator_data_to_send_queue = aggregator_data_to_send_queue
        self._connection_established_event = connection_established_event

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[Any, Any],  # Not directly used by Aggregator
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return TestDataAggregatorRuntime(
            thread_watcher=thread_watcher,
            instance_listener_factory=self._instance_listener_factory, # Corrected parameter passed
            grpc_channel_factory=grpc_channel_factory,
            own_caller_id=self._own_caller_id,
            handshake_done_event=self._handshake_done_event,
            aggregator_received_data_queue=self._aggregator_received_data_queue,
            aggregator_data_to_send_queue=self._aggregator_data_to_send_queue,
            connection_established_event=self._connection_established_event,
        )


@pytest.mark.asyncio
async def test_full_e2e_with_discovery_and_grpc_stream(clear_loop_fixture, caplog):
    """
    Tests the full client-advertises, server-discovers architecture with a
    bidirectional gRPC stream, CallerId handshaking, and data transfer.
    """
    caplog.set_level(logging.INFO)
    logging.info("Starting test_full_e2e_with_discovery_and_grpc_stream")

    DATA_SOURCE_PORT = 50055  # Arbitrary port for the data source
    DATA_SOURCE_SERVICE_NAME = "TestSourceServiceInstance"
    AGGREGATOR_CALLER_ID_STR = "TestAggregatorClient_123"

    # Synchronization primitives
    client_caller_id_event_at_server = (
        asyncio.Event()
    )  # Server sets this when it receives client's CallerId
    handshake_done_event_at_client = (
        asyncio.Event()
    )  # Client sets this when server acks its CallerId
    connection_established_event = (
        asyncio.Event()
    )  # Client sets this when _on_channel_connected

    # Data queues
    # Data sent by Aggregator, received by DataSource
    agg_to_ds_data_q = asyncio.Queue[tensor_pb2.TensorChunk]()
    # Data sent by DataSource, received by Aggregator
    ds_to_agg_data_q = asyncio.Queue[tensor_pb2.TensorChunk]()

    # Storage for server to confirm received client CallerId
    server_caller_id_storage: list[caller_id_pb2.CallerId] = []

    # RuntimeManager with out-of-process config using SplitRuntimeFactoryFactory
    # This factory enables runtimes to potentially run in separate processes.
    # For this test, they will still run in the same process due to start_in_process_async,
    # but RuntimeManager is configured for out-of-process capabilities.

    # Need ThreadPoolExecutor and ThreadWatcher for SplitRuntimeFactoryFactory
    # RuntimeManager creates its own ThreadWatcher if is_testing=True,
    # but SplitRuntimeFactoryFactory needs one at construction.
    from concurrent.futures import ThreadPoolExecutor
    from tsercom.api.split_process.split_runtime_factory_factory import SplitRuntimeFactoryFactory

    # Create a dedicated ThreadWatcher for the test environment
    # This ensures that if RuntimeManager's internal watcher has specific lifecycle assumptions,
    # we don't interfere with them or rely on its internal instance.
    test_thread_watcher = ThreadWatcher()

    # Create a ThreadPoolExecutor for the factory
    # Using a small number of threads for test purposes.
    # Ensure this executor is shutdown properly, though for test scope it might auto-clean.
    # For robust cleanup, could manage it with try/finally or context manager if test was longer.
    test_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SplitFactoryTest")

    split_factory = SplitRuntimeFactoryFactory[Any, Any]( # Using Any for DataTypeT, EventTypeT as per test
        thread_pool=test_thread_pool, thread_watcher=test_thread_watcher
    )

    runtime_manager = RuntimeManager(
        split_runtime_factory_factory=split_factory, # Corrected keyword argument
        is_testing=True  # is_testing also provides a default ThreadWatcher to RuntimeManager itself
    )

    # Fake mDNS Listener setup for the TestDataAggregatorRuntime
    # The TestDataAggregatorRuntime (client) will use this to "discover" the TestDataSourceRuntime (server).
    # The FakeMdnsListener needs to know the port of the service it's faking.
    # Using instance_listener_factory route for DiscoveryHost
    target_service_type_for_discovery = "_e2e_stream_data_source._tcp"  # Used by FakeMdnsListener

    def instance_listener_factory_for_test(aggregator_runtime_as_client: MdnsListener.Client) -> FakeMdnsListener:
        # DATA_SOURCE_PORT and target_service_type_for_discovery are from the outer scope
        return FakeMdnsListener(
            aggregator_runtime_as_client,
            target_service_type_for_discovery,
            DATA_SOURCE_PORT
        )

    # Initializer for TestDataSourceRuntime (Server)
    # server_ready_event is created by TestDataSourceRuntime itself.
    data_source_initializer = TestDataSourceRuntimeInitializer(
        grpc_port=DATA_SOURCE_PORT,
        service_name=DATA_SOURCE_SERVICE_NAME,
        client_caller_id_event=client_caller_id_event_at_server,
        server_received_data_queue=agg_to_ds_data_q,  # Where DS puts data from Agg
        server_data_to_send_queue=ds_to_agg_data_q,  # Where DS gets data to send to Agg
        server_caller_id_storage=server_caller_id_storage,
        # No server_ready_event passed to initializer
    )

    # Initializer for TestDataAggregatorRuntime (Client)
    aggregator_caller_id = caller_id_pb2.CallerId(id=AGGREGATOR_CALLER_ID_STR)
    data_aggregator_initializer = TestDataAggregatorRuntimeInitializer(
        instance_listener_factory=instance_listener_factory_for_test, # Changed from mdns_listener_factory
        own_caller_id=aggregator_caller_id,
        handshake_done_event=handshake_done_event_at_client,
        aggregator_received_data_queue=ds_to_agg_data_q,  # Where Agg puts data from DS
        aggregator_data_to_send_queue=agg_to_ds_data_q,  # Where Agg gets data to send to DS
        connection_established_event=connection_established_event,
    )

    # Register initializers
    ds_handle_f = runtime_manager.register_runtime_initializer(data_source_initializer)
    agg_handle_f = runtime_manager.register_runtime_initializer(
        data_aggregator_initializer
    )

    await runtime_manager.start_in_process_async()  # Starts manager, initializers create runtimes
    assert ds_handle_f.done() and agg_handle_f.done()
    ds_runtime_handle = ds_handle_f.result()
    agg_runtime_handle = agg_handle_f.result()

    # Start the actual runtimes.
    # RuntimeManager with is_testing=True does not start them automatically via initialize_runtimes.
    ds_runtime_handle.start()

    # Wait for the server runtime to signal its gRPC server is ready
    ds_runtime_actual_for_event = ds_runtime_handle._get_runtime_for_test()
    assert isinstance(ds_runtime_actual_for_event, TestDataSourceRuntime)
    logging.info("Test: Waiting for DataSourceRuntime server to be ready...")
    await asyncio.wait_for(ds_runtime_actual_for_event.server_ready_event.wait(), timeout=10.0)
    logging.info("Test: DataSourceRuntime server is ready.")

    agg_runtime_handle.start()

    # --- Test Flow ---
    try:
        # 1. Wait for connection and gRPC handshake
        logging.info("Test: Waiting for connection to be established...")
        await asyncio.wait_for(connection_established_event.wait(), timeout=10.0)
        logging.info("Test: Connection established by Aggregator.")

        logging.info("Test: Waiting for server to receive client's CallerId...")
        await asyncio.wait_for(client_caller_id_event_at_server.wait(), timeout=10.0)
        logging.info("Test: Server confirmed client's CallerId.")
        assert len(server_caller_id_storage) == 1
        assert server_caller_id_storage[0].id == AGGREGATOR_CALLER_ID_STR

        logging.info("Test: Waiting for client to confirm handshake ack...")
        await asyncio.wait_for(handshake_done_event_at_client.wait(), timeout=10.0)
        logging.info("Test: Client confirmed handshake ack from server.")

        # 2. Trigger data transfer: Aggregator (client) to DataSource (server)
        logging.info("Test: Sending data from Aggregator to DataSource...")
        agg_payload_bytes = b"Aggregator_data_chunk_1"
        ts = time_pb2.ServerTimestamp(seconds=1, nanos=1)
        agg_chunk_to_send = tensor_pb2.TensorChunk(
            timestamp=ts,
            starting_index=0,
            data_bytes=agg_payload_bytes,
            compression=tensor_pb2.TensorChunk.CompressionType.NONE,
        )
        # Accessing the internal queue of the runtime via _get_runtime_for_test()
        # This is acceptable for testing purposes to inject data.
        agg_runtime_actual = agg_runtime_handle._get_runtime_for_test()
        assert isinstance(agg_runtime_actual, TestDataAggregatorRuntime)
        await agg_runtime_actual._aggregator_data_to_send_queue.put(agg_chunk_to_send)

        logging.info("Test: Waiting for DataSource to receive data...")
        received_by_ds_chunk = await asyncio.wait_for(agg_to_ds_data_q.get(), timeout=5.0)
        assert received_by_ds_chunk.data_bytes == agg_payload_bytes
        logging.info(f"Test: DataSource received: {received_by_ds_chunk.data_bytes.decode()}")
        agg_to_ds_data_q.task_done()

        # 3. Trigger data transfer: DataSource (server) to Aggregator (client)
        logging.info("Test: Sending data from DataSource to Aggregator...")
        ds_payload_bytes = b"DataSource_data_chunk_1"
        ds_chunk_to_send = tensor_pb2.TensorChunk(
            timestamp=ts,
            starting_index=100,
            data_bytes=ds_payload_bytes,
            compression=tensor_pb2.TensorChunk.CompressionType.NONE,
        )
        # Accessing the internal queue of the runtime via _get_runtime_for_test()
        ds_runtime_actual = ds_runtime_handle._get_runtime_for_test()
        assert isinstance(ds_runtime_actual, TestDataSourceRuntime)
        await ds_runtime_actual._server_data_to_send_queue.put(ds_chunk_to_send)

        logging.info("Test: Waiting for Aggregator to receive data...")
        received_by_agg_chunk = await asyncio.wait_for(ds_to_agg_data_q.get(), timeout=5.0) # ds_to_agg_data_q is correct here
        assert received_by_agg_chunk.data_bytes == ds_payload_bytes
        logging.info(f"Test: Aggregator received: {received_by_agg_chunk.data_bytes.decode()}")
        ds_to_agg_data_q.task_done() # Correct queue to mark task_done on for aggregator receiving

        # 4. Shutdown
        logging.info("Test: Shutting down runtimes...")
        if agg_runtime_handle: # Check if handle exists before stopping
            agg_runtime_handle.stop()
        if ds_runtime_handle: # Check if handle exists before stopping
            ds_runtime_handle.stop()

        await asyncio.sleep(1.0) # Allow time for graceful shutdown

        if runtime_manager: # Check if manager exists before shutdown
            runtime_manager.shutdown()
        logging.info("test_full_e2e_with_discovery_and_grpc_stream completed successfully.")
    finally:
        # Cleanup for ThreadPoolExecutor and ThreadWatcher
        if 'test_thread_pool' in locals() and test_thread_pool:
            test_thread_pool.shutdown(wait=True)

        # ThreadWatcher in tsercom typically has a stop() method that might also join.
        # It's often managed as part of other components (like RuntimeManager or GrpcServicePublisher).
        # If test_thread_watcher was started, it should be stopped.
        # However, ThreadWatcher("TestE2E") might not have been explicitly started if only passed.
        # Let's assume if it has a 'stop' method, it's safe to call.
        # RuntimeManager(is_testing=True) creates its own watcher and manages it.
        # The test_thread_watcher is passed to SplitRuntimeFactoryFactory.
        # SplitRuntimeFactoryFactory doesn't explicitly start/stop it.
        # It's typically started if it's monitoring something.
        # For safety, add a check and attempt to stop if method exists.
        if 'test_thread_watcher' in locals() and hasattr(test_thread_watcher, 'stop') and callable(getattr(test_thread_watcher, 'stop')):
            logging.info("Attempting to stop test_thread_watcher...")
            test_thread_watcher.stop() # type: ignore

        logging.info("Test resources cleanup attempted in finally block.")
