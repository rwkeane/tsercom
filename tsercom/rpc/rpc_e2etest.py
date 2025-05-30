import asyncio
import grpc
import pytest
import pytest_asyncio
import logging  # For server/client logging
from tsercom.threading.aio.global_event_loop import set_tsercom_event_loop, clear_tsercom_event_loop

# Assuming ThreadWatcher can be instantiated directly for test purposes.
# If it requires complex setup, a mock might be needed later.
from tsercom.threading.thread_watcher import ThreadWatcher

from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.rpc.endpoints.test_connection_server import (
    AsyncTestConnectionServer,
)
from tsercom.rpc.proto.generated.v1_71.common_pb2 import (
    TestConnectionCall,
    TestConnectionResponse,
)

# Note: Using v1_71 based on previous exploration. If common_pb2 is not versioned like this
# or the path is different, adjust accordingly.
# For example, if it's from tsercom.rpc.proto directly (less likely for generated):
# from tsercom.rpc.proto import TestConnectionCall, TestConnectionResponse

from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.common.channel_info import (
    ChannelInfo,
)  # Expected by InsecureGrpcChannelFactory

# Configure basic logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder for the service name, to be confirmed or adjusted
TEST_SERVICE_NAME = "dtp.TestConnectionService"
TEST_METHOD_NAME = "TestConnection"
FULL_METHOD_PATH = f"/{TEST_SERVICE_NAME}/{TEST_METHOD_NAME}"

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio


class TestGrpcServicePublisher(GrpcServicePublisher):
    """Subclass of GrpcServicePublisher to capture the assigned port."""

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        addresses: str | list[str] | None = None,
    ):
        super().__init__(watcher, port, addresses)
        self._chosen_port: int | None = None

    def _connect(self) -> bool:
        """Binds the gRPC server and captures the chosen port."""
        # Connect to a port.
        worked = 0
        # Ensure __server is not None before proceeding
        if self._GrpcServicePublisher__server is None:  # type: ignore
            logger.error("Server object not initialized before _connect")
            return False

        # Use 'localhost' to ensure we bind to an interface accessible for local testing
        # and to simplify port capture. The original uses get_all_address_strings().
        # For E2E testing, 'localhost' or '127.0.0.1' is usually sufficient.
        # If specific addresses are needed, this might need adjustment or configuration.
        addresses_to_bind = ["127.0.0.1"]
        if isinstance(self._GrpcServicePublisher__addresses, str):  # type: ignore
            addresses_to_bind = [self._GrpcServicePublisher__addresses]  # type: ignore
        elif self._GrpcServicePublisher__addresses:  # type: ignore
            # If addresses were explicitly passed, use them, but this might be complex
            # if it includes non-local IPs. For now, prioritize simplicity.
            # This logic might need to be more robust if the default get_all_address_strings()
            # is strictly required and returns multiple addresses that must all be bound.
            # For now, we'll try to bind to the first one or '127.0.0.1'.
            pass  # Keep addresses_to_bind as ['127.0.0.1'] for simplicity

        for address in addresses_to_bind:  # Use the simplified list
            try:
                # The port passed to add_insecure_port is self._GrpcServicePublisher__port
                # which could be 0 for dynamic port assignment.
                port_out = self._GrpcServicePublisher__server.add_insecure_port(  # type: ignore
                    f"{address}:{self._GrpcServicePublisher__port}"  # type: ignore
                )
                if (
                    self._chosen_port is None
                ):  # Capture the first successfully bound port
                    self._chosen_port = port_out
                logger.info(
                    f"Running gRPC Server on {address}:{port_out} (expected: {self._GrpcServicePublisher__port})"  # type: ignore
                )
                worked += 1
                # For E2E, binding to one accessible address (like 127.0.0.1) is usually enough.
                # Breaking after the first successful bind simplifies port management.
                break
            except Exception as e:
                if isinstance(e, AssertionError):
                    self._GrpcServicePublisher__watcher.on_exception_seen(e)  # type: ignore
                    raise e
                logger.warning(
                    f"Failed to bind gRPC server to {address}:{self._GrpcServicePublisher__port}. Error: {e}"  # type: ignore
                )
                continue

        if worked == 0:
            logger.error("FAILED to host gRPC Service on any address.")
            return False

        if self._chosen_port is None and worked > 0:
            # Fallback if break was not hit but binding worked (e.g. if using original multiple address logic)
            # This part of the logic might be complex if multiple addresses from get_all_address_strings() are used.
            # However, with the current simplified '127.0.0.1' approach, _chosen_port should be set.
            logger.warning(
                "Port bound, but _chosen_port not explicitly captured. This may occur if binding logic changes."
            )
            # Attempt to re-query or use a fixed port as a last resort if this path is hit.
            # For now, this is a warning. The test might fail if port is None.

        return worked != 0

    @property
    def chosen_port(self) -> int | None:
        return self._chosen_port


@pytest_asyncio.fixture
async def async_test_server():
    """Pytest fixture to start and stop the AsyncTestConnectionServer."""
    watcher = ThreadWatcher()  # Manages threads for the server
    # Use port 0 to let the OS pick an available port
    # Using the subclass to capture the chosen port
    service_publisher = TestGrpcServicePublisher(
        watcher, port=0, addresses="127.0.0.1"
    )

    async_server_impl = AsyncTestConnectionServer()

    def connect_call(server: grpc.aio.Server):
        """Callback to add RPC handlers to the gRPC server."""
        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            async_server_impl.TestConnection,  # The actual method implementation
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )
        generic_handler = grpc.method_handlers_generic_handler(
            TEST_SERVICE_NAME,  # Assumed service name "dtp.TestConnectionService"
            {
                TEST_METHOD_NAME: rpc_method_handler
            },  # Method name "TestConnection"
        )
        server.add_generic_rpc_handlers((generic_handler,))
        logger.info(
            f"gRPC handlers added for service '{TEST_SERVICE_NAME}' method '{TEST_METHOD_NAME}'."
        )

    # server_task = None # Commented out as it's unused
    current_loop = asyncio.get_event_loop()
    set_tsercom_event_loop(current_loop)
    try:
        # Start the server. GrpcServicePublisher.start_async schedules the server start.
        # It's not an async def function itself, so we don't await it here.
        service_publisher.start_async(connect_call)

        # Ensure the chosen port is available by polling briefly.
        # __start_async_impl (called by start_async) will set the port.
        port = service_publisher.chosen_port
        if port is None:
            # Attempt a brief wait in case port assignment is slightly delayed
            await asyncio.sleep(0.1)
            port = service_publisher.chosen_port

        if port is None:
            pytest.fail(
                "Server started but failed to capture the chosen port."
            )

        logger.info(f"AsyncTestConnectionServer started on 127.0.0.1:{port}")
        yield "127.0.0.1", port  # Yield host and port

    finally:
        logger.info("Stopping AsyncTestConnectionServer...")
        # GrpcServicePublisher.stop() is synchronous.
        # For an async server started with start_async, ensure proper async shutdown if available.
        # The current GrpcServicePublisher.stop() calls self.__server.stop(grace=None)
        # For grpc.aio.Server, server.stop(grace) is an awaitable,
        # but GrpcServicePublisher.__server is type hinted as grpc.Server.
        # This might need careful handling if issues arise during shutdown.
        # For now, assuming GrpcServicePublisher.stop() is adequate.

        # We need to ensure that server.stop() is called correctly.
        # If __server is grpc.aio.Server, then stop is a coroutine.
        # GrpcServicePublisher.stop() is not async.
        # This is a potential issue in the original GrpcServicePublisher for async servers.
        # For this E2E test, we'll call it as is.
        # If __server is indeed an grpc.aio.Server, its stop() method should be awaited.
        # Let's assume for now the existing stop() method in GrpcServicePublisher
        # correctly handles stopping either sync or async server by making blocking call.

        # Check if server object exists and has stop method
        if (
            hasattr(service_publisher, "_GrpcServicePublisher__server")
            and service_publisher._GrpcServicePublisher__server is not None
        ):  # type: ignore
            actual_server_obj = service_publisher._GrpcServicePublisher__server  # type: ignore
            if isinstance(actual_server_obj, grpc.aio.Server):
                logger.info("Attempting graceful async server stop...")
                await actual_server_obj.stop(grace=1)  # 1 second grace period
                logger.info("Async server stop completed.")
            else:
                # Synchronous server or unexpected type, use original stop
                logger.info(
                    "Using GrpcServicePublisher's default stop method."
                )
                service_publisher.stop()
        else:
            logger.info(
                "Server object not found or already None, skipping explicit stop call via fixture."
            )

        # watcher.stop() # ThreadWatcher does not have stop()
        logger.info("AsyncTestConnectionServer stopped.")
        clear_tsercom_event_loop() # Clean up global event loop


@pytest.mark.asyncio
async def test_grpc_connection_e2e(async_test_server):
    """
    E2E test for gRPC connection using AsyncTestConnectionServer.
    - Starts a server using the async_test_server fixture.
    - Creates a client that connects to this server.
    - Sends a TestConnectionCall.
    - Verifies a TestConnectionResponse is received.
    """
    host, port = async_test_server
    logger.info(f"Test client connecting to server at {host}:{port}")

    channel_factory = InsecureGrpcChannelFactory()
    channel_info: ChannelInfo | None = (
        None  # Ensure channel_info is defined for finally block
    )

    try:
        # Establish a connection to the server
        channel_info = await channel_factory.find_async_channel(host, port)
        assert channel_info is not None, "Failed to create client channel"
        assert (
            channel_info.channel is not None
        ), "ChannelInfo has no channel object"

        logger.info(f"Client channel created to {host}:{port}")

        # Prepare the request message (empty for TestConnectionCall)
        request = TestConnectionCall()
        logger.info(f"Sending request to {FULL_METHOD_PATH}")

        # Make the gRPC call using the generic client approach
        # The method path is /<package>.<Service>/<Method>
        response = await channel_info.channel.unary_unary(
            FULL_METHOD_PATH,  # e.g., "/dtp.TestConnectionService/TestConnection"
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(request)

        logger.info(f"Received response: {response}")

        # Verify the response
        assert isinstance(
            response, TestConnectionResponse
        ), f"Unexpected response type: {type(response)}"

        logger.info("E2E test assertions passed.")

    except grpc.aio.AioRpcError as e:
        logger.error(f"gRPC call failed: {e.code()} - {e.details()}")
        pytest.fail(f"gRPC call failed: {e.details()}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}")
        pytest.fail(f"An unexpected error occurred: {e}")
    finally:
        if channel_info and channel_info.channel:
            logger.info("Closing client channel.")
            await channel_info.channel.close()
            logger.info("Client channel closed.")
