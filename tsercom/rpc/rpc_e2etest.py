import asyncio
import grpc
import pytest
import pytest_asyncio
import logging  # For server/client logging
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop,
    clear_tsercom_event_loop,
)
from collections.abc import (
    Callable,
    Awaitable,
)  # For type hinting delay_before_retry_func

# Assuming ThreadWatcher can be instantiated directly for test purposes.
# If it requires complex setup, a mock might be needed later.
from tsercom.threading.thread_watcher import ThreadWatcher

from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.util.stopable import Stopable  # Required for TInstanceType
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
        clear_tsercom_event_loop()  # Clean up global event loop


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


# --- New Test Service for Error and Timeout Scenarios ---

# Define service and method names for the new test service
ERROR_TIMEOUT_SERVICE_NAME = "dtp.ErrorAndTimeoutTestService"
TRIGGER_ERROR_METHOD_NAME = "TriggerError"
DELAYED_RESPONSE_METHOD_NAME = "DelayedResponse"

FULL_TRIGGER_ERROR_METHOD_PATH = (
    f"/{ERROR_TIMEOUT_SERVICE_NAME}/{TRIGGER_ERROR_METHOD_NAME}"
)
FULL_DELAYED_RESPONSE_METHOD_PATH = (
    f"/{ERROR_TIMEOUT_SERVICE_NAME}/{DELAYED_RESPONSE_METHOD_NAME}"
)


class ErrorAndTimeoutTestServiceServer:
    """
    A gRPC service implementation for testing error handling and timeouts.
    It reuses TestConnectionCall and TestConnectionResponse for simplicity.
    """

    async def TriggerError(
        self, request: TestConnectionCall, context: grpc.aio.ServicerContext
    ) -> TestConnectionResponse:
        """
        Simulates a server-side error. For now, it always returns INVALID_ARGUMENT.
        Could be extended to take error type from request if needed.
        """
        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME}: TriggerError called. Returning INVALID_ARGUMENT."
        )
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("This is a simulated error from TriggerError.")
        # When an error is set on the context, returning a response message is optional,
        # as the error itself is the primary information conveyed.
        # However, gRPC Python expects a response message type to be returned or an exception raised.
        # Returning an empty response of the correct type is safe.
        return TestConnectionResponse()

    async def DelayedResponse(
        self, request: TestConnectionCall, context: grpc.aio.ServicerContext
    ) -> TestConnectionResponse:
        """
        Simulates a delay before sending a response.
        The actual delay duration could be passed in the request in a real scenario,
        but for this test, we'll use a fixed delay, and the client will try to timeout sooner.
        """
        delay_duration_seconds = 2  # Server will delay for 2 seconds
        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME}: DelayedResponse called. Delaying for {delay_duration_seconds}s."
        )
        await asyncio.sleep(delay_duration_seconds)
        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME}: Delay complete. Sending response."
        )
        return TestConnectionResponse()


@pytest_asyncio.fixture  # Make sure to use pytest_asyncio.fixture for async fixtures
async def error_timeout_test_server():
    """
    Pytest fixture to start and stop the ErrorAndTimeoutTestServiceServer.
    """
    watcher = ThreadWatcher()
    # Using the TestGrpcServicePublisher subclass to capture the chosen port
    service_publisher = TestGrpcServicePublisher(
        watcher, port=0, addresses="127.0.0.1"
    )

    # Instantiate the new service implementation
    service_impl = ErrorAndTimeoutTestServiceServer()

    def connect_call(server: grpc.aio.Server):
        """Callback to add RPC handlers for ErrorAndTimeoutTestServiceServer to the gRPC server."""

        # Handler for TriggerError method
        trigger_error_rpc_handler = grpc.unary_unary_rpc_method_handler(
            service_impl.TriggerError,
            request_deserializer=TestConnectionCall.FromString,  # Reusing existing messages
            response_serializer=TestConnectionResponse.SerializeToString,
        )

        # Handler for DelayedResponse method
        delayed_response_rpc_handler = grpc.unary_unary_rpc_method_handler(
            service_impl.DelayedResponse,
            request_deserializer=TestConnectionCall.FromString,  # Reusing existing messages
            response_serializer=TestConnectionResponse.SerializeToString,
        )

        # Generic handler for the entire service
        generic_handler = grpc.method_handlers_generic_handler(
            ERROR_TIMEOUT_SERVICE_NAME,  # e.g., "dtp.ErrorAndTimeoutTestService"
            {
                TRIGGER_ERROR_METHOD_NAME: trigger_error_rpc_handler,
                DELAYED_RESPONSE_METHOD_NAME: delayed_response_rpc_handler,
            },
        )
        server.add_generic_rpc_handlers((generic_handler,))
        logger.info(
            f"gRPC handlers added for service '{ERROR_TIMEOUT_SERVICE_NAME}'."
        )

    # Setup for event loop if tsercom specific loop management is used (based on previous fixes)
    # This was identified as necessary in the previous test runs for ThreadWatcher
    original_loop = asyncio.get_event_loop_policy().get_event_loop()
    # It's important to use a try-finally block for setting/clearing the tsercom loop
    # to ensure cleanup even if errors occur during fixture setup.
    is_tsercom_loop_managed = False  # Initialize before try block
    try:
        # Attempt to import and set the tsercom event loop.
        # The `replace_policy=True` argument for set_tsercom_event_loop is not in the actual signature
        # based on previous exploration of global_event_loop.py. Removing it.
        # If the global_event_loop.py was intended to have it, this might be a point of version mismatch.
        # For now, adhering to the known signature.
        set_tsercom_event_loop(original_loop)  # Removed replace_policy=True
        is_tsercom_loop_managed = True
    except RuntimeError as e:
        # This can happen if "Only one Global Event Loop may be set"
        logger.warning(
            f"Could not set tsercom global event loop: {e}. This might be okay if already set by another fixture."
        )
        # We don't re-raise here as the loop might have been set by async_test_server if used in the same test session.
        # If it's critical and not set, other parts of tsercom might fail.
    except ImportError:
        logger.warning(
            "tsercom event loop management functions not found. Proceeding with default loop."
        )
        # is_tsercom_loop_managed remains False

    try:
        # Start the server (start_async is not an async def function)
        service_publisher.start_async(connect_call)

        port = service_publisher.chosen_port
        if port is None:
            await asyncio.sleep(0.1)  # Brief wait for port assignment
            port = service_publisher.chosen_port

        if port is None:
            pytest.fail(
                f"{ERROR_TIMEOUT_SERVICE_NAME} server started but failed to capture the chosen port."
            )

        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME} server started on 127.0.0.1:{port}"
        )
        yield "127.0.0.1", port  # Yield host and port

    finally:
        logger.info(f"Stopping {ERROR_TIMEOUT_SERVICE_NAME} server...")
        if (
            hasattr(service_publisher, "_GrpcServicePublisher__server")
            and service_publisher._GrpcServicePublisher__server is not None
        ):  # type: ignore
            actual_server_obj = service_publisher._GrpcServicePublisher__server  # type: ignore
            if isinstance(actual_server_obj, grpc.aio.Server):
                await actual_server_obj.stop(grace=1)
                logger.info(
                    f"Async {ERROR_TIMEOUT_SERVICE_NAME} server stop completed."
                )
            else:
                service_publisher.stop()  # For synchronous server if type was different
        else:
            logger.info(
                f"{ERROR_TIMEOUT_SERVICE_NAME} server object not found, skipping explicit stop."
            )

        if is_tsercom_loop_managed:
            try:
                # Attempt to clear the tsercom event loop only if this fixture instance set it.
                # This check might be simplified if we are sure about fixture scopes and execution order.
                # If set_tsercom_event_loop raises RuntimeError because loop is already set,
                # this fixture instance did not set it, so it should not clear it.
                # The current logic sets `is_tsercom_loop_managed = True` only if set_tsercom_event_loop succeeds.
                clear_tsercom_event_loop()
            except (
                ImportError
            ):  # Should not happen if is_tsercom_loop_managed is True
                logger.error(
                    "Failed to import clear_tsercom_event_loop for cleanup when expected."
                )
            except (
                RuntimeError
            ) as e:  # For example, if loop was already cleared or not set by this instance
                logger.warning(f"Issue clearing tsercom event loop: {e}")

        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME} server fixture cleanup complete."
        )


@pytest.mark.asyncio
async def test_server_returns_grpc_error(error_timeout_test_server):
    """
    Tests that the server can return a specific gRPC error,
    and the client correctly receives and identifies it.
    """
    host, port = error_timeout_test_server
    logger.info(
        f"Test client connecting to error_timeout_test_server at {host}:{port} for error test."
    )

    channel_factory = InsecureGrpcChannelFactory()
    channel_info: ChannelInfo | None = None

    try:
        channel_info = await channel_factory.find_async_channel(host, port)
        assert (
            channel_info is not None
        ), "Failed to create client channel for error test"
        assert (
            channel_info.channel is not None
        ), "ChannelInfo has no channel object for error test"

        logger.info(f"Client channel created to {host}:{port} for error test.")

        request = TestConnectionCall()  # Reusing TestConnectionCall

        logger.info(f"Client calling {FULL_TRIGGER_ERROR_METHOD_PATH}")

        with pytest.raises(grpc.aio.AioRpcError) as e_info:
            await channel_info.channel.unary_unary(
                FULL_TRIGGER_ERROR_METHOD_PATH,
                request_serializer=TestConnectionCall.SerializeToString,
                response_deserializer=TestConnectionResponse.FromString,
            )(request)

        # Verify the details of the AioRpcError
        assert (
            e_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        ), f"Expected INVALID_ARGUMENT, but got {e_info.value.code()}"
        assert (
            "simulated error from TriggerError" in e_info.value.details()
        ), f"Error details mismatch: {e_info.value.details()}"

        logger.info(
            f"Correctly received gRPC error: {e_info.value.code()} - {e_info.value.details()}"
        )

    except Exception as e:
        # Catch any other unexpected errors during the test itself
        logger.error(
            f"An unexpected error occurred during the error handling test: {type(e).__name__} - {e}"
        )
        pytest.fail(
            f"An unexpected error occurred in test_server_returns_grpc_error: {e}"
        )
    finally:
        if channel_info and channel_info.channel:
            logger.info("Closing client channel in error test.")
            await channel_info.channel.close()
            logger.info("Client channel closed in error test.")


@pytest.mark.asyncio
async def test_client_handles_timeout(error_timeout_test_server):
    """
    Tests that the client correctly handles a timeout when the server's
    response is too slow.
    """
    host, port = error_timeout_test_server
    logger.info(
        f"Test client connecting to error_timeout_test_server at {host}:{port} for timeout test."
    )

    channel_factory = InsecureGrpcChannelFactory()
    channel_info: ChannelInfo | None = None

    # The ErrorAndTimeoutTestServiceServer.DelayedResponse is hardcoded to delay for 2 seconds.
    # We'll set the client timeout to be shorter than that.
    client_timeout_seconds = 0.5

    try:
        channel_info = await channel_factory.find_async_channel(host, port)
        assert (
            channel_info is not None
        ), "Failed to create client channel for timeout test"
        assert (
            channel_info.channel is not None
        ), "ChannelInfo has no channel object for timeout test"

        logger.info(
            f"Client channel created to {host}:{port} for timeout test."
        )

        request = TestConnectionCall()  # Reusing TestConnectionCall

        logger.info(
            f"Client calling {FULL_DELAYED_RESPONSE_METHOD_PATH} with timeout {client_timeout_seconds}s."
        )

        with pytest.raises(grpc.aio.AioRpcError) as e_info:
            method_callable = channel_info.channel.unary_unary(
                FULL_DELAYED_RESPONSE_METHOD_PATH,
                request_serializer=TestConnectionCall.SerializeToString,
                response_deserializer=TestConnectionResponse.FromString,
            )
            await method_callable(request, timeout=client_timeout_seconds) # Apply client-side timeout

        # Verify the details of the AioRpcError
        assert (
            e_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        ), f"Expected DEADLINE_EXCEEDED, but got {e_info.value.code()}"

        logger.info(
            f"Correctly received gRPC DEADLINE_EXCEEDED error: {e_info.value.code()}"
        )

    except Exception as e:
        # Catch any other unexpected errors during the test itself
        logger.error(
            f"An unexpected error occurred during the timeout test: {type(e).__name__} - {e}"
        )
        pytest.fail(
            f"An unexpected error occurred in test_client_handles_timeout: {e}"
        )
    finally:
        if channel_info and channel_info.channel:
            logger.info("Closing client channel in timeout test.")
            await channel_info.channel.close()
            logger.info("Client channel closed in timeout test.")


# Using a fixed port for the retrier test simplifies port management during restarts.
# Ensure this port is unlikely to conflict with other tests or services.
RETRIER_TEST_FIXED_PORT = 50052


@pytest_asyncio.fixture
async def retrier_server_controller():
    """
    Pytest fixture that provides control over a gRPC server's lifecycle
    (stop, start/restart) for testing client retrier mechanisms.
    It hosts the basic AsyncTestConnectionServer on a fixed port.
    """
    # watcher = ThreadWatcher() # Not strictly used by this simple server, but good practice
    # For the F841 fix, ThreadWatcher instance is not created if not used.
    # If it were to be used (e.g. passed to TestableDisconnectionRetrier if that took a watcher for its own threads),
    # it would be reinstated. For this simple gRPC server fixture, it's not essential.

    # Store the server instance in a list/dict to allow reassignment in closures
    server_container = {"instance": None}

    # Define the servicer and how to connect it
    service_impl = AsyncTestConnectionServer()

    def _add_servicer_to_server(s: grpc.aio.Server):
        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            service_impl.TestConnection,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )
        generic_handler = grpc.method_handlers_generic_handler(
            TEST_SERVICE_NAME,  # Using the original TestConnectionService
            {TEST_METHOD_NAME: rpc_method_handler},
        )
        s.add_generic_rpc_handlers((generic_handler,))

    async def _start_new_server_instance():
        # Stop existing server if it's running (relevant for restart)
        if server_container["instance"] is not None:
            try:
                # Ensure it's fully stopped before trying to reuse the port
                await server_container["instance"].stop(grace=0.1)
                logger.info(
                    f"Retrier test server (old instance) stopped before restart attempt on port {RETRIER_TEST_FIXED_PORT}."
                )
            except Exception as e:
                logger.warning(
                    f"Error stopping old server instance during restart: {e}"
                )
            server_container["instance"] = None  # Clear old instance

        # Short delay to ensure port is released, especially on some OS/CI environments
        await asyncio.sleep(0.2)

        new_server = (
            grpc.aio.server()
        )  # Removed interceptors for simplicity for this basic server
        _add_servicer_to_server(new_server)
        try:
            new_server.add_insecure_port(
                f"127.0.0.1:{RETRIER_TEST_FIXED_PORT}"
            )
        except Exception as e:
            logger.error(
                f"Failed to bind retrier test server to port {RETRIER_TEST_FIXED_PORT}: {e}"
            )
            pytest.fail(
                f"Retrier test server failed to bind to port {RETRIER_TEST_FIXED_PORT}: {e}"
            )
            return  # Should not be reached due to pytest.fail

        await new_server.start()
        server_container["instance"] = new_server
        logger.info(
            f"Retrier test server started/restarted on 127.0.0.1:{RETRIER_TEST_FIXED_PORT}"
        )

    # Initial server start
    # Manage tsercom event loop as done in other fixtures
    original_loop = asyncio.get_event_loop_policy().get_event_loop()
    is_tsercom_loop_managed = False
    try:
        set_tsercom_event_loop(original_loop)
        is_tsercom_loop_managed = True
    except RuntimeError as e:
        logger.warning(
            f"Retrier fixture: Could not set tsercom global event loop: {e}."
        )
    except ImportError:
        logger.warning(
            "Retrier fixture: tsercom event loop management functions not found."
        )

    async def _stop_server_instance():
        if server_container["instance"]:
            await server_container["instance"].stop(grace=0)  # Quick stop
            logger.info(
                f"Retrier test server stopped on port {RETRIER_TEST_FIXED_PORT}."
            )
        else:
            logger.info(
                "Retrier test server stop called but no instance was running."
            )

    await _start_new_server_instance()  # Start the server for the first time

    controller = {
        "host": "127.0.0.1",
        "port": RETRIER_TEST_FIXED_PORT,
        "get_port": lambda: RETRIER_TEST_FIXED_PORT,  # Port is fixed
        "stop_server": _stop_server_instance,
        "start_server": _start_new_server_instance,  # Function to (re)start the server
    }

    try:
        yield controller
    finally:
        logger.info(
            f"Cleaning up retrier_server_controller. Stopping server on port {RETRIER_TEST_FIXED_PORT} if running."
        )
        if server_container["instance"]:
            await server_container["instance"].stop(grace=1)

        if is_tsercom_loop_managed:
            try:
                clear_tsercom_event_loop()
            except Exception as e:  # Broad exception for cleanup
                logger.warning(
                    f"Retrier fixture: Issue clearing tsercom event loop: {e}"
                )
        logger.info("Retrier_server_controller cleanup complete.")


# 1. Helper class to make grpc.aio.Channel Stopable
class StopableChannelWrapper(Stopable):
    """Wraps a grpc.aio.Channel to make it conform to the Stopable interface."""

    def __init__(self, channel: grpc.aio.Channel):
        if not isinstance(channel, grpc.aio.Channel):
            raise TypeError("Provided channel is not a grpc.aio.Channel")
        self._channel = channel
        self._active = True  # Assume active upon creation

    async def stop(self) -> None:
        if self._active:
            logger.info("StopableChannelWrapper: stopping (closing) channel.")
            await self._channel.close()
            self._active = False
            logger.info("StopableChannelWrapper: channel closed.")
        else:
            logger.info(
                "StopableChannelWrapper: stop() called but channel already inactive/closed."
            )

    @property
    def channel(self) -> grpc.aio.Channel:
        return self._channel

    @property
    def is_active(self) -> bool:
        # This is a simple view, grpc.aio.Channel doesn't have a direct is_active property.
        # We infer based on whether stop() has been called on this wrapper.
        return self._active


# 2. Concrete subclass of ClientDisconnectionRetrier
class TestableDisconnectionRetrier(
    ClientDisconnectionRetrier[StopableChannelWrapper]
):
    """
    A testable subclass of ClientDisconnectionRetrier that manages StopableChannelWrapper instances.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        server_controller_fixture_data: dict,  # Contains host, get_port
        event_loop: asyncio.AbstractEventLoop | None = None,
        max_retries: int = 3,  # Configure for test speed
        # Provide a shorter delay for testing purposes
        delay_before_retry_func: (
            Callable[[], Awaitable[None]] | None
        ) = lambda: asyncio.sleep(0.2),
    ):
        self._server_host = server_controller_fixture_data["host"]
        # get_port is a callable that returns the current port
        self._get_server_port = server_controller_fixture_data["get_port"]
        self._channel_factory = InsecureGrpcChannelFactory()

        # Ensure default retry delay is not None if not provided
        effective_delay_func = delay_before_retry_func or (
            lambda: asyncio.sleep(0.2)
        )

        super().__init__(
            watcher=watcher,
            event_loop=event_loop,
            max_retries=max_retries,
            delay_before_retry_func=effective_delay_func,
        )
        self._managed_instance_wrapper: StopableChannelWrapper | None = None

    async def _connect(self) -> StopableChannelWrapper:
        """Connects to the server and returns a StopableChannelWrapper."""
        current_port = self._get_server_port()
        logger.info(
            f"TestableDisconnectionRetrier: Attempting to connect to {self._server_host}:{current_port}..."
        )

        # InsecureGrpcChannelFactory.find_async_channel can return ChannelInfo or None.
        # If None, it means connection failed (e.g., server down).
        # We need to raise an error that is_server_unavailable_error_func can catch.
        channel_info = await self._channel_factory.find_async_channel(
            self._server_host, current_port
        )

        if channel_info is None or channel_info.channel is None:
            logger.warning(
                f"TestableDisconnectionRetrier: Connection failed to {self._server_host}:{current_port}."
            )
            # Simulate a gRPC error that is_server_unavailable_error would recognize
            # Construct a dummy AioRpcError (normally grpc internals do this)
            # This is a bit of a hack; ideally find_async_channel would raise this.
            # For now, let's assume it might return None, and we convert to an error.
            # A more robust _connect would ensure an actual grpc.aio.AioRpcError is raised
            # by attempting a quick RPC call or health check if find_async_channel is too lenient.
            # For now, we'll rely on find_async_channel's behavior or manually raise.
            # Let's assume find_async_channel itself can raise AioRpcError with UNAVAILABLE
            # if connection is actively refused or times out quickly. If it just returns None
            # for a passive failure, the retrier might not see the right error type.
            # The default is_server_unavailable_error checks for UNAVAILABLE and DEADLINE_EXCEEDED.
            raise grpc.aio.AioRpcError(  # Manually raising to ensure correct error type for retrier
                grpc.StatusCode.UNAVAILABLE,
                initial_metadata=None,
                trailing_metadata=None,
                details=f"Connection failed to {self._server_host}:{current_port}",
            )

        logger.info(
            f"TestableDisconnectionRetrier: Connected to {self._server_host}:{current_port}. Wrapping channel."
        )
        self._managed_instance_wrapper = StopableChannelWrapper(
            channel_info.channel
        )
        return self._managed_instance_wrapper

    def get_current_channel_from_managed_instance(
        self,
    ) -> grpc.aio.Channel | None:
        """Provides access to the channel within the currently managed StopableChannelWrapper."""
        # Access the private __instance from the parent class
        # This is generally not ideal but necessary if the parent doesn't expose it.
        # A better way would be if ClientDisconnectionRetrier had a method like get_instance()
        current_managed_wrapper = self._ClientDisconnectionRetrier__instance  # type: ignore
        if current_managed_wrapper and isinstance(
            current_managed_wrapper, StopableChannelWrapper
        ):
            if current_managed_wrapper.is_active:
                return current_managed_wrapper.channel
        return None


# 3. The E2E Test Function
@pytest.mark.asyncio
async def test_client_retrier_reconnects(retrier_server_controller):
    """
    Tests ClientDisconnectionRetrier's ability to reconnect after server outage.
    """
    server_ctrl = retrier_server_controller
    watcher = ThreadWatcher()  # For the retrier

    current_event_loop = asyncio.get_event_loop()
    retrier = TestableDisconnectionRetrier(
        watcher,
        server_ctrl,  # Pass the whole controller dict which includes host and get_port
        event_loop=current_event_loop,
        max_retries=3,
    )

    # Initial connection
    logger.info("Attempting initial connection via retrier.start()...")
    assert await retrier.start(), "Retrier failed initial start"

    initial_channel = retrier.get_current_channel_from_managed_instance()
    assert (
        initial_channel is not None
    ), "Failed to get channel after initial retrier start"
    logger.info(f"Initial connection successful. Channel: {initial_channel}")

    # Make a successful call
    try:
        response = await initial_channel.unary_unary(
            FULL_METHOD_PATH,  # Using the original TestConnectionService path
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(TestConnectionCall())
        assert isinstance(response, TestConnectionResponse)
        logger.info("Initial gRPC call successful.")
    except grpc.aio.AioRpcError as e:
        pytest.fail(
            f"Initial gRPC call failed unexpectedly: {e.code()} - {e.details()}"
        )

    # Simulate server outage
    logger.info("Simulating server outage: Stopping server...")
    await server_ctrl["stop_server"]()
    # Allow some time for the server to fully stop and release port
    await asyncio.sleep(0.5)
    logger.info("Server stopped.")

    # Attempt a call that should fail and trigger _on_disconnect
    logger.info(
        "Attempting gRPC call during server outage (expected to fail initially)..."
    )
    call_succeeded_during_outage = False
    try:
        # Use the same initial_channel object. The retrier should manage its underlying connection.
        # If the channel object itself becomes unusable, this test design might need adjustment.
        # The idea is that the retrier's _connect will provide a *new* channel to its managed instance.
        # So, after _on_disconnect, we must fetch the new channel via get_current_channel_from_managed_instance().

        # This call is expected to fail.
        await initial_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(TestConnectionCall())
        call_succeeded_during_outage = True  # Should not be reached
    except grpc.aio.AioRpcError as e:
        logger.info(
            f"gRPC call failed as expected during outage: {e.code()} - {e.details()}"
        )
        assert (
            e.code() == grpc.StatusCode.UNAVAILABLE
        ), f"Expected UNAVAILABLE during outage, got {e.code()}"

        # Notify the retrier about the disconnection.
        # This will trigger its retry logic in the background.
        logger.info("Notifying retrier._on_disconnect()...")
        # _on_disconnect will try to reconnect. We run it concurrently.
        on_disconnect_task = current_event_loop.create_task(retrier._on_disconnect(e))

        # While retrier is attempting to reconnect, restart the server.
        logger.info("Restarting server while retrier is in its retry loop...")
        await asyncio.sleep(
            0.1
        )  # Give _on_disconnect a moment to start its first delay/retry
        await server_ctrl["start_server"]()
        logger.info("Server restarted.")

        # Wait for the _on_disconnect task to complete.
        # It should eventually succeed in reconnecting because the server is back.
        await on_disconnect_task
        logger.info("retrier._on_disconnect() task completed.")

    if call_succeeded_during_outage:
        pytest.fail(
            "gRPC call unexpectedly succeeded during server outage before retrier acted."
        )

    # After retrier has reconnected, get the new channel and make a call
    logger.info("Attempting gRPC call after simulated reconnection...")
    reconnected_channel = retrier.get_current_channel_from_managed_instance()
    assert (
        reconnected_channel is not None
    ), "Failed to get channel after retrier's reconnection attempt"

    if initial_channel is reconnected_channel:
        logger.warning(
            "Retrier is using the exact same channel object. This might be okay if the channel object itself can recover, or if _connect re-established its internal state."
        )
    else:
        logger.info(
            "Retrier provided a new channel object after reconnection."
        )

    try:
        response_after_reconnect = await reconnected_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(TestConnectionCall())
        assert isinstance(response_after_reconnect, TestConnectionResponse)
        logger.info("gRPC call after reconnection successful.")
    except grpc.aio.AioRpcError as e:
        pytest.fail(
            f"gRPC call after reconnection failed: {e.code()} - {e.details()}"
        )

    # Clean up
    logger.info("Stopping retrier...")
    await retrier.stop()
    logger.info("Test complete.")
