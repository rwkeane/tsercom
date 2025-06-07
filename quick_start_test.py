# quick_start_test.py
# This script demonstrates a basic client-server interaction using tsercom.
# It sets up a simple gRPC service, starts it, and then a client connects
# to it, sends a request, and receives a response.

import asyncio
import grpc  # For gRPC types and channel creation
import logging  # For basic logging

# Import necessary components from the tsercom library
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.rpc.proto import (
    TestConnectionCall,
    TestConnectionResponse,
)  # For TestConnectionCall and TestConnectionResponse
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop,
    clear_tsercom_event_loop,
)

# --- Configuration ---
# Define a service name and method name for the gRPC interaction.
# These are arbitrary but must be consistent between server and client.
_SERVICE_NAME = "tsercom.quickstart.SimpleService"
_METHOD_NAME = "Communicate"
_FULL_METHOD_PATH = f"/{_SERVICE_NAME}/{_METHOD_NAME}"
_SERVER_ADDRESS = "127.0.0.1"  # Localhost for this example
_SERVER_PORT = 50051  # A common port for gRPC examples

# Configure basic logging to see output from the script.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Server-Side Implementation ---


class SimpleService:
    """
    A simple service implementation.
    This class defines the methods that will be exposed via gRPC.
    """

    async def Communicate(
        self, request: TestConnectionCall, context: grpc.aio.ServicerContext
    ) -> TestConnectionResponse:
        """
        The gRPC method implemented by this service.
        It receives a TestConnectionCall and returns a TestConnectionResponse.
        For this example, the request and response are empty, but it demonstrates
        the communication flow.
        """
        logger.info(f"Server: Communicate method called by {context.peer()}")
        # In a real application, you would process the request here.
        # Since TestConnectionCall is empty, there's no data to inspect from it directly.
        # Similarly, TestConnectionResponse is empty, so we return an empty instance.
        return TestConnectionResponse()


async def run_server(stop_event: asyncio.Event) -> None:
    """
    Sets up and runs the gRPC server.
    """
    logger.info("Server: Initializing...")
    # ThreadWatcher is required by GrpcServicePublisher for managing threads.
    thread_watcher = ThreadWatcher()
    # GrpcServicePublisher is used to host and manage the gRPC service.
    # We specify a port (0 means assign a free port, but we'll use a fixed one for simplicity here)
    # and the address to bind to.
    service_publisher = GrpcServicePublisher(
        watcher=thread_watcher, port=_SERVER_PORT, addresses=_SERVER_ADDRESS
    )

    # Instantiate our simple service.
    simple_service_impl = SimpleService()

    # The connect_call callback is used by GrpcServicePublisher to add RPC method handlers
    # to the gRPC server instance.
    def connect_call(server: grpc.aio.Server) -> None:
        logger.info("Server: Adding RPC handlers...")
        # Create a gRPC method handler for our 'Communicate' method.
        # It specifies:
        #   - The implementation method (simple_service_impl.Communicate)
        #   - How to deserialize requests (TestConnectionCall.FromString)
        #   - How to serialize responses (TestConnectionResponse.SerializeToString)
        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            simple_service_impl.Communicate,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )
        # Create a generic handler that maps service and method names to the actual method handler.
        generic_handler = grpc.method_handlers_generic_handler(
            _SERVICE_NAME,  # The name of our service
            {
                _METHOD_NAME: rpc_method_handler
            },  # Maps method name to its handler
        )
        # Add this generic handler to the server.
        server.add_generic_rpc_handlers((generic_handler,))
        logger.info(
            f"Server: RPC handlers added for service '{_SERVICE_NAME}'."
        )

    try:
        # Start the server asynchronously.
        # The connect_call will be invoked to set up the service.
        await service_publisher.start_async(connect_call)
        logger.info(
            f"Server: Started and listening on http://{_SERVER_ADDRESS}:{_SERVER_PORT}"
        )
        # Keep the server running until the stop_event is set.
        await stop_event.wait()
    except Exception as e:
        logger.error(f"Server: Failed to start or run: {e!r}")
    finally:
        logger.info("Server: Shutting down...")
        # Stop the gRPC server.
        # Note: GrpcServicePublisher's stop() is synchronous.
        # For an async server, ensure proper async shutdown if extending this.
        # The current stop() might block if the server is grpc.aio.server.
        # For this example, we'll call it, but a more robust async shutdown might be needed.
        if (
            hasattr(service_publisher, "_GrpcServicePublisher__server")
            and service_publisher._GrpcServicePublisher__server is not None
        ):
            actual_server_obj = service_publisher._GrpcServicePublisher__server
            if isinstance(actual_server_obj, grpc.aio.Server):
                logger.info("Server: Attempting graceful async server stop...")
                await actual_server_obj.stop(grace=1)  # 1 second grace period
                logger.info("Server: Async server stop completed.")
            else:
                # Fallback for non-async server or if direct control is not exposed as such
                service_publisher.stop()
                logger.info("Server: Synchronous server stop called.")
        else:
            logger.info(
                "Server: Server object not found on publisher, or already stopped."
            )

        # thread_watcher.stop_all_threads() # ThreadWatcher doesn't manage thread lifecycle directly after creation.
        logger.info("Server: Shutdown complete.")


# --- Client-Side Implementation ---


async def run_client() -> None:
    """
    Runs the gRPC client to connect to the server and make a call.
    """
    logger.info("Client: Initializing...")
    server_target = f"{_SERVER_ADDRESS}:{_SERVER_PORT}"
    # Create an insecure gRPC channel to connect to the server.
    # 'insecure' means no encryption (TLS) is used for this example.
    # For production, secure channels (grpc.aio.secure_channel) are recommended.
    async with grpc.aio.insecure_channel(server_target) as channel:
        logger.info(f"Client: Connected to server at {server_target}")

        # Create an empty request message.
        # TestConnectionCall is defined in common_pb2.proto and is an empty message.
        request = TestConnectionCall()
        logger.info("Client: Sending TestConnectionCall to server...")

        try:
            # Make the RPC call.
            # channel.unary_unary indicates a simple request-response call.
            # It requires:
            #   - The full method path (e.g., "/servicename/methodname")
            #   - How to serialize the request (TestConnectionCall.SerializeToString)
            #   - How to deserialize the response (TestConnectionResponse.FromString)
            response = await channel.unary_unary(
                _FULL_METHOD_PATH,
                request_serializer=TestConnectionCall.SerializeToString,
                response_deserializer=TestConnectionResponse.FromString,
            )(
                request, timeout=10
            )  # 10-second timeout for the call

            # TestConnectionResponse is also an empty message in this example.
            # In a real application, you would inspect the fields of the response.
            logger.info(f"Client: Received response from server: {response}")
            assert isinstance(response, TestConnectionResponse)
            logger.info("Client: Communication successful!")
        except grpc.aio.AioRpcError as e:
            logger.error(
                f"Client: gRPC call failed: {e.code()} - {e.details()}"
            )
        except Exception as e:
            logger.error(f"Client: An unexpected error occurred: {e!r}")


# --- Main Execution ---


async def main() -> None:
    """
    Coordinates the server and client startup and shutdown.
    """
    # Set the tsercom global event loop to the current asyncio event loop
    try:
        current_loop = asyncio.get_running_loop()
        set_tsercom_event_loop(current_loop)
        logger.info("Main: Tsercom global event loop set.")

        # Create an asyncio Event to signal the server to stop.
        stop_server_event = asyncio.Event()

        # Start the server in a background task.
        logger.info("Main: Starting server task...")
        server_task = asyncio.create_task(run_server(stop_server_event))

        # Wait a moment for the server to start up.
        # In a production system, you might use more sophisticated readiness checks.
        await asyncio.sleep(1.0)

        # Run the client.
        logger.info("Main: Starting client task...")
        await run_client()

        # Signal the server to stop.
        logger.info("Main: Signaling server to stop...")
        stop_server_event.set()

        # Wait for the server task to complete.
        await server_task
        logger.info("Main: Example finished.")
    finally:
        clear_tsercom_event_loop()
        logger.info("Main: Tsercom global event loop cleared.")


if __name__ == "__main__":
    # Run the main asynchronous function.
    asyncio.run(main())
