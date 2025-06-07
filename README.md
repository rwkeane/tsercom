# TSERCOM
## Time SERies COMmunication

[![CI Tests](https://github.com/rwkeane/tsercom/actions/workflows/python-tests.yml/badge.svg)](https://github.com/rwkeane/tsercom/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/rwkeane/tsercom/branch/main/graph/badge.svg)](https://codecov.io/gh/rwkeane/tsercom)

Tsercom is a Python library designed to simplify the transmission and management of time-series data across networks using gRPC. It provides tools for establishing communication between clients and servers, handling data serialization, managing persistent client identities, and synchronizing timestamps, making it suitable for distributed data science and machine learning applications.

## Key Features

*   **Simplified gRPC Management:** Abstracts away much of the boilerplate for setting up and managing gRPC services and clients.
*   **ZeroConf Client/Server Discovery:** Automatically discover and connect clients and servers on the network using mDNS (optional, via `discovery` module).
*   **Automatic Reconnection:** Includes utilities to help build resilient clients that can handle network disruptions and attempt reconnection (e.g., `ClientDisconnectionRetrier`).
*   **Persistent Client Identity:** Provides mechanisms for managing a consistent `CallerId` for clients.
*   **Timestamp Synchronization:** Offers tools for synchronizing timestamps between server and client instances.
*   **Serialization Utilities:** Includes helpers for serializing common data types to and from protobufs for gRPC transmission.
*   **Process/Thread Isolation:** Supports running communication logic in separate processes or threads, isolating it from the main application, particularly when using the `RuntimeManager` system.

## Installation

You can install Tsercom using pip:
```bash
pip install tsercom
```

For development, clone the repository and install in editable mode with development dependencies:
```bash
git clone https://github.com/rwkeane/tsercom.git
cd tsercom
pip install -e .[dev]
```
This will also install tools like `pytest`, `black`, `ruff`, `mypy`, and `pylint`.

## Quick Start / Basic Usage

This example demonstrates a fundamental client-server interaction using Tsercom's gRPC utilities. It shows how to:
1. Define a simple gRPC service.
2. Host this service using `GrpcServicePublisher`.
3. Create a client that connects to the service.
4. Send a request and receive a response.
5. Manage Tsercom's global event loop.

```python
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
```

To run this example, save it as `quick_start_test.py` and execute `python quick_start_test.py`.

## How It Works / The Idea

Tsercom simplifies building systems that exchange time-series data by providing a framework and tools for common networking tasks. The core philosophy is to:

*   **Abstract Complexity:** Hide the intricacies of network programming (gRPC setup, service discovery, reconnection logic) behind more straightforward APIs. This allows developers to focus on their application-specific data handling and business logic.
*   **Promote Modularity:** Encourage separation of concerns. Communication logic can be developed and managed independently of the core application (e.g., a machine learning model or data processing pipeline). Tsercom's `RuntimeManager` system (shown in older examples, and used internally for more complex scenarios) particularly facilitates running communication components in separate threads or processes, isolating them and improving robustness.
*   **Ensure Robustness:** Incorporate features like persistent client identifiers (`CallerId`) and utilities for automatic reconnections to help build more resilient distributed systems.
*   **Facilitate Integration:** Offer utilities for data serialization (especially for `torch.Tensor` if PyTorch is installed) and timestamp synchronization, which are common needs in time-series applications.

**Typical Use Cases:**
*   **Distributed Machine Learning:** Streaming inference requests to model servers or aggregating training data from multiple sources.
*   **Sensor Networks:** Collecting and processing data from many distributed sensors.
*   **Real-time Data Pipelines:** Building systems where components need to exchange data with low latency.

**Architectural Flexibility:**
While Tsercom provides components like `GrpcServicePublisher` for straightforward client-server setups (as shown in the Quick Start), it also supports more advanced architectures. For instance, the `discovery` module (using mDNS via `zeroconf`) allows for dynamic discovery of services. A common pattern in some Tsercom applications involves "client" processes (data sources) advertising themselves, and "server" processes (data aggregators) discovering and connecting to them. This can be useful for systems where data sources may join or leave the network dynamically. The library provides building blocks that can be composed to fit various distributed system designs.

## Suggested Architecture: Maximizing Tsercom's Potential

While Tsercom supports straightforward client-server setups (as demonstrated in the Quick Start guide), its design truly shines in more dynamic, distributed environments. A powerful and recommended architecture involves a "client-advertises, server-discovers" model. This approach flips the traditional roles, offering significant flexibility and resilience.

In this model:

*   **Tsercom "Client" (Acts as a Data Source/Provider):**
    *   **Runs its own gRPC Server:** Instead of just initiating connections, each data source (e.g., a sensor, a data processing node, a machine learning model server) hosts its own gRPC service. This service is typically managed as a Tsercom `Runtime`.
    *   **Advertises its Service:** The Tsercom "Client" uses an `InstancePublisher` (leveraging mDNS/ZeroConf) to announce its presence and service details (like its IP address, port, and unique `CallerId`) on the local network. This makes it discoverable without needing a central registry or pre-configured addresses.
    *   **Managed by `RuntimeManager`:** The setup, including the gRPC server and the `InstancePublisher`, is often orchestrated using Tsercom's `RuntimeManager` along with a custom `RuntimeInitializer`. This encapsulates the communication logic.
    *   **Persistent Identity:** A `CallerIdentifier` is used to ensure this data source maintains a consistent, recognizable identity across restarts or network changes.

*   **Tsercom "Server" (Acts as a Data Aggregator/Consumer):**
    *   **Discovers Data Sources:** The Tsercom "Server" (e.g., a central data logger, an analytics dashboard, a training job coordinator) uses an `InstanceListener` to dynamically discover available Tsercom "Clients" (data sources) on the network by listening for their mDNS advertisements.
    *   **Connects as a gRPC Client:** Upon discovering a data source, the Tsercom "Server" initiates a gRPC connection *to* that data source.
    *   **Managed by `RuntimeManager`:** This discovery and connection logic is also typically managed within a Tsercom `Runtime` via the `RuntimeManager`.

*   **`RuntimeDataHandler` as the Bridge:**
    *   On both the "Client" (data source) and "Server" (aggregator) sides, a `RuntimeDataHandler` (or a custom implementation of its base class) acts as the crucial bridge. It facilitates the exchange of data and commands between the application's main business logic (e.g., your data generation algorithm or data analysis code) and the isolated Tsercom communication runtime.

### Benefits of this Architecture:

This "client-advertises, server-discovers" approach offers several advantages:

*   **Dynamic Discovery:** Data sources can join (or leave) the network, and the aggregator will automatically discover and connect to them (or handle their disappearance) without manual reconfiguration. This is ideal for environments with ephemeral or mobile nodes.
*   **Resilience to Network Changes:** Data sources can change IP addresses or ports (e.g., due to DHCP or dynamic port assignment). As long as they can re-advertise via mDNS, the aggregator can re-discover and reconnect to them.
*   **Decoupling:** Data producers (Tsercom "Clients") and consumers (Tsercom "Servers") are highly decoupled. They only need to agree on the service definition and the discovery mechanism, not on static network locations.
*   **Scalability:** New data sources can be easily added to the system. They simply start advertising themselves, and the aggregator(s) can discover and integrate them. Similarly, multiple aggregators can discover the same set of data sources.

### Simpler Models Still Viable:

It's important to note that Tsercom still fully supports traditional client-server models where the client initiates a connection to a well-known server address, as shown in the Quick Start. This is perfectly suitable for simpler applications or when dynamic discovery is not a requirement.

However, adopting the "client-advertises, server-discovers" architecture with `RuntimeManager`, `InstancePublisher`, and `InstanceListener` unlocks Tsercom's more advanced capabilities for building robust, scalable, and adaptive distributed systems for time-series data communication.

## Dependencies

Tsercom relies on several key libraries:

*   `grpcio`, `grpcio-status`, `grpcio-tools`: For the core gRPC communication framework.
*   `protobuf`: For working with Protocol Buffers, the data serialization format used by gRPC.
*   `zeroconf`: For mDNS-based service discovery (used by the `tsercom.discovery` module).
*   `ntplib`: Used by the `tsercom.timesync` module for network time synchronization.
*   `psutil`: For system utilities, which can be used internally for process management or monitoring.
*   `typing-extensions`: Provides access to newer typing features for older Python versions.

**Optional Dependencies:**

*   `pytorch`: If PyTorch is installed, Tsercom provides utilities for serializing and deserializing `torch.Tensor` objects.

If you encounter issues with gRPC versions, you might need to regenerate the protobuf-generated Python files. If you have the Tsercom repository cloned, you can do this by running the `scripts/generate_protos.py` script. This may require installing `mypy-protobuf` (`pip install mypy-protobuf`) and ensuring `protoc-gen-mypy` is in your PATH.

## Contributing

Contributions are welcome! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/rwkeane/tsercom).

When contributing code, please ensure that:
*   Your changes pass all existing tests.
*   You add new tests for any new functionality.
*   The code adheres to our style guidelines. We use `black` for formatting, `ruff` for linting, `mypy` for type checking, and `pylint` for further static analysis. Please run these tools locally before submitting your changes.
    *   `black .`
    *   `ruff check . --fix`
    *   `mypy .`
    *   `pylint tsercom quick_start_test.py` (or specify relevant modules/files)

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
