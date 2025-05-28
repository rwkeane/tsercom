# TSERCOM
## Time SERies COMmunication

[![CI Tests](https://github.com/rwkeane/tsercom/actions/workflows/python-tests.yml/badge.svg)](https://github.com/rwkeane/tsercom/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/rwkeane/tsercom/branch/main/graph/badge.svg)](https://codecov.io/gh/rwkeane/tsercom)

Tsercom is a Python library designed to simplify the transmission and management of time-series data across networks using gRPC. It provides tools for establishing communication between clients and servers, handling data serialization, managing persistent client identities, and synchronizing timestamps, making it suitable for distributed data science and machine learning applications.

## Key Features

*   **ZeroConf Client/Server Discovery:** Automatically discover and connect clients and servers on the network using mDNS.
*   **Simplified gRPC Management:** Abstracts away much of the boilerplate for setting up and managing gRPC connections.
*   **Automatic Reconnection:** Handles network disruptions by attempting to reconnect clients and servers automatically.
*   **Persistent Client Identity:** Maintains a consistent `CallerId` for clients, even across disconnections and restarts.
*   **Timestamp Synchronization:** Provides mechanisms for synchronizing timestamps between server and client instances.
*   **Serialization Utilities:** Includes helpers for serializing common data types to and from protobufs for gRPC transmission.
*   **Process/Thread Isolation:** Supports running communication logic in separate processes or threads, isolating it from the main application.

## Installation

```bash
pip install tsercom
```

## Quick Start / Basic Usage

This example demonstrates setting up a simple custom runtime and managing it with `RuntimeManager`.

```python
import asyncio
import time
from datetime import datetime # For timestamp
from typing import Any, Optional, List, Dict # For type hinting
from concurrent.futures import Future

from tsercom.api.runtime_manager import RuntimeManager
# RuntimeHandle is generic, so we'll use its generic form here.
# The type variables will be specialized based on MyRuntimeInitializer.
from tsercom.api.runtime_handle import RuntimeHandle
# RuntimeInitializer is generic.
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.runtime.runtime_config import ServiceType # For RuntimeConfig
# RuntimeDataHandler is generic.
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime import Runtime # Base class for custom runtime (non-generic)
from tsercom.threading.thread_watcher import ThreadWatcher # For exception handling
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory # For gRPC channel
from tsercom.data.annotated_instance import AnnotatedInstance # To wrap data
from tsercom.caller_id.caller_identifier import CallerIdentifier # For CallerId
# ExposedData is the base for data that can be handled by the data_aggregator.
from tsercom.data.exposed_data import ExposedData


# Define a concrete payload type for this example.
ExamplePayloadType = str

class MyCustomRuntime(Runtime): # MyCustomRuntime is not generic
    """
    A custom Runtime implementation.
    This class would typically contain application-specific logic
    for handling data and events.
    """
    def __init__(self,
                 thread_watcher: ThreadWatcher,
                 # This data_handler is expected to be configured for:
                 # TDataType = AnnotatedInstance[ExamplePayloadType]
                 # TEventType = Any (or a specific event type)
                 data_handler: RuntimeDataHandler[AnnotatedInstance[ExamplePayloadType], Any],
                 grpc_channel_factory: GrpcChannelFactory):
        self._thread_watcher = thread_watcher
        self._data_handler = data_handler 
        self._grpc_channel_factory = grpc_channel_factory
        print("MyCustomRuntime initialized.")

    async def start_async(self) -> None:
        """Starts the custom runtime's operations."""
        print("MyCustomRuntime started asynchronously.")
        await asyncio.sleep(0.1) # Simulate some async work
        
        timestamp_val = time.time()
        _dummy_caller_id = CallerIdentifier(id="my_custom_runtime_caller")
        
        data_instance = AnnotatedInstance[ExamplePayloadType](
            data="Hello Tsercom from MyCustomRuntime!",
            caller_id=_dummy_caller_id,
            timestamp=datetime.fromtimestamp(timestamp_val)
        )
        
        # To make this data available to the RuntimeHandle's data_aggregator,
        # MyCustomRuntime would use its self._data_handler. The exact method
        # depends on the concrete type of RuntimeDataHandler provided by the
        # Tsercom framework (e.g., ClientRuntimeDataHandler/ServerRuntimeDataHandler)
        # and how LocalRuntimeFactory configures it.
        # For this conceptual example, we'll note that data is created.
        # A real implementation would involve calling a method on self._data_handler
        # to push `data_instance` into the Tsercom data flow.
        print(f"MyCustomRuntime: Data instance created ('{data_instance.data}'). In a full setup, this would be sent via self._data_handler.")

    async def stop(self) -> None:
        """Stops the custom runtime's operations."""
        print(f"MyCustomRuntime stopping.")
        # Custom cleanup for MyCustomRuntime would go here.

# MyRuntimeInitializer itself is generic, as defined by the library.
# For this example, TDataType is AnnotatedInstance[ExamplePayloadType]
# and TEventType is Any.
class MyRuntimeInitializer(RuntimeInitializer[AnnotatedInstance[ExamplePayloadType], Any]):
    """
    Initializes MyCustomRuntime.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize RuntimeConfig, e.g., as a client.
        super().__init__(service_type=ServiceType.kClient, *args, **kwargs)
        print("MyRuntimeInitializer initialized.")

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[AnnotatedInstance[ExamplePayloadType], Any],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime: # Returns the base, non-generic Runtime
        """Creates and returns an instance of MyCustomRuntime."""
        print("MyRuntimeInitializer: Creating MyCustomRuntime.")
        return MyCustomRuntime(thread_watcher, data_handler, grpc_channel_factory)

async def main():
    print("Starting Tsercom RuntimeManager example...")
    runtime_manager = RuntimeManager()
    my_initializer = MyRuntimeInitializer()
    
    # The RuntimeHandle will be specialized with the types from MyRuntimeInitializer
    handle_future: Future[RuntimeHandle[AnnotatedInstance[ExamplePayloadType], Any]] = \
        runtime_manager.register_runtime_initializer(my_initializer)
    print("RuntimeInitializer registered.")

    print("Starting runtime in-process...")
    # start_in_process_async will use the types from MyRuntimeInitializer
    # to correctly type the RuntimeDataHandler passed to MyCustomRuntime.
    await runtime_manager.start_in_process_async()
    print("RuntimeManager started in-process.")

    try:
        runtime_handle: RuntimeHandle[AnnotatedInstance[ExamplePayloadType], Any] = handle_future.result(timeout=5)
        print(f"Obtained RuntimeHandle: {runtime_handle}")

        # The data_aggregator is part of the RuntimeHandle. Its TDataType is AnnotatedInstance[ExamplePayloadType].
        data_aggregator = runtime_handle.data_aggregator
        
        # Give MyCustomRuntime a moment to potentially send data
        await asyncio.sleep(0.5) 

        print("Attempting to receive data from runtime...")
        try:
            # get_new_data() on this aggregator returns Dict[CallerIdentifier, List[AnnotatedInstance[ExamplePayloadType]]]
            all_new_data: Dict[CallerIdentifier, List[AnnotatedInstance[ExamplePayloadType]]] = data_aggregator.get_new_data()
            data_found = False
            if all_new_data:
                for caller_id, data_list in all_new_data.items():
                    if data_list:
                        for received_data_item in data_list:
                            # received_data_item is an AnnotatedInstance[ExamplePayloadType]
                            print(f"QuickStart: Received data - '{received_data_item.data}' at {received_data_item.timestamp} from {caller_id.id}")
                            data_found = True
                            break 
                    if data_found:
                        break
            if not data_found:
                print("QuickStart: No new data received. (This is expected if MyCustomRuntime's data sending mechanism isn't fully implemented for this example context).")
        except Exception as e:
            print(f"QuickStart: Error receiving data: {e!r}")

    except TimeoutError:
        print("Error: Timed out waiting for RuntimeHandle.")
    except Exception as e:
        print(f"An error occurred: {e!r}")
    finally:
        if 'runtime_handle' in locals() and runtime_handle:
             print("Stopping runtime...")
             runtime_handle.stop()
             print("Runtime stopped.")
        print("Tsercom example finished.")

if __name__ == "__main__":
    asyncio.run(main())
```

## The Idea

Tsercom aims to simplify the complex task of building systems that require real-time or near real-time exchange of time-series data. Such systems are common in distributed machine learning, sensor data aggregation, and financial applications.

The core philosophy is to:
-   **Abstract Complexity:** Hide the intricacies of network programming (gRPC, discovery, reconnection) behind a more straightforward API.
-   **Promote Modularity:** Allow developers to focus on their application-specific data handling logic by providing a framework for communication.
-   **Ensure Robustness:** Incorporate features like persistent client IDs and automatic reconnections to build more resilient systems.
-   **Facilitate Integration:** Offer utilities for data serialization and timestamp management, common needs in time-series applications.

By providing these building blocks, Tsercom allows developers to set up multi-client/multi-server architectures where time-series data can be efficiently disseminated and aggregated. The library encourages isolating network communication (which can be I/O bound and require high performance) from the main application logic (e.g., a machine learning model's computation), often by running Tsercom components in separate threads or processes.

## Suggested Architecture

_This is a conceptual suggestion. Tsercom is flexible and can support various client-server models._

A common, albeit somewhat counter-intuitive, architecture with Tsercom involves **CLIENT instances advertising their services**, and **SERVER instances discovering and connecting to these CLIENTs**.

**Tsercom CLIENT (e.g., a data source like a sensor or an ML model generating predictions):**
1.  **Implements a gRPC Server:** This server is an extension of Tsercom's `Runtime`. It defines how the client responds to requests from a Tsercom SERVER.
2.  **Advertises Itself:** Uses Tsercom's mDNS API (e.g., `InstancePublisher`) to make its gRPC server discoverable on the network.
3.  **Uses `RuntimeInitializer`:** Implements a `RuntimeInitializer` to define how its custom `Runtime` (the gRPC server) is created. This initializer wires up necessary components like `ThreadWatcher` (for error handling back to the main thread/process) and `RuntimeDataHandler` (for passing data between the Tsercom runtime and the application's main logic).
4.  **Managed by `RuntimeManager`:** The `RuntimeManager` takes the `RuntimeInitializer` and starts the client's `Runtime`, potentially in a separate thread or process.

**Tsercom SERVER (e.g., a data aggregator or a central logging service):**
1.  **Implements a gRPC Client:** This client logic, also extending `Runtime`, is responsible for connecting to Tsercom CLIENTs.
2.  **Discovers Clients:** Uses Tsercom's mDNS API (e.g., `InstanceListener`) to find available Tsercom CLIENT services on the network. Upon discovery, it initiates gRPC connections to them.
3.  **Uses `RuntimeInitializer` and `RuntimeManager`:** Similar to the Tsercom CLIENT, these are used to set up and manage the server's own `Runtime` logic.

**Tsercom's Internal Handling:**
-   **Persistent CallerID:** Ensures that if a Tsercom CLIENT disconnects and reconnects, the Tsercom SERVER can recognize it as the same entity.
-   **Timestamp Synchronization:** Helps in ordering and correlating data from various sources by aligning timestamps.
-   **Data Marshalling:** Facilitates data exchange between the main application process and the Tsercom communication process/thread.
-   **Resilience:** Manages reconnections if network issues occur.

**Alternative:** It's also possible for the Tsercom SERVER to host the gRPC server and Tsercom CLIENTs to connect to it. However, the suggested architecture above is often preferred as it simplifies the management of dynamic client appearances and persistent identities.

## Dependencies
Tsercom relies on several key libraries:
- `grpcio`*, `grpcio-status`*, `grpcio-tools`*: For the core gRPC communication.
- `ntplib`: For network time synchronization.
- `zeroconf`: For mDNS-based service discovery.
- `psutil`: For system utilities, potentially used in process management or monitoring.

Optional dependencies:
- `pytorch`: To support serializing and deserializing `torch.Tensor` instances. Only enabled if `pytorch` is installed 

*: If the gRPC dependency version causes issues, updating it usually involves regenerating protobuf files using `scripts/generate_protos.py` (if you have the repository checked out).

## Contributing
Contributions are welcome! If you find a bug, have a feature request, or want to improve the documentation, please open an issue or submit a pull request on the GitHub repository.
