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

## Basic Usage

The basic steps for using Tsercom for a gRPC backed client-server architecture are as follows:
1. Define a simple gRPC service.
2. Host this service using `GrpcServicePublisher`.
3. Create a client that connects to the service.
4. Send a request and receive a response.
5. Manage Tsercom's global event loop.

For example useage, see the [Quick Start Script](https://github.com/rwkeane/tsercom/blob/main/quick_start_test.py) in this repo.

To run this example, save it as `quick_start_test.py` and execute `python quick_start_test.py`.

**Architectural Flexibility:**
While Tsercom provides components like `GrpcServicePublisher` for straightforward client-server setups (as shown in the Quick Start), it also supports more advanced architectures. For instance, the `discovery` module (using mDNS via `zeroconf`) allows for dynamic discovery of services. A common pattern in some Tsercom applications involves "client" processes (data sources) advertising themselves, and "server" processes (data aggregators) discovering and connecting to them. This can be useful for systems where data sources may join or leave the network dynamically. The library provides building blocks that can be composed to fit various distributed system designs.

That being said, there is a suggested architedture 

## Suggested Architecture: Maximizing Tsercom's Potential

While Tsercom supports straightforward client-server setups (as demonstrated in the Quick Start guide), its design truly shines in more dynamic, distributed environments. A powerful and recommended architecture involves a "client-advertises, server-discovers" model. This approach flips the traditional roles, offering significant flexibility and resilience. This "client-advertises, server-discovers" approach offers several advantages:

*   **Dynamic Discovery:** Data sources can join (or leave) the network, and the aggregator will automatically discover and connect to them (or handle their disappearance) without manual reconfiguration. This is ideal for environments with ephemeral or mobile nodes.
*   **Resilience to Network Changes:** Data sources can change IP addresses or ports (e.g., due to DHCP or dynamic port assignment). As long as they can re-advertise via mDNS, the aggregator can re-discover and reconnect to them.
*   **Decoupling:** Data producers (Tsercom "Clients") and consumers (Tsercom "Servers") are highly decoupled. They only need to agree on the service definition and the discovery mechanism, not on static network locations.
*   **Scalability:** New data sources can be easily added to the system. They simply start advertising themselves, and the aggregator(s) can discover and integrate them. Similarly, multiple aggregators can discover the same set of data sources.

For more details about this approach, see [Suggested Architecture](https://github.com/rwkeane/tsercom/blob/main/suggested_architecture.md) in this repo.

### Simpler Models Still Viable:

It's important to note that Tsercom still fully supports traditional client-server models where the client initiates a connection to a well-known server address, as shown in the Quick Start. This is perfectly suitable for simpler applications or when dynamic discovery is not a requirement.

However, adopting the "client-advertises, server-discovers" architecture with `RuntimeManager`, `InstancePublisher`, and `InstanceListener` unlocks Tsercom's more advanced capabilities for building robust, scalable, and adaptive distributed systems for time-series data communication.

### Real-World Examples:

Coming soon! These repos have not yet been made public!

But if you use this library, pleae submit a PR to add a link to your library here!

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

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](https://github.com/rwkeane/tsercom/blob/main/LICENSE) file for details.
