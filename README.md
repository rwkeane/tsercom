# Tsercom: A Toolkit for Robust Time-Series Communication

[![CI Tests](https://github.com/rwkeane/tsercom/actions/workflows/python-tests.yml/badge.svg)](https://github.com/rwkeane/tsercom/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/rwkeane/tsercom/branch/main/graph/badge.svg)](https://codecov.io/gh/rwkeane/tsercom)

**Tsercom** (Time SERies COMmunication) is a high-performance Python toolkit for building robust, distributed applications that stream and synchronize time-series data. It provides composable modules and a powerful, integrated runtime system to handle complex networking challenges like service discovery, out-of-order data, and efficient serialization, allowing you to focus on your core application logic.

While its most advanced features are purpose-built for `torch.Tensor` data, the library is designed to be modular and is fully functional for any data type in environments without PyTorch.

## Core Philosophy

  * **Composable Toolkit:** Use what you need. Each major capability is a self-contained module. Need dynamic service discovery? Use `tsercom.discovery`. Need to handle out-of-order tensor streams? Use `tsercom.tensor`. The modules can be used independently.
  * **Powerful Integration:** While modules are independent, the optional `RuntimeManager` provides a powerful, out-of-process execution environment. When used, it **automatically handles difficult distributed systems problems** like inter-process communication, multi-node time synchronization, and resilient connections "for free".
  * **Empower with Utilities:** Tsercom provides **gRPC utilities**, not a restrictive framework. You define your own `.proto` services and implement your own logic; Tsercom provides the tools to host, connect, and manage those services robustly.

## Key Features

  * **Optimized Tensor Transport:** When `torch` is installed, Tsercom uses `torch.multiprocessing` for near-zero-copy tensor transfer between local processes and an efficient "chunk-based" serialization to minimize network traffic.
  * **Causal Consistency Engine:** The `TensorDemuxer` acts as a state reconstruction engine, correctly rebuilding a historical timeline from sporadic or out-of-order data streams via a cascading update mechanism.
  * **Online Data Augmentation:** Provides utilities for creating smoothed, high-frequency data streams from sparse updates. Includes options from simple linear interpolation (`RuntimeDataAggregator`) to the fully causally-consistent `SmoothedTensorDemuxer`.
  * **Dynamic Service Discovery:** Built-in mDNS (`zeroconf`) support for automatically discovering and connecting to services on a local network.
  * **Stateful Bidirectional Streaming:** Built on gRPC, Tsercom establishes resilient connections with a formal handshake to exchange identity (`CallerId`). This allows for persistent, stateful relationships between components, even across network failures.

## Installation

```bash
# For general-purpose use
pip install tsercom

# To enable all tensor-specific features, install PyTorch
pip install tsercom torch
```

For development, clone the repository and run the setup script, which will install all dependencies and `pre-commit` hooks:

```bash
git clone https://github.com/rwkeane/tsercom.git
cd tsercom
./setup_dev.sh
```

## Anatomy of a Tsercom Connection
When using the discovery and connection modules, the components interact in a clear sequence to establish a robust data stream:
1.  **Publish:** A data source `Runtime` uses an `InstancePublisher` to advertise its service (e.g., `_my-service._tcp.local`) on the network via mDNS.
2.  **Discover:** A data aggregator `Runtime` uses a `DiscoveryHost` to listen for services of that type.
3.  **Connect:** When a service is discovered, the `ServiceConnector` automatically initiates a gRPC connection, creating a communication channel.
4.  **Handshake:** The `_on_channel_connected` callback is triggered. In this step, the components perform a handshake, typically sending a `CallerId` to establish a persistent, stateful session.
5.  **Stream:** With the connection established and identified, both sides can now use their gRPC stubs and servicers to engage in bidirectional streaming of time-series data.

For more details, see the [quick start guide](https://github.com/rwkeane/tsercom/blob/main/quick_start.md).

## Advanced Usage & Architectures

#### Suggested Architecture: Client-Advertises, Server-Discovers

While Tsercom supports traditional client-server models, its design truly shines in more dynamic, distributed environments. A powerful and recommended architecture involves a "client-advertises, server-discovers" model. This approach flips the traditional roles, offering significant flexibility and resilience, which is ideal for environments with ephemeral or mobile nodes like sensor networks or robotics.

This architecture offers several advantages:

  * **Dynamic Discovery:** Data sources can join or leave the network, and aggregators will automatically discover and connect to them.
  * **Resilience to Network Changes:** Data sources can change IP addresses without manual reconfiguration.
  * **Decoupling:** Data producers and consumers are highly decoupled.

For more details about this approach, see the full [Suggested Architecture documentation](https://github.com/rwkeane/tsercom/blob/main/suggested_architecture.md).

## Dependencies

Tsercom relies on several key libraries:

  * `grpcio`, `grpcio-status`, `grpcio-tools`, `protobuf`
  * `sortedcontainers`
  * `zeroconf`
  * `ntplib`
  * `psutil`
  * `typing-extensions`

**Optional Dependencies:**

  * `pytorch`: Required to unlock all tensor-specific features, including optimized inter-process communication and advanced serialization/reconstruction (`TensorMultiplexer`/`TensorDemuxer`).

## Contributing

Contributions are welcome\! When contributing code, please ensure your changes pass our quality gates. We use `black` for formatting and `ruff` for linting. Setting up the local `pre-commit` hooks (`./setup_dev.sh`) is the easiest way to ensure compliance.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](https://github.com/rwkeane/tsercom/blob/main/LICENSE) file for details.