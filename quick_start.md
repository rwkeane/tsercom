# Quick Start for Tsercom

The best way to understand `Tsercom` is to see how its modules work together. This example demonstrates the powerful "client-advertises, server-discovers" pattern. We will create:

1.  A "Data Source" runtime that hosts a gRPC service and advertises it on the network.
2.  A "Data Aggregator" runtime that discovers the Data Source, connects to it as a gRPC client, and calls an RPC.
3.  A `RuntimeManager` to run both in separate processes.

**1. Define the gRPC Service (`echo.proto`)**

```protobuf
syntax = "proto3";
package tsercom_example;

service Echoer {
  rpc Echo (EchoRequest) returns (EchoReply);
}
message EchoRequest { string message = 1; }
message EchoReply { string response = 1; }
```

**2. Implement the Runtimes**

```python
# quick_start.py
import asyncio
import grpc
import time
from typing import Optional

# Assume generated files exist
import echo_service_pb2
import echo_service_pb2_grpc

# --- Tsercom Imports ---
from tsercom.api import RuntimeManager
from tsercom.discovery import DiscoveryHost, ServiceConnector, ServiceInfo
from tsercom.discovery.mdns import InstancePublisher
from tsercom.rpc.connection import ChannelInfo
from tsercom.rpc.grpc_util import GrpcChannelFactoryImpl, GrpcServicePublisher
from tsercom.runtime import Runtime, RuntimeInitializer
from tsercom.threading import NullThreadWatcher


# --- Data Source (Acts as gRPC Server) ---
class EchoServicer(echo_service_pb2_grpc.EchoerServicer):
    async def Echo(self, request, context):
        return echo_service_pb2.EchoReply(response=f"Echoing: {request.message}")

class DataSourceRuntime(Runtime, echo_service_pb2_grpc.EchoerServicer):
    def __init__(self, thread_watcher: ThreadWatcher):
        super().__init__()
        self._port = 50051
        self._publisher = GrpcServicePublisher(thread_watcher, self._port)
        self._mdns_advertiser = InstancePublisher(self._port, "_echo._tcp.local.", "MyEchoService")

    async def start_async(self):
        print("Data Source: Starting gRPC server and advertising via mDNS...")
        await self._publisher.start_async(
            lambda server: echo_service_pb2_grpc.add_EchoerServicer_to_server(self, server)
        )
        await self._mdns_advertiser.publish()
        print("Data Source: Service is live.")

    async def stop(self, exception: Optional[Exception] = None):
        await self._mdns_advertiser.close()
        await self._publisher.stop_async()
        print("Data Source: Stopped.")

    async def Echo(self, request, context):
        # This is the gRPC service logic
        return echo_service_pb2.EchoReply(response=f"Echoing: {request.message}")

class DataSourceInitializer(RuntimeInitializer):
    def create(self,
               thread_watcher: ThreadWatcher,
               data_handler: RuntimeDataHandler,
               grpc_channel_factory: GrpcChannelFactory,) -> Runtime:
        # data_handler is for sending data to / from the Runtime instance.
        return DataSourceRuntime(thread_watcher)


# --- Data Aggregator (Acts as gRPC Client) ---
class DataAggregatorRuntime(Runtime, ServiceConnector.Client):
    def __init__(self, grpc_channel_factory: GrpcChannelFactory):
        super().__init__()
        self._discoverer = DiscoveryHost(service_type="_echo._tcp.local.")
        self._connector = ServiceConnector(self, grpc_channel_factory, self._discoverer)

    async def start_async(self):
        print("Aggregator: Discovering services...")
        await self._connector.start()
        await self._discoverer.start_discovery()

    async def stop(self, exception: Optional[Exception] = None):
        await self._discoverer.stop_discovery()
        await self._connector.stop()
        print("Aggregator: Stopped.")

    async def _on_channel_connected(self, conn_info: ServiceInfo, _, channel_info: ChannelInfo):
        print(f"Aggregator: Connected to {conn_info.name}. Sending RPC...")
        try:
            stub = echo_service_pb2_grpc.EchoerStub(channel_info.channel)
            response = await stub.Echo(echo_service_pb2.EchoRequest(message="Hello from Aggregator"))
            print(f"Aggregator: Got response: '{response.response}'")
            self.response_future.set_result(response.response)
        except grpc.aio.AioRpcError as e:
            print(f"Aggregator: RPC failed: {e}")
            self.response_future.set_exception(e)

class DataAggregatorInitializer(RuntimeInitializer):
    def create(self,
               thread_watcher: ThreadWatcher,
               data_handler: RuntimeDataHandler,
               grpc_channel_factory: GrpcChannelFactory) -> Runtime:
        # data_handler is for sending data to / from the Runtime instance.
        return DataAggregatorRuntime(grpc_channel_factory)


# --- Orchestrate with RuntimeManager --- 
async def main():
    manager = RuntimeManager()
    aggregator_init = DataAggregatorInitializer()
    aggregator_handle_f = manager.register_runtime_initializer(aggregator_init)
    manager.register_runtime_initializer(DataSourceInitializer())

    # Also supports manager.start_in_process_async()
    manager.start_out_of_process_async()
    
    # Used to send data to / from the aggregator. But that's not needed in this example.
    aggregator_handle = aggregator_handle_f.result()

    # Wait for gRPC to do its thing off-process / off-thread
    time.sleep(5)
    
    manager.shutdown()

if __name__ == "__main__":
    main()

```

For a more comprehensive example that demonstrates a full client-server interaction with gRPC and a fake discovery mechanism, see the end-to-end test located at [`tsercom/full_app_e2etest.py`](https://www.google.com/search?q=%5Bhttps://github.com/rwkeane/tsercom/blob/main/tsercom/full_app_e2etest.py%5D\(https://github.com/rwkeane/tsercom/blob/main/tsercom/full_app_e2etest.py\)).