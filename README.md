# TSERCOM
## Time SERies COMmunication
This library provides utilities for transmitting time-series data across a 
network, from client to server and back again (if needed) using gRPC.

## The Idea

Tsercom provides a simplified API for implementing a high-performance
time-series aggregation server, as well as clients for sending this data from
clients. More formally, Tsercom provides a simplified API for hosting a
multi-client, multi-server architecture for the dissemination and aggregation of
time-series data. Specifically, it provides the following:

- ZeroConf client / server [discovery](https://github.com/rwkeane/tsercom/tree/ce5877ee6720c72773d729d0d43d2dd99518b438/tsercom/discovery).
- A simplified API for [gRPC Connection](https://github.com/rwkeane/tsercom/tree/ce5877ee6720c72773d729d0d43d2dd99518b438/tsercom/rpc).
- Simplified reconnection of clients / servers upon network connection issues.
- A `CallerId` associated with each client which persists across client node
disconnections and restarts.
- `TimeStamp` [synchronization](https://github.com/rwkeane/tsercom/tree/ce5877ee6720c72773d729d0d43d2dd99518b438/tsercom/timesync) between each SERVER and CLIENT instance available on
the network.
- Utilities for serializing and deserializing common types to and from
protobufs (for use with gRPC), as well as the proto files which must be imported
in a gRPC Service definition to use these instances.
- [Simple] tooling and APIs to support all of the above in a separate process
(or instead thread, if desired) from the host application!

This library operates on the principle that many machine learning libraries use
a "blocking" model, in that calls are made, and then the thread sits and waits
for a response to be available, while networking calls are expected to be 
high-performance and minimal overhead, so these operations should be isolated
from one-another on separate threads or processes. At the same time, the user
should not need to worry about the complexity of maintaining this parallelism.
The solution is an API where the user may poll for data as needed, without
worrying about the underlying network connection, threading concerns, or other
issues outside of their machine learning model code.

## Suggested Architecture

NOTE: This is a _suggestion_. Tsercom supports a more diverse set of client-
server models. This design just accounts for the most edge cases in the most
common scenarios.

The suggested architecture, admittedly, sounds a bit "backwards". Each Tsercom
CLIENT instance will advertise its presence over gRPC, so that each Tsercom
SERVER can discover and connect to it. To do so, the Tsercom CLIENT will:

1. Implement a gRPC Server that extends [`Runtime`](https://github.com/rwkeane/tsercom/blob/main/tsercom/runtime/runtime.py)
(i.e. it supports `start_async()` to call the actual gRPC Code to host a server,
and `stop()` to stop its execution.)
2. Call into provided [`mDNS` API](https://github.com/rwkeane/tsercom/blob/main/tsercom/discovery/mdns/instance_publisher.py)
to advertise this device's presence to all availble servers (from within
this `Runtime`).
3. Implement a [`RuntimeInitializer`](https://github.com/rwkeane/tsercom/blob/main/tsercom/runtime/runtime_initializer.py)
to create your Runtime instance. This will provide a [`ThreadWatcher`](https://github.com/rwkeane/tsercom/blob/main/tsercom/threading/thread_watcher.py)
(for `Exception` Handling, to ensure exceptions get back to the caller), a 
[`RuntimeDataHandler`](https://github.com/rwkeane/tsercom/blob/main/tsercom/runtime/runtime_data_handler.py)
(to pass data between the main process and the Tsercom process), a [`GrpcChannelFactory`](https://github.com/rwkeane/tsercom/blob/main/tsercom/runtime/runtime_initializer.py), and a [`GrpcChannelFactory`](https://github.com/rwkeane/tsercom/blob/main/tsercom/runtime/runtime_initializer.py)
(to simplify forming `gRPC` connection, if required - not used by the CLIENT in
the suggested architecture) and create your `Runtime` as implemented in 1.
4. Call on [`RuntimeManager`](https://github.com/rwkeane/tsercom/blob/main/tsercom/api/runtime_manager.py)
first `register_runtime_initializer()` to get a [handle](https://github.com/rwkeane/tsercom/blob/main/tsercom/api/runtime_handle.py)
for the Runtime that will be created, and then either `start_in_process()` or
`start_out_of_process()` to use all `RuntimeInitializer` instances that have
been registered to create their corresponding `Runtime` instances in the
appropriate process.

Each Tsercom SERVER will do largely the same thing:

1. Implement a gRPC Client that extends [`Runtime`](https://github.com/rwkeane/tsercom/blob/main/tsercom/runtime/runtime.py)
(i.e. it supports `start_async()` to call the actual gRPC Code to host a server,
and `stop()` to stop its execution.)
2. From within the client, use the provided `mDNS` API by implementing 
[`InstanceListener.Client`](https://github.com/rwkeane/tsercom/blob/main/tsercom/discovery/mdns/instance_listener.py)
to respond to instance discoveries to initialize gRPC Connections.
3. and .4: Exactly as above.

Internally, the library will handle:

- Maintaining a consistent CallerID associated with each Tsercom CLIENT which
persists across CLIENT restarts. 
- Synchronizing TimeStamps between each Tsercom CLIENT and SERVER instance.
- Transmitting data between the differing process (or differing threads) on
which Tsercom and the local client are running.
- Re-connection upon disconnection due to network conditions. 

NOTE: It is perfectly find to reverse this and have the Tsercom SERVER host the
gRPC Server. Doing so will just lose the consistency of CallerID across client
restarts.


## WARNING: This library is still in alpha version. It works for my project use case, but has not yet been fully tested with unit tests. Use at your own risk!


## Dependencies
_NOTE_: If the gRPC dependency here gets out-of-date, it is a 2 minute fix to update it! Just check out the repo, run `scripts/generate_protos.py`, and put up a pull request!