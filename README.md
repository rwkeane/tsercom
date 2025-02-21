# tsercom
This library provides utilities for transmitting time-series data across a 
network, from client to server and back again (if needed) using gRPC.

## WARNING: This library is still in alpha version. It works for my project use case, but has not yet been fully tested with unit tests. Use at your own risk!

This library operates on the principle that many data science libraries use a
"blocking" model, in that calls are made, and then the thread sits and waits for
a response to be available, while networking calls are expected to be 
high-performance and minimal overhead, so these operations should be isolated
from one-another on different threads. At the same time, the user should not
need to worry about the fact that different threads are being used, and can
instead just look for new data as needed, without worrying about the underlying
network connection.

Specifically, this library supports the following:

- Creation of both synchronous and asynchronous gRPC Connections, and utilities
for maintaining these connections.
- Reconnections upon failure.
- Time synchronization between client and server instances using NTP, as well as
utilities to make use of this synchronization.
- Identities for clients, as assigned by servers, and utlities to 
Time Series Communication Utilities for Data Science Applications
- Utilities for serializing and deserializing common types to and from
protobufs (for use with gRPC), as well as the proto files which must be imported
in a gRPC Service definition to use these instances.
- Threading utilities necessary to synchronize between a "main" thread and other
"utility" threads.

## Dependencies
_NOTE_: If the gRPC dependency here gets out-of-date, it is a 2 minute fix to update it! Just check out the repo, run `scripts/generate_protos.py`, and put up a pull request!