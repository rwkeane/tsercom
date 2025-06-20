import dataclasses

import grpc


@dataclasses.dataclass
class ChannelInfo:
    """Encapsulates information about a gRPC channel and its connection endpoint."""

    channel: grpc.Channel
    address: str
    port: int
