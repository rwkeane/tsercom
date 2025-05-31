import dataclasses
import grpc


@dataclasses.dataclass
class ChannelInfo:
    """Encapsulates information about a gRPC channel and its connection endpoint.

    This class serves as a data container for a `grpc.Channel` object along with
    the network address and port it is connected to.
    """

    channel: grpc.Channel
    address: str
    port: int
