import grpc


class ChannelInfo:
    """
    This class acts as a bucket of data, for a gRPC Channel and the address /
    port to which it is connected.
    """

    def __init__(self, channel: grpc.Channel, address: str, port: int):
        self.__channel = channel
        self.__address = address
        self.__port = port

    @property
    def channel(self) -> grpc.Channel:
        return self.__channel

    @property
    def address(self) -> str:
        return self.__address

    @property
    def port(self) -> int:
        return self.__port
