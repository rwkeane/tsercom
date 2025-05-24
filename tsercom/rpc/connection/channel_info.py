import grpc


class ChannelInfo:
    """Encapsulates information about a gRPC channel and its connection endpoint.

    This class serves as a data container for a `grpc.Channel` object along with
    the network address and port it is connected to.
    """

    def __init__(self, channel: grpc.Channel, address: str, port: int) -> None:
        """Initializes a ChannelInfo instance.

        Args:
            channel: The active gRPC channel.
            address: The IP address or hostname of the connected endpoint.
            port: The network port of the connected endpoint.
        """
        # The gRPC channel object.
        self.__channel: grpc.Channel = channel
        # The network address (IP or hostname) of the endpoint.
        self.__address: str = address
        # The network port number of the endpoint.
        self.__port: int = port

    @property
    def channel(self) -> grpc.Channel:
        """Gets the gRPC channel.

        Returns:
            The `grpc.Channel` instance.
        """
        return self.__channel

    @property
    def address(self) -> str:
        """Gets the network address of the endpoint.

        Returns:
            The address string (IP or hostname).
        """
        return self.__address

    @property
    def port(self) -> int:
        """Gets the network port of the endpoint.

        Returns:
            The port number as an integer.
        """
        return self.__port
