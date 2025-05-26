"""Provides ChannelFactorySelector for obtaining gRPC channel factory instances.

This module currently offers a simple selector that returns a default
insecure gRPC channel factory. Future extensions might allow for selection
based on configuration or environment.
"""

from tsercom.rpc.grpc_generated.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_generated.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)


class ChannelFactorySelector:
    """Selects and provides instances of gRPC channel factories.

    Currently, it defaults to providing an `InsecureGrpcChannelFactory`.
    The selection logic may be expanded in the future.
    """

    # TODO: Implement switching.
    # This will require a more significant refactor to allow different channel factory implementations.
    def get_instance(self) -> GrpcChannelFactory:
        """Gets an instance of a GrpcChannelFactory.

        Currently, this method returns an `InsecureGrpcChannelFactory` by default.
        The `# TODO` comment in the original code indicates plans for more
        sophisticated selection logic.

        Returns:
            An instance of `InsecureGrpcChannelFactory`.
        """
        return InsecureGrpcChannelFactory()
