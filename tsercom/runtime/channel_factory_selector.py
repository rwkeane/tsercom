"""Provides ChannelFactorySelector for obtaining gRPC channel factory instances.

This module currently offers a simple selector that returns a default
insecure gRPC channel factory. Future extensions might allow for selection
based on configuration or environment.
"""

from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)


class ChannelFactorySelector:
    """Selects and provides instances of gRPC channel factories.

    Currently, it defaults to providing an `InsecureGrpcChannelFactory`.
    The selection logic may be expanded in the future.
    """

    # TODO(developer): Implement switching logic for channel factories. This will require a more significant refactor to allow selection of different channel factory implementations (e.g., secure, insecure) based on configuration or other criteria.
    def get_instance(self) -> GrpcChannelFactory:
        """Gets an instance of a GrpcChannelFactory.

        Currently, this method returns an `InsecureGrpcChannelFactory` by default.

        Returns:
            An instance of `InsecureGrpcChannelFactory`.
        """
        return InsecureGrpcChannelFactory()
