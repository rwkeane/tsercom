from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.grpc.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)


class ChannelFactorySelector:
    # TODO: Implement switching.
    def get_instance(self) -> GrpcChannelFactory:
        return InsecureGrpcChannelFactory()
