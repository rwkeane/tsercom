from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory


class ChannelFactorySelector:
    def get_instance(self) -> GrpcChannelFactory:
        raise NotImplementedError()
