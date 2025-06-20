# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from tsercom.test.proto.generated.v1_62 import (
    e2e_test_service_pb2 as e2e__test__service__pb2,
)


class E2ETestServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Echo = channel.unary_unary(
            "/tsercom.E2ETestService/Echo",
            request_serializer=e2e__test__service__pb2.EchoRequest.SerializeToString,
            response_deserializer=e2e__test__service__pb2.EchoResponse.FromString,
        )
        self.ServerStreamData = channel.unary_stream(
            "/tsercom.E2ETestService/ServerStreamData",
            request_serializer=e2e__test__service__pb2.StreamDataRequest.SerializeToString,
            response_deserializer=e2e__test__service__pb2.StreamDataResponse.FromString,
        )
        self.ClientStreamData = channel.stream_unary(
            "/tsercom.E2ETestService/ClientStreamData",
            request_serializer=e2e__test__service__pb2.StreamDataRequest.SerializeToString,
            response_deserializer=e2e__test__service__pb2.EchoResponse.FromString,
        )
        self.BidirectionalStreamData = channel.stream_stream(
            "/tsercom.E2ETestService/BidirectionalStreamData",
            request_serializer=e2e__test__service__pb2.StreamDataRequest.SerializeToString,
            response_deserializer=e2e__test__service__pb2.StreamDataResponse.FromString,
        )


class E2ETestServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Echo(self, request, context):
        """A simple unary RPC for echo functionality."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ServerStreamData(self, request, context):
        """A server-streaming RPC."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ClientStreamData(self, request_iterator, context):
        """A client-streaming RPC."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def BidirectionalStreamData(self, request_iterator, context):
        """A bidirectional-streaming RPC."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_E2ETestServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Echo": grpc.unary_unary_rpc_method_handler(
            servicer.Echo,
            request_deserializer=e2e__test__service__pb2.EchoRequest.FromString,
            response_serializer=e2e__test__service__pb2.EchoResponse.SerializeToString,
        ),
        "ServerStreamData": grpc.unary_stream_rpc_method_handler(
            servicer.ServerStreamData,
            request_deserializer=e2e__test__service__pb2.StreamDataRequest.FromString,
            response_serializer=e2e__test__service__pb2.StreamDataResponse.SerializeToString,
        ),
        "ClientStreamData": grpc.stream_unary_rpc_method_handler(
            servicer.ClientStreamData,
            request_deserializer=e2e__test__service__pb2.StreamDataRequest.FromString,
            response_serializer=e2e__test__service__pb2.EchoResponse.SerializeToString,
        ),
        "BidirectionalStreamData": grpc.stream_stream_rpc_method_handler(
            servicer.BidirectionalStreamData,
            request_deserializer=e2e__test__service__pb2.StreamDataRequest.FromString,
            response_serializer=e2e__test__service__pb2.StreamDataResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "tsercom.E2ETestService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class E2ETestService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Echo(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/tsercom.E2ETestService/Echo",
            e2e__test__service__pb2.EchoRequest.SerializeToString,
            e2e__test__service__pb2.EchoResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ServerStreamData(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/tsercom.E2ETestService/ServerStreamData",
            e2e__test__service__pb2.StreamDataRequest.SerializeToString,
            e2e__test__service__pb2.StreamDataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def ClientStreamData(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/tsercom.E2ETestService/ClientStreamData",
            e2e__test__service__pb2.StreamDataRequest.SerializeToString,
            e2e__test__service__pb2.EchoResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def BidirectionalStreamData(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/tsercom.E2ETestService/BidirectionalStreamData",
            e2e__test__service__pb2.StreamDataRequest.SerializeToString,
            e2e__test__service__pb2.StreamDataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
