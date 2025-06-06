"""Provides simple gRPC server components for testing connections."""

import grpc

from tsercom.rpc.proto import TestConnectionCall, TestConnectionResponse


class AsyncTestConnectionServer:
    """An asynchronous server for the TestConnection RPC method."""

    async def TestConnection(
        self, request: TestConnectionCall, context: grpc.aio.ServicerContext
    ) -> TestConnectionResponse:
        """Handles an asynchronous TestConnection request. Simply returns an empty response."""
        return TestConnectionResponse()


class TestConnectionServer:
    """A synchronous server for the TestConnection RPC method."""

    def TestConnection(
        self, request: TestConnectionCall, context: grpc.ServicerContext
    ) -> TestConnectionResponse:
        """Handles a synchronous TestConnection request. Simply returns an empty response."""
        return TestConnectionResponse()
