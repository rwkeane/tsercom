from tsercom.rpc.proto import TestConnectionCall, TestConnectionResponse


class AsyncTestConnectionServer:
    async def TestConnection(  # type: ignore
        self, request: TestConnectionCall, context
    ) -> TestConnectionResponse:
        return TestConnectionResponse()


class TestConnectionServer:
    def TestConnection(  # type: ignore
        self, request: TestConnectionCall, context
    ) -> TestConnectionResponse:
        return TestConnectionResponse()
