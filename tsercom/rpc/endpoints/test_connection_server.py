from tsercom.rpc.proto import TestConnectionCall, TestConnectionResponse


class AsyncTestConnectionServer:
    async def TestConnection(self,
                             request : TestConnectionCall,
                             context) -> TestConnectionResponse:
        return TestConnectionResponse()
    
class TestConnectionServer:
    def TestConnection(self,
                       request : TestConnectionCall,
                       context) -> TestConnectionResponse:
        return TestConnectionResponse()
