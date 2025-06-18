import asyncio
from typing import AsyncIterator, Optional

from tsercom.test.proto.generated.e2e_test_service_pb2 import EchoRequest, EchoResponse, StreamDataRequest, StreamDataResponse
from tsercom.test.proto.generated.e2e_test_service_pb2_grpc import E2ETestServiceServicer

# Placeholder for a data handler or queue to push received data for verification
# In a real test, this would be something the test can access to verify data.
_received_data_queue: asyncio.Queue[str] = asyncio.Queue()

class E2eTestServicer(E2ETestServiceServicer):
    """
    gRPC servicer for the E2ETestService.
    """

    async def Echo(self, request: EchoRequest, context: Optional[object] = None) -> EchoResponse:
        """
        Handles the Echo RPC request.

        Args:
            request: The EchoRequest containing the message to echo.
            context: The gRPC context (optional).

        Returns:
            An EchoResponse containing the echoed message.
        """
        # Push the received message to the queue for verification
        await _received_data_queue.put(request.message)
        return EchoResponse(response=f"Echo: {request.message}")

    async def ServerStreamData(
        self, request: StreamDataRequest, context: Optional[object] = None
    ) -> AsyncIterator[StreamDataResponse]:
        """
        Handles the ServerStreamData RPC request. (Not fully implemented for this example)
        """
        for i in range(3):  # Example: Send 3 data chunks
            await asyncio.sleep(0.1) # Simulate some work
            yield StreamDataResponse(data_chunk=f"Chunk {i} for ID {request.data_id}", sequence_number=i)

    async def ClientStreamData(
        self, request_iterator: AsyncIterator[StreamDataRequest], context: Optional[object] = None
    ) -> EchoResponse:
        """
        Handles the ClientStreamData RPC request. (Not fully implemented for this example)
        """
        count = 0
        async for request in request_iterator:
            # In a real scenario, process request.data_id
            await _received_data_queue.put(f"ClientStream ID: {request.data_id}")
            count += 1
        return EchoResponse(response=f"Received {count} client stream messages.")

    async def BidirectionalStreamData(
        self, request_iterator: AsyncIterator[StreamDataRequest], context: Optional[object] = None
    ) -> AsyncIterator[StreamDataResponse]:
        """
        Handles the BidirectionalStreamData RPC request. (Not fully implemented for this example)
        """
        seq_num = 0
        async for request in request_iterator:
            await _received_data_queue.put(f"BiDiStream ID: {request.data_id}")
            yield StreamDataResponse(data_chunk=f"BiDi Response to {request.data_id}", sequence_number=seq_num)
            seq_num += 1

def get_received_data_queue() -> asyncio.Queue[str]:
    """
    Returns the queue used by the servicer to store received messages.
    This allows tests to access and verify the data.
    """
    return _received_data_queue
