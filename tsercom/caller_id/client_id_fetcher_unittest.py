import asyncio
import pytest
import unittest
import uuid # Import uuid
from unittest.mock import AsyncMock, patch

from tsercom.caller_id.client_id_fetcher import ClientIdFetcher
from tsercom.caller_id.caller_identifier import CallerIdentifier
# Assuming GetIdResponse is generated from a proto file.
# The actual import path might differ based on the project structure.
# Corrected import path based on file structure and grpc version
from tsercom.caller_id.proto.generated.v1_70.caller_id_pb2 import GetIdResponse, CallerId


@pytest.mark.asyncio
class TestClientIdFetcher:
    async def test_successful_fetch(self):
        # Mock the gRPC stub
        mock_stub = AsyncMock()
        id_string = str(uuid.uuid4()) # Use UUID
        caller_id_msg = CallerId()
        caller_id_msg.id = id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is not None
        assert str(caller_id) == id_string # Use str(caller_id)
        mock_stub.GetId.assert_called_once()

    async def test_caching_behavior(self):
        # Mock the gRPC stub
        mock_stub = AsyncMock()
        id_string = str(uuid.uuid4()) # Use UUID
        caller_id_msg = CallerId()
        caller_id_msg.id = id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)

        # First call to fetch and cache
        caller_id1 = await fetcher.get_id_async()
        assert caller_id1 is not None
        assert str(caller_id1) == id_string # Use str(caller_id1)
        mock_stub.GetId.assert_called_once()

        # Second call, should use cached value
        caller_id2 = await fetcher.get_id_async()
        assert caller_id2 is not None
        assert str(caller_id2) == id_string # Use str(caller_id2)
        # Assert GetId was still only called once
        mock_stub.GetId.assert_called_once()
        # Assert the same instance is returned
        assert caller_id1 is caller_id2

    async def test_correct_parsing(self):
        # This test is largely covered by test_successful_fetch,
        # as CallerIdentifier.try_parse is used internally by get_id_async.
        # We can add a specific check if there are complex parsing rules.
        mock_stub = AsyncMock()
        raw_id_string = str(uuid.uuid4()) # Use UUID
        caller_id_msg = CallerId()
        caller_id_msg.id = raw_id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is not None
        # This implicitly tests try_parse
        assert str(caller_id) == raw_id_string # Use str(caller_id)
        
        # Example of more specific parsing check if CallerIdentifier had more fields:
        # parsed_identifier = CallerIdentifier.try_parse(raw_id_string)
        # assert caller_id.some_parsed_field == parsed_identifier.some_parsed_field
        mock_stub.GetId.assert_called_once()

    async def test_fetch_failure_returns_none(self):
        # Mock the gRPC stub to raise an exception
        mock_stub = AsyncMock()
        mock_stub.GetId.side_effect = Exception("gRPC error")

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is None
        mock_stub.GetId.assert_called_once()

    async def test_empty_id_from_server(self):
        # Mock the gRPC stub
        mock_stub = AsyncMock()
        caller_id_msg = CallerId()
        caller_id_msg.id = "" # Empty ID string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        # try_parse returns None for empty string, so fetcher returns None
        assert caller_id is None
        mock_stub.GetId.assert_called_once()

    async def test_concurrent_fetches_call_rpc_once(self):
        # Mock the gRPC stub
        mock_stub = AsyncMock()
        raw_id_string = str(uuid.uuid4()) # Use UUID
        caller_id_msg = CallerId()
        caller_id_msg.id = raw_id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        
        # Introduce a delay to simulate network latency
        original_get_id = mock_stub.GetId
        async def delayed_get_id(*args, **kwargs):
            await asyncio.sleep(0.1) # 100ms delay
            return original_get_id.return_value
        
        mock_stub.GetId.return_value = mock_response
        mock_stub.GetId.side_effect = delayed_get_id # First call will be delayed

        fetcher = ClientIdFetcher(mock_stub)

        # Start multiple fetch calls concurrently
        tasks = [fetcher.get_id_async() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be the same and valid
        for caller_id in results:
            assert caller_id is not None
            assert str(caller_id) == raw_id_string # Use str(caller_id)
            assert caller_id is results[0] # All should be the same cached instance

        # The RPC GetId should have been called only once
        # Due to the side_effect wrapping, we need to check the original mock if it was part of an object
        # or the current mock if it's standalone. Here, mock_stub.GetId is the direct mock.
        # However, since the side_effect is an async function, we need to check its call count
        # if the mock library doesn't automatically delegate call_count for side_effect functions.
        # For AsyncMock, assert_called_once() should work as expected even with an async side_effect.
        # Let's refine this part if tests show issues.
        # We need to ensure the *actual* RPC call is what we're counting.
        # The test setup with side_effect might make mock_stub.GetId.call_count > 1
        # if the side_effect itself is counted.
        # A better way for this specific test might be to check the call count of the *original* GetId
        # if we had wrapped a real method. With a pure mock, this is tricky.
        # Let's assume for now that GetId.assert_called_once() is smart enough.
        # If not, we might need a more complex mock setup (e.g. a counter in the side_effect).

        # Re-evaluating: The lock in ClientIdFetcher should ensure GetId is called once.
        # The side_effect is on mock_stub.GetId.
        mock_stub.GetId.assert_called_once()

        # Call again to ensure cache is now populated and no more calls
        caller_id_after_concurrent = await fetcher.get_id_async()
        assert caller_id_after_concurrent is results[0]
        mock_stub.GetId.assert_called_once() # Still only once

    @patch('tsercom.caller_id.caller_identifier.CallerIdentifier.try_parse')
    async def test_parsing_failure_returns_none(self, mock_try_parse):
        # Mock the gRPC stub
        mock_stub = AsyncMock()
        # This ID will cause try_parse (the real one) to return None if it's not a UUID
        # or if it's specifically handled by the mock_try_parse patch.
        # For this test, we are *explicitly* mocking try_parse to return None,
        # so the actual content of "id_that_will_fail_parsing" doesn't strictly matter
        # for the real try_parse's behavior, only for what's passed to the mock.
        id_for_failed_parse = "id_that_will_fail_parsing"
        caller_id_msg = CallerId()
        caller_id_msg.id = id_for_failed_parse
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        # Mock CallerIdentifier.try_parse to return None
        mock_try_parse.return_value = None

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is None
        mock_stub.GetId.assert_called_once()
        mock_try_parse.assert_called_once_with(id_for_failed_parse)
