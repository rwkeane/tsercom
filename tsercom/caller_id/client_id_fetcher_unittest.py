import asyncio
import pytest

import uuid

from tsercom.caller_id.client_id_fetcher import ClientIdFetcher
from tsercom.caller_id.caller_identifier import CallerIdentifier

# Assuming GetIdResponse is generated from a proto file.
# The actual import path might differ based on the project structure.
# Corrected import path based on file structure and grpc version
from tsercom.caller_id.proto import (
    GetIdResponse,
    CallerId,
)


@pytest.mark.asyncio
class TestClientIdFetcher:
    async def test_successful_fetch(self, mocker):
        # Mock the gRPC stub
        mock_stub = mocker.AsyncMock()
        id_string = str(uuid.uuid4())
        caller_id_msg = CallerId()
        caller_id_msg.id = id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is not None
        assert str(caller_id) == id_string
        mock_stub.GetId.assert_called_once()

    async def test_caching_behavior(self, mocker):
        # Mock the gRPC stub
        mock_stub = mocker.AsyncMock()
        id_string = str(uuid.uuid4())
        caller_id_msg = CallerId()
        caller_id_msg.id = id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)

        # First call to fetch and cache
        caller_id1 = await fetcher.get_id_async()
        assert caller_id1 is not None
        assert str(caller_id1) == id_string
        mock_stub.GetId.assert_called_once()

        # Second call, should use cached value
        caller_id2 = await fetcher.get_id_async()
        assert caller_id2 is not None
        assert str(caller_id2) == id_string
        mock_stub.GetId.assert_called_once()
        assert caller_id1 is caller_id2

    async def test_correct_parsing(self, mocker):
        mock_stub = mocker.AsyncMock()
        raw_id_string = str(uuid.uuid4())
        caller_id_msg = CallerId()
        caller_id_msg.id = raw_id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is not None
        assert str(caller_id) == raw_id_string
        mock_stub.GetId.assert_called_once()

    async def test_fetch_failure_returns_none(self, mocker):
        mock_stub = mocker.AsyncMock()
        mock_stub.GetId.side_effect = Exception("gRPC error")

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is None
        mock_stub.GetId.assert_called_once()

    async def test_empty_id_from_server(self, mocker):
        mock_stub = mocker.AsyncMock()
        caller_id_msg = CallerId()
        caller_id_msg.id = ""
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is None
        mock_stub.GetId.assert_called_once()

    async def test_concurrent_fetches_call_rpc_once(self, mocker):
        mock_stub = mocker.AsyncMock()
        raw_id_string = str(uuid.uuid4())
        caller_id_msg = CallerId()
        caller_id_msg.id = raw_id_string
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)

        original_get_id = mock_stub.GetId

        async def delayed_get_id(*args, **kwargs):
            await asyncio.sleep(0.1)
            return original_get_id.return_value

        mock_stub.GetId.return_value = mock_response
        mock_stub.GetId.side_effect = delayed_get_id

        fetcher = ClientIdFetcher(mock_stub)

        tasks = [fetcher.get_id_async() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        for caller_id in results:
            assert caller_id is not None
            assert str(caller_id) == raw_id_string
            assert caller_id is results[0]

        mock_stub.GetId.assert_called_once()

        caller_id_after_concurrent = await fetcher.get_id_async()
        assert caller_id_after_concurrent is results[0]
        mock_stub.GetId.assert_called_once()

    # Patch decorator needs to be changed to use mocker
    async def test_parsing_failure_returns_none(self, mocker):
        # Mock the gRPC stub
        mock_stub = mocker.AsyncMock()
        id_for_failed_parse = "id_that_will_fail_parsing"
        caller_id_msg = CallerId()
        caller_id_msg.id = id_for_failed_parse
        mock_response = GetIdResponse()
        mock_response.id.CopyFrom(caller_id_msg)
        mock_stub.GetId.return_value = mock_response

        # Use mocker.patch for the context manager or decorator functionality
        # Patching where it's used by the SUT (ClientIdFetcher)
        mock_try_parse = mocker.patch(
            "tsercom.caller_id.client_id_fetcher.CallerIdentifier.try_parse"
        )
        mock_try_parse.return_value = None

        fetcher = ClientIdFetcher(mock_stub)
        caller_id = await fetcher.get_id_async()

        assert caller_id is None
        mock_stub.GetId.assert_called_once()
        mock_try_parse.assert_called_once_with(id_for_failed_parse)
