import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock # AsyncMock for multi-address tests

import grpc # For status codes and AioRpcError
import grpc.aio # For Channel, AioRpcError

from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import InsecureGrpcChannelFactory
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.transport.grpc_channel_test_utils import mock_provider

# Common test data - specific to these tests
TEST_ADDRESS = "localhost"
TEST_PORT = 12345

# --- Tests for InsecureGrpcChannelFactory ---
@pytest.mark.asyncio
async def test_insecure_success(mock_provider):
    factory = InsecureGrpcChannelFactory(mock_provider)
    # mock_provider.mock_aio_channel.channel_ready will resolve by default (from grpc_channel_test_utils)

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    assert channel_info.address == TEST_ADDRESS
    assert channel_info.port == TEST_PORT
    mock_provider.create_insecure_channel_mock.assert_called_once_with(
        target=f"{TEST_ADDRESS}:{TEST_PORT}", options=None
    )
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()

@pytest.mark.asyncio
async def test_insecure_channel_ready_timeout(mock_provider):
    factory = InsecureGrpcChannelFactory(mock_provider)
    mock_provider.mock_aio_channel.channel_ready.side_effect = asyncio.TimeoutError

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.create_insecure_channel_mock.assert_called_once_with(
        target=f"{TEST_ADDRESS}:{TEST_PORT}", options=None
    )
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_insecure_channel_ready_aio_rpc_error(mock_provider):
    factory = InsecureGrpcChannelFactory(mock_provider)
    mock_provider.mock_aio_channel.channel_ready.side_effect = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE, initial_metadata=None, trailing_metadata=None, details=None
    )

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.create_insecure_channel_mock.assert_called_once_with(
        target=f"{TEST_ADDRESS}:{TEST_PORT}", options=None
    )
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_insecure_multiple_addresses_first_fails(mock_provider):
    factory = InsecureGrpcChannelFactory(mock_provider)
    addresses = ["bad_address", TEST_ADDRESS]
    
    failed_channel_mock = MagicMock(spec=grpc.aio.Channel)
    failed_channel_mock.channel_ready = AsyncMock(side_effect=asyncio.TimeoutError)
    failed_channel_mock.close = AsyncMock()

    standard_successful_channel = mock_provider.create_insecure_channel_mock.return_value 
    mock_provider.create_insecure_channel_mock.side_effect = [
        failed_channel_mock,
        standard_successful_channel 
    ]

    channel_info = await factory.find_async_channel(addresses, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    assert channel_info.address == TEST_ADDRESS
    assert mock_provider.create_insecure_channel_mock.call_count == 2
    
    mock_provider.create_insecure_channel_mock.assert_any_call(
        target=f"bad_address:{TEST_PORT}", options=None
    )
    failed_channel_mock.channel_ready.assert_awaited_once()
    failed_channel_mock.close.assert_awaited_once()

    mock_provider.create_insecure_channel_mock.assert_any_call(
        target=f"{TEST_ADDRESS}:{TEST_PORT}", options=None
    )
    standard_successful_channel.channel_ready.assert_awaited_once()
    standard_successful_channel.close.assert_not_called()
