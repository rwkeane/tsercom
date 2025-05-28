import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock # AsyncMock might be needed for multi-address tests if added

import grpc # For status codes and AioRpcError
import grpc.aio # For Channel, AioRpcError

from tsercom.rpc.grpc_util.transport.specific_cert_grpc_channel_factory import SpecificCertGrpcChannelFactory
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.transport.grpc_channel_test_utils import mock_provider

# Common test data - specific to these tests
TEST_ADDRESS = "localhost"
TEST_PORT = 12345
MOCK_SERVER_CERT_PATH = "/path/to/server.crt"
MOCK_CERT_BYTES = b"cert_data"
MOCK_GRPC_CREDS = MagicMock(spec=grpc.ChannelCredentials)


# --- Tests for SpecificCertGrpcChannelFactory ---
@pytest.mark.asyncio
async def test_specific_cert_success(mock_provider):
    factory = SpecificCertGrpcChannelFactory(MOCK_SERVER_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    mock_provider.read_file_content_mock.assert_called_once_with(MOCK_SERVER_CERT_PATH)
    mock_provider.create_ssl_channel_credentials_mock.assert_called_once_with(root_certificates=MOCK_CERT_BYTES)
    mock_provider.create_secure_channel_mock.assert_called_once_with(
        f"{TEST_ADDRESS}:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()

@pytest.mark.asyncio
async def test_specific_cert_file_not_found(mock_provider):
    factory = SpecificCertGrpcChannelFactory(MOCK_SERVER_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = None

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.read_file_content_mock.assert_called_once_with(MOCK_SERVER_CERT_PATH)
    mock_provider.create_ssl_channel_credentials_mock.assert_not_called()

@pytest.mark.asyncio
async def test_specific_cert_cred_creation_failure(mock_provider):
    factory = SpecificCertGrpcChannelFactory(MOCK_SERVER_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = None

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.create_ssl_channel_credentials_mock.assert_called_once_with(root_certificates=MOCK_CERT_BYTES)
    mock_provider.create_secure_channel_mock.assert_not_called()

@pytest.mark.asyncio
async def test_specific_cert_channel_ready_timeout(mock_provider):
    factory = SpecificCertGrpcChannelFactory(MOCK_SERVER_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    mock_provider.mock_aio_channel.channel_ready.side_effect = asyncio.TimeoutError

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_specific_cert_channel_ready_aio_rpc_error(mock_provider):
    factory = SpecificCertGrpcChannelFactory(MOCK_SERVER_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    mock_provider.mock_aio_channel.channel_ready.side_effect = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE, initial_metadata=None, trailing_metadata=None, details=None
    )

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_specific_cert_multiple_addresses_first_fails(mock_provider):
    factory = SpecificCertGrpcChannelFactory(MOCK_SERVER_CERT_PATH, mock_provider)
    addresses = ["bad_address", TEST_ADDRESS]
    
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    
    failed_channel_mock = MagicMock(spec=grpc.aio.Channel)
    failed_channel_mock.channel_ready = AsyncMock(side_effect=asyncio.TimeoutError)
    failed_channel_mock.close = AsyncMock()

    standard_successful_channel = mock_provider.create_secure_channel_mock.return_value
    mock_provider.create_secure_channel_mock.side_effect = [
        failed_channel_mock,
        standard_successful_channel 
    ]

    channel_info = await factory.find_async_channel(addresses, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    assert channel_info.address == TEST_ADDRESS
    assert mock_provider.read_file_content_mock.call_count == 1
    assert mock_provider.create_ssl_channel_credentials_mock.call_count == 1
    assert mock_provider.create_secure_channel_mock.call_count == 2
    
    mock_provider.create_secure_channel_mock.assert_any_call(
        f"bad_address:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    failed_channel_mock.channel_ready.assert_awaited_once()
    failed_channel_mock.close.assert_awaited_once()

    mock_provider.create_secure_channel_mock.assert_any_call(
        f"{TEST_ADDRESS}:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    standard_successful_channel.channel_ready.assert_awaited_once()
    standard_successful_channel.close.assert_not_called()
