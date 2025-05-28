import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock # Keep AsyncMock for failed_channel_mock in multi-address test

import grpc # For status codes and AioRpcError
import grpc.aio # For Channel, AioRpcError

from tsercom.rpc.grpc_util.transport.local_ca_grpc_channel_factory import LocalCaGrpcChannelFactory
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.transport.grpc_channel_test_utils import mock_provider

# Common test data - specific to these tests
TEST_ADDRESS = "localhost"
TEST_PORT = 12345
MOCK_CA_PATH = "/path/to/ca.crt"
MOCK_CERT_BYTES = b"cert_data"
MOCK_GRPC_CREDS = MagicMock(spec=grpc.ChannelCredentials)


# --- Tests for LocalCaGrpcChannelFactory ---
@pytest.mark.asyncio
async def test_local_ca_success(mock_provider): # Type hint for mock_provider is in the fixture
    factory = LocalCaGrpcChannelFactory(MOCK_CA_PATH, mock_provider)
    
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    # mock_provider.mock_aio_channel.channel_ready will resolve by default

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    assert channel_info.address == TEST_ADDRESS
    assert channel_info.port == TEST_PORT
    mock_provider.read_file_content_mock.assert_called_once_with(MOCK_CA_PATH)
    mock_provider.create_ssl_channel_credentials_mock.assert_called_once_with(root_certificates=MOCK_CERT_BYTES)
    mock_provider.create_secure_channel_mock.assert_called_once_with(
        f"{TEST_ADDRESS}:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()

@pytest.mark.asyncio
async def test_local_ca_file_not_found(mock_provider):
    factory = LocalCaGrpcChannelFactory(MOCK_CA_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = None

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.read_file_content_mock.assert_called_once_with(MOCK_CA_PATH)
    mock_provider.create_ssl_channel_credentials_mock.assert_not_called()

@pytest.mark.asyncio
async def test_local_ca_cred_creation_failure(mock_provider):
    factory = LocalCaGrpcChannelFactory(MOCK_CA_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = None

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.create_ssl_channel_credentials_mock.assert_called_once_with(root_certificates=MOCK_CERT_BYTES)
    mock_provider.create_secure_channel_mock.assert_not_called()

@pytest.mark.asyncio
async def test_local_ca_channel_ready_timeout(mock_provider):
    factory = LocalCaGrpcChannelFactory(MOCK_CA_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    mock_provider.mock_aio_channel.channel_ready.side_effect = asyncio.TimeoutError

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_local_ca_channel_ready_aio_rpc_error(mock_provider):
    factory = LocalCaGrpcChannelFactory(MOCK_CA_PATH, mock_provider)
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    # Need to ensure the AioRpcError is properly instantiated
    mock_provider.mock_aio_channel.channel_ready.side_effect = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE, initial_metadata=None, trailing_metadata=None, details=None
    )

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()
    
@pytest.mark.asyncio
async def test_local_ca_multiple_addresses_first_fails(mock_provider):
    factory = LocalCaGrpcChannelFactory(MOCK_CA_PATH, mock_provider)
    addresses = ["bad_address", TEST_ADDRESS]
    
    mock_provider.read_file_content_mock.return_value = MOCK_CERT_BYTES
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    
    # First call to create_secure_channel will use a channel that fails on ready
    failed_channel_mock = MagicMock(spec=grpc.aio.Channel)
    failed_channel_mock.channel_ready = AsyncMock(side_effect=asyncio.TimeoutError)
    failed_channel_mock.close = AsyncMock()

    # Subsequent call to create_secure_channel uses the default mock_aio_channel which succeeds
    # This requires mock_provider.create_secure_channel_mock to return the new mock first,
    # then the standard one.
    # The mock_provider fixture provides a fresh provider for each test, so its
    # create_secure_channel_mock.return_value is the standard self.mock_aio_channel.
    # To have it return different channels for different calls within the SAME test,
    # we set its side_effect here.
    
    # Get the standard successful channel from the provider to use for the second call
    standard_successful_channel = mock_provider.create_secure_channel_mock.return_value

    mock_provider.create_secure_channel_mock.side_effect = [
        failed_channel_mock,
        standard_successful_channel 
    ]

    channel_info = await factory.find_async_channel(addresses, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    assert channel_info.address == TEST_ADDRESS
    assert mock_provider.read_file_content_mock.call_count == 1 # Only called once for CA cert
    assert mock_provider.create_ssl_channel_credentials_mock.call_count == 1 # Only called once
    assert mock_provider.create_secure_channel_mock.call_count == 2 # Called for each address
    
    # Check first call (failed one)
    # The mock_provider.create_secure_channel_mock itself was called with these args
    mock_provider.create_secure_channel_mock.assert_any_call(
        f"bad_address:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    failed_channel_mock.channel_ready.assert_awaited_once()
    failed_channel_mock.close.assert_awaited_once()

    # Check second call (successful one)
    mock_provider.create_secure_channel_mock.assert_any_call(
        f"{TEST_ADDRESS}:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    # standard_successful_channel is mock_provider.mock_aio_channel from the fixture instance
    standard_successful_channel.channel_ready.assert_awaited_once()
    # standard_successful_channel.close should not be called as it succeeded
    standard_successful_channel.close.assert_not_called()
