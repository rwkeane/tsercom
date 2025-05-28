import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock # AsyncMock for multi-address tests

import grpc # For status codes and AioRpcError
import grpc.aio # For Channel, AioRpcError

from tsercom.rpc.grpc_util.transport.client_cert_grpc_channel_factory import ClientCertGrpcChannelFactory
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.rpc.grpc_util.transport.grpc_channel_test_utils import mock_provider

# Common test data - specific to these tests
TEST_ADDRESS = "localhost"
TEST_PORT = 12345
MOCK_CLIENT_KEY_PATH = "/path/to/client.key"
MOCK_CLIENT_CERT_PATH = "/path/to/client.crt"
MOCK_KEY_BYTES = b"key_data"
MOCK_CERT_BYTES = b"cert_data"
MOCK_GRPC_CREDS = MagicMock(spec=grpc.ChannelCredentials)


# --- Tests for ClientCertGrpcChannelFactory ---
@pytest.mark.asyncio
async def test_client_cert_success(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, MOCK_CERT_BYTES]
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)

    assert isinstance(channel_info, ChannelInfo)
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_KEY_PATH)
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_CERT_PATH)
    assert mock_provider.read_file_content_mock.call_count == 2
    mock_provider.create_ssl_channel_credentials_mock.assert_called_once_with(
        root_certificates=None, private_key=MOCK_KEY_BYTES, certificate_chain=MOCK_CERT_BYTES
    )
    mock_provider.create_secure_channel_mock.assert_called_once_with(
        f"{TEST_ADDRESS}:{TEST_PORT}", MOCK_GRPC_CREDS, options=None
    )
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_cert_key_file_not_found(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.side_effect = [None, MOCK_CERT_BYTES] # Key not found

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_KEY_PATH)
    # Factory reads key first, if None, it returns. So cert read_file_content_mock won't be called.
    # Adjusting based on factory's actual behavior (it short-circuits if key is None)
    assert mock_provider.read_file_content_mock.call_count == 1 
    mock_provider.create_ssl_channel_credentials_mock.assert_not_called()


@pytest.mark.asyncio
async def test_client_cert_cert_file_not_found(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, None] # Cert not found

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_KEY_PATH)
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_CERT_PATH)
    assert mock_provider.read_file_content_mock.call_count == 2
    mock_provider.create_ssl_channel_credentials_mock.assert_not_called()


@pytest.mark.asyncio
async def test_client_cert_cred_creation_failure(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, MOCK_CERT_BYTES]
    mock_provider.create_ssl_channel_credentials_mock.return_value = None

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.create_ssl_channel_credentials_mock.assert_called_once_with(
        root_certificates=None, private_key=MOCK_KEY_BYTES, certificate_chain=MOCK_CERT_BYTES
    )
    mock_provider.create_secure_channel_mock.assert_not_called()

@pytest.mark.asyncio
async def test_client_cert_channel_ready_timeout(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, MOCK_CERT_BYTES]
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    mock_provider.mock_aio_channel.channel_ready.side_effect = asyncio.TimeoutError

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_cert_channel_ready_aio_rpc_error(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, MOCK_CERT_BYTES]
    mock_provider.create_ssl_channel_credentials_mock.return_value = MOCK_GRPC_CREDS
    mock_provider.mock_aio_channel.channel_ready.side_effect = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE, initial_metadata=None, trailing_metadata=None, details=None
    )

    channel_info = await factory.find_async_channel(TEST_ADDRESS, TEST_PORT)
    assert channel_info is None
    mock_provider.mock_aio_channel.channel_ready.assert_awaited_once()
    mock_provider.mock_aio_channel.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_cert_multiple_addresses_first_fails(mock_provider):
    factory = ClientCertGrpcChannelFactory(MOCK_CLIENT_KEY_PATH, MOCK_CLIENT_CERT_PATH, mock_provider)
    addresses = ["bad_address", TEST_ADDRESS]
    
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, MOCK_CERT_BYTES, # For first attempt
                                                         MOCK_KEY_BYTES, MOCK_CERT_BYTES] # For second attempt (if files were read per attempt)
                                                         # The factory reads files once at the start. So this side_effect needs care.
    
    # Reset side_effect for read_file_content_mock as it's called once for key and once for cert
    # at the beginning of find_async_channel, not per address.
    mock_provider.read_file_content_mock.side_effect = [MOCK_KEY_BYTES, MOCK_CERT_BYTES]
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
    
    # File reading should happen only once for key and once for cert
    assert mock_provider.read_file_content_mock.call_count == 2 
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_KEY_PATH)
    mock_provider.read_file_content_mock.assert_any_call(MOCK_CLIENT_CERT_PATH)

    # Credential creation should happen only once
    assert mock_provider.create_ssl_channel_credentials_mock.call_count == 1
    
    # Secure channel creation called for each address
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
