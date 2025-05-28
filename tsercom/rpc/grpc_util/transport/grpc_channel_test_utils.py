import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Any, TYPE_CHECKING

import grpc # For type hints and ChannelCredentials
import grpc.aio # For type hints and aio.Channel

# Absolute import for the base class GrpcChannelCredentialsProvider
from tsercom.rpc.grpc_util.grpc_channel_credentials_provider import GrpcChannelCredentialsProvider

if TYPE_CHECKING:
    # This is for pytest_mock.MockerFixture, but unittest.mock is used directly.
    # No specific type hints from TYPE_CHECKING seem immediately necessary for this file's content.
    pass

# Mock Implementation of GrpcChannelCredentialsProvider
class MockGrpcChannelCredentialsProvider(GrpcChannelCredentialsProvider):
    def __init__(self):
        super().__init__() # Though GrpcChannelCredentialsProvider is ABC, super call is good practice
        self.read_file_content_mock = MagicMock(spec=self.read_file_content)
        self.create_ssl_channel_credentials_mock = MagicMock(spec=self.create_ssl_channel_credentials)
        
        # Mock for the channel object itself
        self.mock_aio_channel = MagicMock(spec=grpc.aio.Channel)
        self.mock_aio_channel.channel_ready = AsyncMock()
        self.mock_aio_channel.close = AsyncMock()

        self.create_secure_channel_mock = MagicMock(
            spec=self.create_secure_channel,
            return_value=self.mock_aio_channel
        )
        self.create_insecure_channel_mock = MagicMock(
            spec=self.create_insecure_channel,
            return_value=self.mock_aio_channel
        )

    def read_file_content(self, path: str) -> bytes | None:
        return self.read_file_content_mock(path)

    def create_ssl_channel_credentials(
        self,
        root_certificates: bytes | None = None,
        private_key: bytes | None = None,
        certificate_chain: bytes | None = None,
    ) -> grpc.ChannelCredentials | None:
        return self.create_ssl_channel_credentials_mock(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )

    def create_secure_channel(
        self,
        target: str,
        credentials: 'grpc.ChannelCredentials | None',
        options: list[tuple[str, Any]] | None = None,
    ) -> 'grpc.aio.Channel | None':
        # Reset channel sub-mocks for multiple calls if needed, or ensure they are fresh
        # This is important because the same mock_aio_channel instance is returned.
        self.mock_aio_channel.channel_ready.reset_mock()
        self.mock_aio_channel.close.reset_mock()
        # Ensure side_effects are also cleared if they were set on channel_ready or close
        # for specific test scenarios. For a generic mock, this might not be needed,
        # but if tests modify these, they should be reset.
        # Default behavior of AsyncMock is to return a new awaitable mock each time.
        # If a side_effect was set (like an exception), it should be reset for the next test.
        # However, pytest fixtures usually create a fresh mock_provider for each test.
        # So, this reset is more for cases where the same provider instance is reused.
        self.mock_aio_channel.channel_ready.side_effect = None 
        self.mock_aio_channel.close.side_effect = None
        return self.create_secure_channel_mock(target, credentials, options=options)

    def create_insecure_channel(
        self, target: str, options: list[tuple[str, Any]] | None = None
    ) -> 'grpc.aio.Channel | None':
        self.mock_aio_channel.channel_ready.reset_mock()
        self.mock_aio_channel.close.reset_mock()
        self.mock_aio_channel.channel_ready.side_effect = None
        self.mock_aio_channel.close.side_effect = None
        return self.create_insecure_channel_mock(target, options=options)

@pytest.fixture
def mock_provider() -> MockGrpcChannelCredentialsProvider:
    """Pytest fixture to provide a fresh MockGrpcChannelCredentialsProvider for each test."""
    return MockGrpcChannelCredentialsProvider()

# Common test data can also be moved here if it's truly shared across all factory tests
# For now, keeping it in the original file until individual test files are created.
# Example:
# TEST_ADDRESS = "localhost"
# TEST_PORT = 12345
# MOCK_CERT_BYTES = b"cert_data"
# MOCK_KEY_BYTES = b"key_data"
# MOCK_GRPC_CREDS = MagicMock(spec=grpc.ChannelCredentials)
