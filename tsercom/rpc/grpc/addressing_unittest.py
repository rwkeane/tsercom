import pytest
from unittest.mock import MagicMock, patch

# This mock will be used as the spec for the context instance.
MockServicerContextSpec = MagicMock(name="MockServicerContextSpec")
MockServicerContextSpec.peer = MagicMock()


@pytest.fixture
def mock_context(mocker) -> MagicMock:
    """
    Provides a mock ServicerContext instance for tests.
    The `grpc.aio.ServicerContext` used by the SUT will be patched to this.
    The `peer` method of this instance is also a MagicMock and can be configured per test.
    """
    # Patch 'grpc.aio.ServicerContext' specifically within the 'tsercom.rpc.grpc.addressing' module.
    # This ensures that when 'addressing.py' does 'import grpc' and then uses 'grpc.aio.ServicerContext',
    # it gets our mock.
    
    # We need to mock 'grpc' then 'grpc.aio' then 'grpc.aio.ServicerContext'
    # However, the SUT directly uses 'import grpc' and then 'context: "grpc.aio.ServicerContext"'.
    # So, we need to mock 'grpc' as seen by the SUT.
    
    mock_grpc_root = MagicMock(name="RootGrpcMockForAddressing")
    mock_grpc_aio = MagicMock(name="AioMockForAddressing")
    
    # This is the class that will be used for type hinting and potentially by isinstance if not using spec
    MockServicerContextClass = MagicMock(name="MockedServicerContextClass", spec=MockServicerContextSpec)
    
    mock_grpc_aio.ServicerContext = MockServicerContextClass
    mock_grpc_root.aio = mock_grpc_aio
    
    mocker.patch('tsercom.rpc.grpc.addressing.grpc', mock_grpc_root)
    
    # Return an instance of the mocked ServicerContext class
    # This instance will have a `peer` attribute that is a MagicMock due to the spec.
    # Tests can then configure `mock_context_instance.peer.return_value`.
    mock_context_instance = MockServicerContextClass()
    return mock_context_instance


# Tests for get_client_ip
@pytest.mark.parametrize(
    "peer_string, expected_ip",
    [
        ("ipv4:192.168.1.10:12345", "192.168.1.10"),
        ("ipv6:[::1]:12345", "::1"),
        ("ipv6:[2001:db8::1]:12345", "2001:db8::1"),
        ("unix:/tmp/socket", "localhost"),
        ("unknown:address", None),
        ("ipv4:192.168.1.10", "192.168.1.10"),
        ("", None),
        ("ipv4:", ""),
        ("ipv6:", ""),
        ("ipv4:127.0.0.1:8080", "127.0.0.1"),
        ("ipv6:[::ffff:192.0.2.128]:80", "::ffff:192.0.2.128"),
        ("ipv4::12345", None),
        ("ipv6:::12345", None),
    ],
)
def test_get_client_ip(
    mock_context: MagicMock, peer_string: str, expected_ip: str | None
):
    from tsercom.rpc.grpc.addressing import get_client_ip  # SUT import

    mock_context.peer.return_value = peer_string
    assert get_client_ip(mock_context) == expected_ip


# Tests for get_client_port
@pytest.mark.parametrize(
    "peer_string, expected_port",
    [
        ("ipv4:192.168.1.10:12345", 12345),
        ("ipv6:[::1]:80", 80),
        ("ipv4:192.168.1.10:abc", None),
        ("ipv4:192.168.1.10", None),
        ("ipv4:192.168.1.10:0", 0),
        ("", None),
        ("ipv6:[2001:db8::1]:65535", 65535),
        ("ipv4:127.0.0.1:", None),
        ("unix:/tmp/socket", None),
        ("ipv4:1.2.3.4:12345extra", None),
    ],
)
def test_get_client_port(
    mock_context: MagicMock,
    peer_string: str,
    expected_port: int | None,
):
    from tsercom.rpc.grpc.addressing import get_client_port  # SUT import

    mock_context.peer.return_value = peer_string
    assert get_client_port(mock_context) == expected_port
