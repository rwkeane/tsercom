import sys
import types
if "grpc.addressing_unittest" not in sys.modules:
    dummy_module = types.ModuleType("grpc.addressing_unittest")
    sys.modules["grpc.addressing_unittest"] = dummy_module
import pytest
# from unittest.mock import MagicMock, patch # Removed

# Define a simple class to be used as a spec for the context mock
class ServicerContextSpec:
    peer = None # This attribute will be mocked by MagicMock

@pytest.fixture
def mock_context(mocker): # Removed type hint MagicMock for fixture return, as it's a mock.
    """
    Provides a mock ServicerContext instance for tests.
    The `grpc.aio.ServicerContext` used by the SUT will be patched to this.
    The `peer` method of this instance is also a MagicMock and can be configured per test.
    """
    mock_grpc_root = mocker.MagicMock(name="RootGrpcMockForAddressing")
    mock_grpc_aio = mocker.MagicMock(name="AioMockForAddressing")
    
    # MockServicerContextClass will be a mock class, its instances will conform to ServicerContextSpec
    MockServicerContextClass = mocker.MagicMock(name="MockedServicerContextClass", spec=ServicerContextSpec)
    
    # When an instance of MockServicerContextClass is created, its 'peer' attribute will also be a MagicMock.
    # Example: instance = MockServicerContextClass(); instance.peer will be a MagicMock.
    
    mock_grpc_aio.ServicerContext = MockServicerContextClass 
    mock_grpc_root.aio = mock_grpc_aio
    
    # Patch 'grpc' in the 'tsercom.rpc.grpc.addressing' module to be our mock_grpc_root
    mocker.patch('tsercom.rpc.grpc.addressing.grpc', new=mock_grpc_root)
    
    # Return an instance of the mocked ServicerContext class.
    # This instance will have a `peer` attribute that is a MagicMock due to the spec.
    mock_context_instance = MockServicerContextClass()
    mock_context_instance.peer = mocker.MagicMock(name="mock_context_peer_method") # Explicitly make peer a mock
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
        ("ipv4:192.168.1.10", "192.168.1.10"), # Test case from original: assumes port might be missing
        ("", None), # Empty peer string
        ("ipv4:", ""), # Malformed
        ("ipv6:", ""), # Malformed
        ("ipv4:127.0.0.1:8080", "127.0.0.1"),
        ("ipv6:[::ffff:192.0.2.128]:80", "::ffff:192.0.2.128"), # IPv4-mapped IPv6
        ("ipv4::12345", None), # Malformed
        ("ipv6:::12345", None), # Malformed
    ],
)
def test_get_client_ip(
    mock_context, peer_string: str, expected_ip: str | None # mock_context is now untyped from fixture perspective
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
        ("ipv4:192.168.1.10:abc", None), # Non-integer port
        ("ipv4:192.168.1.10", None),    # No port
        ("ipv4:192.168.1.10:0", 0),     # Port zero
        ("", None),                     # Empty peer string
        ("ipv6:[2001:db8::1]:65535", 65535), # Max valid port
        ("ipv4:127.0.0.1:", None),      # No port number after colon
        ("unix:/tmp/socket", None),     # Unix socket, no port
        ("ipv4:1.2.3.4:12345extra", None), # Extra chars after port
    ],
)
def test_get_client_port(
    mock_context, # mock_context is now untyped from fixture perspective
    peer_string: str,
    expected_port: int | None,
):
    from tsercom.rpc.grpc.addressing import get_client_port  # SUT import

    mock_context.peer.return_value = peer_string
    assert get_client_port(mock_context) == expected_port
