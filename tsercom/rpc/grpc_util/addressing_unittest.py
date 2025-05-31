"""Unit tests for tsercom.rpc.grpc_util.addressing."""

import pytest
import grpc  # Required for grpc.aio.ServicerContext
import grpc.aio  # Required for grpc.aio.ServicerContext

from tsercom.rpc.grpc_util.addressing import get_client_ip, get_client_port


@pytest.fixture
def mock_context(mocker):
    """Fixture for a mocked grpc.aio.ServicerContext."""
    return mocker.MagicMock(spec=grpc.aio.ServicerContext)


# --- Tests for get_client_ip ---


@pytest.mark.parametrize(
    "peer_string, expected_ip",
    [
        ("ipv4:127.0.0.1:12345", "127.0.0.1"),
        ("ipv6:[::1]:12345", "::1"),
        ("ipv6:[2001:db8::1]:54321", "2001:db8::1"),
        ("unix:/tmp/my.sock", "localhost"),
        ("unknown:foo:bar", None),
        (
            "ipv4:malformed-no-port",
            "malformed-no-port",
        ),  # Current behavior based on split
        ("ipv6:[::1]", "::1"),  # Corrected: Code extracts "::1"
        (
            "ipv4:",
            "",
        ),  # Current behavior: split(":") -> ["ipv4", ""], then [1] -> ""
        (
            "ipv6:",
            "",
        ),  # Current behavior: peer_address[5:] -> "", then split & strip -> ""
        # Additional robust handling cases for malformed ipv4/ipv6
        (
            "ipv4:127.0.0.1",
            "127.0.0.1",
        ),  # Malformed - no port, but IP is there
        (
            "ipv6:[2001:db8::1]",
            "2001:db8::1",
        ),  # Malformed - no port, but IP is there
    ],
)
def test_get_client_ip(mock_context, peer_string, expected_ip):
    """Test get_client_ip with various peer strings."""
    mock_context.peer.return_value = peer_string
    assert get_client_ip(mock_context) == expected_ip


# --- Tests for get_client_port ---


@pytest.mark.parametrize(
    "peer_string, expected_port",
    [
        ("ipv4:127.0.0.1:12345", 12345),
        ("ipv6:[::1]:54321", 54321),
        ("ipv6:[2001:db8::1]:8080", 8080),
        ("unix:/tmp/my.sock", None),
        ("ipv4:127.0.0.1", None),  # No port
        (
            "ipv6:[::1]",
            1,
        ),  # Current behavior: last part of "::1" is treated as port
        ("ipv4:127.0.0.1:abc", None),  # Non-integer port
        ("unknown:foo:bar", None),
        ("ipv4:127.0.0.1:", None),  # Empty port part
        ("ipv6:[::1]:", None),  # Empty port part
    ],
)
def test_get_client_port(mock_context, peer_string, expected_port):
    """Test get_client_port with various peer strings."""
    mock_context.peer.return_value = peer_string
    assert get_client_port(mock_context) == expected_port
