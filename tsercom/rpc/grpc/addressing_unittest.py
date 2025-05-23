import sys
import pytest
from unittest.mock import MagicMock

# --- Start of GRPC mock ---
# This section must run before `tsercom.rpc.grpc.addressing` is imported.
# Create a mock for the grpc module and its sub-modules/classes.

# Mock for grpc.aio.ServicerContext
MockServicerContext = MagicMock()
MockServicerContext.__name__ = "ServicerContext"

# Mock for grpc.aio
mock_aio = MagicMock()
mock_aio.ServicerContext = MockServicerContext

# Mock for grpc module
mock_grpc_module = MagicMock()
mock_grpc_module.aio = mock_aio

# Inject the mock_grpc_module into sys.modules.
# This ensures that when 'tsercom.rpc.grpc.addressing' (or any other module)
# tries to 'import grpc' or 'from grpc import aio', it gets our mock.
sys.modules['grpc'] = mock_grpc_module
sys.modules['grpc.aio'] = mock_aio
# --- End of GRPC mock ---

# Now we can safely import the functions to be tested
# This import should now use the mocked 'grpc' module from sys.modules
from tsercom.rpc.grpc.addressing import get_client_ip, get_client_port


@pytest.fixture
def mock_context_instance() -> MagicMock:
    """
    Fixture to create a mock ServicerContext instance.
    This instance will be used as the 'context' argument in the functions under test.
    """
    context_instance = MagicMock()
    # The critical part is that context_instance.peer() must be mockable.
    # The functions get_client_ip/port call context.peer().
    # The type hint in addressing.py is "grpc.aio.ServicerContext",
    # which is now resolved to our MockServicerContext due to sys.modules patching.
    return context_instance

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
        # Additional tricky cases
        ("ipv4:127.0.0.1:8080", "127.0.0.1"),
        ("ipv6:[::ffff:192.0.2.128]:80", "::ffff:192.0.2.128"),
        ("ipv4::12345", None), 
        ("ipv6:::12345", None),
    ],
)
def test_get_client_ip(mock_context_instance: MagicMock, peer_string: str, expected_ip: str | None):
    mock_context_instance.peer.return_value = peer_string
    assert get_client_ip(mock_context_instance) == expected_ip

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
        # Additional tricky cases
        ("ipv6:[2001:db8::1]:65535", 65535),
        ("ipv4:127.0.0.1:", None),
        ("unix:/tmp/socket", None),
        ("ipv4:1.2.3.4:12345extra", None),
    ],
)
def test_get_client_port(mock_context_instance: MagicMock, peer_string: str, expected_port: int | None):
    mock_context_instance.peer.return_value = peer_string
    assert get_client_port(mock_context_instance) == expected_port
