import pytest
from unittest.mock import patch, MagicMock, call
import socket # For AF_INET, AF_INET6 constants

# Functions to be tested are in tsercom.util.ip
# Ensure this path is correct based on your project structure.
# If ip.py is directly under util, then this is fine.
from tsercom.util import ip as ip_util 

# Helper to create a mock address object
def create_mock_address(family, address):
    mock_addr = MagicMock()
    mock_addr.family = family
    mock_addr.address = address
    return mock_addr

class TestIpUtils:

    # --- Tests for get_all_address_strings ---

    @patch('psutil.net_if_addrs')
    def test_get_all_address_strings_no_interfaces(self, mock_net_if_addrs):
        print("\n--- Test: test_get_all_address_strings_no_interfaces ---")
        mock_net_if_addrs.return_value = {} # No interfaces
        
        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")
        
        assert result == []
        mock_net_if_addrs.assert_called_once()
        print("--- Test: test_get_all_address_strings_no_interfaces finished ---")

    @patch('psutil.net_if_addrs')
    def test_get_all_address_strings_interface_no_ipv4(self, mock_net_if_addrs):
        print("\n--- Test: test_get_all_address_strings_interface_no_ipv4 ---")
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(socket.AF_INET6, "::1"),
                create_mock_address(socket.AF_PACKET, "00:11:22:33:44:55") # type: ignore
            ]
        }
        
        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")
        
        assert result == []
        mock_net_if_addrs.assert_called_once()
        print("--- Test: test_get_all_address_strings_interface_no_ipv4 finished ---")

    @patch('psutil.net_if_addrs')
    def test_get_all_address_strings_single_interface_single_ipv4(self, mock_net_if_addrs):
        print("\n--- Test: test_get_all_address_strings_single_interface_single_ipv4 ---")
        ipv4_address = "192.168.1.100"
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(socket.AF_INET, ipv4_address),
                create_mock_address(socket.AF_INET6, "fe80::1")
            ]
        }
        
        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")
        
        assert result == [ipv4_address]
        mock_net_if_addrs.assert_called_once()
        print("--- Test: test_get_all_address_strings_single_interface_single_ipv4 finished ---")

    @patch('psutil.net_if_addrs')
    def test_get_all_address_strings_single_interface_multiple_ipv4(self, mock_net_if_addrs):
        print("\n--- Test: test_get_all_address_strings_single_interface_multiple_ipv4 ---")
        ipv4_address1 = "192.168.1.101"
        ipv4_address2 = "192.168.1.102"
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(socket.AF_INET, ipv4_address1),
                create_mock_address(socket.AF_INET6, "fe80::2"),
                create_mock_address(socket.AF_INET, ipv4_address2)
            ]
        }
        
        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")
        
        assert sorted(result) == sorted([ipv4_address1, ipv4_address2]) # Order might vary
        mock_net_if_addrs.assert_called_once()
        print("--- Test: test_get_all_address_strings_single_interface_multiple_ipv4 finished ---")

    @patch('psutil.net_if_addrs')
    def test_get_all_address_strings_multiple_interfaces_varied_ipv4(self, mock_net_if_addrs):
        print("\n--- Test: test_get_all_address_strings_multiple_interfaces_varied_ipv4 ---")
        ipv4_eth0_1 = "192.168.1.103"
        ipv4_eth1_1 = "10.0.0.5"
        ipv4_eth1_2 = "10.0.0.6"
        
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(socket.AF_INET, ipv4_eth0_1),
                create_mock_address(socket.AF_INET6, "fe80::3")
            ],
            "lo": [
                create_mock_address(socket.AF_INET6, "::1")
            ],
            "eth1": [
                create_mock_address(socket.AF_INET6, "fe80::4"),
                create_mock_address(socket.AF_INET, ipv4_eth1_1),
                create_mock_address(socket.AF_INET, ipv4_eth1_2)
            ],
            "docker0": [ # Interface with no relevant addresses
                create_mock_address(socket.AF_PACKET, "02:42:ac:11:00:02") # type: ignore
            ]
        }
        
        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")
        
        expected_ips = [ipv4_eth0_1, ipv4_eth1_1, ipv4_eth1_2]
        assert sorted(result) == sorted(expected_ips)
        mock_net_if_addrs.assert_called_once()
        print("--- Test: test_get_all_address_strings_multiple_interfaces_varied_ipv4 finished ---")

    @patch('psutil.net_if_addrs')
    def test_get_all_address_strings_loopback_and_regular_ipv4(self, mock_net_if_addrs):
        print("\n--- Test: test_get_all_address_strings_loopback_and_regular_ipv4 ---")
        loopback_ipv4 = "127.0.0.1"
        regular_ipv4 = "172.16.0.10"
        
        mock_net_if_addrs.return_value = {
            "lo": [
                create_mock_address(socket.AF_INET, loopback_ipv4),
                create_mock_address(socket.AF_INET6, "::1")
            ],
            "eth0": [
                create_mock_address(socket.AF_INET, regular_ipv4),
                create_mock_address(socket.AF_INET6, "2001:db8::123")
            ]
        }
        
        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")
        
        expected_ips = [loopback_ipv4, regular_ipv4]
        assert sorted(result) == sorted(expected_ips)
        mock_net_if_addrs.assert_called_once()
        print("--- Test: test_get_all_address_strings_loopback_and_regular_ipv4 finished ---")

    # --- Tests for get_all_addresses ---

    @patch('socket.inet_aton') # Mock socket.inet_aton
    @patch('tsercom.util.ip.get_all_address_strings') # Mock the helper function
    def test_get_all_addresses_no_ip_strings(self, mock_get_strings, mock_inet_aton):
        print("\n--- Test: test_get_all_addresses_no_ip_strings ---")
        mock_get_strings.return_value = [] # No IP strings
        
        result = ip_util.get_all_addresses()
        print(f"  Result: {result}")
        
        assert result == []
        mock_get_strings.assert_called_once()
        mock_inet_aton.assert_not_called() # inet_aton should not be called if no strings
        print("--- Test: test_get_all_addresses_no_ip_strings finished ---")

    @patch('socket.inet_aton') # Mock socket.inet_aton
    @patch('tsercom.util.ip.get_all_address_strings') # Mock the helper function
    def test_get_all_addresses_with_ip_strings(self, mock_get_strings, mock_inet_aton):
        print("\n--- Test: test_get_all_addresses_with_ip_strings ---")
        ip_strings = ["192.168.1.1", "10.0.0.1", "127.0.0.1"]
        mock_get_strings.return_value = ip_strings
        
        # Define a side effect for mock_inet_aton to simulate packing
        def inet_aton_side_effect(ip_str):
            print(f"  mock_inet_aton called with: {ip_str}")
            return f"packed_{ip_str}".encode('utf-8')
        mock_inet_aton.side_effect = inet_aton_side_effect
        
        result = ip_util.get_all_addresses()
        print(f"  Result: {result}")
        
        expected_packed_results = [f"packed_{s}".encode('utf-8') for s in ip_strings]
        assert result == expected_packed_results
        
        mock_get_strings.assert_called_once()
        # Check that inet_aton was called for each IP string
        expected_calls = [call(s) for s in ip_strings]
        mock_inet_aton.assert_has_calls(expected_calls, any_order=False) # Order matters here
        assert mock_inet_aton.call_count == len(ip_strings)
        print("--- Test: test_get_all_addresses_with_ip_strings finished ---")

```

**Summary of Implementation:**
1.  **Imports**: `pytest`, `unittest.mock` utilities, and `socket` are imported. `tsercom.util.ip` is imported as `ip_util`.
2.  **`create_mock_address` Helper**: A small helper function to create `MagicMock` objects that simulate network address structures returned by `psutil.net_if_addrs`.
3.  **`TestIpUtils` Class**:
    *   Contains test methods for `get_all_address_strings` and `get_all_addresses`.
    *   **`get_all_address_strings` Tests**:
        *   Uses `@patch('psutil.net_if_addrs')` to mock the external dependency.
        *   `test_no_interfaces`: Mocks `psutil.net_if_addrs` to return `{}`. Asserts empty list.
        *   `test_interface_no_ipv4`: Mocks interfaces with only IPv6 or other family addresses. Asserts empty list.
        *   `test_single_interface_single_ipv4`: Mocks one interface with one IPv4. Asserts list with that IP.
        *   `test_single_interface_multiple_ipv4`: Mocks one interface with multiple IPv4s. Asserts list with all those IPs.
        *   `test_multiple_interfaces_varied_ipv4`: Mocks several interfaces with a mix of IPv4 and non-IPv4 addresses. Asserts all and only IPv4s are returned.
        *   `test_loopback_and_regular_ipv4`: Specifically tests inclusion of "127.0.0.1" and a typical non-loopback IPv4.
    *   **`get_all_addresses` Tests**:
        *   Uses nested `@patch` decorators for `socket.inet_aton` and `tsercom.util.ip.get_all_address_strings`.
        *   `test_no_ip_strings`: Mocks `get_all_address_strings` to return `[]`. Asserts `get_all_addresses` returns `[]` and `socket.inet_aton` is not called.
        *   `test_with_ip_strings`: Mocks `get_all_address_strings` to return a list of IP strings. `socket.inet_aton` is mocked with a `side_effect` to simulate IP packing. Asserts `inet_aton` is called for each string and the results are correctly aggregated.

The tests cover the scenarios outlined in the task description. Print statements are included for diagnostic purposes during the run.
The tests for `get_all_address_strings` use `sorted()` for list comparison where the order of returned IPs from multiple interfaces/addresses might not be guaranteed by `psutil.net_if_addrs`. For `get_all_addresses`, the order of packed results should match the order of IP strings returned by the (mocked) `get_all_address_strings`.The test file `tsercom/util/ip_unittest.py` has been written with tests for `get_all_address_strings` and `get_all_addresses`.

**For `get_all_address_strings()`:**
- Mocked `psutil.net_if_addrs` for various scenarios:
    - No network interfaces.
    - Interfaces with no IPv4 addresses.
    - Single interface with single/multiple IPv4 addresses.
    - Multiple interfaces with a mix of IPv4 and other address types.
    - Inclusion of loopback ("127.0.0.1") and regular IPv4 addresses.
- Assertions check that the function correctly filters and returns only IPv4 address strings.

**For `get_all_addresses()`:**
- Mocked `tsercom.util.ip.get_all_address_strings` to control the input IP strings.
- Mocked `socket.inet_aton` to simulate IP packing and verify calls.
- Test scenarios:
    - No IP strings provided by `get_all_address_strings`.
    - Multiple IP strings provided, ensuring `socket.inet_aton` is called for each and results are aggregated correctly.

The tests use `unittest.mock.patch` for mocking and `pytest` as the test framework. Helper function `create_mock_address` is used for readability.

I will now run these tests. The print statements included in the tests will provide real-time feedback during execution.
