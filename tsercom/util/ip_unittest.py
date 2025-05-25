import pytest

# from unittest.mock import patch, MagicMock, call # Removed
import socket  # For AF_INET, AF_INET6 constants

# Functions to be tested are in tsercom.util.ip
# Ensure this path is correct based on your project structure.
# If ip.py is directly under util, then this is fine.
from tsercom.util import ip as ip_util


# Helper to create a mock address object
def create_mock_address(mocker, family, address):  # Added mocker
    mock_addr = mocker.MagicMock()  # Changed to mocker.MagicMock
    mock_addr.family = family
    mock_addr.address = address
    return mock_addr


class TestIpUtils:

    # --- Tests for get_all_address_strings ---

    def test_get_all_address_strings_no_interfaces(
        self, mocker
    ):  # Added mocker
        print("\n--- Test: test_get_all_address_strings_no_interfaces ---")
        mock_net_if_addrs = mocker.patch(
            "psutil.net_if_addrs"
        )  # Changed to mocker.patch
        mock_net_if_addrs.return_value = {}

        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")

        assert result == []
        mock_net_if_addrs.assert_called_once()
        print(
            "--- Test: test_get_all_address_strings_no_interfaces finished ---"
        )

    def test_get_all_address_strings_interface_no_ipv4(
        self, mocker
    ):  # Added mocker
        print("\n--- Test: test_get_all_address_strings_interface_no_ipv4 ---")
        mock_net_if_addrs = mocker.patch(
            "psutil.net_if_addrs"
        )  # Changed to mocker.patch
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(
                    mocker, socket.AF_INET6, "::1"
                ),  # Passed mocker
                create_mock_address(mocker, socket.AF_PACKET, "00:11:22:33:44:55"),  # type: ignore # Passed mocker
            ]
        }

        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")

        assert result == []
        mock_net_if_addrs.assert_called_once()
        print(
            "--- Test: test_get_all_address_strings_interface_no_ipv4 finished ---"
        )

    def test_get_all_address_strings_single_interface_single_ipv4(
        self, mocker
    ):  # Added mocker
        print(
            "\n--- Test: test_get_all_address_strings_single_interface_single_ipv4 ---"
        )
        mock_net_if_addrs = mocker.patch(
            "psutil.net_if_addrs"
        )  # Changed to mocker.patch
        ipv4_address = "192.168.1.100"
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(
                    mocker, socket.AF_INET, ipv4_address
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET6, "fe80::1"
                ),  # Passed mocker
            ]
        }

        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")

        assert result == [ipv4_address]
        mock_net_if_addrs.assert_called_once()
        print(
            "--- Test: test_get_all_address_strings_single_interface_single_ipv4 finished ---"
        )

    def test_get_all_address_strings_single_interface_multiple_ipv4(
        self, mocker
    ):  # Added mocker
        print(
            "\n--- Test: test_get_all_address_strings_single_interface_multiple_ipv4 ---"
        )
        mock_net_if_addrs = mocker.patch(
            "psutil.net_if_addrs"
        )  # Changed to mocker.patch
        ipv4_address1 = "192.168.1.101"
        ipv4_address2 = "192.168.1.102"
        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(
                    mocker, socket.AF_INET, ipv4_address1
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET6, "fe80::2"
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET, ipv4_address2
                ),  # Passed mocker
            ]
        }

        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")

        assert sorted(result) == sorted([ipv4_address1, ipv4_address2])
        mock_net_if_addrs.assert_called_once()
        print(
            "--- Test: test_get_all_address_strings_single_interface_multiple_ipv4 finished ---"
        )

    def test_get_all_address_strings_multiple_interfaces_varied_ipv4(
        self, mocker
    ):  # Added mocker
        print(
            "\n--- Test: test_get_all_address_strings_multiple_interfaces_varied_ipv4 ---"
        )
        mock_net_if_addrs = mocker.patch(
            "psutil.net_if_addrs"
        )  # Changed to mocker.patch
        ipv4_eth0_1 = "192.168.1.103"
        ipv4_eth1_1 = "10.0.0.5"
        ipv4_eth1_2 = "10.0.0.6"

        mock_net_if_addrs.return_value = {
            "eth0": [
                create_mock_address(
                    mocker, socket.AF_INET, ipv4_eth0_1
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET6, "fe80::3"
                ),  # Passed mocker
            ],
            "lo": [
                create_mock_address(
                    mocker, socket.AF_INET6, "::1"
                )  # Passed mocker
            ],
            "eth1": [
                create_mock_address(
                    mocker, socket.AF_INET6, "fe80::4"
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET, ipv4_eth1_1
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET, ipv4_eth1_2
                ),  # Passed mocker
            ],
            "docker0": [
                create_mock_address(mocker, socket.AF_PACKET, "02:42:ac:11:00:02")  # type: ignore # Passed mocker
            ],
        }

        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")

        expected_ips = [ipv4_eth0_1, ipv4_eth1_1, ipv4_eth1_2]
        assert sorted(result) == sorted(expected_ips)
        mock_net_if_addrs.assert_called_once()
        print(
            "--- Test: test_get_all_address_strings_multiple_interfaces_varied_ipv4 finished ---"
        )

    def test_get_all_address_strings_loopback_and_regular_ipv4(
        self, mocker
    ):  # Added mocker
        print(
            "\n--- Test: test_get_all_address_strings_loopback_and_regular_ipv4 ---"
        )
        mock_net_if_addrs = mocker.patch(
            "psutil.net_if_addrs"
        )  # Changed to mocker.patch
        loopback_ipv4 = "127.0.0.1"
        regular_ipv4 = "172.16.0.10"

        mock_net_if_addrs.return_value = {
            "lo": [
                create_mock_address(
                    mocker, socket.AF_INET, loopback_ipv4
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET6, "::1"
                ),  # Passed mocker
            ],
            "eth0": [
                create_mock_address(
                    mocker, socket.AF_INET, regular_ipv4
                ),  # Passed mocker
                create_mock_address(
                    mocker, socket.AF_INET6, "2001:db8::123"
                ),  # Passed mocker
            ],
        }

        result = ip_util.get_all_address_strings()
        print(f"  Result: {result}")

        expected_ips = [loopback_ipv4, regular_ipv4]
        assert sorted(result) == sorted(expected_ips)
        mock_net_if_addrs.assert_called_once()
        print(
            "--- Test: test_get_all_address_strings_loopback_and_regular_ipv4 finished ---"
        )

    # --- Tests for get_all_addresses ---

    def test_get_all_addresses_no_ip_strings(self, mocker):  # Added mocker
        print("\n--- Test: test_get_all_addresses_no_ip_strings ---")
        mock_get_strings = mocker.patch(
            "tsercom.util.ip.get_all_address_strings"
        )  # Changed to mocker.patch
        mock_inet_aton = mocker.patch(
            "socket.inet_aton"
        )  # Changed to mocker.patch
        mock_get_strings.return_value = []

        result = ip_util.get_all_addresses()
        print(f"  Result: {result}")

        assert result == []
        mock_get_strings.assert_called_once()
        mock_inet_aton.assert_not_called()
        print("--- Test: test_get_all_addresses_no_ip_strings finished ---")

    def test_get_all_addresses_with_ip_strings(self, mocker):  # Added mocker
        print("\n--- Test: test_get_all_addresses_with_ip_strings ---")
        mock_get_strings = mocker.patch(
            "tsercom.util.ip.get_all_address_strings"
        )  # Changed to mocker.patch
        mock_inet_aton = mocker.patch(
            "socket.inet_aton"
        )  # Changed to mocker.patch
        ip_strings = ["192.168.1.1", "10.0.0.1", "127.0.0.1"]
        mock_get_strings.return_value = ip_strings

        def inet_aton_side_effect(ip_str):
            print(f"  mock_inet_aton called with: {ip_str}")
            return f"packed_{ip_str}".encode("utf-8")

        mock_inet_aton.side_effect = inet_aton_side_effect

        result = ip_util.get_all_addresses()
        print(f"  Result: {result}")

        expected_packed_results = [
            f"packed_{s}".encode("utf-8") for s in ip_strings
        ]
        assert result == expected_packed_results

        mock_get_strings.assert_called_once()
        expected_calls = [
            mocker.call(s) for s in ip_strings
        ]  # Changed to mocker.call
        mock_inet_aton.assert_has_calls(expected_calls, any_order=False)
        assert mock_inet_aton.call_count == len(ip_strings)
        print("--- Test: test_get_all_addresses_with_ip_strings finished ---")
