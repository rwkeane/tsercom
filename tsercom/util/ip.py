"""Utilities for network IP addresses."""

import socket

import psutil  # type: ignore[import-untyped]


def get_all_address_strings() -> list[str]:
    """Retrieves all IPv4 address strings for all network interfaces.

    This function iterates through all network interfaces on the system,
    collects all assigned IPv4 addresses, and returns them as a list of strings.

    Returns:
        A list of IPv4 address strings. Empty if no IPv4 addresses found.
    """
    addresses: list[str] = []
    for _, interface_addresses in psutil.net_if_addrs().items():
        for address in interface_addresses:
            if address.family == socket.AF_INET:
                addresses.append(address.address)
    return addresses


def get_all_addresses() -> list[bytes]:
    """Retrieves all IPv4 addresses for all network interfaces, as bytes.

    This function first calls `get_all_address_strings()` to get the string
    representation of all IPv4 addresses and then converts each address
    string into its packed 32-bit binary format (bytes).

    Returns:
        A list of bytes, where each bytes object is an IPv4 address
        packed in network byte order. Returns an empty list if no
        IPv4 addresses are found.
    """
    return [socket.inet_aton(a) for a in get_all_address_strings()]
