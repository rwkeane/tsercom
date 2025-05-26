"""Utility functions for extracting client IP and port from gRPC context."""

import grpc


def get_client_ip(context: "grpc.aio.ServicerContext") -> str | None:
    """Extracts the client IP address from a gRPC servicer context.

    Parses the peer string to identify IPv4, IPv6, or Unix domain socket addresses.

    Args:
        context: The gRPC servicer context.

    Returns:
        The client's IP address as a string, or 'localhost' for Unix sockets.
        Returns None if the address format is unknown.
    """
    peer_address = context.peer()

    if peer_address.startswith("ipv4:"):
        return peer_address.split(":")[1]  # type: ignore
    elif peer_address.startswith("ipv6:"):
        # For ipv6, format is like "ipv6:[::1]:12345"
        return peer_address[5:].split("]")[0].strip("[")  # type: ignore
    elif peer_address.startswith("unix:"):
        return "localhost"  # Or handle Unix socket addresses as needed
    else:
        return None  # Unknown format


def get_client_port(context: "grpc.aio.ServicerContext") -> int | None:
    """Extracts the client port number from a gRPC servicer context.

    Parses the peer string to find the port.

    Args:
        context: The gRPC servicer context.

    Returns:
        The client's port number as an integer, or None if it cannot be parsed.
    """
    peer_address = context.peer()

    try:
        # For ipv4 like "ipv4:127.0.0.1:12345" or ipv6 like "ipv6:[::1]:12345"
        # The port is the last part after the colon.
        # For unix sockets like "unix:/tmp/socket", this will fail, which is fine.
        result = int(peer_address.split(":")[-1].strip("]"))
        return result
    except (ValueError, IndexError):
        return None
