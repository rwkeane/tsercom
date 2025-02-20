import socket
import psutil

def get_all_address_strings():
    addresses = []
    for _, interface_addresses in psutil.net_if_addrs().items():
        for address in interface_addresses:
            if address.family == socket.AF_INET:
                addresses.append(address.address)
    return addresses

def get_all_addresses():
    return [ socket.inet_aton(a) for a in get_all_address_strings() ]
