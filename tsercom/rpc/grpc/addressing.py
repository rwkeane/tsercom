def get_client_ip(context):
    peer_address = context.peer()

    # Extract the IP address based on the format
    if peer_address.startswith('ipv4:'):
        return peer_address.split(':')[1]
    elif peer_address.startswith('ipv6:'):
        return peer_address[5:].split(']')[0]  # Remove 'ipv6:[' and ']:port'
    elif peer_address.startswith('unix:'):
        return "localhost"  # Or handle Unix socket addresses as needed
    else:
        return None  # Unknown format
