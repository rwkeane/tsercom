import grpc

def get_client_ip(context: grpc.aio.ServicerContext) -> str | None:
    peer_address = context.peer()

    # Extract the IP address based on the format
    if peer_address.startswith("ipv4:"):
        return peer_address.split(":")[1] # type: ignore
    elif peer_address.startswith("ipv6:"):
        return peer_address[5:].split("]")[0]  # type: ignore
    elif peer_address.startswith("unix:"):
        return "localhost"  # Or handle Unix socket addresses as needed
    else:
        return None  # Unknown format
