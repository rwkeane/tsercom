import asyncio
import logging
from typing import TYPE_CHECKING, List, Union # For type hints

# Import grpc.aio for AioRpcError
import grpc.aio

from ...common.channel_info import ChannelInfo # Relative: transport -> grpc_util -> rpc -> common
from ..grpc_channel_factory import GrpcChannelFactory # Relative: transport -> grpc_util -> grpc_channel_factory
from ..grpc_channel_credentials_provider import GrpcChannelCredentialsProvider # Relative: transport -> grpc_util -> grpc_channel_credentials_provider

if TYPE_CHECKING:
    # grpc.aio.Channel is available via grpc.aio import
    pass

class ClientCertGrpcChannelFactory(GrpcChannelFactory):
    """
    A gRPC channel factory that creates secure channels using a client certificate
    for authentication, without validating the server's certificate.
    """

    def __init__(
        self,
        client_key_path: str,
        client_cert_chain_path: str,
        credentials_provider: GrpcChannelCredentialsProvider,
    ):
        """
        Initializes ClientCertGrpcChannelFactory.

        Args:
            client_key_path: Path to the client's private key file.
            client_cert_chain_path: Path to the client's certificate chain file.
            credentials_provider: The provider for gRPC channel credentials.
        """
        super().__init__() # Call super for ABC initialization
        self._client_key_path = client_key_path
        self._client_cert_chain_path = client_cert_chain_path
        self._credentials_provider = credentials_provider

    async def find_async_channel(
        self, addresses: Union[List[str], str], port: int
    ) -> ChannelInfo | None:
        """
        Finds an asynchronous gRPC channel to the specified address(es) and port
        using client certificate for authentication. Server certificate is NOT validated.

        Args:
            addresses: A single address string or a list of address strings to try.
            port: The port number to connect to.

        Returns:
            A ChannelInfo object if a channel is successfully established,
            otherwise None.
        """
        address_list: List[str] = []
        if isinstance(addresses, str):
            address_list.append(addresses)
        else:
            address_list = addresses

        logging.info(
            f"Attempting to find secure channel to addresses {address_list} on port {port} "
            f"using client key: {self._client_key_path} and cert: {self._client_cert_chain_path}. "
            "Server certificate will NOT be validated."
        )

        client_key_bytes = self._credentials_provider.read_file_content(
            self._client_key_path
        )
        if client_key_bytes is None:
            logging.error(
                f"Failed to read client private key from {self._client_key_path}"
            )
            return None

        client_cert_bytes = self._credentials_provider.read_file_content(
            self._client_cert_chain_path
        )
        if client_cert_bytes is None:
            logging.error(
                f"Failed to read client certificate chain from {self._client_cert_chain_path}"
            )
            return None

        # root_certificates=None is crucial for not validating the server certificate
        ssl_credentials = self._credentials_provider.create_ssl_channel_credentials(
            root_certificates=None, # Do not validate server certificate
            private_key=client_key_bytes,
            certificate_chain=client_cert_bytes,
        )
        if ssl_credentials is None:
            logging.error(
                "Failed to create SSL channel credentials using client key/cert "
                "(and no server cert validation)."
            )
            return None

        for current_address in address_list:
            target = f"{current_address}:{port}"
            logging.info(
                f"Attempting to connect to {target} with client cert (no server validation)..."
            )
            channel: grpc.aio.Channel | None = None
            try:
                channel = self._credentials_provider.create_secure_channel(
                    target, ssl_credentials
                )
                if channel:
                    await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
                    logging.info(
                        f"Successfully connected to {target} using client cert "
                        "(no server validation)."
                    )
                    return ChannelInfo(channel, current_address, port)
                else:
                    logging.warning(f"Channel creation returned None for {target}")

            except grpc.aio.AioRpcError as e:
                logging.warning(
                    f"gRPC AioRpcError while trying to connect to {target} "
                    f"with client cert (no server validation): {e.code()} - {e.details()}"
                )
                if channel:
                    await channel.close()
            except asyncio.TimeoutError:
                logging.warning(
                    f"Timeout waiting for channel to be ready for {target} "
                    f"with client cert (no server validation)."
                )
                if channel:
                    await channel.close()
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred while connecting to {target} "
                    f"with client cert (no server validation): {e}"
                )
                if channel:
                    await channel.close()
        
        logging.warning(
            f"Failed to establish a secure connection to any of the addresses "
            f"{address_list} on port {port} using client key {self._client_key_path} "
            f"and cert {self._client_cert_chain_path} (no server validation)."
        )
        return None
