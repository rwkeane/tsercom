import asyncio
import logging
from typing import TYPE_CHECKING, List, Union # Union for type hint

# Import grpc.aio for AioRpcError
import grpc.aio

from ...common.channel_info import ChannelInfo # Corrected relative import
from ..grpc_channel_factory import GrpcChannelFactory
from ..grpc_channel_credentials_provider import GrpcChannelCredentialsProvider

if TYPE_CHECKING:
    # grpc.aio.Channel is already implicitly available via grpc.aio import
    # but keeping explicit if other grpc types were needed for hints
    pass

class LocalCaGrpcChannelFactory(GrpcChannelFactory):
    """
    A gRPC channel factory that creates secure channels using a local CA certificate.
    """

    def __init__(
        self,
        root_ca_cert_path: str,
        credentials_provider: GrpcChannelCredentialsProvider,
    ):
        """
        Initializes LocalCaGrpcChannelFactory.

        Args:
            root_ca_cert_path: Path to the root CA certificate file.
            credentials_provider: The provider for gRPC channel credentials.
        """
        super().__init__()
        self._root_ca_cert_path = root_ca_cert_path
        self._credentials_provider = credentials_provider

    async def find_async_channel(
        self, addresses: Union[List[str], str], port: int
    ) -> ChannelInfo | None:
        """
        Finds an asynchronous gRPC channel to the specified address(es) and port
        using a local CA certificate for secure connection.

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
            f"Attempting to find secure channel to addresses {address_list} on port {port}"
            f" using CA cert: {self._root_ca_cert_path}"
        )

        ca_cert_bytes = self._credentials_provider.read_file_content(
            self._root_ca_cert_path
        )
        if ca_cert_bytes is None:
            logging.error(
                f"Failed to read CA certificate from {self._root_ca_cert_path}"
            )
            return None

        ssl_credentials = self._credentials_provider.create_ssl_channel_credentials(
            root_certificates=ca_cert_bytes
        )
        if ssl_credentials is None:
            logging.error(
                "Failed to create SSL channel credentials using the provided CA certificate."
            )
            return None

        for current_address in address_list:
            target = f"{current_address}:{port}"
            logging.info(f"Attempting to connect to {target} with local CA cert...")
            channel: grpc.aio.Channel | None = None
            try:
                channel = self._credentials_provider.create_secure_channel(
                    target, ssl_credentials
                )
                if channel:
                    # Wait for the channel to be ready
                    # Default timeout for channel_ready is None (waits indefinitely)
                    # Adding a specific timeout
                    await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
                    logging.info(
                        f"Successfully connected to {target} using local CA cert."
                    )
                    return ChannelInfo(channel, current_address, port)
                else:
                    logging.warning(f"Channel creation returned None for {target}")

            except grpc.aio.AioRpcError as e:
                logging.warning(
                    f"gRPC AioRpcError while trying to connect to {target} "
                    f"with local CA: {e.code()} - {e.details()}"
                )
                if channel: # Close channel if created but failed to become ready
                    await channel.close()
            except asyncio.TimeoutError:
                logging.warning(
                    f"Timeout waiting for channel to be ready for {target} "
                    f"with local CA."
                )
                if channel: # Close channel if created but timed out
                    await channel.close()
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred while connecting to {target} "
                    f"with local CA: {e}"
                )
                if channel: # Close channel on other errors
                    await channel.close()
        
        logging.warning(
            f"Failed to establish a secure connection to any of the addresses "
            f"{address_list} on port {port} using CA {self._root_ca_cert_path}"
        )
        return None
