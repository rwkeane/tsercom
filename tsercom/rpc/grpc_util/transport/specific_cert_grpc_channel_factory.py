import asyncio
import logging
from typing import TYPE_CHECKING, List, Union # For type hints

# Import grpc.aio for AioRpcError
import grpc.aio

from ...common.channel_info import ChannelInfo # Relative import from tsercom/rpc/common/
from ..grpc_channel_factory import GrpcChannelFactory # Relative import from tsercom/rpc/grpc_util/
from ..grpc_channel_credentials_provider import GrpcChannelCredentialsProvider # Relative import from tsercom/rpc/grpc_util/

if TYPE_CHECKING:
    # grpc.aio.Channel is available via grpc.aio import
    pass

class SpecificCertGrpcChannelFactory(GrpcChannelFactory):
    """
    A gRPC channel factory that creates secure channels using a specific
    server certificate as the trust anchor.
    """

    def __init__(
        self,
        server_cert_path: str,
        credentials_provider: GrpcChannelCredentialsProvider,
    ):
        """
        Initializes SpecificCertGrpcChannelFactory.

        Args:
            server_cert_path: Path to the server's certificate file. This certificate
                              will be used as the sole trust anchor.
            credentials_provider: The provider for gRPC channel credentials.
        """
        super().__init__()
        self._server_cert_path = server_cert_path
        self._credentials_provider = credentials_provider

    async def find_async_channel(
        self, addresses: Union[List[str], str], port: int
    ) -> ChannelInfo | None:
        """
        Finds an asynchronous gRPC channel to the specified address(es) and port
        using a specific server certificate for secure connection.

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
            f" using server cert: {self._server_cert_path} as trust anchor."
        )

        server_cert_bytes = self._credentials_provider.read_file_content(
            self._server_cert_path
        )
        if server_cert_bytes is None:
            logging.error(
                f"Failed to read server certificate from {self._server_cert_path}"
            )
            return None

        # For gRPC, when providing a specific server certificate as the trust anchor,
        # it's passed via the root_certificates parameter.
        ssl_credentials = self._credentials_provider.create_ssl_channel_credentials(
            root_certificates=server_cert_bytes
        )
        if ssl_credentials is None:
            logging.error(
                "Failed to create SSL channel credentials using the provided server certificate "
                "as trust anchor."
            )
            return None

        for current_address in address_list:
            target = f"{current_address}:{port}"
            logging.info(
                f"Attempting to connect to {target} with server cert {self._server_cert_path}..."
            )
            channel: grpc.aio.Channel | None = None
            try:
                channel = self._credentials_provider.create_secure_channel(
                    target, ssl_credentials
                )
                if channel:
                    await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
                    logging.info(
                        f"Successfully connected to {target} using server cert {self._server_cert_path}."
                    )
                    return ChannelInfo(channel, current_address, port)
                else:
                    logging.warning(f"Channel creation returned None for {target}")

            except grpc.aio.AioRpcError as e:
                logging.warning(
                    f"gRPC AioRpcError while trying to connect to {target} "
                    f"with server cert {self._server_cert_path}: {e.code()} - {e.details()}"
                )
                if channel:
                    await channel.close()
            except asyncio.TimeoutError:
                logging.warning(
                    f"Timeout waiting for channel to be ready for {target} "
                    f"with server cert {self._server_cert_path}."
                )
                if channel:
                    await channel.close()
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred while connecting to {target} "
                    f"with server cert {self._server_cert_path}: {e}"
                )
                if channel:
                    await channel.close()
        
        logging.warning(
            f"Failed to establish a secure connection to any of the addresses "
            f"{address_list} on port {port} using server cert {self._server_cert_path}"
        )
        return None
