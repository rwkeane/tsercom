import abc
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import grpc
    import grpc.aio

class GrpcChannelCredentialsProvider(abc.ABC):
    @abc.abstractmethod
    def read_file_content(self, path: str) -> bytes | None:
        pass

    @abc.abstractmethod
    def create_ssl_channel_credentials(
        self,
        root_certificates: bytes | None = None,
        private_key: bytes | None = None,
        certificate_chain: bytes | None = None,
    ) -> 'grpc.ChannelCredentials | None':
        pass

    @abc.abstractmethod
    def create_secure_channel(
        self,
        target: str,
        credentials: 'grpc.ChannelCredentials | None',
        options: list[tuple[str, Any]] | None = None,
    ) -> 'grpc.aio.Channel':
        pass

    @abc.abstractmethod
    def create_insecure_channel(
        self, target: str, options: list[tuple[str, Any]] | None = None
    ) -> 'grpc.aio.Channel':
        pass

class DefaultGrpcChannelCredentialsProvider(GrpcChannelCredentialsProvider):
    def read_file_content(self, path: str) -> bytes | None:
        try:
            with open(path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
            return None

    def create_ssl_channel_credentials(
        self,
        root_certificates: bytes | None = None,
        private_key: bytes | None = None,
        certificate_chain: bytes | None = None,
    ) -> 'grpc.ChannelCredentials | None':
        # Ensure grpc is imported for runtime use
        import grpc
        return grpc.ssl_channel_credentials(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )

    def create_secure_channel(
        self,
        target: str,
        credentials: 'grpc.ChannelCredentials | None',
        options: list[tuple[str, Any]] | None = None,
    ) -> 'grpc.aio.Channel':
        # Ensure grpc.aio is imported for runtime use
        import grpc.aio
        return grpc.aio.secure_channel(target, credentials, options=options)

    def create_insecure_channel(
        self, target: str, options: list[tuple[str, Any]] | None = None
    ) -> 'grpc.aio.Channel':
        # Ensure grpc.aio is imported for runtime use
        import grpc.aio
        return grpc.aio.insecure_channel(target, options=options)
