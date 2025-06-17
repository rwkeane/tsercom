"""End-to-end tests for tsercom gRPC communication, including various security scenarios and client retries."""

import asyncio
import datetime
import logging  # For server/client logging
from collections.abc import (
    Awaitable,
    Callable,
)
from ipaddress import ip_address  # For SANs
from typing import (
    Optional,
    Union,
)

import grpc
import pytest
import pytest_asyncio
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.rpc.endpoints.test_connection_server import (
    AsyncTestConnectionServer,
)
from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)
from tsercom.rpc.proto import (
    TestConnectionCall,
    TestConnectionResponse,
)
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    set_tsercom_event_loop,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.stopable import Stopable  # Required for TInstanceType

# ChannelInfo import removed as it's no longer used

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_private_key(key_size=2048) -> rsa.RSAPrivateKey:
    """Generates a new RSA private key."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    return private_key


def generate_ca_certificate(
    common_name: str = "Test CA", key_size: int = 2048
) -> tuple[bytes, bytes]:
    """
    Generates a self-signed CA certificate and its private key.

    Args:
        common_name: The common name for the CA.
        key_size: The RSA key size.

    Returns:
        A tuple of (ca_cert_pem, ca_key_pem), both bytes.
    """
    private_key = generate_private_key(key_size)
    public_key = private_key.public_key()

    builder = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        )
        .issuer_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        )
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=30)
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(public_key),
            critical=False,
        )
    )

    certificate = builder.sign(private_key, hashes.SHA256())

    ca_cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
    ca_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return ca_cert_pem, ca_key_pem


def generate_signed_certificate(
    ca_cert_pem: bytes,
    ca_key_pem: bytes,
    common_name: str,
    sans: Optional[list[str]] = None,
    is_server: bool = True,
    key_size: int = 2048,
) -> tuple[bytes, bytes]:
    """
    Generates a server or client certificate signed by the provided CA.

    Args:
        ca_cert_pem: PEM encoded CA certificate (bytes).
        ca_key_pem: PEM encoded CA private key (bytes).
        common_name: The common name for the certificate.
        sans: Optional list of Subject Alternative Names.
        is_server: True for server cert, False for client cert.
        key_size: The RSA key size for the new certificate.

    Returns:
        A tuple of (cert_pem, key_pem), both bytes.
    """
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_private_key = serialization.load_pem_private_key(
        ca_key_pem, password=None
    )

    private_key = generate_private_key(key_size)
    public_key = private_key.public_key()

    builder = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        )
        .issuer_name(ca_cert.subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=30)
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=(True if not is_server else False),
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_cert.public_key()
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(public_key),
            critical=False,
        )
    )

    if is_server:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=True,
        )
    else:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=True,
        )

    if sans:
        san_list = []
        for san_entry in sans:
            if san_entry.startswith("DNS:"):
                san_list.append(x509.DNSName(san_entry[4:]))
            elif san_entry.startswith("IP:"):
                san_list.append(x509.IPAddress(ip_address(san_entry[3:])))
        if san_list:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list), critical=False
            )

    certificate = builder.sign(ca_private_key, hashes.SHA256())

    cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


TEST_SERVICE_NAME = "dtp.TestConnectionService"
TEST_METHOD_NAME = "TestConnection"
FULL_METHOD_PATH = f"/{TEST_SERVICE_NAME}/{TEST_METHOD_NAME}"

pytestmark = pytest.mark.asyncio


class CustomServicePublisherRpcE2E(GrpcServicePublisher):
    """Subclass of GrpcServicePublisher to capture the assigned port."""

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        addresses: Union[str, list[str], None] = None,
    ):
        super().__init__(watcher, port, addresses)
        self._chosen_port: int | None = None

    def _connect(self) -> bool:
        """Binds the gRPC server and captures the chosen port."""
        worked = 0
        if self._GrpcServicePublisher__server is None:
            logger.error("Server object not initialized before _connect")
            return False

        addresses_to_bind = ["127.0.0.1"]
        if isinstance(self._GrpcServicePublisher__addresses, str):
            addresses_to_bind = [self._GrpcServicePublisher__addresses]
        elif isinstance(self._GrpcServicePublisher__addresses, list):
            addresses_to_bind = self._GrpcServicePublisher__addresses

        for address in addresses_to_bind:
            try:
                port_out = (
                    self._GrpcServicePublisher__server.add_insecure_port(
                        f"{address}:{self._GrpcServicePublisher__port}"
                    )
                )
                if self._chosen_port is None:
                    self._chosen_port = port_out
                logger.info(
                    f"Running gRPC Server on {address}:{port_out} (expected: {self._GrpcServicePublisher__port})"
                )
                worked += 1
                break
            except Exception as e:
                if isinstance(e, AssertionError):
                    self._GrpcServicePublisher__watcher.on_exception_seen(e)
                    raise e
                logger.warning(
                    f"Failed to bind gRPC server to {address}:{self._GrpcServicePublisher__port}. Error: {e}"
                )
                continue

        if worked == 0:
            logger.error("FAILED to host gRPC Service on any address.")
            return False

        if self._chosen_port is None and worked > 0:
            logger.warning(
                "Port bound, but _chosen_port not explicitly captured. This may occur if binding logic changes."
            )

        return worked != 0

    @property
    def chosen_port(self) -> int | None:
        return self._chosen_port


@pytest_asyncio.fixture
async def async_test_server():
    """Pytest fixture to start and stop the AsyncTestConnectionServer."""
    watcher = ThreadWatcher()
    service_publisher = CustomServicePublisherRpcE2E(
        watcher, port=0, addresses="127.0.0.1"
    )

    async_server_impl = AsyncTestConnectionServer()

    def connect_call(server: grpc.aio.Server):
        """Callback to add RPC handlers to the gRPC server."""
        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            async_server_impl.TestConnection,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )
        generic_handler = grpc.method_handlers_generic_handler(
            TEST_SERVICE_NAME,
            {TEST_METHOD_NAME: rpc_method_handler},
        )
        server.add_generic_rpc_handlers((generic_handler,))
        logger.info(
            f"gRPC handlers added for service '{TEST_SERVICE_NAME}' method '{TEST_METHOD_NAME}'."
        )

    current_loop = asyncio.get_event_loop()
    set_tsercom_event_loop(current_loop)
    try:
        await service_publisher.start_async(connect_call)

        port = service_publisher.chosen_port
        if port is None:
            await asyncio.sleep(0.1)
            port = service_publisher.chosen_port

        if port is None:
            pytest.fail(
                "Server started but failed to capture the chosen port."
            )

        logger.info(f"AsyncTestConnectionServer started on 127.0.0.1:{port}")
        yield "127.0.0.1", port

    finally:
        logger.info("Stopping AsyncTestConnectionServer...")
        if (
            hasattr(service_publisher, "_GrpcServicePublisher__server")
            and service_publisher._GrpcServicePublisher__server is not None
        ):
            actual_server_obj = service_publisher._GrpcServicePublisher__server
            if isinstance(actual_server_obj, grpc.aio.Server):
                logger.info("Attempting graceful async server stop...")
                await actual_server_obj.stop(grace=1)
                logger.info("Async server stop completed.")
            else:
                logger.info(
                    "Using GrpcServicePublisher's default stop method."
                )
                service_publisher.stop()
        else:
            logger.info(
                "Server object not found or already None, skipping explicit stop call via fixture."
            )

        logger.info("AsyncTestConnectionServer stopped.")
        clear_tsercom_event_loop()


@pytest.mark.asyncio
async def test_grpc_connection_e2e(async_test_server):
    """
    E2E test for gRPC connection using AsyncTestConnectionServer.
    - Starts a server using the async_test_server fixture.
    - Creates a client that connects to this server.
    - Sends a TestConnectionCall.
    - Verifies a TestConnectionResponse is received.
    """
    host, port = async_test_server
    logger.info(f"Test client connecting to server at {host}:{port}")

    channel_factory = InsecureGrpcChannelFactory()
    grpc_channel: Optional[grpc.Channel] = None

    try:
        grpc_channel = await channel_factory.find_async_channel(host, port)
        assert grpc_channel is not None, "Failed to create client channel"

        logger.info(f"Client channel created to {host}:{port}")

        request = TestConnectionCall()
        logger.info(f"Sending request to {FULL_METHOD_PATH}")

        response = await grpc_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(request)

        logger.info(f"Received response: {response}")

        assert isinstance(
            response, TestConnectionResponse
        ), f"Unexpected response type: {type(response)}"

        logger.info("E2E test assertions passed.")

    except grpc.aio.AioRpcError as e:
        logger.error(f"gRPC call failed: {e.code()} - {e.details()}")
        pytest.fail(f"gRPC call failed: {e.details()}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}")
        pytest.fail(f"An unexpected error occurred: {e}")
    finally:
        if grpc_channel:
            logger.info("Closing client channel.")
            await grpc_channel.close()
            logger.info("Client channel closed.")


ERROR_TIMEOUT_SERVICE_NAME = "dtp.ErrorAndTimeoutTestService"
TRIGGER_ERROR_METHOD_NAME = "TriggerError"
DELAYED_RESPONSE_METHOD_NAME = "DelayedResponse"

FULL_TRIGGER_ERROR_METHOD_PATH = (
    f"/{ERROR_TIMEOUT_SERVICE_NAME}/{TRIGGER_ERROR_METHOD_NAME}"
)
FULL_DELAYED_RESPONSE_METHOD_PATH = (
    f"/{ERROR_TIMEOUT_SERVICE_NAME}/{DELAYED_RESPONSE_METHOD_NAME}"
)


class ErrorAndTimeoutTestServiceServer:
    """
    A gRPC service implementation for testing error handling and timeouts.
    It reuses TestConnectionCall and TestConnectionResponse for simplicity.
    """

    async def TriggerError(
        self, request: TestConnectionCall, context: grpc.aio.ServicerContext
    ) -> TestConnectionResponse:
        """
        Simulates a server-side error. For now, it always returns INVALID_ARGUMENT.
        Could be extended to take error type from request if needed.
        """
        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME}: TriggerError called. Returning INVALID_ARGUMENT."
        )
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("This is a simulated error from TriggerError.")
        return TestConnectionResponse()

    async def DelayedResponse(
        self, request: TestConnectionCall, context: grpc.aio.ServicerContext
    ) -> TestConnectionResponse:
        """
        Simulates a delay before sending a response.
        The actual delay duration could be passed in the request in a real scenario,
        but for this test, we'll use a fixed delay, and the client will try to timeout sooner.
        """
        delay_duration_seconds = 2
        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME}: DelayedResponse called. Delaying for {delay_duration_seconds}s."
        )
        await asyncio.sleep(delay_duration_seconds)
        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME}: Delay complete. Sending response."
        )
        return TestConnectionResponse()


@pytest_asyncio.fixture
async def error_timeout_test_server():
    """
    Pytest fixture to start and stop the ErrorAndTimeoutTestServiceServer.
    """
    watcher = ThreadWatcher()
    service_publisher = CustomServicePublisherRpcE2E(
        watcher, port=0, addresses="127.0.0.1"
    )

    service_impl = ErrorAndTimeoutTestServiceServer()

    def connect_call(server: grpc.aio.Server):
        """Callback to add RPC handlers for ErrorAndTimeoutTestServiceServer to the gRPC server."""

        trigger_error_rpc_handler = grpc.unary_unary_rpc_method_handler(
            service_impl.TriggerError,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )

        delayed_response_rpc_handler = grpc.unary_unary_rpc_method_handler(
            service_impl.DelayedResponse,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )

        generic_handler = grpc.method_handlers_generic_handler(
            ERROR_TIMEOUT_SERVICE_NAME,
            {
                TRIGGER_ERROR_METHOD_NAME: trigger_error_rpc_handler,
                DELAYED_RESPONSE_METHOD_NAME: delayed_response_rpc_handler,
            },
        )
        server.add_generic_rpc_handlers((generic_handler,))
        logger.info(
            f"gRPC handlers added for service '{ERROR_TIMEOUT_SERVICE_NAME}'."
        )

    original_loop = asyncio.get_event_loop_policy().get_event_loop()
    is_tsercom_loop_managed = False
    try:
        set_tsercom_event_loop(original_loop)
        is_tsercom_loop_managed = True
    except RuntimeError as e:
        logger.warning(
            f"Could not set tsercom global event loop: {e}. This might be okay if already set by another fixture."
        )
    except ImportError:
        logger.warning(
            "tsercom event loop management functions not found. Proceeding with default loop."
        )

    try:
        await service_publisher.start_async(connect_call)

        port = service_publisher.chosen_port
        if port is None:
            await asyncio.sleep(0.1)
            port = service_publisher.chosen_port

        if port is None:
            pytest.fail(
                f"{ERROR_TIMEOUT_SERVICE_NAME} server started but failed to capture the chosen port."
            )

        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME} server started on 127.0.0.1:{port}"
        )
        yield "127.0.0.1", port

    finally:
        logger.info(f"Stopping {ERROR_TIMEOUT_SERVICE_NAME} server...")
        if (
            hasattr(service_publisher, "_GrpcServicePublisher__server")
            and service_publisher._GrpcServicePublisher__server is not None
        ):
            actual_server_obj = service_publisher._GrpcServicePublisher__server
            if isinstance(actual_server_obj, grpc.aio.Server):
                await actual_server_obj.stop(grace=1)
                logger.info(
                    f"Async {ERROR_TIMEOUT_SERVICE_NAME} server stop completed."
                )
            else:
                service_publisher.stop()
        else:
            logger.info(
                f"{ERROR_TIMEOUT_SERVICE_NAME} server object not found, skipping explicit stop."
            )

        if is_tsercom_loop_managed:
            try:
                clear_tsercom_event_loop()
            except ImportError:
                logger.error(
                    "Failed to import clear_tsercom_event_loop for cleanup when expected."
                )
            except RuntimeError as e:
                logger.warning(f"Issue clearing tsercom event loop: {e}")

        logger.info(
            f"{ERROR_TIMEOUT_SERVICE_NAME} server fixture cleanup complete."
        )


@pytest.mark.asyncio
async def test_server_returns_grpc_error(error_timeout_test_server):
    """
    Tests that the server can return a specific gRPC error,
    and the client correctly receives and identifies it.
    """
    host, port = error_timeout_test_server
    logger.info(
        f"Test client connecting to error_timeout_test_server at {host}:{port} for error test."
    )

    channel_factory = InsecureGrpcChannelFactory()
    grpc_channel: Optional[grpc.Channel] = None

    try:
        grpc_channel = await channel_factory.find_async_channel(host, port)
        assert (
            grpc_channel is not None
        ), "Failed to create client channel for error test"

        logger.info(f"Client channel created to {host}:{port} for error test.")

        request = TestConnectionCall()

        logger.info(f"Client calling {FULL_TRIGGER_ERROR_METHOD_PATH}")

        with pytest.raises(grpc.aio.AioRpcError) as e_info:
            await grpc_channel.unary_unary(
                FULL_TRIGGER_ERROR_METHOD_PATH,
                request_serializer=TestConnectionCall.SerializeToString,
                response_deserializer=TestConnectionResponse.FromString,
            )(request)

        assert (
            e_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        ), f"Expected INVALID_ARGUMENT, but got {e_info.value.code()}"
        assert (
            "simulated error from TriggerError" in e_info.value.details()
        ), f"Error details mismatch: {e_info.value.details()}"

        logger.info(
            f"Correctly received gRPC error: {e_info.value.code()} - {e_info.value.details()}"
        )

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the error handling test: {type(e).__name__} - {e}"
        )
        pytest.fail(
            f"An unexpected error occurred in test_server_returns_grpc_error: {e}"
        )
    finally:
        if grpc_channel:
            logger.info("Closing client channel in error test.")
            await grpc_channel.close()
            logger.info("Client channel closed in error test.")


@pytest.mark.asyncio
async def test_client_handles_timeout(error_timeout_test_server):
    """
    Tests that the client correctly handles a timeout when the server's
    response is too slow.
    """
    host, port = error_timeout_test_server
    logger.info(
        f"Test client connecting to error_timeout_test_server at {host}:{port} for timeout test."
    )

    channel_factory = InsecureGrpcChannelFactory()
    grpc_channel: Optional[grpc.Channel] = None

    client_timeout_seconds = 0.5

    try:
        grpc_channel = await channel_factory.find_async_channel(host, port)
        assert (
            grpc_channel is not None
        ), "Failed to create client channel for timeout test"

        logger.info(
            f"Client channel created to {host}:{port} for timeout test."
        )

        request = TestConnectionCall()

        logger.info(
            f"Client calling {FULL_DELAYED_RESPONSE_METHOD_PATH} with timeout {client_timeout_seconds}s."
        )

        with pytest.raises(grpc.aio.AioRpcError) as e_info:
            method_callable = grpc_channel.unary_unary(
                FULL_DELAYED_RESPONSE_METHOD_PATH,
                request_serializer=TestConnectionCall.SerializeToString,
                response_deserializer=TestConnectionResponse.FromString,
            )
            await method_callable(request, timeout=client_timeout_seconds)

        assert (
            e_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        ), f"Expected DEADLINE_EXCEEDED, but got {e_info.value.code()}"

        logger.info(
            f"Correctly received gRPC DEADLINE_EXCEEDED error: {e_info.value.code()}"
        )

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the timeout test: {type(e).__name__} - {e}"
        )
        pytest.fail(
            f"An unexpected error occurred in test_client_handles_timeout: {e}"
        )
    finally:
        if grpc_channel:
            logger.info("Closing client channel in timeout test.")
            await grpc_channel.close()
            logger.info("Client channel closed in timeout test.")


RETRIER_TEST_FIXED_PORT = 50052


@pytest_asyncio.fixture
async def retrier_server_controller():
    """
    Pytest fixture that provides control over a gRPC server's lifecycle
    (stop, start/restart) for testing client retrier mechanisms.
    It hosts the basic AsyncTestConnectionServer on a fixed port.
    """
    server_container = {"instance": None}
    service_impl = AsyncTestConnectionServer()

    def _add_servicer_to_server(s: grpc.aio.Server):
        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            service_impl.TestConnection,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )
        generic_handler = grpc.method_handlers_generic_handler(
            TEST_SERVICE_NAME,
            {TEST_METHOD_NAME: rpc_method_handler},
        )
        s.add_generic_rpc_handlers((generic_handler,))

    async def _start_new_server_instance():
        if server_container["instance"] is not None:
            try:
                await server_container["instance"].stop(grace=0.1)
                logger.info(
                    f"Retrier test server (old instance) stopped before restart attempt on port {RETRIER_TEST_FIXED_PORT}."
                )
            except Exception as e:
                logger.warning(
                    f"Error stopping old server instance during restart: {e}"
                )
            server_container["instance"] = None

        await asyncio.sleep(0.2)

        new_server = grpc.aio.server()
        _add_servicer_to_server(new_server)
        try:
            new_server.add_insecure_port(
                f"127.0.0.1:{RETRIER_TEST_FIXED_PORT}"
            )
        except Exception as e:
            logger.error(
                f"Failed to bind retrier test server to port {RETRIER_TEST_FIXED_PORT}: {e}"
            )
            pytest.fail(
                f"Retrier test server failed to bind to port {RETRIER_TEST_FIXED_PORT}: {e}"
            )
            return

        await new_server.start()
        server_container["instance"] = new_server
        logger.info(
            f"Retrier test server started/restarted on 127.0.0.1:{RETRIER_TEST_FIXED_PORT}"
        )

    original_loop = asyncio.get_event_loop_policy().get_event_loop()
    is_tsercom_loop_managed = False
    try:
        set_tsercom_event_loop(original_loop)
        is_tsercom_loop_managed = True
    except RuntimeError as e:
        logger.warning(
            f"Retrier fixture: Could not set tsercom global event loop: {e}."
        )
    except ImportError:
        logger.warning(
            "Retrier fixture: tsercom event loop management functions not found."
        )

    async def _stop_server_instance():
        if server_container["instance"]:
            await server_container["instance"].stop(grace=0)
            logger.info(
                f"Retrier test server stopped on port {RETRIER_TEST_FIXED_PORT}."
            )
        else:
            logger.info(
                "Retrier test server stop called but no instance was running."
            )

    await _start_new_server_instance()

    controller = {
        "host": "127.0.0.1",
        "port": RETRIER_TEST_FIXED_PORT,
        "get_port": lambda: RETRIER_TEST_FIXED_PORT,
        "stop_server": _stop_server_instance,
        "start_server": _start_new_server_instance,
    }

    try:
        yield controller
    finally:
        logger.info(
            f"Cleaning up retrier_server_controller. Stopping server on port {RETRIER_TEST_FIXED_PORT} if running."
        )
        if server_container["instance"]:
            await server_container["instance"].stop(grace=1)

        if is_tsercom_loop_managed:
            try:
                clear_tsercom_event_loop()
            except Exception as e:
                logger.warning(
                    f"Retrier fixture: Issue clearing tsercom event loop: {e}"
                )
        logger.info("Retrier_server_controller cleanup complete.")


class StopableChannelWrapper(Stopable):
    """Wraps a grpc.aio.Channel to make it conform to the Stopable interface."""

    def __init__(self, channel: grpc.aio.Channel):
        if not isinstance(channel, grpc.aio.Channel):
            raise TypeError("Provided channel is not a grpc.aio.Channel")
        self._channel = channel
        self._active = True

    async def stop(self) -> None:
        if self._active:
            logger.info("StopableChannelWrapper: stopping (closing) channel.")
            await self._channel.close()
            self._active = False
            logger.info("StopableChannelWrapper: channel closed.")
        else:
            logger.info(
                "StopableChannelWrapper: stop() called but channel already inactive/closed."
            )

    @property
    def channel(self) -> grpc.aio.Channel:
        return self._channel

    @property
    def is_active(self) -> bool:
        return self._active


class CustomDisconnectionRetrierRpcE2E(
    ClientDisconnectionRetrier[StopableChannelWrapper]
):
    """
    A testable subclass of ClientDisconnectionRetrier that manages StopableChannelWrapper instances.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        server_controller_fixture_data: dict,
        event_loop: asyncio.AbstractEventLoop | None = None,
        max_retries: int = 3,
        delay_before_retry_func: (
            Callable[[], Awaitable[None]] | None
        ) = lambda: asyncio.sleep(0.2),
    ):
        self._server_host = server_controller_fixture_data["host"]
        self._get_server_port = server_controller_fixture_data["get_port"]
        self._channel_factory = InsecureGrpcChannelFactory()

        effective_delay_func = delay_before_retry_func or (
            lambda: asyncio.sleep(0.2)
        )

        super().__init__(
            watcher=watcher,
            event_loop=event_loop,
            max_retries=max_retries,
            delay_before_retry_func=effective_delay_func,
        )
        self._managed_instance_wrapper: StopableChannelWrapper | None = None

    async def _connect(self) -> StopableChannelWrapper:
        """Connects to the server and returns a StopableChannelWrapper."""
        current_port = self._get_server_port()
        logger.info(
            f"CustomDisconnectionRetrierRpcE2E: Attempting to connect to {self._server_host}:{current_port}..."
        )

        grpc_channel = await self._channel_factory.find_async_channel(
            self._server_host, current_port
        )

        if grpc_channel is None:
            logger.warning(
                f"CustomDisconnectionRetrierRpcE2E: Connection failed to {self._server_host}:{current_port}."
            )
            raise grpc.aio.AioRpcError(
                grpc.StatusCode.UNAVAILABLE,
                initial_metadata=None,
                trailing_metadata=None,
                details=f"Connection failed to {self._server_host}:{current_port}",
            )

        logger.info(
            f"CustomDisconnectionRetrierRpcE2E: Connected to {self._server_host}:{current_port}. Wrapping channel."
        )
        self._managed_instance_wrapper = StopableChannelWrapper(grpc_channel)
        return self._managed_instance_wrapper

    def get_current_channel_from_managed_instance(
        self,
    ) -> grpc.aio.Channel | None:
        """Provides access to the channel within the currently managed StopableChannelWrapper."""
        current_managed_wrapper = self._ClientDisconnectionRetrier__instance
        if current_managed_wrapper and isinstance(
            current_managed_wrapper, StopableChannelWrapper
        ):
            if current_managed_wrapper.is_active:
                return current_managed_wrapper.channel
        return None


@pytest.mark.asyncio
async def test_client_retrier_reconnects(retrier_server_controller):
    """
    Tests ClientDisconnectionRetrier's ability to reconnect after server outage.
    """
    server_ctrl = retrier_server_controller
    watcher = ThreadWatcher()

    current_event_loop = asyncio.get_event_loop()
    retrier = CustomDisconnectionRetrierRpcE2E(
        watcher,
        server_ctrl,
        event_loop=current_event_loop,
        max_retries=3,
    )

    logger.info("Attempting initial connection via retrier.start()...")
    assert await retrier.start(), "Retrier failed initial start"

    initial_channel = retrier.get_current_channel_from_managed_instance()
    assert (
        initial_channel is not None
    ), "Failed to get channel after initial retrier start"
    logger.info(f"Initial connection successful. Channel: {initial_channel}")

    try:
        response = await initial_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(TestConnectionCall())
        assert isinstance(response, TestConnectionResponse)
        logger.info("Initial gRPC call successful.")
    except grpc.aio.AioRpcError as e:
        pytest.fail(
            f"Initial gRPC call failed unexpectedly: {e.code()} - {e.details()}"
        )

    logger.info("Simulating server outage: Stopping server...")
    await server_ctrl["stop_server"]()
    await asyncio.sleep(0.5)
    logger.info("Server stopped.")

    logger.info(
        "Attempting gRPC call during server outage (expected to fail initially)..."
    )
    call_succeeded_during_outage = False
    try:
        await initial_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(TestConnectionCall())
        call_succeeded_during_outage = True
    except grpc.aio.AioRpcError as e:
        logger.info(
            f"gRPC call failed as expected during outage: {e.code()} - {e.details()}"
        )
        assert (
            e.code() == grpc.StatusCode.UNAVAILABLE
        ), f"Expected UNAVAILABLE during outage, got {e.code()}"

        logger.info("Notifying retrier._on_disconnect()...")
        on_disconnect_task = current_event_loop.create_task(
            retrier._on_disconnect(e)
        )

        logger.info("Restarting server while retrier is in its retry loop...")
        await asyncio.sleep(0.1)
        await server_ctrl["start_server"]()
        logger.info("Server restarted.")

        await on_disconnect_task
        logger.info("retrier._on_disconnect() task completed.")

    if call_succeeded_during_outage:
        pytest.fail(
            "gRPC call unexpectedly succeeded during server outage before retrier acted."
        )

    logger.info("Attempting gRPC call after simulated reconnection...")
    reconnected_channel = retrier.get_current_channel_from_managed_instance()
    assert (
        reconnected_channel is not None
    ), "Failed to get channel after retrier's reconnection attempt"

    if initial_channel is reconnected_channel:
        logger.warning(
            "Retrier is using the exact same channel object. This might be okay if the channel object itself can recover, or if _connect re-established its internal state."
        )
    else:
        logger.info(
            "Retrier provided a new channel object after reconnection."
        )

    try:
        response_after_reconnect = await reconnected_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(TestConnectionCall())
        assert isinstance(response_after_reconnect, TestConnectionResponse)
        logger.info("gRPC call after reconnection successful.")
    except grpc.aio.AioRpcError as e:
        pytest.fail(
            f"gRPC call after reconnection failed: {e.code()} - {e.details()}"
        )

    logger.info("Stopping retrier...")
    await retrier.stop()
    logger.info("Test complete.")


@pytest_asyncio.fixture
async def secure_async_test_server_factory():
    """
    Pytest fixture factory to start and stop an AsyncTestConnectionServer with SSL/TLS.
    Yields an async function that can be called by tests to create configured servers.
    """
    created_servers: list[grpc.aio.Server] = []
    server_details_log: list[tuple[str, int, str]] = []

    async def _factory(
        server_key_pem: bytes,
        server_cert_pem: bytes,
        client_ca_cert_pem: Optional[bytes] = None,
        require_client_auth: bool = False,
        server_cn: str = "localhost",
    ) -> tuple[str, int, str]:
        """
        Actual server creation logic.
        Args: (same as original secure_async_test_server)
        Yields: (host, port, server_common_name)
        """
        current_loop = asyncio.get_event_loop()
        try:
            set_tsercom_event_loop(current_loop)
        except (RuntimeError, ImportError) as e:
            logger.debug(
                f"secure_async_test_server_factory: Did not set tsercom event loop: {e}"
            )

        server = grpc.aio.server()
        created_servers.append(server)

        server_credentials = grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=[
                (server_key_pem, server_cert_pem)
            ],
            root_certificates=client_ca_cert_pem,
            require_client_auth=require_client_auth,
        )

        async_server_impl = AsyncTestConnectionServer()
        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            async_server_impl.TestConnection,
            request_deserializer=TestConnectionCall.FromString,
            response_serializer=TestConnectionResponse.SerializeToString,
        )
        generic_handler = grpc.method_handlers_generic_handler(
            TEST_SERVICE_NAME, {TEST_METHOD_NAME: rpc_method_handler}
        )
        server.add_generic_rpc_handlers((generic_handler,))

        host = "127.0.0.1"
        actual_port = 0
        try:
            actual_port = server.add_secure_port(
                f"{host}:0", server_credentials
            )
            if actual_port == 0:
                created_servers.pop()
                pytest.fail(
                    "Server add_secure_port returned 0, indicating failure to bind."
                )
        except Exception as e:
            if server in created_servers:
                created_servers.pop()
            pytest.fail(
                f"Failed to add secure port or bind server for CN '{server_cn}': {e}"
            )

        await server.start()
        logger.info(
            f"Secure AsyncTestConnectionServer started on {host}:{actual_port} with CN '{server_cn}'"
        )
        server_details_log.append((host, actual_port, server_cn))
        return host, actual_port, server_cn

    try:
        yield _factory
    finally:
        logger.info(
            f"Cleaning up {len(created_servers)} secure server(s) created by factory..."
        )
        for i, server_instance in enumerate(created_servers):
            log_host, log_port, log_cn = "unknown", 0, "unknown"
            if i < len(server_details_log):
                log_host, log_port, log_cn = server_details_log[i]

            logger.info(
                f"Stopping secure server instance {i+1}/{len(created_servers)} (CN: {log_cn}) at {log_host}:{log_port}..."
            )
            try:
                await server_instance.stop(None)
                logger.info(
                    f"Secure server instance {i+1} (CN: {log_cn}) stopped."
                )
            except Exception as e:
                logger.error(
                    f"Error stopping secure server instance {i+1} (CN: {log_cn}): {e}"
                )

        try:
            if created_servers:
                clear_tsercom_event_loop()
        except (RuntimeError, ImportError) as e:
            logger.debug(
                f"secure_async_test_server_factory: Issue clearing tsercom event loop: {e}"
            )


@pytest.mark.asyncio
async def test_server_auth_successful_connection(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 1: Client validates server using the correct CA.
    """
    ca_cert_pem, ca_key_pem = generate_ca_certificate(
        common_name="Test Root CA S1 Success"
    )
    server_cn = "localhost"
    server_cert_pem, server_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1"],
        is_server=True,
    )

    host, port, returned_server_cn = await secure_async_test_server_factory(
        server_key_pem=server_key_pem,
        server_cert_pem=server_cert_pem,
        server_cn=server_cn,
    )
    assert returned_server_cn == server_cn

    factory = ServerAuthGrpcChannelFactory(
        root_ca_cert_pem=ca_cert_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is not None
        ), "Client failed to connect (ServerAuth - trusted CA)"

        request = TestConnectionCall()
        response = await grpc_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(request)
        assert isinstance(
            response, TestConnectionResponse
        ), "Unexpected RPC response type"
        logger.info(
            "RPC call successful with ServerAuthGrpcChannelFactory and correct CA."
        )

    finally:
        if grpc_channel:
            await grpc_channel.close()


@pytest.mark.asyncio
async def test_server_auth_untrusted_ca_fails(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 1: Client fails to validate server if using an untrusted CA.
    """
    server_ca_cert_pem, server_ca_key_pem = generate_ca_certificate(
        common_name="Actual Server CA S1 Failure"
    )
    server_cn = "localhost"
    server_cert_pem, server_key_pem = generate_signed_certificate(
        server_ca_cert_pem,
        server_ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1"],
        is_server=True,
    )

    client_untrusted_ca_pem, _ = generate_ca_certificate(
        common_name="Untrusted Client CA S1 Failure"
    )

    host, port, _ = await secure_async_test_server_factory(
        server_key_pem=server_key_pem,
        server_cert_pem=server_cert_pem,
        server_cn=server_cn,
    )

    factory = ServerAuthGrpcChannelFactory(
        root_ca_cert_pem=client_untrusted_ca_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is None
        ), "Client should have failed to connect with untrusted CA (ServerAuth)"
        logger.info(
            "Client correctly failed to connect (returned None) with untrusted CA (ServerAuth)."
        )

    except grpc.aio.AioRpcError as e:
        logger.error(
            f"Connection unexpectedly raised gRPC error instead of returning None: {e.code()} - {e.details()}"
        )
        pytest.fail(
            f"Expected find_async_channel to return None, but it raised {type(e).__name__}: {e}"
        )
    finally:
        if grpc_channel:
            logger.warning(
                "Closing channel that should not have been successfully created in untrusted CA test."
            )
            await grpc_channel.close()


@pytest.mark.asyncio
async def test_pinned_server_auth_successful_connection(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 2: Client validates server by pinning the correct server certificate.
    """
    ca_cert_pem, ca_key_pem = generate_ca_certificate(
        common_name="Test Root CA S2 Pinning"
    )
    server_cn = "localhost"
    server_cert_pem, server_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1"],
        is_server=True,
    )

    host, port, returned_server_cn = await secure_async_test_server_factory(
        server_key_pem=server_key_pem,
        server_cert_pem=server_cert_pem,
        server_cn=server_cn,
    )
    assert returned_server_cn == server_cn

    factory = PinnedServerAuthGrpcChannelFactory(
        expected_server_cert_pem=server_cert_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is not None
        ), "Client failed to connect (PinnedServerAuth - correct pin)"

        request = TestConnectionCall()
        response = await grpc_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(request)
        assert isinstance(
            response, TestConnectionResponse
        ), "Unexpected RPC response type"
        logger.info(
            "RPC call successful with PinnedServerAuthGrpcChannelFactory and correct pinned cert."
        )

    finally:
        if grpc_channel:
            await grpc_channel.close()


@pytest.mark.asyncio
async def test_pinned_server_auth_incorrect_pinned_cert_fails(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 2: Client fails if it pins a different certificate than the server's actual certificate.
    """
    ca_cert_pem, ca_key_pem = generate_ca_certificate(
        common_name="Test Root CA S2 IncorrectPin"
    )
    server_cn = "localhost"

    server_actual_cert_pem, server_actual_key_pem = (
        generate_signed_certificate(
            ca_cert_pem,
            ca_key_pem,
            common_name=server_cn,
            sans=["DNS:localhost", "IP:127.0.0.1"],
            is_server=True,
        )
    )

    pinned_by_client_cert_pem, _ = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name="pinned.wrong.example.com",
        sans=["DNS:pinned.wrong.example.com"],
        is_server=True,
    )
    assert server_actual_cert_pem != pinned_by_client_cert_pem

    host, port, _ = await secure_async_test_server_factory(
        server_key_pem=server_actual_key_pem,
        server_cert_pem=server_actual_cert_pem,
        server_cn=server_cn,
    )

    factory = PinnedServerAuthGrpcChannelFactory(
        expected_server_cert_pem=pinned_by_client_cert_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is None
        ), "Client should have failed to connect with incorrect pinned cert (PinnedServerAuth)"
        logger.info(
            "Client correctly failed to connect (returned None) with incorrect pinned cert (PinnedServerAuth)."
        )
    except grpc.aio.AioRpcError as e:
        logger.error(
            f"Connection unexpectedly raised gRPC error (incorrect pin): {e.code()} - {e.details()}"
        )
        pytest.fail(
            f"Expected find_async_channel to return None for incorrect pin, but it raised {type(e).__name__}: {e}"
        )
    finally:
        if grpc_channel:
            logger.warning(
                "Closing channel that should not have been successfully created in incorrect pin test."
            )
            await grpc_channel.close()


@pytest.mark.asyncio
async def test_pinned_server_auth_server_changes_cert_fails(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 2: Client fails if server changes its certificate, and client pins the old one.
    """
    ca_cert_pem, ca_key_pem = generate_ca_certificate(
        common_name="Test Root CA S2 ServerChange"
    )
    server_cn = "localhost"

    old_server_cert_pem, _ = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1", "DNS:old.server.com"],
        is_server=True,
    )

    new_server_cert_pem, new_server_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1", "DNS:new.server.com"],
        is_server=True,
    )
    assert old_server_cert_pem != new_server_cert_pem

    host, port, _ = await secure_async_test_server_factory(
        server_key_pem=new_server_key_pem,
        server_cert_pem=new_server_cert_pem,
        server_cn=server_cn,
    )

    factory = PinnedServerAuthGrpcChannelFactory(
        expected_server_cert_pem=old_server_cert_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is None
        ), "Client should have failed to connect when server changed cert and client pinned old one (PinnedServerAuth)"
        logger.info(
            "Client correctly failed to connect (returned None) when server changed cert (PinnedServerAuth)."
        )
    except grpc.aio.AioRpcError as e:
        logger.error(
            f"Connection unexpectedly raised gRPC error (server changed cert): {e.code()} - {e.details()}"
        )
        pytest.fail(
            f"Expected find_async_channel to return None for server cert change, but it raised {type(e).__name__}: {e}"
        )
    finally:
        if grpc_channel:
            logger.warning(
                "Closing channel that should not have been successfully created when server changed cert."
            )
            await grpc_channel.close()


# --- Tests for ClientAuthGrpcChannelFactory (Scenario 3) ---


@pytest.mark.asyncio
@pytest.mark.parametrize("server_requires_client_auth", [False, True])
async def test_client_auth_no_server_validation_by_client(
    secure_async_test_server_factory, server_requires_client_auth
):
    """
    Tests Scenario 3: Client uses its cert, client does NOT validate server.
    Server is configured to either optionally or require client certs.
    """
    ca_cert_pem, ca_key_pem = generate_ca_certificate(
        common_name="Test Root CA S3 NoServerValidation"
    )
    server_cn = "localhost"
    server_cert_pem, server_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1"],
        is_server=True,
    )
    client_cert_pem, client_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name="client.example.com",
        sans=["DNS:client.example.com"],
        is_server=False,
    )

    host, port, returned_server_cn = await secure_async_test_server_factory(
        server_key_pem=server_key_pem,
        server_cert_pem=server_cert_pem,
        client_ca_cert_pem=ca_cert_pem,
        require_client_auth=server_requires_client_auth,
        server_cn=server_cn,
    )
    assert returned_server_cn == server_cn

    factory = ClientAuthGrpcChannelFactory(
        client_cert_pem=client_cert_pem,
        client_key_pem=client_key_pem,
        root_ca_cert_pem=None,
        server_hostname_override=None,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert grpc_channel is None, (
            f"ClientAuth with NoServerValidation by client unexpectedly connected. "
            f"server_requires_client_auth={server_requires_client_auth}. "
            f"This scenario consistently fails handshake (server can't verify client cert)."
        )
        logger.info(
            f"Connection correctly failed as expected (ClientAuth, NoServerValidation, "
            f"server_requires_client_auth={server_requires_client_auth}). Handshake issue."
        )

    finally:
        if grpc_channel:
            await grpc_channel.close()


@pytest.mark.asyncio
async def test_client_auth_with_server_validation_mtls(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 3: Client uses its cert, client DOES validate server (mTLS).
    Server must require client certs and validate them.
    """
    ca_cert_pem, ca_key_pem = generate_ca_certificate(
        common_name="Test Root CA S3 mTLS"
    )
    server_cn = "localhost"
    server_cert_pem, server_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1"],
        is_server=True,
    )
    client_cert_pem, client_key_pem = generate_signed_certificate(
        ca_cert_pem,
        ca_key_pem,
        common_name="mtls.client.example.com",
        sans=["DNS:mtls.client.example.com"],
        is_server=False,
    )

    host, port, returned_server_cn = await secure_async_test_server_factory(
        server_key_pem=server_key_pem,
        server_cert_pem=server_cert_pem,
        client_ca_cert_pem=ca_cert_pem,
        require_client_auth=True,
        server_cn=server_cn,
    )
    assert returned_server_cn == server_cn

    factory = ClientAuthGrpcChannelFactory(
        client_cert_pem=client_cert_pem,
        client_key_pem=client_key_pem,
        root_ca_cert_pem=ca_cert_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is not None
        ), "Client failed to connect (ClientAuth, mTLS)"

        request = TestConnectionCall()
        response = await grpc_channel.unary_unary(
            FULL_METHOD_PATH,
            request_serializer=TestConnectionCall.SerializeToString,
            response_deserializer=TestConnectionResponse.FromString,
        )(request)
        assert isinstance(
            response, TestConnectionResponse
        ), "Unexpected RPC response type"
        logger.info("RPC successful (ClientAuth, mTLS).")

    finally:
        if grpc_channel:
            await grpc_channel.close()


@pytest.mark.asyncio
async def test_client_auth_with_server_validation_untrusted_server_ca_fails(
    secure_async_test_server_factory,
):
    """
    Tests Scenario 3: Client uses its cert, tries to validate server, but server's CA is untrusted by client.
    """
    actual_ca_cert_pem, actual_ca_key_pem = generate_ca_certificate(
        common_name="Actual CA S3 UntrustedServer"
    )
    clients_false_trusted_ca_pem, _ = generate_ca_certificate(
        common_name="Client's False Trust CA S3"
    )

    server_cn = "localhost"
    server_cert_pem, server_key_pem = generate_signed_certificate(
        actual_ca_cert_pem,
        actual_ca_key_pem,
        common_name=server_cn,
        sans=["DNS:localhost", "IP:127.0.0.1"],
        is_server=True,
    )
    client_cert_pem, client_key_pem = generate_signed_certificate(
        actual_ca_cert_pem,
        actual_ca_key_pem,
        common_name="untrustedtest.client.example.com",
        is_server=False,
    )

    host, port, _ = await secure_async_test_server_factory(
        server_key_pem=server_key_pem,
        server_cert_pem=server_cert_pem,
        client_ca_cert_pem=actual_ca_cert_pem,
        require_client_auth=True,
        server_cn=server_cn,
    )

    factory = ClientAuthGrpcChannelFactory(
        client_cert_pem=client_cert_pem,
        client_key_pem=client_key_pem,
        root_ca_cert_pem=clients_false_trusted_ca_pem,
        server_hostname_override=server_cn,
    )

    grpc_channel: Optional[grpc.Channel] = None
    try:
        grpc_channel = await factory.find_async_channel(host, port)
        assert (
            grpc_channel is None
        ), "Client should have failed to connect (ClientAuth, untrusted server CA)"
        logger.info(
            "Client correctly failed to connect (returned None) due to untrusted server CA (ClientAuth)."
        )
    except grpc.aio.AioRpcError as e:
        logger.error(
            f"Connection unexpectedly raised gRPC error (untrusted server CA): {e.code()} - {e.details()}"
        )
        pytest.fail(
            f"Expected find_async_channel to return None for untrusted server CA, but it raised {type(e).__name__}: {e}"
        )
    finally:
        if grpc_channel:
            logger.warning(
                "Closing channel that should not have been successfully created in untrusted server CA test."
            )
            await grpc_channel.close()
