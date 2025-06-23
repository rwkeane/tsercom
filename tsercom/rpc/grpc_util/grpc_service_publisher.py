"""Provides GrpcServicePublisher for hosting gRPC services."""

import asyncio
import logging
from collections.abc import Callable, Iterable

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_health.v1._async import HealthServicer

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.ip import get_all_address_strings

AddServicerCB = Callable[["grpc.Server"], None]


class GrpcServicePublisher:
    """Helper class to publish gRPC services."""

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        addresses: str | Iterable[str] | None = None,
    ):
        """Create a new gRPC Service on ``port`` and network ``addresses``."""
        if addresses is None:
            addresses = get_all_address_strings()
        elif isinstance(addresses, str):
            addresses = [addresses]
        self.__addresses = list(addresses)

        self._health_servicer = HealthServicer()
        self.__port = port
        self.__server: grpc.Server = None
        self.__watcher = watcher

    def start(self, connect_call: AddServicerCB) -> None:
        """Start a synchronous server."""
        self.__server: grpc.Server = grpc.server(  # type: ignore
            self.__watcher.create_tracked_thread_pool_executor(max_workers=10)
        )
        connect_call(self.__server)
        self._connect()
        self.__server.start()

    async def start_async(self, connect_call: AddServicerCB) -> None:
        """Start an asynchronous server and wait for it to be serving.

        Runs on the event loop this coroutine is scheduled on.
        """
        # __start_async_impl is an async method, so it can be directly awaited.
        # It will run on the same event loop that start_async is currently running on.
        await self.__start_async_impl(connect_call)

    async def __start_async_impl(self, connect_call: AddServicerCB) -> None:
        """Start the asynchronous gRPC server internally.

        Configures the server with an exception interceptor and starts it.

        Args:
            connect_call: Callback to add servicer implementations to the server.

        """
        # Moved import here to break potential circular dependency

        from tsercom.rpc.grpc_util.async_grpc_exception_interceptor import (
            AsyncGrpcExceptionInterceptor,
        )

        interceptor = AsyncGrpcExceptionInterceptor(self.__watcher)
        self.__server: grpc.Server = grpc.aio.server(  # type: ignore
            self.__watcher.create_tracked_thread_pool_executor(max_workers=1),
            interceptors=[interceptor],
            maximum_concurrent_rpcs=None,
        )
        connect_call(self.__server)
        health_pb2_grpc.add_HealthServicer_to_server(
            self._health_servicer, self.__server
        )
        # Empty string for service_name sets the overall server health.
        await self._health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        await asyncio.sleep(0)  # Allow gRPC to process the status update internally
        self._connect()
        await self.__server.start()

    async def set_service_health_status(
        self, service: str, status: health_pb2.HealthCheckResponse.ServingStatus
    ) -> None:
        """Explicity sets the health status of a specific service or the overall server.

        Args:
            service: The name of the service to set the status for.
                     An empty string sets the overall server health.
            status: The desired health status (e.g., SERVING, NOT_SERVING).

        """
        if self._health_servicer is None:
            logging.error(
                "Cannot set service status because health servicer is not initialized."
            )
            return

        status_name = health_pb2.HealthCheckResponse.ServingStatus.Name(status)
        logging.info(f"Setting health status for service '{service}' to {status_name}.")
        await self._health_servicer.set(service, status)

        # Allow time for the status to propagate, especially in an async environment.
        await asyncio.sleep(0.1)

    def _connect(self) -> bool:
        """Binds the gRPC server to the configured addresses and port.

        Iterates through specified addresses, attempting to bind the server.
        Logs successes and failures.

        Returns:
            True if the server successfully bound to at least one address,
            False otherwise.

        """
        worked = 0
        for address in self.__addresses:
            try:
                port_out = self.__server.add_insecure_port(f"{address}:{self.__port}")
                logging.info(
                    "Running gRPC Server on %s:%s (expected: %s)",
                    address,
                    port_out,
                    self.__port,
                )
                worked += 1
            except RuntimeError as e:  # More specific for port binding/setup issues
                if isinstance(
                    e, AssertionError
                ):  # AssertionError is a RuntimeError subtype
                    self.__watcher.on_exception_seen(e)
                    raise e
                logging.warning(
                    "Failed to bind gRPC server to %s:%s. Error: %s",
                    address,
                    self.__port,
                    e,
                )
                continue

        if worked == 0:
            logging.error("FAILED to host gRPC Service on any address.")

        return worked != 0

    def stop(self) -> None:
        """Stop the server.

        For grpc.aio.Server, use stop_async() instead.
        This method is intended for synchronous grpc.server.
        """
        if self.__server is None:
            logging.warning(
                "GrpcServicePublisher: Server not started or already stopped "
                "when calling stop()."
            )
            return

        if isinstance(self.__server, grpc.aio.Server):
            logging.error(
                "GrpcServicePublisher: Synchronous stop() called on an "
                "grpc.aio.Server. This is incorrect and will not stop the "
                "server gracefully. Use stop_async() instead."
            )
            return

        logging.info(
            "GrpcServicePublisher: Attempting to stop gRPC Server (sync call)..."
        )
        try:
            self.__server.stop(
                0
            )  # For grpc.Server, argument is grace period in seconds
        except TypeError:
            self.__server.stop()  # Fallback
        logging.info("GrpcServicePublisher: gRPC Server stopped (sync call).")

    async def stop_async(self) -> None:
        """Stop the asynchronous gRPC server gracefully."""
        if self.__server is None:
            logging.warning(
                "GrpcServicePublisher: Server not started or already stopped "
                "when calling stop_async()."
            )
            return

        if not isinstance(self.__server, grpc.aio.Server):
            logging.warning(
                "GrpcServicePublisher: stop_async() called on a non-aio server. "
                "Attempting to use its synchronous stop() method."
            )
            try:
                self.__server.stop(0)
            except TypeError:
                self.__server.stop()  # Fallback
            return

        logging.info(
            "GrpcServicePublisher: Attempting to stop gRPC Server gracefully (async)..."
        )
        try:
            await self.__server.stop(
                grace=1.0
            )  # For grpc.aio.Server, keyword is 'grace'
            logging.info("GrpcServicePublisher: gRPC Server stopped (async).")
        except Exception as e:
            logging.exception(
                "GrpcServicePublisher: Exception during async server stop: %r",
                e,
            )


async def check_grpc_channel_health(
    channel: grpc.aio.Channel, service: str = ""
) -> bool:
    """Check the health of a given gRPC channel for a specific service.

    Args:
        channel: The gRPC channel to check.
        service: The name of the service to check. An empty string checks overall
            server health.

    Returns:
        True if the channel is serving for the specified service, False otherwise
            (including errors).

    """
    if not channel:
        logging.debug(
            f"Health check: Channel is None or empty for service '{service}'."
        )
        return False

    logging.debug(
        f"Health check: Creating HealthStub for channel: {channel}, "
        f"service: '{service}'",
    )
    health_stub = health_pb2_grpc.HealthStub(channel)
    request = health_pb2.HealthCheckRequest(service=service)
    logging.debug(
        f"Health check: Sending HealthCheckRequest: {request} for service '{service}'"
    )

    try:
        response = await health_stub.Check(request, timeout=1.0)
        logging.info(f"Health check: Received response status: {response.status}")
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            logging.debug("Health check: Status is SERVING, returning True.")
            return True
        else:
            service_status_name = health_pb2.HealthCheckResponse.ServingStatus.Name(
                response.status
            )
            logging.warning(
                f"Health check: Status is {service_status_name}, expected SERVING. "
                "Returning False.",
            )
            return False
    except grpc.aio.AioRpcError as e:
        logging.error(
            f"Health check: AioRpcError for channel {channel}: {e.details()} "
            f"(code: {e.code()})",
        )
        return False
    except Exception as e:
        logging.error(
            f"Health check: Unexpected error for channel {channel}: {e}", exc_info=True
        )
        return False
