"""Provides GrpcServicePublisher for hosting gRPC services."""

import logging
from collections.abc import Callable, Iterable

import grpc

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.ip import get_all_address_strings

AddServicerCB = Callable[["grpc.Server"], None]


class GrpcServicePublisher:
    """Helper class to publish gRPC services.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        addresses: str | Iterable[str] | None = None,
    ):
        """Creates a new gRPC Service hosted on a given ``port`` and network
        interfaces assocaited with ``addresses``.
        """
        if addresses is None:
            addresses = get_all_address_strings()
        elif isinstance(addresses, str):
            addresses = [addresses]
        self.__addresses = list(addresses)

        self.__port = port
        self.__server: grpc.Server = None
        self.__watcher = watcher

    def start(self, connect_call: AddServicerCB) -> None:
        """Starts a synchronous server.
        """
        self.__server: grpc.Server = grpc.server(  # type: ignore
            self.__watcher.create_tracked_thread_pool_executor(max_workers=10)
        )
        connect_call(self.__server)
        self._connect()
        self.__server.start()

    async def start_async(self, connect_call: AddServicerCB) -> None:
        """Starts an asynchronous server and waits for it to be serving.
        Runs on the event loop this coroutine is scheduled on.
        """
        # __start_async_impl is an async method, so it can be directly awaited.
        # It will run on the same event loop that start_async is currently running on.
        await self.__start_async_impl(connect_call)

    async def __start_async_impl(self, connect_call: AddServicerCB) -> None:
        """Internal implementation to start the asynchronous gRPC server.

        Configures the server with an exception interceptor and starts it.

        Args:
            connect_call: Callback to add servicer implementations to the server.

        """
        # Moved import here to break potential circular dependency
        # pylint: disable=import-outside-toplevel
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
        self._connect()
        await self.__server.start()

    def _connect(self) -> bool:
        """Binds the gRPC server to the configured addresses and port.

        Iterates through specified addresses, attempting to bind the server.
        Logs successes and failures.

        Returns:
            True if the server successfully bound to at least one address, False otherwise.

        """
        # Connect to a port.
        worked = 0
        for address in self.__addresses:
            try:
                port_out = self.__server.add_insecure_port(
                    f"{address}:{self.__port}"
                )
                logging.info(
                    "Running gRPC Server on %s:%s (expected: %s)",
                    address,
                    port_out,
                    self.__port,
                )
                worked += 1
            except (
                RuntimeError
            ) as e:  # More specific for port binding/setup issues
                if isinstance(
                    e, AssertionError
                ):  # AssertionError is a RuntimeError subtype
                    self.__watcher.on_exception_seen(e)
                    raise e
                # Log other exceptions that prevent binding to a specific address
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
        """Stops the server.
        For grpc.aio.Server, use stop_async() instead.
        This method is intended for synchronous grpc.server.
        """
        if self.__server is None:
            logging.warning(
                "GrpcServicePublisher: Server not started or already stopped when calling stop()."
            )
            return

        if isinstance(self.__server, grpc.aio.Server):
            logging.error(
                "GrpcServicePublisher: Synchronous stop() called on an grpc.aio.Server. "
                "This is incorrect and will not stop the server gracefully. "
                "Use stop_async() instead."
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
        """Stops the asynchronous gRPC server gracefully.
        """
        if self.__server is None:
            logging.warning(
                "GrpcServicePublisher: Server not started or already stopped when calling stop_async()."
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
            # Log the full exception for better debugging
            logging.exception(
                "GrpcServicePublisher: Exception during async server stop: %r",
                e,
            )
