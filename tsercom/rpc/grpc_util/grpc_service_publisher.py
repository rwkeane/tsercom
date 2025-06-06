"""Provides GrpcServicePublisher for hosting gRPC services."""

import asyncio
import logging
from functools import partial
from typing import Callable, Iterable

import grpc

from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.ip import get_all_address_strings

AddServicerCB = Callable[["grpc.Server"], None]


class GrpcServicePublisher:
    """
    Helper class to publish gRPC services.
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        addresses: str | Iterable[str] | None = None,
    ):
        """
        Creates a new gRPC Service hosted on a given ``port`` and network
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
        """
        Starts a synchronous server.
        """
        self.__server: grpc.Server = grpc.server(  # type: ignore
            self.__watcher.create_tracked_thread_pool_executor(max_workers=10)
        )
        connect_call(self.__server)
        self._connect()
        self.__server.start()

    async def start_async(self, connect_call: AddServicerCB) -> None:
        """
        Starts an asynchronous server and waits for it to be serving.
        """
        cf_future = run_on_event_loop(
            partial(self.__start_async_impl, connect_call)
        )
        await asyncio.wrap_future(cf_future)

    async def __start_async_impl(self, connect_call: AddServicerCB) -> None:
        """Internal implementation to start the asynchronous gRPC server.

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
                    f"Running gRPC Server on {address}:{port_out} (expected: {self.__port})"
                )
                worked += 1
            except Exception as e:
                if isinstance(e, AssertionError):
                    self.__watcher.on_exception_seen(e)
                    raise e
                # Log other exceptions that prevent binding to a specific address
                logging.warning(
                    f"Failed to bind gRPC server to {address}:{self.__port}. Error: {e}"
                )
                continue

        if worked == 0:
            logging.error("FAILED to host gRPC Service on any address.")

        return worked != 0

    def stop(self) -> None:
        """
        Stops the server.
        """
        # TODO(review): The current synchronous stop() may not correctly handle
        # graceful shutdown for an asyncio gRPC server (if __server is grpc.aio.Server).
        # Consider making this method async or adding a separate stop_async().
        if self.__server is None:
            raise RuntimeError("Server not started")
        self.__server.stop()  # This is a blocking call for non-async server
        logging.info("gRPC Server stopped.")
