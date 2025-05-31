"""Provides GrpcServicePublisher for hosting gRPC services."""

from functools import partial
from typing import Callable, Iterable, Optional  # Added Optional
import grpc
import logging
import os  # Added os

# Removed: from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.ip import get_all_address_strings

AddServicerCB = Callable[["grpc.Server"], None]


class GrpcServicePublisher:
    """
    This class Is a helper to publish a gRPC Service/
    """

    def __init__(
        self,
        watcher: ThreadWatcher,
        port: int,
        addresses: str | Iterable[str] | None = None,
        server_key_path: Optional[str] = None,  # New
        server_cert_path: Optional[str] = None,  # New
        client_ca_cert_path: Optional[str] = None,  # New
    ):
        """
        Creates a new gRPC Service hosted on a given |port| and network
        interfaces assocaited with |addresses|.
        Can be configured for TLS or mTLS using server_key_path, server_cert_path, and client_ca_cert_path.
        """
        if addresses is None:
            addresses = get_all_address_strings()
        elif isinstance(addresses, str):
            addresses = [addresses]
        self.__addresses = list(addresses)

        self.__port = port
        self.__server: grpc.Server = None  # type: ignore
        self.__watcher = watcher
        self._chosen_port: Optional[int] = None  # Added
        self.__server_key_path = server_key_path
        self.__server_cert_path = server_cert_path
        self.__client_ca_cert_path = client_ca_cert_path

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
        Starts an asynchronous server.
        """
        await self.__start_async_impl(connect_call)

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
        self.__server  # type: ignore
        self._connect()
        await self.__server.start()  # type: ignore

    def _read_file_bytes(self, file_path: Optional[str]) -> Optional[bytes]:
        if file_path is None:
            return None
        if not os.path.exists(file_path):
            # This error will be caught by the caller (_connect)
            raise FileNotFoundError(f"Credential file not found: {file_path}")
        with open(file_path, "rb") as f:
            return f.read()

    def _connect(self) -> bool:
        worked = 0
        if self.__server_key_path and self.__server_cert_path:
            # Attempt to configure a secure port
            try:
                key_bytes = self._read_file_bytes(self.__server_key_path)
                cert_bytes = self._read_file_bytes(self.__server_cert_path)

                # Key and cert bytes must exist if paths were provided
                if (
                    not key_bytes
                ):  # Should be caught by FileNotFoundError in _read_file_bytes if path was bad
                    raise ValueError(
                        f"Server key file loaded as empty: {self.__server_key_path}"
                    )
                if (
                    not cert_bytes
                ):  # Should be caught by FileNotFoundError in _read_file_bytes if path was bad
                    raise ValueError(
                        f"Server cert file loaded as empty: {self.__server_cert_path}"
                    )

                client_ca_bytes = self._read_file_bytes(
                    self.__client_ca_cert_path
                )  # Optional

                server_credentials = grpc.ssl_server_credentials(
                    [(key_bytes, cert_bytes)],
                    root_certificates=client_ca_bytes,
                    require_client_auth=bool(
                        client_ca_bytes
                    ),  # True if client_ca_bytes is not None
                )

                for address in self.__addresses:
                    try:
                        port_out = self.__server.add_secure_port(  # type: ignore
                            f"{address}:{self.__port}", server_credentials
                        )
                        if (
                            worked == 0
                        ):  # Store the first successfully bound port
                            self._chosen_port = port_out
                        logging.info(
                            f"Running SECURE gRPC Server on {address}:{port_out} (expected: {self.__port})"
                        )
                        worked += 1
                    except Exception as e:
                        if isinstance(e, AssertionError):
                            self.__watcher.on_exception_seen(e)
                            raise e
                        logging.warning(
                            f"Failed to bind SECURE gRPC server to {address}:{self.__port}. Error: {e}"
                        )
                        continue

            except (FileNotFoundError, ValueError) as e:
                logging.error(
                    f"Failed to load credentials for secure gRPC server: {e}. Server will not start securely."
                )
                # Do not proceed to insecure fallback if secure was intended but failed.
                if worked == 0:  # Ensure log message if no addresses worked
                    logging.error(
                        "FAILED to host SECURE gRPC Service on any address due to credential error."
                    )
                return (
                    False  # Indicate connection setup failed for secure mode
                )

        else:
            # Configure an insecure port (original logic)
            for address in self.__addresses:
                try:
                    port_out = self.__server.add_insecure_port(  # type: ignore
                        f"{address}:{self.__port}"
                    )
                    if worked == 0:  # Store the first successfully bound port
                        self._chosen_port = port_out
                    logging.info(
                        f"Running INSECURE gRPC Server on {address}:{port_out} (expected: {self.__port})"
                    )
                    worked += 1
                except Exception as e:
                    if isinstance(e, AssertionError):
                        self.__watcher.on_exception_seen(e)
                        raise e
                    logging.warning(
                        f"Failed to bind INSECURE gRPC server to {address}:{self.__port}. Error: {e}"
                    )
                    continue

        if worked == 0:
            # This log might be redundant if specific secure/insecure logs already covered it
            # but serves as a general failure message if no ports were bound.
            logging.error(
                "FAILED to host gRPC Service on any address (final check in _connect)."
            )

        return worked > 0  # Returns True if at least one address was bound

    def get_chosen_port(self) -> Optional[int]:
        return self._chosen_port

    async def stop(self, grace: Optional[float] = None) -> None:
        """
        Stops the server.
        """
        if self.__server is None:
            # It's possible stop is called before start_async completed or if _connect failed.
            logging.info(
                "gRPC Server stop called, but server was not fully started or not running."
            )
            return

        # Check if it's an async server (grpc.aio.Server)
        # Need to import asyncio for iscoroutinefunction if not already available
        import asyncio

        if hasattr(self.__server, "stop") and asyncio.iscoroutinefunction(self.__server.stop):  # type: ignore
            await self.__server.stop(grace)  # type: ignore
        else:
            # For synchronous server, stop is blocking and might not have grace.
            # This part of the code might need to be run in an executor if called from async code.
            # However, start_async sets up an async server, so this branch is less likely with current usage.
            self.__server.stop(0)  # type: ignore
        logging.info("gRPC Server stopped.")
