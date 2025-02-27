from functools import partial
from typing import Callable, Iterable
import grpc

from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.ip import get_all_address_strings

AddServicerCB = Callable[[grpc.Server], None]
class GrpcServicePublisher:
    """
    This class Is a helper to publish a gRPC Service/
    """
    def __init__(self,
                 watcher : ThreadWatcher,
                 port : int,
                 addresses : str | Iterable[str] | None = None):
        """
        Creates a new gRPC Service hosted on a given |port| and network
        interfaces assocaited with |addresses|.
        """
        if addresses is None:
            addresses = get_all_address_strings()
        elif isinstance(addresses, str):
            addresses = [ addresses ]
        self.__addresses = list(addresses)

        self.__port = port
        self.__server : grpc.Server = None
        self.__watcher = watcher

    def start(self, connect_call : AddServicerCB):
        """
        Starts a synchronous server.
        """
        self.__server : grpc.Server = grpc.server(
                self.__watcher.create_tracked_thread_pool_executor(
                        max_workers=10))
        connect_call(self.__server)
        self._connect()
        self.__server.start()

    def start_async(self, connect_call : AddServicerCB):
        """
        Starts an asynchronous server.
        """
        run_on_event_loop(partial(self.__start_async_impl, connect_call))

    async def __start_async_impl(self, connect_call : AddServicerCB):
        interceptor = AsyncGrpcExceptionInterceptor(self.__watcher)
        self.__server : grpc.Server = grpc.aio.server(
                self.__watcher.create_tracked_thread_pool_executor(
                        max_workers=1),
                interceptors = [ interceptor ],
                maximum_concurrent_rpcs = None)
        connect_call(self.__server)
        self.__server
        self._connect()
        await self.__server.start()
        
    def _connect(self) -> bool:
        # Connect to a port.
        worked = 0
        for address in self.__addresses:
            try:
                port_out = self.__server.add_insecure_port(
                        f'{address}:{self.__port}')
                print(f"\tRunning gRPC Server on {address}:{port_out} "
                      f"(expected: {self.__port})")
                worked += 1
            except Exception as e:
                if isinstance(e, AssertionError):
                    self.__watcher.on_exception_seen(e)
                    raise e
                continue

        if worked == 0:
            print("FAILED to host gRPC Service")

        return worked != 0

    def stop(self):
        self.__server.stop()
        print(f"Server stopped!")