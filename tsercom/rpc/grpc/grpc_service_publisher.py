from functools import partial
from typing import Callable, Iterable
import grpc

from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor
from tsercom.threading.task_runner import TaskRunner
from tsercom.util.ip import get_all_address_strings

AddServicerCB = Callable[[grpc.Server], None]
class GrpcServicePublisher:
    """
    This class Is a helper to publish a gRPC Service/
    """
    def __init__(self,
                 task_runner : TaskRunner,
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
        self.__task_runner = task_runner

    def start(self, connect_call : AddServicerCB):
        """
        Starts a synchronous server.
        """
        self.__server : grpc.Server = grpc.server(
                self.__task_runner.create_delegated_thread_pool_executor(
                        max_workers=10))
        connect_call(self.__server)
        self._connect()
        self.__server.start()

    def start_async(self, connect_call : AddServicerCB):
        """
        Starts an asynchronous server.
        """
        self.__task_runner.post_task(
                partial(self.__start_async_impl, connect_call))

    async def __start_async_impl(self, connect_call : AddServicerCB):
        interceptor = AsyncGrpcExceptionInterceptor(self.__task_runner)
        self.__server : grpc.Server = grpc.aio.server(
                self.__task_runner.create_delegated_thread_pool_executor(
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
            except Exception:
                continue

        if worked == 0:
            print("FAILED to host gRPC Service")

        return worked != 0

    def stop(self):
        self.__server.stop()
        print(f"Server stopped!")