from abc import ABC, abstractmethod
import asyncio
from functools import partial
from typing import Callable, Generic, Optional, TypeVar

from tsercom.rpc.connection.client_reconnection_handler import ClientReconnectionManager
from tsercom.rpc.grpc.grpc_caller import delay_before_retry, is_grpc_error, is_server_unavailable_error
from tsercom.threading.aio.aio_utils import get_running_loop_or_none, is_running_on_event_loop, run_on_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.stopable import Stopable


TInstanceType = TypeVar("TInstanceType", bound = Stopable)
class ClientDisconnectionRetrier(
        ABC, Generic[TInstanceType], ClientReconnectionManager):
    def __init__(
            self,
            watcher : ThreadWatcher,
            safe_disconnection_handler : Optional[Callable[[], None]] = None):
        assert issubclass(type(watcher), ThreadWatcher)

        self.__instance : TInstanceType = None
        self.__watcher = watcher
        self.__disconnection_handler = safe_disconnection_handler

        self.__event_loop : asyncio.AbstractEventLoop = None

    @abstractmethod
    def _connect(self) -> TInstanceType:
        pass

    async def start(self) -> bool:
        try:
            self.__event_loop = get_running_loop_or_none()
            assert not self.__event_loop is None

            self.__instance = self._connect()
            assert not self.__instance is None
            assert issubclass(type(self.__instance), Stopable), \
                    type(self.__instance)
            return True
        except Exception as error:
            if not is_server_unavailable_error(error):
                raise error
            
            print("Connection to server FAILED with error", error)
            return False

    async def stop(self):
        if self.__event_loop is None:
            return
        
        if not is_running_on_event_loop(self.__event_loop):
            run_on_event_loop(self.stop, self.__event_loop)
            return

        if not self.__instance is None:
            await self.__instance.stop()
            self.__instance = None
    
    async def _on_disconnect(self, error : Optional[Exception] = None):
        # Jump to the same thread from which this instance was initially created
        # to avoid any weird threading issues or race conditions.
        if not is_running_on_event_loop(self.__event_loop):
            run_on_event_loop(partial(self._on_disconnect, error),
                              self.__event_loop)
            return
        
        # This should never happen, but check just in case.
        assert not error is None, "ERROR: NO EXCEPTION FOUND!"

        # These should NEVER be swallowed. So raise it first.
        if isinstance(error, AssertionError):
            self.__watcher.on_exception_seen(error)
        
        # Since a thread hop might happen, there is a possibility of a race
        # condition here. So check against self.__instance to avoid it.
        if self.__instance is None:
            return
        # Stop the running instance so it can be replaced.
        await self.__instance.stop()

        # If the gRPC connection failed for an unexpected reason, just let it
        # die without crashing the entire runtime.
        if is_grpc_error(error) and not is_server_unavailable_error(error):
            print("WARNING: gRPC Session Ended Unexpectedly:", error)
            if not self.__disconnection_handler is None:
                await self.__disconnection_handler()
            return

        # If its NOT a gRPC error, expose it to the runtime since it was a local
        # crash.
        elif not is_server_unavailable_error(error):
            self.__watcher.on_exception_seen(error)
        
        # If it IS a server unavailable error, retry until the server becomes
        # available. This is done on the same thread from which the instance was
        # initially started to avoid any weirdness if the underlying 
        print("WILL RETRY")  # TODO: Remove this
        
        while True:
            try:
                await delay_before_retry()
                print("RETRYING CONNECTION")  # TODO: Remove this line.
                self.__instance = self._connect()
                break
            except Exception as error:
                if not is_server_unavailable_error(error):
                    raise error