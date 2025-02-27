from abc import ABC
import asyncio
import threading
from typing import Deque, Generic, List, TypeVar

from tsercom.threading.aio.aio_utils import get_running_loop_or_none, is_running_on_event_loop, run_on_event_loop
from tsercom.threading.atomic import Atomic
from tsercom.threading.thread_watcher import ThreadWatcher


kMaxResponses = 30

TResultType = TypeVar('TResultType')
class AsyncPoller(Generic[TResultType], ABC):
    """
    This class provides an asynchronous queueing mechanism, to allow for 
    subscribers to request the next available instance, and asynchronously
    await that until a new instance(s) is published.
    """
    def __init__(self):
        self.__responses = Deque[TResultType]()
        self.__barrier = asyncio.Event()
        self.__lock = threading.Lock()

        self.__is_loop_running = Atomic[bool](False)

        self.__event_loop : asyncio.AbstractEventLoop = None

    @property
    def event_loop(self):
        return self.__event_loop

    def on_available(self, new_instance : TResultType):
        """
        Enqueues a newly available |new_instance|.
        """

        with self.__lock:
            if len(self.__responses) > kMaxResponses:
                self.__responses.popleft()
            self.__responses.append(new_instance)

        if not self.__is_loop_running.get():
            return

        assert not self.__event_loop is None
        run_on_event_loop(self.__set_results_available, self.__event_loop)
        
    async def __set_results_available(self):
        self.__barrier.set()

    def flush(self):
        """
        Flushes any unread data out of the queue.
        """
        with self.__lock:
            self.__responses.clear()

    async def wait_instance(self) -> List[TResultType]:
        """
        Asynchronously waits for new data to be available in the queue.
        """
        # Set the current loop if it has not yet been set.
        if not self.__is_loop_running.get():
            assert self.__event_loop is None
            self.__event_loop = get_running_loop_or_none()
            assert not self.__event_loop is None

            self.__is_loop_running.set(True)

        # Check the current loop.
        assert is_running_on_event_loop(self.__event_loop)

        # Keep trying to pull results until some are found.
        while self.__is_loop_running.get():
            # If there are items to return, get them all.
            queue = None
            with self.__lock:
                if len(self.__responses) > 0:
                    queue = self.__responses
                    self.__responses = Deque[TResultType]()

            # Return the results, but do it outside the mutex to avoid blocking
            # as much as possible.
            if not queue is None:
                responses = []
                while len(queue) > 0:
                    responses.append(queue.popleft())

                assert len(responses) > 0
                return responses
                
            # If there is NO pending item, wait for one to show up.
            await self.__barrier.wait()
            self.__barrier.clear()
        
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        return await self.wait_instance()
    
    def __len__(self):
        with self.__lock:
            return len(self.__responses)