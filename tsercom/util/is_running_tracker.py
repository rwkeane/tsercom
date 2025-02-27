import asyncio
from collections.abc import Coroutine
from functools import partial
import threading
from typing import Any, AsyncIterator, TypeVar

from tsercom.threading.aio.aio_utils import get_running_loop_or_none, is_running_on_event_loop, run_on_event_loop
from tsercom.threading.atomic import Atomic


TReturnType = TypeVar("TReturnType")
class IsRunningTracker(Atomic[bool]):
    """
    This class provides a state-tracker to track whether the using object is
    running or stopped (the initial state), and utilities to work with this
    state.

    NOTE: Only the set(), start(), and stop() methods of this class are thread
    safe. The remaining methods, once called from a given asyncio event loop,
    may only be called from that loop in future.
    """
    def __init__(self):
        # For tracking the current state.
        self.__running_barrier = asyncio.Event()
        self.__stopped_barrier = asyncio.Event()

        # For keeping the event loop with which this instance is associated in
        # sync.
        self.__event_loop_lock = threading.Lock()
        self.__event_loop : asyncio.AbstractEventLoop = None

        super().__init__(False)

    @property
    def is_running(self):
        """
        Returns whether or not this instance is currently running.
        """
        return self.get()

    def start(self):
        """
        Sets this instance to running at the next available opportunity. May be
        called from any thread.
        """
        return self.set(True)
    
    def stop(self):
        """
        Sets this instance to stopped at the next available opportunity. May be
        called from any thread.
        """
        return self.set(False)

    def set(self, value : bool):
        """
        Sets the state of this instance be running when |value| is True and 
        stopped when |value| is False at the next available opportunity. May be
        called from any thread.
        """
        super().set(value)

        with self.__event_loop_lock:
            if self.__event_loop is None:
                return
            
        # Block to ensure the state internally matches the stored value.
        task = run_on_event_loop(partial(self.__set_impl, value),
                                 self.__event_loop)
        
        # To clear the event loop and similar.
        def clear(x : Any):
            with self.__event_loop_lock:
                self.__running_barrier = asyncio.Event()
                self.__stopped_barrier = asyncio.Event()
                
                self.__event_loop = None
        task.add_done_callback(clear)

        # If already on the event loop to which the task gets posted,
        # calling .result() triggers deadlock.
        if not is_running_on_event_loop(self.__event_loop):
            task.result()
    
    async def wait_until_started(self):
        """
        Waits until the event loop has been started before continuing. May only
        be called from a single asyncio loop, along with the other asyncio
        functions of this object.
        """
        if self.get():
            return
        
        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        await self.__running_barrier.wait()
    
    async def wait_until_stopped(self):
        """
        Waits until the event loop has been stopped before continuing. May only
        be called from a single asyncio loop, along with the other asyncio
        functions of this object.
        """
        if not self.get():
            return
        
        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        await self.__stopped_barrier.wait()
            
    async def task_or_stopped(
            self,
            call : Coroutine[Any, Any, TReturnType]) -> TReturnType | None:
        """
        Runs |call| until completion, or until the current instance changes to
        False. May only be called from a single asyncio loop, along with the
        other asyncio functions of this object. Returns the result of |call| on
        completion, and None if this instance is stopped prior to then.
        """
        # Wrap the task in asyncio.shield() to ensure that no task using a lock
        # gets cancelled while holding the lock, resulting in a deadlock. This
        # isn't ideal and might lead to issues later, but for now its the best
        # solution I can come up with.
        call_task = asyncio.shield(asyncio.create_task(call))

        # This SHOULD get around the issue where |call_task| gets dropped
        # resulting in an error.
        if not self.get():
            call_task.cancel()
            return None

        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        # Wait for either |call| to finish or for this instance to stop running.
        stop_check_task = asyncio.create_task(self.__stopped_barrier.wait())
        done, pending = await asyncio.wait(
                [ call_task, stop_check_task ],
                return_when = asyncio.FIRST_COMPLETED)
        
        # Cancel everything not yet done.
        for task in pending:
            task.cancel()
        
        # Return the result if |call| completed, and None otherwise.
        if call_task in done:
            return call_task.result()
        
        return None
    
    async def create_stoppable_iterator(
            self, iterator : AsyncIterator[TReturnType]) -> \
                    AsyncIterator[TReturnType]:
        """
        Creates an iterator that can be used to ensure that iteration stops when
        this instance stops running.
        
        Returns an AsyncIterator such that for each item retrieved from the
        iterator, it is either anext(|iterator|) or None if this instance stops
        running.
        """
        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        return IsRunningTracker.__IteratorWrapper(iterator, self)

    async def __set_impl(self, value : bool):
        if value:
            self.__running_barrier.set()
            self.__stopped_barrier.clear()
        else:
            self.__stopped_barrier.set()
            self.__running_barrier.clear()

    async def __ensure_event_loop_initialized(self):
        if not self.__event_loop is None:
            return
        
        with self.__event_loop_lock:
            if not self.__event_loop is None:
                return
            
            self.__event_loop = get_running_loop_or_none()
            assert not self.__event_loop is None
            value = self.get()
        await self.__set_impl(value)

    class __IteratorWrapper:
        def __init__(self,
                     iterator : AsyncIterator[TReturnType],
                     tracker : 'IsRunningTracker'):
            self.__iterator = iterator
            self.__tracker = tracker

        def __aiter__(self):
            return self
        
        async def __anext__(self):
            result = await self.__tracker.task_or_stopped(
                    anext(self.__iterator))
            if not self.__tracker.get():
                raise StopAsyncIteration()
            
            return result