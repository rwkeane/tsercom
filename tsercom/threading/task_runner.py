import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future
from functools import partial
import threading
import time
from typing import Any, List, ParamSpec, TypeVar
from collections.abc import Callable
import uuid

from util.threading.throwing_thread import ThrowingThread
from util.threading.throwing_thread_pool_executor import ThrowingThreadPoolExecutor

P = ParamSpec('P')
T = TypeVar('T')

class TaskRunner:
    """
    This class provides a simple wrapper around a single threaded thread pool,
    as well as providing a number of utility functions to simplify its use.
    """
    def __init__(self):
        self.__thread_pool_id = uuid.uuid4()
        self.__prefix = f"TaskRunner-{self.__thread_pool_id}"
       
        # Waits for an exception, and passes it to waiting thread when an
        # exception is thrown.
        self.__barrier = threading.Event()
        self.__exceptions_lock = threading.Lock()
        self.__exceptions : List[Exception] = []

        # Create a single-threaded thread pool to back the task runner's main
        # thread.
        self.__executor = self.create_delegated_thread_pool_executor(
                max_workers=1,
                thread_name_prefix=self.__prefix)
        
        # Create the event loop, but don't return until the new thread has
        # started. Else, it creates a race condition with the first call to
        # get |self.__event_loop|.
        self.__event_loop_thread : threading.Thread = None
        self.__event_loop : asyncio.AbstractEventLoop = None

        barrier = threading.Event()
        self.__start_asyncio_loop(barrier)
        barrier.wait()

    def post_task(self,
                  call: Callable[P, T],
                  *args: P.args,
                  **kwargs: P.kwargs) -> Future[T]:
        """
        Runs |call| as soon as possible. If |call| is synchronous, it is run on
        the ThreadPoolExecutor backing this instance. Else, it is run on the
        main EventLoop, which is on a thread owned by this instance. Exceptions
        are returned when run_until_exception() is called.
        """
        if not asyncio.iscoroutinefunction(call):
            return self.__executor.submit(call, *args, **kwargs)
        else:
            return asyncio.run_coroutine_threadsafe(
                    self.__wrap_for_exception(call, *args, **kwargs),
                    self.__event_loop)

    def post_task_with_delay(self,
                             call: Callable[P, T],
                             delay_ms: int | float,
                             *args,
                             **kwargs):
        """Runs |call| as soon as possible after |delay| ms"""
        if delay_ms <= 0:
            self.post_task(call, *args, **kwargs)
            return

        # Delay this call until enough time has passes.
        thread = self.create_short_lived_thread(
                target = partial(self.__delay_and_post_task,
                                 call,
                                 delay_ms,
                                 *args,
                                 **kwargs))
        thread.start()

    def is_running_on_task_runner(self):
        """
        Returns whether or not the current thread is owned by this TaskRunner.
        """
        return self.__prefix in threading.current_thread().name
    
    def is_running_on_task_runner_event_loop(self):
        return asyncio._get_running_loop() == self.__event_loop

    def create_short_lived_thread(
            self, target : Callable[[], None]) -> threading.Thread:
        """
        Creates a threading.Thread instance such that exceptions on that thread
        are exposed as they would be for exceptions occuring on the TaskRunner.
        """
        return ThrowingThread(target = target,
                              on_error_cb = self.on_exception_seen)
    
    def create_delegated_thread_pool_executor(
            self, *args, **kwargs) -> ThreadPoolExecutor:
        """
        Creates a ThreadPool such that each thread of the ThreadPool has error
        handling as would be done for an exception on this TaskRunner instance.
        """
        return ThrowingThreadPoolExecutor(
                error_cb = self.on_exception_seen, *args, **kwargs)

    def run_until_exception(self):
        """
        Runs until an exception is seen, at which point it will be thrown.
        """
        if self.is_running_on_task_runner():
            assert False, "Cannot call run_until_exception() from TaskRunner!"

        while True:
            self.__barrier.wait()
            with self.__exceptions_lock:
                if len(self.__exceptions) == 0:
                    continue

                raise ExceptionGroup("Errors hit in async thread(s)!",
                                     self.__exceptions)

    def on_exception_seen(self, e : Exception):
        with self.__exceptions_lock:
            self.__exceptions.append(e)
            self.__barrier.set()
            
    def __delay_and_post_task(self,
                              call: Callable[[], None],
                              delay_ms: int | float,
                              *args,
                              **kwargs):
        assert not self.is_running_on_task_runner()
        time.sleep(delay_ms / 1000)
        self.post_task(call, *args, **kwargs)
    
    def __start_asyncio_loop(self, barrier : threading.Event):
        def handle_exception(loop, context):
            exception = context.get("exception")
            print("HIT EXCEPTION", exception)
            if exception:
                self.on_exception_seen(exception)
            else:
                print("ERROR! NO EXCEPTION FOUND")

        def start_event_loop():
            self.__event_loop = asyncio.new_event_loop()
            self.__event_loop.set_exception_handler(handle_exception)
            asyncio.set_event_loop(self.__event_loop)

            # The loop is in a good state. Continue execution of the ctor.
            barrier.set()
            self.__event_loop.run_forever()

        self.__event_loop_thread = \
                self.create_short_lived_thread(start_event_loop)
        self.__event_loop_thread.start()

    async def __wrap_for_exception(self,
                                   call: Callable[P, T],
                                   *args: P.args,
                                   **kwargs: P.kwargs) -> T:
        try:
            return await call(*args, **kwargs)
        except Exception as e:
            self.on_exception_seen(e)
        except Warning as e:
            self.on_exception_seen(e)