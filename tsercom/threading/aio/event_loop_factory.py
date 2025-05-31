"""Provides EventLoopFactory for creating and managing asyncio event loops that run in separate threads, monitored by a ThreadWatcher."""

import asyncio
from typing import Any, Optional
import threading
import logging

from tsercom.threading.thread_watcher import ThreadWatcher


# Class responsible for creating and managing an asyncio event loop in a separate thread.
class EventLoopFactory:
    """
    Factory class for creating and managing an asyncio event loop.

    The event loop runs in a separate thread and is monitored by a ThreadWatcher.
    """

    def __init__(self, watcher: "ThreadWatcher") -> None:
        """
        Initializes the EventLoopFactory.

        Args:
            watcher (ThreadWatcher): The ThreadWatcher instance to monitor the event loop thread.

        Raises:
            ValueError: If the watcher argument is None.
            TypeError: If the watcher is not a subclass of ThreadWatcher.
        """
        if watcher is None:
            raise ValueError(
                "Watcher argument cannot be None for EventLoopFactory."
            )
        if not issubclass(type(watcher), ThreadWatcher):
            raise TypeError(
                f"Watcher must be a subclass of ThreadWatcher, got {type(watcher).__name__}."
            )
        self.__watcher = watcher

        # These attributes are initialized in the start_asyncio_loop method.
        # They are typed as Optional because they are None until that method is called.
        self.__event_loop_thread: Optional[threading.Thread] = None
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

    def start_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        """
        Starts an asyncio event loop in a new thread.

        The loop is configured with an exception handler that logs unhandled
        exceptions and notifies the ThreadWatcher.

        Returns:
            asyncio.AbstractEventLoop: The created and running event loop.
        """
        barrier = threading.Event()

        def handle_exception(
            loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            """
            Handles exceptions occurring in the event loop.

            Logs the exception and notifies the ThreadWatcher.

            Args:
                loop (asyncio.AbstractEventLoop): The event loop where the exception occurred.
                context (dict[str, Any]): A dictionary containing exception details.
            """
            exception = context.get("exception")
            message = context.get("message")
            if exception:
                if isinstance(exception, asyncio.CancelledError):
                    raise exception

                logging.error(
                    f"Unhandled exception in event loop: {message}",
                    exc_info=exception,
                )
                self.__watcher.on_exception_seen(exception)
            else:
                logging.critical(
                    f"Event loop exception handler called without an exception. Context message: {message}"
                )

        def start_event_loop() -> None:
            """
            Initializes and runs the asyncio event loop.

            This function is intended to be run in a separate thread.
            It sets up the new event loop, assigns the custom exception handler,
            sets it as the current event loop for the thread, signals that
            the loop is ready, and then runs the loop forever.
            """
            try:
                local_event_loop = asyncio.new_event_loop()

                local_event_loop.set_exception_handler(handle_exception)

                asyncio.set_event_loop(
                    local_event_loop
                )  # Associates loop with this thread's context

                self.__event_loop = local_event_loop

                barrier.set()  # Notifies waiting thread that loop is ready

                local_event_loop.run_forever()  # Starts the event loop
            except Exception:
                raise
            finally:
                if (
                    "local_event_loop" in locals()
                    and hasattr(local_event_loop, "is_closed")
                    and not local_event_loop.is_closed()
                ):
                    if local_event_loop.is_running():
                        local_event_loop.stop()
                    local_event_loop.close()

        self.__event_loop_thread = self.__watcher.create_tracked_thread(
            target=start_event_loop  # Pass the function to be executed
        )
        self.__event_loop_thread.start()

        # Wait until the event loop is set up and running.
        barrier.wait()

        # At this point, self.__event_loop should have been set by start_event_loop.
        # If it's not, something went wrong in thread initialization.
        assert (
            self.__event_loop is not None
        ), "Event loop was not initialized in the thread."
        return self.__event_loop

    def shutdown(
        self, timeout: Optional[float] = 7.0
    ) -> None:  # Increased default timeout
        """
        Stops the event loop and joins its thread.

        This method attempts to gracefully shut down the event loop by:
        1. Cancelling all outstanding tasks on the loop.
        2. Stopping the event loop.
        3. Joining the thread that runs the event loop.

        Args:
            timeout (Optional[float]): The maximum time in seconds to wait for
                                     the thread to join. Defaults to 5.0 seconds.
        """
        logger = logging.getLogger(__name__)
        logger.info(
            f"EventLoopFactory {id(self)}: Initiating shutdown sequence..."
        )

        if not self.__event_loop_thread or not self.__event_loop:
            logger.info(
                f"EventLoopFactory {id(self)}: Loop or thread already None. Shutdown aborted or already completed."
            )
            return

        loop_to_clean = self.__event_loop
        thread_to_join = self.__event_loop_thread

        logger.debug(
            f"EventLoopFactory {id(self)}: Loop to clean: {id(loop_to_clean)}, Thread to join: {thread_to_join.name if thread_to_join else 'None'}"
        )

        if loop_to_clean.is_running():
            logger.info(
                f"EventLoopFactory {id(self)}: Loop {id(loop_to_clean)} is running. Scheduling _shutdown_tasks_and_stop_loop."
            )
            # Schedule the async helper to run on the loop's own thread.
            # Allocate most of the time to the helper, then a smaller fixed time for join.
            join_grace_period = (
                2.0  # seconds to leave for the thread join itself
            )
            if timeout is not None:
                future_timeout = max(
                    1.0, timeout - join_grace_period
                )  # Ensure at least 1s for future
            else:
                future_timeout = 5.0  # Default if overall timeout is None for shutdown method

            future = asyncio.run_coroutine_threadsafe(
                self._shutdown_tasks_and_stop_loop(), loop_to_clean
            )
            try:
                logger.debug(
                    f"EventLoopFactory {id(self)}: Waiting for _shutdown_tasks_and_stop_loop for {future_timeout}s."
                )
                future.result(timeout=future_timeout)
                logger.info(
                    f"EventLoopFactory {id(self)}: _shutdown_tasks_and_stop_loop completed for loop {id(loop_to_clean)}."
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"EventLoopFactory {id(self)}: _shutdown_tasks_and_stop_loop timed out after {future_timeout}s for loop {id(loop_to_clean)}. Forcing stop."
                )
                loop_to_clean.call_soon_threadsafe(loop_to_clean.stop)
            except Exception as e:
                logger.error(
                    f"EventLoopFactory {id(self)}: Error waiting for _shutdown_tasks_and_stop_loop on loop {id(loop_to_clean)}: {e}",
                    exc_info=True,
                )
                loop_to_clean.call_soon_threadsafe(loop_to_clean.stop)
        else:
            logger.info(
                f"EventLoopFactory {id(self)}: Loop {id(loop_to_clean)} was not running when shutdown was called."
            )

        # Join the thread
        if timeout is not None:
            thread_join_timeout = max(
                0.5,
                join_grace_period if timeout >= join_grace_period else timeout,
            )
        else:
            thread_join_timeout = (
                join_grace_period  # Use the grace period if no overall timeout
            )

        if thread_to_join and thread_to_join.is_alive():
            logger.info(
                f"EventLoopFactory {id(self)}: Joining event loop thread '{thread_to_join.name}' (timeout={thread_join_timeout}s)..."
            )
            thread_to_join.join(timeout=thread_join_timeout)
            if thread_to_join.is_alive():
                logger.warning(
                    f"EventLoopFactory {id(self)}: Thread '{thread_to_join.name}' did NOT join/terminate within {thread_join_timeout}s timeout."
                )
            else:
                logger.info(
                    f"EventLoopFactory {id(self)}: Thread '{thread_to_join.name}' joined successfully."
                )
        elif thread_to_join:
            logger.info(
                f"EventLoopFactory {id(self)}: Event loop thread '{thread_to_join.name}' was not alive before join."
            )
        else:
            logger.warning(
                f"EventLoopFactory {id(self)}: self.__event_loop_thread was None, cannot join."
            )

        # Final cleanup of loop object. Its own thread's 'finally' block (in start_event_loop) should have closed it.
        if loop_to_clean and not loop_to_clean.is_closed():
            logger.warning(
                f"EventLoopFactory {id(self)}: Loop {id(loop_to_clean)} was not closed by its thread's finally block. This indicates an issue in the thread's exit path or the loop's state."
            )
            # Avoid closing from this thread if the loop's thread might still be technically alive but unjoined.
            if not thread_to_join or not thread_to_join.is_alive():
                try:
                    logger.info(
                        f"EventLoopFactory {id(self)}: Thread seems dead, attempting to close loop {id(loop_to_clean)} now."
                    )
                    if (
                        loop_to_clean.is_running()
                    ):  # Should not be running if thread joined and stop was effective
                        loop_to_clean.stop()
                    loop_to_clean.close()
                    logger.info(
                        f"EventLoopFactory {id(self)}: Loop {id(loop_to_clean)} closed."
                    )
                except Exception as e:
                    logger.error(
                        f"EventLoopFactory {id(self)}: Error closing loop {id(loop_to_clean)}: {e}",
                        exc_info=True,
                    )
            else:
                logger.error(
                    f"EventLoopFactory {id(self)}: Loop {id(loop_to_clean)} not closed, but its thread {thread_to_join.name if thread_to_join else 'Unknown'} may still be alive. Not closing loop from this thread."
                )

        self.__event_loop = None
        self.__event_loop_thread = None
        logger.info(f"EventLoopFactory {id(self)}: Shutdown method finished.")

    async def _shutdown_tasks_and_stop_loop(self) -> None:
        """
        Async helper to be run on the factory's own event loop to cancel tasks and stop the loop.
        """
        logger = logging.getLogger(__name__)
        if not self.__event_loop:  # Should not happen if called correctly
            logger.error(
                f"EventLoopFactory {id(self)}: _shutdown_tasks_and_stop_loop called but self.__event_loop is None."
            )
            return

        loop = self.__event_loop
        logger.info(
            f"EventLoopFactory {id(self)}: Async helper _shutdown_tasks_and_stop_loop running on loop {id(loop)}."
        )

        tasks_to_cancel = [
            t
            for t in asyncio.all_tasks(loop=loop)
            if t is not asyncio.current_task(loop=loop) and not t.done()
        ]

        if not tasks_to_cancel:
            logger.info(
                f"EventLoopFactory {id(self)}: No tasks to cancel on loop {id(loop)}. Stopping loop."
            )
            if loop.is_running():
                loop.stop()
            return

        logger.info(
            f"EventLoopFactory {id(self)}: Helper coroutine cancelling {len(tasks_to_cancel)} tasks on loop {id(loop)}."
        )
        for task_idx, task in enumerate(tasks_to_cancel):
            # logger.debug(f"EventLoopFactory {self}: Helper cancelling task {task_idx}: {str(task)[:150]}")
            task.cancel()

        try:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info(
                f"EventLoopFactory {id(self)}: Helper coroutine finished gathering cancelled tasks on loop {id(loop)}."
            )
        except Exception as e:
            logger.error(
                f"EventLoopFactory {id(self)}: Exception during asyncio.gather in _shutdown_tasks_and_stop_loop for loop {id(loop)}: {e}",
                exc_info=True,
            )

        if loop.is_running():  # Check again before stopping
            logger.info(
                f"EventLoopFactory {id(self)}: Helper coroutine stopping loop {id(loop)}."
            )
            loop.stop()
        else:
            logger.info(
                f"EventLoopFactory {id(self)}: Loop {id(loop)} already stopped before helper could stop it."
            )
