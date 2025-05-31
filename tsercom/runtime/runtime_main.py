"""Main entry points and initialization logic for Tsercom runtimes."""

import logging  # Import logging
from typing import Any, List
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.channel_factory_selector import ChannelFactorySelector
from tsercom.runtime.client.client_runtime_data_handler import (
    ClientRuntimeDataHandler,
)
from tsercom.api.split_process.split_process_error_watcher_sink import (
    SplitProcessErrorWatcherSink,
)
from tsercom.runtime.server.server_runtime_data_handler import (
    ServerRuntimeDataHandler,
)
from tsercom.threading.aio.aio_utils import run_on_event_loop
from functools import partial
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    create_tsercom_event_loop_from_watcher,
    is_global_event_loop_set,
    get_global_event_loop,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher
import concurrent.futures

# import logging # logging is already imported above
import asyncio  # Added for asyncio.all_tasks

logger = logging.getLogger(__name__)  # Define logger for this module


def initialize_runtimes(
    thread_watcher: ThreadWatcher,
    initializers: List[RuntimeFactory[Any, Any]],
    *,
    is_testing: bool = False,
) -> List[Runtime]:
    """Initializes and starts a list of Tsercom runtimes.

    Args:
        thread_watcher: Monitors threads for the runtimes.
        initializers: A list of `RuntimeFactory` instances to create and start runtimes from.
        is_testing: Boolean flag for testing mode.

    Returns:
        A list of the started `Runtime` instances.
    """
    assert is_global_event_loop_set()

    channel_factory_selector = ChannelFactorySelector()  # Instantiate once

    runtimes: List[Runtime] = []
    for factory_idx, initializer_factory in enumerate(
        initializers
    ):  # initializer_factory is clearer
        # Retrieve GrpcChannelFactoryConfig from the initializer_factory
        # initializer_factory is a RuntimeFactory, which is a RuntimeInitializer, which is a RuntimeConfig
        grpc_config = (
            initializer_factory.grpc_channel_factory_config
        )  # Access the property

        # Create the channel factory using the config
        # The create_factory_from_config method handles None config by returning InsecureGrpcChannelFactory
        channel_factory = channel_factory_selector.create_factory_from_config(
            grpc_config
        )

        data_reader = (
            initializer_factory._remote_data_reader()
        )  # Was remote_data_reader()
        event_poller = (
            initializer_factory._event_poller()
        )  # Was event_poller()

        if initializer_factory.is_client():
            data_handler = ClientRuntimeDataHandler(
                thread_watcher,
                data_reader,
                event_poller,
                is_testing=is_testing,
            )
        elif initializer_factory.is_server():
            data_handler = ServerRuntimeDataHandler(
                data_reader,
                event_poller,
                is_testing=is_testing,
            )
        else:
            raise ValueError("Invalid endpoint type!")

        runtime_instance = initializer_factory.create(
            thread_watcher, data_handler, channel_factory
        )
        runtimes.append(runtime_instance)

    for runtime_idx, runtime in enumerate(runtimes):
        active_loop = get_global_event_loop()

        coro_to_run = runtime.start_async
        future = run_on_event_loop(coro_to_run, event_loop=active_loop)

        def _runtime_start_done_callback(
            f: concurrent.futures.Future,
            thread_watcher: ThreadWatcher,
        ):
            try:
                if f.done() and not f.cancelled():
                    exc = f.exception()
                    if exc is not None:
                        thread_watcher.on_exception_seen(exc)
            except Exception:
                pass

        future.add_done_callback(
            partial(
                _runtime_start_done_callback, thread_watcher=thread_watcher
            )
        )

    return runtimes


def remote_process_main(
    initializers: List[RuntimeFactory[Any, Any]],
    error_queue: MultiprocessQueueSink[Exception],
    *,
    is_testing: bool = False,
) -> None:
    """Main function for a Tsercom runtime operating in a remote process.

    Sets up event loop, error handling via a queue, initializes runtimes,
    and runs until an exception occurs.

    Args:
        initializers: List of `RuntimeFactory` instances.
        error_queue: A `MultiprocessQueueSink` to send exceptions back to the parent.
        is_testing: Boolean flag for testing mode.
    """
    clear_tsercom_event_loop(try_stop_loop=False)

    thread_watcher = ThreadWatcher()
    # Remove redundant create_tsercom_event_loop_from_watcher call here
    # The loop will be created and assigned inside the try block.

    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    runtimes: List[Runtime] = []
    loop = None  # Initialize loop to None
    try:
        # Force-clear the global event loop variable directly before creating a new one in the child process
        # This is to ensure the check within create_tsercom_event_loop_from_watcher passes.
        # This is a more direct way to address the "Only one Global Event Loop may be set" error.
        import tsercom.threading.aio.global_event_loop as global_loop_module

        global_loop_module._INTERNAL_clear_global_event_loop_for_process_start_ONLY()  # Assuming a new helper or direct access

        # Loop creation moved inside the try block to ensure it's defined before use
        loop = create_tsercom_event_loop_from_watcher(
            thread_watcher
        )  # Removed replace_policy
        runtimes = initialize_runtimes(
            thread_watcher, initializers, is_testing=is_testing
        )
        sink.run_until_exception()

    except Exception as e:
        if error_queue:
            try:
                logger.error(
                    f"remote_process_main: Exception caught: {type(e).__name__} - {e}",
                    exc_info=True,
                )
                error_queue.put_nowait(e)
                # Attempt to ensure the item is flushed before the process terminates.
                error_queue.close()
                error_queue.join_thread()
                logger.info(
                    f"remote_process_main: Successfully queued exception {type(e).__name__}."
                )
            except Exception as q_err:
                logger.error(
                    f"remote_process_main: Failed to queue exception {type(e).__name__} due to queue error: {type(q_err).__name__} - {q_err}",
                    exc_info=True,
                )
        else:
            logger.error(
                "remote_process_main: error_queue is None, cannot queue exception.",
                exc_info=True,
            )
        raise  # Re-raise the original exception e to terminate the child process
    finally:
        logger.info("remote_process_main: Starting shutdown sequence.")

        # Stop all runtimes
        try:
            if loop and not loop.is_closed():
                if not loop.is_running():
                    logger.info(
                        "remote_process_main: Loop not running, attempting to run stop tasks for runtimes."
                    )
                    for i, runtime in enumerate(runtimes):
                        try:
                            logger.info(
                                f"Stopping runtime {i} (loop was not running)."
                            )
                            if asyncio.iscoroutinefunction(runtime.stop):
                                loop.run_until_complete(runtime.stop(None))
                            else:
                                runtime.stop(None)
                            logger.info(f"Runtime {i} stopped.")
                        except Exception as e_stop:
                            logger.error(
                                f"Error stopping runtime {i} (loop was not running): {e_stop}",
                                exc_info=True,
                            )
                else:  # Loop is running
                    stop_tasks_futures = []
                    for i, runtime in enumerate(runtimes):
                        try:
                            logger.info(
                                f"Scheduling stop for runtime {i} on running loop."
                            )
                            task = loop.create_task(runtime.stop(None))
                            stop_tasks_futures.append(task)
                        except Exception as e_schedule_stop:
                            logger.error(
                                f"Error scheduling stop for runtime {i}: {e_schedule_stop}",
                                exc_info=True,
                            )

                    if stop_tasks_futures:
                        logger.info(
                            f"Gathering {len(stop_tasks_futures)} runtime stop tasks."
                        )
                        try:
                            loop.run_until_complete(
                                asyncio.gather(
                                    *stop_tasks_futures, return_exceptions=True
                                )
                            )
                            logger.info("Runtime stop tasks gathered.")
                        except Exception as e_gather:
                            logger.error(
                                f"Error waiting for runtime stops: {e_gather}",
                                exc_info=True,
                            )
            else:
                logger.warning(
                    "Loop is None or closed at start of runtime stop sequence."
                )
        except Exception as e_runtime_cleanup:
            logger.error(
                f"General error during runtime cleanup phase: {e_runtime_cleanup}",
                exc_info=True,
            )

        # Stop all factories
        logger.info("Stopping RuntimeFactory instances.")
        for i, factory in enumerate(initializers):
            try:
                factory._stop()
                logger.info(f"Factory {i} stopped.")
            except Exception as e_factory_stop:
                logger.error(
                    f"Error stopping factory {i}: {e_factory_stop}",
                    exc_info=True,
                )

        # Finalize event loop shutdown
        logger.info("Finalizing event loop shutdown.")
        try:
            if loop and not loop.is_closed():
                try:
                    # Cancel pending tasks
                    all_tasks = asyncio.all_tasks(loop=loop)
                    # current_task = asyncio.current_task(loop=loop) # Avoid this if not in a task
                    tasks_to_cancel = {
                        task for task in all_tasks if not task.done()
                    }  # Simpler: cancel all non-done tasks

                    if tasks_to_cancel:
                        logger.info(
                            f"Cancelling {len(tasks_to_cancel)} outstanding tasks."
                        )
                        for task in tasks_to_cancel:
                            task.cancel()
                        # Allow cancellations to be processed
                        loop.run_until_complete(
                            asyncio.gather(
                                *tasks_to_cancel, return_exceptions=True
                            )
                        )
                        logger.info(
                            "Outstanding tasks cancellation processed."
                        )

                    # Run the loop briefly to process any final items after cancellations
                    if (
                        loop.is_running()
                    ):  # ensure it is running before run_until_complete for sleep
                        loop.run_until_complete(asyncio.sleep(0.01, loop=loop))

                except (
                    RuntimeError
                ) as e_task_cancel:  # e.g. if loop is already stopping/closed
                    logger.error(
                        f"RuntimeError during task cancellation: {e_task_cancel}",
                        exc_info=True,
                    )
                finally:  # Ensure loop.close() is attempted
                    if loop.is_running():
                        try:
                            loop.call_soon_threadsafe(
                                loop.stop
                            )  # Request stop if still running
                            # Run one last time to process the stop if call_soon_threadsafe was used
                            # This might not be strictly necessary if run_until_complete was just called for sleep
                            # but can help ensure stop is processed.
                            # loop.run_until_complete(loop.shutdown_asyncgens()) # an alternative for graceful shutdown
                        except RuntimeError as e_loop_stop:
                            logger.error(
                                f"RuntimeError stopping loop: {e_loop_stop}",
                                exc_info=True,
                            )

                    if not loop.is_closed():  # Check again before closing
                        logger.info("Closing event loop.")
                        loop.close()
                    else:
                        logger.info("Event loop was already closed.")
            else:
                logger.info(
                    "Loop is None or already closed before final loop finalization steps."
                )
        except Exception as e_loop_cleanup:
            logger.error(
                f"General error during event loop finalization phase: {e_loop_cleanup}",
                exc_info=True,
            )

        try:
            clear_tsercom_event_loop(try_stop_loop=False)
            logger.info("Global tsercom event loop cleared.")
        except Exception as e_clear_tsercom:
            logger.error(
                f"Error clearing tsercom global event loop: {e_clear_tsercom}",
                exc_info=True,
            )

        logger.info("remote_process_main: Shutdown sequence complete.")
