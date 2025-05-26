"""Main entry points and initialization logic for Tsercom runtimes."""

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
import asyncio  # Added for asyncio.Future
from functools import partial  # Added for functools.partial
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    create_tsercom_event_loop_from_watcher,
    is_global_event_loop_set,
    get_global_event_loop,  # Added for get_global_event_loop
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher


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

    channel_factory_selector = ChannelFactorySelector()
    channel_factory = channel_factory_selector.get_instance()

    runtimes: List[Runtime] = []
    for initializer_factory in initializers:
        data_reader = initializer_factory._remote_data_reader()
        event_poller = initializer_factory._event_poller()

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

    # Ensure asyncio and functools.partial are imported (done above)
    # from tsercom.threading.aio.global_event_loop import get_global_event_loop (done above)
    # from tsercom.threading.thread_watcher import ThreadWatcher (already imported)

    for runtime in runtimes:
        # Use the global event loop which start_in_process would have set.
        # This is important because runtime.start_async is expected to run on that specific loop.
        active_loop = get_global_event_loop()
        future = run_on_event_loop(runtime.start_async, event_loop=active_loop)

        # Define the callback function
        # It needs a reference to the thread_watcher passed to initialize_runtimes
        def _runtime_start_done_callback(
            f: asyncio.Future, tw_ref: ThreadWatcher
        ):
            try:
                # Check if the future completed with an exception
                if f.done() and not f.cancelled():
                    exc = f.exception()
                    if exc is not None:
                        tw_ref.on_exception_seen(exc)
            except Exception:
                # Log errors from the callback itself, if any, to prevent loop crashes.
                # (Requires a logger, or pass for now)
                pass

        # Add the callback, using partial to pass the thread_watcher
        future.add_done_callback(
            partial(_runtime_start_done_callback, tw_ref=thread_watcher)
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
    # Only needed on linux systems.
    clear_tsercom_event_loop()

    thread_watcher = ThreadWatcher()
    create_tsercom_event_loop_from_watcher(thread_watcher)
    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    runtimes: List[Runtime] = []  # Initialize runtimes to an empty list
    try:
        runtimes = initialize_runtimes(
            thread_watcher, initializers, is_testing=is_testing
        )
        # If initialize_runtimes succeeds, then wait for other errors from sink
        sink.run_until_exception()
    except Exception as e:
        if error_queue:  # error_queue is an argument to remote_process_main
            error_queue.put_nowait(e)
        raise  # Important to re-raise so child process indicates error
    finally:
        for runtime in runtimes:
            run_on_event_loop(runtime.stop)
