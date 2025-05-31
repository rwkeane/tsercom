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
    for factory_idx, initializer_factory in enumerate(initializers):
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
    create_tsercom_event_loop_from_watcher(thread_watcher)

    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    runtimes: List[Runtime] = []
    try:
        runtimes = initialize_runtimes(
            thread_watcher, initializers, is_testing=is_testing
        )
        sink.run_until_exception()

    except Exception as e:
        if error_queue:
            error_queue.put_nowait(e)
        raise
    finally:
        for runtime_idx, runtime in enumerate(runtimes):
            run_on_event_loop(partial(runtime.stop, None))

        for factory_idx, factory in enumerate(initializers):
            factory._stop()
