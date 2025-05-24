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
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    create_tsercom_event_loop_from_watcher,
    is_global_event_loop_set,
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
):
    assert is_global_event_loop_set()

    # Get the gRPC Channel Factory.
    channel_factory_selector = ChannelFactorySelector()
    channel_factory = channel_factory_selector.get_instance()

    # Create all runtimes.
    runtimes: List[Runtime] = []
    for initializer in initializers:
        # Create the data handler.
        if initializer.is_client():
            data_handler = ClientRuntimeDataHandler(
                thread_watcher,
                initializer.remote_data_reader,
                initializer.event_poller,
                is_testing=is_testing,
            )
        elif initializer.is_server():
            data_handler = ServerRuntimeDataHandler(
                initializer.remote_data_reader,
                initializer.event_poller,
                is_testing=is_testing,
            )
        else:
            raise ValueError("Invalid endpoint type!")

        # Create the runtime with this data handler.
        runtimes.append(
            initializer.create(thread_watcher, data_handler, channel_factory)
        )

    # Start them all.
    for runtime in runtimes:
        run_on_event_loop(runtime.start_async)

    return runtimes


def remote_process_main(
    initializers: List[RuntimeFactory[Any, Any]],
    error_queue: MultiprocessQueueSink[Exception],
    *,
    is_testing: bool = False,
):
    # Only needed on linux systems.
    clear_tsercom_event_loop()

    # Initialize the global types.
    thread_watcher = ThreadWatcher()
    create_tsercom_event_loop_from_watcher(thread_watcher)
    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    # Create and start all runtimes.
    runtimes: List[Runtime] = initialize_runtimes(
        thread_watcher, initializers, is_testing=is_testing
    )

    # Call into run_until_error and, on error, stop all runtimes.
    try:
        sink.run_until_exception()
    except Exception as e:
        raise e
    finally:
        for runtime in runtimes:
            run_on_event_loop(runtime.stop)
