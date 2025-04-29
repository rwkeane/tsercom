from typing import Any, List, Literal
from tsercom.api.runtime_initializer import RuntimeInitializer
from tsercom.runtime.channel_factory_selector import ChannelFactorySelector
from tsercom.runtime.client.client_runtime_data_handler import ClientRuntimeDataHandler
from tsercom.api.split_process.split_process_error_watcher_sink import SplitProcessErrorWatcherSink
from tsercom.runtime.server.server_runtime_data_handler import ServerRuntimeDataHandler
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.aio.global_event_loop import create_tsercom_event_loop_from_watcher, is_global_event_loop_set
from tsercom.threading.multiprocess.multiprocess_queue_sink import MultiprocessQueueSink
from tsercom.threading.thread_watcher import ThreadWatcher


def initialize_runtimes(thread_watcher : ThreadWatcher,
                        endpoint_type : Literal["client", "server"],
                        initializers : List[RuntimeInitializer[Any, Any]]):
    assert is_global_event_loop_set()

    # Get the gRPC Channel Factory.
    factory_selector = ChannelFactorySelector()
    factory = factory_selector.get_instance()

    # Create the data handler.
    if endpoint_type == "client":
        data_handler = ClientRuntimeDataHandler(thread_watcher)
    elif endpoint_type == "server":
        data_handler = ServerRuntimeDataHandler()
    else:
        raise ValueError(f"Invalid endpoint type: {endpoint_type}")
    
    # Create all runtimes.
    runtimes = [initializer.create(thread_watcher, data_handler, factory) for initializer in initializers]
    
    # Start them all.
    for runtime in runtimes:
        run_on_event_loop(runtime.start_async)
    
    return runtimes

def remote_process_main(endpoint_type : Literal["client", "server"],
                        initializers : List[RuntimeInitializer[Any, Any]],
                        error_queue: MultiprocessQueueSink[Exception],):
    # Initialize the global types.
    thread_watcher = ThreadWatcher()
    create_tsercom_event_loop_from_watcher(thread_watcher)
    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    # Create and start all runtimes.
    runtimes = initialize_runtimes(thread_watcher, endpoint_type, initializers)

    # Call into run_until_error and, on error, stop all runtimes.
    try:
        sink.run_until_exception()
    except Exception as e:
        raise e
    finally:
        for runtime in runtimes:
            run_on_event_loop(runtime.stop)
