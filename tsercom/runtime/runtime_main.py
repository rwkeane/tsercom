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
    initializers: List[RuntimeFactory[Any, Any]], # These are RemoteRuntimeFactory instances in remote process
    *,
    is_testing: bool = False,
):
    assert is_global_event_loop_set()
    print(f"DEBUG: [runtime_main.initialize_runtimes] Starting. Number of initializers: {len(initializers)}")

    # Get the gRPC Channel Factory.
    channel_factory_selector = ChannelFactorySelector()
    channel_factory = channel_factory_selector.get_instance()

    # Create all runtimes.
    runtimes: List[Runtime] = []
    for initializer_factory in initializers: # Renamed to avoid confusion with RuntimeInitializer
        print(f"DEBUG: [runtime_main.initialize_runtimes] Processing initializer_factory: {initializer_factory}, id(initializer_factory): {id(initializer_factory)}")
        # Create the data handler.
        # These calls invoke methods on the RemoteRuntimeFactory instance in the remote process
        # Changed to call underlying methods directly
        data_reader = initializer_factory._remote_data_reader() 
        print(f"DEBUG: [runtime_main.initialize_runtimes] After data_reader = initializer_factory._remote_data_reader(). id(initializer_factory): {id(initializer_factory)}, data_reader: {data_reader}, id(data_reader): {id(data_reader)}")
        
        event_poller = initializer_factory._event_poller()
        print(f"DEBUG: [runtime_main.initialize_runtimes] After event_poller = initializer_factory._event_poller(). id(initializer_factory): {id(initializer_factory)}, event_poller: {event_poller}, id(event_poller): {id(event_poller)}")

        if initializer_factory.is_client():
            print(f"DEBUG: [runtime_main.initialize_runtimes] Creating ClientRuntimeDataHandler. data_reader arg: {data_reader}, id(data_reader): {id(data_reader)}")
            data_handler = ClientRuntimeDataHandler(
                thread_watcher,
                data_reader, # Pass the result of initializer._remote_data_reader()
                event_poller,  # Pass the result of initializer._event_poller()
                is_testing=is_testing,
            )
        elif initializer_factory.is_server():
            print(f"DEBUG: [runtime_main.initialize_runtimes] Creating ServerRuntimeDataHandler. data_reader arg: {data_reader}, id(data_reader): {id(data_reader)}")
            data_handler = ServerRuntimeDataHandler(
                data_reader, # Pass the result of initializer._remote_data_reader()
                event_poller,  # Pass the result of initializer._event_poller()
                is_testing=is_testing,
            )
            print(f"DEBUG: [runtime_main.initialize_runtimes] ServerRuntimeDataHandler created: {data_handler}, id(data_handler): {id(data_handler)}")
        else:
            raise ValueError("Invalid endpoint type!")

        # Create the runtime with this data handler.
        # This calls RemoteRuntimeFactory.create()
        print(f"DEBUG: [runtime_main.initialize_runtimes] Calling initializer_factory.create(). id(initializer_factory): {id(initializer_factory)}")
        runtime_instance = initializer_factory.create(thread_watcher, data_handler, channel_factory)
        runtimes.append(runtime_instance)
        print(f"DEBUG: [runtime_main.initialize_runtimes] Runtime instance appended: {runtime_instance}")


    # Start them all.
    for runtime in runtimes:
        run_on_event_loop(runtime.start_async)
    
    print(f"DEBUG: [runtime_main.initialize_runtimes] Finished.")
    return runtimes


def remote_process_main(
    initializers: List[RuntimeFactory[Any, Any]], # These are RemoteRuntimeFactory instances
    error_queue: MultiprocessQueueSink[Exception],
    *,
    is_testing: bool = False,
):
    # Only needed on linux systems.
    clear_tsercom_event_loop()
    print(f"DEBUG: [runtime_main.remote_process_main] Starting remote process. Number of initializers: {len(initializers)}")
    if initializers:
        print(f"DEBUG: [runtime_main.remote_process_main] First initializer_factory id: {id(initializers[0])}")


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
        print(f"DEBUG: [runtime_main.remote_process_main] Remote process finished cleanup.")
