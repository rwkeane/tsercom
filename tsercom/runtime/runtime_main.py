"""Main entry points and initialization logic for Tsercom runtimes."""

import sys # Added import
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
    reset_global_event_loop_state_for_child_process, # New import
    # clear_tsercom_event_loop is not explicitly used in the new remote_process_main start
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher
import concurrent.futures # Added for type hint in callback


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
    print(f"initialize_runtimes: Initializing runtimes list (id={id(runtimes)}).", flush=True) # Log list creation
    for factory_idx, initializer_factory in enumerate(initializers): # Add enumerate for logging
        print(f"initialize_runtimes: Loop 1: Processing factory_idx={factory_idx}, factory={initializer_factory}", flush=True)
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
        
        print(f"initialize_runtimes: Loop 1: Created data_handler (id={id(data_handler)}) for factory {factory_idx}", flush=True)
        runtime_instance = initializer_factory.create(
            thread_watcher, data_handler, channel_factory
        )
        print(f"initialize_runtimes: Loop 1: Created runtime_instance (id={id(runtime_instance)}, type={type(runtime_instance)}) for factory {factory_idx}", flush=True)
        runtimes.append(runtime_instance)
        print(f"initialize_runtimes: Loop 1: Appended instance. Runtimes list now (id={id(runtimes)}): {runtimes}", flush=True)

    # --- New Detailed Prints ---
    print(f"initialize_runtimes: After Loop 1 (instance creation), runtimes list (id={id(runtimes)}): {runtimes}", flush=True)
    print(f"initialize_runtimes: Length of runtimes list: {len(runtimes)}", flush=True)
    for i, rt_obj in enumerate(runtimes):
        print(f"initialize_runtimes: Runtime object {i} in list: id={id(rt_obj)}, type={type(rt_obj)}", flush=True)
    # --- End New Detailed Prints ---

    # THE FOLLOWING LOOP AND CALLBACK LOGIC IS REMOVED:
    # print(f"initialize_runtimes: About to Loop 2 (schedule start_async). Runtimes list (id={id(runtimes)}): {runtimes}", flush=True)
    # for runtime_idx, runtime in enumerate(runtimes):
    #     print(f"initialize_runtimes: Loop 2: Processing runtime_idx={runtime_idx}, runtime={runtime} (id={id(runtime)})", flush=True)
    #     active_loop = get_global_event_loop()
    #     coro_to_run = runtime.start_async
    #     future = run_on_event_loop(coro_to_run, event_loop=active_loop)
    #     print(f"initialize_runtimes: Loop 2: Scheduled start_async for runtime_idx={runtime_idx}. Future: {future}", flush=True)
    #     def _runtime_start_done_callback(...): ...
    #     future.add_done_callback(...)
    # print(f"initialize_runtimes: Completed Loop 2.", flush=True)
    
    print("initialize_runtimes: Runtimes created. Explicit start_async loop removed. Actual start is delegated to RuntimeCommand.kStart mechanism.", flush=True)
    return runtimes


def remote_process_main(
    initializers: List[RuntimeFactory[Any, Any]],
    error_queue: MultiprocessQueueSink[Exception],
    *,
    is_testing: bool = False,
) -> None:
    reset_global_event_loop_state_for_child_process() # New call

    print("remote_process_main: Entered function and reset global loop state.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    """Main function for a Tsercom runtime operating in a remote process.

    Sets up event loop, error handling via a queue, initializes runtimes,
    and runs until an exception occurs.

    Args:
        initializers: List of `RuntimeFactory` instances.
        error_queue: A `MultiprocessQueueSink` to send exceptions back to the parent.
        is_testing: Boolean flag for testing mode.
    """
    # The initial clear_tsercom_event_loop() is removed as reset_global_event_loop_state_for_child_process handles the state.
    # If a clear is still needed for other reasons, it could be added after reset, but typically reset is enough for a child process.

    thread_watcher = ThreadWatcher()
    print("remote_process_main: About to call create_tsercom_event_loop_from_watcher.", flush=True)
    create_tsercom_event_loop_from_watcher(thread_watcher)
    print("remote_process_main: Returned from create_tsercom_event_loop_from_watcher.", flush=True)

    # It's important to ensure error_queue is valid here.
    # If error_queue is None or problematic, SplitProcessErrorWatcherSink might fail.
    # For now, assume error_queue is correctly passed from RuntimeManager.
    print("remote_process_main: About to create SplitProcessErrorWatcherSink.", flush=True)
    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)
    print("remote_process_main: Created SplitProcessErrorWatcherSink.", flush=True)

    runtimes: List[Runtime] = []
    try:
        print("remote_process_main: About to initialize runtimes.", flush=True)
        runtimes = initialize_runtimes( 
            thread_watcher, initializers, is_testing=is_testing
        )
        print("remote_process_main: Returned from initialize_runtimes.", flush=True)
        print("remote_process_main: Starting sink.run_until_exception()...", flush=True)
        sink.run_until_exception()
        print("remote_process_main: sink.run_until_exception() completed.", flush=True)

    except Exception as e:
        print(f"remote_process_main: Exception caught in try block: {e}", flush=True)
        # This block handles exceptions from initialize_runtimes or sink.run_until_exception()
        # If error_queue is available, put the exception.
        # It will now be covered by the finally block.
        if error_queue: # Ensure error_queue is checked before use
            error_queue.put_nowait(e)
        # Re-raise the exception to ensure the process exits with an error status if needed.
        # Python's try-except-finally ensures 'finally' runs before 'raise' in except.
        raise # This re-raises the caught exception 'e'
    finally:
        print("remote_process_main: Entering finally block for cleanup.", flush=True)
        for runtime_idx, runtime in enumerate(runtimes):
            try:
                print(f"remote_process_main (finally - runtime {runtime_idx}): Stopping runtime {runtime}", flush=True)
                run_on_event_loop(partial(runtime.stop, None))
            except Exception as e_rt_stop:
                print(f"remote_process_main (finally - runtime {runtime_idx}): Error stopping runtime {runtime}: {e_rt_stop}", flush=True)

        for factory_idx, factory in enumerate(initializers):
            try:
                if hasattr(factory, 'stop_command_source') and callable(getattr(factory, 'stop_command_source')):
                    print(f"remote_process_main (finally - factory {factory_idx}): Stopping command source for factory {factory}", flush=True)
                    factory.stop_command_source() # type: ignore
                else:
                    print(f"remote_process_main (finally - factory {factory_idx}): Factory {factory} does not have stop_command_source.", flush=True)
            except Exception as e_cs_stop:
                print(f"remote_process_main (finally - factory {factory_idx}): Error stopping command source for factory {factory}: {e_cs_stop}", flush=True)
        print("remote_process_main: Finally block completed.", flush=True)
    print("remote_process_main: Exiting.", flush=True)
