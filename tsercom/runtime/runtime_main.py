"""Core initialization and execution logic for Tsercom runtimes.

This module provides the primary functions for setting up and launching Tsercom
runtimes, both for in-process and out-of-process (remote) execution scenarios.
It orchestrates the creation of necessary components like data handlers,
event pollers, and channel factories based on the provided `RuntimeFactory`
instances.

Key functions:
- `initialize_runtimes`: Sets up and starts a list of runtimes within the
  calling process, using an existing event loop.
- `remote_process_main`: Serves as the main entry point for a new process
  spawned to host Tsercom runtimes, managing its own event loop and error
  propagation back to the parent process.
"""

import asyncio
import concurrent.futures
from functools import partial
import logging
import signal # For signal handling
from threading import Thread, Event as ThreadingEvent # For the command listener thread and stop_event
from typing import (
    Any,
    List,
    Optional,
)

from tsercom.api.split_process.split_process_error_watcher_sink import (
    SplitProcessErrorWatcherSink,
)
from tsercom.runtime.channel_factory_selector import ChannelFactorySelector
from tsercom.runtime.client.client_runtime_data_handler import (
    ClientRuntimeDataHandler,
)
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.server.server_runtime_data_handler import (
    ServerRuntimeDataHandler,
)
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    create_tsercom_event_loop_from_watcher,
    get_global_event_loop,
    is_global_event_loop_set,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource, # Needed for new control_source argument
)
from tsercom.threading.thread_watcher import ThreadWatcher
from .event_poller_adapter import (
    EventToSerializableAnnInstancePollerAdapter,
)
from .runtime_data_handler import RuntimeDataHandler


logger = logging.getLogger(__name__)

# --- Module-level variables for signal handling ---
MODULE_LEVEL_G_THREAD_WATCHER_REF: Optional[ThreadWatcher] = None
MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED: bool = False

class GracefulShutdownCommand(Exception):
    """Custom exception to signal graceful shutdown from the command listener."""
    pass

def handle_module_level_signal(signum, frame):
    """Module-level signal handler to initiate graceful shutdown."""
    global MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED, MODULE_LEVEL_G_THREAD_WATCHER_REF
    signal_name = signal.Signals(signum).name
    logger.info(
        "Signal %s received by module-level handler in remote process. Initiating graceful shutdown.",
        signal_name,
    )
    MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED = True
    if MODULE_LEVEL_G_THREAD_WATCHER_REF:
        # Inject GracefulShutdownCommand to break ThreadWatcher.run_until_exception()
        MODULE_LEVEL_G_THREAD_WATCHER_REF.on_exception_seen(
            GracefulShutdownCommand(f"Signal {signal_name} received, initiating graceful shutdown via ThreadWatcher.")
        )
    else:
        logger.warning("Global thread watcher reference (MODULE_LEVEL_G_THREAD_WATCHER_REF) not set at time of signal %s.", signal_name)


# pylint: disable=too-many-locals # Initialization involves many components.
def initialize_runtimes(
    thread_watcher: ThreadWatcher,
    initializers: List[RuntimeFactory[Any, Any]],
    *,
    is_testing: bool = False,
) -> List[Runtime]:
    """Initializes, configures, and starts a list of Tsercom runtimes."""
    assert is_global_event_loop_set(), "Global Tsercom event loop must be set."

    channel_factory_selector = ChannelFactorySelector()
    created_runtimes: List[Runtime] = []

    for initializer_factory in initializers:
        # pylint: disable=protected-access
        data_reader = initializer_factory._remote_data_reader()
        event_poller = initializer_factory._event_poller()
        # pylint: enable=protected-access

        auth_config = initializer_factory.auth_config
        channel_factory = channel_factory_selector.create_factory(auth_config)

        adapted_event_poller = EventToSerializableAnnInstancePollerAdapter(
            event_poller
        )

        data_handler: RuntimeDataHandler[Any, Any]
        if initializer_factory.is_client():
            data_handler = ClientRuntimeDataHandler(
                thread_watcher=thread_watcher,
                data_reader=data_reader,
                event_source=adapted_event_poller,
                min_send_frequency_seconds=(
                    initializer_factory.min_send_frequency_seconds
                ),
                is_testing=is_testing,
            )
        elif initializer_factory.is_server():
            data_handler = ServerRuntimeDataHandler(
                data_reader=data_reader,
                event_source=adapted_event_poller,
                min_send_frequency_seconds=(
                    initializer_factory.min_send_frequency_seconds
                ),
                is_testing=is_testing,
            )
        else:
            raise ValueError(
                f"RuntimeFactory {initializer_factory} has an invalid endpoint type."
            )

        runtime_instance = initializer_factory.create(
            thread_watcher,
            data_handler,
            channel_factory,
        )
        created_runtimes.append(runtime_instance)

    for runtime in created_runtimes:
        active_loop = get_global_event_loop()
        future: concurrent.futures.Future[Any] = run_on_event_loop(
            runtime.start_async, event_loop=active_loop
        )

        def _runtime_start_done_callback(
            f: concurrent.futures.Future[Any],
            watcher: ThreadWatcher,
        ) -> None:
            try:
                if f.done() and not f.cancelled():
                    exc = f.exception()
                    if isinstance(exc, Exception):
                        watcher.on_exception_seen(exc)
                    elif exc is not None:
                        logger.warning(
                            "Runtime start_async future completed with a non-Exception "
                            "BaseException: %s. This will not be reported via ThreadWatcher.",
                            type(exc).__name__,
                        )
            except Exception as e_callback:
                logger.error(
                    "Error in _runtime_start_done_callback: %s", e_callback
                )

        future.add_done_callback(
            partial(_runtime_start_done_callback, watcher=thread_watcher)
        )
    return created_runtimes


def _control_command_listener_thread(
    control_source: MultiprocessQueueSource[str],
    thread_watcher_ref: ThreadWatcher,
    stop_event: ThreadingEvent
    ):
    """Target for the thread that listens for control commands."""
    logger.info("Control command listener thread started.")
    try:
        while not stop_event.is_set():
            try:
                command = control_source.get_blocking(timeout=0.5)
                if command == "PREPARE_SHUTDOWN":
                    logger.info("PREPARE_SHUTDOWN command received by listener thread.")
                    if thread_watcher_ref:
                        thread_watcher_ref.on_exception_seen(
                            GracefulShutdownCommand("PREPARE_SHUTDOWN command received")
                        )
                    break
                elif command is None:
                    continue
                else:
                    logger.warning(f"Unknown control command received: {command}")
            except Exception as e: # pylint: disable=broad-exception-caught
                if not stop_event.is_set():
                    logger.error(f"Error in control command listener thread: {e}")
                break
    finally:
        logger.info("Control command listener thread finishing.")


async def _perform_runtime_cleanup(
    active_runtimes: List[Runtime],
    initializers: List[RuntimeFactory[Any, Any]],
    captured_exception: Optional[Exception], # Pass as a mutable container (e.g., a list) if modification is needed
    logger_ref: logging.Logger # Pass logger explicitly
) -> Optional[Exception]: # Return the (potentially updated) captured_exception
    """Helper async function to perform cleanup of runtimes and factories."""
    # This function encapsulates the logic originally in do_cleanup.
    # It's made a top-level async function for easier patching in tests.
    # We need to decide if captured_exception should be modified directly (if mutable) or returned.
    # Returning it might be cleaner.

    current_captured_exception = captured_exception # Work with a local copy

    logger_ref.info("Executing _perform_runtime_cleanup() in remote process.")

    runtime_stop_tasks = [
        runtime.stop(current_captured_exception) for runtime in active_runtimes
    ]
    if runtime_stop_tasks:
        logger_ref.info(f"Attempting to stop {len(runtime_stop_tasks)} runtimes...")
        results = await asyncio.wait_for(
            asyncio.gather(*runtime_stop_tasks, return_exceptions=True),
            timeout=6.0  # Increased timeout
        )
        logger_ref.info("Runtime stop sequence completed.")
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger_ref.error(f"Error stopping runtime {active_runtimes[i]}: {res!r}")
                if current_captured_exception is None: # Capture the first error if none yet
                    current_captured_exception = res
    else:
        logger_ref.info("No active runtimes to stop.")

    logger_ref.info("Stopping factories in remote process.")
    for factory in initializers:
        try:
            factory._stop()
        except Exception as e_factory_stop:
            logger_ref.error(
                "Error stopping factory %s: %s", factory, e_factory_stop
            )
            if current_captured_exception is None: current_captured_exception = e_factory_stop
    logger_ref.info("Factories stopped in remote process.")
    return current_captured_exception


# pylint: disable=too-many-statements, too-many-branches
def remote_process_main(
    initializers: List[RuntimeFactory[Any, Any]],
    error_queue: MultiprocessQueueSink[Exception],
    control_source: MultiprocessQueueSource[str],
    ack_sink: MultiprocessQueueSink[str],
    *,
    is_testing: bool = False,
) -> None:
    """Main entry point for a Tsercom runtime operating in a remote process."""
    global MODULE_LEVEL_G_THREAD_WATCHER_REF, MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED

    clear_tsercom_event_loop(try_stop_loop=False)

    thread_watcher = ThreadWatcher()
    MODULE_LEVEL_G_THREAD_WATCHER_REF = thread_watcher

    signal.signal(signal.SIGTERM, handle_module_level_signal)
    signal.signal(signal.SIGINT, handle_module_level_signal)

    loop = create_tsercom_event_loop_from_watcher(thread_watcher)
    error_sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    active_runtimes: List[Runtime] = []
    captured_exception: Optional[Exception] = None
    shutdown_initiated_gracefully = False # Renamed from _shutdown_signal_received for clarity in this scope

    control_thread_stop_event = ThreadingEvent()
    cmd_listener_thread = Thread(
        target=_control_command_listener_thread,
        args=(control_source, thread_watcher, control_thread_stop_event),
        daemon=True,
        name="ControlCmdListener"
    )
    cmd_listener_thread.start()

    try:
        logger.info("Remote process initializing runtimes.")
        active_runtimes = initialize_runtimes(
            thread_watcher, initializers, is_testing=is_testing
        )
        logger.info("Remote process runtimes initialized. Waiting for exceptions or signals.")
        error_sink.run_until_exception()

    except GracefulShutdownCommand as gsc:
        logger.info(f"Remote process received GracefulShutdownCommand: {gsc}. Proceeding to cleanup.")
        shutdown_initiated_gracefully = True
    except SystemExit as se:
        logger.info(f"Remote process received SystemExit: {se}. Proceeding to graceful shutdown.")
        shutdown_initiated_gracefully = True
        if not MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED : # If SystemExit wasn't from our signal handler
            captured_exception = se
    except Exception as e:
        logger.error(f"Remote process caught unhandled exception: {e}", exc_info=True)
        captured_exception = e
    finally:
        logger.info("Remote process entering finally block for cleanup.")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        control_thread_stop_event.set()
        if cmd_listener_thread.is_alive():
             cmd_listener_thread.join(timeout=2.0)
        if cmd_listener_thread.is_alive():
            logger.warning("Control command listener thread did not exit cleanly.")

        # Call the refactored cleanup function
        if loop and not loop.is_closed():
            cleanup_coro = _perform_runtime_cleanup(active_runtimes, initializers, captured_exception, logger)
            if loop.is_running():
                logger.info("Remote process event loop is running. Scheduling cleanup task.")
                try:
                    cleanup_future = asyncio.run_coroutine_threadsafe(cleanup_coro, loop)
                    # The future's result is the potentially updated captured_exception
                    updated_exception = cleanup_future.result(timeout=7.5)
                    if captured_exception is None and updated_exception is not None:
                        captured_exception = updated_exception
                    logger.info("_perform_runtime_cleanup task completed via run_coroutine_threadsafe.")
                except Exception as e_cleanup_run:
                     logger.error(f"Exception running _perform_runtime_cleanup task via run_coroutine_threadsafe: {e_cleanup_run}")
                     if captured_exception is None: captured_exception = e_cleanup_run
            else:
                logger.info("Remote process event loop is not running but open. Running cleanup with run_until_complete.")
                try:
                    updated_exception = loop.run_until_complete(cleanup_coro)
                    if captured_exception is None and updated_exception is not None:
                        captured_exception = updated_exception
                    logger.info("_perform_runtime_cleanup task completed via run_until_complete.")
                except Exception as e_cleanup_run:
                    logger.error(f"Exception running _perform_runtime_cleanup task via run_until_complete: {e_cleanup_run}")
                    if captured_exception is None: captured_exception = e_cleanup_run
        else:
            logger.warning("Remote process event loop was not available or closed when trying to execute _perform_runtime_cleanup.")

        ack_message = "SHUTDOWN_ERROR" # Default to error
        if shutdown_initiated_gracefully and captured_exception is None :
            ack_message = "SHUTDOWN_READY"
        elif isinstance(captured_exception, GracefulShutdownCommand): # If it was a graceful command, and no other error occurred
             ack_message = "SHUTDOWN_READY"
        elif isinstance(captured_exception, SystemExit) and MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED: # If SystemExit from our signal
            ack_message = "SHUTDOWN_READY"


        logger.info(f"Sending acknowledgment: {ack_message}")
        try:
            # Ensure ack_sink is not None before using
            if ack_sink:
                ack_sink.put_blocking(ack_message, timeout=1.0)
        except Exception as e_ack:
            logger.error(f"Failed to send shutdown acknowledgment: {e_ack}")
        finally:
            try:
                if ack_sink:
                    ack_sink.close()
                    ack_sink.join_thread(timeout=1.0)
            except Exception as e_ack_close:
                logger.error(f"Error closing ack_sink: {e_ack_close}")

        try:
            if control_source:
                control_source.close()
                control_source.join_thread(timeout=1.0)
        except Exception as e_ctrl_close:
            logger.error(f"Error closing control_source: {e_ctrl_close}")

        logger.info("Remote process cleanup attempt in finally block finished.")
        clear_tsercom_event_loop(try_stop_loop=True)
        MODULE_LEVEL_G_THREAD_WATCHER_REF = None

    if captured_exception and not (isinstance(captured_exception, GracefulShutdownCommand) or (isinstance(captured_exception, SystemExit) and MODULE_LEVEL_SHUTDOWN_SIGNAL_RECEIVED)):
        logger.info(f"Remote process propagating captured exception: {type(captured_exception).__name__}: {captured_exception}")
        if error_queue:
            try:
                error_queue.put_nowait(captured_exception)
            except Exception as q_e:
                logger.error(
                    "Failed to put exception onto error_queue during final propagation: %s", q_e
                )

    logger.info("Remote process_main finished.")
