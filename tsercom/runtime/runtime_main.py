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
import logging
from functools import partial
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
from tsercom.threading.thread_watcher import ThreadWatcher

from tsercom.runtime.event_poller_adapter import (
    EventToSerializableAnnInstancePollerAdapter,
)
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals # Initialization involves many components.
def initialize_runtimes(
    thread_watcher: ThreadWatcher,
    initializers: List[RuntimeFactory[Any, Any]],
    *,
    is_testing: bool = False,
) -> List[Runtime]:
    """Initializes, configures, and starts a list of Tsercom runtimes.

    This function iterates through the provided `RuntimeFactory` instances (referred
    to as `initializers` in this context, though they are factories that produce
    `RuntimeInitializer` instances internally). For each factory, it:
    1. Retrieves the data reader and event poller.
    2. Creates a gRPC channel factory based on the runtime\'s auth config.
    3. Determines if the runtime is client or server type.
    4. Instantiates the appropriate `RuntimeDataHandler` (`ClientRuntimeDataHandler`
       or `ServerRuntimeDataHandler`), wrapping the event poller with an
       `EventToSerializableAnnInstancePollerAdapter`.
    5. Calls the factory\'s `create` method to get a `Runtime` instance.
    6. Schedules the `runtime_instance.start_async()` coroutine on the global
       Tsercom event loop and adds a callback to report exceptions to the
       `thread_watcher`.

    Args:
        thread_watcher: The `ThreadWatcher` instance responsible for monitoring
            threads and asynchronous tasks associated with these runtimes.
        initializers: A list of `RuntimeFactory` instances. Each factory is
            responsible for creating a specific type of `Runtime`. The `Any`
            type parameters indicate that this function can handle factories
            for various data and event types.
        is_testing: If True, configures components (like data handlers) for
            testing-specific behaviors (e.g., using fake time sync).

    Returns:
        A list of the initialized and started `Runtime` instances.

    Raises:
        ValueError: If a `RuntimeFactory` does not specify a valid service
            type (neither client nor server).
        AssertionError: If the global Tsercom event loop has not been set
            prior to calling this function.
    """
    assert is_global_event_loop_set(), "Global Tsercom event loop must be set."

    channel_factory_selector = ChannelFactorySelector()
    created_runtimes: List[Runtime] = []

    for initializer_factory in initializers:
        # pylint: disable=protected-access # Accessing factory internals for setup
        data_reader = initializer_factory._remote_data_reader()
        event_poller = initializer_factory._event_poller()
        # pylint: enable=protected-access

        auth_config = initializer_factory.auth_config
        channel_factory = channel_factory_selector.create_factory(auth_config)

        # Adapt the raw event poller from the factory to one that yields
        # SerializableAnnotatedInstance, as expected by data handlers.
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
            # This case should ideally be prevented by RuntimeFactory design.
            raise ValueError(
                f"RuntimeFactory {initializer_factory} has an invalid endpoint type."
            )

        runtime_instance = initializer_factory.create(
            thread_watcher,
            data_handler,
            channel_factory,
        )
        created_runtimes.append(runtime_instance)

    # Schedule the start_async method for all created runtimes.
    for runtime in created_runtimes:
        active_loop = get_global_event_loop()
        # runtime.start_async() is a coroutine that needs to be run.
        future: concurrent.futures.Future[Any] = run_on_event_loop(
            runtime.start_async, event_loop=active_loop
        )

        # Add a callback to propagate exceptions from runtime startup to the thread_watcher.
        def _runtime_start_done_callback(
            f: concurrent.futures.Future[Any],
            watcher: ThreadWatcher,
        ) -> None:
            """Callback to handle completion of a runtime's start_async future."""
            try:
                if f.done() and not f.cancelled():
                    exc = f.exception()
                    if isinstance(exc, Exception):
                        watcher.on_exception_seen(exc)
                    elif exc is not None:  # BaseException but not Exception
                        logger.warning(
                            "Runtime start_async future completed with a non-Exception "
                            "BaseException: %s. This will not be reported via ThreadWatcher.",
                            type(exc).__name__,
                        )
            # pylint: disable=broad-exception-caught
            except Exception as e_callback:
                # Log errors within the callback itself to prevent ThreadWatcher issues.
                logger.error(
                    "Error in _runtime_start_done_callback: %s", e_callback
                )

        future.add_done_callback(
            partial(_runtime_start_done_callback, watcher=thread_watcher)
        )

    return created_runtimes


def remote_process_main(
    initializers: List[RuntimeFactory[Any, Any]],
    error_queue: MultiprocessQueueSink[Exception],
    *,
    is_testing: bool = False,
) -> None:
    """Main entry point for a Tsercom runtime operating in a remote process.

    This function is intended to be the target for a new process created by
    `RuntimeManager.start_out_of_process`. It performs the necessary setup for
    a Tsercom runtime environment within this new process, including:
    1. Clearing any existing global event loop state.
    2. Creating a new `ThreadWatcher` and a new Tsercom global event loop.
    3. Setting up a `SplitProcessErrorWatcherSink` to report exceptions from
       this process back to the parent process via the `error_queue`.
    4. Calling `initialize_runtimes` to set up and start the actual runtimes
       specified by the `initializers`.
    5. Running the error sink until an exception occurs or the process is terminated.

    Exceptions from runtimes are caught and put onto the `error_queue` for
    the parent process to observe.

    Args:
        initializers: A list of `RuntimeFactory` instances that define the
            runtimes to be created and started in this remote process.
        error_queue: A `MultiprocessQueueSink` instance used to send any
            exceptions encountered in this process back to the parent process.
        is_testing: If True, configures components (passed to
            `initialize_runtimes`) for testing-specific behaviors.
    """
    # Ensure a clean slate for the event loop in the new process.
    clear_tsercom_event_loop(
        try_stop_loop=False
    )  # try_stop_loop=False as no loop should be running yet.

    thread_watcher = ThreadWatcher()
    # Create and set the global event loop for this process.
    create_tsercom_event_loop_from_watcher(thread_watcher)

    # Error sink to report exceptions from this process to the parent.
    error_sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    active_runtimes: List[Runtime] = []
    captured_exception: Optional[Exception] = None
    try:
        active_runtimes = initialize_runtimes(
            thread_watcher, initializers, is_testing=is_testing
        )
        # This blocks until an exception is caught by the thread_watcher or
        # the error_sink is stopped.
        error_sink.run_until_exception()

    except Exception as e:
        captured_exception = e

    # Ensure cleanup of runtimes and factories on exit.
    logger.info("Remote process shutting down. Stopping runtimes.")

    asyncs = [runtime.stop(captured_exception) for runtime in active_runtimes]
    aggregate_asyncs = asyncio.wait_for(asyncio.gather(*asyncs), timeout=5)
    asyncs_task = get_global_event_loop().create_task(aggregate_asyncs)
    for factory in initializers:
        try:
            # pylint: disable=protected-access
            factory._stop()
        # pylint: disable=broad-exception-caught
        except Exception as e_factory_stop:
            logger.error(
                "Error stopping factory %s: %s", factory, e_factory_stop
            )

    logger.info("Remote process cleanup complete.")

    # Wait for all stop calls to return or time out.
    try:
        asyncs_task.result()
    except Exception as e:  # pylint: disable=broad-exception-caught
        # TODO: Create AggregateException with both if captured_exception is NOT
        # None.
        if captured_exception is None:
            captured_exception = e

    # Handle exception after finally is done.
    if captured_exception is not None:
        if error_queue:
            try:
                error_queue.put_nowait(captured_exception)
            # pylint: disable=broad-exception-caught
            except Exception as q_e:
                logger.error(
                    "Failed to put exception onto error_queue: %s", q_e
                )
        raise captured_exception
