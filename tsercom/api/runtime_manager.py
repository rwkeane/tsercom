"""Manages the creation and lifecycle of Tsercom runtimes."""

from asyncio import AbstractEventLoop
from concurrent.futures import Future
from functools import partial
from multiprocessing import Process
from typing import Any, Generic, List, TypeVar, Optional # Added Optional

from tsercom.api.initialization_pair import InitializationPair
from tsercom.api.local_process.local_runtime_factory_factory import (
    LocalRuntimeFactoryFactory,
)
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer

# Imports for runtime_main are moved into methods (start_in_process, start_out_of_process)
# to break potential circular dependencies between manager and main execution modules.
from tsercom.threading.aio.aio_utils import get_running_loop_or_none
from tsercom.threading.aio.global_event_loop import (
    create_tsercom_event_loop_from_watcher,
    set_tsercom_event_loop,
)
from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker

# Type variables for generic RuntimeHandle and related classes.
TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class RuntimeManager(ErrorWatcher):
    """Manages the lifecycle of Tsercom runtimes.

    This class is responsible for creating runtimes from `RuntimeInitializer`
    instances. It handles the complexities of starting runtimes either in the
    current process or in a separate process, along with associated error
    monitoring and propagation.
    """

    def __init__(self, *, is_testing: bool = False) -> None:
        """Initializes the RuntimeManager.

        Args:
            is_testing: If True, configures some operations for testing purposes,
                        such as making out-of-process runtimes daemonic.
        """
        super().__init__(is_testing=False)

        # Stores pairs of Futures and their corresponding RuntimeInitializers.
        self.__initializers: list[InitializationPair[Any, Any]] = []
        # Tracks if the manager has started processing runtimes.
        self.__has_started: IsRunningTracker = IsRunningTracker()

        # Manages threads created by this manager.
        self.__thread_watcher: ThreadWatcher = ThreadWatcher()
        # Watches for errors, either from local threads or a remote process.
        self.__error_watcher: Optional[ErrorWatcher] = None
        # Holds the remote process instance if started out-of-process.
        self.__process: Optional[Process] = None

        self.__is_testing: bool = is_testing

    @property
    def has_started(self) -> bool:
        """Indicates whether this manager instance is currently running.

        This method is thread-safe.

        Returns:
            True if the manager has started, False otherwise.
        """
        return self.__has_started.get()

    def register_runtime_initializer(
        self, runtime_initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> Future[RuntimeHandle[TDataType, TEventType]]:
        """Registers a RuntimeInitializer to be managed.

        This method must be called before any of the start methods
        (`start_in_process`, `start_out_of_process`) are invoked.

        Args:
            runtime_initializer: The initializer for the runtime to be created.

        Returns:
            A Future that will be populated with the RuntimeHandle once the
            runtime is initialized.
        
        Raises:
            RuntimeError: If called after the manager has started.
        """
        # Ensure initializers are registered only before starting.
        if self.has_started:
            raise RuntimeError("Cannot register runtime initializer after the manager has started.")

        handle_future = Future[RuntimeHandle[TDataType, TEventType]]()
        pair = InitializationPair[TDataType, TEventType](
            handle_future, runtime_initializer
        )
        self.__initializers.append(pair)

        return handle_future

    async def start_in_process_async(
        self,
    ) -> List[RuntimeHandle[Any, Any]]:
        """Creates and starts all registered runtimes in the current process.

        This method calls the synchronous `start_in_process` method, providing
        it with the currently running asyncio event loop. Tsercom operations
        will run on this event loop.

        The primary mechanism for retrieving `RuntimeHandle` instances is through
        the `Future` objects returned by `register_runtime_initializer`.

        Returns:
            A list containing `RuntimeHandle` instances for any runtimes whose
            initialization `Future` had already completed (i.e., `future.done()` is true)
            at the time of this call. This is checked by attempting to retrieve the
            result with a zero timeout (`future.result(timeout=0)`).
            This list may be empty or incomplete if runtime initializations are
            still pending. The primary method for obtaining all `RuntimeHandle`
            instances remains the `Future` objects returned by
            `register_runtime_initializer`.

        Raises:
            RuntimeError: If no event loop is running when this method is called.
            Exception: If `future.result(timeout=0)` raises an exception because
                       the future completed with an exception (e.g., `CancelledError`
                       or any exception set on the future by the task it was awaiting).
        """
        running_loop = get_running_loop_or_none()
        
        if running_loop is None:
            raise RuntimeError("Could not determine the current running event loop for start_in_process_async.")
            
        # The `start_in_process` method doesn't return handles directly.
        # Handles are obtained via the Futures.
        self.start_in_process(running_loop)
        
        # Collect handles from futures for convenience, if desired by original intent.
        # However, the current start_in_process doesn't facilitate this directly.
        # For now, aligning with start_in_process's void return.
        # If handles were to be returned, it would look like:
        # return [pair.handle_future.result() for pair in self.__initializers]
        return [pair.handle_future.result(timeout=0) for pair in self.__initializers if pair.handle_future.done()]

    def start_in_process(
        self,
        runtime_event_loop: AbstractEventLoop,
    ) -> None:
        """Creates and starts all registered runtimes in the current process.

        Tsercom operations are run on the provided `runtime_event_loop`.
        RuntimeHandles are provided via the Futures returned by
        `register_runtime_initializer`.

        Args:
            runtime_event_loop: The asyncio event loop on which Tsercom
                                operations will run.
        
        Raises:
            RuntimeError: If called after the manager has started.
        """
        if self.has_started:
            raise RuntimeError("RuntimeManager has already been started.")
        self.__has_started.start()

        # Basic initialization for in-process execution.
        set_tsercom_event_loop(runtime_event_loop)
        self.__error_watcher = self.__thread_watcher # Local errors managed by ThreadWatcher.

        # Create factories for local process runtimes.
        thread_pool = (
            self.__thread_watcher.create_tracked_thread_pool_executor(
                max_workers=1 # Single worker for sequential factory creation.
            )
        )
        factory_factory = LocalRuntimeFactoryFactory(thread_pool)
        factories = self.__create_factories(factory_factory)

        # Import and run the main initialization sequence for runtimes.
        # Import is deferred to avoid circular dependencies.
        from tsercom.runtime.runtime_main import (
            initialize_runtimes,
        )

        initialize_runtimes(
            self.__thread_watcher, factories, is_testing=self.__is_testing
        )

    def start_out_of_process(self, start_as_daemon: bool = False) -> None:
        """Creates and starts registered runtimes in a new, separate process.

        Creates runtimes from all registered RuntimeInitializer instances, and
        then starts each created instance in a new process separate from the
        current process. Commands to such runtimes are forwarded from the
        returned Runtime instances, and data received from it can be accessed
        through the RemoteDataAggregator instance available in it.

        Args:
            start_as_daemon: If True, the new process will be a daemon process.
                             Daemonic processes are typically used for background
                             tasks and are terminated when the main program exits.
        
        Raises:
            RuntimeError: If called after the manager has started.
        """
        if self.has_started:
            raise RuntimeError("RuntimeManager has already been started.")
            
        self.__has_started.start()

        # Set up a minimal local Tsercom event loop, primarily for utilities.
        create_tsercom_event_loop_from_watcher(self.__thread_watcher)

        # Prepare for inter-process error communication.
        error_sink, error_source = create_multiprocess_queues()
        self.__error_watcher = SplitProcessErrorWatcherSource(
            self.__thread_watcher, error_source
        )
        self.__error_watcher.start() # Start listening for errors from the remote process.

        # Create factories for split-process runtimes.
        thread_pool = (
            self.__thread_watcher.create_tracked_thread_pool_executor(
                max_workers=1 # Single worker for sequential factory creation.
            )
        )
        factory_factory = SplitRuntimeFactoryFactory(
            thread_pool, self.__thread_watcher
        )
        factories = self.__create_factories(factory_factory)

        # Import and prepare the main function for the remote process.
        # Import is deferred to avoid circular dependencies.
        from tsercom.runtime.runtime_main import (
            remote_process_main,
        )

        # Configure and start the new process.
        self.__process = Process(
            target=partial(
                remote_process_main,
                factories,
                error_sink, # Provide the error queue to the remote process.
                is_testing=self.__is_testing,
            ),
            # Test processes or explicit daemons are set as daemonic.
            daemon=start_as_daemon or self.__is_testing,
        )
        self.__process.start()

    def run_until_exception(self) -> None:
        """Blocks execution until an exception is raised by any managed runtime.
        
        Runs the current thread until an exception as been raised, throwing the
        exception upon receipt. This method is thread-safe and can be called
        from any thread.

        This method is thread-safe. It re-raises the caught exception in the
        calling thread.

        Raises:
            Any exception propagated from the managed runtimes.
            RuntimeError: If the manager hasn't started or the error watcher isn't set.
        """
        if not self.has_started:
            # Added this check for consistency, as __error_watcher depends on has_started
            raise RuntimeError("RuntimeManager has not been started.")
        if self.__error_watcher is None:
            raise RuntimeError("Error watcher is not available. Ensure the RuntimeManager has been properly started.")

        # Delegate to ThreadWatcher to wait for and propagate exceptions.
        self.__thread_watcher.run_until_exception()

    def check_for_exception(self) -> None:
        """Checks for and re-raises any exceptions from managed runtimes.

        If an exception has been caught from any runtime, this method re-raises
        it in the calling thread. Otherwise, it does nothing. This method is
        thread-safe.

        Raises:
            Any exception propagated from the managed runtimes.
            RuntimeError: If the manager has started but the error watcher isn't set.
        """
        if not self.has_started:
            return # No exceptions to check if not started.

        if self.__error_watcher is None:
            # This implies it wasn't started correctly or state is corrupted.
            raise RuntimeError("Error watcher is not available. Ensure the RuntimeManager has been properly started.")

        # Delegate to ThreadWatcher to check and propagate exceptions.
        self.__thread_watcher.check_for_exception()

    def __create_factories(
        self, factory_factory: RuntimeFactoryFactory[Any, Any]
    ) -> List[RuntimeFactory[Any, Any]]:
        """Creates runtime factories using the provided factory_factory.

        Iterates through all registered `InitializationPair`s and uses the
        `factory_factory` to create a `RuntimeFactory` for each. The associated
        `RuntimeHandle` Future within each pair is populated by a
        `RuntimeFuturePopulator` client.

        Args:
            factory_factory: The `RuntimeFactoryFactory` (e.g., local or split)
                             to use for creating individual `RuntimeFactory` instances.

        Returns:
            A list of created `RuntimeFactory` instances.
        """
        results: List[RuntimeFactory[Any, Any]] = []
        for pair in self.__initializers:
            # The RuntimeFuturePopulator acts as the client to receive the handle.
            client = RuntimeFuturePopulator[Any, Any](pair.handle_future)
            factory = factory_factory.create_factory(
                client,
                pair.initializer,
            )
            results.append(factory)
        return results


class RuntimeFuturePopulator(
    RuntimeFactoryFactory.Client[TDataType, TEventType], Generic[TDataType, TEventType]
):
    """A client that populates a Future with a RuntimeHandle when ready.

    This class implements the `RuntimeFactoryFactory.Client` interface. Its sole
    purpose is to set the result of a provided `Future` when the
    `_on_handle_ready` callback is invoked with the `RuntimeHandle`.
    """
    def __init__(self, future: Future[RuntimeHandle[TDataType, TEventType]]) -> None:
        """Initializes the RuntimeFuturePopulator.

        Args:
            future: The `Future` object that will be populated with the
                    `RuntimeHandle`.
        """
        self.__future: Future[RuntimeHandle[TDataType, TEventType]] = future

    def _on_handle_ready(self, handle: RuntimeHandle[TDataType, TEventType]) -> None:
        """Callback invoked when the RuntimeHandle is ready.

        This method sets the provided `handle` as the result of the `Future`
        stored during initialization.

        Args:
            handle: The `RuntimeHandle` that has been successfully created.
        """
        # Set the RuntimeHandle on the future, notifying waiters.
        self.__future.set_result(handle)
