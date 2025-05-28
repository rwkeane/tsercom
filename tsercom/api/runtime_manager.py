"""Manages the creation and lifecycle of Tsercom runtimes."""

import sys  # Added for stderr printing
from asyncio import AbstractEventLoop
from concurrent.futures import Future
from functools import partial
from multiprocessing import Process  # Keep for type hinting if necessary
from typing import Any, Generic, List, TypeVar, Optional

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
    SplitProcessErrorWatcherSource,  # Keep for type hinting if necessary
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from .runtime_manager_helpers import (
    ProcessCreator,
    SplitErrorWatcherSourceFactory,
)


# Imports for runtime_main are moved into methods (start_in_process, start_out_of_process)
# to break potential circular dependencies between manager and main execution modules.
from tsercom.threading.aio.aio_utils import get_running_loop_or_none
from tsercom.threading.aio.global_event_loop import (
    create_tsercom_event_loop_from_watcher,
    set_tsercom_event_loop,
    clear_tsercom_event_loop,  # Added
    get_global_event_loop,  # Added
    is_global_event_loop_set,  # Added
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

    def __init__(
        self,
        *,
        is_testing: bool = False,
        thread_watcher: Optional[ThreadWatcher] = None,
        local_runtime_factory_factory: Optional[
            LocalRuntimeFactoryFactory
        ] = None,
        split_runtime_factory_factory: Optional[
            SplitRuntimeFactoryFactory
        ] = None,
        process_creator: Optional[ProcessCreator] = None,
        split_error_watcher_source_factory: Optional[
            SplitErrorWatcherSourceFactory
        ] = None,
    ) -> None:
        """Initializes the RuntimeManager.

        Args:
            is_testing: If True, configures some operations for testing purposes,
                        such as making out-of-process runtimes daemonic.
            thread_watcher: Optional ThreadWatcher instance.
            local_runtime_factory_factory: Optional factory for local runtimes.
            split_runtime_factory_factory: Optional factory for split-process runtimes.
            process_creator: Optional helper to create processes.
            split_error_watcher_source_factory: Optional factory for SplitProcessErrorWatcherSource.
        """
        super().__init__()

        self.__is_testing: bool = is_testing
        self.__thread_watcher: ThreadWatcher = (
            thread_watcher if thread_watcher is not None else ThreadWatcher()
        )
        self.__process_creator: ProcessCreator = (
            process_creator
            if process_creator is not None
            else ProcessCreator()
        )
        self.__split_error_watcher_source_factory: (
            SplitErrorWatcherSourceFactory
        ) = (
            split_error_watcher_source_factory
            if split_error_watcher_source_factory is not None
            else SplitErrorWatcherSourceFactory()
        )

        if local_runtime_factory_factory is not None:
            self.__local_runtime_factory_factory: (
                LocalRuntimeFactoryFactory
            ) = local_runtime_factory_factory
        else:
            default_local_factory_thread_pool = (
                self.__thread_watcher.create_tracked_thread_pool_executor(
                    max_workers=1
                )
            )
            self.__local_runtime_factory_factory: (
                LocalRuntimeFactoryFactory
            ) = LocalRuntimeFactoryFactory(default_local_factory_thread_pool)

        if split_runtime_factory_factory is not None:
            self.__split_runtime_factory_factory: (
                SplitRuntimeFactoryFactory
            ) = split_runtime_factory_factory
        else:
            default_split_factory_thread_pool = (
                self.__thread_watcher.create_tracked_thread_pool_executor(
                    max_workers=1
                )
            )
            self.__split_runtime_factory_factory: (
                SplitRuntimeFactoryFactory
            ) = SplitRuntimeFactoryFactory(
                default_split_factory_thread_pool, self.__thread_watcher
            )

        self.__initializers: list[InitializationPair[Any, Any]] = []
        self.__has_started: IsRunningTracker = IsRunningTracker()
        self.__error_watcher: Optional[ErrorWatcher] = None
        self.__process: Optional[Process] = (
            None  # Process type hint from multiprocessing
        )

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
            raise RuntimeError(
                "Cannot register runtime initializer after the manager has started."
            )

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
            raise RuntimeError(
                "Could not determine the current running event loop for start_in_process_async."
            )

        # The `start_in_process` method doesn't return handles directly.
        # Handles are obtained via the Futures.
        self.start_in_process(running_loop)

        # Collect handles from futures for convenience, if desired by original intent.
        # However, the current start_in_process doesn't facilitate this directly.
        # For now, aligning with start_in_process's void return.
        # If handles were to be returned, it would look like:
        # return [pair.handle_future.result() for pair in self.__initializers]
        return [
            pair.handle_future.result(timeout=0)
            for pair in self.__initializers
            if pair.handle_future.done()
        ]

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

        set_tsercom_event_loop(runtime_event_loop)
        self.__error_watcher = (
            self.__thread_watcher
        )  # Local errors managed by ThreadWatcher.

        # Use the injected or default-created local_runtime_factory_factory
        factories = self.__create_factories(
            self.__local_runtime_factory_factory
        )

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

        error_sink, error_source = create_multiprocess_queues()
        # Use the factory to create the SplitProcessErrorWatcherSource
        self.__error_watcher = (
            self.__split_error_watcher_source_factory.create(
                self.__thread_watcher, error_source
            )
        )
        self.__error_watcher.start()

        # Use the injected or default-created split_runtime_factory_factory
        factories = self.__create_factories(
            self.__split_runtime_factory_factory
        )

        # Import and prepare the main function for the remote process.
        # Import is deferred to avoid circular dependencies.
        from tsercom.runtime.runtime_main import (
            remote_process_main,
        )

        # Configure and start the new process using ProcessCreator.
        process_target = partial(
            remote_process_main,
            factories,
            error_sink,  # Provide the error queue to the remote process.
            is_testing=self.__is_testing,
        )
        process_daemon = start_as_daemon or self.__is_testing

        self.__process = self.__process_creator.create_process(
            target=process_target,
            args=(),  # remote_process_main is partially applied, so no additional args here
            daemon=process_daemon,
        )
        if self.__process:
            self.__process.start()
        else:
            # Handle process creation failure, e.g., log an error or raise an exception
            # For now, this matches the helper's behavior of returning None on failure
            # and RuntimeManager not explicitly handling it beyond self.__process remaining None.
            # Consider adding error handling/logging here if process creation is critical.
            pass

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
            raise RuntimeError(
                "Error watcher is not available. Ensure the RuntimeManager has been properly started."
            )

        self.__error_watcher.run_until_exception()

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
            return  # No exceptions to check if not started.

        if self.__error_watcher is None:
            # This implies it wasn't started correctly or state is corrupted.
            raise RuntimeError(
                "Error watcher is not available. Ensure the RuntimeManager has been properly started."
            )

        self.__error_watcher.check_for_exception()

    def shutdown(self) -> None:
        """Shuts down the managed process and associated error watcher if applicable.

        This method is intended to clean up resources, particularly for out-of-process
        runtimes. It stops the error watcher if it's a SplitProcessErrorWatcherSource
        and terminates the managed process.
        """
        print("RuntimeManager.shutdown: Starting.", file=sys.stderr)

        # New: Stop ThreadWatcher's event loop and clear global loop
        # Accessing __thread_watcher via its mangled name _RuntimeManager__thread_watcher
        # Assuming _RuntimeManager__thread_watcher exists and has stop_event_loop_thread if it's the correct type.
        if hasattr(self, "_RuntimeManager__thread_watcher") and \
           hasattr(self._RuntimeManager__thread_watcher, 'stop_event_loop_thread'):
            print("RuntimeManager.shutdown: Stopping internal ThreadWatcher's event loop thread.", file=sys.stderr)
            self._RuntimeManager__thread_watcher.stop_event_loop_thread()

        if is_global_event_loop_set(): # from global_event_loop import
            print("RuntimeManager.shutdown: Clearing global Tsercom event loop.", file=sys.stderr)
            clear_tsercom_event_loop() # from global_event_loop import

        # Existing logic for stopping SplitProcessErrorWatcherSource
        # Assuming _RuntimeManager__error_watcher attribute exists (e.g. initialized to None or an object)
        if isinstance(self._RuntimeManager__error_watcher, SplitProcessErrorWatcherSource):
            if self._RuntimeManager__error_watcher.is_running():
                print("RuntimeManager.shutdown: Stopping error watcher source.", file=sys.stderr)
                self._RuntimeManager__error_watcher.stop()
        
        # Existing logic for process termination (with prints)
        # Assuming _RuntimeManager__process attribute exists (e.g. initialized to None or an object)
        if self._RuntimeManager__process and self._RuntimeManager__process.is_alive():
            print(f"RuntimeManager.shutdown: Process {self._RuntimeManager__process.pid} is alive. Attempting to terminate.", file=sys.stderr)
            self._RuntimeManager__process.terminate()
            self._RuntimeManager__process.join(timeout=1.0)
            if self._RuntimeManager__process.is_alive():
                print(f"RuntimeManager.shutdown: Process {self._RuntimeManager__process.pid} not terminated after terminate(), trying kill().", file=sys.stderr)
                self._RuntimeManager__process.kill()
                print(f"RuntimeManager.shutdown: Process {self._RuntimeManager__process.pid} kill signal sent. Joining...", file=sys.stderr)
                self._RuntimeManager__process.join(timeout=1.0)
                if self._RuntimeManager__process.is_alive():
                    print(f"RuntimeManager.shutdown: Process {self._RuntimeManager__process.pid} still alive after kill() and join.", file=sys.stderr)
                else:
                    print(f"RuntimeManager.shutdown: Process {self._RuntimeManager__process.pid} successfully killed and joined.", file=sys.stderr)
            else:
                print(f"RuntimeManager.shutdown: Process {self._RuntimeManager__process.pid} successfully terminated and joined.", file=sys.stderr)
        elif self._RuntimeManager__process: # Process object exists but is not alive
             # Check for .pid only if self._RuntimeManager__process is not None
            pid_info = self._RuntimeManager__process.pid if hasattr(self._RuntimeManager__process, 'pid') and self._RuntimeManager__process.pid is not None else "unknown_pid (process object invalid or None)"
            print(f"RuntimeManager.shutdown: Process {pid_info} was already not alive.", file=sys.stderr)
        else: # self._RuntimeManager__process is None or evaluates to False
            print("RuntimeManager.shutdown: No process to manage (process attribute is None or invalid).", file=sys.stderr)

        self._RuntimeManager__process = None
        print("RuntimeManager.shutdown: Completed.", file=sys.stderr)

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
    RuntimeFactoryFactory.Client[TDataType, TEventType],
    Generic[TDataType, TEventType],
):
    """A client that populates a Future with a RuntimeHandle when ready.

    This class implements the `RuntimeFactoryFactory.Client` interface. Its sole
    purpose is to set the result of a provided `Future` when the
    `_on_handle_ready` callback is invoked with the `RuntimeHandle`.
    """

    def __init__(
        self, future: Future[RuntimeHandle[TDataType, TEventType]]
    ) -> None:
        """Initializes the RuntimeFuturePopulator.

        Args:
            future: The `Future` object that will be populated with the
                    `RuntimeHandle`.
        """
        self.__future: Future[RuntimeHandle[TDataType, TEventType]] = future

    def _on_handle_ready(
        self, handle: RuntimeHandle[TDataType, TEventType]
    ) -> None:
        """Callback invoked when the RuntimeHandle is ready.

        This method sets the provided `handle` as the result of the `Future`
        stored during initialization.

        Args:
            handle: The `RuntimeHandle` that has been successfully created.
        """
        self.__future.set_result(handle)
