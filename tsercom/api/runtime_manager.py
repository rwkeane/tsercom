"""Manages the creation, lifecycle, and error monitoring of Tsercom runtimes.

This module provides the `RuntimeManager` class, which is the primary entry point
for applications to initialize and manage Tsercom runtimes. It supports
starting runtimes either within the same process as the manager or in separate,
isolated processes.
"""

import logging
from asyncio import AbstractEventLoop
from concurrent.futures import Future
from functools import partial
from multiprocessing import Process
from typing import Generic, List, Optional

from tsercom.api.initialization_pair import InitializationPair
from tsercom.api.local_process.local_runtime_factory_factory import (
    LocalRuntimeFactoryFactory,
)

# Import TypeVars from runtime_factory_factory for consistency
from tsercom.api.runtime_factory_factory import (
    DataTypeT,
    EventTypeT,
    RuntimeFactoryFactory,
)
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.api.runtime_manager_helpers import (
    ProcessCreator,
    SplitErrorWatcherSourceFactory,
)
from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.aio_utils import get_running_loop_or_none
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    create_tsercom_event_loop_from_watcher,
    set_tsercom_event_loop,
)
from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker

logger = logging.getLogger(__name__)


class RuntimeManager(ErrorWatcher, Generic[DataTypeT, EventTypeT]):
    """Manages the lifecycle of Tsercom runtimes, supporting in-process and out-of-process execution.

    This class serves as a central coordinator for initializing, starting, and
    monitoring Tsercom runtimes. Users register `RuntimeInitializer` instances,
    and the manager then uses appropriate factories (`LocalRuntimeFactoryFactory`
    or `SplitRuntimeFactoryFactory`) to create and launch these runtimes.

    Key functionalities include:
    - Registering runtime initializers and providing `Future` objects for their
      corresponding `RuntimeHandle`s.
    - Starting runtimes either in the same process (sharing an event loop) or
      in a new, isolated process (for out-of-process execution).
    - Monitoring for exceptions from managed runtimes via a `ThreadWatcher`
      (for in-process) or a `SplitProcessErrorWatcherSource` (for out-of-process).
    - Providing methods to check for exceptions or block until an exception occurs.
    - Handling shutdown of created processes and error watchers.

    Type Args:
        DataTypeT: The type of data objects (bound by `ExposedData`) that the
            managed runtimes will handle.
        EventTypeT: The type of event objects that the managed runtimes will process.
    """

    def __init__(
        self,
        *,
        is_testing: bool = False,
        thread_watcher: Optional[ThreadWatcher] = None,
        local_runtime_factory_factory: Optional[
            LocalRuntimeFactoryFactory[DataTypeT, EventTypeT]
        ] = None,
        split_runtime_factory_factory: Optional[
            SplitRuntimeFactoryFactory[DataTypeT, EventTypeT]
        ] = None,
        process_creator: Optional[ProcessCreator] = None,
        split_error_watcher_source_factory: Optional[
            SplitErrorWatcherSourceFactory
        ] = None,
    ) -> None:
        """Initializes the RuntimeManager.

        Args:
            is_testing: If True, configures certain operations for testing
                environments, such as making out-of-process runtimes daemonic
                by default.
            thread_watcher: An optional, pre-configured `ThreadWatcher` instance.
                If `None`, a new `ThreadWatcher` is created.
            local_runtime_factory_factory: An optional factory for creating
                in-process runtimes. If `None`, a default instance is created.
            split_runtime_factory_factory: An optional factory for creating
                out-of-process (split) runtimes. If `None`, a default instance
                is created.
            process_creator: An optional helper for creating new processes,
                primarily for testing. If `None`, a default `ProcessCreator`
                is used.
            split_error_watcher_source_factory: An optional factory for creating
                `SplitProcessErrorWatcherSource` instances, used for monitoring
                out-of-process runtimes. If `None`, a default factory is used.
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
            self.__local_runtime_factory_factory: RuntimeFactoryFactory[
                DataTypeT, EventTypeT
            ] = local_runtime_factory_factory
        else:
            default_local_factory_thread_pool = (
                self.__thread_watcher.create_tracked_thread_pool_executor(
                    max_workers=1  # Typically for local runtime factory tasks
                )
            )
            self.__local_runtime_factory_factory = LocalRuntimeFactoryFactory(
                default_local_factory_thread_pool
            )

        if split_runtime_factory_factory is not None:
            self.__split_runtime_factory_factory: RuntimeFactoryFactory[
                DataTypeT, EventTypeT
            ] = split_runtime_factory_factory  # Use imported
        else:
            default_split_factory_thread_pool = (
                self.__thread_watcher.create_tracked_thread_pool_executor(
                    max_workers=1  # Typically for split runtime factory tasks
                )
            )
            self.__split_runtime_factory_factory = SplitRuntimeFactoryFactory(
                default_split_factory_thread_pool, self.__thread_watcher
            )

        self.__initializers: List[
            InitializationPair[DataTypeT, EventTypeT]
        ] = []
        self.__has_started: IsRunningTracker = IsRunningTracker()
        self.__error_watcher: Optional[SplitProcessErrorWatcherSource] = None
        self.__process: Optional[Process] = None

    @property
    def has_started(self) -> bool:
        """Indicates whether this manager instance has been started.

        Once started (via `start_in_process` or `start_out_of_process`),
        new runtime initializers cannot be registered. This method is thread-safe.

        Returns:
            True if the manager has been started, False otherwise.
        """
        return self.__has_started.get()

    def register_runtime_initializer(
        self,
        runtime_initializer: RuntimeInitializer[DataTypeT, EventTypeT],
    ) -> Future[RuntimeHandle[DataTypeT, EventTypeT]]:
        """Registers a `RuntimeInitializer` to be managed and launched.

        This method must be called before `start_in_process` or
        `start_out_of_process`. Each registered initializer will result in the
        creation of a runtime when one of the start methods is called.

        Args:
            runtime_initializer: The initializer instance that defines how to
                create a specific `Runtime`.

        Returns:
            A `concurrent.futures.Future` that will eventually be populated with
            the `RuntimeHandle` for the initialized runtime. The future is
            fulfilled after the corresponding runtime factory has been created
            and its handle is ready.

        Raises:
            RuntimeError: If this method is called after the manager has already
                been started.
        """
        if self.has_started:
            raise RuntimeError(
                "Cannot register runtime initializer after the manager has started."
            )

        handle_future = Future[RuntimeHandle[DataTypeT, EventTypeT]]()
        pair = InitializationPair[DataTypeT, EventTypeT](
            handle_future, runtime_initializer
        )
        self.__initializers.append(pair)

        return handle_future

    async def start_in_process_async(self) -> None:
        """Asynchronously creates and starts all registered runtimes in the current process.

        This is a convenience method that calls `start_in_process`, using the
        currently running asyncio event loop. Tsercom operations for these
        runtimes will execute on this loop.

        Note:
            `RuntimeHandle`s for the started runtimes should be obtained from the
            `Future` objects returned by `register_runtime_initializer`.

        Raises:
            RuntimeError: If no asyncio event loop is currently running in the
                calling context.
        """
        running_loop = get_running_loop_or_none()
        if running_loop is None:
            raise RuntimeError(
                "Could not determine the current running event loop for start_in_process_async."
            )
        self.start_in_process(running_loop)

    def start_in_process(
        self,
        runtime_event_loop: AbstractEventLoop,
    ) -> None:
        """Creates and starts all registered runtimes in the current process.

        Tsercom operations for the initialized runtimes will run on the provided
        `runtime_event_loop`. The global Tsercom event loop is set to this loop.
        This method uses the `LocalRuntimeFactoryFactory` to create the necessary
        runtime factories.

        Note:
            `RuntimeHandle`s for the started runtimes should be obtained from the
            `Future` objects returned by `register_runtime_initializer`.

        Args:
            runtime_event_loop: The asyncio event loop on which Tsercom
                operations for these runtimes will execute.

        Raises:
            RuntimeError: If the manager has already been started.
        """
        if self.has_started:
            raise RuntimeError("RuntimeManager has already been started.")
        self.__has_started.start()

        set_tsercom_event_loop(runtime_event_loop)

        factories = self.__create_factories(
            self.__local_runtime_factory_factory
        )

        # Import is deferred to method scope to avoid circular dependencies at module load time.
        from tsercom.runtime.runtime_main import (  # pylint: disable=import-outside-toplevel
            initialize_runtimes,
        )

        initialize_runtimes(
            self.__thread_watcher, factories, is_testing=self.__is_testing
        )

    def start_out_of_process(
        self, start_as_daemon: bool = True
    ) -> None:  # Changed default to True
        """Creates and starts registered runtimes in a new, separate process.

        This method uses the `SplitRuntimeFactoryFactory` to prepare runtime
        factories suitable for inter-process communication. A new process is
        spawned, and the `remote_process_main` function from `tsercom.runtime.runtime_main`
        is executed in that process to initialize and run the runtimes.
        Error monitoring for the remote process is set up using a
        `SplitProcessErrorWatcherSource`.

        Note:
            `RuntimeHandle`s for the started runtimes should be obtained from the
            `Future` objects returned by `register_runtime_initializer`.

        Args:
            start_as_daemon: If True, the new process will be a daemon process. Defaults to `True`.
                Daemonic processes are typically used for background tasks and
                are automatically terminated when the main program exits.
                When `is_testing` is also True, it remains `True`.

        Raises:
            RuntimeError: If the manager has already been started.
        """
        if self.has_started:
            raise RuntimeError("RuntimeManager has already been started.")
        self.__has_started.start()

        create_tsercom_event_loop_from_watcher(self.__thread_watcher)

        error_sink: MultiprocessQueueSink[Exception]
        error_source: MultiprocessQueueSource[Exception]
        error_sink, error_source = create_multiprocess_queues()

        self.__error_watcher = (
            self.__split_error_watcher_source_factory.create(
                self.__thread_watcher, error_source
            )
        )
        self.__error_watcher.start()

        factories = self.__create_factories(
            self.__split_runtime_factory_factory
        )

        # Import is deferred to method scope to avoid circular dependencies at module load time.
        from tsercom.runtime.runtime_main import (  # pylint: disable=import-outside-toplevel
            remote_process_main,
        )

        process_target = partial(
            remote_process_main,
            factories,
            error_sink,
            is_testing=self.__is_testing,
        )
        process_daemon = start_as_daemon or self.__is_testing

        self.__process = self.__process_creator.create_process(
            target=process_target,
            args=(),
            daemon=process_daemon,
        )

        if self.__process:
            self.__process.start()
        else:
            logger.warning(
                "Failed to create process for out-of-process runtime."
            )

    def run_until_exception(self) -> None:
        """Blocks the calling thread until an exception is reported from any managed runtime.

        This method relies on the internal `ThreadWatcher` (for in-process runtimes)
        or the `SplitProcessErrorWatcherSource` (for out-of-process runtimes via
        the `ThreadWatcher`) to signal an exception. When an exception is caught,
        this method re-raises it in the calling thread.

        Raises:
            Exception: The first exception propagated from any managed runtime.
            RuntimeError: If the manager has not been started or if the
                necessary error watching components are not available.
        """
        if not self.has_started:
            raise RuntimeError("RuntimeManager has not been started.")
        if self.__thread_watcher is None:
            raise RuntimeError(
                "Internal ThreadWatcher is None when checking for exceptions after start."
            )
        # The __thread_watcher is initialized in __init__, so it should always be present.
        self.__thread_watcher.run_until_exception()

    def check_for_exception(self) -> None:
        """Checks for and re-raises the first caught exception from managed runtimes.

        If an exception has been propagated from any runtime and caught by the
        manager\'s error watching mechanisms, this method re-raises that
        exception in the calling thread. If no exceptions have been caught,
        this method does nothing. This method is thread-safe.

        Raises:
            Exception: The first exception propagated from any managed runtime, if any.
            RuntimeError: If the manager has been started but the internal
                `ThreadWatcher` is somehow not available (should not happen
                with proper initialization).
        """
        if not self.has_started:
            return
        if self.__thread_watcher is None:
            raise RuntimeError(
                "Internal ThreadWatcher is None when checking for exceptions after start."
            )
        # The __thread_watcher is initialized in __init__.
        self.__thread_watcher.check_for_exception()

    def shutdown(self) -> None:
        """Shuts down the `RuntimeManager` and cleans up associated resources.

        This includes:
        - Terminating any out-of-process runtime process.
        - Stopping the `SplitProcessErrorWatcherSource` if it was used.
        - Clearing the Tsercom global event loop.
        - Potentially other cleanup related to the `ThreadWatcher` if it owned
          resources like thread pools (though ThreadWatcher itself usually
          does not own pools directly, it tracks them).
        """
        logger.info("RuntimeManager.shutdown: Starting shutdown sequence.")

        if self.__process is not None and self.__process.is_alive():
            logger.info("Terminating out-of-process runtime process.")
            self.__process.kill()  # kill() is more forceful than terminate()
            self.__process.join(timeout=5)  # Wait for kill
            if self.__process.is_alive():
                logger.warning(
                    "Out-of-process runtime process did not terminate cleanly after kill()."
                )
            self.__process = None

        if isinstance(self.__error_watcher, SplitProcessErrorWatcherSource):
            if self.__error_watcher.is_running:
                logger.info("Stopping SplitProcessErrorWatcherSource.")
                self.__error_watcher.stop()
        self.__error_watcher = None

        # Clear Tsercom global event loop.
        # This also attempts to stop the loop if it was set by this manager.
        clear_tsercom_event_loop()

        # Note: ThreadWatcher itself doesn't usually need explicit shutdown unless
        # it started its own threads that need joining, which is not typical for its role here.
        # Thread pools created via thread_watcher.create_tracked_thread_pool_executor
        # should be shut down by whoever created them if they are not daemonic.
        # If this RuntimeManager created default factories with new thread pools,
        # those pools should ideally be shut down.
        # LocalRuntimeFactoryFactory and SplitRuntimeFactoryFactory might need shutdown methods
        # if they own non-daemonic thread pools. For now, assuming they manage their own resources
        # or use daemonic threads/pools that exit with the main process.

        logger.info("RuntimeManager.shutdown: Sequence complete.")

    def __create_factories(
        self,
        factory_factory: RuntimeFactoryFactory[DataTypeT, EventTypeT],
    ) -> List[RuntimeFactory[DataTypeT, EventTypeT]]:
        """Creates runtime factories for all registered initializers.

        This internal helper method iterates through each `InitializationPair`
        (containing a `RuntimeInitializer` and a `Future` for its `RuntimeHandle`)
        and uses the provided `factory_factory` (either local or split) to
        create a `RuntimeFactory`.

        A `RuntimeFuturePopulator` is used as the client for the `factory_factory`
        to ensure that the `RuntimeHandle` created by the `RuntimeFactory` is
        used to fulfill the `Future` associated with the initializer.

        Args:
            factory_factory: The `RuntimeFactoryFactory` (e.g.,
                `LocalRuntimeFactoryFactory` or `SplitRuntimeFactoryFactory`)
                to use for creating individual `RuntimeFactory` instances.

        Returns:
            A list of created `RuntimeFactory` instances, one for each
            registered `RuntimeInitializer`.
        """
        results: List[RuntimeFactory[DataTypeT, EventTypeT]] = []
        for pair in self.__initializers:
            populator_client = RuntimeFuturePopulator[DataTypeT, EventTypeT](
                pair.handle_future
            )
            factory = factory_factory.create_factory(
                populator_client,
                pair.initializer,
            )
            results.append(factory)
        return results


class RuntimeFuturePopulator(
    RuntimeFactoryFactory.Client,
    Generic[DataTypeT, EventTypeT],
):
    """A client that populates a Future with a RuntimeHandle once it's ready.

    This class implements the `RuntimeFactoryFactory.Client` interface. Its primary
    role is to act as a bridge between the asynchronous creation of a `RuntimeHandle`
    by a `RuntimeFactory` and a `concurrent.futures.Future` that allows callers
    to synchronously or asynchronously await the availability of the handle.
    """

    def __init__(
        self, future: Future[RuntimeHandle[DataTypeT, EventTypeT]]
    ) -> None:
        """Initializes the RuntimeFuturePopulator.

        Args:
            future: The `Future` object that will be populated with the
                `RuntimeHandle` when `_on_handle_ready` is called.
        """
        self.__future: Future[RuntimeHandle[DataTypeT, EventTypeT]] = future

    def _on_handle_ready(  # type: ignore[override]
        self,
        handle: RuntimeHandle[DataTypeT, EventTypeT],  # Use imported TypeVars
    ) -> None:
        """Callback invoked by a `RuntimeFactory` when its `RuntimeHandle` is ready.

        This method fulfills the `Future` (provided during initialization) with
        the given `handle`.

        Args:
            handle: The `RuntimeHandle` that has been successfully created and
                is ready for use.
        """
        self.__future.set_result(handle)
