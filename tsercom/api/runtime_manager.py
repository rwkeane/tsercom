"""Manages the creation and lifecycle of Tsercom runtimes."""

import logging
from asyncio import AbstractEventLoop
from concurrent.futures import Future
from functools import partial
from multiprocessing import Process  # Keep for type hinting if necessary
from typing import Any, Generic, List, TypeVar, Optional

# Make RuntimeManager Generic by importing TypeVar if not already (it is, for DataTypeT, EventTypeT)

from tsercom.api.initialization_pair import InitializationPair

# Black-formatted import
from tsercom.api.local_process.local_runtime_factory_factory import (
    LocalRuntimeFactoryFactory,
)
from tsercom.api.runtime_factory_factory import RuntimeFactoryFactory

# Black-formatted import
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.runtime_handle import RuntimeHandle

# Black-formatted import
from tsercom.api.runtime_manager_helpers import (
    ProcessCreator,
    SplitErrorWatcherSourceFactory,
)

# Black-formatted import
from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,  # Keep for type hinting if necessary
)
from tsercom.data.exposed_data import ExposedData
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer


# Imports for runtime_main are moved into methods (start_in_process, start_out_of_process)
# to break potential circular dependencies between manager and main execution modules.
from tsercom.threading.aio.aio_utils import get_running_loop_or_none

# Black-formatted import
from tsercom.threading.aio.global_event_loop import (
    create_tsercom_event_loop_from_watcher,
    set_tsercom_event_loop,
    clear_tsercom_event_loop,
)
from tsercom.threading.error_watcher import ErrorWatcher

# Removed incorrect import: from tsercom.system.multiprocess_queue import MultiprocessQueueSink, MultiprocessQueueSource
# Black-formatted import
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)

# Black-formatted import
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)

# Black-formatted import
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker
from tsercom.runtime.runtime_main import initialize_runtimes
from tsercom.runtime.runtime_main import remote_process_main

# Type variables for generic RuntimeHandle and related classes.
DataTypeT = TypeVar("DataTypeT", bound=ExposedData)
EventTypeT = TypeVar("EventTypeT")


logger = logging.getLogger(__name__)


# pylint: disable=R0902 # Essential attributes for manager state
class RuntimeManager(
    ErrorWatcher, Generic[DataTypeT, EventTypeT]
):  # Made Generic
    """Manages the lifecycle of Tsercom runtimes.

    This class is responsible for creating runtimes from `RuntimeInitializer`
    instances. It handles the complexities of starting runtimes either in the
    current process or in a separate process, along with associated error
    monitoring and propagation.
    """

    # pylint: disable=R0913 # Necessary arguments for comprehensive setup
    def __init__(  # Black-formatted signature
        self,
        *,
        is_testing: bool = False,
        thread_watcher: Optional[ThreadWatcher] = None,
        local_runtime_factory_factory: Optional[
            LocalRuntimeFactoryFactory[DataTypeT, EventTypeT]  # Parameterized
        ] = None,
        split_runtime_factory_factory: Optional[
            SplitRuntimeFactoryFactory[DataTypeT, EventTypeT]  # Parameterized
        ] = None,
        process_creator: Optional[ProcessCreator] = None,
        split_error_watcher_source_factory: Optional[
            SplitErrorWatcherSourceFactory
        ] = None,
    ) -> None:
        """Initializes the RuntimeManager.

        Args:
            is_testing: If True, configures some operations for testing,
                        such as making out-of-process runtimes daemonic.
            thread_watcher: Optional ThreadWatcher instance.
            local_runtime_factory_factory: Optional factory for local runtimes.
            split_runtime_factory_factory: Optional factory for split runtimes.
            process_creator: Optional helper to create processes.
            split_error_watcher_source_factory: Factory for error watcher source.
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

        # Field types will be correctly inferred if constructor params are typed
        # However, explicit annotation is good practice if they are complex.
        # For now, let's ensure constructor params are typed and see if mypy infers fields.
        # If redefinition errors persist, we'll explicitly type fields.

        if local_runtime_factory_factory is not None:
            self.__local_runtime_factory_factory = (
                local_runtime_factory_factory
            )
        else:
            default_local_factory_thread_pool = (
                self.__thread_watcher.create_tracked_thread_pool_executor(
                    max_workers=1
                )
            )
            # Default factories are [Any, Any] because DataTypeT/EventTypeT are not known
            # if RuntimeManager is instantiated as RuntimeManager() without specific types.
            self.__local_runtime_factory_factory = LocalRuntimeFactoryFactory[
                Any, Any
            ](default_local_factory_thread_pool)

        if split_runtime_factory_factory is not None:
            self.__split_runtime_factory_factory = (
                split_runtime_factory_factory
            )
        else:
            default_split_factory_thread_pool = (
                self.__thread_watcher.create_tracked_thread_pool_executor(
                    max_workers=1
                )
            )
            self.__split_runtime_factory_factory = SplitRuntimeFactoryFactory[
                Any, Any
            ](default_split_factory_thread_pool, self.__thread_watcher)

        # If RuntimeManager is Generic[DataTypeT, EventTypeT], initializers should store these specific types.
        self.__initializers: list[
            InitializationPair[DataTypeT, EventTypeT]
        ] = []
        self.__has_started: IsRunningTracker = IsRunningTracker()
        self.__error_watcher: Optional[SplitProcessErrorWatcherSource] = None
        self.__process: Optional[Process] = None

    @property
    def has_started(self) -> bool:
        """Indicates whether this manager instance is currently running.

        This method is thread-safe.

        Returns:
            True if the manager has started, False otherwise.
        """
        return self.__has_started.get()

    def register_runtime_initializer(
        self, runtime_initializer: RuntimeInitializer[DataTypeT, EventTypeT]
    ) -> Future[RuntimeHandle[DataTypeT, EventTypeT]]:
        """Registers a RuntimeInitializer to be managed.

        Must be called before `start_in_process` or `start_out_of_process`.

        Args:
            runtime_initializer: The initializer for the runtime.

        Returns:
            A Future that will be populated with the RuntimeHandle once the
            runtime is initialized.

        Raises:
            RuntimeError: If called after the manager has started.
        """
        # Ensure initializers are registered only before starting.
        if self.has_started:
            # Long but readable error message
            raise RuntimeError(
                "Cannot register runtime initializer after the manager has started."
            )

        handle_future = Future[RuntimeHandle[DataTypeT, EventTypeT]]()
        pair = InitializationPair[DataTypeT, EventTypeT](
            handle_future, runtime_initializer
        )
        self.__initializers.append(pair)

        return handle_future

    async def start_in_process_async(
        self,
    ) -> None:
        """Creates and starts all registered runtimes in the current process.

        This method calls `start_in_process` with the current asyncio event
        loop. Tsercom operations will run on this loop.

        RuntimeHandles are retrieved via Futures from `register_runtime_initializer`.

        Returns:
            None. RuntimeHandles are obtained via Futures.

        Raises:
            RuntimeError: If no event loop is running.
            Exception: If a future completed with an exception.
        """
        running_loop = get_running_loop_or_none()

        if running_loop is None:
            # Long but readable error message
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

        # Use the injected or default-created local_runtime_factory_factory
        factories = self.__create_factories(
            self.__local_runtime_factory_factory
        )

        # Import is deferred to avoid circular dependencies.
        # pylint: disable=C0415 # Avoid circular import / late import

        initialize_runtimes(
            self.__thread_watcher, factories, is_testing=self.__is_testing
        )

    def start_out_of_process(self, start_as_daemon: bool = False) -> None:
        """Creates and starts registered runtimes in a new, separate process.

        Creates runtimes from registered initializers and starts them in a new,
        separate process. Commands are forwarded from the returned Runtime
        instances. Data is accessed via RemoteDataAggregator.

        Args:
            start_as_daemon: If True, the new process will be a daemon process.
                             Daemonic processes are for background tasks and
                             terminate when the main program exits.

        Raises:
            RuntimeError: If called after the manager has started.
        """
        if self.has_started:
            raise RuntimeError("RuntimeManager has already been started.")

        self.__has_started.start()

        # Set up a minimal local Tsercom event loop, primarily for utilities.
        create_tsercom_event_loop_from_watcher(self.__thread_watcher)

        error_sink: MultiprocessQueueSink[Exception]
        error_source: MultiprocessQueueSource[Exception]
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
        # pylint: disable=C0415 # Avoid circular import / late import

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
            logger.warning(
                "Failed to create process for out-of-process runtime."
            )

    def run_until_exception(self) -> None:
        """Blocks execution until an exception from any managed runtime.

        Runs the current thread until an exception is raised, then throws it.
        This method is thread-safe.

        It re-raises the caught exception in the calling thread.

        Raises:
            Any exception propagated from the managed runtimes.
            RuntimeError: If manager not started or error watcher not set.
        """
        if not self.has_started:
            # Added this check for consistency, as __thread_watcher depends on has_started
            raise RuntimeError("RuntimeManager has not been started.")
        if self.__thread_watcher is None:  # Reverted to simpler check
            # Long but readable error message
            raise RuntimeError(
                "Error watcher is not available. Ensure the RuntimeManager has been properly started."
            )

        self.__thread_watcher.run_until_exception()

    def check_for_exception(self) -> None:
        """Checks for and re-raises any exceptions from managed runtimes.

        If an exception has been caught from any runtime, this method re-raises
        it in the calling thread. Otherwise, it does nothing. This method is
        thread-safe.

        Raises:
            Any exception propagated from the managed runtimes.
            RuntimeError: If manager started but error watcher not set.
        """
        if not self.has_started:
            return  # No exceptions to check if not started.

        # Reverted: If manager has started, but __thread_watcher is somehow None, it's an issue.
        if self.__thread_watcher is None:
            # Long but readable error message
            raise RuntimeError(
                "Error watcher is not available. Ensure the RuntimeManager has been properly started."
            )

        self.__thread_watcher.check_for_exception()

    def shutdown(self) -> None:
        """Shuts down the managed process and associated error watcher if applicable.

        This method is intended to clean up resources, particularly for out-of-process
        runtimes. It stops the error watcher if it's a
        SplitProcessErrorWatcherSource and terminates the managed process.
        """
        logger.info("RuntimeManager.shutdown: Starting.")

        if self.__process is not None:
            self.__process.kill()

        # Existing logic for stopping SplitProcessErrorWatcherSource
        # Assuming _RuntimeManager__error_watcher attribute exists (e.g. initialized to None or an object)
        if isinstance(self.__error_watcher, SplitProcessErrorWatcherSource):
            if self.__error_watcher.is_running:  # Access as a property
                self.__error_watcher.stop()

        clear_tsercom_event_loop()

    def __create_factories(  # This method now needs to use DataTypeT, EventTypeT from self
        self, factory_factory: RuntimeFactoryFactory[DataTypeT, EventTypeT]
    ) -> List[RuntimeFactory[DataTypeT, EventTypeT]]:
        """Creates runtime factories using the provided factory_factory.

        Iterates through all registered `InitializationPair`s and uses the
        `factory_factory` to create a `RuntimeFactory` for each. The
        associated `RuntimeHandle` Future in each pair is populated by a
        `RuntimeFuturePopulator` client.

        Args:
            factory_factory: The `RuntimeFactoryFactory` (local or split)
                             to use for creating `RuntimeFactory` instances.

        Returns:
            A list of created `RuntimeFactory` instances.
        """
        results: List[RuntimeFactory[DataTypeT, EventTypeT]] = []
        for (
            pair
        ) in (
            self.__initializers
        ):  # pair is InitializationPair[DataTypeT, EventTypeT]
            # The RuntimeFuturePopulator acts as the client to receive the handle.
            client = RuntimeFuturePopulator[DataTypeT, EventTypeT](
                pair.handle_future
            )  # Use DataTypeT, EventTypeT from pair
            factory = factory_factory.create_factory(
                client,  # client is Populator[DataTypeT, EventTypeT]
                pair.initializer,  # initializer is RuntimeInitializer[DataTypeT, EventTypeT]
            )
            results.append(factory)
        return results


# pylint: disable=R0903 # Internal helper class
class RuntimeFuturePopulator(
    RuntimeFactoryFactory.Client,
    Generic[DataTypeT, EventTypeT],  # Make it generic again
):
    """A client that populates a Future with a RuntimeHandle when ready.

    This class implements the `RuntimeFactoryFactory.Client` interface. Its sole
    purpose is to set the result of a provided `Future` when the
    `_on_handle_ready` callback is invoked with the `RuntimeHandle`.
    """

    def __init__(
        self, future: Future[RuntimeHandle[DataTypeT, EventTypeT]]
    ) -> None:
        """Initializes the RuntimeFuturePopulator.

        Args:
            future: The `Future` object that will be populated with the
                    `RuntimeHandle`.
        """
        self.__future: Future[RuntimeHandle[DataTypeT, EventTypeT]] = future

    def _on_handle_ready(  # type: ignore[override]
        self, handle: RuntimeHandle[DataTypeT, EventTypeT]
    ) -> None:
        """Callback invoked when the RuntimeHandle is ready.

        This method sets the provided `handle` as the result of the `Future`
        stored during initialization.

        Args:
            handle: The `RuntimeHandle` that has been successfully created.
        """
        self.__future.set_result(handle)
