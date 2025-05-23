from asyncio import AbstractEventLoop
from concurrent.futures import Future
from functools import partial
from multiprocessing import Process
from typing import Any, Generic, List, TypeVar

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

# Imports for initialize_runtimes and remote_process_main moved into methods
# to break circular dependency.
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


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class RuntimeManager(ErrorWatcher):
    """
    This is the top-level class for managing runtimes for user-defined
    functionality. It is used to create such runtimes from RuntimeInitializer
    instances, and handles the complexity associated with starting either in the
    local or a remote process, as well as all associated error handling.
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.__initializers: list[InitializationPair[Any, Any]] = []
        self.__has_started = IsRunningTracker()

        self.__thread_watcher = ThreadWatcher()
        self.__error_watcher: ErrorWatcher | None = None
        self.__process: Process | None = None

    @property
    def has_started(self) -> bool:
        """
        Returns whether this instance is currently running. This method is
        thread safe.
        """
        return self.__has_started.get()

    def register_runtime_initializer(
        self, runtime_initializer: RuntimeInitializer[TDataType, TEventType]
    ) -> Future[RuntimeHandle[TDataType, TEventType]]:
        """
        Registers a new RuntimeInitializer which should be initialized when this
        instance is started. May only be called prior to this instance starting.

        Returns a Future associated with the RuntimeHandle that can be used to
        control the Runtime this |runtime_initializer| will create. It will
        become valid once either start_in_process() or start_out_of_process()
        has been called.
        """
        assert not self.has_started

        future = Future[RuntimeHandle[TDataType, TEventType]]()
        pair = InitializationPair[TDataType, TEventType](
            future, runtime_initializer
        )
        self.__initializers.append(pair)

        return future

    async def start_in_process_async(
        self,
    ) -> List[RuntimeHandle[Any, Any]]:
        """
        Creates Runtimes from all registered RuntimeInitializer instances, and
        then starts each creaed instance, all in the current process. These
        created instances are then returned. Tsercom operations are run on the
        event loop from which this operation is called.
        """
        running_loop = get_running_loop_or_none()
        assert running_loop is not None
        return self.start_in_process(running_loop)

    def start_in_process(
        self,
        runtime_event_loop: AbstractEventLoop,
    ) -> None:
        """
        Creates Runtimes from all registered RuntimeInitializer instances, and
        then starts each created instance, all in the current process. These
        created instances are then returned. Tsercom operations are run on the
        provided event loop.
        """
        assert not self.has_started
        self.__has_started.start()

        # Initialization.
        set_tsercom_event_loop(runtime_event_loop)
        self.__error_watcher = self.__thread_watcher

        # Create all factories
        thread_pool = (
            self.__thread_watcher.create_tracked_thread_pool_executor(
                max_workers=1
            )
        )
        factory_factory = LocalRuntimeFactoryFactory(thread_pool)
        factories = self.__create_factories(factory_factory)

        # Start running them. Initialization is performed on the local thread
        # but computation will quickly change using to the provided
        # |event_loop|.
        from tsercom.runtime.runtime_main import (
            initialize_runtimes,
        )  # Moved import

        initialize_runtimes(self.__thread_watcher, factories)

    def start_out_of_process(
        self,
    ) -> None:
        """
        Creates runtimes from all registered RuntimeInitializer instances, and
        then starts each creaed instance in a new process separate from the
        current process. Commands to such runtimes are forwarded from the
        returned Runtime instances, and data received from it can be accessed
        through the RemoteDataAggregator instance available in it.
        """
        assert not self.has_started
        self.__has_started.start()

        # Set a local Tsercom event loop. It is rarely used, but is required for
        # some operations.
        create_tsercom_event_loop_from_watcher(self.__thread_watcher)

        # Save the local end of the error watcher and begin listening.
        error_sink, error_source = create_multiprocess_queues()
        self.__error_watcher = SplitProcessErrorWatcherSource(
            self.__thread_watcher, error_source
        )
        self.__error_watcher.start()

        # Wrap all initializers
        thread_pool = (
            self.__thread_watcher.create_tracked_thread_pool_executor(
                max_workers=1
            )
        )
        factory_factory = SplitRuntimeFactoryFactory(
            thread_pool, self.__thread_watcher
        )
        factories = self.__create_factories(factory_factory)

        # Create a new process, passing initializers and error queue endpoint.
        from tsercom.runtime.runtime_main import (
            remote_process_main,
        )  # Moved import

        self.__process = Process(
            target=partial(remote_process_main, factories, error_sink)
        )
        self.__process.start()

    def run_until_exception(self) -> None:
        """
        Runs the current thread until an exception as been raised, throwing the
        exception upon receipt. This method is thread-safe and can be called
        from any thread.
        """
        assert self.has_started
        assert self.__error_watcher is not None

        self.__thread_watcher.run_until_exception()

    def check_for_exception(self) -> None:
        """
        If an exception has been seen, throw it. Else, do nothing. This method
        is thread safe and can be called from any thread.
        """
        if not self.has_started:
            return

        assert self.__error_watcher is not None

        self.__thread_watcher.check_for_exception()

    def __create_factories(
        self, factory_factory: RuntimeFactoryFactory
    ) -> List[RuntimeFactory]:
        results = []
        for pair in self.__initializers:
            factory = factory_factory.create_factory(
                RuntimeFuturePopulator[Any, Any](pair.handle_future),
                pair.initializer,
            )
            results.append(factory)

        return results


class RuntimeFuturePopulator(
    RuntimeFactoryFactory.Client, Generic[TDataType, TEventType]
):
    def __init__(self, future: Future[RuntimeHandle[TDataType, TEventType]]):
        self.__future = future

    def _on_handle_ready(self, handle: RuntimeHandle[TDataType, TEventType]):
        self.__future.set_result(handle)
