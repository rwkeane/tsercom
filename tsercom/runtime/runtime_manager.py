from abc import ABC
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.dummy import Process
from typing import Any, Callable, Dict, Generic, List, Tuple, TypeVar

from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.rpc.grpc.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.runtime.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.runtime.remote_process.wrapped_runtime_initializer import (
    WrappedRuntimeInitializer,
)
from tsercom.runtime.running_runtime import RunningRuntime
from tsercom.runtime.local_process.shim_running_runtime import (
    ShimRunningRuntime,
)
from tsercom.runtime.local_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.aio_utils import get_running_loop_or_none
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop,
)
from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    create_multiprocess_queues,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.time_sync_server import TimeSyncServer


TInitializerType = TypeVar("TInitializerType", bound=RuntimeInitializer)


class RuntimeManager(ABC, Generic[TInitializerType], ErrorWatcher):
    """
    This is the top-level class for managing runtimes for user-defined
    functionality. It is used to create such runtimes from RuntimeInitializer
    instances, and handles the complexity associated with starting either in the
    local or a remote process, as well as all associated error handling.
    """

    def __init__(
        self,
        out_of_process_main: Callable[
            [
                MultiprocessQueueSink[Exception],
                List[WrappedRuntimeInitializer[Any, Any]],
                None,
            ]
        ],
    ):
        super().__init__()

        self.__out_of_process_main = out_of_process_main

        self.__initializers: list[TInitializerType] = []
        self.__has_started = False

        self.__thread_watcher = ThreadWatcher()
        self.__process: Process | None = None

    @property
    def has_started(self) -> bool:
        """
        Returns whether this instance is currently running.
        """
        return self.__has_started

    def register_runtime_initializer(
        self, runtime_initializer: TInitializerType
    ):
        """
        Registers a new RuntimeInitializer which should be initialized when this
        instance is started. May only be called prior to this instance starting.
        """
        assert not self.has_started
        self.__initializers.append(runtime_initializer)

    async def start_in_process_async(
        self,
    ) -> List[RunningRuntime[Any, Any, TInitializerType]]:
        """
        Creates runtimes from all registered RuntimeInitializer instances, and
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
    ) -> Dict[TInitializerType, RunningRuntime[Any, Any, TInitializerType]]:
        """
        Creates runtimes from all registered RuntimeInitializer instances, and
        then starts each creaed instance, all in the current process. These
        created instances are then returned. Tsercom operations are run on the
        provided event loop.
        """
        assert not self.__has_started
        self.__has_started = True

        # Initialization.
        self.__error_watcher = ThreadWatcher()
        set_tsercom_event_loop(runtime_event_loop)

        # Create the gRPC Channel Factory.
        channel_factory = InsecureGrpcChannelFactory()

        # Create the timing server.
        time_server = TimeSyncServer()
        time_started = time_server.start_async()
        if not time_started:
            print("WARNING: TimeSync server failed to start")
        else:
            print("Time Sync server started!")

        clock = time_server.get_synchronized_clock()

        # Create all runtime instances.
        thread_pool = self.__error_watcher.create_tracked_thread_pool_executor(
            max_workers=1
        )
        results = [
            self.__start_initializer(initializer, clock, thread_pool)
            for initializer in self.__initializers
        ]
        return self.__create_runtime_map(results)

    def start_out_of_process(
        self,
    ) -> Dict[TInitializerType, RunningRuntime[Any, Any, TInitializerType]]:
        """
        Creates runtimes from all registered RuntimeInitializer instances, and
        then starts each creaed instance in a new process separate from the
        current process. Commands to such runtimes are forwarded from the
        returned Runtime instances, and data received from it can be accessed
        through the RemoteDataAggregator instance available in it.
        """
        assert not self.__has_started
        self.__has_started = True

        # Wrap all initializers
        thread_pool = self.__error_watcher.create_tracked_thread_pool_executor(
            max_workers=1
        )
        wrapped_initializers = [
            self.__wrap_initializer_for_remote(initializer, thread_pool)
            for initializer in self.__initializers
        ]
        initializers = [
            wrapped_initializers[0]
            for wrapped_initializers in wrapped_initializers
        ]

        # Save the local end of the error watcher and begin listening.
        error_sink, error_source = create_multiprocess_queues()
        self.__error_watcher = SplitProcessErrorWatcherSource(
            self.__thread_watcher, error_source
        )
        self.__error_watcher.start()

        # Create a new process, passing this instance (with replaced
        # initializers) by calling into __out_of_process_main().
        self.__process = Process(
            target=partial(
                self.__out_of_process_main, error_sink, initializers
            )
        )
        self.__process.start()

        # Return all runtimes.
        results = [
            wrapped_initializer[1]
            for wrapped_initializer in wrapped_initializers
        ]
        return self.__create_runtime_map(results)

    def run_until_exception(self) -> None:
        """
        Runs the current thread until an exception as been raised, throwing the
        exception upon receipt.
        """
        assert self.has_started
        assert self.__error_watcher is not None

        self.__error_watcher.run_until_exception()

    def check_for_exception(self) -> None:
        """
        If an exception has been seen, throw it. Else, do nothing. This method
        is thread safe and can be called from any thread.
        """
        assert self.has_started
        assert self.__error_watcher is not None

        self.__error_watcher.check_for_exception()

    def __create_runtime_map(
        self, runtimes: List[RunningRuntime[Any, Any, TInitializerType]]
    ) -> Dict[TInitializerType, RunningRuntime[Any, Any, TInitializerType]]:
        map = {}
        for runtime in runtimes:
            map[runtime.initializer] = runtime
        return map

    def __wrap_initializer_for_remote(
        self,
        initializer: TInitializerType,
        thread_pool: ThreadPoolExecutor,
    ) -> Tuple[
        TInitializerType,
        ShimRunningRuntime[Any, Any, TInitializerType],
    ]:
        # Create the pipes between source and destination.
        event_sink, event_source = create_multiprocess_queues()
        data_sink, data_source = create_multiprocess_queues()
        runtime_command_sink, runtime_command_source = (
            create_multiprocess_queues()
        )

        # Put the first end of each in a to-be-remotely-called initializer.
        wrapped = WrappedRuntimeInitializer(
            initializer, event_source, data_sink, runtime_command_source
        )

        # Return the local ends along with the wrapped instance.
        aggregator = RemoteDataAggregatorImpl[Any](
            thread_pool, initializer.client(), initializer.timeout()
        )
        runtime = ShimRunningRuntime[Any, Any, TInitializerType](
            self.__thread_watcher,
            event_sink,
            data_source,
            runtime_command_sink,
            aggregator,
        )

        return wrapped, runtime

    def __start_initializer(
        self,
        initializer: TInitializerType,
        clock: SynchronizedClock,
        thread_pool: ThreadPoolExecutor,
    ) -> RunningRuntime[Any, Any, TInitializerType]:
        aggregator = RemoteDataAggregatorImpl[Any](
            thread_pool, initializer.client(), initializer.timeout()
        )
        runtime = initializer.create(clock, aggregator)
        return RuntimeWrapper(runtime, aggregator, initializer)
