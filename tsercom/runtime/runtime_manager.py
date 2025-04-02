from abc import ABC
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.dummy import Process
from typing import Generic, List, Tuple, TypeVar, TypeAlias

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.runtime.remote_process.split_process_error_watcher_sink import (
    SplitProcessErrorWatcherSink,
)
from tsercom.runtime.remote_process.wrapped_runtime_initializer import (
    WrappedRuntimeInitializer,
)
from tsercom.runtime.running_runtime import RunningRuntime
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.runtime.local_process.shim_running_runtime import (
    ShimRunningRuntime,
)
from tsercom.runtime.local_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.threading.aio.global_event_loop import (
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
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.server_synchronized_clock import (
    ServerSynchronizedClock,
)


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
InitializerType: TypeAlias = RuntimeInitializer[TDataType, TEventType]


class RuntimeManager(ABC, Generic[TDataType, TEventType], ErrorWatcher):
    # Requires from user:
    #   - timeout_seconds : int
    #   - RemoteDataAggregator.Client
    # Always local. The remaining parameter is a:
    #     thread_pool: ThreadPoolExecutor which can be created here.
    #
    # So add these as required parameters in the RuntimeInitializer class, and
    # only use them locally.
    def __init__(self):
        super().__init__()

        self.__initializers: list[InitializerType] = []
        self.__has_started = False

        self.__thread_watcher = ThreadWatcher()
        self.__process: Process | None = None

    @property
    def has_started(self) -> bool:
        return self.__has_started

    def register_runtime_initializer(
        self, runtime_initializer: InitializerType
    ):
        self.__initializers.append(runtime_initializer)

    def start_in_process(
        self, runtime_event_loop: AbstractEventLoop
    ) -> List[RunningRuntime[TDataType, TEventType]]:
        assert not self.__has_started
        self.__has_started = True

        # Initialization.
        self.__error_watcher = ThreadWatcher()
        set_tsercom_event_loop(runtime_event_loop)

        # Create all runtimes.
        clock = ServerSynchronizedClock()
        thread_pool = self.__error_watcher.create_tracked_thread_pool_executor(
            max_workers=1
        )
        return [
            self.__start_initializer(initializer, clock, thread_pool)
            for initializer in self.__initializers
        ]

    def start_out_of_process(
        self,
    ) -> List[RunningRuntime[TDataType, TEventType]]:
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
        # initializers) by calling into self.__out_of_process_main().
        self.__process = Process(
            target=partial(
                self.__out_of_process_main, error_sink, initializers
            )
        )
        self.__process.start()

        # Return all runtimes.
        return [
            wrapped_initializer[1]
            for wrapped_initializer in wrapped_initializers
        ]

    def run_until_exception(self) -> None:
        assert self.has_started
        assert self.__error_watcher is not None

        self.__error_watcher.run_until_exception()

    async def __out_of_process_main(
        self,
        error_queue: MultiprocessQueueSink[Exception],
        initializers: List[WrappedRuntimeInitializer[TDataType, TEventType]],
    ):
        thread_watcher: ThreadWatcher = ThreadWatcher
        create_tsercom_event_loop_from_watcher(thread_watcher)
        sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

        clock = ServerSynchronizedClock()

        # Start all runtimes.
        runtimes = [
            initializer.create_runtime(clock, thread_watcher)
            for initializer in initializers
        ]
        for runtime in runtimes:
            await runtime.start_async()

        # Call into run_until_error and, on error, stop all runtimes.
        try:
            sink.run_until_exception()
        except Exception as e:
            raise e
        finally:
            for runtime in runtimes:
                runtime.stop()

    def __wrap_initializer_for_remote(
        self, initializer: InitializerType, thread_pool: ThreadPoolExecutor
    ) -> Tuple[InitializerType, ShimRunningRuntime[TDataType, TEventType]]:
        # TODO: Wrap this so that |initializer| takes a pipe input, which writes
        # events to the instance. HAVE IT WRITE INSTANCES USING A SINGLE
        # THREADED THREAD POOL to ensure that writing is done in single
        # thredded manner.

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
        runtime = ShimRunningRuntime[TEventType, TDataType](
            self.__thread_watcher,
            thread_pool,
            event_sink,
            data_source,
            runtime_command_sink,
        )

        return wrapped, runtime

    def __start_initializer(
        self,
        initializer: InitializerType,
        clock: SynchronizedClock,
        thread_pool: ThreadPoolExecutor,
    ) -> RunningRuntime:
        aggregator = RemoteDataAggregatorImpl[TDataType](thread_pool)
        runtime = initializer.create(clock, aggregator)
        return RuntimeWrapper(runtime, aggregator)
