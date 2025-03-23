
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop
from functools import partial
import multiprocessing
from multiprocessing.dummy import Process
from queue import Empty, Full
from typing import Generic, List, Optional, Tuple, TypeVar, TypeAlias

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.runtime.data_reader_sink import DataReaderSink
from tsercom.runtime.running_runtime import RunningRuntime
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_command import RuntimeCommand
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.runtime.shim_running_runtime import ShimRunningRuntime
from tsercom.runtime.split_process_error_watcher_sink import SplitProcessErrorWatcherSink
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.aio.global_event_loop import create_tsercom_event_loop_from_watcher, set_tsercom_event_loop
from tsercom.threading.error_watcher import ErrorWatcher
from tsercom.threading.multiprocess.multiprocess_input_queue import MultiprocessQueueSink
from tsercom.threading.multiprocess.multiprocess_output_queue import MultiprocessQueueSource
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.server_synchronized_clock import ServerSynchronizedClock


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")
InitializerType : TypeAlias = RuntimeInitializer[TDataType, TEventType]
class RuntimeManager(ABC, Generic[TDataType, TEventType]):
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

        self.__initializers : list[InitializerType] = []
        self.__has_started = False

        self.__error_watcher : ErrorWatcher | None = None
        self.__process : Process | None = None

    @property
    def has_started(self) -> bool:
        return self.__has_started

    def register_runtime_initializer(self, runtime_initializer : InitializerType):
        self.__initializers.append(runtime_initializer)

    def start_in_process(self, runtime_event_loop : AbstractEventLoop) -> List[RunningRuntime[TDataType, TEventType]]:
        assert not self.__has_started
        self.__has_started = True

        self.__error_watcher = ThreadWatcher()
        set_tsercom_event_loop(runtime_event_loop)
        return self.__start_in_process_impl(self.__initializers)
    
    def start_out_of_process(self) -> List[RunningRuntime[TDataType, TEventType]]:
        assert not self.__has_started
        self.__has_started = True

        # Wrap all initializers
        wrapped_initializers = [ self.__wrap_initializer_for_remote(initializer) for initializer in self.__initializers ]
        self.__initializers = [ wrapped_initializers[0] for wrapped_initializers in wrapped_initializers ]

        # Create a new process, passing this instance (with replaced
        # initializers) by calling into self.__out_of_process_main().
        self.__process = Process(target=partial(self.__out_of_process_main))
        self.__process.start()

        # Save the local end of the error watcher.
        self.__error_watcher = SplitProcessErrorWatcherSink()

        # Create a new ShimRunningRuntime from the RemoteDataAggregator created
        # from MultiprocessInputQueue[TEventType].
        
        # Then, feed the shim to the QueueDataSource[TDataType] as data_reader.
        
        pass
    
    def __start_in_process_impl(self,
                                init_infos : list['RuntimeManager.InitializationInfo[TDataType, TEventType]']) -> List[Runtime[TEventType]]:
        assert self.__error_watcher is not None
        clock = ServerSynchronizedClock()
        
        # Cache the runtimes locally so the instances aren't garbage collected if remote, since results won't be used.
        return [ init_info.initializer.create(clock, init_info.data_reader) for init_info in init_infos]

    async def __out_of_process_main(
            self,
            error_queue : MultiprocessQueueSink[Exception],
            init_infos : list['RuntimeManager.InitializationInfo[TDataType, TEventType]']):
        self.__error_watcher = ThreadWatcher()
        create_tsercom_event_loop_from_watcher(self.__error_watcher)

        # Delegate to __start_in_process_impl(). Cache results locally so that
        # it doesn't get garbage collected.
        runtimes = self.__start_in_process_impl(init_infos)
        for runtime in runtimes:
            await runtime.start_async()

        # Call into run_until_error and, on error, forward it to the other end
        # of the error pipe.
        try:
            self.__error_watcher.run_until_exception()
        except Exception as e:
            error_queue.put_blocking(e)

        # Stop all running runtimes.
        for runtime in runtimes:
            await runtime.stop()

    class RuntimeInitializerTransport(RuntimeInitializer[TDataType, TEventType]):
        def __init__(self, delegated : RuntimeInitializer[TDataType, TEventType]):
            self.__delegated = delegated

        def create(self,
                   clock : SynchronizedClock,
                   data_reader : RemoteDataReader[TDataType]) -> Runtime[TEventType]:
            return self.__delegated.create()

    def __wrap_initializer_for_remote(self, initializer : InitializerType) -> Tuple[InitializerType, ShimRunningRuntime[TDataType, TEventType]]:
        # TODO: Wrap this so that |initializer| takes a pipe input, which writes
        # events to the instance. HAVE IT WRITE INSTANCES USING A SINGLE
        # THREADED THREAD POOL to ensure that writing is done in single
        # thredded manner.
        
        # Create the pipes between source and destination.

        # Put the first end of each in a to-be-remotely-called initializer.

        # Return the local ends along with the wrapped instance.
        shim_runtime = ShimRunningRuntime(event_sink, runtime_command_sink)
        
        pass

    class InitializationInfo(Generic[TDataType, TEventType]):
        def __init__(self,
                     initializer : InitializerType,
                     data_reader : RemoteDataReader[TDataType]):
            self.__initializer = initializer
            self.__data_reader = data_reader
            
        @property
        def initializer(self) -> InitializerType:
            return self.__initializer
            
        @property
        def data_reader(self) -> RemoteDataReader[TDataType]:
            return self.__data_reader 