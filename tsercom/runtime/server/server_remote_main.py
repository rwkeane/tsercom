from typing import Any, List
from tsercom.rpc.grpc.transport.insecure_grpc_channel_factory import InsecureGrpcChannelFactory
from tsercom.runtime.remote_process.split_process_error_watcher_sink import (
    SplitProcessErrorWatcherSink,
)
from tsercom.runtime.remote_process.wrapped_runtime_initializer import (
    WrappedRuntimeInitializer,
)
from tsercom.threading.aio.global_event_loop import (
    create_tsercom_event_loop_from_watcher,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.server.time_sync_server import TimeSyncServer


async def server_remote_main(
    error_queue: MultiprocessQueueSink[Exception],
    initializers: List[WrappedRuntimeInitializer[Any, Any]],
):
    thread_watcher = ThreadWatcher()
    create_tsercom_event_loop_from_watcher(thread_watcher)
    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    # Create the timing server.
    time_server = TimeSyncServer()
    time_started = time_server.start_async()
    if not time_started:
        print("WARNING: TimeSync server failed to start")
    else:
        print("Time Sync server started!")

    # Create the gRPC Channel Factory.
    channel_factory = InsecureGrpcChannelFactory()

    # Start all runtimes.
    clock = time_server.get_synchronized_clock()
    runtimes = [
        initializer.create_runtime(thread_watcher, channel_factory, clock)
        for initializer in initializers
    ]
    for runtime in runtimes:
        runtime.start_async()

    # Call into run_until_error and, on error, stop all runtimes.
    try:
        sink.run_until_exception()
    except Exception as e:
        raise e
    finally:
        for runtime in runtimes:
            await runtime.stop()
