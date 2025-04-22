from typing import Any, List
from tsercom.rpc.grpc.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)
from tsercom.runtime.remote_process.split_process_error_watcher_sink import (
    SplitProcessErrorWatcherSink,
)
from tsercom.runtime.remote_process.wrapped_runtime_initializer import (
    WrappedRuntimeInitializer,
)
from tsercom.runtime.runtime import Runtime
from tsercom.threading.aio.global_event_loop import (
    create_tsercom_event_loop_from_watcher,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.thread_watcher import ThreadWatcher


async def client_remote_main(
    error_queue: MultiprocessQueueSink[Exception],
    initializers: List[WrappedRuntimeInitializer[Any, Any]],
):
    thread_watcher = ThreadWatcher()
    create_tsercom_event_loop_from_watcher(thread_watcher)
    sink = SplitProcessErrorWatcherSink(thread_watcher, error_queue)

    # Create the gRPC Channel Factory.
    channel_factory = InsecureGrpcChannelFactory()

    # Start all runtimes.
    runtimes: List[Runtime] = [
        initializer.create_runtime(thread_watcher, channel_factory)
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
