import asyncio
from collections.abc import Callable
from concurrent.futures import Future
import datetime
from functools import partial
from threading import Thread
import time
import pytest  # Added for new tests

from tsercom.api.runtime_manager import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.rpc.grpc_generated.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
)
from tsercom.threading.thread_watcher import ThreadWatcher


started = "STARTED"
stopped = "STOPPED"

start_timestamp = datetime.datetime.now() - datetime.timedelta(hours=10)
stop_timestamp = datetime.datetime.now() + datetime.timedelta(minutes=20)

test_id = CallerIdentifier.random()


class FakeData:
    def __init__(self, val: str):
        self.__val = val

    @property
    def value(self):
        return self.__val


class FakeEvent:
    pass


class FakeRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory

        self.__responder: EndpointDataProcessor[FakeData] | None = None

        super().__init__()

    async def start_async(self) -> None:
        await asyncio.sleep(0.01)

        self.__responder = self.__data_handler.register_caller(
            test_id, "0.0.0.0", 443
        )

        data = FakeData(started)
        try:
            await self.__responder.process_data(data, start_timestamp)
        except Exception:
            pass

    async def stop(self) -> None:
        assert self.__responder is not None
        await self.__responder.process_data(FakeData(stopped), stop_timestamp)


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return FakeRuntime(thread_watcher, data_handler, grpc_channel_factory)


# New Helper Classes
class ErrorThrowingRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,
        grpc_channel_factory: GrpcChannelFactory,
        error_message="TestError",
        error_type=RuntimeError,
    ):
        super().__init__()
        self.error_message = error_message
        self.error_type = error_type
        # These arguments are not used but are part of the expected signature for Runtime implementations
        self._thread_watcher = thread_watcher
        self._data_handler = data_handler
        self._grpc_channel_factory = grpc_channel_factory

    async def start_async(self) -> None:
        raise self.error_type(self.error_message)

    async def stop(self) -> None:
        pass


class ErrorThrowingRuntimeInitializer(RuntimeInitializer):
    def __init__(
        self,
        error_message="TestError",
        error_type=RuntimeError,
        service_type="Client",
    ):
        super().__init__(service_type=service_type)
        self.error_message = error_message
        self.error_type = error_type

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return ErrorThrowingRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self.error_message,
            self.error_type,
        )


class FaultyCreateRuntimeInitializer(RuntimeInitializer):
    def __init__(
        self,
        error_message="CreateFailed",
        error_type=TypeError,
        service_type="Client",
    ):
        super().__init__(service_type=service_type)
        self.error_message = error_message
        self.error_type = error_type

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        raise self.error_type(self.error_message)


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_manager.check_for_exception()
    runtime_future = runtime_manager.register_runtime_initializer(
        FakeRuntimeInitializer(service_type="Client")
    )

    assert not runtime_future.done()
    assert not runtime_manager.has_started
    init_call(runtime_manager)
    assert runtime_manager.has_started
    assert runtime_future.done()

    runtime_manager.check_for_exception()
    runtime = runtime_future.result()
    data_aggregator = runtime.data_aggregator
    assert not data_aggregator.has_new_data(
        test_id
    ), "Aggregator should not have new data for test_id before runtime start"
    runtime.start()

    time.sleep(0.5)

    runtime_manager.check_for_exception()
    assert (
        data_aggregator.any_new_data()
    ), "Aggregator should have some new data (any_new_data)"
    assert data_aggregator.has_new_data(
        test_id
    ), f"Aggregator should have new data for test_id ({test_id})"

    values = data_aggregator.get_new_data(test_id)
    assert isinstance(
        values, list
    ), f"Expected list for get_new_data(test_id), got {type(values)}"
    assert len(values) == 1, f"Expected 1 item for test_id, got {len(values)}"

    first = values[0]
    assert isinstance(first, AnnotatedInstance), type(first)
    assert isinstance(first.data, FakeData), type(first.data)
    assert first.data.value == started
    assert first.timestamp == start_timestamp
    assert first.caller_id == test_id

    assert not data_aggregator.has_new_data(
        test_id
    ), f"Aggregator should not have new data for test_id ({test_id}) after get_new_data"
    runtime_manager.check_for_exception()

    runtime.stop()
    runtime_manager.check_for_exception()

    time.sleep(0.5)

    assert data_aggregator.has_new_data(
        test_id
    ), f"Aggregator should have new data (stop message) for test_id ({test_id})"
    values = data_aggregator.get_new_data(test_id)
    assert isinstance(
        values, list
    ), f"Expected list for get_new_data(test_id) for stop, got {type(values)}"
    assert (
        len(values) == 1
    ), f"Expected 1 stop item for test_id, got {len(values)}"

    first = values[0]
    assert isinstance(first, AnnotatedInstance), type(first)
    assert isinstance(first.data, FakeData), type(first.data)
    assert first.data.value == stopped
    assert first.timestamp == stop_timestamp
    assert first.caller_id == test_id

    assert not data_aggregator.has_new_data(
        test_id
    ), f"Aggregator should not have new data for test_id ({test_id}) after get_new_data for stop"


def test_out_of_process_init():
    clear_tsercom_event_loop()
    __check_initialization(RuntimeManager.start_out_of_process)


def test_in_process_init():
    clear_tsercom_event_loop()

    loop_future = Future()

    def _thread_loop_runner(fut: Future):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        finally:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
            loop.close()

    event_thread = Thread(
        target=_thread_loop_runner, args=(loop_future,), daemon=True
    )
    event_thread.start()

    worker_event_loop = loop_future.result(timeout=5)
    __check_initialization(
        partial(
            RuntimeManager.start_in_process,
            runtime_event_loop=worker_event_loop,
        )
    )
    # Cleanup the event loop thread
    if worker_event_loop.is_running():
        worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
    event_thread.join(timeout=1)


# New Test Cases
def test_out_of_process_error_check_for_exception():
    clear_tsercom_event_loop()
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteFailureOops"
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg, error_type=ValueError
        )
    )
    runtime_manager.start_out_of_process()
    time.sleep(1.0)
    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()

    # Process cleanup (defensive)
    process_attr_name = "_RuntimeManager__process"  # Name mangling
    if hasattr(runtime_manager, process_attr_name):
        process = getattr(runtime_manager, process_attr_name)
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=0.1)


def test_out_of_process_error_run_until_exception():
    clear_tsercom_event_loop()
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteRunUntilFailure"
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg, error_type=RuntimeError
        )
    )
    runtime_manager.start_out_of_process()
    with pytest.raises(RuntimeError, match=error_msg):
        runtime_manager.run_until_exception()

    process_attr_name = "_RuntimeManager__process"
    if hasattr(runtime_manager, process_attr_name):
        process = getattr(runtime_manager, process_attr_name)
        if process is not None and process.is_alive():
            process.terminate()  # Terminate should be fine as run_until_exception implies error state
            process.join(timeout=0.1)


def test_in_process_error_check_for_exception():
    clear_tsercom_event_loop()
    loop_future = Future()

    def _thread_loop_runner(fut: Future):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        finally:
            # Ensure all tasks are cancelled before stopping the loop
            # This part can be tricky if tasks don't handle cancellation well
            all_tasks = asyncio.all_tasks(loop)
            if all_tasks:
                for task in all_tasks:
                    task.cancel()
                # Allow tasks to process cancellation
                # loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))

            if not loop.is_closed():
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
                    # Give some time for stop to propagate if needed in some contexts
                    # However, for run_forever, stop should be enough.
                    # If run_until_complete was used, it would wait.
                # Ensure it's fully stopped before closing
                # This might require running the loop briefly if stop() was just called
                # For simplicity in test, assume stop is effective.
            loop.close()

    event_thread = Thread(
        target=_thread_loop_runner, args=(loop_future,), daemon=True
    )
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "InProcessFailureOops"
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg, error_type=ValueError
        )
    )
    runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)

    # Allow time for the error to be processed by the event loop and ThreadWatcher
    # The error is raised in start_async, which is run on the worker_event_loop.
    # The ThreadWatcher is part of the RuntimeManager and should catch it.
    time.sleep(0.3)  # Increased sleep to be safer

    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()

    # Cleanup the event loop thread
    if worker_event_loop.is_running():
        worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
    event_thread.join(timeout=1)


def test_out_of_process_initializer_create_error():
    clear_tsercom_event_loop()
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "CreateOops"
    runtime_manager.register_runtime_initializer(
        FaultyCreateRuntimeInitializer(
            error_message=error_msg, error_type=TypeError
        )
    )
    runtime_manager.start_out_of_process()
    time.sleep(1.0)
    with pytest.raises(TypeError, match=error_msg):
        runtime_manager.check_for_exception()

    process_attr_name = "_RuntimeManager__process"
    if hasattr(runtime_manager, process_attr_name):
        process = getattr(runtime_manager, process_attr_name)
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=0.1)
