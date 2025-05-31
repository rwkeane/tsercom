"""End-to-end tests for Tsercom runtime initialization, data flow, and error handling."""

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
import datetime
from functools import partial
from threading import Thread
import time
import pytest

from tsercom.api.runtime_manager import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
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
        test_id: CallerIdentifier,
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: EndpointDataProcessor[FakeData] | None = None
        self._data_sent = False

        super().__init__()

    def __repr__(self) -> str:
        return f"<FakeRuntime instance at {id(self)}>"

    async def start_async(self) -> None:
        try:
            asyncio.current_task()
        except RuntimeError:
            pass

        await asyncio.sleep(0.01)

        self.__responder = self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 443
        )

        if not self._data_sent:
            fresh_data_value = "FRESH_SIMPLE_DATA_V2"
            fresh_data_object = FakeData(fresh_data_value)
            fresh_timestamp = datetime.datetime.now()

            await self.__responder.process_data(
                fresh_data_object, fresh_timestamp
            )
            self._data_sent = True

    async def stop(self, exception) -> None:
        assert self.__responder is not None
        await self.__responder.process_data(FakeData(stopped), stop_timestamp)


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(self, test_id: CallerIdentifier, service_type="Client"):
        super().__init__(service_type=service_type)
        self._test_id = test_id

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return FakeRuntime(
            thread_watcher, data_handler, grpc_channel_factory, self._test_id
        )


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
        self._thread_watcher = thread_watcher
        self._data_handler = data_handler
        self._grpc_channel_factory = grpc_channel_factory

    async def start_async(self) -> None:
        raise self.error_type(self.error_message)

    async def stop(self, exception) -> None:
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


@pytest.fixture
def clear_loop_fixture():
    clear_tsercom_event_loop()
    yield
    clear_tsercom_event_loop()


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                test_id=current_test_id, service_type="Server"
            )
        )

        assert not runtime_future.done()
        assert not runtime_manager.has_started
        init_call(runtime_manager)
        assert runtime_manager.has_started
        assert runtime_future.done()

        runtime_manager.check_for_exception()
        runtime_handle = runtime_future.result()
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator
        assert not data_aggregator.has_new_data(current_test_id)
        runtime_handle.start()

        data_arrived = False
        max_wait_time = 5.0
        poll_interval = 0.1
        waited_time = 0.0
        while waited_time < max_wait_time:
            has_data_now = data_aggregator.has_new_data(current_test_id)
            if has_data_now:
                data_arrived = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        runtime_manager.check_for_exception()
        assert (
            data_arrived
        ), f"Aggregator did not receive data for test_id ({current_test_id}) within {max_wait_time}s"
        assert data_aggregator.has_new_data(current_test_id)

        values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values, list)
        assert len(values) == 1

        first = values[0]
        assert isinstance(first, AnnotatedInstance)
        assert isinstance(first.data, FakeData)
        expected_fresh_value = "FRESH_SIMPLE_DATA_V2"
        # actual_value_for_log was removed as it was unused
        assert first.data.value == expected_fresh_value
        assert isinstance(first.timestamp, datetime.datetime)
        assert first.caller_id == current_test_id

        assert not data_aggregator.has_new_data(current_test_id)
        runtime_manager.check_for_exception()

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        time.sleep(0.5)

        assert data_aggregator.has_new_data(current_test_id)
        values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values, list)
        assert len(values) == 1

        first = values[0]
        assert isinstance(first, AnnotatedInstance)
        assert isinstance(first.data, FakeData)
        assert first.data.value == stopped
        assert first.timestamp == stop_timestamp
        assert first.caller_id == current_test_id

        assert not data_aggregator.has_new_data(current_test_id)

    except Exception as e:
        raise e
    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception:
                pass
        runtime_manager.shutdown()


def test_out_of_process_init(clear_loop_fixture):
    __check_initialization(RuntimeManager.start_out_of_process)


def test_in_process_init(clear_loop_fixture):
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
    if worker_event_loop.is_running():
        worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
    event_thread.join(timeout=1)


def test_out_of_process_error_check_for_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteFailureOops"

    handle_future = runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg,
            error_type=ValueError,
            service_type="Server",
        )
    )
    runtime_manager.start_out_of_process()

    try:
        runtime_handle = handle_future.result(timeout=2)
        runtime_handle.start()
    except Exception as e_handle:
        pytest.fail(f"Failed to get or start runtime_handle: {e_handle}")

    wait_time_for_error = 1.5
    time.sleep(wait_time_for_error)

    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()


def test_out_of_process_error_run_until_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteRunUntilFailure"
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg, error_type=RuntimeError
        )
    )
    runtime_manager.start_out_of_process()
    with pytest.raises(RuntimeError, match=error_msg):
        runtime_manager.check_for_exception()
        for _ in range(5):
            time.sleep(1)
            runtime_manager.check_for_exception()


def test_in_process_error_check_for_exception(clear_loop_fixture):
    loop_future = Future()

    def _thread_loop_runner(fut: Future):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        finally:
            all_tasks = asyncio.all_tasks(loop)
            if all_tasks:
                for task in all_tasks:
                    task.cancel()

            if not loop.is_closed():
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
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

    time.sleep(0.3)

    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()

    if worker_event_loop.is_running():
        worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
    event_thread.join(timeout=1)


def test_out_of_process_initializer_create_error(clear_loop_fixture):
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
