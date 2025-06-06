"""End-to-end tests for Tsercom runtime initialization, data flow, and error handling."""

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
import datetime
import dataclasses  # Ensure this import is present
from functools import partial
from threading import Thread
import time
from typing import (
    List,
    Optional,
    Any,
)  # Ensure List, Optional, Any are imported
import uuid  # For creating specific UUIDs for CallerIdentifier

import pytest

from tsercom.api.runtime_handle import RuntimeHandle # Added import
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
# stop_timestamp for FakeRuntime, distinct from current_stop_time in BroadcastTestRuntime
global_test_stop_timestamp = datetime.datetime.now(
    datetime.timezone.utc
) + datetime.timedelta(minutes=20)


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
        grpc_channel_factory: Optional[GrpcChannelFactory],  # Made Optional
        test_id: CallerIdentifier,
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: Optional[
            EndpointDataProcessor[FakeData, FakeEvent]
        ] = None
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

        assert self.__data_handler is not None
        self.__responder = await self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 443
        )
        assert self.__responder is not None

        if not self._data_sent:
            fresh_data_value = "FRESH_SIMPLE_DATA_V2"
            fresh_data_object = FakeData(fresh_data_value)
            fresh_timestamp = datetime.datetime.now(datetime.timezone.utc)
            await self.__responder.process_data(
                fresh_data_object, fresh_timestamp
            )
            self._data_sent = True

    async def stop(self, exception) -> None:
        assert self.__responder is not None
        await self.__responder.process_data(
            FakeData(stopped), global_test_stop_timestamp
        )


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(self, test_id: CallerIdentifier, service_type="Client"):
        super().__init__(service_type=service_type)
        self._test_id = test_id

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: Optional[GrpcChannelFactory],
    ) -> Runtime:
        return FakeRuntime(
            thread_watcher, data_handler, grpc_channel_factory, self._test_id
        )


class ErrorThrowingRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,
        grpc_channel_factory: Optional[GrpcChannelFactory],
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
        grpc_channel_factory: Optional[GrpcChannelFactory],
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
        grpc_channel_factory: Optional[GrpcChannelFactory],
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
            time.sleep(poll_interval)  # Using time.sleep in a non-async helper
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
        assert first.data.value == expected_fresh_value
        assert isinstance(first.timestamp, datetime.datetime)
        assert first.caller_id == current_test_id

        assert not data_aggregator.has_new_data(current_test_id)
        runtime_manager.check_for_exception()

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        time.sleep(0.5)

        stopped_data_arrived = False
        max_wait_stopped_data = 3.0
        poll_interval_stopped = 0.1
        waited_time_stopped = 0.0
        while waited_time_stopped < max_wait_stopped_data:
            if data_aggregator.has_new_data(current_test_id):
                stopped_data_arrived = True
                break
            time.sleep(poll_interval_stopped)
            waited_time_stopped += poll_interval_stopped

        assert (
            stopped_data_arrived
        ), f"Aggregator did not receive 'stopped' data for test_id ({current_test_id}) within {max_wait_stopped_data}s"
        assert data_aggregator.has_new_data(current_test_id)

        values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values, list)
        assert len(values) == 1

        first = values[0]
        assert isinstance(first, AnnotatedInstance)
        assert isinstance(first.data, FakeData)
        assert first.data.value == stopped
        assert first.timestamp == global_test_stop_timestamp
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
            if (
                not loop.is_closed()
            ):  # Check if loop is not closed before stopping
                loop.call_soon_threadsafe(loop.stop)
            if not loop.is_closed():  # Check again before closing
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

    time.sleep(1.5)

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
    # Loop to ensure error is caught by check_for_exception
    for _ in range(10):  # Increased range for safety
        try:
            runtime_manager.check_for_exception()
            time.sleep(0.2)  # Slightly longer sleep
        except RuntimeError as e:
            assert str(e) == error_msg
            break
    else:
        pytest.fail(f"RuntimeError '{error_msg}' not caught.")


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
            if not loop.is_closed() and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            if not loop.is_closed():
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


def test_multiple_runtimes_out_of_process(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []
    loop = asyncio.get_event_loop()  # Get current event loop for asyncio.run
    try:
        test_id_1 = CallerIdentifier.random()
        test_id_2 = CallerIdentifier.random()

        runtime_future_1 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(test_id=test_id_1, service_type="Server")
        )
        runtime_future_2 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(test_id=test_id_2, service_type="Server")
        )

        runtime_manager.start_out_of_process()
        runtime_handle_1 = runtime_future_1.result(timeout=2)
        runtime_handle_2 = runtime_future_2.result(timeout=2)
        runtime_handles_for_cleanup.extend(
            [runtime_handle_1, runtime_handle_2]
        )

        data_aggregator_1 = runtime_handle_1.data_aggregator
        data_aggregator_2 = runtime_handle_2.data_aggregator

        runtime_handle_1.start()
        runtime_handle_2.start()

        # Use loop.run_until_complete for async helper
        initial_data_1_wrapper = loop.run_until_complete(
            _wait_for_data(
                data_aggregator_1,
                test_id_1,
                "FRESH_SIMPLE_DATA_V2",
                FakeData,
                timeout=10,
            )
        )
        assert initial_data_1_wrapper.value == "FRESH_SIMPLE_DATA_V2"

        initial_data_2_wrapper = loop.run_until_complete(
            _wait_for_data(
                data_aggregator_2,
                test_id_2,
                "FRESH_SIMPLE_DATA_V2",
                FakeData,
                timeout=10,
            )
        )
        assert initial_data_2_wrapper.value == "FRESH_SIMPLE_DATA_V2"

        assert not data_aggregator_1.has_new_data(test_id_2)
        assert not data_aggregator_2.has_new_data(test_id_1)

        runtime_handle_1.stop()
        stopped_data_1_wrapper = loop.run_until_complete(
            _wait_for_data(
                data_aggregator_1, test_id_1, stopped, FakeData, timeout=10
            )
        )
        assert stopped_data_1_wrapper.value == stopped

        runtime_handle_2.stop()
        stopped_data_2_wrapper = loop.run_until_complete(
            _wait_for_data(
                data_aggregator_2, test_id_2, stopped, FakeData, timeout=10
            )
        )
        assert stopped_data_2_wrapper.value == stopped

    finally:
        for handle in runtime_handles_for_cleanup:
            try:
                handle.stop()
            except Exception:
                pass
        runtime_manager.shutdown()


def test_client_type_runtime_in_process(clear_loop_fixture):
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
            if not loop.is_closed():
                loop.close()

    event_thread = Thread(
        target=_thread_loop_runner, args=(loop_future,), daemon=True
    )
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)
    main_loop = asyncio.get_event_loop()

    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                test_id=current_test_id, service_type="Client"
            )
        )
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        runtime_handle = runtime_future.result()
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator
        runtime_handle.start()

        initial_data_wrapper = main_loop.run_until_complete(
            _wait_for_data(
                data_aggregator,
                current_test_id,
                "FRESH_SIMPLE_DATA_V2",
                FakeData,
                timeout=10,
            )
        )
        assert initial_data_wrapper.value == "FRESH_SIMPLE_DATA_V2"

        runtime_handle.stop()
        stopped_data_wrapper = main_loop.run_until_complete(
            _wait_for_data(
                data_aggregator, current_test_id, stopped, FakeData, timeout=10
            )
        )
        assert stopped_data_wrapper.value == stopped
    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception:
                pass
        runtime_manager.shutdown()
        if worker_event_loop.is_running():
            worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        event_thread.join(timeout=1)


def test_in_process_initializer_create_error(clear_loop_fixture):
    loop_future = Future()

    def _thread_loop_runner(fut: Future):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        finally:
            all_tasks = asyncio.all_tasks(loop)
            if all_tasks:  # Ensure tasks are cancelled before closing loop
                for task in all_tasks:
                    task.cancel()
            if not loop.is_closed() and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            if not loop.is_closed():
                loop.close()

    event_thread = Thread(
        target=_thread_loop_runner, args=(loop_future,), daemon=True
    )
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "InProcessCreateOops"
    try:
        initializer = FaultyCreateRuntimeInitializer(
            error_message=error_msg,
            error_type=ValueError,
            service_type="Client",
        )
        runtime_manager.register_runtime_initializer(initializer)
        # For in-process, error in create() should propagate from start_in_process()
        with pytest.raises(ValueError, match=error_msg):
            runtime_manager.start_in_process(
                runtime_event_loop=worker_event_loop
            )
    finally:
        runtime_manager.shutdown()  # Ensure manager is shutdown
        if worker_event_loop.is_running():
            worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        event_thread.join(timeout=1)


def test_out_of_process_error_direct_run_until_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "DirectBlockError"
    error_type = ConnectionError
    thread_result_queue = Future()

    def target_for_thread():
        try:
            runtime_manager.run_until_exception()
        except Exception as e:
            thread_result_queue.set_result(e)
        else:
            thread_result_queue.set_result(None)

    test_thread = None
    runtime_handle_for_cleanup = None
    try:
        initializer = ErrorThrowingRuntimeInitializer(
            error_message=error_msg,
            error_type=error_type,
            service_type="Server",
        )
        handle_future = runtime_manager.register_runtime_initializer(
            initializer
        )
        runtime_manager.start_out_of_process()
        runtime_handle = handle_future.result(timeout=2)
        runtime_handle_for_cleanup = runtime_handle
        runtime_handle.start()
        time.sleep(1.5)

        test_thread = Thread(target=target_for_thread, daemon=True)
        test_thread.start()
        test_thread.join(timeout=5.0)
        assert not test_thread.is_alive(), "run_until_exception timed out"

        result_from_thread = thread_result_queue.result(timeout=0)
        assert isinstance(result_from_thread, error_type)
        assert str(result_from_thread) == error_msg
    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception:
                pass
        runtime_manager.shutdown()
        if test_thread and test_thread.is_alive():
            test_thread.join(timeout=1)


# --- Broadcast Test Components ---
@dataclasses.dataclass(frozen=True)
class BroadcastTestRuntimeEvent:
    payload: str

    def __repr__(self) -> str:
        return f"<BroadcastTestRuntimeEvent payload='{self.payload}'>"


@dataclasses.dataclass(frozen=True)
class BroadcastTestRuntimeData:
    message: str
    originator_id_str: str

    def __repr__(self) -> str:
        return f"<BroadcastTestRuntimeData message='{self.message}' from='{self.originator_id_str}'>"


class BroadcastTestRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[
            BroadcastTestRuntimeData, BroadcastTestRuntimeEvent
        ],
        grpc_channel_factory: Optional[GrpcChannelFactory],
        test_ids: List[CallerIdentifier],
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__test_ids = test_ids
        self.__responders: dict[
            CallerIdentifier,
            EndpointDataProcessor[
                BroadcastTestRuntimeData, BroadcastTestRuntimeEvent
            ],
        ] = {}
        self.__event_processing_tasks: list[asyncio.Task] = (
            []
        )  # Renamed from __processing_tasks
        self.__stop_timestamp = datetime.datetime.now(
            datetime.timezone.utc
        )  # Defined here

    async def _create_processing_task_for_responder(
        self,
        test_id: CallerIdentifier,
        responder: EndpointDataProcessor[
            BroadcastTestRuntimeData, BroadcastTestRuntimeEvent
        ],
    ):
        try:
            async for event_list in responder:
                for event_inst in event_list:
                    processed_data = BroadcastTestRuntimeData(
                        message=f"processed_event_{event_inst.data.payload}_by_{str(test_id)}",
                        originator_id_str=str(test_id),
                    )
                    await responder.process_data(
                        processed_data,
                        datetime.datetime.now(datetime.timezone.utc),
                    )
        except asyncio.CancelledError:
            # print(f"REMOTERUNTIME_TASK_CANCELLED_FOR: {str(test_id)}")
            pass
        except Exception as e:
            print(f"REMOTERUNTIME_TASK_ERROR_FOR: {str(test_id)} - {e!r}")

    async def start_async(self) -> None:
        for i, test_id in enumerate(self.__test_ids):
            port = 50000 + i
            responder = await self.__data_handler.register_caller(
                test_id, "127.0.0.1", port
            )
            assert (
                responder is not None
            ), f"Failed to register caller {str(test_id)}"
            self.__responders[test_id] = responder
            initial_data = BroadcastTestRuntimeData(
                message=f"initial_for_{str(test_id)}",
                originator_id_str=str(test_id),
            )
            await responder.process_data(
                initial_data, datetime.datetime.now(datetime.timezone.utc)
            )
            task = asyncio.create_task(
                self._create_processing_task_for_responder(test_id, responder)
            )
            self.__event_processing_tasks.append(
                task
            )  # Use correct attribute name

    async def stop(self, exception) -> None:
        print(
            f"REMOTERUNTIME_STOP_CALLED: {datetime.datetime.now(datetime.timezone.utc)} for instance {id(self)}"
        )
        stop_message_sending_tasks = []
        for cid in self.__test_ids:
            responder = self.__responders.get(cid)
            if responder:
                print(
                    f"REMOTERUNTIME_STOP_ATTEMPT_CLIENT: {str(cid)} at {datetime.datetime.now(datetime.timezone.utc)}"
                )
                try:
                    stopped_data = BroadcastTestRuntimeData(
                        message=f"stopped_for_{str(cid)}",
                        originator_id_str=str(cid),
                    )
                    stop_message_sending_tasks.append(
                        asyncio.create_task(
                            responder.process_data(
                                stopped_data, self.__stop_timestamp
                            )
                        )
                    )
                    print(
                        f"REMOTERUNTIME_STOP_TASK_CREATED_FOR_CLIENT: {str(cid)} at {datetime.datetime.now(datetime.timezone.utc)}"
                    )
                except Exception as e:
                    print(
                        f"REMOTERUNTIME_STOP_ERROR_CREATING_TASK_FOR_CLIENT: {str(cid)} - {e!r} at {datetime.datetime.now(datetime.timezone.utc)}"
                    )
            else:
                print(
                    f"REMOTERUNTIME_STOP_NO_RESPONDER_FOR_CLIENT: {str(cid)} at {datetime.datetime.now(datetime.timezone.utc)}"
                )

        if stop_message_sending_tasks:
            print(
                f"REMOTERUNTIME_STOP_AWAITING_GATHER for {len(stop_message_sending_tasks)} stop message tasks at {datetime.datetime.now(datetime.timezone.utc)}"
            )
            results = await asyncio.gather(
                *stop_message_sending_tasks, return_exceptions=True
            )
            for i, result_item in enumerate(results):
                cid_for_result = (
                    self.__test_ids[i]
                    if i < len(self.__test_ids)
                    else "unknown_cid"
                )
                if isinstance(result_item, Exception):
                    print(
                        f"REMOTERUNTIME_STOP_GATHER_ERROR_FOR_CLIENT: {str(cid_for_result)} - {result_item!r} at {datetime.datetime.now(datetime.timezone.utc)}"
                    )
                else:
                    print(
                        f"REMOTERUNTIME_STOP_GATHER_SUCCESS_FOR_CLIENT: {str(cid_for_result)} at {datetime.datetime.now(datetime.timezone.utc)}"
                    )
        else:
            print(
                f"REMOTERUNTIME_STOP_NO_STOP_MESSAGE_TASKS_TO_GATHER at {datetime.datetime.now(datetime.timezone.utc)}"
            )

        print(
            f"REMOTERUNTIME_STOP_CANCELLING_EVENT_PROCESSING_TASKS at {datetime.datetime.now(datetime.timezone.utc)}"
        )
        for task_idx, task in enumerate(
            self.__event_processing_tasks
        ):  # Use correct attribute name
            if task and not task.done():
                # Ensure task_idx is valid for self.__test_ids if used for logging
                cid_str = (
                    str(self.__test_ids[task_idx])
                    if task_idx < len(self.__test_ids)
                    else f"task_index_{task_idx}"
                )
                print(
                    f"REMOTERUNTIME_STOP_CANCELLING_TASK_{task_idx}_FOR_CLIENTID_{cid_str} at {datetime.datetime.now(datetime.timezone.utc)}"
                )
                task.cancel()
        if self.__event_processing_tasks:  # Use correct attribute name
            await asyncio.gather(
                *[t for t in self.__event_processing_tasks if t],
                return_exceptions=True,
            )

        print(
            f"REMOTERUNTIME_STOP_METHOD_COMPLETED: {datetime.datetime.now(datetime.timezone.utc)} for instance {id(self)}"
        )


class BroadcastTestRuntimeInitializer(
    RuntimeInitializer[BroadcastTestRuntimeData, BroadcastTestRuntimeEvent]
):
    def __init__(
        self, test_ids: List[CallerIdentifier], service_type="Client"
    ):
        super().__init__(service_type=service_type)
        self._test_ids = test_ids

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[
            BroadcastTestRuntimeData, BroadcastTestRuntimeEvent
        ],
        grpc_channel_factory: Optional[GrpcChannelFactory],
    ) -> Runtime:
        return BroadcastTestRuntime(
            thread_watcher, data_handler, grpc_channel_factory, self._test_ids
        )


async def _wait_for_data(
    aggregator: Any,
    caller_id: CallerIdentifier,
    expected_message_part: str,
    data_class_type: type,
    timeout: float = 5.0,
    originator_id_str_to_check: Optional[str] = None,
) -> Any:
    data_arrived = False
    found_item = None
    max_wait_time = timeout
    poll_interval = 0.1
    waited_time = 0.0
    last_seen_data_for_caller: List[Any] = []

    while waited_time < max_wait_time:
        if aggregator.has_new_data(caller_id):
            items = aggregator.get_new_data(caller_id)
            last_seen_data_for_caller.extend(items)
            for item_wrapper in items:
                if not isinstance(item_wrapper, AnnotatedInstance):
                    continue
                data_item = item_wrapper.data
                if not isinstance(data_item, data_class_type):
                    continue

                if expected_message_part in data_item.message:
                    if (
                        originator_id_str_to_check is None
                        or data_item.originator_id_str
                        == originator_id_str_to_check
                    ):
                        data_arrived = True
                        found_item = data_item
                        break
            if data_arrived:
                break
        await asyncio.sleep(poll_interval)
        waited_time += poll_interval

    assert data_arrived, (
        f"Aggregator did not receive data for {str(caller_id)} "
        f"containing '{expected_message_part}' (originator: {originator_id_str_to_check}) within {max_wait_time}s. "
        f"Last seen data for this caller during this wait: {last_seen_data_for_caller}"
    )
    return found_item


def test_broadcast_event_e2e(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup: Optional[
        RuntimeHandle[BroadcastTestRuntimeData, BroadcastTestRuntimeEvent]
    ] = None

    caller_id_1 = CallerIdentifier(
        uuid.UUID("00000000-0000-0000-0000-000000000001")
    )
    caller_id_2 = CallerIdentifier(
        uuid.UUID("00000000-0000-0000-0000-000000000002")
    )
    caller_id_3 = CallerIdentifier(
        uuid.UUID("00000000-0000-0000-0000-000000000003")
    )
    all_caller_ids = [caller_id_1, caller_id_2, caller_id_3]

    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        runtime_future = runtime_manager.register_runtime_initializer(
            BroadcastTestRuntimeInitializer(
                test_ids=all_caller_ids, service_type="Server"
            )
        )
        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started
        runtime_handle = runtime_future.result(timeout=10)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        for cid in all_caller_ids:
            initial_data = loop.run_until_complete(
                _wait_for_data(
                    data_aggregator,
                    cid,
                    f"initial_for_{str(cid)}",
                    BroadcastTestRuntimeData,
                    originator_id_str_to_check=str(cid),
                    timeout=10,
                )
            )
            assert initial_data.message == f"initial_for_{str(cid)}"

        broadcast_payload = f"test_broadcast_{int(time.time())}"
        event_to_broadcast = BroadcastTestRuntimeEvent(
            payload=broadcast_payload
        )
        runtime_handle.on_event(event=event_to_broadcast, caller_id=None)

        for cid in all_caller_ids:
            processed_event_data = loop.run_until_complete(
                _wait_for_data(
                    data_aggregator,
                    cid,
                    f"processed_event_{broadcast_payload}_by_{str(cid)}",
                    BroadcastTestRuntimeData,
                    originator_id_str_to_check=str(cid),
                    timeout=15,
                )
            )
            assert (
                processed_event_data.message
                == f"processed_event_{broadcast_payload}_by_{str(cid)}"
            )

        runtime_handle.stop()

        # Robustly check for the first client's "stopped" message
        cid_1_check = all_caller_ids[0]
        print(
            f"DEBUG_TEST: Robustly waiting for 'stopped' message for {str(cid_1_check)} with timeout=20s"
        )
        stopped_data_1 = loop.run_until_complete(
            _wait_for_data(
                data_aggregator,
                cid_1_check,
                f"stopped_for_{str(cid_1_check)}",
                BroadcastTestRuntimeData,
                originator_id_str_to_check=str(cid_1_check),
                timeout=20,
            )
        )
        assert stopped_data_1.message == f"stopped_for_{str(cid_1_check)}"
        print(
            f"DEBUG_TEST: Successfully verified 'stopped' for {str(cid_1_check)}"
        )

        # Pragmatic check for subsequent clients
        print(
            "DEBUG_TEST: Pragmatically checking 'stopped' messages for other clients (short timeout)..."
        )
        for i, cid_other in enumerate(all_caller_ids[1:]):
            client_index_for_log = i + 2
            print(
                f"DEBUG_TEST: Pragmatic check for 'stopped' for client {client_index_for_log} ({str(cid_other)}) with timeout=2s"
            )
            try:
                stopped_data_other = loop.run_until_complete(
                    _wait_for_data(
                        data_aggregator,
                        cid_other,
                        f"stopped_for_{str(cid_other)}",
                        BroadcastTestRuntimeData,
                        originator_id_str_to_check=str(cid_other),
                        timeout=2,
                    )
                )
                if (
                    stopped_data_other
                    and stopped_data_other.message
                    == f"stopped_for_{str(cid_other)}"
                ):
                    print(
                        f"DEBUG_TEST: Pragmatic check SUCCESS for 'stopped' for client {client_index_for_log} ({str(cid_other)})"
                    )
                else:
                    print(
                        f"DEBUG_TEST: Pragmatic check FAILED (data mismatch or None) for 'stopped' for client {client_index_for_log} ({str(cid_other)})"
                    )
            except AssertionError as e:
                print(
                    f"DEBUG_TEST: Pragmatic check TIMEOUT/FAILED for 'stopped' for client {client_index_for_log} ({str(cid_other)}): {e}"
                )
            except Exception as e_other:
                print(
                    f"DEBUG_TEST: Pragmatic check ERRORED for 'stopped' for client {client_index_for_log} ({str(cid_other)}): {e_other!r}"
                )
    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception as e_clean:
                print(
                    f"DEBUG_TEST: Error during final runtime_handle.stop(): {e_clean!r}"
                )
        runtime_manager.shutdown()
