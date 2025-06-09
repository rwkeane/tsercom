"""End-to-end tests for Tsercom runtime initialization, data flow, and error handling."""

import asyncio
import datetime
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import Optional, List, Dict

import pytest

from tsercom.api.runtime_manager import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.event_instance import EventInstance
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
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
FRESH_SIMPLE_DATA_V2 = "FRESH_SIMPLE_DATA_V2"

start_timestamp = datetime.datetime.now(
    datetime.timezone.utc
) - datetime.timedelta(hours=10)
stop_timestamp = datetime.datetime.now(
    datetime.timezone.utc
) + datetime.timedelta(minutes=20)


class FakeData:
    def __init__(self, val: str):
        self.__val = val

    @property
    def value(self):
        return self.__val


class FakeEvent(EventInstance[dict]):
    """Fake event for testing. EventInstance.data will store a dictionary."""

    def __init__(
        self,
        data: dict,
        caller_id: Optional[CallerIdentifier] = None,
        timestamp: Optional[datetime.datetime] = None,
    ):
        super().__init__(
            data=data,
            caller_id=(
                caller_id
                if caller_id is not None
                else CallerIdentifier.random()
            ),
            timestamp=(
                timestamp
                if timestamp is not None
                else datetime.datetime.now(datetime.timezone.utc)
            ),
        )
        self.event_value = data.get("value", data.get("type", str(data)))


class FakeRuntime(Runtime):  # This is now the multi-ID capable FakeRuntime
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
        caller_ids: List[CallerIdentifier],
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__caller_ids = caller_ids

        self.__responders_map: Dict[
            CallerIdentifier, EndpointDataProcessor[FakeData, FakeEvent]
        ] = {}
        self.__event_processing_tasks: List[asyncio.Task] = []

        self._start_async_called = False
        super().__init__()

    def __repr__(self) -> str:
        return f"<FakeRuntime instance for {self.__caller_ids} at {id(self)}>"

    async def start_async(self) -> None:
        if self._start_async_called:
            print(
                f"FakeRuntime for {self.__caller_ids}: start_async already called. Skipping."
            )
            return
        self._start_async_called = True
        print(f"FakeRuntime for {self.__caller_ids}: start_async called.")

        for cid_idx, cid in enumerate(self.__caller_ids):
            print(
                f"FakeRuntime: Registering responder for CID {cid} ({cid_idx+1}/{len(self.__caller_ids)})..."
            )
            responder = await self.__data_handler.register_caller(
                cid, f"fake_endpoint_for_{cid}", 12345 + cid_idx
            )
            self.__responders_map[cid] = responder
            print(f"FakeRuntime: Responder registered for CID {cid}.")

            print(
                f"FakeRuntime: Sending initial data for CID {cid}: {FRESH_SIMPLE_DATA_V2}"
            )
            fresh_data_object = FakeData(FRESH_SIMPLE_DATA_V2)
            fresh_timestamp = datetime.datetime.now(datetime.timezone.utc)
            await responder.process_data(fresh_data_object, fresh_timestamp)
            print(f"FakeRuntime: Initial data sent for CID {cid}.")

            async def process_events_for_cid(
                current_cid: CallerIdentifier,
                current_responder: EndpointDataProcessor,
            ):
                print(
                    f"FakeRuntime: Event processing loop started for CID {current_cid}."
                )
                try:
                    while True:
                        async for event_batch in current_responder:
                            print(
                                f"FakeRuntime (CID {current_cid}): Received event batch: {event_batch}"
                            )
                            for event_item in event_batch:
                                received_event_payload = event_item.data

                                print(
                                    f"FakeRuntime (CID {current_cid}): Raw event_item.data type: {type(received_event_payload)}"
                                )
                                print(
                                    f"FakeRuntime (CID {current_cid}): Raw event_item.data value: {received_event_payload}"
                                )

                                if isinstance(
                                    received_event_payload, FakeEvent
                                ):
                                    print(
                                        f"FakeRuntime (CID {current_cid}): It IS a FakeEvent instance."
                                    )
                                    actual_data_dict = (
                                        received_event_payload.data
                                    )
                                    print(
                                        f"FakeRuntime (CID {current_cid}): FakeEvent.data (payload_dict) type: {type(actual_data_dict)}"
                                    )
                                    print(
                                        f"FakeRuntime (CID {current_cid}): FakeEvent.data (payload_dict) value: {actual_data_dict}"
                                    )

                                    if (
                                        isinstance(actual_data_dict, dict)
                                        and actual_data_dict.get("type")
                                        == "broadcast_test_event"
                                    ):
                                        ack_data_val = (
                                            f"event_received_by_{current_cid}"
                                        )
                                        ack_data = FakeData(ack_data_val)
                                        print(
                                            f"FakeRuntime (CID {current_cid}): Broadcast FakeEvent (type='{actual_data_dict.get('type')}') received. Sending ack: {ack_data_val}"
                                        )
                                        await current_responder.process_data(
                                            ack_data,
                                            datetime.datetime.now(
                                                datetime.timezone.utc
                                            ),
                                        )
                                    else:
                                        print(
                                            f"FakeRuntime (CID {current_cid}): Is FakeEvent, but not broadcast_test_event. Type: {actual_data_dict.get('type') if isinstance(actual_data_dict, dict) else 'N/A'}"
                                        )
                                else:
                                    print(
                                        f"FakeRuntime (CID {current_cid}): It is NOT a FakeEvent instance. Type was {type(received_event_payload)}"
                                    )
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    print(
                        f"FakeRuntime (CID {current_cid}): Event processing loop cancelled."
                    )
                    raise
                except Exception as e:
                    print(
                        f"FakeRuntime (CID {current_cid}): Exception in event processing loop: {e}"
                    )
                    await asyncio.sleep(0.1)

            task = asyncio.create_task(process_events_for_cid(cid, responder))
            self.__event_processing_tasks.append(task)
            print(
                f"FakeRuntime: Event processing task created for CID {cid}: {task}"
            )

        print(f"FakeRuntime for {self.__caller_ids}: start_async finished.")

    async def stop(self, exception) -> None:
        print(
            f"FakeRuntime for {self.__caller_ids}: Stop called. Exception: {exception}"
        )

        for task in self.__event_processing_tasks:
            if not task.done():
                print(f"FakeRuntime: Cancelling event processing task {task}")
                task.cancel()
                try:
                    await task
                    print(
                        f"FakeRuntime: Event processing task {task} awaited successfully after cancel."
                    )
                except asyncio.CancelledError:
                    print(
                        f"FakeRuntime: Event processing task {task} confirmed cancelled."
                    )
                except Exception as e:
                    print(
                        f"FakeRuntime: Exception while awaiting cancelled event task {task}: {e}"
                    )
        self.__event_processing_tasks.clear()

        for cid, responder in self.__responders_map.items():
            if responder is not None:
                print(f"FakeRuntime: Sending 'stopped' data for CID {cid}.")
                await responder.process_data(FakeData(stopped), stop_timestamp)

        print(f"FakeRuntime for {self.__caller_ids} fully stopped.")

    def on_event(
        self, event: FakeEvent, caller_id: Optional[CallerIdentifier] = None
    ) -> None:
        print(
            f"FakeRuntime for {self.__caller_ids}: on_event directly called (caller_id: {caller_id}). This is unusual for E2E tests."
        )
        if self.__data_handler:
            serializable_event = SerializableAnnotatedInstance(
                data=event.data, caller_id=caller_id
            )
            main_event_source = getattr(
                self.__data_handler,
                "_RuntimeDataHandlerBase__event_source",
                None,
            )
            if main_event_source:
                main_event_source.on_available(serializable_event)
                print(
                    f"FakeRuntime ({self.__caller_ids}): Event directly injected to main event source for dispatch."
                )


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(
        self,
        caller_ids: List[CallerIdentifier],
        service_type="Client",
        raise_on_start=False,
        raise_on_event=False,
    ):
        super().__init__(service_type=service_type)  # Removed config_name
        self._caller_ids = caller_ids
        self.raise_on_start = raise_on_start
        self.raise_on_event = raise_on_event

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        runtime_instance = FakeRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self._caller_ids,
        )
        return runtime_instance


class ErrorThrowingRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,
        grpc_channel_factory: GrpcChannelFactory,
        test_id: CallerIdentifier,
        error_message="TestError",
        error_type=RuntimeError,
    ):
        super().__init__()
        self.test_id = test_id  # Keep test_id if used by ErrorThrowingRuntime logic/logging
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
        test_id: CallerIdentifier,
        error_message="TestError",
        error_type=RuntimeError,
        service_type="Client",
    ):
        super().__init__(service_type=service_type)  # Removed config_name
        self._test_id = test_id
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
            test_id=self._test_id,
            error_message=self.error_message,
            error_type=self.error_type,
        )


class FaultyCreateRuntimeInitializer(RuntimeInitializer):
    def __init__(
        self,
        test_id: CallerIdentifier,
        error_message="CreateFailed",
        error_type=TypeError,
        service_type="Client",
    ):
        super().__init__(service_type=service_type)  # Removed config_name
        self._test_id = test_id
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


@pytest.mark.asyncio
async def test_broadcast_event_e2e(clear_loop_fixture):
    del clear_loop_fixture
    print("Starting test_broadcast_event_e2e...")
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []

    caller_id_list = [CallerIdentifier.random(), CallerIdentifier.random()]
    print(f"Generated CallerIDs for the test: {caller_id_list}")

    try:
        print("Registering FakeRuntimeInitializer with multiple CallerIDs...")
        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                caller_ids=caller_id_list, service_type="Server"
            )
        )
        print("Initializer registered.")

        print("Starting runtime manager out of process...")
        runtime_manager.start_out_of_process()
        print("Runtime manager started.")

        print("Getting runtime handle...")
        runtime_handle = runtime_future.result(timeout=25)
        runtime_handles_for_cleanup.append(runtime_handle)
        print("Runtime handle obtained.")

        data_aggregator = runtime_handle.data_aggregator
        print("Using data aggregator from the runtime_handle.")

        print("Starting runtime_handle...")
        runtime_handle.start()
        print("runtime_handle started.")

        print("Waiting for initial data from all managed CallerIDs...")
        for cid in caller_id_list:
            await __wait_for_initial_data(
                data_aggregator, cid, timeout_seconds=30
            )
            data_aggregator.get_new_data(cid)
            print(f"Initial data received and cleared for {cid}.")

        print("Sending broadcast event...")
        broadcast_event_data = {
            "type": "broadcast_test_event",
            "content": "TestBroadcastEvent_E2E",
        }
        broadcast_event = FakeEvent(data=broadcast_event_data)
        runtime_handle.on_event(broadcast_event, caller_id=None)
        print(
            f"Broadcast event with data {broadcast_event_data} sent via runtime_handle with caller_id=None."
        )

        print("Waiting for acknowledgements from all managed CallerIDs...")
        received_acks = {cid: False for cid in caller_id_list}
        expected_acks = {
            cid: f"event_received_by_{cid}" for cid in caller_id_list
        }

        start_time = time.monotonic()
        all_acks_received = False
        while time.monotonic() - start_time < 40:
            for cid in caller_id_list:
                if not received_acks[cid] and data_aggregator.has_new_data(
                    cid
                ):
                    values = data_aggregator.get_new_data(cid)
                    for item in values:
                        if item.data.value == expected_acks[cid]:
                            received_acks[cid] = True
                            print(f"Received ack for {cid}: {item.data.value}")
                            break
            if all(received_acks.values()):
                all_acks_received = True
                print("All acknowledgements received.")
                break
            await asyncio.sleep(0.5)

        for cid in caller_id_list:
            assert received_acks[
                cid
            ], f"Acknowledgement '{expected_acks[cid]}' NOT received for {cid}."
        assert all_acks_received, "Not all acknowledgements were received."
        print("Assertions for acknowledgements passed.")

    except Exception as e:
        print(f"Exception in test_broadcast_event_e2e: {e}")
        raise
    finally:
        print("Broadcast test: Starting cleanup...")
        for handle in runtime_handles_for_cleanup:
            try:
                print("Stopping handle...")
                handle.stop()
                print("Handle stopped.")
            except Exception as e:
                print(f"Error stopping handle during cleanup: {e}")
        print("Shutting down runtime manager...")
        runtime_manager.shutdown()
        print("Cleanup complete for test_broadcast_event_e2e.")


async def __wait_for_initial_data(
    data_aggregator, caller_id: CallerIdentifier, timeout_seconds: float
):
    print(
        f"__wait_for_initial_data for {caller_id} using timeout {timeout_seconds}s..."
    )
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        if data_aggregator.has_new_data(caller_id):
            temp_data = data_aggregator.get_new_data(caller_id)
            is_initial = False
            for item in temp_data:
                if item.data.value == FRESH_SIMPLE_DATA_V2:
                    is_initial = True
                    break
            if is_initial:
                print(
                    f"Initial data '{FRESH_SIMPLE_DATA_V2}' found for {caller_id}."
                )
                return
            else:
                print(
                    f"Data found for {caller_id} but was not initial: {[d.data.value for d in temp_data]}. Re-checking."
                )
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"Initial data '{FRESH_SIMPLE_DATA_V2}' not found for {caller_id} within {timeout_seconds}s"
    )


def _wait_for_data_sync(
    data_aggregator,
    caller_id: CallerIdentifier,
    expected_value: str,
    timeout_seconds: float,
):
    print(
        f"_wait_for_data_sync for {caller_id} expecting '{expected_value}' using timeout {timeout_seconds}s..."
    )
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        if data_aggregator.has_new_data(caller_id):
            items = data_aggregator.get_new_data(caller_id)
            for item in items:
                if item.data.value == expected_value:
                    print(
                        f"Found expected data '{expected_value}' for {caller_id}."
                    )
                    return True
            print(
                f"Data found for {caller_id} but not '{expected_value}': {[item.data.value for item in items]}"
            )
        time.sleep(0.1)
    print(f"Timeout waiting for data '{expected_value}' for {caller_id}.")
    return False


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        print(f"__check_initialization: Using CallerID: {current_test_id}")
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                caller_ids=[current_test_id], service_type="Server"
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

        assert _wait_for_data_sync(
            data_aggregator, current_test_id, FRESH_SIMPLE_DATA_V2, 10
        ), f"Initial data '{FRESH_SIMPLE_DATA_V2}' not received for {current_test_id}"
        assert not data_aggregator.has_new_data(
            current_test_id
        ), "Data aggregator should be empty after get_new_data."

        runtime_manager.check_for_exception()
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        assert _wait_for_data_sync(
            data_aggregator, current_test_id, stopped, 10
        ), f"'stopped' data not received for {current_test_id}"
        assert not data_aggregator.has_new_data(current_test_id)

    except Exception as e:
        print(f"__check_initialization failed: {e}")
        raise
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
    test_id = CallerIdentifier.random()

    handle_future = runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            test_id=test_id,
            error_message=error_msg,
            error_type=ValueError,
            service_type="Server",
        )
    )
    runtime_manager.start_out_of_process()

    try:
        runtime_handle = handle_future.result(timeout=10)
        runtime_handle.start()
    except Exception as e_handle:
        print(f"Note: Failed to get or start runtime_handle: {e_handle}")

    time.sleep(1.5)
    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()


def test_out_of_process_error_run_until_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteRunUntilFailure"
    test_id = CallerIdentifier.random()
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            test_id=test_id, error_message=error_msg, error_type=RuntimeError
        )
    )
    runtime_manager.start_out_of_process()
    time.sleep(0.5)

    with pytest.raises(RuntimeError, match=error_msg):
        runtime_manager.check_for_exception()
        for _ in range(5):
            time.sleep(0.2)
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
    test_id = CallerIdentifier.random()
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            test_id=test_id, error_message=error_msg, error_type=ValueError
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        time.sleep(0.3)
        runtime_manager.check_for_exception()

    if worker_event_loop.is_running():
        worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
    event_thread.join(timeout=1)


def test_out_of_process_initializer_create_error(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "CreateOops"
    test_id = CallerIdentifier.random()
    runtime_manager.register_runtime_initializer(
        FaultyCreateRuntimeInitializer(
            test_id=test_id, error_message=error_msg, error_type=TypeError
        )
    )
    runtime_manager.start_out_of_process()
    time.sleep(1.0)
    with pytest.raises(TypeError, match=error_msg):
        runtime_manager.check_for_exception()


@pytest.mark.asyncio
async def test_multiple_runtimes_out_of_process(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []

    try:
        test_id_1 = CallerIdentifier.random()
        test_id_2 = CallerIdentifier.random()

        runtime_future_1 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                caller_ids=[test_id_1], service_type="Server"
            )
        )
        runtime_future_2 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                caller_ids=[test_id_2], service_type="Server"
            )
        )

        assert not runtime_future_1.done()
        assert not runtime_future_2.done()
        assert not runtime_manager.has_started

        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started
        assert runtime_future_1.done()
        assert runtime_future_2.done()

        runtime_manager.check_for_exception()

        runtime_handle_1 = runtime_future_1.result(timeout=10)
        runtime_handle_2 = runtime_future_2.result(timeout=10)
        runtime_handles_for_cleanup.extend(
            [runtime_handle_1, runtime_handle_2]
        )

        data_aggregator_1 = runtime_handle_1.data_aggregator
        data_aggregator_2 = runtime_handle_2.data_aggregator

        runtime_handle_1.start()
        runtime_handle_2.start()

        await __wait_for_initial_data(
            data_aggregator_1, test_id_1, timeout_seconds=10
        )
        data_aggregator_1.get_new_data(test_id_1)
        assert not data_aggregator_1.has_new_data(
            test_id_2
        ), "Aggregator 1 should not have data for test_id_2."

        await __wait_for_initial_data(
            data_aggregator_2, test_id_2, timeout_seconds=10
        )
        data_aggregator_2.get_new_data(test_id_2)
        assert not data_aggregator_2.has_new_data(
            test_id_1
        ), "Aggregator 2 should not have data for test_id_1."

        runtime_manager.check_for_exception()

        runtime_handle_1.stop()
        await __wait_for_stopped_data(
            data_aggregator_1, test_id_1, timeout_seconds=10
        )
        data_aggregator_1.get_new_data(test_id_1)

        runtime_handle_2.stop()
        await __wait_for_stopped_data(
            data_aggregator_2, test_id_2, timeout_seconds=10
        )
        data_aggregator_2.get_new_data(test_id_2)

        runtime_manager.check_for_exception()

    finally:
        for handle in runtime_handles_for_cleanup:
            try:
                handle.stop()
            except Exception:
                pass
        runtime_manager.shutdown()


async def __wait_for_stopped_data(data_aggregator, caller_id, timeout_seconds):
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        if data_aggregator.has_new_data(caller_id):
            values = data_aggregator.get_new_data(caller_id)
            for item in values:
                if item.data.value == stopped:
                    expected_stop_ts = stop_timestamp
                    item_ts = item.timestamp
                    if (
                        item_ts.tzinfo is None
                        and expected_stop_ts.tzinfo is not None
                    ):
                        expected_stop_ts = expected_stop_ts.replace(
                            tzinfo=None
                        )
                    elif (
                        item_ts.tzinfo is not None
                        and expected_stop_ts.tzinfo is None
                    ):
                        expected_stop_ts = expected_stop_ts.replace(
                            tzinfo=datetime.timezone.utc
                        )

                    assert item_ts == expected_stop_ts
                    assert item.caller_id == caller_id
                    print(f"'stopped' data found for {caller_id}")
                    return
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"'stopped' data not found for {caller_id} within {timeout_seconds}s"
    )


@pytest.mark.asyncio
async def test_client_type_runtime_in_process(clear_loop_fixture):
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

    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        runtime_manager.check_for_exception()

        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                caller_ids=[current_test_id], service_type="Client"
            )
        )

        assert not runtime_future.done()
        assert not runtime_manager.has_started

        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)

        assert runtime_manager.has_started
        assert runtime_future.done()

        runtime_manager.check_for_exception()
        runtime_handle = runtime_future.result()
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        assert not data_aggregator.has_new_data(current_test_id)
        runtime_handle.start()

        await __wait_for_initial_data(
            data_aggregator, current_test_id, timeout_seconds=10
        )
        data_aggregator.get_new_data(current_test_id)
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        await __wait_for_stopped_data(
            data_aggregator, current_test_id, timeout_seconds=10
        )
        assert not data_aggregator.has_new_data(current_test_id)

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
    error_msg = "InProcessCreateOops"
    test_id = CallerIdentifier.random()

    try:
        initializer = FaultyCreateRuntimeInitializer(
            test_id=test_id,
            error_message=error_msg,
            error_type=ValueError,
            service_type="Client",
        )
        runtime_manager.register_runtime_initializer(initializer)

        with pytest.raises(ValueError, match=error_msg):
            runtime_manager.start_in_process(
                runtime_event_loop=worker_event_loop
            )
            time.sleep(0.3)
            runtime_manager.check_for_exception()
    finally:
        runtime_manager.shutdown()
        if worker_event_loop.is_running():
            worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        event_thread.join(timeout=1)


def test_out_of_process_error_direct_run_until_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "DirectBlockError"
    error_type = ConnectionError
    thread_result_queue = Future()
    test_id = CallerIdentifier.random()

    def target_for_thread():
        try:
            runtime_manager.run_until_exception()
            thread_result_queue.set_result(None)
        except Exception as e:
            thread_result_queue.set_result(e)

    test_thread = None
    runtime_handle_for_cleanup = None

    try:
        initializer = ErrorThrowingRuntimeInitializer(
            test_id=test_id,
            error_message=error_msg,
            error_type=error_type,
            service_type="Server",
        )
        handle_future = runtime_manager.register_runtime_initializer(
            initializer
        )
        runtime_manager.start_out_of_process()
        runtime_handle = handle_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        runtime_handle.start()

        time.sleep(1.5)

        test_thread = Thread(target=target_for_thread, daemon=True)
        test_thread.start()
        test_thread.join(timeout=5.0)

        if test_thread.is_alive():
            pytest.fail("run_until_exception timed out / deadlocked.")

        result_from_thread = thread_result_queue.result(timeout=0)

        assert isinstance(
            result_from_thread, error_type
        ), f"Expected {error_type}, got {type(result_from_thread)}"
        assert (
            str(result_from_thread) == error_msg
        ), f"Expected '{error_msg}', got '{str(result_from_thread)}'"

    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception:
                pass
        runtime_manager.shutdown()
        if test_thread and test_thread.is_alive():
            print(
                "Warning: Test thread for run_until_exception did not exit cleanly."
            )
        elif test_thread:
            test_thread.join(timeout=1)
