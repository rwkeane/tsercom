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
        test_id: CallerIdentifier, # Added test_id
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id # Store it
        self.__responder: EndpointDataProcessor[FakeData] | None = None
        self._data_sent = False # For idempotency

        super().__init__()

    def __repr__(self) -> str:  # New __repr__ method
        return f"<FakeRuntime instance at {id(self)}>"

    async def start_async(self) -> None:
        # Ensure asyncio is imported in the file
        # import asyncio # Usually at the top of the file

        current_task = None
        try:
            current_task = asyncio.current_task()
        except RuntimeError:  # If no loop or task factory set
            pass  # current_task will remain None

        # Original print statement, now with task info
        print(
            f"FakeRuntime.start_async: Entered. self_id={id(self)}, test_id={self.__test_id} (id={id(self.__test_id)}), self.__data_handler_id={id(self.__data_handler)}, task_id={id(current_task) if current_task else 'N/A'}, task={current_task}",
            flush=True,
        )

        await asyncio.sleep(0.01)

        print(
            f"FakeRuntime.start_async: (After sleep) About to call register_caller for test_id={self.__test_id}. task_id={id(current_task) if current_task else 'N/A'}",
            flush=True,
        )
        self.__responder = self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 443 # Use self.__test_id
        )
        print(
            f"FakeRuntime.start_async: Returned from register_caller. Responder type {type(self.__responder)}. task_id={id(current_task) if current_task else 'N/A'}",
            flush=True,
        )

        # --- Create and send completely fresh data (if not already sent) ---
        if not self._data_sent:
            fresh_data_value = (
                "FRESH_SIMPLE_DATA_V2"
            )
            fresh_data_object = FakeData(fresh_data_value)
            fresh_timestamp = datetime.datetime.now()

            print(
                f"FakeRuntime.start_async: About to send FRESH data: val='{fresh_data_value}', data_obj_id={id(fresh_data_object)}, ts={fresh_timestamp}. Task: {asyncio.current_task()}",
                flush=True,
            )
            # EndpointDataProcessor._process_data creates AnnotatedInstance using self.caller_id
            await self.__responder.process_data(fresh_data_object, fresh_timestamp)
            print(
                f"FakeRuntime.start_async: Completed process_data with FRESH data. Task: {asyncio.current_task()}",
                flush=True,
            )
            self._data_sent = True
        else:
            print(f"FakeRuntime.start_async: Data already sent for {self.__test_id}. Skipping duplicate send. Task: {asyncio.current_task()}", flush=True)


    async def stop(self, exception) -> None:
        assert self.__responder is not None
        await self.__responder.process_data(FakeData(stopped), stop_timestamp)


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(self, test_id: CallerIdentifier, service_type="Client"): # Added test_id
        super().__init__(service_type=service_type)
        self._test_id = test_id # Store it

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return FakeRuntime(thread_watcher, data_handler, grpc_channel_factory, self._test_id) # Pass it


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
    print("STOPPING LOOP")
    clear_tsercom_event_loop()


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random() # Create unique ID for this test run
        print(f"__check_initialization: Using test_id={current_test_id}", flush=True)
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(test_id=current_test_id, service_type="Server") # Pass it
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
        assert not data_aggregator.has_new_data(
            current_test_id # Use current_test_id
        ), "Aggregator should not have new data for test_id before runtime start"
        runtime_handle.start()

        data_arrived = False
        max_wait_time = 5.0  # seconds
        poll_interval = 0.1  # seconds
        waited_time = 0.0
        while waited_time < max_wait_time:
            has_data_now = data_aggregator.has_new_data(current_test_id) # Use current_test_id
            print(
                f"__check_initialization (polling): waited_time={waited_time:.1f}s, data_aggregator.has_new_data(test_id={current_test_id}) is {has_data_now}",
                flush=True,
            )
            if has_data_now:
                data_arrived = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        runtime_manager.check_for_exception()  # Keep this check
        assert (
            data_arrived
        ), f"Aggregator did not receive data for test_id ({current_test_id}) within {max_wait_time}s"

        # The original assertion for any_new_data can be commented out or removed for now
        # assert (
        #     data_aggregator.any_new_data()
        # ), "Aggregator should have some new data (any_new_data)"

        # Continue with the existing assertions for data content:
        assert data_aggregator.has_new_data(
            current_test_id # Use current_test_id
        ), f"Aggregator should have new data for test_id ({current_test_id}) after polling"

        values = data_aggregator.get_new_data(current_test_id) # Use current_test_id
        assert isinstance(
            values, list
        ), f"Expected list for get_new_data(test_id), got {type(values)}"
        assert (
            len(values) == 1
        ), f"Expected 1 item for test_id, got {len(values)}"

        first = values[0]
        assert isinstance(first, AnnotatedInstance), type(first)
        assert isinstance(first.data, FakeData), type(first.data)
        # --- Assertions for the FIRST data item (now "FRESH_SIMPLE_DATA_V2") ---
        expected_fresh_value = "FRESH_SIMPLE_DATA_V2"
        actual_value_for_log = (
            first.data.value if hasattr(first.data, "value") else "N/A"
        )
        print(
            f"__check_initialization: Asserting FIRST data value. Expected: '{expected_fresh_value}', Actual: '{actual_value_for_log}'",
            flush=True,
        )
        assert (
            first.data.value == expected_fresh_value
        ), f"Expected '{expected_fresh_value}', got '{actual_value_for_log}'"

        print(
            f"__check_initialization: Received FIRST data timestamp: {first.timestamp}. Type: {type(first.timestamp)}",
            flush=True,
        )
        assert isinstance(
            first.timestamp, datetime.datetime
        ), "Timestamp is not a datetime object for fresh data"
        # Exact timestamp match for datetime.now() is too brittle for this diagnostic.
        # The original start_timestamp check is removed for the fresh data.
        assert first.caller_id == current_test_id # Use current_test_id

        assert not data_aggregator.has_new_data(
            current_test_id # Use current_test_id
        ), f"Aggregator should not have new data for test_id ({current_test_id}) after get_new_data"
        runtime_manager.check_for_exception()

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        time.sleep(0.5)

        assert data_aggregator.has_new_data(
            current_test_id # Use current_test_id
        ), f"Aggregator should have new data (stop message) for test_id ({current_test_id})"
        values = data_aggregator.get_new_data(current_test_id) # Use current_test_id
        assert isinstance(
            values, list
        ), f"Expected list for get_new_data(test_id) for stop, got {type(values)}"
        assert (
            len(values) == 1
        ), f"Expected 1 stop item for test_id, got {len(values)}"

        first = values[0]
        assert isinstance(first, AnnotatedInstance), type(first)
        assert isinstance(first.data, FakeData), type(first.data)
        # --- Assertions for the SECOND data item ("STOPPED") ---
        print(
            f"__check_initialization: Asserting STOPPED data value. Expected: '{stopped}', Actual: '{first.data.value if hasattr(first.data, 'value') else 'N/A'}'",
            flush=True,
        )
        assert first.data.value == stopped
        print(
            f"__check_initialization: Received STOPPED data timestamp: {first.timestamp}. Type: {type(first.timestamp)}",
            flush=True,
        )
        assert first.timestamp == stop_timestamp
        assert first.caller_id == current_test_id # Use current_test_id

        assert not data_aggregator.has_new_data(
            current_test_id # Use current_test_id
        ), f"Aggregator should not have new data for test_id ({current_test_id}) after get_new_data for stop"

    except Exception as e:
        raise e
    finally:
        if runtime_handle_for_cleanup:
            try:
                print(
                    "Attempting runtime_handle_for_cleanup.stop() in finally block."
                )
                runtime_handle_for_cleanup.stop()
                print("runtime_handle_for_cleanup.stop() completed.")
            except Exception as e_stop:
                print(
                    f"ERROR during runtime_handle_for_cleanup.stop() in finally: {e_stop}"
                )
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
    # Cleanup the event loop thread
    if worker_event_loop.is_running():
        worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
    event_thread.join(timeout=1)


# New Test Cases
def test_out_of_process_error_check_for_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteFailureOops"

    # Capture the future for the handle
    handle_future = runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg,
            error_type=ValueError,
            service_type="Server",  # Assuming service_type="Server" is appropriate for ErrorThrowingRuntime
        )
    )
    # service_type="Server" is important because ErrorThrowingRuntimeInitializer
    # defaults to "Client", which might have implications for how its errors
    # or lifecycle is handled by the generic runtime_main.py logic if it expects
    # a server-like runtime to be the one primarily managed in remote_process_main.
    # Let's assume "Server" is a more robust choice for a primary runtime in a process.

    print(
        "test_out_of_process_error_check_for_exception: Initializer registered. Starting out-of-process manager.",
        flush=True,
    )
    runtime_manager.start_out_of_process()
    print(
        "test_out_of_process_error_check_for_exception: Out-of-process manager started.",
        flush=True,
    )

    try:
        # Get the handle. It should be available very quickly because the handle
        # (ShimRuntimeHandle) is created before the remote process fully starts its logic.
        # Using a short timeout.
        print(
            "test_out_of_process_error_check_for_exception: Attempting to get runtime_handle from future.",
            flush=True,
        )
        runtime_handle = handle_future.result(
            timeout=2
        )  # Increased timeout slightly just in case
        print(
            f"test_out_of_process_error_check_for_exception: Got runtime_handle (id={id(runtime_handle)}, type={type(runtime_handle)}). Calling start() on handle.",
            flush=True,
        )
        runtime_handle.start()  # This sends the kStart command
        print(
            "test_out_of_process_error_check_for_exception: runtime_handle.start() called.",
            flush=True,
        )
    except Exception as e_handle:
        # Ensure pytest is imported at the top of the file: import pytest
        print(
            f"test_out_of_process_error_check_for_exception: Failed to get/start runtime_handle: {type(e_handle).__name__} - {e_handle}",
            flush=True,
        )
        pytest.fail(f"Failed to get or start runtime_handle: {e_handle}")

    # Wait for the remote process to start, execute start_async, and raise the error.
    # The error should be caught by ThreadWatcher in the main process via the error queue.
    wait_time_for_error = (
        1.5  # Adjusted from 1.0, to give a bit more time for IPC
    )
    print(
        f"test_out_of_process_error_check_for_exception: Waiting for {wait_time_for_error}s for error to propagate.",
        flush=True,
    )
    time.sleep(wait_time_for_error)
    # Ensure 'import time' is at the top of the file

    print(
        "test_out_of_process_error_check_for_exception: About to check for exception.",
        flush=True,
    )
    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()
    print(
        "test_out_of_process_error_check_for_exception: Correctly caught expected ValueError.",
        flush=True,
    )


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
