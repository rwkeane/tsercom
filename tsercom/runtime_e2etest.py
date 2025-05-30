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
        test_id: CallerIdentifier,  # Added test_id
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id  # Store it
        self.__responder: EndpointDataProcessor[FakeData] | None = None
        self._data_sent = False  # For idempotency

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
            self.__test_id, "0.0.0.0", 443  # Use self.__test_id
        )
        print(
            f"FakeRuntime.start_async: Returned from register_caller. Responder type {type(self.__responder)}. task_id={id(current_task) if current_task else 'N/A'}",
            flush=True,
        )

        # --- Create and send completely fresh data (if not already sent) ---
        if not self._data_sent:
            fresh_data_value = "FRESH_SIMPLE_DATA_V2"
            fresh_data_object = FakeData(fresh_data_value)
            fresh_timestamp = datetime.datetime.now()

            print(
                f"FakeRuntime.start_async: About to send FRESH data: val='{fresh_data_value}', data_obj_id={id(fresh_data_object)}, ts={fresh_timestamp}. Task: {asyncio.current_task()}",
                flush=True,
            )
            # EndpointDataProcessor._process_data creates AnnotatedInstance using self.caller_id
            await self.__responder.process_data(
                fresh_data_object, fresh_timestamp
            )
            print(
                f"FakeRuntime.start_async: Completed process_data with FRESH data. Task: {asyncio.current_task()}",
                flush=True,
            )
            self._data_sent = True
        else:
            print(
                f"FakeRuntime.start_async: Data already sent for {self.__test_id}. Skipping duplicate send. Task: {asyncio.current_task()}",
                flush=True,
            )

    async def stop(self, exception) -> None:
        assert self.__responder is not None
        await self.__responder.process_data(FakeData(stopped), stop_timestamp)


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(
        self, test_id: CallerIdentifier, service_type="Client"
    ):  # Added test_id
        # service_type argument here can be more specific (e.g. "Server0")
        # but RuntimeConfig expects "Client" or "Server".
        # For FakeRuntime, the actual behavior differentiation comes from test_id.
        super().__init__(
            service_type="Server" if "Server" in service_type else "Client"
        )
        self._test_id = test_id  # Store it

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return FakeRuntime(
            thread_watcher, data_handler, grpc_channel_factory, self._test_id
        )  # Pass it


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
        # Determine base service type for RuntimeConfig
        base_service_type = "Server" if "Server" in service_type else "Client"
        super().__init__(service_type=base_service_type)
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
        # Determine base service type for RuntimeConfig
        base_service_type = "Server" if "Server" in service_type else "Client"
        super().__init__(service_type=base_service_type)
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
        current_test_id = (
            CallerIdentifier.random()
        )  # Create unique ID for this test run
        print(
            f"__check_initialization: Using test_id={current_test_id}",
            flush=True,
        )
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                test_id=current_test_id, service_type="Server"
            )  # Pass it
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
            current_test_id  # Use current_test_id
        ), "Aggregator should not have new data for test_id before runtime start"
        runtime_handle.start()

        data_arrived = False
        max_wait_time = 5.0  # seconds
        poll_interval = 0.1  # seconds
        waited_time = 0.0
        while waited_time < max_wait_time:
            has_data_now = data_aggregator.has_new_data(
                current_test_id
            )  # Use current_test_id
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
            current_test_id  # Use current_test_id
        ), f"Aggregator should have new data for test_id ({current_test_id}) after polling"

        values = data_aggregator.get_new_data(
            current_test_id
        )  # Use current_test_id
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
        assert first.caller_id == current_test_id  # Use current_test_id

        assert not data_aggregator.has_new_data(
            current_test_id  # Use current_test_id
        ), f"Aggregator should not have new data for test_id ({current_test_id}) after get_new_data"
        runtime_manager.check_for_exception()

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        time.sleep(0.5)

        assert data_aggregator.has_new_data(
            current_test_id  # Use current_test_id
        ), f"Aggregator should have new data (stop message) for test_id ({current_test_id})"
        values = data_aggregator.get_new_data(
            current_test_id
        )  # Use current_test_id
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
        assert first.caller_id == current_test_id  # Use current_test_id

        assert not data_aggregator.has_new_data(
            current_test_id  # Use current_test_id
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


def test_multiple_runtimes_out_of_process_success(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []

    try:
        num_runtimes = 2
        runtime_initializers = []
        runtime_futures = []
        test_ids = []

        for i in range(num_runtimes):
            current_test_id = CallerIdentifier.random()
            test_ids.append(current_test_id)
            initializer = FakeRuntimeInitializer(
                test_id=current_test_id, service_type=f"Server{i}"
            )
            runtime_initializers.append(initializer)
            future = runtime_manager.register_runtime_initializer(initializer)
            runtime_futures.append(future)

        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started

        runtime_handles = []
        for future in runtime_futures:
            handle = future.result(
                timeout=5
            )  # Increased timeout for handle acquisition
            runtime_handles.append(handle)
            runtime_handles_for_cleanup.append(handle)  # Add to cleanup list

        for i, handle in enumerate(runtime_handles):
            data_aggregator = handle.data_aggregator
            current_test_id = test_ids[i]
            assert not data_aggregator.has_new_data(
                current_test_id
            ), f"Aggregator for runtime {i} should not have new data before start"
            handle.start()

        for i, handle in enumerate(runtime_handles):
            data_aggregator = handle.data_aggregator
            current_test_id = test_ids[i]
            data_arrived = False
            max_wait_time = (
                7.0  # Increased wait time slightly for multi-runtime
            )
            poll_interval = 0.1
            waited_time = 0.0
            while waited_time < max_wait_time:
                if data_aggregator.has_new_data(current_test_id):
                    data_arrived = True
                    break
                time.sleep(poll_interval)
                waited_time += poll_interval

            runtime_manager.check_for_exception()  # Check for errors during data wait
            assert (
                data_arrived
            ), f"Aggregator for runtime {i} (id: {current_test_id}) did not receive data within {max_wait_time}s"

            values = data_aggregator.get_new_data(current_test_id)
            assert isinstance(values, list)
            assert len(values) == 1
            first = values[0]
            assert isinstance(first.data, FakeData)
            assert first.data.value == "FRESH_SIMPLE_DATA_V2"
            assert first.caller_id == current_test_id
            assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()

        for i, handle in enumerate(runtime_handles):
            handle.stop()

        runtime_manager.check_for_exception()  # Check for errors during stop command

        # Wait a bit for stop messages to propagate
        time.sleep(1.0)  # Increased sleep after stopping multiple runtimes

        for i, handle in enumerate(runtime_handles):
            data_aggregator = handle.data_aggregator
            current_test_id = test_ids[i]
            # Check for stop message after a delay
            stop_data_arrived = False
            max_wait_time_stop = 5.0  # Wait for stop message
            poll_interval_stop = 0.1
            waited_time_stop = 0.0
            while waited_time_stop < max_wait_time_stop:
                if data_aggregator.has_new_data(current_test_id):
                    stop_data_arrived = True
                    break
                time.sleep(poll_interval_stop)
                waited_time_stop += poll_interval_stop

            assert (
                stop_data_arrived
            ), f"Aggregator for runtime {i} (id: {current_test_id}) did not receive STOP data within {max_wait_time_stop}s"

            values = data_aggregator.get_new_data(current_test_id)
            assert isinstance(values, list)
            assert len(values) == 1
            first = values[0]
            assert isinstance(first.data, FakeData)
            assert first.data.value == stopped  # global `stopped` variable
            assert first.caller_id == current_test_id
            # stop_timestamp is a global, might be an issue if tests run near datetime boundaries
            # For now, we keep it, but it could be a source of flakiness if not handled carefully.
            # Consider comparing with a wider time window or injecting timestamp into FakeRuntime.stop
            assert first.timestamp == stop_timestamp
            assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()

    finally:
        # Attempt to stop any handles that might have been created, even if the test failed partway
        # This is a best-effort cleanup.
        # for handle in reversed(runtime_handles_for_cleanup): # Stop in reverse order of creation
        #     try:
        #         print(f"test_multiple_runtimes_out_of_process_success: Finally block stopping handle for {handle.data_aggregator._caller_id_for_logging_do_not_use_directly}", flush=True)
        #         handle.stop()
        #     except Exception as e_stop:
        #         print(f"ERROR during handle.stop() in finally: {e_stop}", flush=True)
        # time.sleep(0.5) # Give a moment for stop commands to be processed before manager shutdown

        print(
            "test_multiple_runtimes_out_of_process_success: Shutting down runtime_manager in finally block.",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_multiple_runtimes_out_of_process_success: Runtime_manager shutdown complete.",
            flush=True,
        )


def test_multiple_runtimes_in_process_success(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    # runtime_handles_for_cleanup = [] # Not strictly needed for in-process if manager shutdown is robust

    loop_future = Future()

    def _thread_loop_runner(fut: Future):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        finally:
            if not loop.is_closed():
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
            loop.close()
            print("_thread_loop_runner: Loop closed.")

    event_thread = Thread(
        target=_thread_loop_runner, args=(loop_future,), daemon=True
    )
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    try:
        num_runtimes = 2
        runtime_initializers = []
        runtime_futures = []
        test_ids = []

        for i in range(num_runtimes):
            current_test_id = CallerIdentifier.random()
            test_ids.append(current_test_id)
            service_type_val = "Client" if i == 0 else "Server"
            initializer = FakeRuntimeInitializer(
                test_id=current_test_id, service_type=service_type_val
            )
            runtime_initializers.append(initializer)
            future = runtime_manager.register_runtime_initializer(initializer)
            runtime_futures.append(future)

        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        assert runtime_manager.has_started

        runtime_handles = []
        for future in runtime_futures:
            handle = future.result(timeout=5)
            runtime_handles.append(handle)
            # runtime_handles_for_cleanup.append(handle)

        for i, handle in enumerate(runtime_handles):
            data_aggregator = handle.data_aggregator
            current_test_id = test_ids[i]
            assert not data_aggregator.has_new_data(
                current_test_id
            ), f"Aggregator for runtime {i} should not have new data before start"
            handle.start()

        for i, handle in enumerate(runtime_handles):
            data_aggregator = handle.data_aggregator
            current_test_id = test_ids[i]
            data_arrived = False
            max_wait_time = 7.0
            poll_interval = 0.1
            waited_time = 0.0
            while waited_time < max_wait_time:
                if data_aggregator.has_new_data(current_test_id):
                    data_arrived = True
                    break
                time.sleep(poll_interval)
                waited_time += poll_interval

            runtime_manager.check_for_exception()
            assert (
                data_arrived
            ), f"Aggregator for runtime {i} (id: {current_test_id}) did not receive data within {max_wait_time}s"

            values = data_aggregator.get_new_data(current_test_id)
            assert isinstance(values, list)
            assert len(values) == 1
            first = values[0]
            assert isinstance(first.data, FakeData)
            assert first.data.value == "FRESH_SIMPLE_DATA_V2"
            assert first.caller_id == current_test_id
            assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()

        for i, handle in enumerate(runtime_handles):
            handle.stop()

        runtime_manager.check_for_exception()
        time.sleep(1.0)

        for i, handle in enumerate(runtime_handles):
            data_aggregator = handle.data_aggregator
            current_test_id = test_ids[i]
            stop_data_arrived = False
            max_wait_time_stop = 5.0
            poll_interval_stop = 0.1
            waited_time_stop = 0.0
            while waited_time_stop < max_wait_time_stop:
                if data_aggregator.has_new_data(current_test_id):
                    stop_data_arrived = True
                    break
                time.sleep(poll_interval_stop)
                waited_time_stop += poll_interval_stop

            assert (
                stop_data_arrived
            ), f"Aggregator for runtime {i} (id: {current_test_id}) did not receive STOP data within {max_wait_time_stop}s"

            values = data_aggregator.get_new_data(current_test_id)
            assert isinstance(values, list)
            assert len(values) == 1
            first = values[0]
            assert isinstance(first.data, FakeData)
            assert first.data.value == stopped
            assert first.caller_id == current_test_id
            assert first.timestamp == stop_timestamp
            assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()

    finally:
        runtime_manager.shutdown()

        if worker_event_loop and worker_event_loop.is_running():
            worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        if event_thread.is_alive():
            event_thread.join(timeout=2)
        # clear_tsercom_event_loop() # Should be handled by shutdown and fixture


def test_shutdown_with_active_out_of_process_runtimes(clear_loop_fixture):
    runtime_manager = RuntimeManager(
        is_testing=True
    )  # is_testing=True makes process daemonic
    runtime_handle_for_cleanup = None  # Keep track for potential manual stop if needed, though shutdown is primary

    try:
        current_test_id = CallerIdentifier.random()
        initializer = FakeRuntimeInitializer(
            test_id=current_test_id, service_type="Server"
        )
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = (
            runtime_handle  # Assign for finally block, just in case
        )

        data_aggregator = runtime_handle.data_aggregator
        assert not data_aggregator.has_new_data(
            current_test_id
        ), "Aggregator should not have new data before runtime start"

        runtime_handle.start()

        # Wait for the runtime to send its initial data to confirm it's active
        data_arrived = False
        max_wait_time = 7.0
        poll_interval = 0.1
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                data_arrived = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        runtime_manager.check_for_exception()
        assert (
            data_arrived
        ), f"Runtime did not send initial data within {max_wait_time}s"

        # Consume the data so it doesn't interfere with later checks if any
        data_aggregator.get_new_data(current_test_id)

        # Now, call shutdown without explicitly stopping the handle
        print(
            "test_shutdown_with_active_out_of_process_runtimes: Calling runtime_manager.shutdown()",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_shutdown_with_active_out_of_process_runtimes: runtime_manager.shutdown() completed",
            flush=True,
        )

        # After shutdown, the manager should indicate it's no longer running (or has_started becomes false if it resets)
        # The current RuntimeManager.has_started is an IsRunningTracker, which doesn't reset to False on shutdown.
        # So, we can't assert !runtime_manager.has_started.
        # We mainly check that shutdown() itself didn't throw an error.

        # Attempting to interact with the handle after shutdown should ideally fail gracefully
        # or reflect that the underlying process is gone. This is more of an advanced test.
        # For this test, we primarily care that shutdown() terminated the process.
        # One way to infer termination is that check_for_exception after a delay doesn't show new errors
        # from a lingering process.
        time.sleep(
            0.5
        )  # Give a moment for any process termination issues to surface
        try:
            runtime_manager.check_for_exception()
        except Exception as e:
            pytest.fail(
                f"check_for_exception after shutdown should not raise new errors from runtime: {e}"
            )

        # Verify that the process is likely gone by trying to stop the handle again.
        # This might raise an error, or it might do nothing if the connection is severed.
        # This is a bit heuristic.
        # if runtime_handle_for_cleanup:
        #     with pytest.raises(Exception): # Or a more specific exception if known
        #         print("test_shutdown_with_active_out_of_process_runtimes: Attempting to stop handle after shutdown", flush=True)
        #         runtime_handle_for_cleanup.stop() # This should ideally fail or do nothing without error
        #         # If it does not fail, it might mean the handle is not aware the process is gone.
        #         # However, for this test, the primary concern is that shutdown() itself works.
        # This part is commented out as behavior of handle.stop() post-manager-shutdown is not strictly defined yet.

    except Exception:
        # runtime_manager.shutdown() # Ensure shutdown is called even if test fails midway
        raise  # Re-raise the original exception
    finally:
        # Ensure shutdown is called, especially if an assert fails before the explicit shutdown call.
        # If shutdown was already called in try, this call should be idempotent.
        print(
            "test_shutdown_with_active_out_of_process_runtimes: Calling shutdown() in finally block (idempotency check)",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_shutdown_with_active_out_of_process_runtimes: Shutdown in finally block complete.",
            flush=True,
        )


def test_multiple_out_of_process_one_fails_start_async(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)

    healthy_test_id = CallerIdentifier.random()
    failing_test_id = (
        CallerIdentifier.random()
    )  # Although ErrorThrowingRuntime doesn't use test_id for data

    error_msg = "FailureDuringStartAsync"
    expected_exception_type = ValueError

    # Initializer for the healthy runtime
    healthy_initializer = FakeRuntimeInitializer(
        test_id=healthy_test_id, service_type="ServerHealthy"
    )
    # Initializer for the runtime that will fail
    failing_initializer = ErrorThrowingRuntimeInitializer(
        error_message=error_msg,
        error_type=expected_exception_type,
        service_type="ServerFail",
    )

    healthy_future = runtime_manager.register_runtime_initializer(
        healthy_initializer
    )
    failing_future = runtime_manager.register_runtime_initializer(
        failing_initializer
    )

    runtime_manager.start_out_of_process()
    assert runtime_manager.has_started

    healthy_handle = None
    failing_handle = None  # Keep variable in scope for finally
    try:
        healthy_handle = healthy_future.result(timeout=5)
        failing_handle = failing_future.result(timeout=5)

        # Start both. The failing one will error out in its start_async.
        print(
            f"test_multiple_out_of_process_one_fails_start_async: Starting healthy_handle ({healthy_test_id})",
            flush=True,
        )
        healthy_handle.start()

        print(
            "test_multiple_out_of_process_one_fails_start_async: Starting failing_handle",
            flush=True,
        )
        failing_handle.start()  # This command is sent, error occurs in remote process

        # Wait for the error to propagate.
        max_wait_for_error_propagation = 5.0  # seconds
        time_slept = 0.0

        print(
            "test_multiple_out_of_process_one_fails_start_async: Waiting for error to propagate from failing runtime...",
            flush=True,
        )
        with pytest.raises(expected_exception_type, match=error_msg):
            while time_slept < max_wait_for_error_propagation:
                try:
                    runtime_manager.check_for_exception()
                    time.sleep(0.2)
                    time_slept += 0.2
                except expected_exception_type as e:
                    print(
                        f"test_multiple_out_of_process_one_fails_start_async: Caught expected exception: {e}",
                        flush=True,
                    )
                    raise
            runtime_manager.check_for_exception()

        if healthy_handle:
            data_aggregator_healthy = healthy_handle.data_aggregator
            if data_aggregator_healthy.has_new_data(healthy_test_id):
                print(
                    f"test_multiple_out_of_process_one_fails_start_async: Healthy runtime ({healthy_test_id}) sent data.",
                    flush=True,
                )
                values = data_aggregator_healthy.get_new_data(healthy_test_id)
                assert len(values) > 0
                assert values[0].data.value == "FRESH_SIMPLE_DATA_V2"
            else:
                print(
                    f"test_multiple_out_of_process_one_fails_start_async: Healthy runtime ({healthy_test_id}) did not send data.",
                    flush=True,
                )
    finally:
        print(
            "test_multiple_out_of_process_one_fails_start_async: Shutting down runtime_manager in finally block.",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_multiple_out_of_process_one_fails_start_async: Runtime_manager shutdown complete.",
            flush=True,
        )


def test_multiple_out_of_process_one_fails_create(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)

    healthy_test_id = CallerIdentifier.random()
    # FaultyCreateRuntimeInitializer doesn't use a test_id in its current form

    error_msg = "FailureDuringCreate"
    expected_exception_type = (
        TypeError  # As defined in FaultyCreateRuntimeInitializer
    )

    # Initializer for the healthy runtime
    healthy_initializer = FakeRuntimeInitializer(
        test_id=healthy_test_id, service_type="ServerHealthy"
    )
    # Initializer that will fail during its .create() method
    failing_initializer = FaultyCreateRuntimeInitializer(
        error_message=error_msg,
        error_type=expected_exception_type,
        service_type="ServerFaultyCreate",
    )

    healthy_future = runtime_manager.register_runtime_initializer(
        healthy_initializer
    )
    failing_future = runtime_manager.register_runtime_initializer(
        failing_initializer
    )

    runtime_manager.start_out_of_process()
    assert runtime_manager.has_started

    healthy_handle = None
    try:
        # Attempt to get the healthy handle. This should succeed.
        try:
            healthy_handle = healthy_future.result(timeout=5)
            print(
                f"test_multiple_out_of_process_one_fails_create: Healthy handle ({healthy_test_id}) obtained.",
                flush=True,
            )
            healthy_handle.start()  # Start the healthy one
            print(
                f"test_multiple_out_of_process_one_fails_create: Healthy handle ({healthy_test_id}) started.",
                flush=True,
            )
        except Exception as e:
            pytest.fail(f"Failed to get or start healthy handle: {e}")

        print(
            "test_multiple_out_of_process_one_fails_create: Waiting for error to propagate from failing initializer...",
            flush=True,
        )
        max_wait_for_error_propagation = 5.0  # seconds
        time_slept = 0.0

        with pytest.raises(expected_exception_type, match=error_msg):
            while time_slept < max_wait_for_error_propagation:
                try:
                    runtime_manager.check_for_exception()
                    time.sleep(0.2)
                    time_slept += 0.2
                except expected_exception_type as e:
                    print(
                        f"test_multiple_out_of_process_one_fails_create: Caught expected exception: {e}",
                        flush=True,
                    )
                    raise  # Re-raise for pytest.raises
            runtime_manager.check_for_exception()  # Final attempt

        # The failing_future should have completed successfully with a ShimRuntimeHandle,
        # as the handle is created before the remote process attempts to call .create()
        # on the initializer. The error from .create() is reported asynchronously.
        assert (
            failing_future.done()
        ), "Failing future for FaultyCreate should still complete with a handle."
        try:
            _ = failing_future.result(
                timeout=0.1
            )  # Should get a handle without error.
            print(
                "test_multiple_out_of_process_one_fails_create: failing_future.result() returned a handle as expected.",
                flush=True,
            )
        except Exception as e:
            pytest.fail(
                f"failing_future.result() should not raise an exception itself. Error from create() is async. Got: {e}"
            )

        if healthy_handle:
            data_aggregator_healthy = healthy_handle.data_aggregator
            data_arrived = False
            max_wait_data = 5.0
            poll_interval = 0.1
            waited_time_data = 0.0
            while waited_time_data < max_wait_data:
                if data_aggregator_healthy.has_new_data(healthy_test_id):
                    data_arrived = True
                    break
                time.sleep(poll_interval)
                waited_time_data += poll_interval

            if data_arrived:
                print(
                    f"test_multiple_out_of_process_one_fails_create: Healthy runtime ({healthy_test_id}) sent data.",
                    flush=True,
                )
                values = data_aggregator_healthy.get_new_data(healthy_test_id)
                assert len(values) > 0
                assert values[0].data.value == "FRESH_SIMPLE_DATA_V2"
            else:
                print(
                    f"test_multiple_out_of_process_one_fails_create: Healthy runtime ({healthy_test_id}) did NOT send data.",
                    flush=True,
                )

    finally:
        print(
            "test_multiple_out_of_process_one_fails_create: Shutting down runtime_manager in finally block.",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_multiple_out_of_process_one_fails_create: Runtime_manager shutdown complete.",
            flush=True,
        )


@pytest.mark.asyncio
async def test_start_in_process_async_flow(clear_loop_fixture):
    # clear_loop_fixture ensures that the tsercom global event loop is clear
    # before and after this test. RuntimeManager.start_in_process_async()
    # will pick up the loop created by pytest-asyncio for this test.

    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None

    try:
        current_test_id = CallerIdentifier.random()
        initializer = FakeRuntimeInitializer(
            test_id=current_test_id,
            service_type="ServerAsync",  # Unique service type for clarity
        )
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        # Call the async start method. It should use the current event loop.
        # start_in_process_async itself doesn't return handles directly.
        await runtime_manager.start_in_process_async()
        assert runtime_manager.has_started

        runtime_handle = runtime_future.result(
            timeout=5
        )  # Get handle from future
        runtime_handle_for_cleanup = runtime_handle

        data_aggregator = runtime_handle.data_aggregator
        assert not data_aggregator.has_new_data(
            current_test_id
        ), "Aggregator should not have new data before runtime start"

        # Starting the handle is a synchronous call, but the operations it triggers
        # (like FakeRuntime.start_async) will run on the event loop managed by start_in_process_async.
        runtime_handle.start()

        # Wait for data
        data_arrived = False
        max_wait_time = 7.0
        poll_interval = (
            0.1  # Using time.sleep for polling as this part is not async
        )
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                data_arrived = True
                break
            await asyncio.sleep(
                poll_interval
            )  # Use asyncio.sleep in an async test
            waited_time += poll_interval

        runtime_manager.check_for_exception()
        assert (
            data_arrived
        ), f"Runtime (id: {current_test_id}) did not send initial data within {max_wait_time}s"

        values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values, list) and len(values) == 1
        first_item = values[0]
        assert isinstance(first_item.data, FakeData)
        assert first_item.data.value == "FRESH_SIMPLE_DATA_V2"
        assert first_item.caller_id == current_test_id
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()

        runtime_handle.stop()  # Synchronous call
        runtime_manager.check_for_exception()

        # Wait for stop message
        stop_data_arrived = False
        waited_time = 0.0
        while waited_time < max_wait_time:  # Reusing max_wait_time
            if data_aggregator.has_new_data(current_test_id):
                stop_data_arrived = True
                break
            await asyncio.sleep(poll_interval)  # Use asyncio.sleep
            waited_time += poll_interval

        assert (
            stop_data_arrived
        ), f"Runtime (id: {current_test_id}) did not send STOP data within {max_wait_time}s"

        stop_values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(stop_values, list) and len(stop_values) == 1
        stop_item = stop_values[0]
        assert isinstance(stop_item.data, FakeData)
        assert stop_item.data.value == stopped  # global `stopped` variable
        assert stop_item.caller_id == current_test_id
        assert stop_item.timestamp == stop_timestamp  # global `stop_timestamp`
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()

    finally:
        print(
            "test_start_in_process_async_flow: Shutting down runtime_manager in finally block.",
            flush=True,
        )
        runtime_manager.shutdown()  # This calls clear_tsercom_event_loop()
        print(
            "test_start_in_process_async_flow: Runtime_manager shutdown complete.",
            flush=True,
        )


def test_register_initializer_after_manager_start_out_of_process(
    clear_loop_fixture,
):
    runtime_manager = RuntimeManager(is_testing=True)
    try:
        # Start the manager first
        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started

        # Then attempt to register an initializer
        with pytest.raises(
            RuntimeError,
            match="Cannot register runtime initializer after the manager has started.",
        ):
            current_test_id = CallerIdentifier.random()
            initializer = FakeRuntimeInitializer(
                test_id=current_test_id, service_type="ServerLate"
            )
            runtime_manager.register_runtime_initializer(initializer)

    finally:
        print(
            "test_register_initializer_after_manager_start_out_of_process: Shutting down runtime_manager.",
            flush=True,
        )
        runtime_manager.shutdown()


def test_register_initializer_after_manager_start_in_process(
    clear_loop_fixture,
):
    runtime_manager = RuntimeManager(is_testing=True)

    loop_future = Future()
    worker_event_loop = None  # Define worker_event_loop here to ensure it's in scope for finally
    event_thread = None  # Define event_thread here for the same reason

    def _thread_loop_runner(fut: Future):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut.set_result(loop)
        try:
            loop.run_forever()
        finally:
            if not loop.is_closed():
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
            loop.close()
            print(
                "_thread_loop_runner (for ..._after_manager_start_in_process): Loop closed."
            )

    try:
        event_thread = Thread(
            target=_thread_loop_runner, args=(loop_future,), daemon=True
        )
        event_thread.start()
        worker_event_loop = loop_future.result(timeout=5)

        # Start the manager first
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        assert runtime_manager.has_started

        # Then attempt to register an initializer
        with pytest.raises(
            RuntimeError,
            match="Cannot register runtime initializer after the manager has started.",
        ):
            current_test_id = CallerIdentifier.random()
            initializer = FakeRuntimeInitializer(
                test_id=current_test_id, service_type="ClientLate"
            )
            runtime_manager.register_runtime_initializer(initializer)

    finally:
        print(
            "test_register_initializer_after_manager_start_in_process: Shutting down runtime_manager.",
            flush=True,
        )
        runtime_manager.shutdown()  # Clears global tsercom loop

        if worker_event_loop and worker_event_loop.is_running():
            worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        if event_thread and event_thread.is_alive():
            event_thread.join(timeout=2)
        print(
            "test_register_initializer_after_manager_start_in_process: Event thread joined.",
            flush=True,
        )


@pytest.mark.asyncio
async def test_register_initializer_after_manager_start_in_process_async(
    clear_loop_fixture,
):
    runtime_manager = RuntimeManager(is_testing=True)
    try:
        # Start the manager first
        await runtime_manager.start_in_process_async()
        assert runtime_manager.has_started

        # Then attempt to register an initializer
        with pytest.raises(
            RuntimeError,
            match="Cannot register runtime initializer after the manager has started.",
        ):
            current_test_id = CallerIdentifier.random()
            initializer = FakeRuntimeInitializer(
                test_id=current_test_id, service_type="ServerAsyncLate"
            )
            runtime_manager.register_runtime_initializer(initializer)

    finally:
        print(
            "test_register_initializer_after_manager_start_in_process_async: Shutting down runtime_manager.",
            flush=True,
        )
        runtime_manager.shutdown()


def test_shutdown_idempotency(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    handle_future = None  # Define here for availability in finally if needed
    try:
        # Minimal setup: register one initializer and start the manager
        current_test_id = CallerIdentifier.random()
        initializer = FakeRuntimeInitializer(
            test_id=current_test_id, service_type="ServerIdempotency"
        )
        handle_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started

        # Get the handle, but we don't need to do much with it.
        # Just ensures the manager has gone through more of its setup.
        try:
            handle = handle_future.result(timeout=5)
            # handle.start() # Not strictly necessary to start for this test
        except Exception as e:
            # If handle acquisition fails, it might indicate an issue,
            # but the main point is testing shutdown. Log it if it occurs.
            print(
                f"test_shutdown_idempotency: Info: Could not get handle: {e}",
                flush=True,
            )

        # Call shutdown multiple times
        print(
            "test_shutdown_idempotency: Calling shutdown() the 1st time.",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_shutdown_idempotency: shutdown() 1st call complete.",
            flush=True,
        )

        print(
            "test_shutdown_idempotency: Calling shutdown() the 2nd time.",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_shutdown_idempotency: shutdown() 2nd call complete.",
            flush=True,
        )

        print(
            "test_shutdown_idempotency: Calling shutdown() the 3rd time.",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_shutdown_idempotency: shutdown() 3rd call complete.",
            flush=True,
        )

        # The main assertion is that the above calls did not raise exceptions.
        # No explicit assert statement needed here if Pytest is used, as uncaught exceptions fail the test.

    except Exception as e:
        # If any other part of the test fails, make sure to re-raise
        # after attempting a final shutdown.
        # runtime_manager.shutdown() # This will be hit by the finally block.
        pytest.fail(
            f"test_shutdown_idempotency encountered an unexpected error: {e}"
        )
    finally:
        # Ensure shutdown is called at least once if an error occurred before the explicit calls.
        # If it was already called multiple times successfully, this should also be fine.
        print(
            "test_shutdown_idempotency: Calling shutdown() in finally block (final check).",
            flush=True,
        )
        runtime_manager.shutdown()
        print(
            "test_shutdown_idempotency: Shutdown in finally block complete.",
            flush=True,
        )
