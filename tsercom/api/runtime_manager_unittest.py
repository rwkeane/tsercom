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
from tsercom.runtime.client.client_runtime_data_handler import ClientRuntimeDataHandler
# Add if ServerRuntimeDataHandler is needed for completeness, though FakeRuntimeInitializer defaults to Client
# from tsercom.runtime.server.server_runtime_data_handler import ServerRuntimeDataHandler 
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
# FakeEvent is already defined, used for TEventType in AsyncPoller
# from tsercom.data.annotated_instance import AnnotatedInstance # For TDataType in RemoteDataReader - already imported
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
        test_id_override: CallerIdentifier,
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id_override
        self._start_async_called_count = 0
        self._register_caller_called_count = 0 # Also add for register_caller

        self.__responder: EndpointDataProcessor[FakeData] | None = None

        super().__init__()

    @property
    def test_id(self) -> CallerIdentifier:
        return self.__test_id

    async def start_async(self) -> None:
        self._start_async_called_count += 1
        if self._start_async_called_count > 1:
            print(f"WARNING: FakeRuntime ({self.__test_id}) start_async called {self._start_async_called_count} times!")
            # Consider raising a custom exception here to make the test fail differently if this happens
            # raise Exception(f"FakeRuntime ({self.__test_id}) start_async called multiple times!")

        await asyncio.sleep(0.01)

        # Wrap the register_caller part to count it too
        self._register_caller_called_count += 1
        if self._register_caller_called_count > 1:
             print(f"WARNING: FakeRuntime ({self.__test_id}) register_caller in start_async about to be called {self._register_caller_called_count} times!")

        self.__responder = self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 443
        )

        data = FakeData(started)
        try:
            await self.__responder.process_data(data, start_timestamp)
        except Exception as e:
            print(f"FakeRuntime ({self.__test_id}) error processing data in start_async: {e}")
            pass

    async def stop(self) -> None:
        assert self.__responder is not None
        await self.__responder.process_data(FakeData(stopped), stop_timestamp)


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(self, service_type="Client"):
        super().__init__(service_type=service_type)
        # self.last_created_test_id: CallerIdentifier | None = None # No longer needed here

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        current_test_id = CallerIdentifier.random()
        # self.last_created_test_id = current_test_id # No longer needed here
        return FakeRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            test_id_override=current_test_id,
        )

    def create_data_handler(
        self,
        thread_watcher: ThreadWatcher,
        data_reader: RemoteDataReader[AnnotatedInstance[FakeData]], # Matches ClientRuntimeDataHandler expectation
        event_poller: AsyncPoller[FakeEvent], # Matches ClientRuntimeDataHandler expectation
        is_testing: bool,
    ) -> RuntimeDataHandler[FakeData, FakeEvent]:
        if self.is_client():
            return ClientRuntimeDataHandler[FakeData, FakeEvent](
                thread_watcher=thread_watcher,
                data_reader=data_reader,
                event_source=event_poller, # ClientRuntimeDataHandler takes event_source
                is_testing=is_testing,
            )
        else:
            # If FakeRuntimeInitializer could be a server, implement this:
            # raise NotImplementedError("Server FakeRuntimeInitializer not fully implemented for data_handler")
            # For now, assume it's always client as per its __init__ default.
            # If a ServerRuntimeDataHandler is needed, it would be instantiated here.
            # For this specific test case, client is sufficient.
            # Fallback or error if not client, though __init__ defaults to "Client"
            raise ValueError("FakeRuntimeInitializer is expected to be Client type for this test context.")


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
    fake_initializer = FakeRuntimeInitializer(service_type="Client")
    runtime_future = runtime_manager.register_runtime_initializer(
        fake_initializer
    )

    assert not runtime_future.done()
    assert not runtime_manager.has_started
    init_call(runtime_manager)
    assert runtime_manager.has_started
    assert runtime_future.done()

    runtime_manager.check_for_exception()
    runtime = runtime_future.result()
    # assert isinstance(runtime, FakeRuntime), "Runtime instance should be FakeRuntime" # This will fail for wrapped runtimes
    data_aggregator = runtime.data_aggregator
    # runtime_internal_test_id = runtime.test_id  # Cannot access test_id directly from wrapper
    # assert runtime_internal_test_id is not None, "Runtime instance should have a test_id"

    # Initial check: no data for any ID (or specifically the global test_id if it's used elsewhere)
    assert not data_aggregator.any_new_data(), "Aggregator should not have any data before runtime start"

    runtime.start()
    time.sleep(0.5)  # Give time for data to be processed

    runtime_manager.check_for_exception()
    assert (
        data_aggregator.any_new_data()
    ), "Aggregator should have some new data after runtime start"

    # Discover the runtime_internal_test_id from the first piece of data
    # get_new_data(None) returns Dict[CallerIdentifier, List[TDataType]]
    all_new_data_map = data_aggregator.get_new_data(None) 
    assert all_new_data_map, "Should have received some data map from get_new_data(None)"

    runtime_internal_test_id = None
    first_annotated_instance = None
    
    for cid, instances in all_new_data_map.items():
        if instances: # Found a caller ID with data
            runtime_internal_test_id = cid
            first_annotated_instance = instances[0]
            # Verify the 'started' message as soon as we find it
            assert isinstance(first_annotated_instance.data, FakeData), type(first_annotated_instance.data)
            assert first_annotated_instance.data.value == started, \
                f"Expected first message to be '{started}', got '{first_annotated_instance.data.value}'"
            assert first_annotated_instance.timestamp == start_timestamp
            assert first_annotated_instance.caller_id == runtime_internal_test_id
            # If there was more than one instance for this cid, it's still in 'instances'
            # and data_aggregator.has_new_data(cid) would be true if instances[1:] existed.
            # The original test implies only one 'started' message.
            if len(instances) > 1:
                logging.warning(f"Found more than one initial message for {cid}: {instances}")
            break # Found the ID and the 'started' message
            
    assert runtime_internal_test_id is not None, "Could not determine runtime_internal_test_id from get_new_data(None)"
    assert first_annotated_instance is not None, "Could not get first annotated instance from get_new_data(None)"

    # get_new_data(None) clears all data it returns. So, for the specific runtime_internal_test_id, 
    # if only one message (the 'started' message) was present, it should now be clear.
    assert not data_aggregator.has_new_data(
         runtime_internal_test_id
    ), f"Aggregator should not have new data for {runtime_internal_test_id} after get_new_data(None) processed it"
    runtime_manager.check_for_exception()

    runtime.stop()
    runtime_manager.check_for_exception()

    time.sleep(0.5) # Give time for stop message

    assert data_aggregator.has_new_data(
        runtime_internal_test_id
    ), f"Aggregator should have new data (stop message) for {runtime_internal_test_id}"
    
    stop_values = data_aggregator.get_new_data(runtime_internal_test_id)
    assert isinstance(
        stop_values, list
    ), f"Expected list for get_new_data({runtime_internal_test_id}) for stop, got {type(stop_values)}"
    assert (
        len(stop_values) == 1
    ), f"Expected 1 stop item for {runtime_internal_test_id}, got {len(stop_values)}"

    stop_first = stop_values[0]
    assert isinstance(stop_first, AnnotatedInstance), type(stop_first)
    assert isinstance(stop_first.data, FakeData), type(stop_first.data)
    assert stop_first.data.value == stopped
    assert stop_first.timestamp == stop_timestamp
    assert stop_first.caller_id == runtime_internal_test_id

    assert not data_aggregator.has_new_data(
        runtime_internal_test_id
    ), f"Aggregator should not have new data for {runtime_internal_test_id} after get_new_data for stop"


def test_out_of_process_init(): # Renamed back
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
