import asyncio
from collections.abc import Callable
from concurrent.futures import Future
import datetime
from functools import partial
from threading import Thread
import time

from tsercom.api.runtime_manager import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
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
        # Wait to allow initialization to complete. Not needed in the current
        # infra, but its a reasonable assumption so just include this here to
        # document that it's allowed for future.
        await asyncio.sleep(0.01)

        # Register a fake endpoint to allow returning data.
        self.__responder = self.__data_handler.register_caller(
            test_id, "0.0.0.0", 443
        )

        # Return the data.
        data = FakeData(started)
        print(f"DEBUG: [FakeRuntime.start_async] Processing data: {data.value} for caller_id: {test_id}")
        try:
            await self.__responder.process_data(data, start_timestamp)
            print(f"DEBUG: [FakeRuntime.start_async] Successfully processed data: {data.value} for caller_id: {test_id}")
        except Exception as e:
            print(f"DEBUG: [FakeRuntime.start_async] Error processing data for {test_id}: {e}")


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


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    # 1. Object Instantiation
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_manager.check_for_exception()
    runtime_future = runtime_manager.register_runtime_initializer(
        FakeRuntimeInitializer(service_type="Client") # service_type="Client" to match ServerRuntimeDataHandler expectation
    )

    # Create the runtime.
    assert not runtime_future.done()
    assert not runtime_manager.has_started
    init_call(runtime_manager)
    assert runtime_manager.has_started
    assert runtime_future.done()

    # Start the runtime.
    runtime_manager.check_for_exception()
    runtime = runtime_future.result()
    data_aggregator = runtime.data_aggregator
    # Initial check: no data should be present for test_id specifically, or overall.
    # Using test_id is more precise if the aggregator could have other data.
    assert not data_aggregator.has_new_data(test_id), "Aggregator should not have new data for test_id before runtime start"
    runtime.start()

    # Sleep to allow the runtime to process the call and respond.
    time.sleep(0.5) # Keep increased sleep

    # Check that the runtime sent its "starting" message.
    runtime_manager.check_for_exception()
    assert data_aggregator.any_new_data(), "Aggregator should have some new data (any_new_data)" # General check
    assert data_aggregator.has_new_data(test_id), f"Aggregator should have new data for test_id ({test_id})" # Specific check

    values = data_aggregator.get_new_data(test_id) # Get data specifically for test_id
    assert isinstance(values, list), f"Expected list for get_new_data(test_id), got {type(values)}"
    assert len(values) == 1, f"Expected 1 item for test_id, got {len(values)}"
    
    first = values[0]
    assert isinstance(first, AnnotatedInstance), type(first)
    assert isinstance(first.data, FakeData), type(first.data)
    assert first.data.value == started
    assert first.timestamp == start_timestamp
    assert first.caller_id == test_id

    # After get_new_data(test_id), there should be no new data for test_id
    assert not data_aggregator.has_new_data(test_id), f"Aggregator should not have new data for test_id ({test_id}) after get_new_data"
    runtime_manager.check_for_exception()

    # Stop the runtime.
    runtime.stop()
    runtime_manager.check_for_exception()

    # Sleep to allow the runtime to process the call and respond.
    time.sleep(0.5) # Keep increased sleep

    # Check that the runtime sent its "stopping" message.
    assert data_aggregator.has_new_data(test_id), f"Aggregator should have new data (stop message) for test_id ({test_id})"
    values = data_aggregator.get_new_data(test_id) # Get data specifically for test_id
    assert isinstance(values, list), f"Expected list for get_new_data(test_id) for stop, got {type(values)}"
    assert len(values) == 1, f"Expected 1 stop item for test_id, got {len(values)}"

    first = values[0]
    assert isinstance(first, AnnotatedInstance), type(first)
    assert isinstance(first.data, FakeData), type(first.data)
    assert first.data.value == stopped
    assert first.timestamp == stop_timestamp
    assert first.caller_id == test_id

    assert not data_aggregator.has_new_data(test_id), f"Aggregator should not have new data for test_id ({test_id}) after get_new_data for stop"


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
