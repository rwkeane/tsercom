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
        # Wait to allow initialization to complete.
        await asyncio.sleep(0.01)

        # Register a fake endpoint to allow returning data.
        self.__responder = self.__data_handler.register_caller(
            test_id, "0.0.0.0", 443
        )

        # Return the data.
        data = FakeData(started)
        await self.__responder.process_data(data, start_timestamp)

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
    runtime_manager = RuntimeManager()
    runtime_future = runtime_manager.register_runtime_initializer(
        FakeRuntimeInitializer(service_type="Client")
    )
    assert not runtime_future.done()
    init_call(runtime_manager)

    assert runtime_future.done()
    runtime = runtime_future.result()

    data_aggregator = runtime.data_aggregator
    assert not data_aggregator.has_new_data()
    runtime.start()

    # Sleep to allow the runtime to process the call and respond.
    time.sleep(0.1)

    assert data_aggregator.has_new_data()
    values = data_aggregator.get_new_data()
    assert len(values.keys()) == 1
    for key, val in values.items():
        assert isinstance(key, CallerIdentifier)
        assert isinstance(val, AnnotatedInstance)
        assert isinstance(val.data, FakeData)
        assert val.data.value == started
        assert val.timestamp == start_timestamp
        assert val.caller_id == test_id

    assert not data_aggregator.has_new_data()
    runtime.stop()

    # Sleep to allow the runtime to process the call and respond.
    time.sleep(0.1)

    assert data_aggregator.has_new_data()
    values = data_aggregator.get_new_data()
    assert len(values.keys()) == 1
    for key, val in values.items():
        assert isinstance(key, CallerIdentifier)
        assert isinstance(val, AnnotatedInstance)
        assert isinstance(val.data, FakeData)
        assert val.data.value == stopped
        assert val.timestamp == stop_timestamp
        assert val.caller_id == test_id


def test_out_of_process_init():
    __check_initialization(RuntimeManager.start_out_of_process)


def test_in_process_init():
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
