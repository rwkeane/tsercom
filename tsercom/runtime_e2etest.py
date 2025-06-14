"""End-to-end tests for Tsercom runtime initialization, data flow, and error handling."""

import asyncio
import datetime
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from threading import Thread

import pytest
import torch

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

start_timestamp = datetime.datetime.now(
    datetime.timezone.utc
) - datetime.timedelta(hours=10)
stop_timestamp = datetime.datetime.now(
    datetime.timezone.utc
) + datetime.timedelta(minutes=20)

test_id = (
    CallerIdentifier.random()
)  # This global test_id seems to be for general use, will create specific ones in tests


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

        self.__responder = await self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 443
        )

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


# --- Torch Tensor Runtime and Initializer ---
class TorchTensorRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
        test_id: CallerIdentifier,
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: EndpointDataProcessor[torch.Tensor] | None = None
        self.sent_tensor: torch.Tensor | None = None
        self.stopped_tensor: torch.Tensor | None = None

    async def start_async(self) -> None:
        await asyncio.sleep(0.01)
        self.__responder = await self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 444
        )
        assert self.__responder is not None

        self.sent_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        await self.__responder.process_data(self.sent_tensor, timestamp)

    async def stop(self, exception) -> None:
        assert self.__responder is not None
        self.stopped_tensor = torch.tensor([[-1.0, -1.0]])
        fixed_stop_ts = datetime.datetime.now(datetime.timezone.utc)
        await self.__responder.process_data(self.stopped_tensor, fixed_stop_ts)


class TorchTensorRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, FakeEvent]
):
    def __init__(self, test_id: CallerIdentifier, service_type="Client"):
        super().__init__(service_type=service_type)
        self._test_id = test_id
        self.expected_sent_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.expected_stopped_tensor = torch.tensor([[-1.0, -1.0]])

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return TorchTensorRuntime(
            thread_watcher, data_handler, grpc_channel_factory, self._test_id
        )


# --- End Torch Tensor Runtime and Initializer ---


# --- String Data, Torch Event Runtime and Initializer ---
class StrDataTorchEventRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[str, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
        test_id: CallerIdentifier,
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: EndpointDataProcessor[str, torch.Tensor] | None = (
            None
        )
        self._listener_task: asyncio.Task | None = None
        self.expected_event_tensor_sum_str: str | None = None

    async def _event_listener_loop(self):
        if self.__responder is None:
            print(
                f"ERROR: {self.__class__.__name__} responder not initialized for event loop."
            )
            return

        print(
            f"DEBUG: {self.__class__.__name__} starting event listener loop for {self.__test_id}"
        )
        try:
            async for event_list in self.__responder:
                for annotated_event in event_list:
                    print(
                        f"DEBUG: {self.__class__.__name__} received event: {type(annotated_event.data)}"
                    )
                    if isinstance(annotated_event.data, torch.Tensor):
                        event_tensor = annotated_event.data
                        self.expected_event_tensor_sum_str = f"processed_tensor_event_{event_tensor.sum().item()}"
                        print(
                            f"DEBUG: {self.__class__.__name__} processing tensor event, sum str: {self.expected_event_tensor_sum_str}"
                        )
                        if self.__responder:
                            await self.__responder.process_data(
                                self.expected_event_tensor_sum_str,
                                datetime.datetime.now(datetime.timezone.utc),
                            )
                            print(
                                f"DEBUG: {self.__class__.__name__} sent data response: {self.expected_event_tensor_sum_str}"
                            )
        except asyncio.CancelledError:
            print(
                f"DEBUG: {self.__class__.__name__} event listener loop cancelled for {self.__test_id}"
            )
        except Exception as e:
            print(
                f"ERROR: {self.__class__.__name__} exception in event_listener_loop: {e}"
            )

    async def start_async(self) -> None:
        await asyncio.sleep(0.01)
        self.__responder = await self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 446
        )
        assert (
            self.__responder is not None
        ), f"{self.__class__.__name__} failed to register caller."
        self._listener_task = asyncio.create_task(self._event_listener_loop())
        print(f"DEBUG: {self.__class__.__name__} started for {self.__test_id}")

    async def stop(self, exception) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self.__responder:
            try:
                await self.__responder.process_data(
                    f"stopped_{self.__class__.__name__}",
                    datetime.datetime.now(datetime.timezone.utc),
                )
            except Exception as e:
                print(
                    f"ERROR: {self.__class__.__name__} error during stop's process_data: {e}"
                )
        print(f"DEBUG: {self.__class__.__name__} stopped for {self.__test_id}")


class StrDataTorchEventRuntimeInitializer(
    RuntimeInitializer[str, torch.Tensor]
):
    def __init__(self, test_id: CallerIdentifier, service_type="Server"):
        super().__init__(service_type=service_type)
        self._test_id = test_id

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[str, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return StrDataTorchEventRuntime(
            thread_watcher, data_handler, grpc_channel_factory, self._test_id
        )


# --- End String Data, Torch Event Runtime and Initializer ---


# --- Torch Data, Torch Event Runtime and Initializer ---
class TorchDataTorchEventRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
        test_id: CallerIdentifier,
        initializer: "TorchDataTorchEventRuntimeInitializer",
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.initializer = initializer
        self.__responder: (
            EndpointDataProcessor[torch.Tensor, torch.Tensor] | None
        ) = None
        self._listener_task: asyncio.Task | None = None

    async def _event_listener_loop(self):
        if self.__responder is None:
            print(
                f"ERROR: {self.__class__.__name__} responder not initialized for event loop."
            )
            return
        print(
            f"DEBUG: {self.__class__.__name__} starting event listener loop for {self.__test_id}"
        )
        try:
            async for event_list in self.__responder:
                for annotated_event in event_list:
                    print(
                        f"DEBUG: {self.__class__.__name__} received event: {type(annotated_event.data)}"
                    )
                    if isinstance(annotated_event.data, torch.Tensor):
                        event_tensor = annotated_event.data
                        response_tensor = (
                            self.initializer.event_response_tensor_base
                            + event_tensor.sum().item()
                        )
                        print(
                            f"DEBUG: {self.__class__.__name__} processing tensor event, response: {response_tensor}"
                        )
                        if self.__responder:
                            await self.__responder.process_data(
                                response_tensor,
                                datetime.datetime.now(datetime.timezone.utc),
                            )
                            print(
                                f"DEBUG: {self.__class__.__name__} sent data response: {response_tensor}"
                            )
        except asyncio.CancelledError:
            print(
                f"DEBUG: {self.__class__.__name__} event listener loop cancelled for {self.__test_id}"
            )
        except Exception as e:
            print(
                f"ERROR: {self.__class__.__name__} exception in event_listener_loop: {e}"
            )

    async def start_async(self) -> None:
        await asyncio.sleep(0.01)
        self.__responder = await self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 447
        )
        assert (
            self.__responder is not None
        ), f"{self.__class__.__name__} failed to register caller."

        print(
            f"DEBUG: {self.__class__.__name__} sending initial data: {self.initializer.initial_data_tensor}"
        )
        await self.__responder.process_data(
            self.initializer.initial_data_tensor,
            datetime.datetime.now(datetime.timezone.utc),
        )

        self._listener_task = asyncio.create_task(self._event_listener_loop())
        print(f"DEBUG: {self.__class__.__name__} started for {self.__test_id}")

    async def stop(self, exception) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self.__responder:
            try:
                print(
                    f"DEBUG: {self.__class__.__name__} sending stopped indicator: {self.initializer.stopped_tensor_indicator}"
                )
                await self.__responder.process_data(
                    self.initializer.stopped_tensor_indicator,
                    datetime.datetime.now(datetime.timezone.utc),
                )
            except Exception as e:
                print(
                    f"ERROR: {self.__class__.__name__} error during stop's process_data: {e}"
                )
        print(f"DEBUG: {self.__class__.__name__} stopped for {self.__test_id}")


class TorchDataTorchEventRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(self, test_id: CallerIdentifier, service_type="Server"):
        super().__init__(service_type=service_type)
        self._test_id = test_id
        self.initial_data_tensor = torch.tensor([[250.0, 350.0]])
        self.event_response_tensor_base = torch.tensor([700.0, 800.0])
        self.stopped_tensor_indicator = torch.tensor([[-999.0]])

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return TorchDataTorchEventRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self._test_id,
            self,
        )


# --- End Torch Data, Torch Event Runtime and Initializer ---


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


# New classes for broadcast test
class BroadcastTestFakeRuntime(Runtime):
    __test__ = False  # Tell pytest this is not a test class

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
        initial_caller_ids: list[CallerIdentifier],
    ):
        super().__init__()
        self._thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self._grpc_channel_factory = grpc_channel_factory
        self.initial_caller_ids = initial_caller_ids
        self._responders: dict[
            CallerIdentifier, EndpointDataProcessor[FakeData, FakeEvent]
        ] = {}
        self._listener_tasks: list[asyncio.Task] = []

    async def start_async(self) -> None:
        for i, cid in enumerate(self.initial_caller_ids):
            port = 50000 + i  # Assign a unique dummy port
            try:
                responder = await self.__data_handler.register_caller(
                    cid, "0.0.0.0", port
                )
                if responder is None:
                    # Handle case where responder might be None, e.g. log or skip
                    print(f"Warning: Responder for {cid} is None.")
                    continue
                self._responders[cid] = responder
                task = asyncio.create_task(
                    self._event_listener_loop(cid, responder)
                )
                self._listener_tasks.append(task)
            except Exception as e:
                # Log or handle registration errors
                print(
                    f"Error registering caller {cid} or starting listener: {e}"
                )

    async def _event_listener_loop(
        self,
        caller_id: CallerIdentifier,
        responder: EndpointDataProcessor[FakeData, FakeEvent],
    ):
        await asyncio.sleep(0.01)  # Yield control at the beginning
        try:
            async for event_list in responder:
                for annotated_event in event_list:
                    if isinstance(annotated_event.data, FakeEvent):
                        # Ensure annotated_event.caller_id is handled if it can be None
                        # For broadcast, original caller_id on event might be None
                        # but we are processing for a specific responder's caller_id here.
                        print(
                            f"Listener for {caller_id}: Received FakeEvent. Processing..."
                        )
                        processed_data = FakeData(
                            f"event_for_{str(caller_id)[:8]}"
                        )
                        await responder.process_data(
                            processed_data,
                            datetime.datetime.now(datetime.timezone.utc),
                        )
                        print(
                            f"LISTENER_DBG: Listener for {caller_id} processed data for FakeEvent"
                        )
                        await asyncio.sleep(0)  # Yield control
        except asyncio.CancelledError:
            print(f"Listener for {caller_id}: Cancelled.")
            # Log cancellation if necessary
            pass
        except Exception as e:
            # Log other exceptions during event listening
            print(f"Error in event listener for {caller_id}: {e}")

    async def stop(self, exception) -> None:
        for task in self._listener_tasks:
            if not task.done():
                task.cancel()
        if self._listener_tasks:
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)

        for cid, responder in self._responders.items():
            try:
                await responder.process_data(
                    FakeData(f"stopped_{str(cid)[:8]}"),
                    datetime.datetime.now(datetime.timezone.utc),
                )
            except Exception as e:
                # Log error during stop processing
                print(f"Error processing stop data for {cid}: {e}")


class BroadcastTestFakeRuntimeInitializer(
    RuntimeInitializer[FakeData, FakeEvent]
):
    __test__ = False  # Tell pytest this is not a test class

    def __init__(
        self, initial_caller_ids: list[CallerIdentifier], service_type="Server"
    ):
        super().__init__(service_type=service_type)
        self.initial_caller_ids = initial_caller_ids

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return BroadcastTestFakeRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self.initial_caller_ids,
        )


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        runtime_manager.check_for_exception()
        initializer = FakeRuntimeInitializer(
            test_id=current_test_id, service_type="Server"
        )
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
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
        assert first.data.value == expected_fresh_value
        assert isinstance(first.timestamp, datetime.datetime)
        assert first.caller_id == current_test_id

        assert not data_aggregator.has_new_data(current_test_id)
        runtime_manager.check_for_exception()

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        time.sleep(0.5)  # Initial sleep

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


def test_out_of_process_torch_tensor_transport(clear_loop_fixture):
    """Validates torch.Tensor transport with an out-of-process runtime."""
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    current_test_id = CallerIdentifier.random()
    initializer = TorchTensorRuntimeInitializer(
        test_id=current_test_id, service_type="Server"
    )

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        assert not runtime_future.done()
        assert not runtime_manager.has_started
        runtime_manager.start_out_of_process()
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
        received_annotated_instance = None
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                all_data = data_aggregator.get_new_data(current_test_id)
                if all_data:
                    received_annotated_instance = all_data[0]
                    data_arrived = True
                    break
            time.sleep(poll_interval)
            waited_time += poll_interval

        runtime_manager.check_for_exception()
        assert (
            data_arrived
        ), f"Aggregator did not receive tensor data for test_id ({current_test_id}) within {max_wait_time}s"

        assert received_annotated_instance is not None
        assert isinstance(received_annotated_instance, AnnotatedInstance)
        assert isinstance(
            received_annotated_instance.data, torch.Tensor
        ), f"Received data is not a torch.Tensor, type: {type(received_annotated_instance.data)}"
        assert torch.equal(
            received_annotated_instance.data, initializer.expected_sent_tensor
        ), f"Received tensor {received_annotated_instance.data} does not match expected {initializer.expected_sent_tensor}"
        assert isinstance(
            received_annotated_instance.timestamp, datetime.datetime
        )
        assert received_annotated_instance.caller_id == current_test_id
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        stopped_data_arrived = False
        waited_time = 0.0
        received_stopped_annotated_instance = None
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                all_stopped_data = data_aggregator.get_new_data(
                    current_test_id
                )
                if all_stopped_data:
                    received_stopped_annotated_instance = all_stopped_data[0]
                    stopped_data_arrived = True
                    break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            stopped_data_arrived
        ), f"Aggregator did not receive 'stopped' tensor data for test_id ({current_test_id}) within {max_wait_time}s"

        assert received_stopped_annotated_instance is not None
        assert isinstance(
            received_stopped_annotated_instance, AnnotatedInstance
        )
        assert isinstance(
            received_stopped_annotated_instance.data, torch.Tensor
        ), f"Received 'stopped' data is not a torch.Tensor, type: {type(received_stopped_annotated_instance.data)}"
        assert torch.equal(
            received_stopped_annotated_instance.data,
            initializer.expected_stopped_tensor,
        ), f"Received stopped tensor {received_stopped_annotated_instance.data} does not match expected {initializer.expected_stopped_tensor}"
        assert isinstance(
            received_stopped_annotated_instance.timestamp, datetime.datetime
        )
        assert received_stopped_annotated_instance.caller_id == current_test_id
        assert not data_aggregator.has_new_data(current_test_id)

    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception:
                pass
        runtime_manager.shutdown()


@pytest.mark.usefixtures("clear_loop_fixture")
def test_out_of_process_torch_event_transport(clear_loop_fixture):
    """Validates torch.Tensor transport for EventTypeT with an out-of-process runtime."""
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    current_test_id = CallerIdentifier.random()

    initializer = StrDataTorchEventRuntimeInitializer(
        test_id=current_test_id, service_type="Server"
    )
    expected_event_tensor = torch.tensor([10.0, 20.0, 30.0])
    expected_sum_val = expected_event_tensor.sum().item()
    expected_response_str = f"processed_tensor_event_{expected_sum_val}"

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        assert not runtime_future.done()
        runtime_manager.start_out_of_process()
        assert runtime_future.done()

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        time.sleep(0.5)

        event_timestamp = datetime.datetime.now(datetime.timezone.utc)
        print(
            f"DEBUG_TEST: Sending event: {expected_event_tensor} to {current_test_id}"
        )
        runtime_handle.on_event(
            expected_event_tensor, current_test_id, timestamp=event_timestamp
        )

        data_arrived = False
        max_wait_time = 10.0
        poll_interval = 0.2
        waited_time = 0.0
        received_annotated_instance = None

        while waited_time < max_wait_time:
            runtime_manager.check_for_exception()
            if data_aggregator.has_new_data(current_test_id):
                all_data = data_aggregator.get_new_data(current_test_id)
                if all_data:
                    for item in all_data:
                        if (
                            isinstance(item.data, str)
                            and item.data == expected_response_str
                        ):
                            received_annotated_instance = item
                            data_arrived = True
                            break
                    if data_arrived:
                        break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            data_arrived
        ), f"Aggregator did not receive expected string data '{expected_response_str}' for test_id ({current_test_id}) within {max_wait_time}s. Last data: {data_aggregator.get_new_data(current_test_id) if data_aggregator.has_new_data(current_test_id) else 'None'}"

        assert received_annotated_instance is not None
        assert isinstance(received_annotated_instance, AnnotatedInstance)
        assert isinstance(received_annotated_instance.data, str)
        assert received_annotated_instance.data == expected_response_str
        assert isinstance(
            received_annotated_instance.timestamp, datetime.datetime
        )
        assert received_annotated_instance.caller_id == current_test_id

    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception as e:
                print(f"Error during cleanup stop: {e}")
        runtime_manager.shutdown()


@pytest.mark.usefixtures("clear_loop_fixture")
def test_out_of_process_torch_data_torch_event_transport(clear_loop_fixture):
    """Validates torch.Tensor for both DataTypeT and EventTypeT with an out-of-process runtime."""
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    current_test_id = CallerIdentifier.random()

    initializer = TorchDataTorchEventRuntimeInitializer(
        test_id=current_test_id, service_type="Server"
    )

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        assert not runtime_future.done()
        runtime_manager.start_out_of_process()
        assert runtime_future.done()

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        # 1. Verify Initial Data Tensor
        initial_data_arrived = False
        max_wait_time = 10.0
        poll_interval = 0.2
        waited_time = 0.0
        received_initial_data_instance = None

        while waited_time < max_wait_time:
            runtime_manager.check_for_exception()
            if data_aggregator.has_new_data(current_test_id):
                all_data = data_aggregator.get_new_data(current_test_id)
                if all_data:
                    for item in all_data:
                        if torch.equal(
                            item.data, initializer.initial_data_tensor
                        ):
                            received_initial_data_instance = item
                            initial_data_arrived = True
                            break
                    if initial_data_arrived:
                        break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            initial_data_arrived
        ), f"Aggregator did not receive initial data tensor for test_id ({current_test_id}) within {max_wait_time}s."
        assert received_initial_data_instance is not None
        assert isinstance(received_initial_data_instance.data, torch.Tensor)
        assert torch.equal(
            received_initial_data_instance.data,
            initializer.initial_data_tensor,
        )

        # 2. Send Tensor Event and Verify Response
        event_tensor_to_send = torch.tensor([25.0, 75.0])
        event_timestamp = datetime.datetime.now(datetime.timezone.utc)
        runtime_handle.on_event(
            event_tensor_to_send, current_test_id, timestamp=event_timestamp
        )

        event_response_arrived = False
        waited_time = 0.0
        received_event_response_instance = None
        expected_event_response = (
            initializer.event_response_tensor_base
            + event_tensor_to_send.sum().item()
        )

        while waited_time < max_wait_time:
            runtime_manager.check_for_exception()
            if data_aggregator.has_new_data(current_test_id):
                all_data = data_aggregator.get_new_data(current_test_id)
                if all_data:
                    for item in all_data:
                        if isinstance(item.data, torch.Tensor) and torch.equal(
                            item.data, expected_event_response
                        ):
                            received_event_response_instance = item
                            event_response_arrived = True
                            break
                    if event_response_arrived:
                        break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            event_response_arrived
        ), f"Aggregator did not receive event response tensor for test_id ({current_test_id}) within {max_wait_time}s."
        assert received_event_response_instance is not None
        assert isinstance(received_event_response_instance.data, torch.Tensor)
        assert torch.equal(
            received_event_response_instance.data, expected_event_response
        )

        # 3. Verify Stopped Tensor
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        stopped_data_arrived = False
        waited_time = 0.0
        received_stopped_data_instance = None
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                all_data = data_aggregator.get_new_data(current_test_id)
                if all_data:
                    for item in all_data:
                        if isinstance(item.data, torch.Tensor) and torch.equal(
                            item.data, initializer.stopped_tensor_indicator
                        ):
                            received_stopped_data_instance = item
                            stopped_data_arrived = True
                            break
                    if stopped_data_arrived:
                        break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            stopped_data_arrived
        ), f"Aggregator did not receive stopped tensor indicator for test_id ({current_test_id}) within {max_wait_time}s."
        assert received_stopped_data_instance is not None
        assert isinstance(received_stopped_data_instance.data, torch.Tensor)
        assert torch.equal(
            received_stopped_data_instance.data,
            initializer.stopped_tensor_indicator,
        )

    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception as e:
                print(f"Error during cleanup stop: {e}")
        runtime_manager.shutdown()


# --- Start: New E2E Stress Test ---
@pytest.mark.usefixtures("clear_loop_fixture")
def test_out_of_process_torch_tensor_stress(clear_loop_fixture):
    """Stress tests torch.Tensor transport for data and events with multiple events."""
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    current_test_id = CallerIdentifier.random()

    initializer = TorchDataTorchEventRuntimeInitializer(
        test_id=current_test_id, service_type="Server"
    )

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(
            initializer
        )

        assert not runtime_future.done()
        runtime_manager.start_out_of_process()
        assert runtime_future.done()

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        # 1. Verify Initial Data Tensor
        initial_data_arrived = False
        max_wait_time = 10.0
        poll_interval = 0.2
        waited_time = 0.0
        received_initial_data_items = []

        while waited_time < max_wait_time:
            runtime_manager.check_for_exception()
            if data_aggregator.has_new_data(current_test_id):
                new_items = data_aggregator.get_new_data(current_test_id)
                received_initial_data_items.extend(new_items)
                # Check if the specific initial tensor is among received items
                for item in new_items:
                    if torch.equal(item.data, initializer.initial_data_tensor):
                        initial_data_arrived = True
                        break
                if initial_data_arrived:
                    break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            initial_data_arrived
        ), f"Aggregator did not receive initial data tensor for test_id ({current_test_id}) within {max_wait_time}s."
        # The initial data might have been cleared from aggregator if other data (event responses) arrived quickly.
        # The key is that it was sent and runtime proceeded.

        # 2. Stress Send Multiple Tensor Events
        num_events_to_send = 50
        sent_event_sums = []
        expected_responses_map = {}

        for i in range(num_events_to_send):
            event_tensor = torch.tensor([float(i + 5), float(i + 5) * 1.5])
            event_sum = event_tensor.sum().item()
            sent_event_sums.append(event_sum)

            expected_response_tensor = (
                initializer.event_response_tensor_base + event_sum
            )
            expected_responses_map[event_sum] = expected_response_tensor

            runtime_handle.on_event(event_tensor, current_test_id)
            time.sleep(0.01)  # Small delay

        # 3. Collect and Verify All Event Responses
        received_event_responses_map = {}
        deadline = time.monotonic() + 25.0  # 25-second timeout

        all_received_data_for_stress_phase = []

        while (
            time.monotonic() < deadline
            and len(received_event_responses_map) < num_events_to_send
        ):
            runtime_manager.check_for_exception()
            if data_aggregator.has_new_data(current_test_id):
                new_data_list = data_aggregator.get_new_data(current_test_id)
                all_received_data_for_stress_phase.extend(new_data_list)
                for item in new_data_list:
                    if torch.equal(item.data, initializer.initial_data_tensor):
                        continue
                    if torch.equal(
                        item.data, initializer.stopped_tensor_indicator
                    ):
                        continue

                    assert isinstance(
                        item.data, torch.Tensor
                    ), f"Received non-tensor data: {item.data}"
                    response_tensor = item.data

                    # Try to match response_tensor to one of the expected_responses_map values
                    # This is more robust if sums are not perfectly unique or if base tensor has non-zero elements.
                    found_match = False
                    for es, expected_resp_t in expected_responses_map.items():
                        if torch.equal(response_tensor, expected_resp_t):
                            if (
                                es not in received_event_responses_map
                            ):  # Store only first match for this sum
                                received_event_responses_map[es] = (
                                    response_tensor
                                )
                                found_match = True
                            break
                    # If not found by direct match, try deriving sum (less robust if base is complex)
                    if not found_match:
                        # This derivation is only valid if event_response_tensor_base is just added (no element-wise mult)
                        # and response_tensor - initializer.event_response_tensor_base results in a scalar or tensor that sums to event_sum
                        try:
                            sum_tensor_candidate = (
                                response_tensor
                                - initializer.event_response_tensor_base
                            )
                            if (
                                sum_tensor_candidate.numel() == 1
                                or (
                                    sum_tensor_candidate.ndim == 1
                                    and sum_tensor_candidate.shape[0] == 1
                                )
                                or (
                                    sum_tensor_candidate.ndim == 2
                                    and sum_tensor_candidate.shape[0] == 1
                                    and sum_tensor_candidate.shape[1] == 1
                                )
                            ):
                                event_sum_from_response = (
                                    sum_tensor_candidate.sum().item()
                                )
                                if (
                                    event_sum_from_response
                                    not in received_event_responses_map
                                    and event_sum_from_response
                                    in expected_responses_map
                                ):
                                    received_event_responses_map[
                                        event_sum_from_response
                                    ] = response_tensor
                        except (
                            Exception
                        ):  # Broad exception if tensor math fails
                            pass  # Could not derive sum, might be an unexpected tensor

            if len(received_event_responses_map) < num_events_to_send:
                time.sleep(0.1)

        assert (
            len(received_event_responses_map) == num_events_to_send
        ), f"Expected {num_events_to_send} responses, got {len(received_event_responses_map)}. \nExpected sums: {sorted(list(expected_responses_map.keys()))} \nReceived sums: {sorted(list(received_event_responses_map.keys()))}"

        for event_sum, expected_tensor in expected_responses_map.items():
            assert (
                event_sum in received_event_responses_map
            ), f"Response for event sum {event_sum} not found."
            assert torch.equal(
                received_event_responses_map[event_sum], expected_tensor
            ), f"Mismatch for event sum {event_sum}. Expected {expected_tensor}, got {received_event_responses_map[event_sum]}"

        # 4. Verify Stopped Tensor
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        stopped_data_arrived = False
        waited_time = 0.0
        received_stopped_data_instance = None
        # Collect any remaining data, looking for the stop indicator
        while waited_time < max_wait_time:  # Reuse max_wait_time
            if data_aggregator.has_new_data(current_test_id):
                all_data = data_aggregator.get_new_data(current_test_id)
                if all_data:
                    for item in all_data:
                        if isinstance(item.data, torch.Tensor) and torch.equal(
                            item.data, initializer.stopped_tensor_indicator
                        ):
                            received_stopped_data_instance = item
                            stopped_data_arrived = True
                            break
                    if stopped_data_arrived:
                        break
            if stopped_data_arrived:  # Break outer loop if found
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            stopped_data_arrived
        ), f"Aggregator did not receive stopped tensor indicator for test_id ({current_test_id}) within {max_wait_time}s."
        assert received_stopped_data_instance is not None
        assert isinstance(received_stopped_data_instance.data, torch.Tensor)
        assert torch.equal(
            received_stopped_data_instance.data,
            initializer.stopped_tensor_indicator,
        )

    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception as e:
                print(f"Error during cleanup stop: {e}")
        runtime_manager.shutdown()


# --- End: New E2E Stress Test ---


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


def test_multiple_runtimes_out_of_process(clear_loop_fixture):
    """
    Verify that RuntimeManager can manage multiple out-of-process runtimes,
    that their data is correctly aggregated and distinguishable, and that they
    operate independently.
    """
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []

    try:
        test_id_1 = CallerIdentifier.random()
        test_id_2 = CallerIdentifier.random()

        # Register two FakeRuntimeInitializers
        runtime_future_1 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(test_id=test_id_1, service_type="Server")
        )
        runtime_future_2 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(test_id=test_id_2, service_type="Server")
        )

        assert not runtime_future_1.done()
        assert not runtime_future_2.done()
        assert not runtime_manager.has_started

        runtime_manager.start_out_of_process()
        assert runtime_manager.has_started
        assert runtime_future_1.done()
        assert runtime_future_2.done()

        runtime_manager.check_for_exception()

        runtime_handle_1 = runtime_future_1.result(timeout=1)
        runtime_handle_2 = runtime_future_2.result(timeout=1)
        runtime_handles_for_cleanup.extend(
            [runtime_handle_1, runtime_handle_2]
        )

        data_aggregator_1 = runtime_handle_1.data_aggregator
        data_aggregator_2 = runtime_handle_2.data_aggregator

        # Start both runtimes
        runtime_handle_1.start()
        runtime_handle_2.start()

        # --- Verify Data for Runtime 1 ---
        data_arrived_1 = False
        max_wait_time = 5.0
        poll_interval = 0.1
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator_1.has_new_data(test_id_1):
                data_arrived_1 = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            data_arrived_1
        ), f"Aggregator 1 did not receive data for test_id_1 ({test_id_1}) within {max_wait_time}s"
        assert data_aggregator_1.has_new_data(test_id_1)
        assert not data_aggregator_1.has_new_data(
            test_id_2
        ), "Aggregator 1 should not have data for test_id_2 yet"

        values_1 = data_aggregator_1.get_new_data(test_id_1)
        assert isinstance(values_1, list) and len(values_1) == 1
        first_1 = values_1[0]
        assert isinstance(first_1, AnnotatedInstance) and isinstance(
            first_1.data, FakeData
        )
        assert first_1.data.value == "FRESH_SIMPLE_DATA_V2"
        assert isinstance(first_1.timestamp, datetime.datetime)
        assert first_1.caller_id == test_id_1
        assert not data_aggregator_1.has_new_data(test_id_1)

        # --- Verify Data for Runtime 2 ---
        data_arrived_2 = False
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator_2.has_new_data(
                test_id_2
            ):  # Check aggregator 2 for test_id_2
                data_arrived_2 = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            data_arrived_2
        ), f"Aggregator 2 did not receive data for test_id_2 ({test_id_2}) within {max_wait_time}s"
        assert data_aggregator_2.has_new_data(test_id_2)
        if data_aggregator_1 is not data_aggregator_2:
            assert not data_aggregator_2.has_new_data(
                test_id_1
            ), "Aggregator 2 should not have data for test_id_1"

        values_2 = data_aggregator_2.get_new_data(test_id_2)
        assert isinstance(values_2, list) and len(values_2) == 1
        first_2 = values_2[0]
        assert isinstance(first_2, AnnotatedInstance) and isinstance(
            first_2.data, FakeData
        )
        assert first_2.data.value == "FRESH_SIMPLE_DATA_V2"
        assert isinstance(first_2.timestamp, datetime.datetime)
        assert first_2.caller_id == test_id_2
        assert not data_aggregator_2.has_new_data(test_id_2)

        runtime_manager.check_for_exception()

        # --- Stop Runtime 1 and Verify "stopped" Data ---
        runtime_handle_1.stop()
        stopped_data_arrived_1 = False
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator_1.has_new_data(test_id_1):
                stopped_data_arrived_1 = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            stopped_data_arrived_1
        ), f"Aggregator 1 did not receive 'stopped' data for test_id_1 ({test_id_1}) within {max_wait_time}s"
        values_stop_1 = data_aggregator_1.get_new_data(test_id_1)
        assert isinstance(values_stop_1, list) and len(values_stop_1) == 1
        first_stop_1 = values_stop_1[0]
        assert (
            isinstance(first_stop_1.data, FakeData)
            and first_stop_1.data.value == stopped
        )
        assert first_stop_1.caller_id == test_id_1
        assert not data_aggregator_1.has_new_data(test_id_1)

        # --- Stop Runtime 2 and Verify "stopped" Data ---
        runtime_handle_2.stop()
        stopped_data_arrived_2 = False
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator_2.has_new_data(test_id_2):
                stopped_data_arrived_2 = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            stopped_data_arrived_2
        ), f"Aggregator 2 did not receive 'stopped' data for test_id_2 ({test_id_2}) within {max_wait_time}s"
        values_stop_2 = data_aggregator_2.get_new_data(test_id_2)
        assert isinstance(values_stop_2, list) and len(values_stop_2) == 1
        first_stop_2 = values_stop_2[0]
        assert (
            isinstance(first_stop_2.data, FakeData)
            and first_stop_2.data.value == stopped
        )
        assert first_stop_2.caller_id == test_id_2
        assert not data_aggregator_2.has_new_data(test_id_2)

        runtime_manager.check_for_exception()

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
                test_id=current_test_id, service_type="Client"
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

        data_arrived = False
        max_wait_time = 5.0
        poll_interval = 0.1
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                data_arrived = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            data_arrived
        ), f"Aggregator did not receive 'FRESH' data for test_id ({current_test_id}) within {max_wait_time}s"
        assert data_aggregator.has_new_data(current_test_id)

        values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values, list) and len(values) == 1
        first = values[0]
        assert isinstance(first, AnnotatedInstance) and isinstance(
            first.data, FakeData
        )
        assert first.data.value == "FRESH_SIMPLE_DATA_V2"
        assert isinstance(first.timestamp, datetime.datetime)
        assert first.caller_id == current_test_id
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        stopped_data_arrived = False
        waited_time = 0.0
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id):
                stopped_data_arrived = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        assert (
            stopped_data_arrived
        ), f"Aggregator did not receive 'stopped' data for test_id ({current_test_id}) within {max_wait_time}s"

        values_stop = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values_stop, list) and len(values_stop) == 1
        first_stop = values_stop[0]
        assert isinstance(first_stop, AnnotatedInstance) and isinstance(
            first_stop.data, FakeData
        )
        assert first_stop.data.value == stopped
        assert first_stop.timestamp == stop_timestamp
        assert first_stop.caller_id == current_test_id
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
    called_check_for_exception = False

    try:
        initializer = FaultyCreateRuntimeInitializer(
            error_message=error_msg,
            error_type=ValueError,
            service_type="Client",
        )
        runtime_manager.register_runtime_initializer(initializer)
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        time.sleep(0.3)

        with pytest.raises(ValueError, match=error_msg):
            called_check_for_exception = True
            runtime_manager.check_for_exception()

        assert (
            called_check_for_exception
        ), "check_for_exception was expected to be called and raise."

    except ValueError as e:
        if str(e) == error_msg and not called_check_for_exception:
            print(
                f"Caught expected error '{error_msg}' directly from start_in_process, not from check_for_exception as test was re-specified."
            )
            pass
        else:
            raise

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

        if test_thread.is_alive():
            pytest.fail("run_until_exception timed out / deadlocked.")

        result_from_thread = thread_result_queue.result(timeout=0)

        assert isinstance(
            result_from_thread, error_type
        ), f"Expected exception {error_type}, but got {type(result_from_thread)}"
        assert (
            str(result_from_thread) == error_msg
        ), f"Expected error message '{error_msg}', but got '{str(result_from_thread)}'"

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
            pass
        elif test_thread:
            test_thread.join(timeout=1)


def test_event_broadcast_e2e(clear_loop_fixture):
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
                    if not task.done():
                        task.cancel()
            if all_tasks:
                loop.run_until_complete(
                    asyncio.gather(*all_tasks, return_exceptions=True)
                )
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
            loop.close()

    event_thread = Thread(
        target=_thread_loop_runner, args=(loop_future,), daemon=True
    )
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []

    try:
        caller_id1 = CallerIdentifier.random()
        caller_id2 = CallerIdentifier.random()
        all_caller_ids = [caller_id1, caller_id2]
        pending_ids_to_receive_event = {str(cid) for cid in all_caller_ids}

        initializer = BroadcastTestFakeRuntimeInitializer(
            all_caller_ids, service_type="Server"
        )
        handle_future = runtime_manager.register_runtime_initializer(
            initializer
        )
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        runtime_handle = handle_future.result(timeout=10)
        runtime_handles_for_cleanup.append(runtime_handle)

        data_aggregator = runtime_handle.data_aggregator
        runtime_handle.start()
        time.sleep(1.0)

        broadcast_event = FakeEvent()
        runtime_handle.on_event(broadcast_event, caller_id=None)
        time.sleep(1.5)

        max_wait_time = 10.0
        poll_interval = 0.2
        waited_time = 0.0

        while waited_time < max_wait_time and pending_ids_to_receive_event:
            runtime_manager.check_for_exception()
            for cid_obj in all_caller_ids:
                cid_key = str(cid_obj)
                if cid_key not in pending_ids_to_receive_event:
                    continue

                if data_aggregator.has_new_data(cid_obj):
                    received_datas = data_aggregator.get_new_data(cid_obj)
                    assert len(received_datas) >= 1
                    event_data_found = False
                    for item in received_datas:
                        assert isinstance(item, AnnotatedInstance)
                        assert isinstance(item.data, FakeData)
                        if item.data.value == f"event_for_{str(cid_obj)[:8]}":
                            event_data_found = True
                            break
                    if event_data_found:
                        pending_ids_to_receive_event.remove(cid_key)
            if not pending_ids_to_receive_event:
                break
            time.sleep(poll_interval)
            waited_time += poll_interval
        assert (
            not pending_ids_to_receive_event
        ), f"Not all callers received the broadcast event. Missing: {pending_ids_to_receive_event}"
    finally:
        for handle_item in runtime_handles_for_cleanup:
            try:
                if handle_item:
                    handle_item.stop()
            except Exception as e:
                print(f"Error during handle.stop() for {handle_item}: {e}")
                pass
        runtime_manager.shutdown()
        if event_thread.is_alive():
            event_thread.join(timeout=2)


# Ensure a newline at the end of the file
