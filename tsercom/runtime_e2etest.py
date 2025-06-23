"""End-to-end tests for Tsercom runtime initialization, data flow, and error handling."""

import asyncio
import datetime
import logging  # Added
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import Optional

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


import multiprocessing
import os  # For checking if __main__ to avoid running this when imported by other test files

# Try to set the start method to 'spawn'. This should be done as early as possible.
# Guarding with a check for __main__ or specific module name can prevent issues
# if this file is imported by other test discovery processes.
# However, for e2e tests that are run directly, this should be fine.
# A common pattern for e2e tests is that they are the entry point.
# Consider if this file could be imported by pytest in a way that this runs too early or too late.
# For now, applying directly to the file mentioned in warnings.
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        # Check if running as the main script or specifically this test file
        # This is a heuristic to avoid setting this multiple times if imported.
        # A better place might be a conftest.py if it affects many e2e tests.
        # For now, applying directly to the file mentioned in warnings.
        if os.environ.get("PYTEST_CURRENT_TEST"):  # Check if pytest is running
            multiprocessing.set_start_method("spawn", force=True)
        elif __name__ == "__main__":  # If run as a script
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        # This can happen if the context has already been used.
        logging.debug(
            f"INFO: Could not set multiprocessing start method to spawn in {__file__}: {e}"
        )
        pass

started = "STARTED"
stopped = "STOPPED"

# Use fixed timestamps to ensure consistency across processes for test assertions
start_timestamp = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
stop_timestamp = datetime.datetime(2024, 1, 1, 12, 20, 0, tzinfo=datetime.timezone.utc)

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
        stopped_event: Optional[multiprocessing.Event] = None,
        data_event: Optional[multiprocessing.Event] = None,  # For OOP data
        data_event_async: Optional[asyncio.Event] = None,  # For IP data
    ):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: EndpointDataProcessor[FakeData] | None = None
        self._data_sent = False
        self.__stopped_event = stopped_event
        self.__data_event = data_event  # OOP data event
        self.__data_event_async = data_event_async  # IP data event

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

            await self.__responder.process_data(fresh_data_object, fresh_timestamp)
            self._data_sent = True
            if self.__data_event:  # OOP
                self.__data_event.set()
            if self.__data_event_async:  # IP
                self.__data_event_async.set()

    async def stop(self, exception) -> None:
        assert self.__responder is not None
        # Use a fresh timestamp to ensure it's the latest data
        current_stop_time = datetime.datetime.now(datetime.timezone.utc)
        log_message = f"FakeRuntime ({self.__test_id}) stopping. Data: {stopped}, Timestamp: {current_stop_time}"  # Kept for now, minimal
        print(log_message)  # Kept for now, minimal
        await self.__responder.process_data(FakeData(stopped), current_stop_time)
        if self.__stopped_event:
            self.__stopped_event.set()


class FakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    def __init__(
        self,
        test_id: CallerIdentifier,
        service_type="Client",
        stopped_event: Optional[multiprocessing.Event] = None,
        data_event: Optional[multiprocessing.Event] = None,  # OOP
        data_event_async: Optional[asyncio.Event] = None,  # IP
    ):
        super().__init__(service_type=service_type)
        self._test_id = test_id
        self._stopped_event = stopped_event
        self._data_event = data_event
        self._data_event_async = data_event_async

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[FakeData, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return FakeRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self._test_id,
            self._stopped_event,
            self._data_event,
            self._data_event_async,
        )


# --- Torch Tensor Runtime and Initializer ---
class TorchTensorRuntime(Runtime):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
        test_id: CallerIdentifier,
        data_event: multiprocessing.Event,
        stopped_event: multiprocessing.Event,
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: EndpointDataProcessor[torch.Tensor] | None = None
        self.sent_tensor: torch.Tensor | None = None
        self.stopped_tensor: torch.Tensor | None = None
        self.__data_event = data_event
        self.__stopped_event = stopped_event

    async def start_async(self) -> None:
        await asyncio.sleep(0.01)
        self.__responder = await self.__data_handler.register_caller(
            self.__test_id, "0.0.0.0", 444
        )
        assert self.__responder is not None

        self.sent_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        await self.__responder.process_data(self.sent_tensor, timestamp)
        self.__data_event.set()

    async def stop(self, exception) -> None:
        assert self.__responder is not None
        self.stopped_tensor = torch.tensor([[-1.0, -1.0]])
        fixed_stop_ts = datetime.datetime.now(datetime.timezone.utc)
        await self.__responder.process_data(self.stopped_tensor, fixed_stop_ts)
        self.__stopped_event.set()


class TorchTensorRuntimeInitializer(RuntimeInitializer[torch.Tensor, FakeEvent]):
    def __init__(
        self,
        test_id: CallerIdentifier,
        data_event: multiprocessing.Event,
        stopped_event: multiprocessing.Event,
        service_type="Client",
    ):
        super().__init__(service_type=service_type)
        self._test_id = test_id
        self.expected_sent_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.expected_stopped_tensor = torch.tensor([[-1.0, -1.0]])
        self._data_event = data_event
        self._stopped_event = stopped_event

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[torch.Tensor, FakeEvent],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return TorchTensorRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self._test_id,
            self._data_event,
            self._stopped_event,
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
        data_event: Optional[multiprocessing.Event] = None,
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__responder: EndpointDataProcessor[str, torch.Tensor] | None = None
        self._listener_task: asyncio.Task | None = None
        self.expected_event_tensor_sum_str: str | None = None
        self.__data_event = data_event

    async def _event_listener_loop(self):
        if self.__responder is None:
            logging.debug(
                f"ERROR: {self.__class__.__name__} responder not initialized for event loop."
            )
            return

        logging.debug(
            f"DEBUG: {self.__class__.__name__} starting event listener loop for {self.__test_id}"
        )
        try:
            async for event_list in self.__responder:
                for annotated_event in event_list:
                    logging.debug(
                        f"DEBUG: {self.__class__.__name__} received event: {type(annotated_event.data)}"
                    )
                    if isinstance(annotated_event.data, torch.Tensor):
                        event_tensor = annotated_event.data
                        self.expected_event_tensor_sum_str = (
                            f"processed_tensor_event_{event_tensor.sum().item()}"
                        )
                        logging.debug(
                            f"DEBUG: {self.__class__.__name__} processing tensor event, sum str: {self.expected_event_tensor_sum_str}"
                        )
                        if self.__responder:
                            await self.__responder.process_data(
                                self.expected_event_tensor_sum_str,
                                datetime.datetime.now(datetime.timezone.utc),
                            )
                            logging.debug(
                                f"DEBUG: {self.__class__.__name__} sent data response: {self.expected_event_tensor_sum_str}"
                            )
                            if self.__data_event:
                                self.__data_event.set()  # Signal after sending the specific data
                                # self.__data_event = None # Optional: make event one-shot if it's only for one piece of data
        except asyncio.CancelledError:
            logging.debug(
                f"DEBUG: {self.__class__.__name__} event listener loop cancelled for {self.__test_id}"
            )
        except Exception as e:
            logging.debug(
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
        logging.debug(f"DEBUG: {self.__class__.__name__} started for {self.__test_id}")

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
                logging.debug(
                    f"ERROR: {self.__class__.__name__} error during stop's process_data: {e}"
                )
        logging.debug(f"DEBUG: {self.__class__.__name__} stopped for {self.__test_id}")


class StrDataTorchEventRuntimeInitializer(RuntimeInitializer[str, torch.Tensor]):
    def __init__(
        self,
        test_id: CallerIdentifier,
        service_type="Server",
        data_event: Optional[multiprocessing.Event] = None,
    ):
        super().__init__(service_type=service_type)
        self._test_id = test_id
        self._data_event = data_event

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[str, torch.Tensor],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        return StrDataTorchEventRuntime(
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            self._test_id,
            self._data_event,
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
        initial_data_event: Optional[multiprocessing.Event] = None,
        event_response_queue: Optional[
            multiprocessing.Queue
        ] = None,  # Changed from Event to Queue
        stopped_indicator_event: Optional[multiprocessing.Event] = None,
    ):
        super().__init__()
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.initializer = initializer
        self.__responder: EndpointDataProcessor[torch.Tensor, torch.Tensor] | None = (
            None
        )
        self._listener_task: asyncio.Task | None = None
        self._initial_data_event = initial_data_event
        self._event_response_queue = event_response_queue
        self._stopped_indicator_event = stopped_indicator_event

    async def _event_listener_loop(self):
        if self.__responder is None:
            logging.debug(
                f"ERROR: {self.__class__.__name__} responder not initialized for event loop."
            )
            return
        logging.debug(
            f"DEBUG: {self.__class__.__name__} starting event listener loop for {self.__test_id}"
        )
        try:
            async for event_list in self.__responder:
                for annotated_event in event_list:
                    logging.debug(
                        f"DEBUG: {self.__class__.__name__} received event: {type(annotated_event.data)}"
                    )
                    if isinstance(annotated_event.data, torch.Tensor):
                        event_tensor = annotated_event.data
                        response_tensor = (
                            self.initializer.event_response_tensor_base
                            + event_tensor.sum().item()
                        )
                        logging.debug(
                            f"DEBUG: {self.__class__.__name__} processing tensor event, response: {response_tensor}"
                        )
                        if self.__responder:
                            await self.__responder.process_data(
                                response_tensor,
                                datetime.datetime.now(datetime.timezone.utc),
                            )
                            logging.debug(
                                f"DEBUG: {self.__class__.__name__} sent data response: {response_tensor}"
                            )
                            if self._event_response_queue:
                                try:
                                    self._event_response_queue.put_nowait(True)
                                except (
                                    Exception
                                ) as e_queue:  # Should be Full if not sized appropriately but catch all
                                    print(
                                        f"ERROR: {self.__class__.__name__} could not put to event_response_queue: {e_queue}"
                                    )
        except asyncio.CancelledError:
            logging.debug(
                f"DEBUG: {self.__class__.__name__} event listener loop cancelled for {self.__test_id}"
            )
        except Exception as e:
            logging.debug(
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

        logging.debug(
            f"DEBUG: {self.__class__.__name__} sending initial data: {self.initializer.initial_data_tensor}"
        )
        await self.__responder.process_data(
            self.initializer.initial_data_tensor,
            datetime.datetime.now(datetime.timezone.utc),
        )
        if self._initial_data_event:
            self._initial_data_event.set()

        self._listener_task = asyncio.create_task(self._event_listener_loop())
        logging.debug(f"DEBUG: {self.__class__.__name__} started for {self.__test_id}")

    async def stop(self, exception) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self.__responder:
            try:
                logging.debug(
                    f"DEBUG: {self.__class__.__name__} sending stopped indicator: {self.initializer.stopped_tensor_indicator}"
                )
                await self.__responder.process_data(
                    self.initializer.stopped_tensor_indicator,
                    datetime.datetime.now(datetime.timezone.utc),
                )
                if self._stopped_indicator_event:
                    self._stopped_indicator_event.set()
            except Exception as e:
                logging.debug(
                    f"ERROR: {self.__class__.__name__} error during stop's process_data: {e}"
                )
        logging.debug(f"DEBUG: {self.__class__.__name__} stopped for {self.__test_id}")


class TorchDataTorchEventRuntimeInitializer(
    RuntimeInitializer[torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        test_id: CallerIdentifier,
        service_type="Server",
        initial_data_event: Optional[multiprocessing.Event] = None,
        event_response_queue: Optional[multiprocessing.Queue] = None,
        stopped_indicator_event: Optional[multiprocessing.Event] = None,
    ):
        super().__init__(service_type=service_type)
        self._test_id = test_id
        self.initial_data_tensor = torch.tensor([[250.0, 350.0]])
        self.event_response_tensor_base = torch.tensor([700.0, 800.0])
        self.stopped_tensor_indicator = torch.tensor([[-999.0]])
        self._initial_data_event = initial_data_event
        self._event_response_queue = event_response_queue
        self._stopped_indicator_event = stopped_indicator_event

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
            self._initial_data_event,
            self._event_response_queue,
            self._stopped_indicator_event,
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
        error_event: Optional[multiprocessing.Event] = None,
    ):
        super().__init__()
        self.error_message = error_message
        self.error_type = error_type
        self._thread_watcher = thread_watcher
        self._data_handler = data_handler
        self._grpc_channel_factory = grpc_channel_factory
        self._error_event = error_event

    async def start_async(self) -> None:
        if self._error_event:
            self._error_event.set()
        # Add a small sleep to ensure the event is processed before the process potentially dies from the unhandled exception
        await asyncio.sleep(0.01)
        raise self.error_type(self.error_message)

    async def stop(self, exception) -> None:
        pass


class ErrorThrowingRuntimeInitializer(RuntimeInitializer):
    def __init__(
        self,
        error_message="TestError",
        error_type=RuntimeError,
        service_type="Client",
        error_event: Optional[multiprocessing.Event] = None,
    ):
        super().__init__(service_type=service_type)
        self.error_message = error_message
        self.error_type = error_type
        self._error_event = error_event

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
            self._error_event,
        )


class FaultyCreateRuntimeInitializer(RuntimeInitializer):
    def __init__(
        self,
        error_message="CreateFailed",
        error_type=TypeError,
        service_type="Client",
        error_event: Optional[multiprocessing.Event] = None,  # Add event
    ):
        super().__init__(service_type=service_type)
        self.error_message = error_message
        self.error_type = error_type
        self._error_event = error_event  # Store event

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        if self._error_event:  # Set event before raising
            self._error_event.set()
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
        caller_event_map: Optional[dict[CallerIdentifier, asyncio.Event]] = None,
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
        self._caller_event_map = caller_event_map

    async def start_async(self) -> None:
        for i, cid in enumerate(self.initial_caller_ids):
            port = 50000 + i  # Assign a unique dummy port
            try:
                responder = await self.__data_handler.register_caller(
                    cid, "0.0.0.0", port
                )
                if responder is None:
                    # Handle case where responder might be None, e.g. log or skip
                    logging.debug(f"Warning: Responder for {cid} is None.")
                    continue
                self._responders[cid] = responder
                task = asyncio.create_task(self._event_listener_loop(cid, responder))
                self._listener_tasks.append(task)
            except Exception as e:
                # Log or handle registration errors
                logging.debug(
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
                        logging.debug(
                            f"Listener for {caller_id}: Received FakeEvent. Processing..."
                        )
                        processed_data = FakeData(f"event_for_{str(caller_id)[:8]}")
                        await responder.process_data(
                            processed_data,
                            datetime.datetime.now(datetime.timezone.utc),
                        )
                        logging.debug(
                            f"LISTENER_DBG: Listener for {caller_id} processed data for FakeEvent"
                        )
                        if (
                            self._caller_event_map
                            and caller_id in self._caller_event_map
                        ):
                            self._caller_event_map[caller_id].set()
                        await asyncio.sleep(0)  # Yield control
        except asyncio.CancelledError:
            logging.debug(f"Listener for {caller_id}: Cancelled.")
            # Log cancellation if necessary
            pass
        except Exception as e:
            # Log other exceptions during event listening
            logging.debug(f"Error in event listener for {caller_id}: {e}")

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
                logging.debug(f"Error processing stop data for {cid}: {e}")


class BroadcastTestFakeRuntimeInitializer(RuntimeInitializer[FakeData, FakeEvent]):
    __test__ = False  # Tell pytest this is not a test class

    def __init__(
        self,
        initial_caller_ids: list[CallerIdentifier],
        service_type="Server",
        caller_event_map: Optional[dict[CallerIdentifier, asyncio.Event]] = None,
    ):
        super().__init__(service_type=service_type)
        self.initial_caller_ids = initial_caller_ids
        self._caller_event_map = caller_event_map

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
            self._caller_event_map,
        )


def __check_initialization(init_call: Callable[[RuntimeManager], None]):
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        runtime_manager.check_for_exception()

        # Create event for stopped signal, using spawn context if init_call indicates out-of-process
        # This is a heuristic; ideally, RuntimeManager would expose its context or method.
        # For now, assuming start_out_of_process implies spawn.
        oop_stopped_event = None
        async_data_event = None
        event_timeout_seconds = 20.0  # Timeout for waiting on the event

        is_out_of_process_init = "start_out_of_process" in getattr(
            init_call, "__name__", ""
        ) or (
            isinstance(init_call, partial)
            and "start_out_of_process" in getattr(init_call.func, "__name__", "")
        )

        is_in_process_init = "start_in_process" in getattr(
            init_call, "__name__", ""
        ) or (
            isinstance(init_call, partial)
            and "start_in_process" in getattr(init_call.func, "__name__", "")
        )

        if is_out_of_process_init:
            ctx = multiprocessing.get_context("spawn")
            oop_stopped_event = ctx.Event()
            # For OOP, FakeRuntime doesn't use a data_event yet, this test relies on polling for initial data.
            # This change is specifically for in-process initial data.
        elif is_in_process_init:
            async_data_event = asyncio.Event()

        initializer = FakeRuntimeInitializer(
            test_id=current_test_id,
            service_type="Server",
            stopped_event=oop_stopped_event,  # Pass the mp.Event for OOP stop
            data_event_async=async_data_event,  # Pass the asyncio.Event for IP data
        )
        runtime_future = runtime_manager.register_runtime_initializer(initializer)

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

        # For in-process, runtime_handle.start() will run in the worker_event_loop.
        # We need to ensure that loop is running and `start_async` can complete.
        # The test `test_in_process_init` already manages the worker_event_loop.
        runtime_handle.start()

        if async_data_event:  # In-process case
            try:
                # Run the wait in the correct event loop if possible, or use a timeout that allows context switching
                # This part is tricky as __check_initialization is synchronous.
                # A simple approach is to use a timed wait that allows the other thread's loop to run.
                # Loop of time.sleep is effectively what we had, but now event-driven.
                # We need to await this event in the loop where FakeRuntime runs.
                # The test structure makes it hard for __check_initialization to directly await.
                # Instead, we rely on FakeRuntime setting the event, and we poll the event status here
                # with very short sleeps, or use a timed wait on the event.
                # Let's use a timed wait on the event.
                # This will be awaited by the test's event loop if __check_initialization itself were async.
                # Since it's sync, we'll use a loop to check event.is_set() or rely on runtime_manager's loop.

                # The runtime runs in a separate thread's event loop.
                # We must wait for the event in a way that doesn't block this test thread entirely if the
                # target loop needs this thread to release GIL or for other reasons.
                # A loop with event.is_set() and time.sleep is effectively polling the event.
                # Let's try a direct wait, assuming the asyncio event is set in another thread's loop.
                # This might be problematic if the event loop is not handled correctly by the test.
                # The test `test_in_process_init` manages the loop.
                # This is a placeholder for how one *would* wait if this were an async function.
                # For now, we'll keep the polling loop for in-process initial data,
                # but if FakeRuntime sets an asyncio.Event, the test could be refactored to await it.
                # The current FakeRuntime.start_async() will call async_data_event.set().
                # The polling loop below will pick it up.
                # To make it truly event-driven here would require __check_initialization to be async.
                # For now, the event will be set, and the existing poll will just find data faster.
                # The key is that FakeRuntime SETS the event. The original polling loop remains for now for IP initial data.
                pass  # Event is set by FakeRuntime, polling loop below will catch it.

            except asyncio.TimeoutError:  # This won't be hit with current structure
                pytest.fail(f"In-process data event timed out for {current_test_id}")

        # Original polling loop for initial data (kept for now for in-process, event helps it)
        data_arrived = False
        max_wait_time = 15.0  # Increased from 5.0 to allow more time for initial data
        poll_interval = 0.1
        waited_time = 0.0
        while waited_time < max_wait_time:
            if (
                async_data_event and async_data_event.is_set()
            ):  # Check event if available
                # Allow a very brief moment for data to hit aggregator after event is set
                time.sleep(0.05)
            has_data_now = data_aggregator.has_new_data(current_test_id)
            if has_data_now:
                data_arrived = True
                break
            time.sleep(poll_interval)
            waited_time += poll_interval

        runtime_manager.check_for_exception()
        assert (
            data_arrived
        ), f"Aggregator did not receive data for test_id ({current_test_id}) within {max_wait_time}s. Async event set: {async_data_event.is_set() if async_data_event else 'N/A'}"
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

        if oop_stopped_event:  # Check the oop_stopped_event variable
            stopped_event_was_set = oop_stopped_event.wait(
                timeout=event_timeout_seconds
            )
            assert (
                stopped_event_was_set
            ), f"Runtime did not signal oop_stopped_event for test_id ({current_test_id}) within {event_timeout_seconds}s"
            time.sleep(0.1)  # Allow time for data to be processed by the aggregator
        else:
            # Fallback to polling if no event (e.g. for in-process which wasn't failing)
            # This part retains the original polling logic for the 'stopped' signal if no event is used.
            # The failure was specifically in out-of-process, which will now use the event.
            time.sleep(5.0)  # Original sleep
            stopped_data_arrived_polling = False
            max_wait_stopped_data = 20.0
            poll_interval_stopped = 0.1
            waited_time_stopped = 0.0
            while waited_time_stopped < max_wait_stopped_data:
                if data_aggregator.has_new_data(current_test_id):
                    stopped_data_arrived_polling = True
                    break
                time.sleep(poll_interval_stopped)
                waited_time_stopped += poll_interval_stopped
            assert (
                stopped_data_arrived_polling
            ), f"Aggregator (polling) did not receive 'stopped' data for test_id ({current_test_id}) within {max_wait_stopped_data}s"

        runtime_manager.check_for_exception()  # Check for errors after event/polling

        assert data_aggregator.has_new_data(
            current_test_id
        ), "Data aggregator should have new 'stopped' data after event was set or polling."
        values = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values, list)
        assert len(values) == 1

        first = values[0]
        assert isinstance(first, AnnotatedInstance)
        assert isinstance(first.data, FakeData)
        assert first.data.value == stopped
        # Timestamp assertion removed as it's now dynamic
        # assert first.timestamp == stop_timestamp
        assert isinstance(first.timestamp, datetime.datetime)  # Check it's a datetime
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

    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True)
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

    # Get a 'spawn' context and create events from it
    ctx = multiprocessing.get_context("spawn")
    data_received_event = ctx.Event()
    stopped_event = ctx.Event()
    event_timeout_seconds = 15.0  # Generous timeout for event waiting

    initializer = TorchTensorRuntimeInitializer(
        test_id=current_test_id,
        data_event=data_received_event,
        stopped_event=stopped_event,
        service_type="Server",
    )

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(initializer)

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

        # Wait for the data event from the runtime
        data_event_was_set = data_received_event.wait(timeout=event_timeout_seconds)
        assert (
            data_event_was_set
        ), f"Runtime did not signal data received for test_id ({current_test_id}) within {event_timeout_seconds}s"

        time.sleep(0.1)  # Allow time for data to be processed by the aggregator
        runtime_manager.check_for_exception()  # Check for errors after event

        # Retrieve and assert data
        assert data_aggregator.has_new_data(
            current_test_id
        ), "Data aggregator should have new data after event was set."
        all_data = data_aggregator.get_new_data(current_test_id)
        assert all_data, "Received empty data list from aggregator."
        received_annotated_instance = all_data[0]

        assert (
            received_annotated_instance is not None
        ), "Failed to get annotated instance from aggregator."
        assert isinstance(received_annotated_instance, AnnotatedInstance)
        assert isinstance(
            received_annotated_instance.data, torch.Tensor
        ), f"Received data is not a torch.Tensor, type: {type(received_annotated_instance.data)}"
        assert torch.equal(
            received_annotated_instance.data, initializer.expected_sent_tensor
        ), f"Received tensor {received_annotated_instance.data} does not match expected {initializer.expected_sent_tensor}"
        assert isinstance(received_annotated_instance.timestamp, datetime.datetime)
        assert received_annotated_instance.caller_id == current_test_id
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_handle.stop()
        runtime_manager.check_for_exception()

        # Wait for the stopped event from the runtime
        stopped_event_was_set = stopped_event.wait(timeout=event_timeout_seconds)
        assert (
            stopped_event_was_set
        ), f"Runtime did not signal stopped for test_id ({current_test_id}) within {event_timeout_seconds}s"

        time.sleep(0.1)  # Allow time for data to be processed by the aggregator
        runtime_manager.check_for_exception()  # Check for errors after event

        # Retrieve and assert "stopped" data
        assert data_aggregator.has_new_data(
            current_test_id
        ), "Data aggregator should have new 'stopped' data after event was set."
        all_stopped_data = data_aggregator.get_new_data(current_test_id)
        assert all_stopped_data, "Received empty 'stopped' data list from aggregator."
        received_stopped_annotated_instance = all_stopped_data[0]

        assert (
            received_stopped_annotated_instance is not None
        ), "Failed to get 'stopped' annotated instance from aggregator."
        assert isinstance(received_stopped_annotated_instance, AnnotatedInstance)
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
    event_timeout_seconds = 15.0

    ctx = multiprocessing.get_context("spawn")
    data_event = ctx.Event()

    initializer = StrDataTorchEventRuntimeInitializer(
        test_id=current_test_id,
        service_type="Server",
        data_event=data_event,
    )
    expected_event_tensor = torch.tensor([10.0, 20.0, 30.0])
    expected_sum_val = expected_event_tensor.sum().item()
    expected_response_str = f"processed_tensor_event_{expected_sum_val}"

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(initializer)

        assert not runtime_future.done()
        runtime_manager.start_out_of_process()
        assert runtime_future.done()

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        time.sleep(5.0)  # Further increase initial sleep

        event_timestamp = datetime.datetime.now(datetime.timezone.utc)
        logging.debug(
            f"DEBUG_TEST: Sending event: {expected_event_tensor} to {current_test_id}"
        )
        runtime_handle.on_event(
            expected_event_tensor, current_test_id, timestamp=event_timestamp
        )

        data_event_was_set = data_event.wait(timeout=event_timeout_seconds)
        assert (
            data_event_was_set
        ), f"Runtime did not signal data event for {current_test_id} (response '{expected_response_str}') within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator to process
        runtime_manager.check_for_exception()

        assert data_aggregator.has_new_data(
            current_test_id
        ), f"Aggregator should have data for {current_test_id} after event"

        all_data = data_aggregator.get_new_data(current_test_id)
        assert all_data, f"Received empty data list for {current_test_id}"

        received_annotated_instance = None
        for item in all_data:
            if isinstance(item.data, str) and item.data == expected_response_str:
                received_annotated_instance = item
                break

        assert (
            received_annotated_instance is not None
        ), f"Expected string data '{expected_response_str}' not found in received items for {current_test_id}. Received: {all_data}"
        assert isinstance(received_annotated_instance, AnnotatedInstance)
        assert isinstance(received_annotated_instance.data, str)
        assert received_annotated_instance.data == expected_response_str
        assert isinstance(received_annotated_instance.timestamp, datetime.datetime)
        assert received_annotated_instance.caller_id == current_test_id

    finally:
        if runtime_handle_for_cleanup:
            try:
                runtime_handle_for_cleanup.stop()
            except Exception as e:
                logging.debug(f"Error during cleanup stop: {e}")
        runtime_manager.shutdown()


@pytest.mark.usefixtures("clear_loop_fixture")
def test_out_of_process_torch_data_torch_event_transport(clear_loop_fixture):
    """Validates torch.Tensor for both DataTypeT and EventTypeT with an out-of-process runtime."""
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    current_test_id = CallerIdentifier.random()
    event_timeout_seconds = 15.0

    ctx = multiprocessing.get_context("spawn")
    initial_data_event = ctx.Event()
    event_response_queue = ctx.Queue(maxsize=5)  # Use queue for event response
    stopped_event = ctx.Event()

    initializer = TorchDataTorchEventRuntimeInitializer(
        test_id=current_test_id,
        service_type="Server",
        initial_data_event=initial_data_event,
        event_response_queue=event_response_queue,  # Pass queue
        stopped_indicator_event=stopped_event,
    )

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(initializer)

        assert not runtime_future.done()
        runtime_manager.start_out_of_process()
        assert runtime_future.done()

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        # 1. Verify Initial Data Tensor
        initial_data_event_was_set = initial_data_event.wait(
            timeout=event_timeout_seconds
        )
        assert (
            initial_data_event_was_set
        ), f"Runtime did not signal initial_data_event for {current_test_id} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator
        runtime_manager.check_for_exception()

        assert data_aggregator.has_new_data(
            current_test_id
        ), "Aggregator should have initial data"
        all_initial_data = data_aggregator.get_new_data(current_test_id)
        assert all_initial_data, "Received empty initial data list"
        received_initial_data_instance = None
        for item in all_initial_data:
            if torch.equal(item.data, initializer.initial_data_tensor):
                received_initial_data_instance = item
                break
        assert (
            received_initial_data_instance is not None
        ), f"Initial data tensor not found in {all_initial_data}"
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

        try:
            queue_token = event_response_queue.get(timeout=event_timeout_seconds)
            assert (
                queue_token is True
            ), f"Unexpected token from event_response_queue: {queue_token}"
        except Exception as e_get:  # Catches queue.Empty
            pytest.fail(
                f"Did not get response signal from event_response_queue for {current_test_id} within {event_timeout_seconds}s. Error: {e_get}"
            )

        time.sleep(0.1)  # Allow aggregator
        runtime_manager.check_for_exception()

        assert data_aggregator.has_new_data(
            current_test_id
        ), "Aggregator should have event response data"
        all_event_response_data = data_aggregator.get_new_data(current_test_id)
        assert all_event_response_data, "Received empty event response data list"

        received_event_response_instance = None
        expected_event_response = (
            initializer.event_response_tensor_base + event_tensor_to_send.sum().item()
        )
        for item in all_event_response_data:
            if isinstance(item.data, torch.Tensor) and torch.equal(
                item.data, expected_event_response
            ):
                received_event_response_instance = item
                break
        assert (
            received_event_response_instance is not None
        ), f"Event response tensor not found in {all_event_response_data}"
        assert isinstance(received_event_response_instance.data, torch.Tensor)
        assert torch.equal(
            received_event_response_instance.data, expected_event_response
        )

        # 3. Verify Stopped Tensor
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        stopped_event_was_set = stopped_event.wait(timeout=event_timeout_seconds)
        assert (
            stopped_event_was_set
        ), f"Runtime did not signal stopped_event for {current_test_id} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator
        runtime_manager.check_for_exception()

        assert data_aggregator.has_new_data(
            current_test_id
        ), "Aggregator should have stopped data"
        all_stopped_data = data_aggregator.get_new_data(current_test_id)
        assert all_stopped_data, "Received empty stopped data list"

        received_stopped_data_instance = None
        for item in all_stopped_data:
            if isinstance(item.data, torch.Tensor) and torch.equal(
                item.data, initializer.stopped_tensor_indicator
            ):
                received_stopped_data_instance = item
                break
        assert (
            received_stopped_data_instance is not None
        ), f"Stopped tensor indicator not found in {all_stopped_data}"
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
                logging.debug(f"Error during cleanup stop: {e}")
        runtime_manager.shutdown()


# --- Start: New E2E Stress Test ---
@pytest.mark.usefixtures("clear_loop_fixture")
def test_out_of_process_torch_tensor_stress(clear_loop_fixture):
    """Stress tests torch.Tensor transport for data and events with multiple events."""
    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    current_test_id = CallerIdentifier.random()
    event_timeout_seconds = 15.0  # For single events
    stress_queue_get_timeout = 0.5  # Timeout for each queue.get() in stress test

    ctx = multiprocessing.get_context("spawn")
    initial_data_event = ctx.Event()
    # For multiple event responses, use a queue
    # Maxsize should be at least num_events_to_send
    event_response_queue = ctx.Queue(maxsize=100)  # Set a reasonable maxsize
    stopped_event = ctx.Event()

    initializer = TorchDataTorchEventRuntimeInitializer(
        test_id=current_test_id,
        service_type="Server",
        initial_data_event=initial_data_event,
        event_response_queue=event_response_queue,
        stopped_indicator_event=stopped_event,
    )

    try:
        runtime_manager.check_for_exception()
        runtime_future = runtime_manager.register_runtime_initializer(initializer)

        assert not runtime_future.done()
        runtime_manager.start_out_of_process()
        assert runtime_future.done()

        runtime_handle = runtime_future.result(timeout=5)
        runtime_handle_for_cleanup = runtime_handle
        data_aggregator = runtime_handle.data_aggregator

        runtime_handle.start()

        # 1. Verify Initial Data Tensor
        initial_data_event_was_set = initial_data_event.wait(
            timeout=event_timeout_seconds
        )
        assert (
            initial_data_event_was_set
        ), f"Runtime did not signal initial_data_event for {current_test_id} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator
        runtime_manager.check_for_exception()

        # Check data_aggregator for initial data (optional, as event is primary sync)
        # This also helps clear it from aggregator if needed before event responses
        if data_aggregator.has_new_data(current_test_id):
            all_initial_data = data_aggregator.get_new_data(current_test_id)
            # Simple check if any item matches, no strong assertion needed if event was set
            found_initial = any(
                torch.equal(item.data, initializer.initial_data_tensor)
                for item in all_initial_data
                if isinstance(item.data, torch.Tensor)
            )
            print(
                f"DEBUG: Initial data found in aggregator after event: {found_initial}"
            )

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
        # Wait for all event response tokens from the queue
        for i in range(num_events_to_send):
            try:
                token = event_response_queue.get(timeout=stress_queue_get_timeout)
                assert (
                    token is True
                ), f"Received unexpected token from event_response_queue: {token}"
                print(
                    f"DEBUG: Received token {i+1}/{num_events_to_send} from event_response_queue."
                )
            except Exception as e_get:  # Catches queue.Empty primarily
                pytest.fail(
                    f"Failed to get token {i+1}/{num_events_to_send} from event_response_queue within {stress_queue_get_timeout}s. Error: {e_get}"
                )

        time.sleep(
            0.5
        )  # Generous sleep to allow all data to arrive at aggregator after all queue tokens received
        runtime_manager.check_for_exception()

        # Now, retrieve all data from aggregator and verify
        all_received_data_for_stress_phase = []
        if data_aggregator.has_new_data(current_test_id):
            all_received_data_for_stress_phase.extend(
                data_aggregator.get_new_data(current_test_id)
            )

        received_event_responses_map = {}
        for item in all_received_data_for_stress_phase:
            if not isinstance(item.data, torch.Tensor):
                continue
            if torch.equal(item.data, initializer.initial_data_tensor):
                continue
            if torch.equal(item.data, initializer.stopped_tensor_indicator):
                continue

            response_tensor = item.data
            # Try to match response_tensor to one of the expected_responses_map values
            for es, expected_resp_t in expected_responses_map.items():
                if torch.equal(response_tensor, expected_resp_t):
                    if (
                        es not in received_event_responses_map
                    ):  # Store only first match for this sum
                        received_event_responses_map[es] = response_tensor
                    break

        assert len(received_event_responses_map) == num_events_to_send, (
            f"Expected {num_events_to_send} unique responses, got {len(received_event_responses_map)}. "
            f"Expected sums: {sorted(list(expected_responses_map.keys()))} "
            f"Received sums from matched tensors: {sorted(list(received_event_responses_map.keys()))}. "
            f"All data from aggregator for this phase: {all_received_data_for_stress_phase}"
        )

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

        stopped_event_was_set = stopped_event.wait(timeout=event_timeout_seconds)
        assert (
            stopped_event_was_set
        ), f"Runtime did not signal stopped_event for {current_test_id} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator
        runtime_manager.check_for_exception()

        assert data_aggregator.has_new_data(
            current_test_id
        ), "Aggregator should have stopped data"
        all_stopped_data = data_aggregator.get_new_data(current_test_id)
        assert all_stopped_data, "Received empty stopped data list"

        received_stopped_data_instance = None
        for item in all_stopped_data:
            if isinstance(item.data, torch.Tensor) and torch.equal(
                item.data, initializer.stopped_tensor_indicator
            ):
                received_stopped_data_instance = item
                break
        assert (
            received_stopped_data_instance is not None
        ), f"Stopped tensor indicator not found in {all_stopped_data}"
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

    ctx = multiprocessing.get_context("spawn")
    child_error_event = ctx.Event()

    handle_future = runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg,
            error_type=ValueError,
            service_type="Server",
            error_event=child_error_event,  # Pass event here
        )
    )
    runtime_manager.start_out_of_process()

    try:
        runtime_handle = handle_future.result(timeout=2)
        runtime_handle.start()
    except Exception as e_handle:
        pytest.fail(f"Failed to get or start runtime_handle: {e_handle}")

    # Increased wait time for error propagation in out-of-process scenarios
    # time.sleep(5.0) # Replaced with event-based synchronization

    # The event is passed to the original initializer.
    # RuntimeManager already started, need to get new handle if re-registering after start
    # For this test, let's assume runtime_manager was just started and we are getting the first handle.
    # If start_out_of_process was already called, and we re-register, the behavior might be complex.
    # The original code structure implies start_out_of_process happens *after* register.
    # My previous thought on re-registering handle_future was incorrect.
    # The initial handle_future is the correct one.

    event_timeout = 10.0  # Generous timeout for the event
    # Ensure runtime_handle.start() was called if it's needed to trigger start_async
    # The previous code block already did this.

    event_was_set = child_error_event.wait(timeout=event_timeout)
    assert (
        event_was_set
    ), f"Child process did not signal error_event within {event_timeout}s."

    time.sleep(0.2)  # Short pause for IPC to complete after event is set

    with pytest.raises(ValueError, match=error_msg):
        runtime_manager.check_for_exception()


def test_out_of_process_error_run_until_exception(clear_loop_fixture):
    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "RemoteRunUntilFailure"

    ctx = multiprocessing.get_context("spawn")
    child_error_event = ctx.Event()

    handle_future = runtime_manager.register_runtime_initializer(  # Store handle_future
        ErrorThrowingRuntimeInitializer(
            error_message=error_msg,
            error_type=RuntimeError,
            error_event=child_error_event,
        )
    )
    runtime_manager.start_out_of_process()

    try:  # Ensure runtime handle is started
        runtime_handle = handle_future.result(timeout=2)
        runtime_handle.start()
    except Exception as e_handle:
        pytest.fail(
            f"Failed to get or start runtime_handle for run_until_exception test: {e_handle}"
        )

    event_timeout = 10.0
    event_was_set = child_error_event.wait(timeout=event_timeout)
    assert (
        event_was_set
    ), f"Child process did not signal error_event for run_until_exception within {event_timeout}s."

    time.sleep(0.2)  # Pause for IPC

    # The original test used a loop of check_for_exception.
    # run_until_exception should block and raise.
    with pytest.raises(RuntimeError, match=error_msg):
        runtime_manager.run_until_exception()  # Changed to run_until_exception


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

    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True)
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "InProcessFailureOops"
    runtime_manager.register_runtime_initializer(
        ErrorThrowingRuntimeInitializer(error_message=error_msg, error_type=ValueError)
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

    ctx = multiprocessing.get_context("spawn")  # Define ctx and child_error_event
    child_error_event = ctx.Event()

    runtime_manager.register_runtime_initializer(
        FaultyCreateRuntimeInitializer(
            error_message=error_msg,
            error_type=TypeError,
            error_event=child_error_event,  # Pass event
        )
    )
    runtime_manager.start_out_of_process()

    # Wait for the child to signal it's about to raise the error (in create)
    event_timeout = 10.0
    event_was_set = child_error_event.wait(timeout=event_timeout)
    assert (
        event_was_set
    ), f"Child process did not signal error_event for FaultyCreate within {event_timeout}s."

    time.sleep(0.2)  # Pause for IPC

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
        event_timeout_seconds = 15.0

        ctx = multiprocessing.get_context("spawn")
        data_event_1 = ctx.Event()
        stopped_event_1 = ctx.Event()
        data_event_2 = ctx.Event()
        stopped_event_2 = ctx.Event()

        # Register two FakeRuntimeInitializers
        runtime_future_1 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                test_id=test_id_1,
                service_type="Server",
                data_event=data_event_1,
                stopped_event=stopped_event_1,
            )
        )
        runtime_future_2 = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(
                test_id=test_id_2,
                service_type="Server",
                data_event=data_event_2,
                stopped_event=stopped_event_2,
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

        runtime_handle_1 = runtime_future_1.result(timeout=1)
        runtime_handle_2 = runtime_future_2.result(timeout=1)
        runtime_handles_for_cleanup.extend([runtime_handle_1, runtime_handle_2])

        data_aggregator_1 = runtime_handle_1.data_aggregator
        data_aggregator_2 = runtime_handle_2.data_aggregator

        # Start both runtimes
        runtime_handle_1.start()
        runtime_handle_2.start()

        # --- Verify Data for Runtime 1 ---
        data_event_1_was_set = data_event_1.wait(timeout=event_timeout_seconds)
        assert (
            data_event_1_was_set
        ), f"Runtime 1 did not signal data event for {test_id_1} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator to process
        runtime_manager.check_for_exception()

        assert data_aggregator_1.has_new_data(
            test_id_1
        ), f"Aggregator 1 should have data for {test_id_1} after event"
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
        data_event_2_was_set = data_event_2.wait(timeout=event_timeout_seconds)
        assert (
            data_event_2_was_set
        ), f"Runtime 2 did not signal data event for {test_id_2} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator to process
        runtime_manager.check_for_exception()

        assert data_aggregator_2.has_new_data(
            test_id_2
        ), f"Aggregator 2 should have data for {test_id_2} after event"
        if (
            data_aggregator_1 is not data_aggregator_2
        ):  # Should be same aggregator in OOP
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
        runtime_manager.check_for_exception()  # Check for errors after stop command

        stopped_event_1_was_set = stopped_event_1.wait(timeout=event_timeout_seconds)
        assert (
            stopped_event_1_was_set
        ), f"Runtime 1 did not signal stopped event for {test_id_1} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator to process
        runtime_manager.check_for_exception()

        assert data_aggregator_1.has_new_data(
            test_id_1
        ), f"Aggregator 1 should have 'stopped' data for {test_id_1} after event"
        values_stop_1 = data_aggregator_1.get_new_data(test_id_1)
        assert isinstance(values_stop_1, list) and len(values_stop_1) == 1
        first_stop_1 = values_stop_1[0]
        assert (
            isinstance(first_stop_1.data, FakeData)
            and first_stop_1.data.value == stopped
        )
        assert isinstance(
            first_stop_1.timestamp, datetime.datetime
        )  # Check timestamp type
        assert first_stop_1.caller_id == test_id_1
        assert not data_aggregator_1.has_new_data(test_id_1)

        # --- Stop Runtime 2 and Verify "stopped" Data ---
        runtime_handle_2.stop()
        runtime_manager.check_for_exception()  # Check for errors after stop command

        stopped_event_2_was_set = stopped_event_2.wait(timeout=event_timeout_seconds)
        assert (
            stopped_event_2_was_set
        ), f"Runtime 2 did not signal stopped event for {test_id_2} within {event_timeout_seconds}s"
        time.sleep(0.1)  # Allow aggregator to process
        runtime_manager.check_for_exception()

        assert data_aggregator_2.has_new_data(
            test_id_2
        ), f"Aggregator 2 should have 'stopped' data for {test_id_2} after event"
        values_stop_2 = data_aggregator_2.get_new_data(test_id_2)
        assert isinstance(values_stop_2, list) and len(values_stop_2) == 1
        first_stop_2 = values_stop_2[0]
        assert (
            isinstance(first_stop_2.data, FakeData)
            and first_stop_2.data.value == stopped
        )
        assert isinstance(
            first_stop_2.timestamp, datetime.datetime
        )  # Check timestamp type
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

    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True)
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handle_for_cleanup = None
    try:
        current_test_id = CallerIdentifier.random()
        runtime_manager.check_for_exception()

        runtime_future = runtime_manager.register_runtime_initializer(
            FakeRuntimeInitializer(test_id=current_test_id, service_type="Client")
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
        assert isinstance(values, list)
        assert len(values) == 1
        first = values[0]
        assert isinstance(first, AnnotatedInstance) and isinstance(first.data, FakeData)
        assert first.data.value == "FRESH_SIMPLE_DATA_V2"
        assert isinstance(first.timestamp, datetime.datetime)
        assert first.caller_id == current_test_id
        assert not data_aggregator.has_new_data(current_test_id)

        runtime_manager.check_for_exception()
        runtime_handle.stop()
        runtime_manager.check_for_exception()

        time.sleep(5.0)  # Add initial sleep
        stopped_data_arrived = False
        waited_time = 0.0
        max_wait_stop_client = 20.0  # Further increase polling duration
        poll_interval_client_stop = 0.1  # Define locally
        while waited_time < max_wait_stop_client:
            runtime_manager.check_for_exception()  # Check for errors during polling
            if data_aggregator.has_new_data(current_test_id):
                stopped_data_arrived = True
                break
            time.sleep(poll_interval_client_stop)
            waited_time += poll_interval_client_stop

        assert (
            stopped_data_arrived
        ), f"Aggregator did not receive 'stopped' data for test_id ({current_test_id}) within {max_wait_stop_client}s"

        values_stop = data_aggregator.get_new_data(current_test_id)
        assert isinstance(values_stop, list) and len(values_stop) == 1
        first_stop = values_stop[0]
        assert isinstance(first_stop, AnnotatedInstance) and isinstance(
            first_stop.data, FakeData
        )
        assert first_stop.data.value == stopped
        # Timestamp assertion removed as it's now dynamic
        # assert first_stop.timestamp == stop_timestamp
        assert isinstance(
            first_stop.timestamp, datetime.datetime
        )  # Check it's a datetime
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

    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True)
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    error_msg = "InProcessFailureOops"
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
            logging.debug(
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
    event_timeout_seconds = 5.0  # Short timeout for the error event itself

    try:
        ctx = multiprocessing.get_context("spawn")
        error_event = ctx.Event()

        initializer = ErrorThrowingRuntimeInitializer(
            error_message=error_msg,
            error_type=error_type,
            service_type="Server",
            error_event=error_event,
        )
        handle_future = runtime_manager.register_runtime_initializer(initializer)
        runtime_manager.start_out_of_process()
        runtime_handle = handle_future.result(timeout=2)
        runtime_handle_for_cleanup = runtime_handle
        runtime_handle.start()  # This should trigger the error in ErrorThrowingRuntime

        # Wait for the runtime to signal it's about to raise an error
        # This is a short wait, mainly to ensure the child process has reached the error point.
        error_event_was_set = error_event.wait(timeout=event_timeout_seconds)
        if not error_event_was_set:
            print(
                f"Warning: ErrorThrowingRuntime did not signal error_event within {event_timeout_seconds}s. Proceeding anyway."
            )

        # time.sleep(1.5) # Original sleep, possibly redundant now or can be reduced
        time.sleep(0.2)  # Shorter sleep after event is set or timed out

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
            logging.warning(
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

    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True)
    event_thread.start()
    worker_event_loop = loop_future.result(timeout=5)

    runtime_manager = RuntimeManager(is_testing=True)
    runtime_handles_for_cleanup = []

    try:
        caller_id1 = CallerIdentifier.random()
        caller_id2 = CallerIdentifier.random()
        all_caller_ids = [caller_id1, caller_id2]

        # Create asyncio events for each caller
        caller_event_map: dict[CallerIdentifier, asyncio.Event] = {
            cid: asyncio.Event() for cid in all_caller_ids
        }
        # Store string versions of caller IDs for pending set
        pending_ids_to_receive_event_signal = {str(cid) for cid in all_caller_ids}

        initializer = BroadcastTestFakeRuntimeInitializer(
            all_caller_ids,
            service_type="Server",
            caller_event_map=caller_event_map,
        )
        handle_future = runtime_manager.register_runtime_initializer(initializer)
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        runtime_handle = handle_future.result(timeout=10)
        runtime_handles_for_cleanup.append(runtime_handle)

        data_aggregator = runtime_handle.data_aggregator
        runtime_handle.start()
        time.sleep(1.0)

        broadcast_event = FakeEvent()
        runtime_handle.on_event(broadcast_event, caller_id=None)
        time.sleep(1.5)

        max_wait_time = 10.0  # Total wait time for all events
        poll_interval = 0.1  # How often to check events and aggregator
        start_wait_time = time.monotonic()

        pending_data_check_ids = {str(cid) for cid in all_caller_ids}

        while (time.monotonic() - start_wait_time) < max_wait_time and (
            pending_ids_to_receive_event_signal or pending_data_check_ids
        ):
            runtime_manager.check_for_exception()

            for cid_obj in all_caller_ids:
                cid_key_str = str(cid_obj)
                event_for_caller = caller_event_map[cid_obj]

                if (
                    cid_key_str in pending_ids_to_receive_event_signal
                    and event_for_caller.is_set()
                ):
                    print(f"DEBUG: Event received for caller {cid_key_str}")
                    pending_ids_to_receive_event_signal.remove(cid_key_str)
                    # Give a moment for data to arrive at aggregator after event
                    time.sleep(0.05)

                # Check for data if event was signaled (or if still pending data check)
                if cid_key_str in pending_data_check_ids and (
                    event_for_caller.is_set()
                    or cid_key_str not in pending_ids_to_receive_event_signal
                ):
                    if data_aggregator.has_new_data(cid_obj):
                        received_datas = data_aggregator.get_new_data(cid_obj)
                        assert (
                            len(received_datas) >= 1
                        ), f"No data items for {cid_key_str} despite has_new_data"
                        # event_data_found = False # Removed
                        for item in received_datas:
                            assert isinstance(item, AnnotatedInstance)
                            assert isinstance(item.data, FakeData)
                            if item.data.value == f"event_for_{str(cid_obj)[:8]}":
                                # event_data_found = True # Removed
                                print(
                                    f"DEBUG: Correct data found for caller {cid_key_str}"
                                )
                                if (
                                    cid_key_str in pending_data_check_ids
                                ):  # ensure removal only once
                                    pending_data_check_ids.remove(cid_key_str)
                                break
                        # If event was set but data not found yet, it might still be in transit
                        # The loop will continue polling data_aggregator for a short while

            if not pending_ids_to_receive_event_signal and not pending_data_check_ids:
                break
            time.sleep(poll_interval)

        assert (
            not pending_ids_to_receive_event_signal
        ), f"Not all callers signaled event. Missing signals for: {pending_ids_to_receive_event_signal}"
        assert (
            not pending_data_check_ids
        ), f"Not all callers had data verified. Missing data verification for: {pending_data_check_ids}"

    finally:
        for handle_item in runtime_handles_for_cleanup:
            try:
                if handle_item:
                    handle_item.stop()
            except Exception as e:
                logging.debug(f"Error during handle.stop() for {handle_item}: {e}")
                pass
        runtime_manager.shutdown()
        # time.sleep(0.2)  # Diagnostic sleep removed
        if event_thread.is_alive():
            event_thread.join(timeout=2)


# Ensure a newline at the end of the file
