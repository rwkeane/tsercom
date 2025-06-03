import asyncio
import pytest
import grpc.aio  # For ServicerContext

from unittest.mock import (
    MagicMock,
    AsyncMock,
    Mock,  # Added Mock
)
from typing import (
    Optional,
    Any,
    TypeVar,
    List,  # Added List
)
import datetime

# Patches
from unittest.mock import patch

# SUT
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase

# Dependencies to be mocked or used
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.timesync.common.proto import ServerTimestamp
from tsercom.runtime.id_tracker import IdTracker  # Added
from tsercom.runtime.caller_processor_registry import (
    CallerProcessorRegistry,
)  # Added

# For gRPC context testing
# import grpc # Already imported via grpc.aio
from grpc.aio import ServicerContext

# Modules for patching
# import tsercom.rpc.grpc_util.addressing as grpc_addressing_module # Not used directly in this diff

# Type variable for data
DataType = TypeVar("DataType")
TEventType = TypeVar("TEventType")  # For generic type hints


# Test Subclass of RuntimeDataHandlerBase
class TestableRuntimeDataHandler(
    RuntimeDataHandlerBase[DataType, TEventType]  # Use TEventType
):
    __test__ = False  # Mark this class as not a test class for pytest

    def __init__(
        self,
        data_reader: RemoteDataReader[DataType],
        event_source: AsyncPoller[SerializableAnnotatedInstance[TEventType]],
        id_tracker: IdTracker,
        processor_registry: CallerProcessorRegistry,
        mocker,  # Keep mocker for other mocks
    ):
        super().__init__(
            data_reader, event_source, id_tracker, processor_registry
        )
        self.mock_register_caller_impl = mocker.AsyncMock(
            name="_register_caller_impl"
        )
        self.mock_unregister_caller_impl = mocker.MagicMock(
            name="_unregister_caller_impl", return_value=True
        )
        self.mock_try_get_caller_id_impl = mocker.MagicMock(
            name="_try_get_caller_id_impl"
        )

        # This mock is for the parent class's _on_data_ready if we need to assert calls to it.
        # However, the actual method being tested by test_handler_on_data_ready_calls_reader_on_data_ready
        # is the one on RuntimeDataHandlerBase, not this mock.
        # This specific mock on self might not be needed if we test via the base method directly.
        # For now, keeping it if other tests might rely on it.
        self._on_data_ready_mock = mocker.AsyncMock(
            name="instance_on_data_ready_mock"
        )

    # Abstract method implementations for the testable subclass
    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[
        DataType, TEventType
    ]:  # Match base class signature
        return await self.mock_register_caller_impl(caller_id, endpoint, port)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        return self.mock_unregister_caller_impl(caller_id)

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> Optional[CallerIdentifier]:
        return self.mock_try_get_caller_id_impl(endpoint, port)

    # If TestableRuntimeDataHandler's _on_data_ready was being mocked for some tests:
    async def _on_data_ready(self, data: AnnotatedInstance[DataType]) -> None:
        await self._on_data_ready_mock(data)
        # To also call the real base implementation if needed for some tests (unlikely for unit tests):
        # await super()._on_data_ready(data)


# Fixtures moved to module level for wider access
@pytest.fixture
def mock_data_reader_fixture(mocker) -> RemoteDataReader:
    reader = mocker.MagicMock(spec=RemoteDataReader)
    reader._on_data_ready = mocker.MagicMock(
        name="data_reader_on_data_ready_mock"
    )
    return reader


@pytest.fixture
def mock_event_source_fixture(mocker) -> AsyncPoller:
    poller = mocker.AsyncMock(spec=AsyncPoller)
    poller.__aiter__ = Mock(return_value=poller)
    poller.__anext__ = AsyncMock()
    return poller


@pytest.fixture
def mock_servicer_context(mocker) -> ServicerContext:
    context = mocker.AsyncMock(spec=ServicerContext)
    context.peer = mocker.MagicMock(return_value="ipv4:127.0.0.1:12345")
    return context


class TestRuntimeDataHandlerBaseBehavior:

    @pytest.fixture
    def mock_id_tracker(self, mocker) -> IdTracker:
        return mocker.MagicMock(spec=IdTracker)

    @pytest.fixture
    def mock_processor_registry(self, mocker) -> CallerProcessorRegistry:
        registry = mocker.MagicMock(spec=CallerProcessorRegistry)
        mock_poller_for_registry = mocker.AsyncMock(spec=AsyncPoller)
        # SUT calls on_available, not push, on the processor (poller)
        mock_poller_for_registry.on_available = mocker.MagicMock(
            name="processor_on_available_mock"
        )
        registry.get_or_create_processor = mocker.MagicMock(
            name="get_or_create_processor",
            return_value=mock_poller_for_registry,
        )
        return registry

    @pytest.fixture
    def mock_caller_id(self, mocker) -> CallerIdentifier:
        return mocker.MagicMock(
            spec=CallerIdentifier, name="MockCallerIdInstance"
        )

    @pytest.fixture
    def patch_get_client_ip(self, mocker):
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            autospec=True,
        )
        yield mock_get_ip

    @pytest.fixture
    def patch_get_client_port(self, mocker):
        mock_get_port = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            autospec=True,
        )
        yield mock_get_port

    @pytest.fixture
    def handler(
        self,
        mock_data_reader_fixture,
        mock_event_source_fixture,
        mock_id_tracker,
        mock_processor_registry,
        mocker,
    ):  # Changed to use _fixture names
        # Patch asyncio.create_task before TestableRuntimeDataHandler is instantiated
        with patch("asyncio.create_task") as mock_create_task:
            handler_instance = TestableRuntimeDataHandler(
                mock_data_reader_fixture,
                mock_event_source_fixture,
                mock_id_tracker,
                mock_processor_registry,
                mocker,  # Changed to use _fixture names
            )
            handler_instance.mock_create_task = (
                mock_create_task  # Attach for assertions
            )
            yield handler_instance

    # Fixtures for _RuntimeDataHandlerBase__ConcreteDataProcessor tests
    @pytest.fixture
    def mock_sync_clock(self, mocker):
        clock = mocker.MagicMock(spec=SynchronizedClock)
        clock.desync = mocker.MagicMock(
            name="sync_clock_desync"
        )  # Changed to MagicMock
        return clock

    @pytest.fixture
    def test_caller_id_instance(
        self,
    ):  # A concrete CallerIdentifier for processor tests
        return (
            CallerIdentifier.random()
        )  # Assuming CallerIdentifier has a factory or suitable constructor

    @pytest.fixture
    def data_processor_with_mock_poller(  # Renamed fixture
        self,
        handler: TestableRuntimeDataHandler,
        test_caller_id_instance: CallerIdentifier,
        mock_sync_clock: SynchronizedClock,
        mocker,
    ):
        # Patch AsyncPoller specific to the context of creating this data_processor
        # This mock_poller_instance will be what __ConcreteDataProcessor.__internal_poller becomes.
        mock_poller_instance = mocker.AsyncMock(spec=AsyncPoller)
        # Configure its __anext__ as it will be awaited.
        # Specific tests will further configure side_effect or return_value of this __anext__.
        mock_poller_instance.__anext__ = mocker.AsyncMock(
            name="__anext__ on internal poller"
        )

        with patch(
            "tsercom.runtime.runtime_data_handler_base.AsyncPoller",
            return_value=mock_poller_instance,
        ) as mock_AsyncPoller_class:
            # _create_data_processor is part of RuntimeDataHandlerBase and will now use the patched AsyncPoller
            processor_instance = handler._create_data_processor(
                test_caller_id_instance, mock_sync_clock
            )
            # Attach the mock poller instance to the processor for test assertions, if needed,
            # though ideally we assert behavior via the processor's methods.
            # For testing __anext__, we need to control this mock_poller_instance.
            # This also helps verify that the patched poller was indeed used.
            # Removing this assertion as it might be causing issues,
            # the tests themselves will verify interaction with the mock_poller_instance.
            # assert processor_instance._RuntimeDataHandlerBase__ConcreteDataProcessor__internal_poller is mock_poller_instance
            return (
                processor_instance,
                mock_poller_instance,
            )  # Return both for tests

    # --- Tests for RuntimeDataHandlerBase direct methods ---
    def test_constructor_and_init_task(
        self,
        handler: TestableRuntimeDataHandler,  # handler fixture already gets these
        mock_data_reader_fixture,
        mock_event_source_fixture,
        mock_id_tracker,
        mock_processor_registry,
    ):
        assert (
            handler._RuntimeDataHandlerBase__data_reader
            is mock_data_reader_fixture
        )  # Check against what handler got
        assert (
            handler._event_source is mock_event_source_fixture
        )  # Check against what handler got
        assert handler._id_tracker is mock_id_tracker
        assert (
            handler._RuntimeDataHandlerBase__processor_registry
            is mock_processor_registry
        )
        # Check that asyncio.create_task was called
        handler.mock_create_task.assert_called_once()
        # Check that it was called with the correct coroutine method
        # Access the coroutine object passed to the first call's first argument
        args, _ = handler.mock_create_task.call_args
        called_coro = args[0]
        # Check the name of the coroutine function/method
        # For a bound method, __name__ gives the method name, and __self__ points to the instance.
        assert (
            called_coro.__name__
            == "_RuntimeDataHandlerBase__propagate_instances"
        )  # Changed to assert
        # It's good practice to also ensure it's a coroutine, though create_task would fail otherwise
        assert asyncio.iscoroutine(
            called_coro
        ), "Argument to create_task was not a coroutine"  # Changed to assert

    @pytest.mark.asyncio
    async def test_register_caller_endpoint_port(  # No mocker needed directly here
        self,
        handler: TestableRuntimeDataHandler,
        mock_caller_id: CallerIdentifier,
    ):
        endpoint_str = "10.0.0.1"
        port_num = 9999
        # The mock_register_caller_impl is on the handler instance (TestableRuntimeDataHandler)
        mock_processor_instance = MagicMock(spec=EndpointDataProcessor)
        handler.mock_register_caller_impl.return_value = (
            mock_processor_instance
        )

        returned_processor = (
            await handler.register_caller(  # This calls the public method
                mock_caller_id, endpoint_str, port_num
            )
        )
        # Assert the concrete implementation's mock was called via the public method
        handler.mock_register_caller_impl.assert_called_once_with(
            mock_caller_id, endpoint_str, port_num
        )
        assert returned_processor is mock_processor_instance

    @pytest.mark.asyncio
    async def test_register_caller_grpc_context_success(
        self,
        handler: TestableRuntimeDataHandler,
        mock_caller_id: CallerIdentifier,
        mock_servicer_context: ServicerContext,
        patch_get_client_ip,
        patch_get_client_port,
    ):
        expected_ip = "1.2.3.4"
        expected_port = 5678
        patch_get_client_ip.return_value = expected_ip
        patch_get_client_port.return_value = expected_port

        mock_processor_instance = MagicMock(spec=EndpointDataProcessor)
        handler.mock_register_caller_impl.return_value = (
            mock_processor_instance
        )

        returned_processor = await handler.register_caller(
            mock_caller_id, context=mock_servicer_context
        )
        patch_get_client_ip.assert_called_once_with(mock_servicer_context)
        patch_get_client_port.assert_called_once_with(mock_servicer_context)
        handler.mock_register_caller_impl.assert_called_once_with(
            mock_caller_id, expected_ip, expected_port
        )
        assert returned_processor is mock_processor_instance

    @pytest.mark.asyncio
    async def test_register_caller_grpc_context_no_ip(
        self,
        handler: TestableRuntimeDataHandler,
        mock_caller_id: CallerIdentifier,
        mock_servicer_context: ServicerContext,
        patch_get_client_ip,
        patch_get_client_port,
    ):
        patch_get_client_ip.return_value = None
        patch_get_client_port.return_value = 5678

        returned_processor = await handler.register_caller(  # Added await
            mock_caller_id, context=mock_servicer_context
        )
        patch_get_client_ip.assert_called_once_with(mock_servicer_context)
        patch_get_client_port.assert_called_once_with(
            mock_servicer_context
        )  # Still called by current logic
        handler.mock_register_caller_impl.assert_not_called()
        assert returned_processor is None

    def test_check_for_caller_id(  # Unchanged, should still work
        self,
        handler: TestableRuntimeDataHandler,
        mock_caller_id: CallerIdentifier,
    ):
        endpoint_str = "test_ep"
        port_num = 1122
        handler.mock_try_get_caller_id_impl.return_value = mock_caller_id
        result = handler.check_for_caller_id(endpoint_str, port_num)
        handler.mock_try_get_caller_id_impl.assert_called_once_with(
            endpoint_str, port_num
        )
        assert result is mock_caller_id

    @pytest.mark.asyncio
    async def test_handler_on_data_ready_calls_reader_on_data_ready(
        self,
        handler: TestableRuntimeDataHandler,
        mock_data_reader_fixture: RemoteDataReader,
        mocker,
    ):
        mock_annotated_instance = mocker.MagicMock(spec=AnnotatedInstance)
        # The handler fixture already sets up the SUT with mock_data_reader_fixture.
        # We assert that the one IN THE SUT was called.
        await RuntimeDataHandlerBase._on_data_ready(
            handler, mock_annotated_instance
        )  # Call the actual method

        # mock_data_reader_fixture is the same object that was passed to the handler
        mock_data_reader_fixture._on_data_ready.assert_called_once_with(
            mock_annotated_instance
        )

    # --- Tests for _RuntimeDataHandlerBase__ConcreteDataProcessor ---

    @pytest.mark.asyncio
    async def test_processor_desynchronize(
        self, data_processor_with_mock_poller, mock_sync_clock, mocker
    ):
        data_processor, _ = (
            data_processor_with_mock_poller  # Already using new name and unpacking
        )
        mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
        expected_datetime = datetime.datetime.now(datetime.timezone.utc)
        mock_sync_clock.desync.return_value = expected_datetime

        result_dt = await data_processor.desynchronize(mock_server_ts)

        mock_sync_clock.desync.assert_called_once_with(mock_server_ts)
        assert result_dt is expected_datetime

    @pytest.mark.asyncio
    async def test_processor_deregister_caller(
        self,
        data_processor_with_mock_poller,
        handler: TestableRuntimeDataHandler,
        test_caller_id_instance: CallerIdentifier,
    ):
        data_processor, _ = (
            data_processor_with_mock_poller  # Already using new name and unpacking
        )
        await data_processor.deregister_caller()
        handler.mock_unregister_caller_impl.assert_called_once_with(
            test_caller_id_instance
        )

    @pytest.mark.asyncio
    async def test_processor_process_data_with_datetime(
        self,
        data_processor_with_mock_poller,
        handler: TestableRuntimeDataHandler,
        test_caller_id_instance: CallerIdentifier,
    ):
        data_processor, _ = (
            data_processor_with_mock_poller  # Already using new name and unpacking
        )
        test_payload = "test_payload_data"
        test_dt = datetime.datetime.now(datetime.timezone.utc)
        # To test the actual _on_data_ready of the base for this data_processor:
        # We need to ensure the data_processor's internal __data_handler calls the correct _on_data_ready
        # The handler._on_data_ready_mock is on the TestableRuntimeDataHandler instance itself.
        # The data_processor (an instance of __ConcreteDataProcessor) has a reference to its
        # parent handler. We need to mock the parent handler's _on_data_ready.
        # The current `handler._on_data_ready_mock` is suitable if data_processor correctly calls it.

        # Let's assume the data_processor's __data_handler is the 'handler' fixture instance.
        # We need to check if handler._on_data_ready_mock is called.
        # Or, if we want to test the *base* _on_data_ready, we'd need to spy on mock_data_reader.

        # Re-directing the test to assert the mock on the TestableRuntimeDataHandler instance
        # that was passed to the __ConcreteDataProcessor.
        # `data_processor` was created with `handler` as its `data_handler`.
        # `handler` is an instance of `TestableRuntimeDataHandler`.
        # `TestableRuntimeDataHandler` has `_on_data_ready_mock`.

        await data_processor.process_data(test_payload, test_dt)

        handler._on_data_ready_mock.assert_called_once()
        args, _ = handler._on_data_ready_mock.call_args
        annotated_instance: AnnotatedInstance = args[0]

        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == test_dt

    @pytest.mark.asyncio
    async def test_processor_process_data_with_server_timestamp(  # Uses handler._on_data_ready_mock
        self,
        data_processor_with_mock_poller,
        handler: TestableRuntimeDataHandler,  # The parent handler
        test_caller_id_instance: CallerIdentifier,
        mock_sync_clock: SynchronizedClock,  # Clock used by data_processor
        mocker,
    ):
        data_processor, _ = data_processor_with_mock_poller  # Unpack
        test_payload = "payload_with_server_ts"
        mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
        expected_desynced_dt = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=5)
        mock_sync_clock.desync.return_value = expected_desynced_dt

        await data_processor.process_data(test_payload, mock_server_ts)

        mock_sync_clock.desync.assert_called_once_with(mock_server_ts)
        handler._on_data_ready_mock.assert_called_once()  # Asserting the mock on TestableRuntimeDataHandler
        args, _ = handler._on_data_ready_mock.call_args
        annotated_instance: AnnotatedInstance = args[0]

        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == expected_desynced_dt

    @pytest.mark.asyncio
    async def test_processor_process_data_no_timestamp(  # Uses handler._on_data_ready_mock
        self,
        data_processor_with_mock_poller,
        handler: TestableRuntimeDataHandler,  # The parent handler
        test_caller_id_instance: CallerIdentifier,
        mocker,
    ):
        data_processor, _ = data_processor_with_mock_poller  # Unpack
        test_payload = "payload_no_ts"
        fixed_now = datetime.datetime.now(datetime.timezone.utc)

        # Patching datetime.now used by EndpointDataProcessor.process_data
        mock_datetime_now = mocker.patch(
            "tsercom.runtime.endpoint_data_processor.datetime"
        )
        mock_datetime_now.now.return_value = fixed_now
        # Need to also ensure timezone.utc is available or mock it if that's how it's used.
        # The SUT uses datetime.now(timezone.utc). The above mock should cover `datetime.now`.
        # If `timezone` itself is imported as `from datetime import timezone` and then used,
        # that would need separate handling if `datetime.timezone.utc` was the issue.
        # But `datetime.now.assert_called_once_with(datetime.timezone.utc)` will check this.

        await data_processor.process_data(test_payload, timestamp=None)

        mock_datetime_now.now.assert_called_once_with(datetime.timezone.utc)
        handler._on_data_ready_mock.assert_called_once()
        args, _ = handler._on_data_ready_mock.call_args
        annotated_instance: AnnotatedInstance = args[0]

        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == fixed_now

    # --- Tests for __ConcreteDataProcessor AsyncIterator interface ---
    @pytest.mark.asyncio
    async def test_concrete_processor_aiter_returns_self(
        self, data_processor_with_mock_poller
    ):
        """Tests that __aiter__ returns self."""
        concrete_processor, _ = data_processor_with_mock_poller
        assert concrete_processor.__aiter__() is concrete_processor

    @pytest.mark.asyncio
    async def test_concrete_processor_anext_retrieves_from_poller(
        self, data_processor_with_mock_poller, mocker
    ):
        """Tests that __anext__ retrieves items from the internal poller."""
        concrete_processor, mock_internal_poller = (
            data_processor_with_mock_poller
        )

        expected_items = [mocker.MagicMock(spec=SerializableAnnotatedInstance)]
        mock_internal_poller.__anext__.return_value = (
            expected_items  # Configure the mock poller's __anext__
        )

        retrieved_items = await concrete_processor.__anext__()

        assert retrieved_items == expected_items
        mock_internal_poller.__anext__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concrete_processor_anext_raises_stop_async_iteration(
        self, data_processor_with_mock_poller, mocker
    ):
        """Tests that __anext__ raises StopAsyncIteration when poller signals end."""
        concrete_processor, mock_internal_poller = (
            data_processor_with_mock_poller
        )

        mock_internal_poller.__anext__.side_effect = (
            StopAsyncIteration  # Poller's __anext__ raises this
        )

        with pytest.raises(StopAsyncIteration):
            await concrete_processor.__anext__()
        mock_internal_poller.__anext__.assert_awaited_once()

    # Helper for testing __aiter__ more easily
    async def awaitable_aiter(
        async_iterable,
    ):  # Made this a static method or move out of class
        return async_iterable.__aiter__()

    # --- New tests for propagation and routing ---

    @pytest.mark.asyncio
    async def test_propagate_instances_routes_events(
        self,
        handler: TestableRuntimeDataHandler,
        mock_event_source_fixture: AsyncMock,
        mocker,
    ):
        """Tests that __propagate_instances correctly polls event_source and routes instances."""
        # mock_event_source_fixture is the one passed to the handler
        instance1 = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        instance2 = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        instance3 = mocker.MagicMock(spec=SerializableAnnotatedInstance)

        mock_event_source_fixture.__anext__.side_effect = (
            [  # Use mock_event_source_fixture
                [instance1, instance2],
                [instance3],
                StopAsyncIteration,  # Signal end of async iteration
            ]
        )

        # Mock the actual routing method to check calls
        handler._RuntimeDataHandlerBase__route_instance = AsyncMock(
            name="__route_instance_mock"
        )

        # Directly call __propagate_instances for testing its logic
        # asyncio.create_task was already tested in test_constructor_and_init_task
        await handler._RuntimeDataHandlerBase__propagate_instances()

        # Assertions
        assert handler._RuntimeDataHandlerBase__route_instance.call_count == 3
        handler._RuntimeDataHandlerBase__route_instance.assert_any_call(
            instance1
        )
        handler._RuntimeDataHandlerBase__route_instance.assert_any_call(
            instance2
        )
        handler._RuntimeDataHandlerBase__route_instance.assert_any_call(
            instance3
        )

        # Ensure __anext__ was called until StopAsyncIteration
        assert mock_event_source_fixture.__anext__.call_count == 3

    @pytest.mark.asyncio
    async def test_route_instance_pushes_to_processor(
        self,
        handler: TestableRuntimeDataHandler,
        mock_processor_registry: MagicMock,
        mocker,
    ):
        """Tests that __route_instance gets a processor and pushes the instance."""
        mock_caller_id_val = create_caller_id_for_test(
            "test_caller_prop"
        )  # Helper for concrete ID
        mock_instance = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        mock_instance.caller_id = mock_caller_id_val

        # mock_processor_registry fixture already configures get_or_create_processor
        # to return a poller. We get that poller to check assertions on it.
        mock_target_poller = (
            mock_processor_registry.get_or_create_processor.return_value
        )

        await handler._RuntimeDataHandlerBase__route_instance(mock_instance)

        mock_processor_registry.get_or_create_processor.assert_called_once_with(
            mock_caller_id_val
        )
        # The SUT calls on_available on the processor (poller)
        mock_target_poller.on_available.assert_called_once_with(mock_instance)

    @pytest.mark.asyncio
    async def test_route_instance_handles_no_processor(
        self,
        handler: TestableRuntimeDataHandler,
        mock_processor_registry: MagicMock,
        mocker,
        capsys,
    ):
        """Tests __route_instance when processor_registry returns None (unexpected case)."""
        mock_caller_id_val = create_caller_id_for_test("test_caller_no_proc")
        mock_instance = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        mock_instance.caller_id = mock_caller_id_val

        mock_processor_registry.get_or_create_processor.return_value = (
            None  # Simulate no processor
        )

        await handler._RuntimeDataHandlerBase__route_instance(mock_instance)

        mock_processor_registry.get_or_create_processor.assert_called_once_with(
            mock_caller_id_val
        )

        # Check for the warning/error print
        captured = capsys.readouterr()
        assert (
            f"CRITICAL: No processor obtained for caller_id {mock_caller_id_val} from registry."
            in captured.out
        )


# Helper function for creating concrete CallerIdentifier for tests that need it
def create_caller_id_for_test(
    id_str: str,
) -> CallerIdentifier:  # id_str is label
    # For tests needing distinct CallerIdentifier instances where the exact UUID value
    # from the string is not critical for the test logic itself.
    return CallerIdentifier.random()


# Helper function for creating concrete CallerIdentifier for tests that need it
def create_caller_id_for_test(
    id_str: str,
) -> CallerIdentifier:  # id_str is label
    # For tests needing distinct CallerIdentifier instances where the exact UUID value
    # from the string is not critical for the test logic itself.
    return CallerIdentifier.random()


# Helper awaitable_aiter removed.


# Module-level fixtures
@pytest.fixture
def mock_data_reader_fixture(mocker) -> RemoteDataReader:
    reader = mocker.MagicMock(spec=RemoteDataReader)
    reader._on_data_ready = mocker.MagicMock(
        name="data_reader_on_data_ready_mock"
    )
    return reader


@pytest.fixture
def mock_event_source_fixture(mocker) -> AsyncPoller:
    poller = mocker.AsyncMock(spec=AsyncPoller)
    poller.__aiter__ = Mock(return_value=poller)
    poller.__anext__ = AsyncMock()
    return poller


@pytest.fixture
def mock_servicer_context(mocker) -> ServicerContext:  # Moved to module level
    context = mocker.AsyncMock(spec=ServicerContext)
    context.peer = mocker.MagicMock(return_value="ipv4:127.0.0.1:12345")
    return context


# Fixtures specific to ConcreteRuntimeDataHandler tests (can also be module level)
@pytest.fixture
def mock_concrete_id_tracker(mocker):
    return mocker.MagicMock(spec=IdTracker)


@pytest.fixture
def mock_concrete_processor_registry(
    mocker,
):  # Specific for ConcreteRuntimeDataHandler tests
    registry = mocker.MagicMock(spec=CallerProcessorRegistry)
    registry.get_or_create_processor = mocker.MagicMock(
        name="get_or_create_processor_concrete"
    )
    return registry


# Minimal concrete implementation for testing some parts, if TestableRuntimeDataHandler is too complex
# or if we want to test without extensive mocking of abstract methods.
class ConcreteRuntimeDataHandler(
    RuntimeDataHandlerBase[str, str]
):  # TDataType=str, TEventType=str
    __test__ = False  # Not a test itself

    def __init__(
        self, data_reader, event_source, id_tracker, processor_registry, mocker
    ):
        super().__init__(
            data_reader, event_source, id_tracker, processor_registry
        )
        self._register_caller_mock = mocker.AsyncMock()  # Changed to AsyncMock
        self._unregister_caller_mock = mocker.MagicMock(return_value=True)
        self._try_get_caller_id_mock = mocker.MagicMock()

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[str, str]:  # Match types
        return await self._register_caller_mock(
            caller_id, endpoint, port
        )  # Added await

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        return self._unregister_caller_mock(caller_id)

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self._try_get_caller_id_mock(endpoint, port)


@pytest.fixture  # This was the fixture that had issues finding other fixtures
def handler_fixture(
    mock_data_reader_fixture,
    mock_event_source_fixture,
    mock_concrete_id_tracker,
    mock_concrete_processor_registry,
    mocker,
):
    # Patch asyncio.create_task before ConcreteRuntimeDataHandler is instantiated
    with patch("asyncio.create_task") as mock_create_task:
        handler = ConcreteRuntimeDataHandler(
            mock_data_reader_fixture,  # Corrected: Use the actual fixture name
            mock_event_source_fixture,  # Corrected: Use the actual fixture name
            mock_concrete_id_tracker,
            mock_concrete_processor_registry,
            mocker,
        )
        handler.mock_create_task = mock_create_task
        yield handler


# Test class for the public register_caller method's argument parsing (still relevant)
class TestRuntimeDataHandlerBaseRegisterCaller:
    """Tests for the public register_caller method of RuntimeDataHandlerBase (via ConcreteRuntimeDataHandler)."""

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_with_endpoint_port_success(  # Changed to async
        self, handler_fixture: ConcreteRuntimeDataHandler, mocker
    ):
        caller_id = CallerIdentifier.random()
        endpoint = "127.0.0.1"
        port = 8080
        mock_processor_instance = mocker.MagicMock(spec=EndpointDataProcessor)
        handler_fixture._register_caller_mock.return_value = (
            mock_processor_instance
        )

        result = await handler_fixture.register_caller(  # Added await
            caller_id, endpoint=endpoint, port=port
        )
        handler_fixture._register_caller_mock.assert_called_once_with(
            caller_id, endpoint, port
        )
        assert result == mock_processor_instance

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_with_context_success(  # Changed to async
        self,
        handler_fixture: ConcreteRuntimeDataHandler,
        mock_servicer_context: ServicerContext,
        mocker,
    ):
        caller_id = CallerIdentifier.random()
        expected_ip = "192.168.0.1"
        expected_port = 1234
        # Ensure these patch calls are correct for the SUT module
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=expected_ip,
        )
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=expected_port,
        )

        mock_processor_instance = mocker.MagicMock(spec=EndpointDataProcessor)
        handler_fixture._register_caller_mock.return_value = (
            mock_processor_instance
        )

        result = await handler_fixture.register_caller(
            caller_id, context=mock_servicer_context
        )  # Added await

        # get_client_ip and get_client_port are patched, so their calls are on the patch objects
        # This test assumes they are correctly called by the SUT if inputs are right.
        handler_fixture._register_caller_mock.assert_called_once_with(
            caller_id, expected_ip, expected_port
        )
        assert result == mock_processor_instance

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_with_context_ip_none_returns_none(  # Changed to async
        self,
        handler_fixture: ConcreteRuntimeDataHandler,
        mock_servicer_context: ServicerContext,
        mocker,
    ):
        caller_id = CallerIdentifier.random()
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=None,
        )
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=1234,
        )

        result = await handler_fixture.register_caller(
            caller_id, context=mock_servicer_context
        )  # Added await

        handler_fixture._register_caller_mock.assert_not_called()
        assert result is None

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_with_context_port_none_raises_value_error(  # Changed to async
        self,
        handler_fixture: ConcreteRuntimeDataHandler,
        mock_servicer_context: ServicerContext,
        mocker,
    ):
        caller_id = CallerIdentifier.random()
        expected_ip = "192.168.0.1"
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=expected_ip,
        )
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=None,
        )

        with pytest.raises(ValueError) as excinfo:
            await handler_fixture.register_caller(
                caller_id, context=mock_servicer_context
            )  # Added await

        assert (
            f"Could not determine client port from context for endpoint {expected_ip}"
            in str(excinfo.value)
        )
        handler_fixture._register_caller_mock.assert_not_called()

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_mutex_args_endpoint_context(  # Changed to async
        self,
        handler_fixture: ConcreteRuntimeDataHandler,
        mock_servicer_context: ServicerContext,
    ):
        caller_id = CallerIdentifier.random()
        with pytest.raises(
            ValueError,
            match="Cannot specify context via both args and kwargs, or with endpoint/port.",
        ):
            await handler_fixture.register_caller(
                caller_id,
                endpoint="1.2.3.4",
                port=123,
                context=mock_servicer_context,
            )  # Added await

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_mutex_args_none(
        self, handler_fixture: ConcreteRuntimeDataHandler
    ):  # Changed to async
        caller_id = CallerIdentifier.random()
        with pytest.raises(
            ValueError, match="Exactly one of .* must be provided."
        ):
            await handler_fixture.register_caller(caller_id)  # Added await

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_endpoint_without_port(
        self, handler_fixture: ConcreteRuntimeDataHandler
    ):  # Changed to async
        caller_id = CallerIdentifier.random()
        with pytest.raises(
            ValueError,
            match="If 'endpoint' is provided, 'port' must also be provided",
        ):
            await handler_fixture.register_caller(
                caller_id, endpoint="1.2.3.4"
            )  # Added await

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_port_without_endpoint(
        self, handler_fixture: ConcreteRuntimeDataHandler
    ):  # Changed to async
        caller_id = CallerIdentifier.random()
        with pytest.raises(
            ValueError,
            match="If 'endpoint' is provided, 'port' must also be provided, and vice-versa.",
        ):
            await handler_fixture.register_caller(
                caller_id, port=1234
            )  # Added await

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_context_is_not_servicer_context_raises_type_error(  # Changed to async
        self, handler_fixture: ConcreteRuntimeDataHandler, mocker
    ):
        caller_id = CallerIdentifier.random()
        not_a_servicer_context = object()

        mocker.patch("tsercom.runtime.runtime_data_handler_base.get_client_ip")
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port"
        )

        with pytest.raises(
            TypeError,
            match="Expected context to be an instance of grpc.aio.ServicerContext",
        ):
            await handler_fixture.register_caller(
                caller_id, context=not_a_servicer_context
            )  # Added await

        handler_fixture._register_caller_mock.assert_not_called()
