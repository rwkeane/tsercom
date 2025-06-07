import asyncio
import pytest

# Import missing functions as well
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)
import grpc.aio  # For ServicerContext

from typing import (
    Optional,
    Any,
    TypeVar,
    List,  # Added for type hints in ConcreteTestProcessor
    AsyncIterator,  # Added for type hints in ConcreteTestProcessor
)
import datetime

# SUT
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase

# Dependencies to be mocked or used
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from google.protobuf.timestamp_pb2 import (
    Timestamp as GrpcTimestamp,
)
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.timesync.common.proto import ServerTimestamp

import grpc  # For grpc.StatusCode
from grpc.aio import ServicerContext


DataType = TypeVar("DataType")


# Helper class for mocking async iterators in tests
class MockAsyncIterator(
    AsyncIterator[List[Any]]
):  # Type with List[Any] for batches
    def __init__(
        self, items_to_yield: List[List[Any]]
    ):  # Takes a list of batches
        self._items = iter(items_to_yield)

    def __aiter__(self) -> "MockAsyncIterator":
        return self

    async def __anext__(self) -> List[Any]:  # Yields a batch (list)
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


# Test Subclass of RuntimeDataHandlerBase
class TestableRuntimeDataHandler(RuntimeDataHandlerBase[DataType, Any]):
    __test__ = False

    def __init__(
        self,
        data_reader: RemoteDataReader[DataType],
        event_source: AsyncPoller[Any],
        mocker,
    ):
        super().__init__(data_reader, event_source)
        self.mock_register_caller = mocker.AsyncMock()
        self.mock_unregister_caller = mocker.AsyncMock(return_value=True)
        self.mock_try_get_caller_id = mocker.MagicMock(
            name="_try_get_caller_id_impl"
        )
        self._on_data_ready = (
            mocker.AsyncMock(  # This mock is for the SUT's _on_data_ready
                name="handler_sut_on_data_ready_mock"
            )
        )

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> Optional[EndpointDataProcessor]:
        return await self.mock_register_caller(caller_id, endpoint, port)

    async def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        return await self.mock_unregister_caller(caller_id)

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> Optional[CallerIdentifier]:
        return self.mock_try_get_caller_id(endpoint, port)


class TestRuntimeDataHandlerBaseBehavior:

    @pytest.fixture(autouse=True)
    def manage_event_loop(self):
        loop = None
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("existing loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            initial_loop_was_set_by_this_fixture = False
            if not is_global_event_loop_set():  # Check before setting
                set_tsercom_event_loop(loop)
                initial_loop_was_set_by_this_fixture = True

            yield loop
        finally:
            if (
                initial_loop_was_set_by_this_fixture
            ):  # Only clear if this fixture set it
                clear_tsercom_event_loop()

            # Loop closing logic - be cautious with loops from pytest-asyncio
            if (
                loop
                and not getattr(loop, "_default_loop", False)
                and not loop.is_closed()
                and initial_loop_was_set_by_this_fixture  # Only manage loops this fixture created/set policy for
            ):
                if loop.is_running():  # pragma: no cover
                    loop.call_soon_threadsafe(loop.stop)
                # loop.close() # Avoid closing if it might be pytest-asyncio's

    @pytest.fixture
    def mock_data_reader(self, mocker):
        reader = mocker.MagicMock(spec=RemoteDataReader)
        reader._on_data_ready = mocker.MagicMock(
            name="data_reader_on_data_ready_on_reader_mock"
        )
        return reader

    @pytest.fixture
    def mock_event_source(self, mocker):
        # Use a basic AsyncMock. __aiter__ and __anext__ will be AsyncMocks by default.
        poller = mocker.AsyncMock()
        return poller

    @pytest.fixture
    def mock_caller_id(self, mocker):
        return mocker.MagicMock(
            spec=CallerIdentifier, name="MockCallerIdInstance"
        )

    @pytest.fixture
    def mock_servicer_context(self, mocker):
        context = mocker.AsyncMock(spec=ServicerContext)
        context.peer = mocker.MagicMock(return_value="ipv4:127.0.0.1:12345")
        return context

    @pytest.fixture
    def handler(self, mock_data_reader, mock_event_source, mocker):
        # Patch run_on_event_loop called by RuntimeDataHandlerBase.__init__
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.run_on_event_loop"
        )
        return TestableRuntimeDataHandler(
            mock_data_reader, mock_event_source, mocker
        )

    @pytest.fixture
    def mock_sync_clock(self, mocker):
        clock = mocker.MagicMock(spec=SynchronizedClock)
        clock.desync = mocker.MagicMock(name="sync_clock_desync")
        return clock

    @pytest.fixture
    def test_caller_id_instance(
        self,
    ):
        return CallerIdentifier.random()

    @pytest.fixture
    def data_processor(
        self,
        handler,  # Uses the handler fixture from TestRuntimeDataHandlerBaseBehavior
        test_caller_id_instance,
        mock_sync_clock,
        mocker,
    ):
        mock_poller_for_dp = mocker.MagicMock(spec=AsyncPoller)
        # Mock the get method on the actual IdTracker instance used by the handler
        handler._RuntimeDataHandlerBase__id_tracker.get = mocker.MagicMock(  # type: ignore
            return_value=(
                "dummy_ip",
                1234,
                mock_poller_for_dp,
            )
        )
        return handler._create_data_processor(
            test_caller_id_instance, mock_sync_clock
        )

    # --- Tests for RuntimeDataHandlerBase direct methods ---
    def test_constructor(self, handler, mock_data_reader, mock_event_source):
        assert handler._RuntimeDataHandlerBase__data_reader is mock_data_reader  # type: ignore
        assert handler._RuntimeDataHandlerBase__event_source is mock_event_source  # type: ignore

    @pytest.mark.asyncio
    async def test_async_iteration_with_event_source(
        self, handler, mock_event_source, mocker
    ):
        item1 = mocker.MagicMock(name="EventItem1")
        item2 = mocker.MagicMock(name="EventItem2")
        mock_event_source.__anext__.side_effect = [
            [item1],  # Batch 1
            [item2],  # Batch 2
            StopAsyncIteration,
        ]
        collected_items = []
        async for item_batch in handler:
            collected_items.extend(item_batch)  # Assuming items are in a list
        assert collected_items == [item1, item2]
        # __anext__ is called for each item + one more for StopAsyncIteration
        assert mock_event_source.__anext__.call_count == 3
        # The handler's __aiter__ is called, which then uses the event_source.
        # Direct check on mock_event_source.__aiter__ might be tricky if handler is the primary iterator.
        # The important part is that __anext__ on the source was called as expected.
        # If handler.__aiter__ simply returns self.__event_source, then __aiter__ on event_source would be called.
        # RuntimeDataHandlerBase.__aiter__ returns self.
        # RuntimeDataHandlerBase.__anext__ calls self.__event_source.__anext__().
        # So, mock_event_source.__aiter__ is NOT called by `async for item_batch in handler`.
        # The assertion mock_event_source.__aiter__.assert_called_once() is incorrect.
        # We are asserting the behavior of handler as an async iterator.

    def test_check_for_caller_id(self, handler, mock_caller_id):
        endpoint_str = "test_ep"
        port_num = 1122
        handler.mock_try_get_caller_id.return_value = mock_caller_id
        result = handler.check_for_caller_id(endpoint_str, port_num)
        handler.mock_try_get_caller_id.assert_called_once_with(
            endpoint_str, port_num
        )
        assert result is mock_caller_id

    @pytest.mark.asyncio
    async def test_handler_on_data_ready_calls_reader_on_data_ready(
        self, handler, mock_data_reader, mocker
    ):
        mock_annotated_instance = mocker.MagicMock(spec=AnnotatedInstance)
        handler._RuntimeDataHandlerBase__data_reader = mock_data_reader
        await RuntimeDataHandlerBase._on_data_ready(  # type: ignore
            handler, mock_annotated_instance
        )
        mock_data_reader._on_data_ready.assert_called_once_with(
            mock_annotated_instance
        )

    # --- Tests for _RuntimeDataHandlerBase__DataProcessorImpl (inner class) ---
    # These tests use the data_processor fixture

    @pytest.mark.asyncio
    async def test_processor_desynchronize(
        self, data_processor, mock_sync_clock, mocker
    ):
        mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
        # Ensure the mock_server_ts.timestamp has ToDatetime if it's accessed
        mock_grpc_ts = mocker.MagicMock(spec=GrpcTimestamp)
        mock_grpc_ts.ToDatetime.return_value = datetime.datetime.now(
            datetime.timezone.utc
        )
        mock_server_ts.timestamp = mock_grpc_ts

        expected_datetime = datetime.datetime.now(datetime.timezone.utc)
        mock_sync_clock.desync.return_value = expected_datetime

        result_dt = await data_processor.desynchronize(
            mock_server_ts, context=None
        )

        mock_sync_clock.desync.assert_called_once()
        args, _ = mock_sync_clock.desync.call_args
        assert isinstance(args[0], SynchronizedTimestamp)
        assert (
            args[0].timestamp
            == mock_server_ts.timestamp.ToDatetime.return_value
        )
        assert result_dt is expected_datetime

    @pytest.mark.asyncio
    async def test_processor_desynchronize_invalid_ts_with_context_aborts(
        self, data_processor, mocker
    ):
        mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
        mock_context = mocker.AsyncMock(spec=ServicerContext)

        # Patch SynchronizedTimestamp.try_parse to return None
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.SynchronizedTimestamp.try_parse",
            return_value=None,
        )

        result = await data_processor.desynchronize(
            mock_server_ts, context=mock_context
        )

        mock_context.abort.assert_awaited_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid timestamp provided"
        )
        assert result is None
        # Ensure desync on the clock was not called
        # The data_processor fixture gets mock_sync_clock implicitly
        # We need to access it via the handler that created the data_processor,
        # or ensure the fixture setup makes it available if we need to assert on it.
        # For this test, the primary check is abort and return None.
        # If try_parse returns None, desync shouldn't be reached.

    @pytest.mark.asyncio
    async def test_processor_desynchronize_invalid_ts_no_context_returns_none(
        self, data_processor, mocker
    ):
        mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)

        # Patch SynchronizedTimestamp.try_parse to return None
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.SynchronizedTimestamp.try_parse",
            return_value=None,
        )

        result = await data_processor.desynchronize(
            mock_server_ts, context=None
        )
        assert result is None
        # Similar to above, asserting abort was NOT called is tricky if no mock_context
        # was created and passed. The main check is that it returns None and doesn't error.

    @pytest.mark.asyncio
    async def test_processor_deregister_caller(
        self, data_processor, handler, test_caller_id_instance
    ):
        await data_processor.deregister_caller()
        handler.mock_unregister_caller.assert_called_once_with(
            test_caller_id_instance
        )

    @pytest.mark.asyncio
    async def test_processor_process_data_with_datetime(
        self, data_processor, handler, test_caller_id_instance
    ):
        test_payload = "test_payload_data"
        test_dt = datetime.datetime.now(datetime.timezone.utc)
        await data_processor.process_data(test_payload, test_dt)
        handler._on_data_ready.assert_called_once()
        args, _ = handler._on_data_ready.call_args
        annotated_instance: AnnotatedInstance = args[0]
        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == test_dt

    @pytest.mark.asyncio
    async def test_processor_process_data_with_server_timestamp(
        self,
        data_processor,
        handler,
        test_caller_id_instance,
        mock_sync_clock,
        mocker,
    ):
        test_payload = "payload_with_server_ts"
        mock_server_ts = mocker.MagicMock(spec=ServerTimestamp)
        mock_grpc_ts = mocker.MagicMock(spec=GrpcTimestamp)
        mock_grpc_ts.ToDatetime.return_value = datetime.datetime.now(
            datetime.timezone.utc
        )
        mock_server_ts.timestamp = mock_grpc_ts

        expected_desynced_dt = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=5)
        mock_sync_clock.desync.return_value = expected_desynced_dt
        await data_processor.process_data(test_payload, mock_server_ts)
        mock_sync_clock.desync.assert_called_once()
        args, _ = mock_sync_clock.desync.call_args
        assert isinstance(args[0], SynchronizedTimestamp)
        assert (
            args[0].timestamp
            == mock_server_ts.timestamp.ToDatetime.return_value
        )
        handler._on_data_ready.assert_called_once()
        args, _ = handler._on_data_ready.call_args
        annotated_instance: AnnotatedInstance = args[0]
        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == expected_desynced_dt

    @pytest.mark.asyncio
    async def test_processor_process_data_no_timestamp(
        self,
        data_processor,
        handler,
        test_caller_id_instance,
        mocker,
    ):
        test_payload = "payload_no_ts"
        fixed_now = datetime.datetime.now(datetime.timezone.utc)
        mock_datetime_class = mocker.patch(
            "tsercom.runtime.endpoint_data_processor.datetime"
        )
        mock_datetime_class.now.return_value = fixed_now
        # Ensure timezone.utc is correctly patche/available on the mocked datetime
        mock_datetime_class.timezone.utc = datetime.timezone.utc

        await data_processor.process_data(test_payload, timestamp=None)
        mock_datetime_class.now.assert_called_once_with(datetime.timezone.utc)
        handler._on_data_ready.assert_called_once()
        args, _ = handler._on_data_ready.call_args
        annotated_instance: AnnotatedInstance = args[0]
        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == fixed_now

    # --- Tests for __dispatch_poller_data_loop ---
    @pytest.mark.asyncio
    async def test_dispatch_loop_event_for_known_caller(self, handler, mocker):
        test_caller_id = CallerIdentifier.random()
        mock_event_item = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        mock_event_item.caller_id = test_caller_id

        # Use the custom MockAsyncIterator
        mock_event_batches = [[mock_event_item]]
        handler._RuntimeDataHandlerBase__event_source = MockAsyncIterator(mock_event_batches)  # type: ignore

        mock_per_caller_poller = mocker.MagicMock(spec=AsyncPoller)
        mock_per_caller_poller.on_available = mocker.MagicMock()

        handler._RuntimeDataHandlerBase__id_tracker.try_get = mocker.MagicMock(  # type: ignore
            return_value=("ip", 123, mock_per_caller_poller)
        )

        await handler._RuntimeDataHandlerBase__dispatch_poller_data_loop()  # type: ignore

        handler._RuntimeDataHandlerBase__id_tracker.try_get.assert_called_once_with(test_caller_id)  # type: ignore
        mock_per_caller_poller.on_available.assert_called_once_with(
            mock_event_item
        )

    @pytest.mark.asyncio
    async def test_dispatch_loop_event_for_unknown_caller(
        self, handler, mocker
    ):
        mock_event_item = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        mock_event_item.caller_id = CallerIdentifier.random()

        mock_event_batches = [[mock_event_item]]
        handler._RuntimeDataHandlerBase__event_source = MockAsyncIterator(mock_event_batches)  # type: ignore

        handler._RuntimeDataHandlerBase__id_tracker.try_get = mocker.MagicMock(  # type: ignore
            return_value=None
        )
        await handler._RuntimeDataHandlerBase__dispatch_poller_data_loop()  # type: ignore
        handler._RuntimeDataHandlerBase__id_tracker.try_get.assert_called_once_with(mock_event_item.caller_id)  # type: ignore

    @pytest.mark.asyncio
    async def test_dispatch_loop_event_caller_found_poller_none(
        self, handler, mocker
    ):
        mock_event_item = mocker.MagicMock(spec=SerializableAnnotatedInstance)
        mock_event_item.caller_id = CallerIdentifier.random()

        mock_event_batches = [[mock_event_item]]
        handler._RuntimeDataHandlerBase__event_source = MockAsyncIterator(mock_event_batches)  # type: ignore

        handler._RuntimeDataHandlerBase__id_tracker.try_get = mocker.MagicMock(  # type: ignore
            return_value=("ip", 123, None)
        )
        await handler._RuntimeDataHandlerBase__dispatch_poller_data_loop()  # type: ignore
        handler._RuntimeDataHandlerBase__id_tracker.try_get.assert_called_once_with(mock_event_item.caller_id)  # type: ignore

    # --- Tests for _create_data_processor error conditions ---
    def test_create_data_processor_id_not_in_tracker(self, handler, mocker):
        test_caller_id = CallerIdentifier.random()
        mock_clock = mocker.MagicMock(spec=SynchronizedClock)

        handler._RuntimeDataHandlerBase__id_tracker.get = mocker.MagicMock(  # type: ignore
            side_effect=KeyError("ID not found")
        )
        with pytest.raises(KeyError, match="ID not found"):
            handler._create_data_processor(test_caller_id, mock_clock)
        handler._RuntimeDataHandlerBase__id_tracker.get.assert_called_once_with(test_caller_id)  # type: ignore

    def test_create_data_processor_poller_is_none_in_tracker(
        self, handler, mocker
    ):
        test_caller_id = CallerIdentifier.random()
        mock_clock = mocker.MagicMock(spec=SynchronizedClock)

        handler._RuntimeDataHandlerBase__id_tracker.get = mocker.MagicMock(  # type: ignore
            return_value=("ip", 123, None)
        )
        with pytest.raises(
            ValueError,
            match="No data poller found in IdTracker for {}".format(
                test_caller_id
            ),
        ):
            handler._create_data_processor(test_caller_id, mock_clock)
        handler._RuntimeDataHandlerBase__id_tracker.get.assert_called_once_with(test_caller_id)  # type: ignore


# Minimal concrete implementation for testing register_caller argument parsing
class ConcreteRuntimeDataHandler(RuntimeDataHandlerBase[str, str]):
    __test__ = False  # Not a test class itself

    def __init__(self, data_reader, event_source, mocker):
        super().__init__(data_reader, event_source)
        self._register_caller_mock = mocker.AsyncMock(
            spec=self._register_caller
        )
        self._unregister_caller_mock = mocker.AsyncMock(
            spec=self._unregister_caller
        )
        self._try_get_caller_id_mock = mocker.MagicMock(
            spec=self._try_get_caller_id
        )

    async def _register_caller(  # Make it async
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> Optional[EndpointDataProcessor]:
        return await self._register_caller_mock(
            caller_id, endpoint, port
        )  # Added await

    async def _unregister_caller(
        self, caller_id: CallerIdentifier
    ) -> bool:  # Make it async
        return await self._unregister_caller_mock(caller_id)  # Added await

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self._try_get_caller_id_mock(endpoint, port)


@pytest.fixture
def mock_data_reader_fixture(mocker):
    return mocker.MagicMock(spec=RemoteDataReader[AnnotatedInstance[str]])


@pytest.fixture
def mock_event_source_fixture(mocker):
    # Use a basic AsyncMock. __aiter__ and __anext__ will be AsyncMocks by default.
    # Tests using this fixture for iteration need to configure __anext__.side_effect.
    mock_poller = mocker.AsyncMock()
    return mock_poller  # Corrected indentation


@pytest.fixture
def mock_context_fixture(mocker):
    return mocker.MagicMock(spec=grpc.aio.ServicerContext)


@pytest.fixture
def handler_fixture(
    mock_data_reader_fixture, mock_event_source_fixture, mocker
):
    # Patch run_on_event_loop called by RuntimeDataHandlerBase.__init__
    mocker.patch("tsercom.runtime.runtime_data_handler_base.run_on_event_loop")
    return ConcreteRuntimeDataHandler(
        mock_data_reader_fixture, mock_event_source_fixture, mocker
    )


@pytest.fixture
def mock_endpoint_processor_fixture(mocker):
    return mocker.MagicMock(spec=EndpointDataProcessor)


class TestRuntimeDataHandlerBaseRegisterCaller:
    """Tests for the register_caller method of RuntimeDataHandlerBase."""

    @pytest.fixture(autouse=True)
    def manage_event_loop(
        self, request
    ):  # Add request to check for asyncio marker
        from tsercom.threading.aio.global_event_loop import (
            is_global_event_loop_set,
            set_tsercom_event_loop,
            clear_tsercom_event_loop,
        )  # Import here

        if request.node.get_closest_marker("asyncio"):
            # If it's an asyncio test, conftest.manage_tsercom_loop should handle it.
            # This fixture instance should then do nothing further with set/clear for tsercom global loop.
            # It might still provide 'loop' to the test if the test requests it directly,
            # which would be the loop from pytest-asyncio.
            try:
                loop = asyncio.get_running_loop()  # Loop from pytest-asyncio
                yield loop
                return
            except RuntimeError:  # Should not happen in an async test
                pass  # Fall through to sync test logic if really no loop

        # For non-asyncio tests in this class, or if above failed (should not for async)
        loop = None
        initial_loop_was_set_by_this_fixture = False
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if not is_global_event_loop_set():
                set_tsercom_event_loop(loop)
                initial_loop_was_set_by_this_fixture = True
            yield loop
        finally:
            if initial_loop_was_set_by_this_fixture:
                clear_tsercom_event_loop()

            if (
                loop
                and not getattr(loop, "_default_loop", False)
                and not loop.is_closed()
                and initial_loop_was_set_by_this_fixture
            ):
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
                # loop.close()

    @pytest.mark.asyncio
    async def test_register_caller_with_endpoint_port_success(
        self, handler_fixture, mock_endpoint_processor_fixture
    ):
        caller_id = CallerIdentifier.random()
        endpoint = "127.0.0.1"
        port = 8080
        handler_fixture._register_caller_mock.return_value = (
            mock_endpoint_processor_fixture
        )

        result = await handler_fixture.register_caller(
            caller_id, endpoint=endpoint, port=port
        )

        handler_fixture._register_caller_mock.assert_awaited_once_with(  # Changed to assert_awaited_once_with
            caller_id, endpoint, port
        )
        assert result == mock_endpoint_processor_fixture

    @pytest.mark.asyncio
    async def test_register_caller_with_context_success(
        self,
        handler_fixture,
        mock_context_fixture,
        mock_endpoint_processor_fixture,
        mocker,
    ):
        caller_id = CallerIdentifier.random()
        expected_ip = "192.168.0.1"
        expected_port = 1234
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=expected_ip,
        )
        mock_get_port = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=expected_port,
        )
        handler_fixture._register_caller_mock.return_value = (
            mock_endpoint_processor_fixture
        )

        result = await handler_fixture.register_caller(
            caller_id, context=mock_context_fixture
        )

        mock_get_ip.assert_called_once_with(mock_context_fixture)
        mock_get_port.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_awaited_once_with(  # Changed to assert_awaited_once_with
            caller_id, expected_ip, expected_port
        )
        assert result == mock_endpoint_processor_fixture

    @pytest.mark.asyncio
    async def test_register_caller_with_context_ip_none_returns_none(
        self,
        handler_fixture,
        mock_context_fixture,
        mocker,
    ):
        caller_id = CallerIdentifier.random()
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=None,
        )
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=1234,
        )

        result = await handler_fixture.register_caller(
            caller_id, context=mock_context_fixture
        )

        mock_get_ip.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_register_caller_with_context_port_none_raises_value_error(
        self,
        handler_fixture,
        mock_context_fixture,
        mocker,
    ):
        caller_id = CallerIdentifier.random()
        expected_ip = "192.168.0.1"
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=expected_ip,
        )
        mock_get_port = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=None,
        )

        with pytest.raises(ValueError) as excinfo:
            await handler_fixture.register_caller(
                caller_id, context=mock_context_fixture
            )
        expected_msg_part = (
            "Could not get client port from context for endpoint: "
            + expected_ip
        )
        assert (expected_msg_part + ".") == str(excinfo.value)
        mock_get_ip.assert_called_once_with(mock_context_fixture)
        mock_get_port.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_caller_mutex_args_endpoint_context(
        self, handler_fixture, mock_context_fixture
    ):
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            await handler_fixture.register_caller(
                caller_id,
                endpoint="1.2.3.4",
                port=123,
                context=mock_context_fixture,
            )
        assert (
            "Cannot use context via args & kwargs, or with endpoint/port."
            in str(excinfo.value)
        )

    @pytest.mark.asyncio
    async def test_register_caller_mutex_args_none(self, handler_fixture):
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            await handler_fixture.register_caller(caller_id)
        assert (
            "Provide (endpoint and port) or context, but not both or neither."
            in str(excinfo.value)
        )

    @pytest.mark.asyncio
    async def test_register_caller_endpoint_without_port(
        self, handler_fixture
    ):
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            await handler_fixture.register_caller(
                caller_id, endpoint="1.2.3.4"
            )
        assert (
            "If 'endpoint' is provided, 'port' must also be, and vice-versa."
            in str(excinfo.value)
        )

    @pytest.mark.asyncio
    async def test_register_caller_port_without_endpoint(
        self, handler_fixture
    ):
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            await handler_fixture.register_caller(caller_id, port=1234)
        assert (
            "If 'endpoint' is provided, 'port' must also be, and vice-versa."
            in str(excinfo.value)
        )

    @pytest.mark.asyncio
    async def test_register_caller_context_is_not_servicer_context_raises_type_error(
        self, handler_fixture, mocker
    ):
        caller_id = CallerIdentifier.random()
        not_a_servicer_context = object()
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value="1.2.3.4",
        )
        mock_get_port = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=1234,
        )
        with pytest.raises(TypeError) as excinfo:
            await handler_fixture.register_caller(
                caller_id, context=not_a_servicer_context
            )
        assert (
            "Expected context: grpc.aio.ServicerContext, got object."
            in str(excinfo.value)
        )
        mock_get_ip.assert_not_called()
        mock_get_port.assert_not_called()
        handler_fixture._register_caller_mock.assert_not_called()
