import asyncio
import pytest
import grpc.aio  # For ServicerContext

from unittest.mock import (
    MagicMock,
    AsyncMock,
)  # Added for direct use in setup_method
from typing import (
    Optional,
    Any,
    TypeVar,
)  # Removed Tuple, List as they are not used here
import datetime

# SUT
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase

# Dependencies to be mocked or used
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance

# ServerTimestamp removed, SerializableAnnotatedInstance will be used for spec if appropriate, or no spec.
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)  # For type hints
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.timesync.common.proto import ServerTimestamp  # Added import

# For gRPC context testing
import grpc  # For grpc.StatusCode, if testable - not directly used in DataProcessorImpl tests
from grpc.aio import ServicerContext

# Modules for patching
import tsercom.rpc.grpc_util.addressing as grpc_addressing_module

# Type variable for data
DataType = TypeVar("DataType")


# Test Subclass of RuntimeDataHandlerBase
class TestableRuntimeDataHandler(
    RuntimeDataHandlerBase[DataType, Any]
):  # Added Any for TEventType
    __test__ = False  # Mark this class as not a test class for pytest

    def __init__(
        self,
        data_reader: RemoteDataReader[
            DataType
        ],  # Type hints kept as they are in current file
        event_source: AsyncPoller[Any],  # Type hints kept
        mocker,
    ):
        super().__init__(data_reader, event_source)
        # Mocks as per prompt's specification
        self.mock_register_caller = mocker.AsyncMock()
        self.mock_unregister_caller = mocker.AsyncMock(
            return_value=True
        )  # Ensure it returns bool
        self.mock_wait_for_data = mocker.MagicMock(
            return_value=mocker.AsyncMock()
        )  # Using AsyncMock for async_stub
        self.mock_get_data = mocker.MagicMock(
            return_value=mocker.AsyncMock()
        )  # Using AsyncMock for async_stub
        self.mock_handle_data = mocker.MagicMock(
            return_value=mocker.AsyncMock()
        )  # Using AsyncMock for async_stub

        # Preserving other original mocks from this class's __init__
        self.mock_try_get_caller_id = mocker.MagicMock(
            name="_try_get_caller_id_impl"
        )
        self._on_data_ready = mocker.AsyncMock(name="handler_on_data_ready_mock")  # type: ignore, Changed to AsyncMock

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> Optional[EndpointDataProcessor]:  # Corrected reference
        # print(f"TestableRuntimeDataHandler._register_caller called with: {caller_id}, {endpoint}, {port}") # Reduced verbosity
        return await self.mock_register_caller(caller_id, endpoint, port)

    async def _unregister_caller(
        self, caller_id: CallerIdentifier
    ) -> bool:  # Signature changed to bool
        # print(f"TestableRuntimeDataHandler._unregister_caller called with: {caller_id}") # Reduced verbosity
        return await self.mock_unregister_caller(caller_id)  # Added return

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> Optional[CallerIdentifier]:
        # print(f"TestableRuntimeDataHandler._try_get_caller_id called with: {endpoint}, {port}") # Reduced verbosity
        return self.mock_try_get_caller_id(endpoint, port)


class TestRuntimeDataHandlerBaseBehavior:  # Removed @pytest.mark.asyncio from class

    @pytest.fixture
    def mock_data_reader(self, mocker):
        reader = mocker.MagicMock(spec=RemoteDataReader)
        # The SUT's _on_data_ready calls self.__data_reader._on_data_ready.
        # So, we mock this method on the reader instance passed to the SUT.
        reader._on_data_ready = mocker.MagicMock(
            name="data_reader_on_data_ready_on_reader_mock"
        )
        return reader

    @pytest.fixture
    def mock_event_source(self, mocker):
        poller = mocker.MagicMock(spec=AsyncPoller)
        poller.__anext__ = mocker.AsyncMock(name="event_source_anext")
        return poller

    @pytest.fixture
    def mock_caller_id(
        self, mocker
    ):  # Renamed to avoid conflict with test_caller_id_fixture
        return mocker.MagicMock(
            spec=CallerIdentifier, name="MockCallerIdInstance"
        )

    @pytest.fixture
    def mock_servicer_context(self, mocker):
        context = mocker.AsyncMock(spec=ServicerContext)
        context.peer = mocker.MagicMock(return_value="ipv4:127.0.0.1:12345")
        return context

    @pytest.fixture
    def patch_get_client_ip(self, mocker):
        # Patch where it's used in the SUT
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            autospec=True,
        )
        yield mock_get_ip

    @pytest.fixture
    def patch_get_client_port(self, mocker):
        # Patch where it's used in the SUT
        mock_get_port = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            autospec=True,
        )
        yield mock_get_port

    @pytest.fixture
    def handler(self, mock_data_reader, mock_event_source, mocker):
        return TestableRuntimeDataHandler(
            mock_data_reader, mock_event_source, mocker
        )

    # Fixtures for DataProcessorImpl tests
    @pytest.fixture
    def mock_sync_clock(self, mocker):
        clock = mocker.MagicMock(spec=SynchronizedClock)
        clock.desync = mocker.AsyncMock(
            name="sync_clock_desync"
        )  # desync is async
        return clock

    @pytest.fixture
    def test_caller_id_instance(
        self,
    ):  # A concrete CallerIdentifier for processor tests
        return (
            CallerIdentifier.random()
        )  # Assuming CallerIdentifier has a factory or suitable constructor

    @pytest.fixture
    def data_processor(
        self, handler, test_caller_id_instance, mock_sync_clock
    ):
        # Create an instance of the inner class for testing
        # The _create_data_processor method is part of RuntimeDataHandlerBase
        return handler._create_data_processor(
            test_caller_id_instance, mock_sync_clock
        )

    # --- Tests for RuntimeDataHandlerBase direct methods ---
    def test_constructor(
        self, handler, mock_data_reader, mock_event_source
    ):  # No asyncio mark
        assert handler._RuntimeDataHandlerBase__data_reader is mock_data_reader  # type: ignore
        assert handler._RuntimeDataHandlerBase__event_source is mock_event_source  # type: ignore

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_endpoint_port(
        self, handler, mock_caller_id, mocker  # Added mocker
    ):
        endpoint_str = "10.0.0.1"
        port_num = 9999
        mock_processor_instance = mocker.MagicMock(spec=EndpointDataProcessor)
        handler.mock_register_caller.return_value = mock_processor_instance
        returned_processor = await handler.register_caller(
            mock_caller_id, endpoint_str, port_num
        )
        handler.mock_register_caller.assert_called_once_with(
            mock_caller_id, endpoint_str, port_num
        )
        assert returned_processor is mock_processor_instance

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_grpc_context_success(
        self,
        handler,
        mock_caller_id,
        mock_servicer_context,
        patch_get_client_ip,
        patch_get_client_port,
        mocker,  # Added mocker
    ):
        expected_ip = "1.2.3.4"
        expected_port = 5678
        patch_get_client_ip.return_value = expected_ip
        patch_get_client_port.return_value = expected_port
        mock_processor_instance = mocker.MagicMock(spec=EndpointDataProcessor)
        handler.mock_register_caller.return_value = mock_processor_instance
        returned_processor = await handler.register_caller(
            mock_caller_id,
            context=mock_servicer_context,  # Use keyword argument
        )
        patch_get_client_ip.assert_called_once_with(mock_servicer_context)
        patch_get_client_port.assert_called_once_with(mock_servicer_context)
        handler.mock_register_caller.assert_called_once_with(
            mock_caller_id, expected_ip, expected_port
        )
        assert returned_processor is mock_processor_instance

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_register_caller_grpc_context_no_ip(
        self,
        handler,
        mock_caller_id,
        mock_servicer_context,
        patch_get_client_ip,
        patch_get_client_port,  # patch_get_client_port is needed here
    ):
        patch_get_client_ip.return_value = None
        patch_get_client_port.return_value = (
            5678  # Does not matter if IP is None
        )
        returned_processor = handler.register_caller(  # Removed await
            mock_caller_id, context=mock_servicer_context
        )
        patch_get_client_ip.assert_called_once_with(mock_servicer_context)
        # The SUT currently calls get_client_port even if get_client_ip returns None.
        # The important check is that register_caller still returns None.
        patch_get_client_port.assert_called_once_with(mock_servicer_context)
        handler.mock_register_caller.assert_not_called()
        assert returned_processor is None

    def test_get_data_iterator_returns_self(self, handler):  # No asyncio mark
        assert handler.get_data_iterator() is handler

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_async_iteration_with_event_source(
        self, handler, mock_event_source, mocker
    ):
        item1 = mocker.MagicMock(name="EventItem1")
        item2 = mocker.MagicMock(name="EventItem2")
        mock_event_source.__anext__.side_effect = [
            item1,
            item2,
            StopAsyncIteration,
        ]
        collected_items = []
        async for item in handler.get_data_iterator():
            collected_items.append(item)
        assert collected_items == [item1, item2]
        assert mock_event_source.__anext__.call_count == 3

    def test_check_for_caller_id(
        self, handler, mock_caller_id
    ):  # No asyncio mark
        endpoint_str = "test_ep"
        port_num = 1122
        handler.mock_try_get_caller_id.return_value = mock_caller_id
        result = handler.check_for_caller_id(endpoint_str, port_num)
        handler.mock_try_get_caller_id.assert_called_once_with(
            endpoint_str, port_num
        )
        assert result is mock_caller_id

    # Test for the SUT's _on_data_ready, not the reader's.
    # The SUT's _on_data_ready calls the reader's _on_data_ready.
    @pytest.mark.asyncio  # Added asyncio mark
    async def test_handler_on_data_ready_calls_reader_on_data_ready(  # Made async
        self, handler, mock_data_reader, mocker
    ):
        print(
            "\n--- Test: test_handler_on_data_ready_calls_reader_on_data_ready ---"
        )
        mock_annotated_instance = mocker.MagicMock(spec=AnnotatedInstance)

        # Ensure the handler instance uses the mock_data_reader
        handler._RuntimeDataHandlerBase__data_reader = mock_data_reader  # type: ignore

        # Call the actual _on_data_ready method from RuntimeDataHandlerBase
        # using the 'handler' instance. This bypasses TestableRuntimeDataHandler's
        # own mock of _on_data_ready.
        await RuntimeDataHandlerBase._on_data_ready(
            handler, mock_annotated_instance
        )

        mock_data_reader._on_data_ready.assert_called_once_with(
            mock_annotated_instance
        )
        print(
            "--- Test: test_handler_on_data_ready_calls_reader_on_data_ready finished ---"
        )

    # --- Tests for _RuntimeDataHandlerBase__DataProcessorImpl ---

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_processor_desynchronize(
        self, data_processor, mock_sync_clock, mocker
    ):
        print("\n--- Test: test_processor_desynchronize ---")
        mock_server_ts = mocker.MagicMock(
            spec=SerializableAnnotatedInstance
        )  # Changed spec
        expected_datetime = datetime.datetime.now(datetime.timezone.utc)
        mock_sync_clock.desync.return_value = expected_datetime

        result_dt = await data_processor.desynchronize(mock_server_ts)

        mock_sync_clock.desync.assert_called_once_with(mock_server_ts)
        assert result_dt is expected_datetime
        print("--- Test: test_processor_desynchronize finished ---")

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_processor_deregister_caller(
        self, data_processor, handler, test_caller_id_instance
    ):
        print("\n--- Test: test_processor_deregister_caller ---")
        await data_processor.deregister_caller()
        handler.mock_unregister_caller.assert_called_once_with(
            test_caller_id_instance
        )
        print("--- Test: test_processor_deregister_caller finished ---")

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_processor_process_data_with_datetime(
        self, data_processor, handler, test_caller_id_instance
    ):
        print("\n--- Test: test_processor_process_data_with_datetime ---")
        test_payload = "test_payload_data"
        test_dt = datetime.datetime.now(datetime.timezone.utc)

        await data_processor.process_data(test_payload, test_dt)

        handler._on_data_ready.assert_called_once()
        # Check the AnnotatedInstance passed to handler._on_data_ready
        args, _ = handler._on_data_ready.call_args
        annotated_instance: AnnotatedInstance = args[0]

        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == test_dt
        print(
            "--- Test: test_processor_process_data_with_datetime finished ---"
        )

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_processor_process_data_with_server_timestamp(
        self,
        data_processor,
        handler,
        test_caller_id_instance,
        mock_sync_clock,
        mocker,  # Added mocker
    ):
        print(
            "\n--- Test: test_processor_process_data_with_server_timestamp ---"
        )
        test_payload = "payload_with_server_ts"
        mock_server_ts = mocker.MagicMock(
            spec=ServerTimestamp  # Changed spec to ServerTimestamp
        )
        expected_desynced_dt = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=5)

        # Mock the desynchronize method of the specific processor instance
        # Or, rely on mock_sync_clock.desync which is used by processor.desynchronize
        mock_sync_clock.desync.return_value = expected_desynced_dt

        await data_processor.process_data(test_payload, mock_server_ts)

        mock_sync_clock.desync.assert_called_once_with(mock_server_ts)
        handler._on_data_ready.assert_called_once()
        args, _ = handler._on_data_ready.call_args
        annotated_instance: AnnotatedInstance = args[0]

        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == expected_desynced_dt
        print(
            "--- Test: test_processor_process_data_with_server_timestamp finished ---"
        )

    @pytest.mark.asyncio  # Added asyncio mark
    async def test_processor_process_data_no_timestamp(
        self,
        data_processor,
        handler,
        test_caller_id_instance,
        mocker,  # Added mocker
    ):
        print("\n--- Test: test_processor_process_data_no_timestamp ---")
        test_payload = "payload_no_ts"

        # Patch datetime.datetime class in the module where .now() is called
        # (assumed to be endpoint_data_processor.py) to control the timestamp for "now".
        fixed_now = datetime.datetime.now(datetime.timezone.utc)

        # Mock the datetime.datetime class itself.
        # This assumes 'from datetime import datetime, timezone' is in endpoint_data_processor.py
        # or 'import datetime' and then 'datetime.datetime.now()' is used.
        # If 'from datetime.datetime import now' is used, the target would be different.
        mock_datetime_class = mocker.patch(
            "tsercom.runtime.endpoint_data_processor.datetime"  # Target the datetime class
        )
        # Configure the .now() method on the mocked class instance to return fixed_now.
        # When the SUT calls datetime.now(timezone.utc), it will use this mock.
        mock_datetime_class.now.return_value = fixed_now

        await data_processor.process_data(test_payload, timestamp=None)

        mock_datetime_class.now.assert_called_once_with(datetime.timezone.utc)

        handler._on_data_ready.assert_called_once()
        args, _ = handler._on_data_ready.call_args
        annotated_instance: AnnotatedInstance = args[0]

        assert isinstance(annotated_instance, AnnotatedInstance)
        assert annotated_instance.data == test_payload
        assert annotated_instance.caller_id is test_caller_id_instance
        assert annotated_instance.timestamp == fixed_now
        # Check it's recent (within a small delta if not patching now())
        # time_delta = datetime.datetime.now(datetime.timezone.utc) - annotated_instance.timestamp
        # assert time_delta.total_seconds() < 0.1
        print(
            "--- Test: test_processor_process_data_no_timestamp finished ---"
        )


"""Tests for RuntimeDataHandlerBase."""


# Minimal concrete implementation for testing
class ConcreteRuntimeDataHandler(RuntimeDataHandlerBase[str, str]):
    def __init__(self, data_reader, event_source, mocker):  # Added mocker
        super().__init__(data_reader, event_source)
        self._register_caller_mock = mocker.MagicMock(
            spec=self._register_caller
        )
        self._unregister_caller_mock = mocker.MagicMock(
            spec=self._unregister_caller
        )
        self._try_get_caller_id_mock = mocker.MagicMock(
            spec=self._try_get_caller_id
        )

    def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> Optional[
        EndpointDataProcessor
    ]:  # Corrected reference and made Optional to match base class for test stub
        return self._register_caller_mock(caller_id, endpoint, port)

    def _unregister_caller(self, caller_id: CallerIdentifier) -> bool:
        return self._unregister_caller_mock(caller_id)

    def _try_get_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        return self._try_get_caller_id_mock(endpoint, port)


@pytest.fixture
def mock_data_reader_fixture(mocker):
    return mocker.MagicMock(spec=RemoteDataReader[AnnotatedInstance[str]])


@pytest.fixture
def mock_event_source_fixture(mocker):
    return mocker.MagicMock(
        spec=AsyncPoller[SerializableAnnotatedInstance[str]]
    )


@pytest.fixture
def mock_context_fixture(mocker):
    return mocker.MagicMock(spec=grpc.aio.ServicerContext)


@pytest.fixture
def handler_fixture(
    mock_data_reader_fixture, mock_event_source_fixture, mocker
):
    return ConcreteRuntimeDataHandler(
        mock_data_reader_fixture, mock_event_source_fixture, mocker
    )


@pytest.fixture
def mock_endpoint_processor_fixture(mocker):
    return mocker.MagicMock(spec=EndpointDataProcessor)


class TestRuntimeDataHandlerBaseRegisterCaller:
    """Tests for the register_caller method of RuntimeDataHandlerBase."""

    def test_register_caller_with_endpoint_port_success(
        self, handler_fixture, mock_endpoint_processor_fixture
    ):
        caller_id = CallerIdentifier.random()
        endpoint = "127.0.0.1"
        port = 8080
        handler_fixture._register_caller_mock.return_value = (
            mock_endpoint_processor_fixture
        )

        result = handler_fixture.register_caller(
            caller_id, endpoint=endpoint, port=port
        )

        handler_fixture._register_caller_mock.assert_called_once_with(
            caller_id, endpoint, port
        )
        assert result == mock_endpoint_processor_fixture

    def test_register_caller_with_context_success(
        self,
        handler_fixture,
        mock_context_fixture,
        mock_endpoint_processor_fixture,
        mocker,  # Ensure mocker is passed here
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

        result = handler_fixture.register_caller(
            caller_id, context=mock_context_fixture
        )

        mock_get_ip.assert_called_once_with(mock_context_fixture)
        mock_get_port.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_called_once_with(
            caller_id, expected_ip, expected_port
        )
        assert result == mock_endpoint_processor_fixture

    def test_register_caller_with_context_ip_none_returns_none(
        self,
        handler_fixture,
        mock_context_fixture,
        mocker,  # Ensure mocker is passed here
    ):
        caller_id = CallerIdentifier.random()
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value=None,
        )
        mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=1234,
        )  # Mock get_client_port as well

        result = handler_fixture.register_caller(
            caller_id, context=mock_context_fixture
        )

        mock_get_ip.assert_called_once_with(mock_context_fixture)
        # Depending on implementation, get_client_port might not be called if IP is None.
        # If it is called, its return value doesn't prevent returning None here.
        handler_fixture._register_caller_mock.assert_not_called()
        assert result is None

    def test_register_caller_with_context_port_none_raises_value_error(
        self,
        handler_fixture,
        mock_context_fixture,
        mocker,  # Ensure mocker is passed here
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
            handler_fixture.register_caller(
                caller_id, context=mock_context_fixture
            )

        assert (
            f"Could not determine client port from context for endpoint {expected_ip}"
            in str(excinfo.value)
        )
        mock_get_ip.assert_called_once_with(mock_context_fixture)
        mock_get_port.assert_called_once_with(mock_context_fixture)
        handler_fixture._register_caller_mock.assert_not_called()

    # Argument validation tests (already implemented in RuntimeDataHandlerBase by previous subtask)
    def test_register_caller_mutex_args_endpoint_context(
        self, handler_fixture, mock_context_fixture
    ):
        """Test providing both endpoint and context raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(
                caller_id,
                endpoint="1.2.3.4",
                port=123,
                context=mock_context_fixture,
            )
        assert (
            "Exactly one of 'endpoint'/'port' combination or 'context' must be provided"
            in str(excinfo.value)
        )

    def test_register_caller_mutex_args_none(self, handler_fixture):
        """Test providing neither endpoint/port nor context raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(
                caller_id
            )  # No endpoint, port or context
        assert (
            "Exactly one of 'endpoint'/'port' combination or 'context' must be provided"
            in str(excinfo.value)
        )

    def test_register_caller_endpoint_without_port(self, handler_fixture):
        """Test providing endpoint without port raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(
                caller_id, endpoint="1.2.3.4"
            )  # Port is None
        assert (
            "If 'endpoint' is provided, 'port' must also be provided"
            in str(excinfo.value)
        )

    def test_register_caller_port_without_endpoint(self, handler_fixture):
        """Test providing port without endpoint raises ValueError."""
        caller_id = CallerIdentifier.random()
        with pytest.raises(ValueError) as excinfo:
            handler_fixture.register_caller(
                caller_id, port=1234
            )  # Endpoint is None
        assert (
            "Exactly one of 'endpoint'/'port' combination or 'context' must be provided"  # Updated error message
            in str(excinfo.value)
        )

    def test_register_caller_context_is_not_servicer_context_raises_type_error(
        self, handler_fixture, mocker  # Ensure mocker is passed here
    ):
        """Test that if context is not None, it must be a ServicerContext."""
        caller_id = CallerIdentifier.random()
        not_a_servicer_context = object()  # Some other object type

        # These can return valid values, the type check for context should happen before.
        mock_get_ip = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_ip",
            return_value="1.2.3.4",
        )
        mock_get_port = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.get_client_port",
            return_value=1234,
        )

        with pytest.raises(TypeError) as excinfo:
            handler_fixture.register_caller(
                caller_id, context=not_a_servicer_context
            )

        assert (
            "Expected context to be an instance of grpc.aio.ServicerContext"
            in str(excinfo.value)
        )
        mock_get_ip.assert_not_called()  # Should fail before trying to use context
        mock_get_port.assert_not_called()
        handler_fixture._register_caller_mock.assert_not_called()
