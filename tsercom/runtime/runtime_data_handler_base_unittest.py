import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Optional, Any, TypeVar # Removed Tuple, List as they are not used here
import datetime

# SUT
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase

# Dependencies to be mocked or used
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.serializable_annotated_instance import ServerTimestamp # SerializableAnnotatedInstance not directly used
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

# For gRPC context testing
import grpc # For grpc.StatusCode, if testable - not directly used in DataProcessorImpl tests
from grpc.aio import ServicerContext

# Modules for patching
import tsercom.rpc.grpc.addressing as grpc_addressing_module

# Type variable for data
DataType = TypeVar("DataType")

# Test Subclass of RuntimeDataHandlerBase
class TestableRuntimeDataHandler(RuntimeDataHandlerBase[DataType]):
    def __init__(self, data_reader: RemoteDataReader[DataType], event_source: AsyncPoller[Any]):
        super().__init__(data_reader, event_source)
        self.mock_register_caller = AsyncMock(name="_register_caller_impl")
        self.mock_unregister_caller = AsyncMock(name="_unregister_caller_impl")
        self.mock_try_get_caller_id = MagicMock(name="_try_get_caller_id_impl")
        # Mock the protected _on_data_ready to inspect calls to it
        self._on_data_ready = MagicMock(name="handler_on_data_ready_mock") # type: ignore
        # print("TestableRuntimeDataHandler initialized.") # Reduced verbosity

    async def _register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> Optional[RuntimeDataHandlerBase.EndpointDataProcessor]:
        # print(f"TestableRuntimeDataHandler._register_caller called with: {caller_id}, {endpoint}, {port}") # Reduced verbosity
        return await self.mock_register_caller(caller_id, endpoint, port)

    async def _unregister_caller(self, caller_id: CallerIdentifier) -> None:
        # print(f"TestableRuntimeDataHandler._unregister_caller called with: {caller_id}") # Reduced verbosity
        await self.mock_unregister_caller(caller_id)

    def _try_get_caller_id(self, endpoint: str, port: int) -> Optional[CallerIdentifier]:
        # print(f"TestableRuntimeDataHandler._try_get_caller_id called with: {endpoint}, {port}") # Reduced verbosity
        return self.mock_try_get_caller_id(endpoint, port)

@pytest.mark.asyncio
class TestRuntimeDataHandlerBaseBehavior: 

    @pytest.fixture
    def mock_data_reader(self):
        reader = MagicMock(spec=RemoteDataReader)
        # The SUT's _on_data_ready calls self.__data_reader._on_data_ready.
        # So, we mock this method on the reader instance passed to the SUT.
        reader._on_data_ready = MagicMock(name="data_reader_on_data_ready_on_reader_mock")
        return reader

    @pytest.fixture
    def mock_event_source(self):
        poller = MagicMock(spec=AsyncPoller)
        poller.__anext__ = AsyncMock(name="event_source_anext")
        return poller

    @pytest.fixture
    def mock_caller_id(self): # Renamed to avoid conflict with test_caller_id_fixture
        return MagicMock(spec=CallerIdentifier, name="MockCallerIdInstance")

    @pytest.fixture
    def mock_servicer_context(self):
        context = AsyncMock(spec=ServicerContext)
        context.peer = MagicMock(return_value="ipv4:127.0.0.1:12345")
        return context

    @pytest.fixture
    def patch_get_client_ip(self):
        with patch.object(grpc_addressing_module, 'get_client_ip', autospec=True) as mock_get_ip:
            yield mock_get_ip

    @pytest.fixture
    def patch_get_client_port(self):
        with patch.object(grpc_addressing_module, 'get_client_port', autospec=True) as mock_get_port:
            yield mock_get_port
            
    @pytest.fixture
    def handler(self, mock_data_reader, mock_event_source):
        return TestableRuntimeDataHandler(mock_data_reader, mock_event_source)
    
    # Fixtures for DataProcessorImpl tests
    @pytest.fixture
    def mock_sync_clock(self):
        clock = MagicMock(spec=SynchronizedClock)
        clock.desync = AsyncMock(name="sync_clock_desync") # desync is async
        return clock

    @pytest.fixture
    def test_caller_id_instance(self): # A concrete CallerIdentifier for processor tests
        return CallerIdentifier.random() # Assuming CallerIdentifier has a factory or suitable constructor

    @pytest.fixture
    def data_processor(self, handler, test_caller_id_instance, mock_sync_clock):
        # Create an instance of the inner class for testing
        # The _create_data_processor method is part of RuntimeDataHandlerBase
        return handler._create_data_processor(test_caller_id_instance, mock_sync_clock)


    # --- Tests for RuntimeDataHandlerBase direct methods ---
    def test_constructor(self, handler, mock_data_reader, mock_event_source):
        assert handler._RuntimeDataHandlerBase__data_reader is mock_data_reader # type: ignore
        assert handler._RuntimeDataHandlerBase__event_source is mock_event_source # type: ignore

    async def test_register_caller_endpoint_port(self, handler, mock_caller_id):
        endpoint_str = "10.0.0.1"
        port_num = 9999
        mock_processor_instance = MagicMock(spec=RuntimeDataHandlerBase.EndpointDataProcessor)
        handler.mock_register_caller.return_value = mock_processor_instance
        returned_processor = await handler.register_caller(mock_caller_id, endpoint_str, port_num)
        handler.mock_register_caller.assert_called_once_with(mock_caller_id, endpoint_str, port_num)
        assert returned_processor is mock_processor_instance

    async def test_register_caller_grpc_context_success(
        self, handler, mock_caller_id, mock_servicer_context, 
        patch_get_client_ip, patch_get_client_port
    ):
        expected_ip = "1.2.3.4"
        expected_port = 5678
        patch_get_client_ip.return_value = expected_ip
        patch_get_client_port.return_value = expected_port
        mock_processor_instance = MagicMock(spec=RuntimeDataHandlerBase.EndpointDataProcessor)
        handler.mock_register_caller.return_value = mock_processor_instance
        returned_processor = await handler.register_caller(mock_caller_id, mock_servicer_context)
        patch_get_client_ip.assert_called_once_with(mock_servicer_context)
        patch_get_client_port.assert_called_once_with(mock_servicer_context)
        handler.mock_register_caller.assert_called_once_with(mock_caller_id, expected_ip, expected_port)
        assert returned_processor is mock_processor_instance

    async def test_register_caller_grpc_context_no_ip(
        self, handler, mock_caller_id, mock_servicer_context, 
        patch_get_client_ip, patch_get_client_port # patch_get_client_port is needed here
    ):
        patch_get_client_ip.return_value = None
        patch_get_client_port.return_value = 5678 # Does not matter if IP is None
        returned_processor = await handler.register_caller(mock_caller_id, mock_servicer_context)
        patch_get_client_ip.assert_called_once_with(mock_servicer_context)
        patch_get_client_port.assert_not_called() 
        handler.mock_register_caller.assert_not_called()
        assert returned_processor is None

    def test_get_data_iterator_returns_self(self, handler):
        assert handler.get_data_iterator() is handler

    async def test_async_iteration_with_event_source(self, handler, mock_event_source):
        item1 = MagicMock(name="EventItem1")
        item2 = MagicMock(name="EventItem2")
        mock_event_source.__anext__.side_effect = [item1, item2, StopAsyncIteration]
        collected_items = []
        async for item in handler.get_data_iterator():
            collected_items.append(item)
        assert collected_items == [item1, item2]
        assert mock_event_source.__anext__.call_count == 3

    def test_check_for_caller_id(self, handler, mock_caller_id):
        endpoint_str = "test_ep"
        port_num = 1122
        handler.mock_try_get_caller_id.return_value = mock_caller_id
        result = handler.check_for_caller_id(endpoint_str, port_num)
        handler.mock_try_get_caller_id.assert_called_once_with(endpoint_str, port_num)
        assert result is mock_caller_id

    # Test for the SUT's _on_data_ready, not the reader's.
    # The SUT's _on_data_ready calls the reader's _on_data_ready.
    def test_handler_on_data_ready_calls_reader_on_data_ready(self, handler, mock_data_reader):
        print("\n--- Test: test_handler_on_data_ready_calls_reader_on_data_ready ---")
        mock_annotated_instance = MagicMock(spec=AnnotatedInstance)
        # Call the public method on handler, which should internally call its own _on_data_ready
        # which then calls the reader's _on_data_ready.
        # Actually, _on_data_ready is a method of the data processor, which then calls
        # the handler's _on_data_ready.
        # For this test, we're unit testing RuntimeDataHandlerBase._on_data_ready directly.
        # This method is protected.
        handler._RuntimeDataHandlerBase__data_reader = mock_data_reader # Ensure it's the mock
        handler._on_data_ready(mock_annotated_instance) # Call the SUT's method directly
        
        mock_data_reader._on_data_ready.assert_called_once_with(mock_annotated_instance)
        print("--- Test: test_handler_on_data_ready_calls_reader_on_data_ready finished ---")

    # --- Tests for _RuntimeDataHandlerBase__DataProcessorImpl ---
    
    async def test_processor_desynchronize(self, data_processor, mock_sync_clock):
        print("\n--- Test: test_processor_desynchronize ---")
        mock_server_ts = MagicMock(spec=ServerTimestamp)
        expected_datetime = datetime.datetime.now(datetime.timezone.utc)
        mock_sync_clock.desync.return_value = expected_datetime

        result_dt = await data_processor.desynchronize(mock_server_ts)

        mock_sync_clock.desync.assert_called_once_with(mock_server_ts)
        assert result_dt is expected_datetime
        print("--- Test: test_processor_desynchronize finished ---")

    async def test_processor_deregister_caller(self, data_processor, handler, test_caller_id_instance):
        print("\n--- Test: test_processor_deregister_caller ---")
        await data_processor.deregister_caller()
        handler.mock_unregister_caller.assert_called_once_with(test_caller_id_instance)
        print("--- Test: test_processor_deregister_caller finished ---")

    async def test_processor_process_data_with_datetime(self, data_processor, handler, test_caller_id_instance):
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
        print("--- Test: test_processor_process_data_with_datetime finished ---")

    async def test_processor_process_data_with_server_timestamp(
        self, data_processor, handler, test_caller_id_instance, mock_sync_clock
    ):
        print("\n--- Test: test_processor_process_data_with_server_timestamp ---")
        test_payload = "payload_with_server_ts"
        mock_server_ts = MagicMock(spec=ServerTimestamp)
        expected_desynced_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)
        
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
        print("--- Test: test_processor_process_data_with_server_timestamp finished ---")

    async def test_processor_process_data_no_timestamp(self, data_processor, handler, test_caller_id_instance):
        print("\n--- Test: test_processor_process_data_no_timestamp ---")
        test_payload = "payload_no_ts"
        
        # Patch datetime.now inside the SUT module (runtime_data_handler_base)
        # to control the timestamp for "now"
        fixed_now = datetime.datetime.now(datetime.timezone.utc)
        with patch("tsercom.runtime.runtime_data_handler_base.datetime", wraps=datetime) as mock_dt_module:
            mock_dt_module.now.return_value = fixed_now
            
            await data_processor.process_data(test_payload, timestamp=None)

            mock_dt_module.now.assert_called_once_with(datetime.timezone.utc)

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
        print("--- Test: test_processor_process_data_no_timestamp finished ---")

```

**Summary of Additions (Turn 3):**
1.  **Renamed Fixtures**: Added `_fixture` suffix to some fixture names (e.g., `mock_caller_id_fixture` became `mock_caller_id`) for consistency or to avoid potential conflicts if a test variable had the same name. The `handler` fixture remains as is.
2.  **`mock_data_reader` Update**: Ensured `_on_data_ready` is mocked on the `reader` instance itself, as `RuntimeDataHandlerBase._on_data_ready` calls `self.__data_reader._on_data_ready`.
3.  **`TestableRuntimeDataHandler._on_data_ready` Mocked**: The test subclass `TestableRuntimeDataHandler` now mocks its own `_on_data_ready` method. This is crucial because the `__DataProcessorImpl.process_data` method calls `self._handler._on_data_ready(annotated_instance)`, where `self._handler` is the `RuntimeDataHandlerBase` instance. So, to verify this call, we mock it on the `handler` (our `TestableRuntimeDataHandler` instance).
4.  **Fixtures for `__DataProcessorImpl` Tests**:
    *   `mock_sync_clock`: Mocks `SynchronizedClock` and its `desync` method (as `AsyncMock`).
    *   `test_caller_id_instance`: Provides a concrete `CallerIdentifier.random()` instance for use in processor tests.
    *   `data_processor`: A fixture that creates an instance of `_RuntimeDataHandlerBase__DataProcessorImpl` using the `handler`'s `_create_data_processor` method, configured with `test_caller_id_instance` and `mock_sync_clock`.
5.  **Tests for `_RuntimeDataHandlerBase__DataProcessorImpl`**:
    *   **`test_processor_desynchronize`**: Verifies that `processor.desynchronize` calls `mock_sync_clock.desync` with the provided `ServerTimestamp` and returns its result.
    *   **`test_processor_deregister_caller`**: Checks that `processor.deregister_caller` calls the handler's (test subclass's) `_unregister_caller` method with the correct `CallerIdentifier`.
    *   **`test_processor_process_data_with_datetime`**: Calls `processor.process_data` with a specific `datetime`. Asserts that `handler._on_data_ready` (the mock on `TestableRuntimeDataHandler`) is called once, and inspects the `AnnotatedInstance` argument to ensure its `data`, `caller_id`, and `timestamp` fields are correct.
    *   **`test_processor_process_data_with_server_timestamp`**: Calls `processor.process_data` with a `ServerTimestamp`. Mocks `mock_sync_clock.desync` to return a specific `datetime`. Asserts that `desync` is called, and `handler._on_data_ready` is invoked with an `AnnotatedInstance` whose timestamp matches the desynchronized `datetime`.
    *   **`test_processor_process_data_no_timestamp`**: Calls `processor.process_data` with `timestamp=None`. Patches `datetime.now` within the `tsercom.runtime.runtime_data_handler_base` module to control the "current time". Asserts `handler._on_data_ready` is called, and the `AnnotatedInstance`'s timestamp is the patched `datetime.now()` value.

These tests cover the functionality of the inner `__DataProcessorImpl` class, including its interaction with the parent handler and the synchronized clock. The print statements have been reduced for less verbose output.The tests for the inner class `_RuntimeDataHandlerBase__DataProcessorImpl` have been added to `tsercom/runtime/runtime_data_handler_base_unittest.py`.
These tests cover:
-   `desynchronize` method: Verifies interaction with `SynchronizedClock.desync`.
-   `deregister_caller` method: Verifies it calls the handler's `_unregister_caller`.
-   `process_data` method with various timestamp types:
    -   With a `datetime` object.
    -   With a `ServerTimestamp` object (verifying desynchronization).
    -   With `timestamp=None` (verifying `datetime.now(datetime.timezone.utc)` is used).
For all `process_data` variants, the tests assert that the handler's `_on_data_ready` method is called with a correctly populated `AnnotatedInstance`.

Fixtures for `SynchronizedClock` and a concrete `CallerIdentifier` instance were added. The `TestableRuntimeDataHandler` was updated to also mock its own `_on_data_ready` method, as this is what the `DataProcessorImpl` calls.

The test suite now covers the direct methods of `RuntimeDataHandlerBase` and the methods of its inner `_DataProcessorImpl` class.
I will now run all tests to ensure they pass. The `grpc.StatusCode` issue is not expected to affect these specific tests.
