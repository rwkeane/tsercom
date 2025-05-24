import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call as mock_call_obj # Renamed to avoid conflict
import uuid # For generating valid UUIDs for tests

import grpc # For grpc.StatusCode and grpc.aio.ServicerContext
from grpc.aio import ServicerContext # For type hinting

# SUT (Subject Under Test)
from tsercom.rpc.serialization.caller_id_extraction import (
    extract_id_from_call,
    extract_id_from_first_call,
    CALLER_ID_METADATA_KEY # If needed for custom extractor tests involving metadata
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.util.is_running_tracker import IsRunningTracker

# Assuming GrpcCallerIdMessage is the actual protobuf message class
# Adjust path if necessary based on previous findings (e.g., v1_70, v1_62)
# For now, using the path from the problem description.
try:
    from tsercom.caller_id.proto.generated.v1_70.caller_id_pb2 import CallerId as GrpcCallerIdMessage
except ImportError: # pragma: no cover
    # Fallback or define a mock if specific version not found, for basic structure
    # This is a workaround for potential environment differences not related to the SUT's logic.
    # In a real CI, the correct proto version should be available.
    print("Warning: Could not import GrpcCallerIdMessage from v1_70, using MagicMock as placeholder.")
    GrpcCallerIdMessage = MagicMock(name="MockGrpcCallerIdMessage")
    # If GrpcCallerIdMessage is MagicMock, tests relying on its constructor need care.
    # For tests where it's just an attribute holder, MagicMock(id=...) might suffice.


# Helper to create an async iterator from a list
async def async_iterator_from_list(items):
    for item in items:
        # print(f"Async iterator yielding: {item}") # Diagnostic
        await asyncio.sleep(0) # Ensure it behaves like a true async iterator
        yield item

@pytest.mark.asyncio
class TestCallerIdExtraction:

    @pytest.fixture
    def mock_servicer_context(self):
        context = AsyncMock(spec=ServicerContext)
        context.abort = AsyncMock(name="servicer_context_abort")
        # Add other context methods if SUT uses them (e.g., invocation_metadata)
        context.invocation_metadata = MagicMock(return_value=[]) # Default empty metadata
        return context

    @pytest.fixture
    def mock_is_running_tracker(self):
        tracker = MagicMock(spec=IsRunningTracker)
        # create_stoppable_iterator should ideally return an async iterator
        # For simplicity, we'll have it return the input iterator directly by default,
        # or a wrapper that can be controlled if needed for specific tests.
        async def default_stoppable_iterator_impl(iterator, stop_event=None): # stop_event is not used by SUT
            async for item in iterator:
                yield item
        
        tracker.create_stoppable_iterator = MagicMock(side_effect=default_stoppable_iterator_impl, name="mock_create_stoppable_iterator")
        tracker.get = MagicMock(return_value=True, name="mock_is_running_get") # Default to "running"
        return tracker

    @pytest.fixture
    def mock_call_object(self):
        # A simple MagicMock that can have an 'id' attribute set
        return MagicMock(name="MockCallObject")

    # --- Tests for extract_id_from_call ---

    def test_extract_id_success_default_extractor(self, mock_servicer_context, mock_call_object):
        print("\n--- Test: test_extract_id_success_default_extractor ---")
        valid_uuid_str = str(uuid.uuid4())
        # If GrpcCallerIdMessage is a real class, this is how you'd make it.
        # If it's a MagicMock, this will just work due to MagicMock's nature.
        mock_call_object.id = GrpcCallerIdMessage(id=valid_uuid_str) if callable(GrpcCallerIdMessage) else MagicMock(id=valid_uuid_str)

        caller_id = extract_id_from_call(mock_call_object, mock_servicer_context)

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_extract_id_success_default_extractor finished ---")

    def test_extract_id_missing_id_default_extractor(self, mock_servicer_context, mock_call_object):
        print("\n--- Test: test_extract_id_missing_id_default_extractor ---")
        mock_call_object.id = None # Simulate missing ID field

        caller_id = extract_id_from_call(mock_call_object, mock_servicer_context)

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID"
        )
        print("--- Test: test_extract_id_missing_id_default_extractor finished ---")
    
    def test_extract_id_call_object_has_no_id_attr(self, mock_servicer_context):
        print("\n--- Test: test_extract_id_call_object_has_no_id_attr ---")
        call_without_id_attr = MagicMock(spec=[]) # No 'id' attribute defined

        caller_id = extract_id_from_call(call_without_id_attr, mock_servicer_context)
        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID" # Or "Invalid CallerID type" depending on SUT
        )
        print("--- Test: test_extract_id_call_object_has_no_id_attr finished ---")


    def test_extract_id_malformed_id_try_parse_returns_none(self, mock_servicer_context, mock_call_object):
        print("\n--- Test: test_extract_id_malformed_id_try_parse_returns_none ---")
        # This relies on CallerIdentifier.try_parse returning None for "invalid-id-format"
        mock_call_object.id = GrpcCallerIdMessage(id="invalid-id-format") if callable(GrpcCallerIdMessage) else MagicMock(id="invalid-id-format")

        caller_id = extract_id_from_call(mock_call_object, mock_servicer_context)

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )
        print("--- Test: test_extract_id_malformed_id_try_parse_returns_none finished ---")

    def test_extract_id_with_validation_success(self, mock_servicer_context, mock_call_object):
        print("\n--- Test: test_extract_id_with_validation_success ---")
        valid_uuid_str = str(uuid.uuid4())
        validate_against_id = CallerIdentifier.try_parse(valid_uuid_str)
        assert validate_against_id is not None # Ensure our test ID is valid

        mock_call_object.id = GrpcCallerIdMessage(id=valid_uuid_str) if callable(GrpcCallerIdMessage) else MagicMock(id=valid_uuid_str)

        caller_id = extract_id_from_call(
            mock_call_object, mock_servicer_context, validate_against_id=validate_against_id
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert caller_id == validate_against_id
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_extract_id_with_validation_success finished ---")

    def test_extract_id_with_validation_failure(self, mock_servicer_context, mock_call_object):
        print("\n--- Test: test_extract_id_with_validation_failure ---")
        uuid_str_A = str(uuid.uuid4())
        uuid_str_B = str(uuid.uuid4())
        
        validate_against_id = CallerIdentifier.try_parse(uuid_str_A)
        mock_call_object.id = GrpcCallerIdMessage(id=uuid_str_B) if callable(GrpcCallerIdMessage) else MagicMock(id=uuid_str_B)

        caller_id = extract_id_from_call(
            mock_call_object, mock_servicer_context, validate_against_id=validate_against_id
        )

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )
        print("--- Test: test_extract_id_with_validation_failure finished ---")

    def test_extract_id_custom_extractor(self, mock_servicer_context, mock_call_object):
        print("\n--- Test: test_extract_id_custom_extractor ---")
        valid_uuid_str = str(uuid.uuid4())
        # Call object has ID in a custom field
        mock_call_object.custom_field_data = GrpcCallerIdMessage(id=valid_uuid_str) if callable(GrpcCallerIdMessage) else MagicMock(id=valid_uuid_str)
        
        custom_extractor = lambda x: x.custom_field_data # Extracts GrpcCallerIdMessage

        caller_id = extract_id_from_call(
            mock_call_object, mock_servicer_context, extractor=custom_extractor
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_extract_id_custom_extractor finished ---")

    # --- Tests for extract_id_from_first_call ---

    async def test_extract_first_success(self, mock_servicer_context, mock_is_running_tracker, mock_call_object):
        print("\n--- Test: test_extract_first_success ---")
        valid_uuid_str = str(uuid.uuid4())
        call1 = MagicMock(name="Call1")
        call1.id = GrpcCallerIdMessage(id=valid_uuid_str) if callable(GrpcCallerIdMessage) else MagicMock(id=valid_uuid_str)
        call2 = MagicMock(name="Call2") # Subsequent call, ID not relevant for this test part

        iterator = async_iterator_from_list([call1, call2])
        
        caller_id, first_call, processed_iterator = await extract_id_from_first_call(
            iterator, mock_servicer_context, mock_is_running_tracker
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        assert first_call is call1
        mock_servicer_context.abort.assert_not_called()
        # Check that create_stoppable_iterator was called
        mock_is_running_tracker.create_stoppable_iterator.assert_called_once_with(iterator)
        
        # Verify the processed_iterator contains the remaining items (call2)
        remaining_items = [item async for item in processed_iterator]
        assert remaining_items == [call2]
        print("--- Test: test_extract_first_success finished ---")

    async def test_extract_first_iterator_empty(self, mock_servicer_context, mock_is_running_tracker):
        print("\n--- Test: test_extract_first_iterator_empty ---")
        iterator = async_iterator_from_list([])
        
        caller_id, first_call, processed_iterator = await extract_id_from_first_call(
            iterator, mock_servicer_context, mock_is_running_tracker
        )

        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED, "First call never received!"
        )
        # Check that create_stoppable_iterator was called even for empty iterator
        mock_is_running_tracker.create_stoppable_iterator.assert_called_once_with(iterator)
        
        remaining_items = [item async for item in processed_iterator]
        assert not remaining_items
        print("--- Test: test_extract_first_iterator_empty finished ---")

    async def test_extract_first_is_running_becomes_false(self, mock_servicer_context, mock_is_running_tracker, mock_call_object):
        print("\n--- Test: test_extract_first_is_running_becomes_false ---")
        # mock_call_object won't be used as iteration stops early
        iterator = async_iterator_from_list([mock_call_object]) 
        
        # Simulate is_running_tracker.get() returning False
        mock_is_running_tracker.get.return_value = False 

        caller_id, first_call, processed_iterator = await extract_id_from_first_call(
            iterator, mock_servicer_context, mock_is_running_tracker
        )

        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_not_called() # Abort not called if tracker stops it
        mock_is_running_tracker.get.assert_called() # Should be checked at least once
        
        remaining_items = [item async for item in processed_iterator]
        assert not remaining_items # Iterator should appear empty or not advance
        print("--- Test: test_extract_first_is_running_becomes_false finished ---")

    async def test_extract_first_id_extraction_fails_on_first_item(
        self, mock_servicer_context, mock_is_running_tracker, mock_call_object
    ):
        print("\n--- Test: test_extract_first_id_extraction_fails_on_first_item ---")
        call1 = MagicMock(name="Call1_BadID")
        call1.id = GrpcCallerIdMessage(id="invalid-id-format") if callable(GrpcCallerIdMessage) else MagicMock(id="invalid-id-format")
        
        iterator = async_iterator_from_list([call1])

        caller_id, first_call, processed_iterator = await extract_id_from_first_call(
            iterator, mock_servicer_context, mock_is_running_tracker
        )

        assert caller_id is None
        assert first_call is call1 # first_call is returned even if ID extraction fails
        # Abort is called by the inner extract_id_from_call
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )
        
        remaining_items = [item async for item in processed_iterator]
        assert not remaining_items # Iterator consumed the first (bad) item
        print("--- Test: test_extract_first_id_extraction_fails_on_first_item finished ---")

    async def test_extract_first_general_exception_in_iterator(
        self, mock_servicer_context, mock_is_running_tracker, mock_call_object
    ):
        print("\n--- Test: test_extract_first_general_exception_in_iterator ---")
        test_exception = ValueError("Test error in iterator")
        
        async def faulty_iterator():
            # Yield one valid item before failing, to ensure first_call is processed
            valid_uuid_str = str(uuid.uuid4())
            call_good = MagicMock(name="GoodCall")
            call_good.id = GrpcCallerIdMessage(id=valid_uuid_str) if callable(GrpcCallerIdMessage) else MagicMock(id=valid_uuid_str)
            yield call_good
            raise test_exception

        iterator = faulty_iterator()
        
        # We expect the ValueError to propagate if it happens after the first item.
        # If it happens trying to get the *first* item, SUT aborts with CANCELLED.
        # Let's modify faulty_iterator to fail *while trying to get the first item*.
        async def faulty_iterator_fails_first():
            print("Faulty iterator starting...")
            await asyncio.sleep(0) # Ensure it's an async generator
            raise test_exception
            yield "never_yielded" # pragma: no cover
        
        iterator_fails_first = faulty_iterator_fails_first()

        caller_id, first_call, processed_iterator = await extract_id_from_first_call(
            iterator_fails_first, mock_servicer_context, mock_is_running_tracker
        )
        
        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED, "Error processing first call!"
        )
        mock_is_running_tracker.create_stoppable_iterator.assert_called_once_with(iterator_fails_first)
        
        # Check that the processed_iterator is now empty or yields nothing
        processed_items = [item async for item in processed_iterator]
        assert not processed_items
        print("--- Test: test_extract_first_general_exception_in_iterator finished ---")

```

**Summary of Implementation (Turn 2):**
1.  **Imports**: Added necessary modules. `GrpcCallerIdMessage` is conditionally imported with a `MagicMock` fallback to handle potential environment differences (though in a real CI, the correct version should be present).
2.  **Helper `async_iterator_from_list`**: Created this async generator.
3.  **Fixtures**:
    *   `mock_servicer_context`: Mocks `grpc.aio.ServicerContext` and its `abort` method. Also mocks `invocation_metadata()` as it's accessed by SUT.
    *   `mock_is_running_tracker`: Mocks `IsRunningTracker`, its `create_stoppable_iterator` (to return the input iterator wrapped in a default async generator behavior), and `get` (to control loop termination).
    *   `mock_call_object`: A `MagicMock` for simulating gRPC call objects.
4.  **`TestCallerIdExtraction` Class**:
    *   **`extract_id_from_call` Tests**:
        *   `test_extract_id_success_default_extractor`: Valid UUID in `call.id`. Asserts `CallerIdentifier` instance is returned, no abort.
        *   `test_extract_id_missing_id_default_extractor`: `call.id` is `None`. Asserts `None` returned, context aborted with `INVALID_ARGUMENT`, "Missing CallerID".
        *   `test_extract_id_call_object_has_no_id_attr`: Call object lacks `id` attribute. Asserts `None` returned, context aborted.
        *   `test_extract_id_malformed_id_try_parse_returns_none`: `call.id` contains an invalid UUID string. Relies on `CallerIdentifier.try_parse` to return `None`. Asserts `None` returned, context aborted with `INVALID_ARGUMENT`, "Invalid CallerID received".
        *   `test_extract_id_with_validation_success`: `validate_against_id` matches extracted ID. Asserts `CallerIdentifier` returned, no abort.
        *   `test_extract_id_with_validation_failure`: `validate_against_id` does not match. Asserts `None` returned, context aborted.
        *   `test_extract_id_custom_extractor`: Uses a lambda to extract ID from `call.custom_field_data`. Asserts `CallerIdentifier` returned.
    *   **`extract_id_from_first_call` Tests**:
        *   `test_extract_first_success`: Valid ID in the first item of an async iterator. Asserts correct `caller_id`, `first_call`, and that the `processed_iterator` contains subsequent items. `create_stoppable_iterator` is checked.
        *   `test_extract_first_iterator_empty`: Input iterator is empty. Asserts `caller_id` and `first_call` are `None`, context aborted with `CANCELLED`.
        *   `test_extract_first_is_running_becomes_false`: `is_running_tracker.get()` returns `False`. Asserts `caller_id` and `first_call` are `None`, no abort.
        *   `test_extract_first_id_extraction_fails_on_first_item`: First item has malformed ID. Asserts `caller_id` is `None`, `first_call` is the problematic item, context aborted by inner `extract_id_from_call`.
        *   `test_extract_first_general_exception_in_iterator`: Iterator raises an exception when trying to get the first item. Asserts `caller_id` and `first_call` are `None`, context aborted with `CANCELLED`.

This suite covers the specified scenarios. The direct use of `grpc.StatusCode` will test if the `AttributeError` related to it is still an issue.The test file `tsercom/rpc/serialization/caller_id_extraction_unittest.py` has been implemented with tests for `extract_id_from_call` and `extract_id_from_first_call`.

**Key implementations:**
-   **Mocking Strategy**:
    -   `grpc.aio.ServicerContext`: Mocked, focusing on its `abort` method and `invocation_metadata()`.
    -   `CallerIdentifier.try_parse`: Relied on its actual behavior for malformed ID strings; direct patching was not implemented in this pass but can be added if finer control over `try_parse` returning `None` is needed for specific non-UUID "invalid" strings.
    -   `IsRunningTracker`: Mocked, controlling behavior of `create_stoppable_iterator` (to pass through the iterator by default) and `get` (to simulate the tracker stopping).
    -   Call objects (`request` or items from `request_iterator`): Simulated using `MagicMock`, allowing attributes like `id` (or custom fields for custom extractors) to be set.
    -   `GrpcCallerIdMessage`: Imported from `tsercom.caller_id.proto.generated.v1_70.caller_id_pb2`. A `MagicMock` fallback was included for robustness if the specific proto version isn't found, though tests assume the real class is available for `isinstance` checks if they were more stringent (currently, they rely on duck typing for `id` attribute).
-   **Helper Function**: `async_iterator_from_list` to easily create async iterators for testing `extract_id_from_first_call`.
-   **Test Scenarios for `extract_id_from_call`**:
    -   Successful extraction with default extractor.
    -   Missing `id` attribute or `id` attribute being `None` on the call object.
    -   Malformed ID string that `CallerIdentifier.try_parse` would reject.
    -   Successful validation against a provided `CallerIdentifier`.
    -   Failed validation against a provided `CallerIdentifier`.
    -   Successful extraction using a custom `extractor` function.
-   **Test Scenarios for `extract_id_from_first_call`**:
    -   Successful extraction from the first item of an async iterator.
    -   Handling of an empty async iterator.
    -   Behavior when `is_running_tracker.get()` returns `False` (simulating termination).
    -   ID extraction failing for the first item in the iterator (e.g., malformed ID).
    -   Handling of a general exception raised by the input async iterator when fetching the first item.
-   **Error Handling**: Assertions check for calls to `context.abort` with appropriate `grpc.StatusCode` values (`INVALID_ARGUMENT`, `CANCELLED`) and messages.

The tests use `pytest` and `pytest-asyncio`. Direct use of `grpc.StatusCode` might trigger the previously seen `AttributeError` if the `grpc` module isn't fully initialized when `pytest` collects/imports tests. This run will confirm that.

I will now run these tests. Print statements are included for diagnostics.
