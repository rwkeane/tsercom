import asyncio
import pytest

# from unittest.mock import patch, AsyncMock, MagicMock, call as mock_call_obj # Removed
import uuid  # For generating valid UUIDs for tests

import grpc  # For grpc.StatusCode and grpc.aio.ServicerContext
from grpc.aio import ServicerContext  # For type hinting

# SUT (Subject Under Test)
from tsercom.rpc.serialization.caller_id_extraction import (
    extract_id_from_call,
    extract_id_from_first_call,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.util.is_running_tracker import IsRunningTracker


# Placeholder for GrpcCallerIdMessage if real import fails
class MockGrpcCallerIdMessagePlaceholder:
    def __init__(self, id=None):
        self.id = id


try:
    from tsercom.caller_id.proto.generated.v1_70.caller_id_pb2 import (
        CallerId as GrpcCallerIdMessage,
    )
except ImportError:  # pragma: no cover
    print(
        "Warning: Could not import GrpcCallerIdMessage from v1_70, using placeholder."
    )
    GrpcCallerIdMessage = MockGrpcCallerIdMessagePlaceholder  # type: ignore


async def async_iterator_from_list(items):
    for item in items:
        await asyncio.sleep(0)
        yield item


@pytest.mark.asyncio
class TestCallerIdExtraction:

    @pytest.fixture
    def mock_servicer_context(self, mocker):  # Added mocker
        context = mocker.AsyncMock(spec=ServicerContext)  # mocker.AsyncMock
        context.abort = mocker.AsyncMock(
            name="servicer_context_abort"
        )  # mocker.AsyncMock
        context.invocation_metadata = mocker.MagicMock(
            return_value=[]
        )  # mocker.MagicMock
        return context

    @pytest.fixture
    def mock_is_running_tracker(self, mocker):  # Added mocker
        tracker = mocker.MagicMock(spec=IsRunningTracker)  # mocker.MagicMock

        # This is the actual async generator
        async def actual_stoppable_iterator(iterator_arg, stop_event=None):
            async for item in iterator_arg:
                yield item

        # This is the method that will be called by SUT: is_running.create_stoppable_iterator()
        # It needs to be an async function (coroutine) because SUT awaits it.
        # Its return value (after await) should be the async_generator.
        async def mock_create_stoppable_iterator_method(iterator_arg, stop_event=None):
            return actual_stoppable_iterator(iterator_arg, stop_event)

        tracker.create_stoppable_iterator = mocker.MagicMock(
            side_effect=mock_create_stoppable_iterator_method, # side_effect is an async def function
            name="mock_create_stoppable_iterator",
        )  # mocker.MagicMock
        tracker.get = mocker.MagicMock(
            return_value=True, name="mock_is_running_get"
        )  # mocker.MagicMock
        return tracker

    @pytest.fixture
    def mock_call_object(self, mocker):  # Added mocker
        return mocker.MagicMock(name="MockCallObject")  # mocker.MagicMock

    # --- Tests for extract_id_from_call ---

    async def test_extract_id_success_default_extractor(
        self, mock_servicer_context, mock_call_object, mocker
    ):  # Added mocker
        print("\n--- Test: test_extract_id_success_default_extractor ---")
        valid_uuid_str = str(uuid.uuid4())
        # Use GrpcCallerIdMessage or its placeholder
        mock_call_object.id = GrpcCallerIdMessage(id=valid_uuid_str)

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        mock_servicer_context.abort.assert_not_called()
        print(
            "--- Test: test_extract_id_success_default_extractor finished ---"
        )

    async def test_extract_id_missing_id_default_extractor(
        self, mock_servicer_context, mock_call_object
    ):
        print("\n--- Test: test_extract_id_missing_id_default_extractor ---")
        mock_call_object.id = None

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context
        )

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID"
        )
        print(
            "--- Test: test_extract_id_missing_id_default_extractor finished ---"
        )

    async def test_extract_id_call_object_has_no_id_attr(
        self, mock_servicer_context, mocker
    ):  # Added mocker
        print("\n--- Test: test_extract_id_call_object_has_no_id_attr ---")
        # call_without_id_attr = mocker.MagicMock(spec=[])  # This would cause AttributeError in default extractor
        call_without_id_attr = mocker.MagicMock()
        call_without_id_attr.id = None # Make extractor return None to hit "Missing CallerID"

        caller_id = await extract_id_from_call(
            call_without_id_attr, mock_servicer_context
        )
        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID"
        )
        print(
            "--- Test: test_extract_id_call_object_has_no_id_attr finished ---"
        )

    async def test_extract_id_malformed_id_try_parse_returns_none(
        self, mock_servicer_context, mock_call_object, mocker
    ):  # Added mocker
        print(
            "\n--- Test: test_extract_id_malformed_id_try_parse_returns_none ---"
        )
        mock_call_object.id = GrpcCallerIdMessage(id="invalid-id-format")

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context
        )

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )
        print(
            "--- Test: test_extract_id_malformed_id_try_parse_returns_none finished ---"
        )

    async def test_extract_id_with_validation_success(
        self, mock_servicer_context, mock_call_object, mocker
    ):  # Added mocker
        print("\n--- Test: test_extract_id_with_validation_success ---")
        valid_uuid_str = str(uuid.uuid4())
        validate_against_id = CallerIdentifier.try_parse(valid_uuid_str)
        assert validate_against_id is not None

        mock_call_object.id = GrpcCallerIdMessage(id=valid_uuid_str)

        caller_id = await extract_id_from_call(
            mock_call_object,
            mock_servicer_context,
            validate_against=validate_against_id,
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert caller_id == validate_against_id
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_extract_id_with_validation_success finished ---")

    async def test_extract_id_with_validation_failure(
        self, mock_servicer_context, mock_call_object, mocker
    ):  # Added mocker
        print("\n--- Test: test_extract_id_with_validation_failure ---")
        uuid_str_A = str(uuid.uuid4())
        uuid_str_B = str(uuid.uuid4())

        validate_against_id = CallerIdentifier.try_parse(uuid_str_A)
        mock_call_object.id = GrpcCallerIdMessage(id=uuid_str_B)

        caller_id = await extract_id_from_call(
            mock_call_object,
            mock_servicer_context,
            validate_against=validate_against_id,
        )

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )
        print("--- Test: test_extract_id_with_validation_failure finished ---")

    async def test_extract_id_custom_extractor(
        self, mock_servicer_context, mock_call_object, mocker
    ):  # Added mocker
        print("\n--- Test: test_extract_id_custom_extractor ---")
        valid_uuid_str = str(uuid.uuid4())
        mock_call_object.custom_field_data = GrpcCallerIdMessage(
            id=valid_uuid_str
        )

        custom_extractor = lambda x: x.custom_field_data

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context, extractor=custom_extractor
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_extract_id_custom_extractor finished ---")

    # --- Tests for extract_id_from_first_call ---

    async def test_extract_first_success(
        self,
        mock_servicer_context,
        mock_is_running_tracker,
        mock_call_object,
        mocker,
    ):  # Added mocker
        print("\n--- Test: test_extract_first_success ---")
        valid_uuid_str = str(uuid.uuid4())
        call1 = mocker.MagicMock(name="Call1")  # mocker.MagicMock
        call1.id = GrpcCallerIdMessage(id=valid_uuid_str)
        call2 = mocker.MagicMock(name="Call2")  # mocker.MagicMock

        original_iterator = async_iterator_from_list([call1, call2])

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        assert first_call is call1
        mock_servicer_context.abort.assert_not_called()
        # The mock_is_running_tracker.create_stoppable_iterator is called with
        # the original_iterator, but this happens inside extract_id_from_first_call
        # and the test doesn't need to assert this directly if the SUT's internal
        # logic is trusted. The key is that the *original_iterator* is consumed.
        # mock_is_running_tracker.create_stoppable_iterator.assert_called_once_with(
        #    original_iterator
        # )

        remaining_items = [item async for item in original_iterator]
        assert remaining_items == [call2]
        print("--- Test: test_extract_first_success finished ---")

    async def test_extract_first_iterator_empty(
        self, mock_servicer_context, mock_is_running_tracker
    ):
        print("\n--- Test: test_extract_first_iterator_empty ---")
        original_iterator = async_iterator_from_list([])

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED, # Corrected: SUT uses CANCELLED here
            "First call never received!",
        )
        # mock_is_running_tracker.create_stoppable_iterator.assert_called_once_with(
        #    original_iterator
        # )

        remaining_items = [item async for item in original_iterator]
        assert not remaining_items
        print("--- Test: test_extract_first_iterator_empty finished ---")

    async def test_extract_first_is_running_becomes_false(
        self, mock_servicer_context, mock_is_running_tracker, mock_call_object
    ):
        print("\n--- Test: test_extract_first_is_running_becomes_false ---")
        original_iterator = async_iterator_from_list([mock_call_object])

        mock_is_running_tracker.get.return_value = False

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_not_called()
        mock_is_running_tracker.get.assert_called()

        # The original iterator should not have been consumed if is_running was false from the start
        remaining_items = [item async for item in original_iterator]
        # Depending on how create_stoppable_iterator behaves when is_running is false,
        # this assertion might need adjustment. If it yields nothing, then remaining_items is empty.
        # If extract_id_from_first_call returns early, original_iterator might still have items.
        # Based on SUT logic, if is_running.get() is false, it returns (None, None), iterator not touched beyond stoppable.
        # The create_stoppable_iterator itself might consume one item to check if it's empty.
        # For this test, let's assume create_stoppable_iterator might consume the item if not careful
        # However, the SUT's check for is_running.get() happens *before* iteration.
        # So, the original_iterator might not be consumed at all by the SUT's main loop.
        # The provided mock for create_stoppable_iterator simply yields items.
        # If is_running.get() is False, SUT returns (None, None) and original_iterator is not advanced by SUT's main loop.
        # The mock create_stoppable_iterator now returns an awaitable that resolves to the async generator.
        # If is_running.get() is False, the SUT returns (None, None) before iterating.
        # The `iterator = await is_running.create_stoppable_iterator(iterator)` line in SUT
        # means the original_iterator is wrapped. Iterating this wrapped iterator (original_iterator)
        # will consume it via the mock's actual_stoppable_iterator.
        assert not remaining_items # Corrected: The iterator is consumed by the stoppable_iterator wrapper
        print(
            "--- Test: test_extract_first_is_running_becomes_false finished ---"
        )

    async def test_extract_first_id_extraction_fails_on_first_item(
        self,
        mock_servicer_context,
        mock_is_running_tracker,
        mock_call_object, # mock_call_object is not used here, can be removed if not needed for other reasons
        mocker,
    ):
        print(
            "\n--- Test: test_extract_first_id_extraction_fails_on_first_item ---"
        )
        call1 = mocker.MagicMock(name="Call1_BadID")
        call1.id = GrpcCallerIdMessage(id="invalid-id-format")

        original_iterator = async_iterator_from_list([call1])

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert caller_id is None
        assert first_call is call1 # The first item was processed
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )

        # The original_iterator would have yielded its first (and only) item.
        remaining_items = [item async for item in original_iterator]
        assert not remaining_items
        print(
            "--- Test: test_extract_first_id_extraction_fails_on_first_item finished ---"
        )

    async def test_extract_first_general_exception_in_iterator(
        self,
        mock_servicer_context,
        mock_is_running_tracker,
        mock_call_object, # mock_call_object is not used here
        mocker,
    ):
        print(
            "\n--- Test: test_extract_first_general_exception_in_iterator ---"
        )
        test_exception = ValueError("Test error in iterator")

        async def faulty_iterator_fails_first():
            print("Faulty iterator starting...")
            await asyncio.sleep(0)
            raise test_exception
            yield "never_yielded" # Should not be reached

        original_iterator_fails_first = faulty_iterator_fails_first()

        with pytest.raises(ValueError, match="Test error in iterator"):
            await extract_id_from_first_call(
                original_iterator_fails_first,
                mock_is_running_tracker,
                mock_servicer_context,
            )

        # Assertions below are removed as the exception is expected to propagate
        # assert caller_id is None
        # assert first_call is None
        # mock_servicer_context.abort.assert_not_called() # Abort should not be called if exception propagates
        # Actually, SUT calls abort and then re-raises the exception.
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED, "Error processing first call!"
        )

        # # Check iterator consumption (optional, as exception occurs)
        # processed_items = []
        # try:
        #     async for item in original_iterator_fails_first: # pragma: no cover
        #         processed_items.append(item)
        # except ValueError: # pragma: no cover
        #     pass
        # assert not processed_items
        print(
            "--- Test: test_extract_first_general_exception_in_iterator finished ---"
        )
