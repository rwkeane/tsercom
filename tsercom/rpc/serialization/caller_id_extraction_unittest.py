"""Unit tests for caller ID extraction utilities."""

import asyncio
import pytest
import uuid

import grpc
from grpc.aio import ServicerContext

from tsercom.rpc.serialization.caller_id_extraction import (
    extract_id_from_call,
    extract_id_from_first_call,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.util.is_running_tracker import IsRunningTracker


class MockGrpcCallerIdMessagePlaceholder:
    def __init__(self, id=None):
        self.id = id


from tsercom.caller_id.proto import (
    CallerId as GrpcCallerIdMessage,
)


async def async_iterator_from_list(items):
    for item in items:
        await asyncio.sleep(0)
        yield item


@pytest.mark.asyncio
class TestCallerIdExtraction:

    @pytest.fixture
    def mock_servicer_context(self, mocker):
        context = mocker.AsyncMock(spec=ServicerContext)
        context.abort = mocker.AsyncMock(name="servicer_context_abort")
        context.invocation_metadata = mocker.MagicMock(return_value=[])
        return context

    @pytest.fixture
    def mock_is_running_tracker(self, mocker):
        tracker = mocker.MagicMock(spec=IsRunningTracker)

        async def actual_stoppable_iterator(iterator_arg, stop_event=None):
            async for item in iterator_arg:
                yield item

        async def mock_create_stoppable_iterator_method(
            iterator_arg, stop_event=None
        ):
            return actual_stoppable_iterator(iterator_arg, stop_event)

        tracker.create_stoppable_iterator = mocker.MagicMock(
            side_effect=mock_create_stoppable_iterator_method,
            name="mock_create_stoppable_iterator",
        )
        tracker.get = mocker.MagicMock(
            return_value=True, name="mock_is_running_get"
        )
        return tracker

    @pytest.fixture
    def mock_call_object(self, mocker):
        return mocker.MagicMock(name="MockCallObject")

    # --- Tests for extract_id_from_call ---

    async def test_extract_id_success_default_extractor(
        self, mock_servicer_context, mock_call_object, mocker
    ):
        valid_uuid_str = str(uuid.uuid4())
        mock_call_object.id = GrpcCallerIdMessage(id=valid_uuid_str)

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        mock_servicer_context.abort.assert_not_called()

    async def test_extract_id_missing_id_default_extractor(
        self, mock_servicer_context, mock_call_object
    ):
        mock_call_object.id = None

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context
        )

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID"
        )

    async def test_extract_id_call_object_has_no_id_attr(
        self, mock_servicer_context, mocker
    ):
        call_without_id_attr = mocker.MagicMock()
        call_without_id_attr.id = None

        caller_id = await extract_id_from_call(
            call_without_id_attr, mock_servicer_context
        )
        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Missing CallerID"
        )

    async def test_extract_id_malformed_id_try_parse_returns_none(
        self, mock_servicer_context, mock_call_object, mocker
    ):
        mock_call_object.id = GrpcCallerIdMessage(id="invalid-id-format")

        caller_id = await extract_id_from_call(
            mock_call_object, mock_servicer_context
        )

        assert caller_id is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )

    async def test_extract_id_with_validation_success(
        self, mock_servicer_context, mock_call_object, mocker
    ):
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

    async def test_extract_id_with_validation_failure(
        self, mock_servicer_context, mock_call_object, mocker
    ):
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

    async def test_extract_id_custom_extractor(
        self, mock_servicer_context, mock_call_object, mocker
    ):
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

    # --- Tests for extract_id_from_first_call ---

    async def test_extract_first_success(
        self,
        mock_servicer_context,
        mock_is_running_tracker,
        mock_call_object,
        mocker,
    ):
        valid_uuid_str = str(uuid.uuid4())
        call1 = mocker.MagicMock(name="Call1")
        call1.id = GrpcCallerIdMessage(id=valid_uuid_str)
        call2 = mocker.MagicMock(name="Call2")

        original_iterator = async_iterator_from_list([call1, call2])

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert isinstance(caller_id, CallerIdentifier)
        assert str(caller_id) == valid_uuid_str
        assert first_call is call1
        mock_servicer_context.abort.assert_not_called()

        remaining_items = [item async for item in original_iterator]
        assert remaining_items == [call2]

    async def test_extract_first_iterator_empty(
        self, mock_servicer_context, mock_is_running_tracker
    ):
        original_iterator = async_iterator_from_list([])

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED,
            "First call never received!",
        )

        remaining_items = [item async for item in original_iterator]
        assert not remaining_items

    async def test_extract_first_is_running_becomes_false(
        self, mock_servicer_context, mock_is_running_tracker, mock_call_object
    ):
        original_iterator = async_iterator_from_list([mock_call_object])

        mock_is_running_tracker.get.return_value = False

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert caller_id is None
        assert first_call is None
        mock_servicer_context.abort.assert_not_called()
        mock_is_running_tracker.get.assert_called()

        remaining_items = [item async for item in original_iterator]
        assert not remaining_items

    async def test_extract_first_id_extraction_fails_on_first_item(
        self,
        mock_servicer_context,
        mock_is_running_tracker,
        mock_call_object,
        mocker,
    ):
        call1 = mocker.MagicMock(name="Call1_BadID")
        call1.id = GrpcCallerIdMessage(id="invalid-id-format")

        original_iterator = async_iterator_from_list([call1])

        caller_id, first_call = await extract_id_from_first_call(
            original_iterator, mock_is_running_tracker, mock_servicer_context
        )

        assert caller_id is None
        assert first_call is call1
        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT, "Invalid CallerID received"
        )

        remaining_items = [item async for item in original_iterator]
        assert not remaining_items

    async def test_extract_first_general_exception_in_iterator(
        self,
        mock_servicer_context,
        mock_is_running_tracker,
        mock_call_object,
        mocker,
    ):
        test_exception = ValueError("Test error in iterator")

        async def faulty_iterator_fails_first():
            await asyncio.sleep(0)
            raise test_exception
            yield "never_yielded"

        original_iterator_fails_first = faulty_iterator_fails_first()

        with pytest.raises(ValueError, match="Test error in iterator"):
            await extract_id_from_first_call(
                original_iterator_fails_first,
                mock_is_running_tracker,
                mock_servicer_context,
            )

        mock_servicer_context.abort.assert_called_once_with(
            grpc.StatusCode.CANCELLED, "Error processing first call!"
        )
