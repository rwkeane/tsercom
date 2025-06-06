"""Unit tests for EventToSerializableAnnInstancePollerAdapter."""

import asyncio
import datetime
from unittest.mock import Mock, AsyncMock

import pytest

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.event_instance import EventInstance
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.runtime.event_poller_adapter import (
    EventToSerializableAnnInstancePollerAdapter,
)
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


@pytest.fixture
def mock_source_poller():
    """Fixture for a mock source poller."""
    return AsyncMock(spec=AsyncPoller)


@pytest.fixture
def adapter(mock_source_poller):
    """Fixture for the EventToSerializableAnnInstancePollerAdapter."""
    return EventToSerializableAnnInstancePollerAdapter(mock_source_poller)


class TestEventToSerializableAnnInstancePollerAdapter:
    """Tests for EventToSerializableAnnInstancePollerAdapter."""

    def test_convert_event_instance_with_caller_id(self, adapter):
        """Test conversion when EventInstance has a CallerIdentifier."""
        caller_id = CallerIdentifier.random()
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        event_data = "test_event_data"
        event_inst = EventInstance(
            data=event_data, caller_id=caller_id, timestamp=timestamp
        )

        result = adapter._convert_event_instance(event_inst)

        assert isinstance(result, SerializableAnnotatedInstance)
        assert result.data == event_data
        assert result.caller_id == caller_id
        assert isinstance(result.timestamp, SynchronizedTimestamp)
        # Check if timestamp conversion is reasonable (e.g., same microsecond)
        # This depends on how SynchronizedTimestamp internally handles datetime
        assert result.timestamp.as_datetime().replace(
            tzinfo=datetime.timezone.utc
        ) == timestamp.replace(tzinfo=datetime.timezone.utc)

    def test_convert_event_instance_with_none_caller_id(self, adapter):
        """Test conversion when EventInstance has caller_id=None."""
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        event_data = "test_event_data_none_id"
        event_inst = EventInstance(
            data=event_data, caller_id=None, timestamp=timestamp
        )

        # This should no longer raise ValueError
        result = adapter._convert_event_instance(event_inst)

        assert isinstance(result, SerializableAnnotatedInstance)
        assert result.data == event_data
        assert result.caller_id is None
        assert isinstance(result.timestamp, SynchronizedTimestamp)
        assert result.timestamp.as_datetime().replace(
            tzinfo=datetime.timezone.utc
        ) == timestamp.replace(tzinfo=datetime.timezone.utc)

    @pytest.mark.asyncio
    async def test_anext_calls_convert(self, adapter, mock_source_poller):
        """Test that __anext__ calls _convert_event_instance for each event."""
        caller_id1 = CallerIdentifier.random()
        caller_id2 = None
        ts1 = datetime.datetime.now(datetime.timezone.utc)
        ts2 = ts1 + datetime.timedelta(seconds=1)

        event_inst1 = EventInstance("data1", caller_id1, ts1)
        event_inst2 = EventInstance("data2", caller_id2, ts2)

        mock_source_poller.__anext__.return_value = [event_inst1, event_inst2]

        # Mock _convert_event_instance to check its calls
        adapter._convert_event_instance = Mock(
            wraps=adapter._convert_event_instance
        )

        result_list = await adapter.__anext__()

        assert len(result_list) == 2
        adapter._convert_event_instance.assert_any_call(event_inst1)
        adapter._convert_event_instance.assert_any_call(event_inst2)

        assert result_list[0].data == "data1"
        assert result_list[0].caller_id == caller_id1
        assert result_list[1].data == "data2"
        assert result_list[1].caller_id is None

    @pytest.mark.asyncio
    async def test_aiter_returns_self(self, adapter):
        """Test that __aiter__ returns the adapter itself."""
        assert adapter.__aiter__() is adapter

    # Consider adding a test for an empty list from source_poller if important
    @pytest.mark.asyncio
    async def test_anext_empty_list_from_source(
        self, adapter, mock_source_poller
    ):
        """Test __anext__ with an empty list from the source poller."""
        mock_source_poller.__anext__.return_value = []
        result_list = await adapter.__anext__()
        assert len(result_list) == 0
        # Ensure _convert_event_instance was not called
        adapter._convert_event_instance = Mock()
        adapter._convert_event_instance.assert_not_called()

# Minimal test for SerializableAnnotatedInstance constructor changes
# This should ideally be in its own test file if one existed.
def test_serializable_annotated_instance_optional_caller_id():
    """Test SerializableAnnotatedInstance with and without caller_id."""
    data = "test_data"
    cid = CallerIdentifier.random()
    # Create a valid SynchronizedTimestamp for testing
    current_dt = datetime.datetime.now(datetime.timezone.utc)
    ts = SynchronizedTimestamp(current_dt)

    # With CallerIdentifier
    inst1 = SerializableAnnotatedInstance(data=data, caller_id=cid, timestamp=ts)
    assert inst1.caller_id == cid

    # With None caller_id
    inst2 = SerializableAnnotatedInstance(data=data, caller_id=None, timestamp=ts)
    assert inst2.caller_id is None
