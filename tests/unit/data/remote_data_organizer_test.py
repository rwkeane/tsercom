import datetime
import functools
import logging  # Ensure this is present for caplog
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Union  # Added List, Optional, Union

import pytest
from pytest_mock import MockerFixture
from sortedcontainers import SortedList

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.util.is_running_tracker import IsRunningTracker


class DummyCallerIdentifier(CallerIdentifier):
    def __init__(self, id_str: str) -> None:
        self.id_str = id_str

    def __hash__(self) -> int:
        return hash(self.id_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DummyCallerIdentifier):
            return NotImplemented
        return self.id_str == other.id_str

    def __repr__(self) -> str:
        return f"DummyCallerIdentifier('{self.id_str}')"

    @property
    def name(self) -> str:
        return self.id_str


class DummyExposedDataForOrganizerTests(ExposedData):
    __test__ = False
    data: float  # Changed from value to data to match interpolation logic expectations

    def __init__(
        self,
        caller_id: DummyCallerIdentifier,
        timestamp: datetime.datetime,
        data: float = 0.0,  # Changed from value to data
    ) -> None:
        super().__init__(caller_id, timestamp)
        self.data = data  # Changed from value to data

    def __repr__(self) -> str:
        caller_name = getattr(self.caller_id, "name", str(self.caller_id))
        data_repr = getattr(self, "data", "N/A")  # Changed from value to data
        return f"DummyExposedDataForOrganizerTests(caller_id='{caller_name}', timestamp='{self.timestamp}', data={data_repr})"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DummyExposedDataForOrganizerTests):
            return NotImplemented
        # Ensure comparison involves self.data for consistent sorting if timestamps are equal
        if self.timestamp == other.timestamp:
            # Assuming self.data is comparable, which it is (float)
            return self.data < other.data  # Changed from value to data
        return self.timestamp < other.timestamp


@pytest.fixture
def mock_thread_pool(mocker: MockerFixture) -> ThreadPoolExecutor:
    mock_pool = mocker.MagicMock(spec=ThreadPoolExecutor)

    def immediate_submit(
        func_to_call: functools.partial[Any] | Callable[..., Any],
        *args_for_func: Any,
        **kwargs_for_func: Any,
    ) -> Any:
        if isinstance(func_to_call, functools.partial):
            return func_to_call()  # No args needed if already in partial
        return func_to_call(*args_for_func, **kwargs_for_func)

    mock_pool.submit = mocker.MagicMock(side_effect=immediate_submit)
    return mock_pool


@pytest.fixture
def mock_caller_id() -> DummyCallerIdentifier:
    return DummyCallerIdentifier("test_caller_1")


@pytest.fixture
def mock_client(mocker: MockerFixture) -> RemoteDataOrganizer.Client:
    # Correctly mock the abstract Client class
    # Create a concrete mock that implements the abstract method
    class MockClientImpl(RemoteDataOrganizer.Client):
        def _on_data_available(
            self,
            data_organizer: "RemoteDataOrganizer[DummyExposedDataForOrganizerTests]",
        ) -> None:
            pass  # Mock implementation, or use mocker.MagicMock() if methods need to be tracked

    # Mock the _on_data_available method on an instance of the concrete mock
    mock_client_instance = MockClientImpl()
    mocker.patch.object(mock_client_instance, "_on_data_available")  # type: ignore[attr-defined]
    return mock_client_instance


@pytest.fixture
def mock_is_running_tracker(mocker: MockerFixture) -> IsRunningTracker:
    mock_tracker_instance = mocker.MagicMock(spec=IsRunningTracker)
    mock_tracker_instance.get.return_value = True  # type: ignore[attr-defined]
    # Patch the constructor of IsRunningTracker to return our mock instance
    mocker.patch(
        "tsercom.data.remote_data_organizer.IsRunningTracker",
        return_value=mock_tracker_instance,
    )
    return mock_tracker_instance


@pytest.fixture
def organizer(
    mock_thread_pool: ThreadPoolExecutor,
    mock_caller_id: DummyCallerIdentifier,
    mock_client: RemoteDataOrganizer.Client,  # Use the mocked client
    mock_is_running_tracker: IsRunningTracker,  # Ensure this is passed if used by constructor
) -> RemoteDataOrganizer[DummyExposedDataForOrganizerTests]:
    org = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=mock_client,  # Pass the client here
    )
    # Ensure __is_running is properly initialized if start() isn't called by default in constructor
    # or if the mock_is_running_tracker isn't automatically used.
    # Based on RemoteDataOrganizer, IsRunningTracker is instantiated internally.
    # The mock_is_running_tracker fixture already patches the class, so new instances will be that mock.
    org.start()  # Call start to set the internal IsRunningTracker's state via the mock
    return org


@pytest.fixture
def organizer_no_client(
    mock_thread_pool: ThreadPoolExecutor,
    mock_caller_id: DummyCallerIdentifier,
    mock_is_running_tracker: IsRunningTracker,  # Ensure this is passed
) -> RemoteDataOrganizer[DummyExposedDataForOrganizerTests]:
    org = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=None,  # Explicitly no client
    )
    org.start()
    return org


BASE_TIME_UTC: datetime.datetime = datetime.datetime(
    2023,
    1,
    1,
    0,
    0,
    0,
    tzinfo=datetime.timezone.utc,
)


def create_data(
    caller_id: DummyCallerIdentifier,
    timestamp_input: Union[int, float, datetime.datetime],
    data_value: float = 0.0,  # Renamed from value_id for clarity with DummyExposedData.data
) -> DummyExposedDataForOrganizerTests:
    ts: datetime.datetime
    if isinstance(timestamp_input, (int, float)):
        ts = BASE_TIME_UTC + datetime.timedelta(seconds=timestamp_input)
    elif isinstance(
        timestamp_input,
        type(BASE_TIME_UTC),  # Use type() for mock-proof datetime check
    ):
        ts = timestamp_input
        if ts.tzinfo is None:  # Ensure timezone awareness
            ts = ts.replace(tzinfo=datetime.timezone.utc)
    else:
        msg = f"timestamp_input must be int, float, or datetime, got {type(timestamp_input)}"
        raise TypeError(msg)

    return DummyExposedDataForOrganizerTests(
        caller_id=caller_id,
        timestamp=ts,
        data=data_value,  # Pass to 'data' parameter of DummyExposedData
    )


# --- Test Cases ---
def test_initialization(
    mock_thread_pool: ThreadPoolExecutor,
    mock_caller_id: DummyCallerIdentifier,
    mock_client: RemoteDataOrganizer.Client,  # Use the fixture
    # mock_is_running_tracker is not directly passed to constructor, it's patched
) -> None:
    # Need IsRunningTracker to be patched for this test too if it's created inside RemoteDataOrganizer
    # The mock_is_running_tracker fixture (even if not explicitly in args here) should handle this if it's auto-used.
    # To be safe, let's ensure it's active by adding it as an argument to the test.
    # However, the organizer fixture already handles the IsRunningTracker patching.
    # We are testing the constructor directly here.

    # Re-patch IsRunningTracker for this specific test's scope if not using the 'organizer' fixture
    # This ensures that the RemoteDataOrganizer called here uses a MagicMock for IsRunningTracker
    mocker_fixture = MockerFixture(
        None
    )  # Create a local MockerFixture instance for this test.
    mock_irt_instance = mocker_fixture.MagicMock(spec=IsRunningTracker)
    mock_irt_instance.get.return_value = True  # type: ignore[attr-defined]
    mocker_fixture.patch(
        "tsercom.data.remote_data_organizer.IsRunningTracker",
        return_value=mock_irt_instance,
    )

    organizer_instance = RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=mock_client,
    )
    assert organizer_instance._RemoteDataOrganizer__thread_pool is mock_thread_pool  # type: ignore[attr-defined]
    assert organizer_instance.caller_id is mock_caller_id
    assert organizer_instance._RemoteDataOrganizer__client is mock_client  # type: ignore[attr-defined]
    assert isinstance(organizer_instance._RemoteDataOrganizer__data, SortedList)  # type: ignore[attr-defined]
    assert len(organizer_instance._RemoteDataOrganizer__data) == 0  # type: ignore[attr-defined]
    # Verify __last_access initialization
    expected_min_time = datetime.datetime.min.replace(
        tzinfo=datetime.timezone.utc
    )
    assert organizer_instance._RemoteDataOrganizer__last_access == expected_min_time  # type: ignore[attr-defined]
    # Check that the internally created IsRunningTracker is the patched one
    assert organizer_instance._RemoteDataOrganizer__is_running is mock_irt_instance  # type: ignore[attr-defined]


def test_start(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    # mock_is_running_tracker is implicitly used by the 'organizer' fixture
) -> None:
    # The 'organizer' fixture already calls start().
    # We want to test the effect of start() on the mock_is_running_tracker.
    # The mock_is_running_tracker fixture patches the class, so organizer.__is_running IS the mock.
    mock_irt = organizer._RemoteDataOrganizer__is_running  # type: ignore[attr-defined]
    mock_irt.start.reset_mock()  # type: ignore[attr-defined]
    organizer.start()  # Call it again to check the mock
    mock_irt.start.assert_called_once()  # type: ignore[attr-defined]


def test_stop(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
) -> None:
    mock_irt = organizer._RemoteDataOrganizer__is_running  # type: ignore[attr-defined]
    organizer.stop()
    mock_irt.stop.assert_called_once()  # type: ignore[attr-defined]


def test_on_data_ready_submits_to_thread_pool(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_thread_pool: ThreadPoolExecutor,  # Fixture for the thread pool
    mock_caller_id: DummyCallerIdentifier,  # Fixture for the caller ID
) -> None:
    data = create_data(mock_caller_id, 0, data_value=1.0)
    mock_thread_pool.submit.reset_mock()  # type: ignore[attr-defined]
    # Call the public method that should use the thread pool
    organizer._on_data_ready(data)  # type: ignore[attr-defined]
    # Check that submit was called with the correct internal method and data
    mock_thread_pool.submit.assert_called_once_with(  # type: ignore[attr-defined]
        organizer._RemoteDataOrganizer__on_data_ready_impl, data  # type: ignore[attr-defined]
    )


def test_on_data_ready_impl_not_running(
    organizer_no_client: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    # Ensure the organizer's IsRunningTracker mock returns False
    mock_irt = organizer_no_client._RemoteDataOrganizer__is_running  # type: ignore[attr-defined]
    mock_irt.get.return_value = False  # type: ignore[attr-defined]

    data = create_data(mock_caller_id, 0, data_value=1.0)
    # Call the internal implementation method directly for this test
    organizer_no_client._RemoteDataOrganizer__on_data_ready_impl(data)  # type: ignore[attr-defined]

    # Assert that data was not added because the organizer is "not running"
    assert len(organizer_no_client._RemoteDataOrganizer__data) == 0  # type: ignore[attr-defined]


def test_on_data_ready_impl_adds_to_empty_list_and_notifies(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
    mock_client: RemoteDataOrganizer.Client,  # Get the mocked client
) -> None:
    data = create_data(mock_caller_id, 0, data_value=1.0)

    # Reset mock before action, if client methods are called by other setup parts
    # mock_client itself is a MagicMock if methods are directly on it, or its methods are mocks.
    # The fixture now makes _on_data_available a MagicMock on the instance.
    getattr(mock_client, "_on_data_available").reset_mock()  # type: ignore[attr-defined]

    organizer._RemoteDataOrganizer__on_data_ready_impl(data)  # type: ignore[attr-defined]

    assert len(organizer._RemoteDataOrganizer__data) == 1  # type: ignore[attr-defined]
    assert organizer._RemoteDataOrganizer__data[0] == data  # type: ignore[attr-defined]

    # Verify that the client was notified
    getattr(mock_client, "_on_data_available").assert_called_once_with(organizer)  # type: ignore[attr-defined]


def test_on_data_ready_impl_adds_data_maintaining_sort_order(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
    mock_client: RemoteDataOrganizer.Client,
) -> None:
    data_t1 = create_data(mock_caller_id, 10, data_value=1.0)  # timestamp 10
    data_t2 = create_data(mock_caller_id, 20, data_value=2.0)  # timestamp 20

    getattr(mock_client, "_on_data_available").reset_mock()  # type: ignore[attr-defined]

    # Add in reverse order of timestamp to test sorting
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)  # type: ignore[attr-defined]
    # For SortedList, client should be notified on each add if it's the latest.
    # Let's refine this: notification should happen if it's the latest OR an update.
    # The logic in __on_data_ready_impl for item_is_latest handles this.
    # For this test, we primarily care about the final state and one notification.
    # We can reset and check after all additions if needed, but SortedList handles order.

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)  # type: ignore[attr-defined]

    assert len(organizer._RemoteDataOrganizer__data) == 2  # type: ignore[attr-defined]
    assert organizer._RemoteDataOrganizer__data[0] == data_t1  # type: ignore[attr-defined] # Should be the one with ts=10
    assert organizer._RemoteDataOrganizer__data[1] == data_t2  # type: ignore[attr-defined] # Should be the one with ts=20

    # Check client notification. It should be called after each add that qualifies.
    # If data_t2 makes it the latest, it's called. If data_t1 is older, it's still added.
    # The original code notifies if 'item_is_latest' OR if it's an update to an existing timestamp.
    # When data_t2 (ts=20) is added to empty, it's latest. Client notified.
    # When data_t1 (ts=10) is added, it's not latest. Client not notified for this specific add by that condition.
    # Let's check: data_t2 is latest. Notification. data_t1 is not. No notification.
    # This means assert_called_once_with might be too strict if multiple calls happen.
    # The original code has `data_inserted_or_updated` flag.
    # If data_t2 is added: data_inserted_or_updated = True. Notify.
    # If data_t1 is added (it's older): data_inserted_or_updated = False (item_is_latest is False for t1). Notify not called.
    # So, it should be called once.
    getattr(mock_client, "_on_data_available").assert_called_once_with(organizer)  # type: ignore[attr-defined]


def test_on_data_ready_impl_replaces_data_with_same_timestamp(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
    mock_client: RemoteDataOrganizer.Client,
) -> None:
    ts = 10
    data_initial = create_data(mock_caller_id, ts, data_value=1.0)
    data_replacement = create_data(
        mock_caller_id, ts, data_value=2.0
    )  # Same ts, new data

    # Add initial data
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_initial)  # type: ignore[attr-defined]
    getattr(mock_client, "_on_data_available").reset_mock()  # type: ignore[attr-defined]

    # Add replacement data
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_replacement)  # type: ignore[attr-defined]

    assert len(organizer._RemoteDataOrganizer__data) == 1  # type: ignore[attr-defined]
    # The new data should replace the old one.
    # Accessing data property of the item in SortedList:
    # (organizer._RemoteDataOrganizer__data[0] is DummyExposedDataForOrganizerTests)
    assert organizer._RemoteDataOrganizer__data[0].data == 2.0  # type: ignore[attr-defined]
    assert organizer._RemoteDataOrganizer__data[0] == data_replacement  # type: ignore[attr-defined]

    # Client should be notified of the update
    getattr(mock_client, "_on_data_available").assert_called_once_with(organizer)  # type: ignore[attr-defined]


def test_has_new_data_empty(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
) -> None:
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    # Set __last_access to a specific time, e.g., BASE_TIME_UTC or current time
    organizer._RemoteDataOrganizer__last_access = BASE_TIME_UTC  # type: ignore[attr-defined]
    assert organizer.has_new_data() is False


def test_has_new_data_all_accessed(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    ts_now = BASE_TIME_UTC + datetime.timedelta(seconds=10)
    # Data older than or same as __last_access
    data_old = create_data(
        mock_caller_id, ts_now - datetime.timedelta(seconds=1), data_value=1.0
    )

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data_old)  # type: ignore[attr-defined]
    # __last_access is more recent than or same as the data's timestamp
    organizer._RemoteDataOrganizer__last_access = ts_now  # type: ignore[attr-defined]

    assert organizer.has_new_data() is False


def test_has_new_data_new_available(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    data_new_ts = BASE_TIME_UTC + datetime.timedelta(seconds=10)
    data_new = create_data(mock_caller_id, data_new_ts, data_value=1.0)

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data_new)  # type: ignore[attr-defined]
    # __last_access is older than the new data's timestamp
    organizer._RemoteDataOrganizer__last_access = data_new_ts - datetime.timedelta(seconds=1)  # type: ignore[attr-defined]

    assert organizer.has_new_data() is True


def test_get_new_data_empty(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
) -> None:
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    assert organizer.get_new_data() == []


def test_get_new_data_retrieves_newer_updates_last_access(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    # Timestamps for data items
    ts0 = BASE_TIME_UTC
    ts1 = BASE_TIME_UTC + datetime.timedelta(seconds=10)
    ts2 = BASE_TIME_UTC + datetime.timedelta(seconds=20)

    data0 = create_data(mock_caller_id, ts0, data_value=0.0)
    data1 = create_data(mock_caller_id, ts1, data_value=1.0)
    data2 = create_data(mock_caller_id, ts2, data_value=2.0)

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data0)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data1)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data2)  # type: ignore[attr-defined]

    # Set __last_access to the timestamp of data1
    organizer._RemoteDataOrganizer__last_access = data1.timestamp  # type: ignore[attr-defined]

    new_data_list: List[DummyExposedDataForOrganizerTests] = organizer.get_new_data()  # type: ignore[assignment]

    # Should only retrieve data2 (since data0 and data1 are <= __last_access)
    # The method returns in reverse chronological order (most recent first)
    assert len(new_data_list) == 1
    assert new_data_list[0] == data2  # data2 is the newest

    # __last_access should be updated to the timestamp of the most recent item retrieved (data2)
    assert organizer._RemoteDataOrganizer__last_access == data2.timestamp  # type: ignore[attr-defined]


def test_get_most_recent_data_empty(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
) -> None:
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    assert organizer.get_most_recent_data() is None


def test_get_most_recent_data_returns_last_item_in_sortedlist(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    data1 = create_data(mock_caller_id, 0, data_value=1.0)  # ts = 0
    data2 = create_data(
        mock_caller_id, 1, data_value=2.0
    )  # ts = 1 (most recent)

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data1)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data2)  # type: ignore[attr-defined]

    most_recent = organizer.get_most_recent_data()
    assert most_recent is not None
    assert most_recent == data2


def test_get_data_for_timestamp_empty(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
) -> None:
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    assert organizer.get_data_for_timestamp(BASE_TIME_UTC) is None


def test_get_data_for_timestamp_older_than_all(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    # Data starts at ts=10
    data = create_data(mock_caller_id, 10, data_value=1.0)
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data)  # type: ignore[attr-defined]

    # Query for ts=5 (older than any data)
    ts_query = BASE_TIME_UTC + datetime.timedelta(seconds=5)
    assert organizer.get_data_for_timestamp(ts_query) is None


def test_get_data_for_timestamp_finds_correct_item(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    # Data at ts=0, ts=10, ts=20
    data1 = create_data(mock_caller_id, 0, data_value=1.0)  # ts=0
    data2 = create_data(mock_caller_id, 10, data_value=2.0)  # ts=10
    data3 = create_data(mock_caller_id, 20, data_value=3.0)  # ts=20

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data1)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data2)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data3)  # type: ignore[attr-defined]

    # Query at ts=15. Should return data2 (floor behavior: item with largest ts <= query_ts)
    query_ts_15 = BASE_TIME_UTC + datetime.timedelta(seconds=15)
    assert organizer.get_data_for_timestamp(query_ts_15) == data2

    # Query for exact timestamp of data2 (ts=10)
    assert organizer.get_data_for_timestamp(data2.timestamp) == data2

    # Query at ts=25. Should return data3
    query_ts_25 = BASE_TIME_UTC + datetime.timedelta(seconds=25)
    assert organizer.get_data_for_timestamp(query_ts_25) == data3

    # Query for exact timestamp of data1 (ts=0)
    assert organizer.get_data_for_timestamp(data1.timestamp) == data1


def test_timeout_old_data_removes_old_items(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
    mocker: MockerFixture,  # For mocking datetime.now()
) -> None:
    # Mock current time for consistent testing of timeout logic
    # Let current time be ts=100
    current_time_mock_val = BASE_TIME_UTC + datetime.timedelta(seconds=100)

    # Patch datetime.datetime.now specifically within the remote_data_organizer module
    datetime_now_mock = mocker.patch(
        "tsercom.data.remote_data_organizer.datetime.datetime"
    )
    datetime_now_mock.now.return_value = current_time_mock_val

    timeout_seconds = 30
    # Items older than ts=70 (100-30) should be removed.
    # ts_old is 69, so it should be removed.
    # ts_kept1 is 70, so it should be kept.
    # ts_kept2 is 80, so it should be kept.

    ts_old = BASE_TIME_UTC + datetime.timedelta(
        seconds=69
    )  # Should be removed
    ts_kept1 = BASE_TIME_UTC + datetime.timedelta(seconds=70)  # Should be kept
    ts_kept2 = BASE_TIME_UTC + datetime.timedelta(seconds=80)  # Should be kept

    data_old = create_data(mock_caller_id, ts_old, data_value=1.0)
    data_kept1 = create_data(mock_caller_id, ts_kept1, data_value=2.0)
    data_kept2 = create_data(mock_caller_id, ts_kept2, data_value=3.0)

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data_old)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data_kept1)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(data_kept2)  # type: ignore[attr-defined]

    # Call the internal method that performs the timeout logic
    organizer._RemoteDataOrganizer__timeout_old_data(timeout_seconds)  # type: ignore[attr-defined]

    assert len(organizer._RemoteDataOrganizer__data) == 2  # type: ignore[attr-defined]
    assert data_old not in organizer._RemoteDataOrganizer__data  # type: ignore[attr-defined]
    assert data_kept1 in organizer._RemoteDataOrganizer__data  # type: ignore[attr-defined]
    assert data_kept2 in organizer._RemoteDataOrganizer__data  # type: ignore[attr-defined]
    # Check order as well
    assert list(organizer._RemoteDataOrganizer__data) == [data_kept1, data_kept2]  # type: ignore[attr-defined]


def test_on_data_ready_impl_inserts_out_of_order_data_correctly_sortedlist(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
    # mock_client is not strictly needed here as we are testing data state, not notifications
) -> None:
    # Timestamps: 2, 4, 3, 1
    data_t2 = create_data(mock_caller_id, 2, data_value=2.0)
    data_t4 = create_data(mock_caller_id, 4, data_value=4.0)
    data_t3 = create_data(mock_caller_id, 3, data_value=3.0)
    data_t1 = create_data(mock_caller_id, 1, data_value=1.0)

    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]

    # Insert in a jumbled order
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t4)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)  # type: ignore[attr-defined]

    # Expected order: t1, t2, t3, t4
    expected_order = [data_t1, data_t2, data_t3, data_t4]
    actual_order = list(organizer._RemoteDataOrganizer__data)  # type: ignore[attr-defined]

    assert (
        actual_order == expected_order
    ), f"Data not in expected sorted order. Got: {actual_order}"


# --- New Fixture and Tests for get_interpolated_at ---


@pytest.fixture
def organizer_for_interpolation_tests(
    # Cannot use 'organizer' fixture directly if we need to ensure it's clean and set up specifically.
    # Re-create a similar setup or ensure 'organizer' is appropriately reset.
    # Using the existing 'organizer' fixture should be fine if we clear its data.
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,  # For creating data
) -> RemoteDataOrganizer[DummyExposedDataForOrganizerTests]:
    # Clear any data that might be there from other test setups using the same 'organizer'
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]

    # Populate with some data. Values are floats.
    # create_data uses 'data_value' for the 'data' attribute of DummyExposedDataForOrganizerTests
    organizer._RemoteDataOrganizer__data.add(create_data(mock_caller_id, 10, data_value=100.0))  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(create_data(mock_caller_id, 20, data_value=200.0))  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(create_data(mock_caller_id, 30, data_value=300.0))  # type: ignore[attr-defined]
    # Data: [ (ts10,data100), (ts20,data200), (ts30,data300) ]
    return organizer


# Test cases for get_interpolated_at


def test_interpolate_empty_data(
    organizer: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],  # A fresh organizer will be empty
) -> None:
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined] # Ensure it's empty
    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=15)
    assert organizer.get_interpolated_at(query_ts) is None


def test_interpolate_before_first_keyframe(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
) -> None:
    # Data in fixture starts at ts=10, data=100.0
    query_ts = BASE_TIME_UTC + datetime.timedelta(
        seconds=5
    )  # Query before first keyframe

    interpolated = organizer_for_interpolation_tests.get_interpolated_at(
        query_ts
    )
    assert interpolated is not None

    # Behavior: returns a deepcopy of the first item
    expected_first_item = organizer_for_interpolation_tests._RemoteDataOrganizer__data[0]  # type: ignore[attr-defined]
    assert (
        interpolated.timestamp == expected_first_item.timestamp
    )  # Should be ts of first item
    assert interpolated.data == expected_first_item.data  # type: ignore[attr-defined]
    assert interpolated is not expected_first_item  # Must be a deepcopy


def test_interpolate_after_last_keyframe(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
) -> None:
    # Data in fixture ends at ts=30, data=300.0
    query_ts = BASE_TIME_UTC + datetime.timedelta(
        seconds=35
    )  # Query after last keyframe

    interpolated = organizer_for_interpolation_tests.get_interpolated_at(
        query_ts
    )
    assert interpolated is not None

    # Behavior: returns a deepcopy of the last item
    expected_last_item = organizer_for_interpolation_tests._RemoteDataOrganizer__data[-1]  # type: ignore[attr-defined]
    assert (
        interpolated.timestamp == expected_last_item.timestamp
    )  # Should be ts of last item
    assert interpolated.data == expected_last_item.data  # type: ignore[attr-defined]
    assert interpolated is not expected_last_item  # Must be a deepcopy


def test_interpolate_exact_match_keyframe(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
) -> None:
    # Query for ts=20, which is an exact match for (ts20, data200)
    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=20)

    interpolated = organizer_for_interpolation_tests.get_interpolated_at(
        query_ts
    )
    assert interpolated is not None

    # Behavior: returns a deepcopy of the matched item.
    # The timestamp of the *returned* item should be the query_ts itself.
    # The data should be that of the matched keyframe.
    matched_item_original = organizer_for_interpolation_tests._RemoteDataOrganizer__data[1]  # type: ignore[attr-defined] # This is (ts20, data200)
    assert (
        interpolated.timestamp == query_ts
    )  # Timestamp is the query timestamp
    assert interpolated.data == matched_item_original.data  # type: ignore[attr-defined] # Data from original keyframe
    assert interpolated is not matched_item_original  # Must be a deepcopy


def test_interpolate_successful_between_two_points(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
    mock_caller_id: DummyCallerIdentifier,  # To check the caller_id of the interpolated item
) -> None:
    # Test 1: Midpoint interpolation
    # Query at ts=15 (between (ts10,data100) and (ts20,data200))
    # Expected data: 100 + 0.5 * (200-100) = 150
    query_ts_1 = BASE_TIME_UTC + datetime.timedelta(seconds=15)
    interpolated_1 = organizer_for_interpolation_tests.get_interpolated_at(
        query_ts_1
    )

    assert interpolated_1 is not None
    assert (
        interpolated_1.timestamp == query_ts_1
    )  # Timestamp of the new interpolated item
    assert (
        interpolated_1.caller_id == mock_caller_id
    )  # Should be copied from neighbors
    assert pytest.approx(interpolated_1.data) == 150.0  # type: ignore[attr-defined]

    # Test 2: Interpolation at a specific ratio
    # Query at ts=27 (between (ts20,data200) and (ts30,data300))
    # Time difference from left (ts20): 27 - 20 = 7 seconds
    # Total interval: 30 - 20 = 10 seconds
    # Ratio: 7 / 10 = 0.7
    # Expected data: 200 + 0.7 * (300-200) = 200 + 0.7 * 100 = 200 + 70 = 270
    query_ts_2 = BASE_TIME_UTC + datetime.timedelta(seconds=27)
    interpolated_2 = organizer_for_interpolation_tests.get_interpolated_at(
        query_ts_2
    )

    assert interpolated_2 is not None
    assert interpolated_2.timestamp == query_ts_2
    assert pytest.approx(interpolated_2.data) == 270.0  # type: ignore[attr-defined]


def test_interpolate_with_identical_bracketing_keyframes_data(
    # Use a fresh organizer instance to set up specific data
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    organizer._RemoteDataOrganizer__data.clear()  # type: ignore[attr-defined]

    # Keyframes with identical data values
    item1 = create_data(
        mock_caller_id, 10, data_value=100.0
    )  # (ts10, data100)
    item2 = create_data(
        mock_caller_id, 20, data_value=100.0
    )  # (ts20, data100) - same data
    organizer._RemoteDataOrganizer__data.add(item1)  # type: ignore[attr-defined]
    organizer._RemoteDataOrganizer__data.add(item2)  # type: ignore[attr-defined]

    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=15)  # Midpoint query
    interpolated = organizer.get_interpolated_at(query_ts)

    assert interpolated is not None
    assert interpolated.timestamp == query_ts
    # Interpolated data should be 100.0 (since both keyframes have 100.0)
    assert pytest.approx(interpolated.data) == 100.0  # type: ignore[attr-defined]


def test_interpolate_data_type_not_supporting_arithmetic(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
    caplog: pytest.LogCaptureFixture,  # To check log messages
) -> None:
    # The .data attribute in DummyExposedDataForOrganizerTests is float.
    # To test this, we need to simulate a situation where item_left.data or item_right.data
    # is not numeric. This requires modifying the items in the __data list directly,
    # or mocking getattr. Direct modification is simpler for this test.

    # Get references to actual items in the list for modification.
    # These are DummyExposedDataForOrganizerTests instances.
    item_to_corrupt_1 = organizer_for_interpolation_tests._RemoteDataOrganizer__data[0]  # type: ignore[attr-defined] # (ts10, data100)
    item_to_corrupt_2 = organizer_for_interpolation_tests._RemoteDataOrganizer__data[1]  # type: ignore[attr-defined] # (ts20, data200)

    original_data_1 = item_to_corrupt_1.data  # Store original float data
    original_data_2 = item_to_corrupt_2.data

    try:
        # Temporarily change 'data' attribute to a non-numeric type (e.g., string)
        # This is a bit hacky as DummyExposedDataForOrganizerTests expects float.
        # It relies on Python's dynamic typing to allow this change at runtime for the test.
        # We use setattr because 'data' is defined in the class.
        setattr(item_to_corrupt_1, "data", "non_numeric_string1")  # type: ignore[arg-type]
        # Keep item_to_corrupt_2.data as float to test one side non-numeric
        # Or change both: setattr(item_to_corrupt_2, 'data', "non_numeric_string2")

        query_ts = BASE_TIME_UTC + datetime.timedelta(
            seconds=15
        )  # Query between item 0 and 1

        caplog.clear()  # Clear any previous logs
        with caplog.at_level(
            logging.ERROR
        ):  # Ensure logger captures ERROR level
            interpolated = (
                organizer_for_interpolation_tests.get_interpolated_at(query_ts)
            )

        assert (
            interpolated is None
        )  # Should return None if interpolation fails due to type error

        # Check for the specific log message from RemoteDataOrganizer
        # The message is "Data payloads for interpolation are not numeric..."
        assert any(
            "Data payloads for interpolation are not numeric" in record.message
            for record in caplog.records
        ), "Expected log message about non-numeric data not found."

    finally:
        # Restore original data to ensure the fixture remains clean for other tests
        setattr(item_to_corrupt_1, "data", original_data_1)
        setattr(item_to_corrupt_2, "data", original_data_2)
