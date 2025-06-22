import pytest
import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import functools
import re
from sortedcontainers import SortedList
import logging

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.util.is_running_tracker import IsRunningTracker

BASE_TIME_UTC = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)


# Assuming CallerIdentifier.py is missing, define a functional dummy for tests.
class DummyCallerIdentifier:
    def __init__(self, id_str: str):
        self.id_str = id_str

    def __hash__(self):
        return hash(self.id_str)

    def __eq__(self, other):
        return isinstance(other, DummyCallerIdentifier) and self.id_str == other.id_str

    def __repr__(self):
        return f"DummyCallerIdentifier('{self.id_str}')"

    @property
    def name(self) -> str:
        return self.id_str


# --- Concrete Dummy ExposedData for type checks and usage ---
class DummyExposedDataForOrganizerTests(ExposedData):
    __test__ = False
    # _caller_id: DummyCallerIdentifier | None # No longer needed with property
    # _timestamp: datetime.datetime # No longer needed with property
    data: float

    def __init__(self, caller_id: DummyCallerIdentifier | None, timestamp: datetime.datetime, data_val: float = 0.0):
        self._caller_id_val = caller_id
        self._timestamp_val = timestamp
        self.data = data_val

    @property
    def caller_id(self) -> DummyCallerIdentifier | None:
        """Return the identifier of the instance that generated this data."""
        return self._caller_id_val

    @property
    def timestamp(self) -> datetime.datetime:
        """Return the timestamp when this data was generated or recorded."""
        return self._timestamp_val

    @timestamp.setter
    def timestamp(self, value: datetime.datetime) -> None:
        """Set the timestamp."""
        self._timestamp_val = value

    # Ensure 'data' is a direct attribute that can be set after deepcopy
    # The __init__ already sets self.data = data_val

    def __repr__(self):
        # Use the property to access caller_id and timestamp safely
        caller_id_repr = self.caller_id.id_str if self.caller_id else "None"
        timestamp_repr = self.timestamp if hasattr(self, '_timestamp_val') else "N/A" # Check if initialized
        data_repr = getattr(self, "data", "N/A")
        return f"DummyExposedDataForOrganizerTests(caller_id='{caller_id_repr}', timestamp='{timestamp_repr}', data={data_repr})"


# --- Fixtures ---


@pytest.fixture
def mock_thread_pool(mocker):
    mock_pool = mocker.MagicMock(spec=ThreadPoolExecutor)

    def immediate_submit(func_to_call, *args_for_func, **kwargs_for_func):
        return func_to_call(*args_for_func, **kwargs_for_func)

    mock_pool.submit = mocker.MagicMock(side_effect=immediate_submit)
    return mock_pool


@pytest.fixture
def mock_caller_id():
    return DummyCallerIdentifier("test_caller_1")


@pytest.fixture
def mock_client(mocker):
    return mocker.MagicMock(spec=RemoteDataOrganizer.Client)


@pytest.fixture
def mock_is_running_tracker(mocker):
    mock_tracker_instance = mocker.MagicMock(spec=IsRunningTracker)
    mock_tracker_instance.get.return_value = False
    mocker.patch(
        "tsercom.data.remote_data_organizer.IsRunningTracker",
        return_value=mock_tracker_instance,
    )
    return mock_tracker_instance


@pytest.fixture
def organizer(mock_thread_pool, mock_caller_id, mock_client, mock_is_running_tracker):
    org = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=mock_client,
    )
    return org


@pytest.fixture
def organizer_no_client(mock_thread_pool, mock_caller_id, mock_is_running_tracker):
    org = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool, caller_id=mock_caller_id, client=None
    )
    return org


# --- Helper to create data ---
def create_data(caller_id, timestamp_input, data_val=0.0):
    if isinstance(timestamp_input, datetime.datetime):
        ts = timestamp_input
    else:
        base_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
        ts = base_time + datetime.timedelta(seconds=timestamp_input)
    # Pass data_val to the constructor directly
    instance = DummyExposedDataForOrganizerTests(caller_id, ts, data_val=data_val)
    return instance


# --- Test Cases ---


# 1. Initialization
def test_initialization(
    mock_thread_pool, mock_caller_id, mock_client, mock_is_running_tracker
):
    organizer_instance = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=mock_client,
    )
    assert organizer_instance._RemoteDataOrganizer__thread_pool is mock_thread_pool
    assert organizer_instance.caller_id is mock_caller_id
    assert organizer_instance._RemoteDataOrganizer__client is mock_client
    assert isinstance(organizer_instance._RemoteDataOrganizer__data, SortedList)
    assert len(organizer_instance._RemoteDataOrganizer__data) == 0
    assert (
        organizer_instance._RemoteDataOrganizer__last_access
        == datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    )
    assert (
        organizer_instance._RemoteDataOrganizer__is_running is mock_is_running_tracker
    )


def test_initialization_no_client(
    mock_thread_pool, mock_caller_id, mock_is_running_tracker
):
    organizer_instance = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool, caller_id=mock_caller_id, client=None
    )
    assert organizer_instance._RemoteDataOrganizer__client is None


# 2. start() and stop()
def test_start(organizer, mock_is_running_tracker):
    organizer.start()
    mock_is_running_tracker.start.assert_called_once()


def test_start_asserts_if_already_running(organizer, mock_is_running_tracker):
    mock_is_running_tracker.start.side_effect = RuntimeError(
        "IsRunningTracker already started."
    )
    with pytest.raises(RuntimeError, match="IsRunningTracker already started."):
        organizer.start()


def test_stop(organizer, mock_is_running_tracker):
    organizer.start()
    organizer.stop()
    mock_is_running_tracker.stop.assert_called_once()


def test_stop_asserts_if_not_running(organizer, mock_is_running_tracker):
    mock_is_running_tracker.stop.side_effect = RuntimeError(
        "IsRunningTracker not running."
    )
    with pytest.raises(RuntimeError, match="IsRunningTracker not running."):
        organizer.stop()


# 3. _on_data_ready() and __on_data_ready_impl()
def test_on_data_ready_submits_to_thread_pool(
    organizer, mock_thread_pool, mock_caller_id, mocker
):
    data = create_data(mock_caller_id, datetime.datetime.now())
    mock_thread_pool.submit = (
        mocker.MagicMock()
    )  # Re-mock to ensure fresh state for submit
    organizer._on_data_ready(data)
    mock_thread_pool.submit.assert_called_once_with(
        organizer._RemoteDataOrganizer__on_data_ready_impl, data
    )


def test_on_data_ready_impl_not_running(
    organizer_no_client, mock_is_running_tracker, mock_caller_id
):
    mock_is_running_tracker.get.return_value = False
    data = create_data(mock_caller_id, datetime.datetime.now())
    organizer_no_client._RemoteDataOrganizer__on_data_ready_impl(data)
    assert len(organizer_no_client._RemoteDataOrganizer__data) == 0


def test_on_data_ready_impl_adds_to_empty_deque_and_notifies_client(  # Name needs update post-deque
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    data = create_data(mock_caller_id, datetime.datetime.now(), data_val=1)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data)
    assert len(organizer._RemoteDataOrganizer__data) == 1
    assert organizer._RemoteDataOrganizer__data[0] == data
    mock_client._on_data_available.assert_called_once_with(organizer)


def test_on_data_ready_impl_adds_newer_data_to_front(  # Name needs update, behavior changes
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    ts_now = datetime.datetime.now()
    data_old = create_data(
        mock_caller_id, ts_now - datetime.timedelta(seconds=10), data_val=1
    )
    data_new = create_data(mock_caller_id, ts_now, data_val=2)

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_old)
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_new)

    assert len(organizer._RemoteDataOrganizer__data) == 2
    # SortedList order: oldest first
    assert organizer._RemoteDataOrganizer__data[0] == data_old
    assert organizer._RemoteDataOrganizer__data[1] == data_new
    mock_client._on_data_available.assert_called_once_with(organizer)


def test_on_data_ready_impl_ignores_older_data(
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    ts_now = datetime.datetime.now()
    data_current_first = create_data(mock_caller_id, ts_now, data_val=1)
    data_older = create_data(
        mock_caller_id, ts_now - datetime.timedelta(seconds=10), data_val=2
    )

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_current_first)
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_older)

    assert len(organizer._RemoteDataOrganizer__data) == 2
    # SortedList order: oldest first
    assert organizer._RemoteDataOrganizer__data[0] == data_older
    assert organizer._RemoteDataOrganizer__data[1] == data_current_first
    mock_client._on_data_available.assert_not_called()


def test_on_data_ready_impl_replaces_data_with_same_timestamp(
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    ts_now = datetime.datetime.now()
    data_initial = create_data(mock_caller_id, ts_now, data_val=1)
    data_replacement = create_data(mock_caller_id, ts_now, data_val=2)

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_initial)
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_replacement)

    assert len(organizer._RemoteDataOrganizer__data) == 1
    assert organizer._RemoteDataOrganizer__data[0] == data_replacement
    assert organizer._RemoteDataOrganizer__data[0].data == 2
    mock_client._on_data_available.assert_called_once_with(organizer)


def test_on_data_ready_impl_no_client_no_notification(
    organizer_no_client, mock_is_running_tracker, mock_caller_id
):
    mock_is_running_tracker.get.return_value = True
    data = create_data(mock_caller_id, datetime.datetime.now())
    try:
        organizer_no_client._RemoteDataOrganizer__on_data_ready_impl(data)
    except Exception as e:
        pytest.fail(f"__on_data_ready_impl failed with no client: {e}")
    assert len(organizer_no_client._RemoteDataOrganizer__data) == 1


# 4. has_new_data()
def test_has_new_data_empty_deque(organizer):  # Name can be updated
    assert organizer.has_new_data() is False


def test_has_new_data_all_accessed(organizer, mock_caller_id):
    ts_now = datetime.datetime.now()
    organizer._RemoteDataOrganizer__last_access = ts_now
    data_old = create_data(mock_caller_id, ts_now - datetime.timedelta(seconds=1))
    organizer._RemoteDataOrganizer__data.add(data_old)
    assert organizer.has_new_data() is False


def test_has_new_data_new_available(organizer, mock_caller_id):
    ts_now = datetime.datetime.now()
    organizer._RemoteDataOrganizer__last_access = ts_now - datetime.timedelta(seconds=1)
    data_new = create_data(mock_caller_id, ts_now)
    organizer._RemoteDataOrganizer__data.add(data_new)
    assert organizer.has_new_data() is True


# 5. get_new_data()
def test_get_new_data_empty_deque(organizer):  # Name can be updated
    assert organizer.get_new_data() == []


def test_get_new_data_retrieves_newer_than_last_access(organizer, mock_caller_id):
    data1_ts = datetime.datetime(2023, 1, 1, 10, 0, 0)
    data2_ts = datetime.datetime(2023, 1, 1, 10, 0, 1)
    data0_ts = datetime.datetime(2023, 1, 1, 9, 59, 59)

    data0 = create_data(mock_caller_id, data0_ts, data_val=0)
    data1 = create_data(mock_caller_id, data1_ts, data_val=1)
    data2 = create_data(mock_caller_id, data2_ts, data_val=2)

    organizer._RemoteDataOrganizer__data.add(data0)
    organizer._RemoteDataOrganizer__data.add(data1)
    organizer._RemoteDataOrganizer__data.add(data2)

    organizer._RemoteDataOrganizer__last_access = data1_ts
    new_data_list = organizer.get_new_data()
    assert len(new_data_list) == 1
    assert new_data_list[0] == data2  # get_new_data returns newest first
    assert organizer._RemoteDataOrganizer__last_access == data2_ts


# 6. get_most_recent_data()
def test_get_most_recent_data_empty(organizer):
    assert organizer.get_most_recent_data() is None


def test_get_most_recent_data_returns_first_item(
    organizer, mock_caller_id
):  # Name needs update
    data1 = create_data(mock_caller_id, datetime.datetime.now(), data_val=1)
    data2 = create_data(
        mock_caller_id,
        datetime.datetime.now() + datetime.timedelta(seconds=1),
        data_val=2,
    )
    organizer._RemoteDataOrganizer__data.add(data1)
    organizer._RemoteDataOrganizer__data.add(data2)
    assert organizer.get_most_recent_data() == data2  # SortedList[-1] is newest


# 7. get_data_for_timestamp()
def test_get_data_for_timestamp_empty(organizer):
    assert organizer.get_data_for_timestamp(datetime.datetime.now()) is None


def test_get_data_for_timestamp_older_than_all(organizer, mock_caller_id):
    ts_data = datetime.datetime(2023, 1, 1, 12, 0, 0)
    data = create_data(mock_caller_id, ts_data)
    organizer._RemoteDataOrganizer__data.add(data)
    ts_query = ts_data - datetime.timedelta(seconds=10)
    assert organizer.get_data_for_timestamp(ts_query) is None


def test_get_data_for_timestamp_finds_correct_item(organizer, mock_caller_id):
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)
    ts3 = datetime.datetime(2023, 1, 1, 12, 0, 20)
    data1 = create_data(mock_caller_id, ts1, data_val=1)
    data2 = create_data(mock_caller_id, ts2, data_val=2)
    data3 = create_data(mock_caller_id, ts3, data_val=3)

    organizer._RemoteDataOrganizer__data.clear()
    organizer._RemoteDataOrganizer__data.add(data1)
    organizer._RemoteDataOrganizer__data.add(data2)
    organizer._RemoteDataOrganizer__data.add(data3)

    query_ts_between_2_3 = ts2 + datetime.timedelta(seconds=5)
    assert organizer.get_data_for_timestamp(query_ts_between_2_3) == data2
    assert organizer.get_data_for_timestamp(ts2) == data2
    query_ts_after_all = ts3 + datetime.timedelta(seconds=5)
    assert organizer.get_data_for_timestamp(query_ts_after_all) == data3
    assert organizer.get_data_for_timestamp(ts1) == data1


# 8. _on_triggered()
def test_on_triggered_submits_to_thread_pool(organizer, mock_thread_pool, mocker):
    timeout_val = 30
    mock_thread_pool.submit = mocker.MagicMock()  # Re-mock
    organizer._on_triggered(timeout_val)
    mock_thread_pool.submit.assert_called_once_with(
        mocker.ANY
    )  # functools.partial makes direct comparison hard
    submitted_callable = mock_thread_pool.submit.call_args[0][0]
    assert isinstance(submitted_callable, functools.partial)
    assert submitted_callable.func == organizer._RemoteDataOrganizer__timeout_old_data
    assert submitted_callable.args == (timeout_val,)


def test_timeout_old_data_removes_old_items_standalone(
    organizer, mock_caller_id, mocker, mock_is_running_tracker
):
    current_time_mock_val = datetime.datetime(2023, 1, 1, 12, 0, 0)
    mock_is_running_tracker.get.return_value = True
    datetime_module_mock = mocker.MagicMock(name="datetime_module_mock")
    datetime_class_mock = mocker.MagicMock(name="datetime_class_mock")
    datetime_class_mock.now.return_value = current_time_mock_val
    datetime_module_mock.datetime = datetime_class_mock
    datetime_module_mock.timedelta = datetime.timedelta
    mocker.patch(
        "tsercom.data.remote_data_organizer.datetime", new=datetime_module_mock
    )

    timeout_seconds = 30
    ts_old = current_time_mock_val - datetime.timedelta(seconds=31)
    ts_kept1 = current_time_mock_val - datetime.timedelta(seconds=29)
    ts_kept2 = current_time_mock_val - datetime.timedelta(seconds=1)
    data_old = create_data(mock_caller_id, ts_old, data_val=1)
    data_kept1 = create_data(mock_caller_id, ts_kept1, data_val=2)
    data_kept2 = create_data(mock_caller_id, ts_kept2, data_val=3)

    organizer._RemoteDataOrganizer__data.clear()
    organizer._RemoteDataOrganizer__data.add(data_old)
    organizer._RemoteDataOrganizer__data.add(data_kept1)
    organizer._RemoteDataOrganizer__data.add(data_kept2)

    organizer._RemoteDataOrganizer__timeout_old_data(timeout_seconds)
    assert len(organizer._RemoteDataOrganizer__data) == 2
    assert data_old not in organizer._RemoteDataOrganizer__data
    assert data_kept1 in organizer._RemoteDataOrganizer__data
    assert data_kept2 in organizer._RemoteDataOrganizer__data
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_kept1,
        data_kept2,
    ]  # Sorted: kept1, kept2


def test_timeout_old_data_empty_deque(organizer):  # Name can be updated
    try:
        organizer._RemoteDataOrganizer__timeout_old_data(30)
    except Exception as e:
        pytest.fail(f"__timeout_old_data failed on empty deque: {e}")
    assert len(organizer._RemoteDataOrganizer__data) == 0


def test_on_triggered_partial_call_integration(
    organizer, mock_caller_id, mocker, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    current_time_mock_val = datetime.datetime(2023, 1, 1, 12, 0, 0)
    datetime_module_mock = mocker.MagicMock(name="datetime_module_mock")
    datetime_class_mock = mocker.MagicMock(name="datetime_class_mock")
    datetime_class_mock.now.return_value = current_time_mock_val
    datetime_module_mock.datetime = datetime_class_mock
    datetime_module_mock.timedelta = datetime.timedelta
    mocker.patch(
        "tsercom.data.remote_data_organizer.datetime", new=datetime_module_mock
    )

    timeout_seconds = 10
    ts_to_timeout = current_time_mock_val - datetime.timedelta(seconds=11)
    data_to_timeout = create_data(mock_caller_id, ts_to_timeout)

    organizer._RemoteDataOrganizer__data.add(data_to_timeout)
    assert len(organizer._RemoteDataOrganizer__data) == 1

    organizer._on_triggered(timeout_seconds)

    assert len(organizer._RemoteDataOrganizer__data) == 0
    datetime_module_mock.datetime.now.assert_called_once()


def test_on_data_ready_raises_type_error_for_invalid_data_type(
    organizer, mock_thread_pool
):
    invalid_data = "not_exposed_data_object"
    with pytest.raises(
        TypeError, match="Expected new_data to be an instance of ExposedData"
    ):
        organizer._on_data_ready(invalid_data)
    mock_thread_pool.submit.assert_not_called()


def test_on_data_ready_raises_assertion_error_for_caller_id_mismatch(
    organizer, mock_caller_id, mock_thread_pool, mocker
):
    mismatched_caller_id = DummyCallerIdentifier("mismatched_id")
    assert mismatched_caller_id != mock_caller_id
    data_with_wrong_id = create_data(mismatched_caller_id, datetime.datetime.now())
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Data's caller_id '{repr(mismatched_caller_id)}' does not match organizer's '{repr(mock_caller_id)}'"
        ),
    ):
        organizer._on_data_ready(data_with_wrong_id)
    mock_thread_pool.submit.assert_not_called()


def test_timeout_old_data_does_nothing_if_not_running(organizer, mock_caller_id):
    internal_mock_is_running_tracker = organizer._RemoteDataOrganizer__is_running
    internal_mock_is_running_tracker.get.return_value = False
    old_ts = datetime.datetime.now() - datetime.timedelta(days=1)
    data_old = create_data(mock_caller_id, old_ts)
    organizer._RemoteDataOrganizer__data.add(data_old)
    assert len(organizer._RemoteDataOrganizer__data) == 1
    organizer._RemoteDataOrganizer__timeout_old_data(timeout_seconds=30)
    assert len(organizer._RemoteDataOrganizer__data) == 1
    internal_mock_is_running_tracker.get.assert_called_once()


# --- Tests for Out-of-Order Data Handling ---


def test_on_data_ready_impl_inserts_out_of_order_data_correctly(
    organizer, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    now = datetime.datetime.now(datetime.timezone.utc)
    data_t2 = create_data(mock_caller_id, now, data_val=2)
    data_t4 = create_data(
        mock_caller_id, now + datetime.timedelta(seconds=20), data_val=4
    )
    data_t3 = create_data(
        mock_caller_id, now + datetime.timedelta(seconds=10), data_val=3
    )
    data_t1 = create_data(
        mock_caller_id, now - datetime.timedelta(seconds=10), data_val=1
    )

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t4)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)

    expected_order = [data_t1, data_t2, data_t3, data_t4]  # SortedList order
    actual_order = list(organizer._RemoteDataOrganizer__data)
    assert actual_order == expected_order


def test_on_data_ready_impl_no_callback_for_older_data(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    now = datetime.datetime.now(datetime.timezone.utc)
    data_t2 = create_data(mock_caller_id, now, data_val=2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    mock_client._on_data_available.assert_called_once_with(organizer)
    mock_client.reset_mock()

    data_t1 = create_data(
        mock_caller_id, now - datetime.timedelta(seconds=10), data_val=1
    )
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    mock_client._on_data_available.assert_not_called()

    data_t1_5 = create_data(
        mock_caller_id, now - datetime.timedelta(seconds=5), data_val=15
    )
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1_5)
    mock_client._on_data_available.assert_not_called()

    expected_order = [data_t1, data_t1_5, data_t2]  # SortedList order
    actual_order = list(organizer._RemoteDataOrganizer__data)
    assert actual_order == expected_order


def test_on_data_ready_impl_callback_for_new_or_same_timestamp_data(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    now = datetime.datetime.now(datetime.timezone.utc)
    data_t1 = create_data(mock_caller_id, now, data_val=1)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    mock_client._on_data_available.assert_called_once_with(organizer)
    mock_client.reset_mock()

    data_t2_ts = now + datetime.timedelta(seconds=10)
    data_t2 = create_data(mock_caller_id, data_t2_ts, data_val=2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    mock_client._on_data_available.assert_called_once_with(organizer)
    mock_client.reset_mock()

    data_t2_updated = create_data(mock_caller_id, data_t2_ts, data_val=3)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2_updated)
    mock_client._on_data_available.assert_called_once_with(
        organizer
    )  # Update should notify

    expected_data_state = [data_t1, data_t2_updated]  # SortedList order
    actual_data_state = list(organizer._RemoteDataOrganizer__data)
    assert actual_data_state == expected_data_state


# --- More Complex Out-of-Order Scenarios ---


def test_batch_old_then_newer_then_oldest(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    data_t0 = create_data(mock_caller_id, base_ts, data_val=0)
    data_t1 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=1), data_val=1
    )
    data_t2 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=2), data_val=2
    )
    data_t3 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=3), data_val=3
    )
    data_t5 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=5), data_val=5
    )
    data_t6 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=6), data_val=6
    )
    data_t7 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=7), data_val=7
    )
    data_t8 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=8), data_val=8
    )

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t5)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t6)
    assert mock_client._on_data_available.call_count == 2
    assert list(organizer._RemoteDataOrganizer__data) == [data_t5, data_t6]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t2,
        data_t3,
        data_t5,
        data_t6,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t7)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t8)
    assert mock_client._on_data_available.call_count == 2
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t2,
        data_t3,
        data_t5,
        data_t6,
        data_t7,
        data_t8,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t0)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t0,
        data_t1,
        data_t2,
        data_t3,
        data_t5,
        data_t6,
        data_t7,
        data_t8,
    ]
    mock_client.reset_mock()


def test_interspersed_new_and_out_of_order_data(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    data_t3 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=3), data_val=3
    )
    data_t5 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=5), data_val=5
    )
    data_t7 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=7), data_val=7
    )
    data_t10 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=10), data_val=10
    )
    data_t12 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=12), data_val=12
    )
    data_t12_replace = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=12), data_val=120
    )

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t10)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [data_t10]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t5)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [data_t5, data_t10]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t12)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t5,
        data_t10,
        data_t12,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t3,
        data_t5,
        data_t10,
        data_t12,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t7)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t3,
        data_t5,
        data_t7,
        data_t10,
        data_t12,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t12_replace)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t3,
        data_t5,
        data_t7,
        data_t10,
        data_t12_replace,
    ]
    mock_client.reset_mock()


def test_out_of_order_data_with_duplicate_timestamps_inserted_among_existing(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    t10 = base_ts + datetime.timedelta(seconds=10)
    t30 = base_ts + datetime.timedelta(seconds=30)
    t50 = base_ts + datetime.timedelta(seconds=50)
    t100 = base_ts + datetime.timedelta(seconds=100)
    data_t10 = create_data(mock_caller_id, t10, data_val=10)
    data_t30a = create_data(mock_caller_id, t30, data_val=301)
    data_t30b = create_data(mock_caller_id, t30, data_val=302)
    data_t50 = create_data(mock_caller_id, t50, data_val=50)
    data_t50_replace = create_data(mock_caller_id, t50, data_val=500)
    data_t100 = create_data(mock_caller_id, t100, data_val=100)

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t10)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t50)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t100)
    assert mock_client._on_data_available.call_count == 3
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t10,
        data_t50,
        data_t100,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t30a)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t10,
        data_t30a,
        data_t50,
        data_t100,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t30b)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t10,
        data_t30b,
        data_t50,
        data_t100,
    ]
    mock_client.reset_mock()  # Update should notify
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t50_replace)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t10,
        data_t30b,
        data_t50_replace,
        data_t100,
    ]
    mock_client.reset_mock()  # Update should notify


def test_adding_data_becomes_new_oldest_item_repeatedly(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    data_t2 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=2), data_val=2
    )
    data_t3 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=3), data_val=3
    )
    data_t4 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=4), data_val=4
    )
    data_t5 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=5), data_val=5
    )
    data_t6 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=6), data_val=6
    )

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t5)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [data_t5]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t4)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [data_t4, data_t5]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t3,
        data_t4,
        data_t5,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t2,
        data_t3,
        data_t4,
        data_t5,
    ]
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t6)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t2,
        data_t3,
        data_t4,
        data_t5,
        data_t6,
    ]
    mock_client.reset_mock()


# --- New Fixture and Tests for get_interpolated_at ---


@pytest.fixture
def organizer_for_interpolation_tests(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> RemoteDataOrganizer[DummyExposedDataForOrganizerTests]:
    organizer._RemoteDataOrganizer__data.clear()
    # Data will be sorted by timestamp due to SortedList: (10,100), (20,200), (30,300)
    organizer._RemoteDataOrganizer__data.add(
        create_data(
            mock_caller_id,
            BASE_TIME_UTC + datetime.timedelta(seconds=10),
            data_val=100.0,
        )
    )
    organizer._RemoteDataOrganizer__data.add(
        create_data(
            mock_caller_id,
            BASE_TIME_UTC + datetime.timedelta(seconds=20),
            data_val=200.0,
        )
    )
    organizer._RemoteDataOrganizer__data.add(
        create_data(
            mock_caller_id,
            BASE_TIME_UTC + datetime.timedelta(seconds=30),
            data_val=300.0,
        )
    )
    return organizer


def test_interpolate_empty_data_new(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
) -> None:
    organizer._RemoteDataOrganizer__data.clear()
    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=15)
    assert organizer.get_interpolated_at(query_ts) is None


def test_interpolate_before_first_keyframe_new(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
) -> None:
    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=5)
    interpolated = organizer_for_interpolation_tests.get_interpolated_at(query_ts)
    assert interpolated is not None
    expected_item = organizer_for_interpolation_tests._RemoteDataOrganizer__data[0]
    assert interpolated.timestamp == expected_item.timestamp
    assert interpolated.data == expected_item.data
    assert interpolated is not expected_item


def test_interpolate_after_last_keyframe_new(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
) -> None:
    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=35)
    interpolated = organizer_for_interpolation_tests.get_interpolated_at(query_ts)
    assert interpolated is not None
    expected_item = organizer_for_interpolation_tests._RemoteDataOrganizer__data[-1]
    assert interpolated.timestamp == expected_item.timestamp
    assert interpolated.data == expected_item.data
    assert interpolated is not expected_item


def test_interpolate_exact_match_keyframe_new(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
) -> None:
    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=20)
    interpolated = organizer_for_interpolation_tests.get_interpolated_at(query_ts)
    assert interpolated is not None
    expected_item = organizer_for_interpolation_tests._RemoteDataOrganizer__data[1]
    assert (
        interpolated.timestamp == expected_item.timestamp
    )  # RDO returns original item's timestamp for exact match
    assert interpolated.data == expected_item.data
    assert interpolated is not expected_item


def test_interpolate_successful_between_two_points_new(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    query_ts_1 = BASE_TIME_UTC + datetime.timedelta(seconds=15)
    interpolated_1 = organizer_for_interpolation_tests.get_interpolated_at(query_ts_1)
    assert interpolated_1 is not None
    assert interpolated_1.timestamp == query_ts_1
    assert interpolated_1.caller_id == mock_caller_id
    assert pytest.approx(interpolated_1.data) == 150.0

    query_ts_2 = BASE_TIME_UTC + datetime.timedelta(seconds=27)
    interpolated_2 = organizer_for_interpolation_tests.get_interpolated_at(query_ts_2)
    assert interpolated_2 is not None
    assert interpolated_2.timestamp == query_ts_2
    assert pytest.approx(interpolated_2.data) == 270.0


def test_interpolate_with_identical_bracketing_keyframes_data_new(
    organizer: RemoteDataOrganizer[DummyExposedDataForOrganizerTests],
    mock_caller_id: DummyCallerIdentifier,
) -> None:
    organizer._RemoteDataOrganizer__data.clear()
    item1 = create_data(
        mock_caller_id,
        BASE_TIME_UTC + datetime.timedelta(seconds=10),
        data_val=100.0,
    )
    item2 = create_data(
        mock_caller_id,
        BASE_TIME_UTC + datetime.timedelta(seconds=20),
        data_val=100.0,
    )
    organizer._RemoteDataOrganizer__data.add(item1)
    organizer._RemoteDataOrganizer__data.add(item2)

    query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=15)
    interpolated = organizer.get_interpolated_at(query_ts)
    assert interpolated is not None
    assert interpolated.timestamp == query_ts
    assert pytest.approx(interpolated.data) == 100.0


def test_interpolate_data_type_not_supporting_arithmetic_new(
    organizer_for_interpolation_tests: RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    item_to_corrupt = organizer_for_interpolation_tests._RemoteDataOrganizer__data[0]
    original_data_attr = item_to_corrupt.data

    try:
        item_to_corrupt.data = "non_numeric_string"

        query_ts = BASE_TIME_UTC + datetime.timedelta(seconds=15)
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            interpolated = organizer_for_interpolation_tests.get_interpolated_at(
                query_ts
            )

        assert interpolated is None
        assert any(
            "Data payloads for interpolation are not numeric" in record.message
            for record in caplog.records
        )  # Adjusted log message
    finally:
        item_to_corrupt.data = original_data_attr
