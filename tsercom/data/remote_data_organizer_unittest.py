import pytest
import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import functools  # For checking partial
import re  # For re.escape

# Import actual classes from tsercom
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.util.is_running_tracker import IsRunningTracker  # For mocking


# Assuming CallerIdentifier.py is missing, define a functional dummy for tests.
class DummyCallerIdentifier:
    def __init__(self, id_str: str):
        self.id_str = id_str

    def __hash__(self):
        return hash(self.id_str)

    def __eq__(self, other):
        return (
            isinstance(other, DummyCallerIdentifier)
            and self.id_str == other.id_str
        )

    def __repr__(self):
        return f"DummyCallerIdentifier('{self.id_str}')"

    @property
    def name(self) -> str:
        return self.id_str


# --- Concrete Dummy ExposedData for type checks and usage ---
class DummyExposedDataForOrganizerTests(ExposedData):
    __test__ = False  # Mark this class as not a test class for pytest
    # __init__ removed to be pytest-compatible
    # value attribute will be set in create_data or directly in tests

    def __repr__(self):
        # Ensure 'value' attribute exists for repr, provide a default if not
        value_repr = getattr(self, "value", "N/A")
        return f"DummyExposedDataForOrganizerTests(caller_id='{self.caller_id.id_str}', timestamp='{self.timestamp}', value={value_repr})"


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
def organizer(
    mock_thread_pool, mock_caller_id, mock_client, mock_is_running_tracker
):
    org = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=mock_client,
    )
    return org


@pytest.fixture
def organizer_no_client(
    mock_thread_pool, mock_caller_id, mock_is_running_tracker
):
    org = RemoteDataOrganizer[DummyExposedDataForOrganizerTests](
        thread_pool=mock_thread_pool, caller_id=mock_caller_id, client=None
    )
    return org


# --- Helper to create data ---
def create_data(caller_id, timestamp_input, value_id=0):
    # If timestamp_input is already a datetime, use it. Otherwise, treat as offset for tests.
    if isinstance(timestamp_input, datetime.datetime):
        ts = timestamp_input
    else:  # Assume int/float offset from a base time for simplicity in some tests
        base_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
        ts = base_time + datetime.timedelta(seconds=timestamp_input)
    instance = DummyExposedDataForOrganizerTests(caller_id, ts)
    instance.value = value_id  # Set value directly
    return instance


# --- Test Cases ---


# 1. Initialization
def test_initialization(
    mock_thread_pool, mock_caller_id, mock_client, mock_is_running_tracker
):
    organizer_instance = RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ](
        thread_pool=mock_thread_pool,
        caller_id=mock_caller_id,
        client=mock_client,
    )
    assert (
        organizer_instance._RemoteDataOrganizer__thread_pool
        is mock_thread_pool
    )
    assert organizer_instance.caller_id is mock_caller_id
    assert organizer_instance._RemoteDataOrganizer__client is mock_client
    assert isinstance(organizer_instance._RemoteDataOrganizer__data, deque)
    assert len(organizer_instance._RemoteDataOrganizer__data) == 0
    assert (
        organizer_instance._RemoteDataOrganizer__last_access
        == datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    )
    assert (
        organizer_instance._RemoteDataOrganizer__is_running
        is mock_is_running_tracker
    )


def test_initialization_no_client(
    mock_thread_pool, mock_caller_id, mock_is_running_tracker
):
    organizer_instance = RemoteDataOrganizer[
        DummyExposedDataForOrganizerTests
    ](thread_pool=mock_thread_pool, caller_id=mock_caller_id, client=None)
    assert organizer_instance._RemoteDataOrganizer__client is None


# 2. start() and stop()
def test_start(organizer, mock_is_running_tracker):
    organizer.start()
    # Check that the 'start' method of the mock_is_running_tracker was called
    mock_is_running_tracker.start.assert_called_once()


def test_start_asserts_if_already_running(organizer, mock_is_running_tracker):
    # Configure the mock's start() method to raise RuntimeError
    # This simulates the behavior of IsRunningTracker.start() when called on an already started instance
    mock_is_running_tracker.start.side_effect = RuntimeError(
        "IsRunningTracker already started."
    )
    with pytest.raises(
        RuntimeError, match="IsRunningTracker already started."
    ):
        organizer.start()


def test_stop(organizer, mock_is_running_tracker):
    # First, call start() to ensure the mock_is_running_tracker's start() is called
    # and to put the organizer in a state where stop() is meaningful.
    organizer.start()
    # Then call stop()
    organizer.stop()
    # Check that the 'stop' method of the mock_is_running_tracker was called
    mock_is_running_tracker.stop.assert_called_once()


def test_stop_asserts_if_not_running(organizer, mock_is_running_tracker):
    # Configure the mock's stop() method to raise RuntimeError
    # This simulates the behavior of IsRunningTracker.stop() when called on an already stopped instance
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
    mock_thread_pool.submit = mocker.MagicMock()
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


def test_on_data_ready_impl_adds_to_empty_deque_and_notifies_client(
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    data = create_data(mock_caller_id, datetime.datetime.now(), value_id=1)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data)
    assert len(organizer._RemoteDataOrganizer__data) == 1
    assert organizer._RemoteDataOrganizer__data[0] == data
    mock_client._on_data_available.assert_called_once_with(organizer)


def test_on_data_ready_impl_adds_newer_data_to_front(
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    ts_now = datetime.datetime.now()
    data_old = create_data(
        mock_caller_id, ts_now - datetime.timedelta(seconds=10), value_id=1
    )
    data_new = create_data(mock_caller_id, ts_now, value_id=2)

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_old)
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_new)

    assert len(organizer._RemoteDataOrganizer__data) == 2
    assert organizer._RemoteDataOrganizer__data[0] == data_new
    assert organizer._RemoteDataOrganizer__data[1] == data_old
    mock_client._on_data_available.assert_called_once_with(organizer)


def test_on_data_ready_impl_ignores_older_data(
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    ts_now = datetime.datetime.now()
    data_current_first = create_data(mock_caller_id, ts_now, value_id=1)
    data_older = create_data(
        mock_caller_id, ts_now - datetime.timedelta(seconds=10), value_id=2
    )

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_current_first)
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_older)

    # With the new logic, older data is inserted.
    assert len(organizer._RemoteDataOrganizer__data) == 2
    # The order should be newest first.
    assert organizer._RemoteDataOrganizer__data[0] == data_current_first
    assert organizer._RemoteDataOrganizer__data[1] == data_older
    # Client should still not be called for out-of-order older data.
    mock_client._on_data_available.assert_not_called()


def test_on_data_ready_impl_replaces_data_with_same_timestamp(
    organizer, mock_is_running_tracker, mock_caller_id, mock_client
):
    mock_is_running_tracker.get.return_value = True
    ts_now = datetime.datetime.now()
    data_initial = create_data(mock_caller_id, ts_now, value_id=1)
    data_replacement = create_data(mock_caller_id, ts_now, value_id=2)

    organizer._RemoteDataOrganizer__on_data_ready_impl(data_initial)
    mock_client.reset_mock()
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_replacement)

    assert len(organizer._RemoteDataOrganizer__data) == 1
    assert organizer._RemoteDataOrganizer__data[0] == data_replacement
    assert organizer._RemoteDataOrganizer__data[0].value == 2
    # Client should be notified of the update.
    mock_client._on_data_available.assert_called_once_with(organizer)


def test_on_data_ready_impl_no_client_no_notification(
    organizer_no_client, mock_is_running_tracker, mock_caller_id
):
    mock_is_running_tracker.get.return_value = True
    data = create_data(mock_caller_id, datetime.datetime.now())
    try:
        organizer_no_client._RemoteDataOrganizer__on_data_ready_impl(data)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"__on_data_ready_impl failed with no client: {e}")
    assert len(organizer_no_client._RemoteDataOrganizer__data) == 1


# 4. has_new_data()
def test_has_new_data_empty_deque(organizer):
    assert organizer.has_new_data() is False


def test_has_new_data_all_accessed(organizer, mock_caller_id):
    ts_now = datetime.datetime.now()
    organizer._RemoteDataOrganizer__last_access = ts_now
    data_old = create_data(
        mock_caller_id, ts_now - datetime.timedelta(seconds=1)
    )
    organizer._RemoteDataOrganizer__data.appendleft(data_old)
    assert organizer.has_new_data() is False


def test_has_new_data_new_available(organizer, mock_caller_id):
    ts_now = datetime.datetime.now()
    organizer._RemoteDataOrganizer__last_access = ts_now - datetime.timedelta(
        seconds=1
    )
    data_new = create_data(mock_caller_id, ts_now)
    organizer._RemoteDataOrganizer__data.appendleft(data_new)
    assert organizer.has_new_data() is True


# 5. get_new_data()
def test_get_new_data_empty_deque(organizer):
    assert organizer.get_new_data() == []


def test_get_new_data_retrieves_newer_than_last_access(
    organizer, mock_caller_id
):
    data1_ts = datetime.datetime(2023, 1, 1, 10, 0, 0)
    data2_ts = datetime.datetime(2023, 1, 1, 10, 0, 1)
    data0_ts = datetime.datetime(2023, 1, 1, 9, 59, 59)

    data0 = create_data(mock_caller_id, data0_ts, value_id=0)
    data1 = create_data(mock_caller_id, data1_ts, value_id=1)
    data2 = create_data(mock_caller_id, data2_ts, value_id=2)

    organizer._RemoteDataOrganizer__data.appendleft(data0)
    organizer._RemoteDataOrganizer__data.appendleft(data1)
    organizer._RemoteDataOrganizer__data.appendleft(data2)

    organizer._RemoteDataOrganizer__last_access = data1_ts
    new_data_list = organizer.get_new_data()
    assert len(new_data_list) == 1
    assert new_data_list[0] == data2
    # __last_access should be updated to the timestamp of the newest item retrieved (data2)
    assert organizer._RemoteDataOrganizer__last_access == data2_ts


# 6. get_most_recent_data()
def test_get_most_recent_data_empty(organizer):
    assert organizer.get_most_recent_data() is None


def test_get_most_recent_data_returns_first_item(organizer, mock_caller_id):
    data1 = create_data(mock_caller_id, datetime.datetime.now(), value_id=1)
    data2 = create_data(
        mock_caller_id,
        datetime.datetime.now() + datetime.timedelta(seconds=1),
        value_id=2,
    )
    organizer._RemoteDataOrganizer__data.appendleft(data1)
    organizer._RemoteDataOrganizer__data.appendleft(data2)
    assert organizer.get_most_recent_data() == data2


# 7. get_data_for_timestamp()
def test_get_data_for_timestamp_empty(organizer):
    assert organizer.get_data_for_timestamp(datetime.datetime.now()) is None


def test_get_data_for_timestamp_older_than_all(organizer, mock_caller_id):
    ts_data = datetime.datetime(2023, 1, 1, 12, 0, 0)
    data = create_data(mock_caller_id, ts_data)
    organizer._RemoteDataOrganizer__data.appendleft(data)
    ts_query = ts_data - datetime.timedelta(seconds=10)
    assert organizer.get_data_for_timestamp(ts_query) is None


def test_get_data_for_timestamp_finds_correct_item(organizer, mock_caller_id):
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)
    ts3 = datetime.datetime(2023, 1, 1, 12, 0, 20)
    data1 = create_data(mock_caller_id, ts1, value_id=1)
    data2 = create_data(mock_caller_id, ts2, value_id=2)
    data3 = create_data(mock_caller_id, ts3, value_id=3)

    organizer._RemoteDataOrganizer__data.clear()
    organizer._RemoteDataOrganizer__data.appendleft(data1)
    organizer._RemoteDataOrganizer__data.appendleft(data2)
    organizer._RemoteDataOrganizer__data.appendleft(data3)
    # Deque: [data3, data2, data1] (newest to oldest)

    query_ts_between_2_3 = ts2 + datetime.timedelta(seconds=5)  # 12:00:15
    assert organizer.get_data_for_timestamp(query_ts_between_2_3) == data2

    # For ts2 (12:00:10), data2 is the most recent item at or before this time.
    assert organizer.get_data_for_timestamp(ts2) == data2

    query_ts_after_all = ts3 + datetime.timedelta(seconds=5)
    assert organizer.get_data_for_timestamp(query_ts_after_all) == data3

    # For ts1 (12:00:00), data1 is the most recent item at or before this time.
    assert organizer.get_data_for_timestamp(ts1) == data1


# 8. _on_triggered()
def test_on_triggered_submits_to_thread_pool(
    organizer, mock_thread_pool, mocker
):
    timeout_val = 30
    mock_thread_pool.submit = mocker.MagicMock()
    organizer._on_triggered(timeout_val)

    mock_thread_pool.submit.assert_called_once_with(mocker.ANY)
    submitted_callable = mock_thread_pool.submit.call_args[0][0]
    assert isinstance(submitted_callable, functools.partial)
    assert (
        submitted_callable.func
        == organizer._RemoteDataOrganizer__timeout_old_data
    )
    assert submitted_callable.args == (timeout_val,)

    # 9. __timeout_old_data()
    def test_timeout_old_data_removes_old_items(
        organizer, mock_caller_id, mocker, mock_is_running_tracker
    ):  # Added mock_is_running_tracker
        current_time_mock_val = datetime.datetime(2023, 1, 1, 12, 0, 0)

        # Ensure the organizer is "running" for this test
        mock_is_running_tracker.get.return_value = True

        datetime_module_mock = mocker.MagicMock(name="datetime_module_mock")
        datetime_class_mock = mocker.MagicMock(name="datetime_class_mock")
        datetime_class_mock.now.return_value = current_time_mock_val
        datetime_module_mock.datetime = datetime_class_mock
        datetime_module_mock.timedelta = datetime.timedelta
        mocker.patch(
            "tsercom.data.remote_data_organizer.datetime",
            new=datetime_module_mock,
        )

        timeout_seconds = 30

        # Timestamps relative to current_time_mock_val
        ts_old = current_time_mock_val - datetime.timedelta(seconds=31)
        ts_kept1 = current_time_mock_val - datetime.timedelta(seconds=29)
        ts_kept2 = current_time_mock_val - datetime.timedelta(seconds=1)

        data_old = create_data(mock_caller_id, ts_old, value_id=1)
        data_kept1 = create_data(mock_caller_id, ts_kept1, value_id=2)
        data_kept2 = create_data(mock_caller_id, ts_kept2, value_id=3)

        # Setup deque: [newest, ..., oldest] -> [data_kept2, data_kept1, data_old]
        organizer._RemoteDataOrganizer__data.clear()
        organizer._RemoteDataOrganizer__data.appendleft(data_old)
        organizer._RemoteDataOrganizer__data.appendleft(data_kept1)
        organizer._RemoteDataOrganizer__data.appendleft(data_kept2)

        organizer._RemoteDataOrganizer__timeout_old_data(timeout_seconds)

        assert len(organizer._RemoteDataOrganizer__data) == 2
        assert data_old not in organizer._RemoteDataOrganizer__data
        assert data_kept1 in organizer._RemoteDataOrganizer__data
        assert data_kept2 in organizer._RemoteDataOrganizer__data
        assert list(organizer._RemoteDataOrganizer__data) == [
            data_kept2,
            data_kept1,
        ]


def test_timeout_old_data_empty_deque(organizer):
    try:
        organizer._RemoteDataOrganizer__timeout_old_data(30)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"__timeout_old_data failed on empty deque: {e}")
    assert len(organizer._RemoteDataOrganizer__data) == 0


def test_on_triggered_partial_call_integration(
    organizer,
    mock_caller_id,
    mocker,
    mock_is_running_tracker,  # Added mock_is_running_tracker
):
    # Ensure the organizer is "running" for this test, so __timeout_old_data executes
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

    # Add to the right (oldest) for pop()
    organizer._RemoteDataOrganizer__data.append(data_to_timeout)
    assert len(organizer._RemoteDataOrganizer__data) == 1

    organizer._on_triggered(timeout_seconds)

    assert len(organizer._RemoteDataOrganizer__data) == 0
    datetime_module_mock.datetime.now.assert_called_once()


# --- New Tests ---


def test_on_data_ready_raises_type_error_for_invalid_data_type(
    organizer, mock_thread_pool
):
    """
    Tests that _on_data_ready() raises a TypeError if new_data is not an instance of ExposedData.
    """
    invalid_data = "not_exposed_data_object"  # A simple string

    with pytest.raises(
        TypeError,
        match="Expected new_data to be an instance of ExposedData",
    ):
        organizer._on_data_ready(invalid_data)

    mock_thread_pool.submit.assert_not_called()


def test_on_data_ready_raises_assertion_error_for_caller_id_mismatch(
    organizer,
    mock_caller_id,
    mock_thread_pool,
    mocker,  # mock_caller_id is the one organizer was initialized with
):
    """
    Tests that _on_data_ready() raises an AssertionError if the data's caller_id
    does not match the organizer's caller_id.
    """
    mismatched_caller_id = DummyCallerIdentifier("mismatched_id")
    # Ensure it's different from the organizer's mock_caller_id
    assert mismatched_caller_id != mock_caller_id

    data_with_wrong_id = create_data(
        mismatched_caller_id, datetime.datetime.now()
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Data's caller_id '{repr(mismatched_caller_id)}' does not match organizer's '{repr(mock_caller_id)}'"
        ),
    ):  # The SUT message does not have a trailing dot.
        organizer._on_data_ready(data_with_wrong_id)

    mock_thread_pool.submit.assert_not_called()


def test_timeout_old_data_does_nothing_if_not_running(
    organizer,
    mock_caller_id,  # mock_is_running_tracker is part of organizer fixture
):
    """
    Tests that __timeout_old_data does nothing if the organizer is not running.
    """
    # Get the mock_is_running_tracker from the organizer instance
    internal_mock_is_running_tracker = (
        organizer._RemoteDataOrganizer__is_running
    )
    internal_mock_is_running_tracker.get.return_value = (
        False  # Set to not running
    )

    old_ts = datetime.datetime.now() - datetime.timedelta(days=1)
    data_old = create_data(mock_caller_id, old_ts)

    # Manually add data to the internal deque
    organizer._RemoteDataOrganizer__data.append(data_old)
    assert len(organizer._RemoteDataOrganizer__data) == 1

    organizer._RemoteDataOrganizer__timeout_old_data(timeout_seconds=30)

    assert (
        len(organizer._RemoteDataOrganizer__data) == 1
    ), "Data should not have been removed"
    internal_mock_is_running_tracker.get.assert_called_once()  # Should be called once at the start of __timeout_old_data


# --- Tests for Out-of-Order Data Handling ---


def test_on_data_ready_impl_inserts_out_of_order_data_correctly(
    organizer, mock_caller_id, mock_is_running_tracker
):
    """
    Tests that __on_data_ready_impl correctly inserts out-of-order data
    while maintaining reverse chronological order.
    """
    mock_is_running_tracker.get.return_value = True
    now = datetime.datetime.now(datetime.timezone.utc)

    data_t2 = create_data(mock_caller_id, now, value_id=2)
    data_t4 = create_data(
        mock_caller_id, now + datetime.timedelta(seconds=20), value_id=4
    )
    data_t3 = create_data(
        mock_caller_id, now + datetime.timedelta(seconds=10), value_id=3
    )
    data_t1 = create_data(
        mock_caller_id, now - datetime.timedelta(seconds=10), value_id=1
    )

    # Add t2, then t4 (newest)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t4)
    # Current order: [data_t4, data_t2]

    # Add t3 (out-of-order, between t4 and t2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    # Expected order: [data_t4, data_t3, data_t2]

    # Add t1 (out-of-order, oldest)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    # Expected order: [data_t4, data_t3, data_t2, data_t1]

    expected_order = [data_t4, data_t3, data_t2, data_t1]
    actual_order = list(organizer._RemoteDataOrganizer__data)

    assert len(actual_order) == len(expected_order)
    for expected, actual in zip(expected_order, actual_order):
        assert (
            expected.timestamp == actual.timestamp
        ), f"Timestamp mismatch: expected {expected.timestamp}, got {actual.timestamp}"
        assert (
            expected.value == actual.value
        ), f"Value mismatch: expected {expected.value}, got {actual.value}"


def test_on_data_ready_impl_no_callback_for_older_data(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    """
    Tests that the client callback (_on_data_available) is NOT called when
    older, out-of-order data is inserted.
    """
    mock_is_running_tracker.get.return_value = True
    now = datetime.datetime.now(datetime.timezone.utc)

    data_t2 = create_data(mock_caller_id, now, value_id=2)
    # Initial data, should trigger callback
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    mock_client._on_data_available.assert_called_once_with(organizer)
    mock_client.reset_mock()

    # Insert older data
    data_t1 = create_data(
        mock_caller_id, now - datetime.timedelta(seconds=10), value_id=1
    )
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    mock_client._on_data_available.assert_not_called()

    # Insert another older data point, but newer than t1
    data_t1_5 = create_data(
        mock_caller_id, now - datetime.timedelta(seconds=5), value_id=15
    )
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1_5)
    mock_client._on_data_available.assert_not_called()  # Still not called since reset

    # Verify final order to be sure: [data_t2, data_t1_5, data_t1]
    expected_order = [data_t2, data_t1_5, data_t1]
    actual_order = list(organizer._RemoteDataOrganizer__data)
    assert len(actual_order) == len(expected_order)
    for i, item in enumerate(expected_order):
        assert item.value == actual_order[i].value


def test_on_data_ready_impl_callback_for_new_or_same_timestamp_data(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    """
    Tests that the client callback (_on_data_available) IS called for
    new data or data with the same timestamp as the current newest.
    """
    mock_is_running_tracker.get.return_value = True
    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Initial data
    data_t1 = create_data(mock_caller_id, now, value_id=1)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    mock_client._on_data_available.assert_called_once_with(organizer)
    mock_client.reset_mock()

    # 2. Newer data
    data_t2_ts = now + datetime.timedelta(seconds=10)
    data_t2 = create_data(mock_caller_id, data_t2_ts, value_id=2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    mock_client._on_data_available.assert_called_once_with(organizer)
    mock_client.reset_mock()

    # 3. Data with same timestamp as current newest (data_t2)
    data_t2_updated = create_data(
        mock_caller_id, data_t2_ts, value_id=3
    )  # Same ts, different value
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2_updated)
    mock_client._on_data_available.assert_called_once_with(organizer)

    # Verify final state of data
    # Expected: [data_t2_updated, data_t1]
    # (data_t2 was replaced by data_t2_updated)
    expected_data_state = [data_t2_updated, data_t1]
    actual_data_state = list(organizer._RemoteDataOrganizer__data)
    assert len(actual_data_state) == len(expected_data_state)
    for i, item in enumerate(expected_data_state):
        assert item.value == actual_data_state[i].value


# --- More Complex Out-of-Order Scenarios ---


def test_batch_old_then_newer_then_oldest(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    data_t0 = create_data(mock_caller_id, base_ts, value_id=0)
    data_t1 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=1), value_id=1
    )
    data_t2 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=2), value_id=2
    )
    data_t3 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=3), value_id=3
    )
    data_t5 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=5), value_id=5
    )
    data_t6 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=6), value_id=6
    )
    data_t7 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=7), value_id=7
    )
    data_t8 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=8), value_id=8
    )

    # Initial: Add data_t5, then data_t6
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t5)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t6)
    assert mock_client._on_data_available.call_count == 2
    assert list(organizer._RemoteDataOrganizer__data) == [data_t6, data_t5]
    mock_client.reset_mock()

    # Batch 1 (out-of-order): Add data_t2, then data_t3
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t6,
        data_t5,
        data_t3,
        data_t2,
    ]

    # Batch 2 (new): Add data_t7, then data_t8
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t7)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t8)
    assert mock_client._on_data_available.call_count == 2
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t8,
        data_t7,
        data_t6,
        data_t5,
        data_t3,
        data_t2,
    ]
    mock_client.reset_mock()

    # Batch 3 (out-of-order, oldest): Add data_t0, then data_t1
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t0)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t1)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t8,
        data_t7,
        data_t6,
        data_t5,
        data_t3,
        data_t2,
        data_t1,
        data_t0,
    ]


def test_interspersed_new_and_out_of_order_data(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    data_t3 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=3), value_id=3
    )
    data_t5 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=5), value_id=5
    )
    data_t7 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=7), value_id=7
    )
    data_t10 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=10), value_id=10
    )
    data_t12 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=12), value_id=12
    )
    data_t12_replace = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=12), value_id=120
    )  # Same timestamp as data_t12, different value

    # 1. Add data_t10
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t10)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [data_t10]
    mock_client.reset_mock()

    # 2. Add data_t5 (old)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t5)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [data_t10, data_t5]

    # 3. Add data_t12 (new)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t12)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t12,
        data_t10,
        data_t5,
    ]
    mock_client.reset_mock()

    # 4. Add data_t3 (old)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t12,
        data_t10,
        data_t5,
        data_t3,
    ]

    # 5. Add data_t7 (old, between t5 and t10)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t7)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t12,
        data_t10,
        data_t7,
        data_t5,
        data_t3,
    ]

    # 6. Add data_t12_replace (same as newest)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t12_replace)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t12_replace,
        data_t10,
        data_t7,
        data_t5,
        data_t3,
    ]


def test_out_of_order_data_with_duplicate_timestamps_inserted_among_existing(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    t10 = base_ts + datetime.timedelta(seconds=10)
    t30 = base_ts + datetime.timedelta(seconds=30)
    t50 = base_ts + datetime.timedelta(seconds=50)
    t100 = base_ts + datetime.timedelta(seconds=100)

    data_t10 = create_data(mock_caller_id, t10, value_id=10)
    data_t30a = create_data(
        mock_caller_id, t30, value_id=301
    )  # First item with t30
    data_t30b = create_data(
        mock_caller_id, t30, value_id=302
    )  # Second item with t30
    data_t50 = create_data(mock_caller_id, t50, value_id=50)
    data_t50_replace = create_data(
        mock_caller_id, t50, value_id=500
    )  # Replacement for t50
    data_t100 = create_data(mock_caller_id, t100, value_id=100)

    # Initial: Add data_t10, data_t50, data_t100
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t10)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t50)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t100)
    assert mock_client._on_data_available.call_count == 3
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t100,
        data_t50,
        data_t10,
    ]
    mock_client.reset_mock()

    # Add data_t30a (old, should be inserted between t50 and t10)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t30a)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t100,
        data_t50,
        data_t30a,
        data_t10,
    ]

    # Add data_t30b (old, same timestamp as t30a).
    # bisect_right places it before existing t30a when thinking ascending.
    # So, in descending deque, it will be after the original t30a from the "left" side.
    # Deque index: len(self.__data) - insertion_idx_ascending
    # timestamps_ascending: [t10, t30a, t50, t100]
    # bisect_right(timestamps_ascending, t30) -> index 2 (after t30a)
    # deque_index = 4 - 2 = 2. self.__data.insert(2, data_t30b)
    # Initial: [t100, t50, t30a, t10]
    # Insert at index 2: [t100, t50, data_t30b, t30a, t10]
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t30b)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t100,
        data_t50,
        data_t30b,  # data_t30b is inserted before data_t30a due to bisect_right logic
        data_t30a,
        data_t10,
    ]

    # Add data_t50_replace (old, because t100 is newest; ts same as t50)
    # bisect_right places it before existing t50 when thinking ascending.
    # So, in descending deque, it will be after the original t50 from the "left" side.
    # timestamps_ascending: [t10, t30a, t30b, t50, t100]
    # bisect_right(timestamps_ascending, t50) -> index 4 (after t50)
    # deque_index = 5 - 4 = 1. self.__data.insert(1, data_t50_replace)
    # Initial: [t100, t50, data_t30b, t30a, t10]
    # Insert at index 1: [t100, data_t50_replace, t50, data_t30b, t30a, t10]
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t50_replace)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t100,
        data_t50_replace,  # data_t50_replace is inserted before data_t50
        data_t50,
        data_t30b,
        data_t30a,
        data_t10,
    ]


def test_adding_data_becomes_new_oldest_item_repeatedly(
    organizer, mock_client, mock_caller_id, mock_is_running_tracker
):
    mock_is_running_tracker.get.return_value = True
    base_ts = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    data_t2 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=2), value_id=2
    )
    data_t3 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=3), value_id=3
    )
    data_t4 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=4), value_id=4
    )
    data_t5 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=5), value_id=5
    )
    data_t6 = create_data(
        mock_caller_id, base_ts + datetime.timedelta(seconds=6), value_id=6
    )

    # Initial: Add data_t5
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t5)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [data_t5]
    mock_client.reset_mock()

    # Add data_t4 (old)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t4)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [data_t5, data_t4]

    # Add data_t3 (old)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t3)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t5,
        data_t4,
        data_t3,
    ]

    # Add data_t2 (old)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t2)
    mock_client._on_data_available.assert_not_called()
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t5,
        data_t4,
        data_t3,
        data_t2,
    ]

    # Add data_t6 (new)
    organizer._RemoteDataOrganizer__on_data_ready_impl(data_t6)
    mock_client._on_data_available.assert_called_once_with(organizer)
    assert list(organizer._RemoteDataOrganizer__data) == [
        data_t6,
        data_t5,
        data_t4,
        data_t3,
        data_t2,
    ]
