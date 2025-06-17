import pytest

from tsercom.api.split_process.data_reader_source import DataReaderSource
from tsercom.util.is_running_tracker import (
    IsRunningTracker,
)  # Import real one for isinstance check

# from tsercom.utils.is_running_tracker import IsRunningTracker # Faked
# from tsercom.utils.multiprocess_queue_source import MultiprocessQueueSource # Faked
# from tsercom.threading.thread_watcher import ThreadWatcher # Faked
# from tsercom.data.remote_data_reader import RemoteDataReader # Faked
# from tsercom.data.exposed_data import ExposedData # Faked


class FakeExposedData:
    """Simple fake for type hinting if needed by ExposedData constraints."""

    pass


class FakeThread:
    def __init__(self, target, args=()):
        self.target = target
        self.args = args
        self.daemon = False
        self._started = False
        self._joined = False
        self._is_alive = False  # Add is_alive attribute

    def start(self):
        self._started = True
        self._is_alive = True  # Set to alive on start
        # For tests, we'll call target directly rather than actually threading

    def join(self, timeout=None):
        # In a real scenario, join blocks. For this fake, we can simulate
        # behavior based on how stop() is supposed to work with __is_running.
        # If the target loop respects __is_running, it should exit.
        # We can simulate that it eventually "joins" by setting _is_alive to False.
        # For the timeout test, we'll mock this further.
        self._joined = True
        self._is_alive = False  # Assume join means it's no longer alive

    def is_alive(self):  # Add is_alive method
        return self._is_alive


class FakeThreadWatcher:
    def __init__(self):
        self.created_thread_target = None
        self.created_thread_args = None
        self.created_thread_daemon = None
        self.fake_thread = None
        # Initialize on_exception_seen as a MagicMock
        try:
            # This will work if mocker is available in this scope,
            # otherwise, it needs to be passed or imported.
            # For a fixture, this is tricky. Let's assume tests will patch/set it.
            # For direct use in tests, test will have 'mocker' fixture.
            from unittest.mock import MagicMock

            self.on_exception_seen = MagicMock()
        except ImportError:  # pragma: no cover
            self.on_exception_seen = None  # Fallback if no mock
        # self.on_exception_seen_called_with = None # Replaced by MagicMock
        # self.on_exception_seen_call_count = 0 # Replaced by MagicMock

    def create_tracked_thread(self, target, args=(), daemon=True):
        self.created_thread_target = target
        self.created_thread_args = args
        self.created_thread_daemon = daemon
        self.fake_thread = FakeThread(target=target, args=args)
        self.fake_thread.daemon = daemon
        return self.fake_thread


class FakeMultiprocessQueueSource:
    def __init__(self, return_values=None):
        self.get_blocking_call_count = 0
        self._return_values = return_values if return_values else []
        self._return_idx = 0

    def get_blocking(
        self, timeout: float = 0.1
    ):  # Changed: timeout_seconds to timeout
        self.get_blocking_call_count += 1
        if self._return_idx < len(self._return_values):
            val = self._return_values[self._return_idx]
            self._return_idx += 1
            return val
        return None  # Default if no more values or empty list

    def add_return_value(self, value):
        self._return_values.append(value)

    def set_return_values(self, values):
        self._return_values = list(values)
        self._return_idx = 0


class FakeRemoteDataReader:
    def __init__(self):
        self.on_data_ready_called_with = []
        self.on_data_ready_call_count = 0

    def _on_data_ready(self, data):
        self.on_data_ready_called_with.append(data)
        self.on_data_ready_call_count += 1


class FakeIsRunningTracker:
    def __init__(self, initial_value=False):
        self._is_running = initial_value
        self.start_call_count = 0
        self.stop_call_count = 0
        self.get_call_count = 0

    def get(self):
        self.get_call_count += 1
        return self._is_running

    def set_is_running(self, value: bool):
        self._is_running = value

    def start(self):
        self.start_call_count += 1
        self.set_is_running(True)

    def stop(self):
        self.stop_call_count += 1
        self.set_is_running(False)


@pytest.fixture
def fake_queue():
    return FakeMultiprocessQueueSource()


@pytest.fixture
def fake_data_reader():
    return FakeRemoteDataReader()


@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_is_running_tracker():
    return FakeIsRunningTracker()


# from tsercom.util.is_running_tracker import ( # Moved import to top
#     IsRunningTracker,
# )  # Import real one for isinstance check


@pytest.fixture
def data_source(
    fake_queue, fake_data_reader, fake_watcher
):  # Removed fake_is_running_tracker from direct injection
    # IsRunningTracker is created internally, tests might need to replace it if direct control is needed.
    return DataReaderSource[FakeExposedData](
        queue=fake_queue,
        data_reader=fake_data_reader,
        watcher=fake_watcher,
        # is_running_tracker is NOT passed here
    )


# Test __init__
def test_init(
    data_source, fake_queue, fake_data_reader, fake_watcher
):  # Removed fake_is_running_tracker
    assert data_source._DataReaderSource__queue is fake_queue
    assert data_source._DataReaderSource__data_reader is fake_data_reader
    assert data_source._DataReaderSource__watcher is fake_watcher
    assert isinstance(
        data_source._DataReaderSource__is_running, IsRunningTracker
    )  # Check type


# Test is_running property
def test_is_running_property(
    data_source, fake_is_running_tracker
):  # Still use fake for this test
    # Replace the internal tracker with our fake one for this test
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    fake_is_running_tracker.set_is_running(True)
    assert data_source.is_running is True
    assert fake_is_running_tracker.get_call_count == 1

    fake_is_running_tracker.set_is_running(False)
    assert data_source.is_running is False
    assert fake_is_running_tracker.get_call_count == 2

    data_source._DataReaderSource__is_running = original_tracker  # Restore


# Test start() method
def test_start_method(
    data_source, fake_watcher, fake_is_running_tracker
):  # Still use fake for this test
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    data_source.start()

    assert (
        fake_watcher.created_thread_target
        == data_source._DataReaderSource__poll_for_data
    )
    assert fake_watcher.fake_thread is not None
    assert fake_watcher.fake_thread._started is True
    assert fake_is_running_tracker.start_call_count == 1
    assert fake_is_running_tracker.get() is True

    data_source._DataReaderSource__is_running = original_tracker  # Restore


# Test stop() method
def test_stop_method(
    data_source, fake_is_running_tracker
):  # Still use fake for this test
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    # First start it, so stop has an effect on a typically running state
    fake_is_running_tracker.set_is_running(True)  # Manually set the fake

    data_source.stop()
    assert fake_is_running_tracker.stop_call_count == 1
    assert fake_is_running_tracker.get() is False

    data_source._DataReaderSource__is_running = original_tracker  # Restore


# Tests for __poll_for_data behavior
def test_poll_for_data_single_item_then_stop(
    data_source, fake_queue, fake_data_reader, fake_is_running_tracker
):
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    test_item = FakeExposedData()
    fake_queue.set_return_values([test_item])

    is_running_sequence = [True, False]
    original_get_method = (
        fake_is_running_tracker.get
    )  # Store the original get method of the fake

    # Temporarily replace the get method on the fake tracker instance
    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False

    fake_is_running_tracker.get = sequenced_get

    data_source._DataReaderSource__poll_for_data()

    assert fake_data_reader.on_data_ready_call_count == 1
    assert fake_data_reader.on_data_ready_called_with[0] is test_item
    assert fake_queue.get_blocking_call_count == 1

    fake_is_running_tracker.get = (
        original_get_method  # Restore original get method on the fake
    )
    data_source._DataReaderSource__is_running = (
        original_tracker  # Restore original tracker
    )


def test_poll_for_data_timeout_then_stop(
    data_source, fake_queue, fake_data_reader, fake_is_running_tracker
):
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    fake_queue.set_return_values([None])

    is_running_sequence = [True, False]
    original_get_method = fake_is_running_tracker.get

    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False

    fake_is_running_tracker.get = sequenced_get

    data_source._DataReaderSource__poll_for_data()

    assert fake_data_reader.on_data_ready_call_count == 0
    assert fake_queue.get_blocking_call_count == 1

    fake_is_running_tracker.get = original_get_method
    data_source._DataReaderSource__is_running = original_tracker


def test_poll_for_data_multiple_items_then_stop(
    data_source, fake_queue, fake_data_reader, fake_is_running_tracker
):
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    item1 = FakeExposedData()
    item2 = FakeExposedData()
    fake_queue.set_return_values([item1, item2])

    is_running_sequence = [True, True, False]
    original_get_method = fake_is_running_tracker.get

    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False

    fake_is_running_tracker.get = sequenced_get

    data_source._DataReaderSource__poll_for_data()

    assert fake_data_reader.on_data_ready_call_count == 2
    assert fake_data_reader.on_data_ready_called_with[0] is item1
    assert fake_data_reader.on_data_ready_called_with[1] is item2
    assert fake_queue.get_blocking_call_count == 2

    fake_is_running_tracker.get = original_get_method
    data_source._DataReaderSource__is_running = original_tracker


def test_poll_for_data_stop_mid_processing(
    data_source, fake_queue, fake_data_reader, fake_is_running_tracker
):
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    item1 = FakeExposedData()
    item2 = FakeExposedData()
    fake_queue.set_return_values([item1, item2])

    fake_is_running_tracker.set_is_running(True)

    original_get_blocking = fake_queue.get_blocking
    stop_was_called_during_get = False

    def get_blocking_and_stop(
        timeout=0.1,
    ):  # Changed: timeout_seconds to timeout
        nonlocal stop_was_called_during_get
        data = original_get_blocking(
            timeout=timeout
        )  # Use corrected signature
        if data is item1 and not stop_was_called_during_get:
            data_source.stop()
            stop_was_called_during_get = True
        return data

    fake_queue.get_blocking = get_blocking_and_stop

    data_source._DataReaderSource__poll_for_data()

    assert fake_data_reader.on_data_ready_call_count == 1
    assert fake_data_reader.on_data_ready_called_with[0] is item1
    assert fake_queue.get_blocking_call_count == 1
    assert fake_is_running_tracker.get() is False
    assert stop_was_called_during_get is True

    fake_queue.get_blocking = original_get_blocking
    data_source._DataReaderSource__is_running = original_tracker


# Test default timeout for get_blocking
def test_poll_for_data_default_timeout(
    data_source, fake_queue, fake_is_running_tracker
):
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    captured_timeout = None
    original_get_blocking = fake_queue.get_blocking

    def get_blocking_capture_timeout(
        timeout=0.1,
    ):  # Ensure this matches the fake's signature
        nonlocal captured_timeout
        captured_timeout = timeout
        return original_get_blocking(
            timeout=timeout
        )  # Call with corrected signature

    fake_queue.get_blocking = get_blocking_capture_timeout

    is_running_sequence = [True, False]
    original_get_method = fake_is_running_tracker.get

    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False

    fake_is_running_tracker.get = sequenced_get

    fake_queue.set_return_values([None])
    data_source._DataReaderSource__poll_for_data()

    assert (
        captured_timeout == 1
    )  # Default timeout in DataReaderSource.__poll_for_data is 1

    fake_queue.get_blocking = original_get_blocking
    fake_is_running_tracker.get = original_get_method
    data_source._DataReaderSource__is_running = original_tracker


# Test that thread is created as daemon
def test_start_creates_daemon_thread(
    data_source, fake_watcher
):  # fake_is_running_tracker not needed here
    # No need to replace internal tracker if we are just checking thread daemon status
    data_source.start()
    assert fake_watcher.fake_thread is not None
    assert (
        fake_watcher.fake_thread.daemon is True
    )  # Default for create_tracked_thread is True


def test_stop_join_timeout(
    data_source, fake_watcher, fake_is_running_tracker, mocker
):
    """
    Tests that stop() raises RuntimeError if the polling thread does not join
    within the timeout.
    """
    # Replace the internal tracker with our fake one
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    # Call start() to create and start the thread
    fake_is_running_tracker.set_is_running(
        True
    )  # So start() thinks it can run
    data_source.start()

    assert (
        fake_watcher.fake_thread is not None
    ), "Thread should have been created"

    # Mock thread.join to do nothing (simulating it never finishes on its own)
    mocker.patch.object(
        fake_watcher.fake_thread, "join", side_effect=lambda timeout=None: None
    )
    # Mock thread.is_alive to always return True
    mocker.patch.object(
        fake_watcher.fake_thread, "is_alive", return_value=True
    )

    # Updated regex to match the actual error message more broadly
    with pytest.raises(
        RuntimeError,
        match=r"DataReaderSource thread for queue .* did not terminate within 5 seconds.",
    ):
        data_source.stop()  # This call will set __is_running to False and then try to join

    # Ensure join was called with the correct timeout
    fake_watcher.fake_thread.join.assert_called_once_with(timeout=5.0)
    # is_alive would have been checked after join
    fake_watcher.fake_thread.is_alive.assert_called_once()

    # Restore original tracker
    data_source._DataReaderSource__is_running = original_tracker


def test_poll_for_data_reader_on_data_ready_exception(
    data_source,
    fake_queue,
    fake_data_reader,
    fake_watcher,
    fake_is_running_tracker,
    mocker,
):
    """
    Tests that if data_reader._on_data_ready raises an exception, it's caught,
    reported to the watcher, and the poll loop terminates.
    """
    original_tracker = data_source._DataReaderSource__is_running
    data_source._DataReaderSource__is_running = fake_is_running_tracker

    test_exception = ValueError("Reader error")
    fake_data_reader._on_data_ready = mocker.Mock(
        side_effect=test_exception
    )  # More robust mock

    test_item = FakeExposedData()
    fake_queue.set_return_values([test_item])

    # Ensure the loop runs once then stops to check the exception effect
    is_running_sequence = [True, False]
    # We need to control the fake_is_running_tracker's get method
    original_get_method = fake_is_running_tracker.get

    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False  # pragma: no cover

    fake_is_running_tracker.get = sequenced_get

    # Ensure fake_watcher.on_exception_seen is a mock for assertion
    # (It should be by default now due to change in FakeThreadWatcher.__init__)
    # If it was None due to import error (unlikely in test env), this would fail.
    assert hasattr(fake_watcher, "on_exception_seen") and callable(
        fake_watcher.on_exception_seen.assert_called_once_with
    )

    with pytest.raises(ValueError, match="Reader error"):
        data_source._DataReaderSource__poll_for_data()

    fake_data_reader._on_data_ready.assert_called_once_with(test_item)
    fake_watcher.on_exception_seen.assert_called_once_with(test_exception)

    # Restore original get method and tracker
    fake_is_running_tracker.get = original_get_method
    data_source._DataReaderSource__is_running = original_tracker
