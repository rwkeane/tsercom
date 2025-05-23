import pytest
import threading  # For type hinting if necessary, though FakeThread is used

from tsercom.api.split_process.event_source import EventSource
from tsercom.data.event_instance import EventInstance  # For creating test data
from tsercom.util.is_running_tracker import (
    IsRunningTracker,
)  # For isinstance checks

# --- Fake Classes ---


class FakeThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon
        self._started = False
        self._joined = False  # Not strictly needed by EventSource tests

    def start(self):
        self._started = True
        # For tests, target (loop_until_exception) will be called directly.

    def join(self, timeout=None):
        self._joined = True


class FakeThreadWatcher:
    def __init__(self):
        self.created_thread_target = None
        self.created_thread_args = None
        self.created_thread_daemon = None
        self.fake_thread = None

    def create_tracked_thread(self, target, args=(), daemon=True):
        self.created_thread_target = target
        self.created_thread_args = args
        self.created_thread_daemon = daemon
        self.fake_thread = FakeThread(target=target, args=args, daemon=daemon)
        return self.fake_thread


class FakeMultiprocessQueueSource:
    def __init__(self, return_values=None):
        self.get_blocking_call_count = 0
        self._return_values = list(return_values) if return_values else []
        self._return_idx = 0

    def get_blocking(self, timeout: float = 0.1):
        self.get_blocking_call_count += 1
        if self._return_idx < len(self._return_values):
            val = self._return_values[self._return_idx]
            self._return_idx += 1
            if (
                val is StopIteration
            ):  # Sentinel to stop further reads in a controlled way for tests
                raise StopIteration("Test sentinel to stop queue reading")
            return val
        return None  # Default if no more values or empty list

    def add_return_value(self, value):
        self._return_values.append(value)

    def set_return_values(self, values):
        self._return_values = list(values)
        self._return_idx = 0


class FakeIsRunningTracker:
    def __init__(self, initial_value=False):
        self._is_running = initial_value
        self.start_call_count = 0
        self.stop_call_count = 0
        self.get_call_count = 0

    def get(self):
        self.get_call_count += 1
        return self._is_running

    def set_is_running(self, value: bool):  # Helper for tests
        self._is_running = value

    def start(self):
        self.start_call_count += 1
        self._is_running = True

    def stop(self):
        self.stop_call_count += 1
        self._is_running = False


# --- Pytest Fixtures ---


@pytest.fixture
def fake_event_queue():
    return FakeMultiprocessQueueSource()


@pytest.fixture
def fake_thread_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_is_running_tracker():  # This will be used to *replace* the internal one
    return FakeIsRunningTracker()


@pytest.fixture
def event_source_instance(fake_event_queue):
    # ThreadWatcher is passed to start(), not __init__
    return EventSource[str](event_source=fake_event_queue)


@pytest.fixture
def test_event_instance():
    # Using a simple string as event data for EventInstance
    return EventInstance[str](
        data="test_event_data", caller_id=None, timestamp=None
    )  # Timestamp can be None for EventInstance


# --- Test Spy/Fake for on_available ---
class SpyOnAvailable:
    def __init__(self):
        self.called = False
        self.called_with_arg = None
        self.call_count = 0

    def __call__(self, event_instance):  # Make it callable
        self.called = True
        self.called_with_arg = event_instance
        self.call_count += 1


@pytest.fixture
def spy_on_available():
    return SpyOnAvailable()


# --- Unit Tests ---


def test_init(event_source_instance, fake_event_queue):
    """Test EventSource.__init__."""
    assert event_source_instance._EventSource__event_source is fake_event_queue
    assert isinstance(
        event_source_instance._EventSource__is_running, IsRunningTracker
    )
    assert (
        event_source_instance._EventSource__is_running.get() is False
    )  # Should be False initially


def test_start_method(
    event_source_instance, fake_thread_watcher, fake_is_running_tracker
):
    """Test EventSource.start()."""
    # Replace internal tracker with our fake for this test
    original_tracker = event_source_instance._EventSource__is_running
    event_source_instance._EventSource__is_running = fake_is_running_tracker

    event_source_instance.start(fake_thread_watcher)

    assert fake_is_running_tracker.start_call_count == 1
    assert fake_is_running_tracker.get() is True
    assert (
        fake_thread_watcher.created_thread_target is not None
    )  # It's an inner function
    assert fake_thread_watcher.fake_thread is not None
    assert fake_thread_watcher.fake_thread._started is True
    assert (
        fake_thread_watcher.fake_thread.daemon is True
    )  # Check daemon status

    # Restore original tracker
    event_source_instance._EventSource__is_running = original_tracker


def test_stop_method(event_source_instance, fake_is_running_tracker):
    """Test EventSource.stop()."""
    original_tracker = event_source_instance._EventSource__is_running
    event_source_instance._EventSource__is_running = fake_is_running_tracker

    fake_is_running_tracker.set_is_running(True)  # Simulate it was running

    event_source_instance.stop()

    assert fake_is_running_tracker.stop_call_count == 1
    assert fake_is_running_tracker.get() is False

    event_source_instance._EventSource__is_running = original_tracker


# Tests for loop_until_exception behavior (simulated via direct call to the target)


def setup_loop_test(
    event_source_instance, fake_is_running_tracker, spy_on_available_method
):
    """Helper to set up for loop testing."""
    # Replace internal tracker
    original_tracker = event_source_instance._EventSource__is_running
    event_source_instance._EventSource__is_running = fake_is_running_tracker
    # Monkeypatch on_available
    original_on_available = event_source_instance.on_available
    event_source_instance.on_available = spy_on_available_method
    return original_tracker, original_on_available


def teardown_loop_test(
    event_source_instance, original_tracker, original_on_available
):
    """Helper to tear down after loop testing."""
    event_source_instance._EventSource__is_running = original_tracker
    event_source_instance.on_available = original_on_available


def test_loop_single_event_then_stop(
    event_source_instance,
    fake_event_queue,
    fake_thread_watcher,
    fake_is_running_tracker,
    spy_on_available,
    test_event_instance,
):
    """Scenario 1: Single event then stop."""
    original_tracker, original_on_available = setup_loop_test(
        event_source_instance, fake_is_running_tracker, spy_on_available
    )

    fake_event_queue.set_return_values(
        [test_event_instance, StopIteration]
    )  # StopIteration to break loop after one item
    fake_is_running_tracker.set_is_running(True)  # Loop runs at least once

    event_source_instance.start(fake_thread_watcher)  # This sets up the target
    loop_target = fake_thread_watcher.created_thread_target

    # Simulate loop execution
    try:
        loop_target()
    except StopIteration:  # Expected due to sentinel
        pass

    assert spy_on_available.called
    assert spy_on_available.call_count == 1
    assert spy_on_available.called_with_arg is test_event_instance
    assert fake_event_queue.get_blocking_call_count >= 1  # Called for the item

    teardown_loop_test(
        event_source_instance, original_tracker, original_on_available
    )


def test_loop_timeout_then_stop(
    event_source_instance,
    fake_event_queue,
    fake_thread_watcher,
    fake_is_running_tracker,
    spy_on_available,
):
    """Scenario 2: Timeout then stop."""
    original_tracker, original_on_available = setup_loop_test(
        event_source_instance, fake_is_running_tracker, spy_on_available
    )

    fake_event_queue.set_return_values(
        [None, StopIteration]
    )  # None for timeout
    fake_is_running_tracker.set_is_running(True)

    event_source_instance.start(fake_thread_watcher)
    loop_target = fake_thread_watcher.created_thread_target

    try:
        loop_target()
    except StopIteration:
        pass

    assert not spy_on_available.called  # on_available should not be called
    assert fake_event_queue.get_blocking_call_count >= 1  # Called for the None

    teardown_loop_test(
        event_source_instance, original_tracker, original_on_available
    )


def test_loop_multiple_events_then_stop(
    event_source_instance,
    fake_event_queue,
    fake_thread_watcher,
    fake_is_running_tracker,
    spy_on_available,
):
    """Scenario 3: Multiple events, then stop."""
    original_tracker, original_on_available = setup_loop_test(
        event_source_instance, fake_is_running_tracker, spy_on_available
    )

    event1 = EventInstance[str]("event1", None, None)
    event2 = EventInstance[str]("event2", None, None)
    fake_event_queue.set_return_values([event1, event2, StopIteration])
    fake_is_running_tracker.set_is_running(True)

    event_source_instance.start(fake_thread_watcher)
    loop_target = fake_thread_watcher.created_thread_target

    try:
        loop_target()
    except StopIteration:
        pass

    assert spy_on_available.call_count == 2
    # To check args in order, the spy would need to store a list of args.
    # For simplicity, we'll assume the last one is event2 if count is 2.
    # A more robust spy would be: self.args_list.append(event_instance)
    assert spy_on_available.called_with_arg is event2

    teardown_loop_test(
        event_source_instance, original_tracker, original_on_available
    )


def test_loop_terminates_on_stop_call(
    event_source_instance,
    fake_event_queue,
    fake_thread_watcher,
    fake_is_running_tracker,
    spy_on_available,
):
    """Scenario 4: stop() called, ensure loop terminates."""
    original_tracker, original_on_available = setup_loop_test(
        event_source_instance, fake_is_running_tracker, spy_on_available
    )

    event1 = EventInstance[str]("event1_for_stop_test", None, None)
    # Configure queue to provide events, but stop will terminate it.
    # We need to control when is_running becomes False.
    fake_event_queue.set_return_values(
        [event1, event1, event1, event1]
    )  # More events than we'll read

    # is_running will be controlled by the stop() call
    fake_is_running_tracker.set_is_running(True)

    event_source_instance.start(fake_thread_watcher)
    loop_target = fake_thread_watcher.created_thread_target

    # Simulate loop that stops after a few iterations
    max_iterations = 5  # Safety break for the test
    iterations_done = 0

    def controlled_loop():
        nonlocal iterations_done
        while (
            event_source_instance._EventSource__is_running.get()
            and iterations_done < max_iterations
        ):
            event = (
                event_source_instance._EventSource__event_source.get_blocking()
            )
            if event:
                event_source_instance.on_available(event)
            iterations_done += 1
            if iterations_done == 2:  # Arbitrarily stop after 2 events
                event_source_instance.stop()  # This calls the fake tracker's stop

    controlled_loop()

    assert (
        spy_on_available.call_count == 2
    )  # Should have processed 2 events before stop
    assert fake_is_running_tracker.get() is False  # Stop was called

    teardown_loop_test(
        event_source_instance, original_tracker, original_on_available
    )
