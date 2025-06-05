import pytest

from tsercom.api.split_process.split_process_error_watcher_sink import (
    SplitProcessErrorWatcherSink,
)

# --- Fake Classes ---


class FakeThreadWatcher:
    def __init__(self):
        self.run_until_exception_called = False
        self.run_until_exception_call_count = 0
        self._exception_to_raise = None

    def run_until_exception(self):
        self.run_until_exception_called = True
        self.run_until_exception_call_count += 1
        if self._exception_to_raise:
            raise self._exception_to_raise

    def set_exception_to_raise(self, exception_instance):
        self._exception_to_raise = exception_instance


class FakeMultiprocessQueueSink:
    def __init__(self):
        self.put_nowait_called_with = None
        self.put_nowait_call_count = 0
        # put_nowait in this context (for exceptions) usually doesn't care about return value
        # as it's a "best effort" to report the error.

    def put_nowait(self, data):
        self.put_nowait_called_with = data
        self.put_nowait_call_count += 1
        # The real queue returns a bool, but it's not checked by SplitProcessErrorWatcherSink


# --- Pytest Fixtures ---


@pytest.fixture
def fake_thread_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_exception_queue():
    return FakeMultiprocessQueueSink()


@pytest.fixture
def error_watcher_sink(fake_thread_watcher, fake_exception_queue):
    return SplitProcessErrorWatcherSink(
        thread_watcher=fake_thread_watcher,
        exception_queue=fake_exception_queue,
    )


# --- Unit Tests ---


def test_init(error_watcher_sink, fake_thread_watcher, fake_exception_queue):
    """Test SplitProcessErrorWatcherSink.__init__."""
    assert (
        error_watcher_sink._SplitProcessErrorWatcherSink__thread_watcher
        is fake_thread_watcher
    )  # Corrected name
    assert (
        error_watcher_sink._SplitProcessErrorWatcherSink__queue
        is fake_exception_queue
    )  # Corrected name


def test_run_until_exception_no_exception_scenario(
    error_watcher_sink, fake_thread_watcher, fake_exception_queue
):
    """Test run_until_exception() when no exception is raised by the watcher."""
    # Configure watcher to not raise an error (default for FakeThreadWatcher)
    # fake_thread_watcher.set_exception_to_raise(None) # Default behavior

    error_watcher_sink.run_until_exception()

    assert fake_thread_watcher.run_until_exception_called
    assert fake_thread_watcher.run_until_exception_call_count == 1
    assert (
        fake_exception_queue.put_nowait_call_count == 0
    )  # Should NOT be called


def test_run_until_exception_with_exception_scenario(
    error_watcher_sink, fake_thread_watcher, fake_exception_queue
):
    """Test run_until_exception() when an exception is raised by the watcher."""
    test_exception = RuntimeError("Test Error")
    fake_thread_watcher.set_exception_to_raise(test_exception)

    with pytest.raises(RuntimeError, match="Test Error") as excinfo:
        error_watcher_sink.run_until_exception()

    # Verify the correct exception was raised
    assert excinfo.value is test_exception

    # Verify ThreadWatcher.run_until_exception was called
    assert fake_thread_watcher.run_until_exception_called
    assert fake_thread_watcher.run_until_exception_call_count == 1

    # Verify exception_queue.put_nowait was called with the exception
    assert fake_exception_queue.put_nowait_call_count == 1
    assert fake_exception_queue.put_nowait_called_with is test_exception


def test_run_until_exception_queue_put_fails(
    error_watcher_sink, fake_thread_watcher, fake_exception_queue, mocker
):
    """
    Tests that run_until_exception re-raises the original exception even if
    reporting it to the queue fails.
    """
    original_exception = SystemError("Original error from watcher")
    fake_thread_watcher.set_exception_to_raise(original_exception)

    queue_exception = ValueError("Queue put failed")
    # Patch the put_nowait method on the specific instance of FakeMultiprocessQueueSink
    fake_exception_queue.put_nowait = mocker.MagicMock(
        side_effect=queue_exception
    )

    # Expect the original_exception to be re-raised
    with pytest.raises(
        SystemError, match="Original error from watcher"
    ) as excinfo:
        error_watcher_sink.run_until_exception()

    assert excinfo.value is original_exception

    # Assert that put_nowait was called with the original_exception
    fake_exception_queue.put_nowait.assert_called_once_with(original_exception)

    # Logging of the queue_exception is done by the main library, not asserted here
    # unless we explicitly patch and check the logger within SplitProcessErrorWatcherSink.
    # The primary check is that the original error propagates.
