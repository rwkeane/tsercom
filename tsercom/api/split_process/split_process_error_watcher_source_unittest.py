import pytest
import importlib

# Module to be tested & whose attributes will be patched
import tsercom.api.split_process.split_process_error_watcher_source as spews_module
from tsercom.api.split_process.split_process_error_watcher_source import SplitProcessErrorWatcherSource
from tsercom.util.is_running_tracker import IsRunningTracker # For isinstance checks

# --- Fake Classes ---

class FakeThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon
        self._started = False

    def start(self):
        self._started = True


class FakeThreadWatcher:
    def __init__(self):
        self.on_exception_seen_called_with = None
        self.on_exception_seen_call_count = 0
        self.created_thread_target = None
        self.created_thread_args = None
        self.created_thread_daemon = None
        self.fake_thread = None

    def on_exception_seen(self, exception_instance):
        self.on_exception_seen_called_with = exception_instance
        self.on_exception_seen_call_count += 1

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
            if val is StopIteration: # Sentinel for loop control
                raise StopIteration("Test sentinel to stop queue reading")
            return val
        return None

    def set_return_values(self, values):
        self._return_values = list(values)
        self._return_idx = 0


class FakeIsRunningTracker:
    def __init__(self):
        self._is_running = False
        self.start_call_count = 0
        self.stop_call_count = 0
        self.get_call_count = 0

    def get(self):
        self.get_call_count += 1
        return self._is_running

    def is_running(self): # Some trackers might use this name
        self.get_call_count +=1
        return self._is_running

    def set_is_running(self, value: bool): # Test helper
        self._is_running = value

    def start(self):
        self.start_call_count += 1
        self._is_running = True

    def stop(self):
        self.stop_call_count += 1
        self._is_running = False


# --- Pytest Fixtures ---

@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()

@pytest.fixture
def fake_exception_queue():
    return FakeMultiprocessQueueSource()

@pytest.fixture
def patch_is_running_tracker_in_spews_module(request):
    """Monkeypatches IsRunningTracker in the split_process_error_watcher_source module's namespace."""
    original_class = getattr(spews_module, "IsRunningTracker", None)
    setattr(spews_module, "IsRunningTracker", FakeIsRunningTracker)
    def cleanup():
        if original_class: setattr(spews_module, "IsRunningTracker", original_class)
        elif hasattr(spews_module, "IsRunningTracker"): delattr(spews_module, "IsRunningTracker")
    request.addfinalizer(cleanup)


@pytest.fixture
def error_source(fake_watcher, fake_exception_queue, patch_is_running_tracker_in_spews_module):
    # patch_is_running_tracker_in_spews_module ensures FakeIsRunningTracker is used
    return SplitProcessErrorWatcherSource(
        thread_watcher=fake_watcher,
        exception_queue=fake_exception_queue
    )

# --- Unit Tests ---

def test_init(error_source, fake_watcher, fake_exception_queue):
    """Test SplitProcessErrorWatcherSource.__init__."""
    assert error_source._SplitProcessErrorWatcherSource__thread_watcher is fake_watcher # Corrected attribute name
    assert error_source._SplitProcessErrorWatcherSource__queue is fake_exception_queue # Corrected attribute name
    # Check that IsRunningTracker was instantiated (and is our Fake due to patch)
    internal_tracker = error_source._SplitProcessErrorWatcherSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)
    assert internal_tracker.get() is False # Should be False initially


def test_start_method(error_source, fake_watcher):
    """Test SplitProcessErrorWatcherSource.start()."""
    # patch_is_running_tracker_in_spews_module is auto-used by error_source fixture
    internal_tracker = error_source._SplitProcessErrorWatcherSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker) # Pre-condition from fixture

    error_source.start()

    assert internal_tracker.start_call_count == 1
    assert fake_watcher.created_thread_target is not None # Target is an inner function
    assert callable(fake_watcher.created_thread_target)
    assert fake_watcher.fake_thread is not None
    assert fake_watcher.fake_thread._started is True
    # Daemon status is not explicitly set by SplitProcessErrorWatcherSource, defaults to ThreadWatcher's behavior


def test_stop_method(error_source):
    """Test SplitProcessErrorWatcherSource.stop()."""
    # First, start it so __is_running is in a state to be stopped
    error_source.start() # This uses FakeIsRunningTracker due to fixture patch
    internal_tracker = error_source._SplitProcessErrorWatcherSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)
    assert internal_tracker.get() is True # Should be running after start

    error_source.stop()
    assert internal_tracker.stop_call_count == 1
    assert internal_tracker.get() is False


def test_is_running_method(error_source):
    """Test SplitProcessErrorWatcherSource.is_running()."""
    internal_tracker = error_source._SplitProcessErrorWatcherSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)

    # Before start
    assert error_source.is_running() is False
    assert internal_tracker.get_call_count == 1 # is_running() calls get()

    # After start
    error_source.start()
    assert error_source.is_running() is True
    assert internal_tracker.get_call_count == 2 # get() called again (once in start, once in is_running)
                                               # Actually, start calls tracker.start() which sets internal bool.
                                               # is_running() calls get().
                                               # Let's re-verify get_call_count logic for FakeIsRunningTracker
                                               # FakeIsRunningTracker.start() doesn't call get().
                                               # So, after start(), first is_running() makes get_call_count = 2.

    # After stop
    error_source.stop()
    assert error_source.is_running() is False
    assert internal_tracker.get_call_count == 3 # get() called again
    

# Helper for testing loop_until_exception
def setup_loop_test(error_source, fake_watcher):
    error_source.start() # This starts the thread and sets up the tracker
    internal_tracker = error_source._SplitProcessErrorWatcherSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)
    loop_target = fake_watcher.created_thread_target
    assert callable(loop_target)
    internal_tracker.set_is_running(True) # Ensure loop can run
    return internal_tracker, loop_target


def test_loop_exception_received(error_source, fake_watcher, fake_exception_queue):
    """Scenario 1: Exception received from queue."""
    internal_tracker, loop_target = setup_loop_test(error_source, fake_watcher)
    test_exception = RuntimeError("Test Exception from Queue")
    fake_exception_queue.set_return_values([test_exception, StopIteration])

    with pytest.raises(StopIteration): # To stop the loop for the test
        loop_target()

    assert fake_watcher.on_exception_seen_call_count == 1
    assert fake_watcher.on_exception_seen_called_with is test_exception
    assert fake_exception_queue.get_blocking_call_count >= 1


def test_loop_queue_timeout(error_source, fake_watcher, fake_exception_queue):
    """Scenario 2: Queue timeout (returns None)."""
    internal_tracker, loop_target = setup_loop_test(error_source, fake_watcher)
    fake_exception_queue.set_return_values([None, StopIteration]) # Simulate timeout

    with pytest.raises(StopIteration):
        loop_target()

    assert fake_watcher.on_exception_seen_call_count == 0 # Should not be called
    assert fake_exception_queue.get_blocking_call_count >= 1


def test_loop_termination_on_is_running_false(error_source, fake_watcher, fake_exception_queue):
    """Scenario 3: Loop termination when is_running becomes False."""
    internal_tracker, loop_target = setup_loop_test(error_source, fake_watcher)
    test_exception = RuntimeError("Another Test Exception")
    fake_exception_queue.set_return_values([test_exception, test_exception]) # Provide some items

    # Simulate is_running: True for one item, then False
    is_running_sequence = [True, False] 
    original_get_method = internal_tracker.get
    
    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False # Default if sequence exhausted
    internal_tracker.get = sequenced_get
    
    loop_target() # Should process one item then exit gracefully

    assert fake_watcher.on_exception_seen_call_count == 1 # Processed one exception
    assert fake_watcher.on_exception_seen_called_with is test_exception
    assert fake_exception_queue.get_blocking_call_count == 1 # Read one item

    internal_tracker.get = original_get_method # Restore for other tests if any
    internal_tracker.set_is_running(False) # Ensure it's reset for other tests
