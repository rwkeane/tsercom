import pytest

# Import the module to be tested and whose attributes will be patched
import tsercom.api.split_process.runtime_command_source as rcs_module
from tsercom.api.split_process.runtime_command_source import (
    RuntimeCommandSource,
)
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.runtime.runtime import Runtime

# --- Fake Classes ---


class FakeRuntime(Runtime):
    def __init__(self):
        self.start_async_called = False
        self.start_async_call_count = 0
        self.stop_called = False
        self.stop_call_count = 0

    def start_async(self):
        self.start_async_called = True
        self.start_async_call_count += 1

    def stop(self, exception):
        self.stop_called = True
        self.stop_call_count += 1


class FakeMultiprocessQueueSource:
    def __init__(self, return_values=None):
        self.get_blocking_call_count = 0
        self._return_values = list(return_values) if return_values else []
        self._return_idx = 0

    def get_blocking(
        self, timeout: float = 0.1
    ):  # timeout name matches usage in RuntimeCommandSource
        self.get_blocking_call_count += 1
        if self._return_idx < len(self._return_values):
            val = self._return_values[self._return_idx]
            self._return_idx += 1
            if val is StopIteration:  # Sentinel for loop control in tests
                raise StopIteration("Test sentinel to stop queue reading")
            return val
        return None

    def add_return_value(self, value):
        self._return_values.append(value)

    def set_return_values(self, values):
        self._return_values = list(values)
        self._return_idx = 0


class FakeThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon  # RuntimeCommandSource sets daemon=True
        self._started = False
        self.join_called = False
        self.join_timeout = None

    def start(self):
        self._started = True
        self._is_alive = True  # Assume alive once started

    def join(self, timeout=None):
        self.join_called = True
        self.join_timeout = timeout
        # In a fake, we don't actually block or join anything.
        # Typically, after join (even with timeout), is_alive would be false if thread finished.
        self._is_alive = False
        pass

    def is_alive(self):
        return self._is_alive


class FakeThreadWatcher:
    def __init__(self):
        self.created_thread_target = None
        self.created_thread_args = None
        self.created_thread_daemon = None
        self.fake_thread = None
        self.last_exception_seen = None  # For on_exception_seen

    def create_tracked_thread(self, target, args=(), daemon=True):
        self.created_thread_target = target
        self.created_thread_args = args
        self.created_thread_daemon = daemon
        self.fake_thread = FakeThread(target=target, args=args, daemon=daemon)
        return self.fake_thread

    def on_exception_seen(self, exception: Exception) -> None:
        """Records the exception seen by the thread watcher."""
        self.last_exception_seen = exception


class FakeIsRunningTracker:  # No global variable needed here
    def __init__(self, name="default_fake_tracker"):
        self.name = name
        self._is_running = False
        self.start_call_count = 0
        self.stop_call_count = 0
        self.get_call_count = 0

    def get(self):
        self.get_call_count += 1
        return self._is_running

    def set_is_running(self, value: bool):  # Test helper
        self._is_running = value

    def start(self):
        self.start_call_count += 1
        self._is_running = True

    def stop(self):
        self.stop_call_count += 1
        self._is_running = False


# --- Pytest Fixtures ---


@pytest.fixture
def fake_command_queue():
    return FakeMultiprocessQueueSource()


@pytest.fixture
def fake_runtime_instance():
    return FakeRuntime()


@pytest.fixture
def fake_thread_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def patch_run_on_event_loop_in_rcs_module(request):
    """Monkeypatches run_on_event_loop in the runtime_command_source module."""
    original_func = getattr(rcs_module, "run_on_event_loop", None)
    captured_callables = []

    def fake_run_on_event_loop(callable_to_run, *args, **kwargs):
        captured_callables.append(callable_to_run)
        if callable(callable_to_run):
            callable_to_run()

    setattr(rcs_module, "run_on_event_loop", fake_run_on_event_loop)

    def cleanup():
        if original_func:
            setattr(rcs_module, "run_on_event_loop", original_func)
        elif hasattr(
            rcs_module, "run_on_event_loop"
        ):  # If we added it, delete it
            delattr(rcs_module, "run_on_event_loop")

    request.addfinalizer(cleanup)
    return captured_callables


@pytest.fixture
def patch_is_running_tracker_in_rcs_module(request):
    """Monkeypatches IsRunningTracker in the runtime_command_source module's namespace."""
    original_class = getattr(rcs_module, "IsRunningTracker", None)
    setattr(rcs_module, "IsRunningTracker", FakeIsRunningTracker)

    def cleanup():
        if original_class:
            setattr(rcs_module, "IsRunningTracker", original_class)
        elif hasattr(
            rcs_module, "IsRunningTracker"
        ):  # If we added it, delete it
            delattr(rcs_module, "IsRunningTracker")

    request.addfinalizer(cleanup)


@pytest.fixture
def command_source(fake_command_queue):
    # patch_is_running_tracker_in_rcs_module will be applied by tests that need it
    source = RuntimeCommandSource(runtime_command_queue=fake_command_queue)
    return source


# --- Unit Tests ---


def test_init(command_source, fake_command_queue):
    """Test RuntimeCommandSource.__init__."""
    assert (
        command_source._RuntimeCommandSource__runtime_command_queue
        is fake_command_queue
    )
    assert command_source._RuntimeCommandSource__is_running is None
    assert command_source._RuntimeCommandSource__runtime is None


def test_start_async(
    command_source,
    fake_thread_watcher,
    fake_runtime_instance,
    patch_is_running_tracker_in_rcs_module,
):
    """Test start_async() method."""
    # patch_is_running_tracker_in_rcs_module ensures IsRunningTracker class is FakeIsRunningTracker for this test

    command_source.start_async(
        fake_thread_watcher, fake_runtime_instance
    )  # FakeIsRunningTracker is created here

    internal_fake_tracker = command_source._RuntimeCommandSource__is_running
    assert internal_fake_tracker is not None, (
        "Internal tracker not set after start_async"
    )
    assert isinstance(internal_fake_tracker, FakeIsRunningTracker), (
        "Internal tracker is not a FakeIsRunningTracker"
    )

    assert (
        command_source._RuntimeCommandSource__runtime is fake_runtime_instance
    )
    assert internal_fake_tracker.start_call_count == 1

    assert callable(
        fake_thread_watcher.created_thread_target
    )  # Check if a function was set
    assert fake_thread_watcher.fake_thread is not None
    assert fake_thread_watcher.fake_thread._started is True
    assert fake_thread_watcher.fake_thread.daemon is True

    # Test calling start_async twice
    with pytest.raises(AssertionError):
        command_source.start_async(fake_thread_watcher, fake_runtime_instance)


def test_stop_async(
    command_source,
    fake_thread_watcher,
    fake_runtime_instance,
    patch_is_running_tracker_in_rcs_module,
):
    """Test stop_async() method."""
    command_source.start_async(fake_thread_watcher, fake_runtime_instance)
    internal_fake_tracker = command_source._RuntimeCommandSource__is_running
    assert isinstance(internal_fake_tracker, FakeIsRunningTracker), (
        "Internal tracker is not a FakeIsRunningTracker"
    )

    command_source.stop_async()
    assert internal_fake_tracker.stop_call_count == 1


def test_stop_async_before_start_raises_assertion_error(command_source):
    """Test stop_async() raises AssertionError if called before start_async."""
    with pytest.raises(AssertionError):
        command_source.stop_async()


# Helper for watch_commands tests
def setup_watch_commands_test(
    command_source, fake_thread_watcher, fake_runtime_instance
):
    # patch_is_running_tracker_in_rcs_module should be active in the calling test function
    command_source.start_async(fake_thread_watcher, fake_runtime_instance)
    internal_fake_tracker = command_source._RuntimeCommandSource__is_running
    assert isinstance(internal_fake_tracker, FakeIsRunningTracker), (
        "Internal tracker is not FakeIsRunningTracker in setup. Ensure calling test uses patch_is_running_tracker_in_rcs_module."
    )
    loop_target_function = fake_thread_watcher.created_thread_target
    assert callable(loop_target_function)
    return internal_fake_tracker, loop_target_function


def test_watch_commands_kstart(
    command_source,
    fake_command_queue,
    fake_runtime_instance,
    fake_thread_watcher,
    patch_run_on_event_loop_in_rcs_module,
    patch_is_running_tracker_in_rcs_module,
):
    internal_fake_tracker, loop_target = setup_watch_commands_test(
        command_source, fake_thread_watcher, fake_runtime_instance
    )

    fake_command_queue.set_return_values(
        [RuntimeCommand.kStart, StopIteration]
    )
    internal_fake_tracker.set_is_running(True)

    with pytest.raises(StopIteration):
        loop_target()  # Call the actual thread target

    assert len(patch_run_on_event_loop_in_rcs_module) == 1
    assert fake_runtime_instance.start_async_called
    assert fake_runtime_instance.start_async_call_count == 1


def test_watch_commands_kstop(
    command_source,
    fake_command_queue,
    fake_runtime_instance,
    fake_thread_watcher,
    patch_run_on_event_loop_in_rcs_module,
    patch_is_running_tracker_in_rcs_module,
):
    internal_fake_tracker, loop_target = setup_watch_commands_test(
        command_source, fake_thread_watcher, fake_runtime_instance
    )

    fake_command_queue.set_return_values([RuntimeCommand.kStop, StopIteration])
    internal_fake_tracker.set_is_running(True)

    with pytest.raises(StopIteration):
        loop_target()  # Call the actual thread target

    assert len(patch_run_on_event_loop_in_rcs_module) == 1
    assert fake_runtime_instance.stop_called
    assert fake_runtime_instance.stop_call_count == 1


def test_watch_commands_unknown_command_raises_value_error(
    command_source,
    fake_command_queue,
    fake_runtime_instance,
    fake_thread_watcher,
    patch_run_on_event_loop_in_rcs_module,
    patch_is_running_tracker_in_rcs_module,
):
    internal_fake_tracker, loop_target = setup_watch_commands_test(
        command_source, fake_thread_watcher, fake_runtime_instance
    )

    fake_command_queue.set_return_values([-1, StopIteration])
    internal_fake_tracker.set_is_running(True)

    with pytest.raises(ValueError):
        loop_target()  # Call the actual thread target

    assert len(patch_run_on_event_loop_in_rcs_module) == 0


def test_watch_commands_queue_timeout(
    command_source,
    fake_command_queue,
    fake_runtime_instance,
    fake_thread_watcher,
    patch_run_on_event_loop_in_rcs_module,
    patch_is_running_tracker_in_rcs_module,
):
    internal_fake_tracker, loop_target = setup_watch_commands_test(
        command_source, fake_thread_watcher, fake_runtime_instance
    )

    fake_command_queue.set_return_values([None, StopIteration])
    internal_fake_tracker.set_is_running(True)

    with pytest.raises(StopIteration):
        loop_target()  # Call the actual thread target

    assert len(patch_run_on_event_loop_in_rcs_module) == 0


def test_watch_commands_loop_termination(
    command_source,
    fake_command_queue,
    fake_runtime_instance,
    fake_thread_watcher,
    patch_run_on_event_loop_in_rcs_module,
    patch_is_running_tracker_in_rcs_module,
):
    internal_fake_tracker, loop_target = setup_watch_commands_test(
        command_source, fake_thread_watcher, fake_runtime_instance
    )

    fake_command_queue.set_return_values(
        [RuntimeCommand.kStart, RuntimeCommand.kStart]
    )

    is_running_sequence = [True, True, False]
    original_get_method = internal_fake_tracker.get

    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False

    internal_fake_tracker.get = sequenced_get

    loop_target()  # Call the actual thread target

    assert fake_runtime_instance.start_async_call_count == 2
    assert fake_command_queue.get_blocking_call_count == 2

    internal_fake_tracker.get = original_get_method
    internal_fake_tracker.set_is_running(False)
