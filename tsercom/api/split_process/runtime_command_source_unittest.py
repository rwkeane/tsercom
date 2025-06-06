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
        self._is_alive = False  # Initial state

    def start(self):
        self._started = True
        self._is_alive = True  # Assume alive once started

    def join(self, timeout=None):
        self.join_called = True
        self.join_timeout = timeout
        self._is_alive = False  # Simulate thread joined
        pass

    def is_alive(self):
        return self._is_alive


class FakeThreadWatcher:
    def __init__(self):
        self.created_thread_target = None
        self.created_thread_args = None
        self.created_thread_daemon = None
        self.fake_thread = None
        try:
            from unittest.mock import MagicMock

            self.on_exception_seen = MagicMock()
        except ImportError:  # pragma: no cover
            self.on_exception_seen = None

    def create_tracked_thread(self, target, args=(), daemon=True):
        self.created_thread_target = target
        self.created_thread_args = args
        self.created_thread_daemon = daemon
        self.fake_thread = FakeThread(target=target, args=args, daemon=daemon)
        return self.fake_thread

    # This method was defined twice in the read_files output, keeping one.
    # def on_exception_seen(self, exception: Exception) -> None:
    #     """Records the exception seen by the thread watcher."""
    #     self.last_exception_seen = exception


class FakeIsRunningTracker:
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
        elif hasattr(rcs_module, "run_on_event_loop"):
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
        elif hasattr(rcs_module, "IsRunningTracker"):
            delattr(rcs_module, "IsRunningTracker")

    request.addfinalizer(cleanup)


@pytest.fixture
def command_source(fake_command_queue):
    source = RuntimeCommandSource(runtime_command_queue=fake_command_queue)
    return source


# --- Unit Tests ---


def test_init(command_source, fake_command_queue):
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
    command_source.start_async(fake_thread_watcher, fake_runtime_instance)
    internal_fake_tracker = command_source._RuntimeCommandSource__is_running
    assert internal_fake_tracker is not None
    assert isinstance(internal_fake_tracker, FakeIsRunningTracker)
    assert (
        command_source._RuntimeCommandSource__runtime is fake_runtime_instance
    )
    assert internal_fake_tracker.start_call_count == 1
    assert callable(fake_thread_watcher.created_thread_target)
    assert fake_thread_watcher.fake_thread is not None
    assert fake_thread_watcher.fake_thread._started is True
    assert fake_thread_watcher.fake_thread.daemon is True
    with pytest.raises(AssertionError):
        command_source.start_async(fake_thread_watcher, fake_runtime_instance)


def test_stop_async(
    command_source,
    fake_thread_watcher,
    fake_runtime_instance,
    patch_is_running_tracker_in_rcs_module,
):
    command_source.start_async(fake_thread_watcher, fake_runtime_instance)
    internal_fake_tracker = command_source._RuntimeCommandSource__is_running
    assert isinstance(internal_fake_tracker, FakeIsRunningTracker)
    command_source.stop_async()
    assert internal_fake_tracker.stop_call_count == 1


def test_stop_async_before_start_raises_assertion_error(command_source):
    with pytest.raises(AssertionError):
        command_source.stop_async()


def setup_watch_commands_test(
    command_source, fake_thread_watcher, fake_runtime_instance
):
    command_source.start_async(fake_thread_watcher, fake_runtime_instance)
    internal_fake_tracker = command_source._RuntimeCommandSource__is_running
    assert isinstance(internal_fake_tracker, FakeIsRunningTracker)
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
    fake_command_queue.set_return_values([RuntimeCommand.START, StopIteration])
    internal_fake_tracker.set_is_running(True)
    with pytest.raises(StopIteration):
        loop_target()
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
    fake_command_queue.set_return_values([RuntimeCommand.STOP, StopIteration])
    internal_fake_tracker.set_is_running(True)
    with pytest.raises(StopIteration):
        loop_target()
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
        loop_target()
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
        loop_target()
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
        [RuntimeCommand.START, RuntimeCommand.START]
    )
    is_running_sequence = [True, True, False]
    original_get_method = internal_fake_tracker.get

    def sequenced_get():
        if is_running_sequence:
            return is_running_sequence.pop(0)
        return False

    internal_fake_tracker.get = sequenced_get
    loop_target()
    assert fake_runtime_instance.start_async_call_count == 2
    assert fake_command_queue.get_blocking_call_count == 2
    internal_fake_tracker.get = original_get_method
    internal_fake_tracker.set_is_running(False)


def test_stop_async_join_timeout(
    command_source,
    fake_thread_watcher,
    fake_runtime_instance,
    patch_is_running_tracker_in_rcs_module,
    mocker,
):
    """
    Tests that stop_async() raises RuntimeError if the command processing thread
    does not join within the timeout.
    """
    # Start the command source. This will initialize __is_running using the patched FakeIsRunningTracker
    command_source.start_async(fake_thread_watcher, fake_runtime_instance)

    # Now __is_running should be an instance of FakeIsRunningTracker
    assert isinstance(
        command_source._RuntimeCommandSource__is_running, FakeIsRunningTracker
    ), "Internal __is_running tracker should be a FakeIsRunningTracker instance after start_async"
    internal_fake_tracker = command_source._RuntimeCommandSource__is_running

    assert (
        internal_fake_tracker.get() is True
    ), "Tracker should be running after start_async"
    assert (
        fake_thread_watcher.fake_thread is not None
    ), "Thread should have been created"

    # Mock thread.join to do nothing (simulating it never finishes on its own)
    mocker.patch.object(
        fake_thread_watcher.fake_thread,
        "join",
        side_effect=lambda timeout=None: None,
    )
    # Mock thread.is_alive to always return True
    mocker.patch.object(
        fake_thread_watcher.fake_thread, "is_alive", return_value=True
    )

    with pytest.raises(
        RuntimeError,
        match=r"RuntimeCommandSource thread for queue .* did not terminate within 5 seconds.",
    ):
        command_source.stop_async()  # This will set __is_running to False and then try to join

    fake_thread_watcher.fake_thread.join.assert_called_once_with(timeout=5.0)
    fake_thread_watcher.fake_thread.is_alive.assert_called_once()


def test_watch_commands_runtime_command_exception(
    command_source,
    fake_command_queue,
    fake_runtime_instance,
    fake_thread_watcher,
    patch_run_on_event_loop_in_rcs_module,
    patch_is_running_tracker_in_rcs_module,
    mocker,
):
    """
    Tests that if a runtime command (e.g., runtime.start_async) raises an exception,
    it's caught, reported to the watcher, and the loop terminates by re-raising.
    """
    internal_fake_tracker, loop_target = setup_watch_commands_test(
        command_source, fake_thread_watcher, fake_runtime_instance
    )

    test_exception = ValueError("Runtime start_async error")
    fake_runtime_instance.start_async = mocker.MagicMock(
        side_effect=test_exception
    )

    fake_command_queue.set_return_values([RuntimeCommand.START, StopIteration])
    internal_fake_tracker.set_is_running(True)

    assert hasattr(fake_thread_watcher, "on_exception_seen") and callable(
        fake_thread_watcher.on_exception_seen.assert_called_once_with
    )

    with pytest.raises(ValueError, match="Runtime start_async error"):
        loop_target()

    fake_runtime_instance.start_async.assert_called_once()
    fake_thread_watcher.on_exception_seen.assert_called_once_with(
        test_exception
    )
    assert internal_fake_tracker.get() is True
