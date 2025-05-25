import pytest
import importlib

# Module to be tested
import tsercom.api.split_process.runtime_data_source as rds_module
from tsercom.api.split_process.runtime_data_source import RuntimeDataSource
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.util.is_running_tracker import (
    IsRunningTracker,
)  # For isinstance checks

# --- Fake Classes ---


class FakeRuntime:
    def __init__(self):
        self.start_async_called = False
        self.start_async_call_count = 0
        self.stop_called = False
        self.stop_call_count = 0
        self.on_event_called_with = None
        self.on_event_call_count = 0

    def start_async(self):
        self.start_async_called = True
        self.start_async_call_count += 1

    def stop(self):
        self.stop_called = True
        self.stop_call_count += 1

    def on_event(
        self, event_data
    ):  # Assuming event_data is the direct payload
        self.on_event_called_with = event_data
        self.on_event_call_count += 1


class FakeMultiprocessQueueSource:
    def __init__(self, name="FakeQueue", return_values=None):
        self.name = name
        self.get_blocking_call_count = 0
        self._return_values = list(return_values) if return_values else []
        self._return_idx = 0

    def get_blocking(self, timeout: float = 0.1):
        self.get_blocking_call_count += 1
        if self._return_idx < len(self._return_values):
            val = self._return_values[self._return_idx]
            self._return_idx += 1
            if val is StopIteration:  # Sentinel for loop control
                raise StopIteration(f"Test sentinel from {self.name}")
            return val
        return None

    def set_return_values(self, values):
        self._return_values = list(values)
        self._return_idx = 0


class FakeThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.submitted_tasks = []  # List of (function, args, kwargs)
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("ThreadPoolExecutor already shutdown")
        # For testing, execute immediately and store
        task_info = {"fn": fn, "args": args, "kwargs": kwargs, "result": None}
        try:
            # Execute immediately
            task_info["result"] = fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover
            task_info["exception"] = e
            # Depending on test needs, might re-raise or allow inspection
        self.submitted_tasks.append(task_info)

        # To mimic Future API if needed, but not required by these tests directly
        class FakeFuture:
            def result(self, timeout=None):
                if "exception" in task_info:
                    raise task_info["exception"]
                return task_info["result"]

        return FakeFuture()

    def shutdown(self, wait=True):
        self._shutdown = True

    @property
    def last_submitted_fn(self):
        return self.submitted_tasks[-1]["fn"] if self.submitted_tasks else None

    @property
    def last_submitted_args(self):
        return (
            self.submitted_tasks[-1]["args"]
            if self.submitted_tasks
            else tuple()
        )


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
        self.thread_pool_executor_created_with_max_workers = None
        self.fake_thread_pool_executor = None  # Store the one it "creates"

        self.created_threads_info = (
            []
        )  # List of dicts: {"target": t, "args": a, "daemon": d, "thread": ft}
        self.watch_commands_target = None
        self.watch_events_target = None

    def create_tracked_thread_pool_executor(self, max_workers):
        self.thread_pool_executor_created_with_max_workers = max_workers
        # Return a specific fake executor instance for tests to inspect
        self.fake_thread_pool_executor = FakeThreadPoolExecutor(
            max_workers=max_workers
        )
        return self.fake_thread_pool_executor

    def create_tracked_thread(self, target, args=(), daemon=True):
        fake_thread = FakeThread(target=target, args=args, daemon=daemon)
        thread_info = {
            "target": target,
            "args": args,
            "daemon": daemon,
            "thread": fake_thread,
        }
        self.created_threads_info.append(thread_info)

        # Try to identify targets based on typical usage in RuntimeDataSource
        # This is a bit heuristic for testability; depends on how distinct the targets are.
        # For this test, we'll assume the first thread created is watch_commands, second is watch_events.
        if len(self.created_threads_info) == 1:
            self.watch_commands_target = target
        elif len(self.created_threads_info) == 2:
            self.watch_events_target = target

        return fake_thread


class FakeIsRunningTracker:
    def __init__(self):
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
def fake_event_q_source():
    return FakeMultiprocessQueueSource(name="EventQ")


@pytest.fixture
def fake_command_q_source():
    return FakeMultiprocessQueueSource(name="CommandQ")


@pytest.fixture
def fake_runtime_instance():
    return FakeRuntime()


@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def patch_is_running_tracker_in_rds_module(request):
    """Monkeypatches IsRunningTracker in the runtime_data_source module's namespace."""
    original_class = getattr(rds_module, "IsRunningTracker", None)
    setattr(rds_module, "IsRunningTracker", FakeIsRunningTracker)

    def cleanup():
        if original_class:
            setattr(rds_module, "IsRunningTracker", original_class)
        elif hasattr(rds_module, "IsRunningTracker"):
            delattr(rds_module, "IsRunningTracker")

    request.addfinalizer(cleanup)


@pytest.fixture
def data_source(
    fake_watcher,
    fake_event_q_source,
    fake_command_q_source,
    patch_is_running_tracker_in_rds_module,
):
    # patch_is_running_tracker_in_rds_module ensures FakeIsRunningTracker is used
    return RuntimeDataSource(
        thread_watcher=fake_watcher,
        event_queue=fake_event_q_source,
        runtime_command_queue=fake_command_q_source,
    )


# --- Unit Tests ---


def test_init(
    data_source, fake_watcher, fake_event_q_source, fake_command_q_source
):
    assert data_source._RuntimeDataSource__thread_watcher is fake_watcher
    assert data_source._RuntimeDataSource__event_queue is fake_event_q_source
    assert (
        data_source._RuntimeDataSource__runtime_command_queue
        is fake_command_q_source
    )
    # IsRunningTracker is instantiated in __init__
    assert isinstance(
        data_source._RuntimeDataSource__is_running, FakeIsRunningTracker
    )  # Due to patch in data_source fixture
    assert (
        data_source._RuntimeDataSource__is_running.get() is False
    )  # Should be False initially
    assert data_source._RuntimeDataSource__thread_pool is None
    assert data_source._RuntimeDataSource__runtime is None


def test_start_async(data_source, fake_runtime_instance, fake_watcher):
    # patch_is_running_tracker_in_rds_module is auto-used by data_source fixture

    data_source.start_async(fake_runtime_instance)

    internal_tracker = data_source._RuntimeDataSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)
    assert internal_tracker.start_call_count == 1

    assert data_source._RuntimeDataSource__runtime is fake_runtime_instance
    assert fake_watcher.thread_pool_executor_created_with_max_workers == 1
    assert (
        data_source._RuntimeDataSource__thread_pool
        is fake_watcher.fake_thread_pool_executor
    )  # Check instance used

    assert len(fake_watcher.created_threads_info) == 2

    # Check first thread (assumed watch_commands)
    thread_info_commands = fake_watcher.created_threads_info[0]
    assert callable(
        thread_info_commands["target"]
    )  # Target should be the inner watch_commands
    assert thread_info_commands["thread"]._started is True
    assert thread_info_commands["thread"].daemon is True

    # Check second thread (assumed watch_events)
    thread_info_events = fake_watcher.created_threads_info[1]
    assert callable(
        thread_info_events["target"]
    )  # Target should be the inner watch_events
    assert thread_info_events["thread"]._started is True
    assert thread_info_events["thread"].daemon is True

    with pytest.raises(AssertionError):
        data_source.start_async(fake_runtime_instance)  # Call twice


def test_stop_async(data_source, fake_runtime_instance, fake_watcher):
    data_source.start_async(fake_runtime_instance)  # Needs to be started first
    internal_tracker = data_source._RuntimeDataSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)

    data_source.stop_async()
    assert internal_tracker.stop_call_count == 1
    # The source code for RuntimeDataSource.stop_async() does NOT call thread_pool.shutdown().
    # So, we cannot assert that fake_watcher.fake_thread_pool_executor._shutdown is True.
    # If this behavior is desired, the main code needs to change. For now, test existing behavior.
    assert (
        fake_watcher.fake_thread_pool_executor._shutdown is True
    )  # Shutdown IS called by SUT


def test_stop_async_before_start_raises_error(data_source):
    with pytest.raises(AssertionError):
        data_source.stop_async()


# Helper for testing loop methods
def setup_loop_test(
    data_source, fake_runtime_instance, fake_watcher, loop_type="commands"
):
    data_source.start_async(fake_runtime_instance)
    internal_tracker = data_source._RuntimeDataSource__is_running
    assert isinstance(internal_tracker, FakeIsRunningTracker)

    if loop_type == "commands":
        loop_target = fake_watcher.watch_commands_target
    elif loop_type == "events":
        loop_target = fake_watcher.watch_events_target
    else:  # pragma: no cover
        raise ValueError("Invalid loop_type for test setup")

    assert callable(
        loop_target
    ), f"Target for {loop_type} not captured or not callable"
    internal_tracker.set_is_running(True)  # Ensure loop can run once
    return (
        internal_tracker,
        loop_target,
        fake_watcher.fake_thread_pool_executor,
    )


def test_watch_commands_kstart(
    data_source, fake_runtime_instance, fake_watcher, fake_command_q_source
):
    internal_tracker, loop_target, executor = setup_loop_test(
        data_source, fake_runtime_instance, fake_watcher, "commands"
    )
    fake_command_q_source.set_return_values(
        [RuntimeCommand.kStart, StopIteration]
    )

    with pytest.raises(StopIteration):
        loop_target()

    assert len(executor.submitted_tasks) == 1
    assert executor.last_submitted_fn == fake_runtime_instance.start_async
    assert fake_runtime_instance.start_async_call_count == 1


def test_watch_commands_kstop(
    data_source, fake_runtime_instance, fake_watcher, fake_command_q_source
):
    internal_tracker, loop_target, executor = setup_loop_test(
        data_source, fake_runtime_instance, fake_watcher, "commands"
    )
    fake_command_q_source.set_return_values(
        [RuntimeCommand.kStop, StopIteration]
    )

    with pytest.raises(StopIteration):
        loop_target()

    assert len(executor.submitted_tasks) == 1
    assert executor.last_submitted_fn == fake_runtime_instance.stop
    assert fake_runtime_instance.stop_call_count == 1


def test_watch_commands_unknown_raises_valueerror(
    data_source, fake_runtime_instance, fake_watcher, fake_command_q_source
):
    internal_tracker, loop_target, executor = setup_loop_test(
        data_source, fake_runtime_instance, fake_watcher, "commands"
    )
    fake_command_q_source.set_return_values(
        [-1, StopIteration]
    )  # Invalid command

    with pytest.raises(ValueError):  # Loop should raise this
        loop_target()
    assert len(executor.submitted_tasks) == 0


def test_watch_commands_timeout(
    data_source, fake_runtime_instance, fake_watcher, fake_command_q_source
):
    internal_tracker, loop_target, executor = setup_loop_test(
        data_source, fake_runtime_instance, fake_watcher, "commands"
    )
    fake_command_q_source.set_return_values(
        [None, StopIteration]
    )  # Queue timeout

    with pytest.raises(StopIteration):
        loop_target()
    assert len(executor.submitted_tasks) == 0


def test_watch_events_event_received(
    data_source, fake_runtime_instance, fake_watcher, fake_event_q_source
):
    internal_tracker, loop_target, executor = setup_loop_test(
        data_source, fake_runtime_instance, fake_watcher, "events"
    )
    test_event = "my_sample_event_data"
    fake_event_q_source.set_return_values([test_event, StopIteration])

    with pytest.raises(StopIteration):
        loop_target()

    assert len(executor.submitted_tasks) == 1
    assert executor.last_submitted_fn == fake_runtime_instance.on_event
    assert executor.last_submitted_args == (test_event,)
    assert fake_runtime_instance.on_event_call_count == 1
    assert fake_runtime_instance.on_event_called_with == test_event


def test_watch_events_timeout(
    data_source, fake_runtime_instance, fake_watcher, fake_event_q_source
):
    internal_tracker, loop_target, executor = setup_loop_test(
        data_source, fake_runtime_instance, fake_watcher, "events"
    )
    fake_event_q_source.set_return_values(
        [None, StopIteration]
    )  # Queue timeout

    with pytest.raises(StopIteration):
        loop_target()
    assert len(executor.submitted_tasks) == 0
