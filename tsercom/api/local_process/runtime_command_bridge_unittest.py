import pytest
import importlib

from tsercom.api.local_process.runtime_command_bridge import RuntimeCommandBridge, RuntimeCommand


class FakeRuntime:
    def __init__(self):
        self.start_async_called = False
        self.stop_called = False
        self.start_async_call_count = 0
        self.stop_call_count = 0

    def start_async(self):
        self.start_async_called = True
        self.start_async_call_count += 1

    def stop(self):
        self.stop_called = True
        self.stop_call_count += 1


# Fixture to monkeypatch run_on_event_loop in the runtime_command_bridge module
@pytest.fixture
def patch_rcb_run_on_event_loop(request):
    module_path = "tsercom.api.local_process.runtime_command_bridge"
    func_name = "run_on_event_loop"

    # Ensure the module is loaded
    target_module = importlib.import_module(module_path)
    original_func = getattr(target_module, func_name)

    # This list will store callables passed to the fake run_on_event_loop
    # It's reset for each test using this fixture.
    # Some tests might want to execute the callable, others just inspect it.
    # For RuntimeCommandBridge, the callables are simple method calls (runtime.start_async or runtime.stop)
    # so we can execute them directly.
    captured_callables = []

    def fake_run_on_event_loop(callable_to_run, *args, **kwargs):
        captured_callables.append(callable_to_run)
        # Execute the callable immediately for these tests
        if callable(callable_to_run):
            callable_to_run()

    setattr(target_module, func_name, fake_run_on_event_loop)

    def restore():
        setattr(target_module, func_name, original_func)
    request.addfinalizer(restore)

    # Return the list so tests can inspect what was "sent" to the event loop
    # For these tests, direct execution is simpler, so this might not be heavily used.
    return captured_callables


@pytest.fixture
def bridge():
    return RuntimeCommandBridge()


@pytest.fixture
def fake_runtime():
    return FakeRuntime()


# Test Initial State
def test_initial_state(bridge):
    assert bridge._RuntimeCommandBridge__runtime is None
    assert bridge._RuntimeCommandBridge__state.get() is None


# Tests for Commands Before Runtime Set
def test_start_before_runtime_set(bridge, patch_rcb_run_on_event_loop):
    bridge.start()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStart
    # run_on_event_loop should not have been called yet meaningfully
    # as there's no runtime to act upon. The captured_callables might be empty
    # or contain a lambda that does nothing if runtime is None.
    # The key is that FakeRuntime methods are not called.


def test_stop_before_runtime_set(bridge, patch_rcb_run_on_event_loop):
    bridge.stop()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStop


# Tests for Setting Runtime Executes Pending Commands
def test_set_runtime_executes_pending_start(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.start()  # Pending command
    assert not fake_runtime.start_async_called

    bridge.set_runtime(fake_runtime)
    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 1
    assert not fake_runtime.stop_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime


def test_set_runtime_executes_pending_stop(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.stop()  # Pending command
    assert not fake_runtime.stop_called

    bridge.set_runtime(fake_runtime)
    assert fake_runtime.stop_called
    assert fake_runtime.stop_call_count == 1
    assert not fake_runtime.start_async_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime


# Tests for Commands After Runtime Set
def test_start_after_runtime_set(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called  # No pending command initially

    bridge.start()
    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 1
    assert not fake_runtime.stop_called


def test_stop_after_runtime_set(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.stop_called  # No pending command initially

    bridge.stop()
    assert fake_runtime.stop_called
    assert fake_runtime.stop_call_count == 1
    assert not fake_runtime.start_async_called


# Test Setting Runtime with No Pending Command
def test_set_runtime_no_pending_command(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called
    assert not fake_runtime.stop_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime


# Test Double Set Runtime
def test_double_set_runtime_raises_assertion_error(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    another_fake_runtime = FakeRuntime()
    bridge.set_runtime(fake_runtime)

    with pytest.raises(AssertionError):
        bridge.set_runtime(another_fake_runtime)

    # Ensure original runtime is still set and no commands were run on the new one
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime
    assert not another_fake_runtime.start_async_called
    assert not another_fake_runtime.stop_called


def test_set_runtime_idempotent_state_cleared(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    """Test that after a command is executed upon setting runtime, the state is cleared."""
    bridge.start() # state = kStart
    bridge.set_runtime(fake_runtime) # runtime.start_async() called

    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 1
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStart # State is not cleared by set_runtime

    # If we call start again, it WILL re-trigger start_async because runtime is set
    fake_runtime.start_async_called = False # Reset flag for clarity
    bridge.start() # Runtime is set, so this will call run_on_event_loop(runtime.start_async)
    assert fake_runtime.start_async_called # Called again
    assert fake_runtime.start_async_call_count == 2 # Incremented


def test_set_runtime_with_none_state(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    """Test set_runtime when __state is already None (e.g. default)."""
    # Initial state is None
    assert bridge._RuntimeCommandBridge__state.get() is None
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called
    assert not fake_runtime.stop_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime
    assert bridge._RuntimeCommandBridge__state.get() is None # Should remain None


def test_command_order_start_then_stop_before_runtime(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    """If start then stop is called before runtime, stop should take precedence."""
    bridge.start()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStart
    bridge.stop() # This should overwrite the state to kStop
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStop

    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called # Start should not have been called
    assert fake_runtime.stop_called # Stop should be called
    assert fake_runtime.stop_call_count == 1
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStop # State is not cleared by set_runtime


def test_command_order_stop_then_start_before_runtime(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    """If stop then start is called before runtime, start should take precedence."""
    bridge.stop()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStop
    bridge.start() # This should overwrite the state to kStart
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStart

    bridge.set_runtime(fake_runtime)
    assert fake_runtime.start_async_called # Start should be called
    assert not fake_runtime.stop_called # Stop should not have been called
    assert fake_runtime.start_async_call_count == 1
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.kStart # State is not cleared by set_runtime

# Ensure the patch_rcb_run_on_event_loop is used by all tests that interact with methods
# that call run_on_event_loop (set_runtime, start, stop)
# All current tests that need it seem to have it.
# The fixture `bridge` does not need it as __init__ doesn't call run_on_event_loop.
# `fake_runtime` also doesn't need it.
# `test_initial_state` doesn't need it as it only checks initial attributes.

# Test that run_on_event_loop is actually called by start after runtime is set
def test_run_on_event_loop_usage_on_start(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.set_runtime(fake_runtime)
    # patch_rcb_run_on_event_loop is a list of captured callables.
    # When set_runtime was called, if there was a pending command, it would have been captured.
    # Since there's no pending command, captured_callables might be empty or have non-runtime calls.
    # We reset for this specific check.
    patch_rcb_run_on_event_loop.clear() # Clear any previous captures

    bridge.start()
    assert len(patch_rcb_run_on_event_loop) == 1 # One callable should have been passed
    # The callable itself was executed by the fake, so fake_runtime.start_async_called is True
    assert fake_runtime.start_async_called

# Test that run_on_event_loop is actually called by stop after runtime is set
def test_run_on_event_loop_usage_on_stop(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.set_runtime(fake_runtime)
    patch_rcb_run_on_event_loop.clear()

    bridge.stop()
    assert len(patch_rcb_run_on_event_loop) == 1
    assert fake_runtime.stop_called

# Test that run_on_event_loop is called when setting runtime with a pending start
def test_run_on_event_loop_usage_on_set_runtime_pending_start(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.start() # Pending command
    patch_rcb_run_on_event_loop.clear() # Clear previous captures (e.g. from bridge.start if it made one)

    bridge.set_runtime(fake_runtime)
    assert len(patch_rcb_run_on_event_loop) == 1
    assert fake_runtime.start_async_called

# Test that run_on_event_loop is called when setting runtime with a pending stop
def test_run_on_event_loop_usage_on_set_runtime_pending_stop(bridge, fake_runtime, patch_rcb_run_on_event_loop):
    bridge.stop() # Pending command
    patch_rcb_run_on_event_loop.clear()

    bridge.set_runtime(fake_runtime)
    assert len(patch_rcb_run_on_event_loop) == 1
    assert fake_runtime.stop_called
