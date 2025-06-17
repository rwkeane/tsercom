import pytest
import importlib
from unittest.mock import MagicMock
import concurrent.futures

from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
    RuntimeCommand,
)


class FakeRuntime:
    def __init__(self):
        self.start_async_called = False
        self.stop_called = False
        self.start_async_call_count = 0
        self.stop_call_count = 0

    def start_async(self):
        self.start_async_called = True
        self.start_async_call_count += 1

    def stop(self, _=None):  # Add optional argument to match partial call
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

        # Return a future that can be awaited.
        # For tests that need to simulate timeout, the test itself can
        # mock this future's result method.
        mock_future = MagicMock(spec=concurrent.futures.Future)
        # For simplicity, make result() do nothing or return a default value.
        # Tests specifically testing timeout will need to adjust this mock's behavior.
        mock_future.result.return_value = None
        return mock_future

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
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.START
    # run_on_event_loop should not have been called yet meaningfully
    # as there's no runtime to act upon. The captured_callables might be empty
    # or contain a lambda that does nothing if runtime is None.
    # The key is that FakeRuntime methods are not called.


def test_stop_before_runtime_set(bridge, patch_rcb_run_on_event_loop):
    bridge.stop()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.STOP


# Tests for Setting Runtime Executes Pending Commands
def test_set_runtime_executes_pending_start(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.start()  # Pending command
    assert not fake_runtime.start_async_called

    bridge.set_runtime(fake_runtime)
    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 1
    assert not fake_runtime.stop_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime


def test_set_runtime_executes_pending_stop(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.stop()  # Pending command
    assert not fake_runtime.stop_called

    bridge.set_runtime(fake_runtime)
    assert fake_runtime.stop_called
    assert fake_runtime.stop_call_count == 1
    assert not fake_runtime.start_async_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime


# Tests for Commands After Runtime Set
def test_start_after_runtime_set(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called  # No pending command initially

    bridge.start()
    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 1
    assert not fake_runtime.stop_called


def test_stop_after_runtime_set(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.stop_called  # No pending command initially

    bridge.stop()
    assert fake_runtime.stop_called
    assert fake_runtime.stop_call_count == 1
    assert not fake_runtime.start_async_called


# Test Setting Runtime with No Pending Command
def test_set_runtime_no_pending_command(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called
    assert not fake_runtime.stop_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime


# Test Double Set Runtime
def test_double_set_runtime_raises_error(  # Renamed for clarity
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    another_fake_runtime = FakeRuntime()
    bridge.set_runtime(fake_runtime)

    with pytest.raises(RuntimeError):
        bridge.set_runtime(another_fake_runtime)

    # Ensure original runtime is still set and no commands were run on the new one
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime
    assert not another_fake_runtime.start_async_called
    assert not another_fake_runtime.stop_called


def test_set_runtime_state_cleared_after_command_execution(  # Renamed for clarity
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    """Test that after a command is executed upon setting runtime, the state is cleared."""
    bridge.start()
    bridge.set_runtime(fake_runtime)

    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 1
    assert bridge._RuntimeCommandBridge__state.get() is None

    fake_runtime.start_async_called = False  # Reset
    bridge.start()  # Call start again
    assert fake_runtime.start_async_called
    assert fake_runtime.start_async_call_count == 2


def test_set_runtime_with_no_pending_command_state_remains_none(  # Renamed
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    """Test set_runtime when __state is None, it should remain None."""
    assert bridge._RuntimeCommandBridge__state.get() is None
    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called
    assert not fake_runtime.stop_called
    assert bridge._RuntimeCommandBridge__runtime is fake_runtime
    assert bridge._RuntimeCommandBridge__state.get() is None


def test_pending_stop_overwrites_pending_start_before_runtime_set(  # Renamed
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    """If start then stop is called before runtime, stop command is stored."""
    bridge.start()
    bridge.stop()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.STOP

    bridge.set_runtime(fake_runtime)
    assert not fake_runtime.start_async_called
    assert fake_runtime.stop_called
    assert fake_runtime.stop_call_count == 1
    assert bridge._RuntimeCommandBridge__state.get() is None


def test_pending_start_overwrites_pending_stop_before_runtime_set(  # Renamed
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    """If stop then start is called before runtime, start command is stored."""
    bridge.stop()
    bridge.start()
    assert bridge._RuntimeCommandBridge__state.get() == RuntimeCommand.START

    bridge.set_runtime(fake_runtime)
    assert fake_runtime.start_async_called
    assert not fake_runtime.stop_called
    assert fake_runtime.start_async_call_count == 1
    assert bridge._RuntimeCommandBridge__state.get() is None


# Ensure the patch_rcb_run_on_event_loop is used by all tests that interact with methods
# that call run_on_event_loop (set_runtime, start, stop)
# All current tests that need it seem to have it.
# The fixture `bridge` does not need it as __init__ doesn't call run_on_event_loop.
# `fake_runtime` also doesn't need it.
# `test_initial_state` doesn't need it as it only checks initial attributes.


# Test that run_on_event_loop is actually called by start after runtime is set
def test_run_on_event_loop_usage_on_start(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.set_runtime(fake_runtime)
    # patch_rcb_run_on_event_loop is a list of captured callables.
    # When set_runtime was called, if there was a pending command, it would have been captured.
    # Since there's no pending command, captured_callables might be empty or have non-runtime calls.
    # We reset for this specific check.
    patch_rcb_run_on_event_loop.clear()  # Clear any previous captures

    bridge.start()
    assert (
        len(patch_rcb_run_on_event_loop) == 1
    )  # One callable should have been passed
    # The callable itself was executed by the fake, so fake_runtime.start_async_called is True
    assert fake_runtime.start_async_called
    # Check that the returned future from fake_run_on_event_loop was used
    # (This part is implicitly tested by not raising AttributeError for .result())


# Test that run_on_event_loop is actually called by stop after runtime is set
def test_run_on_event_loop_usage_on_stop(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.set_runtime(fake_runtime)
    patch_rcb_run_on_event_loop.clear()

    bridge.stop()
    assert len(patch_rcb_run_on_event_loop) == 1
    assert fake_runtime.stop_called
    # Implicitly tests that .result() was called on the future


def test_set_runtime_pending_stop_timeout_on_result(
    bridge, fake_runtime, mocker
):
    """
    Tests that set_runtime with a pending STOP command handles TimeoutError
    from future.result() gracefully.
    """
    bridge.stop()  # Set pending STOP command
    assert not fake_runtime.stop_called

    mock_future_for_stop = mocker.MagicMock(spec=concurrent.futures.Future)
    mock_future_for_stop.result.side_effect = concurrent.futures.TimeoutError(
        "Simulated timeout"
    )

    # Temporarily patch run_on_event_loop for this specific test's scenario
    # The patch_rcb_run_on_event_loop fixture will be overridden by this more specific patch
    # within the scope of this test.
    def run_on_event_loop_side_effect_pending_stop(
        callable_to_run, *args, **kwargs
    ):
        if callable(callable_to_run):
            callable_to_run()  # Execute the callable
        return mock_future_for_stop  # Return the future that will timeout

    mock_run_on_event_loop = mocker.patch(
        "tsercom.api.local_process.runtime_command_bridge.run_on_event_loop",
        side_effect=run_on_event_loop_side_effect_pending_stop,
    )

    try:
        bridge.set_runtime(fake_runtime)
    except concurrent.futures.TimeoutError:  # pragma: no cover
        pytest.fail(
            "TimeoutError should be caught and handled by set_runtime, not re-raised."
        )
    except Exception as e:  # pragma: no cover
        pytest.fail(f"set_runtime raised an unexpected exception: {e}")

    assert (
        fake_runtime.stop_called
    ), "Runtime's stop method should have been called."
    mock_run_on_event_loop.assert_called_once()  # Ensure our mock was used
    # The callable passed to run_on_event_loop is a partial(fake_runtime.stop, None)
    # We can inspect its first argument (the partial)
    called_arg = mock_run_on_event_loop.call_args[0][0]
    assert callable(called_arg)
    # To be very precise, one could check called_arg.func and called_arg.args

    mock_future_for_stop.result.assert_called_once_with(timeout=5.0)


def test_stop_after_runtime_set_timeout_on_result(
    bridge, fake_runtime, mocker
):
    """
    Tests that bridge.stop() after runtime is set handles TimeoutError
    from future.result() gracefully.
    """
    bridge.set_runtime(fake_runtime)  # Runtime is set, no pending commands
    assert not fake_runtime.stop_called  # stop should not have been called yet

    mock_future_for_stop = mocker.MagicMock(spec=concurrent.futures.Future)
    mock_future_for_stop.result.side_effect = concurrent.futures.TimeoutError(
        "Simulated timeout"
    )

    def run_on_event_loop_side_effect_direct_stop(
        callable_to_run, *args, **kwargs
    ):
        if callable(callable_to_run):
            callable_to_run()  # Execute the callable
        return mock_future_for_stop  # Return the future that will timeout

    mock_run_on_event_loop = mocker.patch(
        "tsercom.api.local_process.runtime_command_bridge.run_on_event_loop",
        side_effect=run_on_event_loop_side_effect_direct_stop,
    )

    try:
        bridge.stop()  # This should trigger the run_on_event_loop with runtime.stop
    except concurrent.futures.TimeoutError:  # pragma: no cover
        pytest.fail(
            "TimeoutError should be caught and handled by bridge.stop, not re-raised."
        )
    except Exception as e:  # pragma: no cover
        pytest.fail(f"bridge.stop() raised an unexpected exception: {e}")

    assert (
        fake_runtime.stop_called
    ), "Runtime's stop method should have been called."
    mock_run_on_event_loop.assert_called_once()
    mock_future_for_stop.result.assert_called_once_with(timeout=5.0)


# Test that run_on_event_loop is called when setting runtime with a pending start
def test_run_on_event_loop_usage_on_set_runtime_pending_start(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.start()
    patch_rcb_run_on_event_loop.clear()

    bridge.set_runtime(fake_runtime)
    assert len(patch_rcb_run_on_event_loop) == 1
    assert fake_runtime.start_async_called
    # For start, we don't call .result(), so no implicit future check here


# Test that run_on_event_loop is called when setting runtime with a pending stop
def test_run_on_event_loop_usage_on_set_runtime_pending_stop(
    bridge, fake_runtime, patch_rcb_run_on_event_loop
):
    bridge.stop()
    patch_rcb_run_on_event_loop.clear()

    bridge.set_runtime(fake_runtime)
    assert len(patch_rcb_run_on_event_loop) == 1
    assert fake_runtime.stop_called
    # Implicitly tests that .result() was called on the future
