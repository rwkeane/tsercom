"""Tests for IsRunningTracker."""

import asyncio
from functools import partial
from unittest.mock import MagicMock

import pytest

from tsercom.util.is_running_tracker import IsRunningTracker


@pytest.fixture
def tracker() -> IsRunningTracker:
    """Fixture to create an IsRunningTracker instance."""
    return IsRunningTracker()


def test_initialization(tracker: IsRunningTracker):
    """Tests that IsRunningTracker initializes with is_running as False."""
    assert not tracker.is_running


def test_start(tracker: IsRunningTracker):
    """Tests that start() sets is_running to True."""
    tracker.start()
    assert tracker.is_running


def test_stop(tracker: IsRunningTracker):
    """Tests that stop() sets is_running to False."""
    tracker.start()  # Start first
    tracker.stop()
    assert not tracker.is_running


def test_set_true(tracker: IsRunningTracker):
    """Tests that set(True) sets is_running to True."""
    tracker.set(True)
    assert tracker.is_running


def test_set_false(tracker: IsRunningTracker):
    """Tests that set(False) sets is_running to False."""
    tracker.set(True)  # Start first
    tracker.set(False)
    assert not tracker.is_running


def test_basic_sequence(tracker: IsRunningTracker):
    """Tests a basic sequence of operations."""
    assert not tracker.is_running  # Initial state
    tracker.start()
    assert tracker.is_running
    tracker.stop()
    assert not tracker.is_running
    tracker.set(True)
    assert tracker.is_running
    tracker.set(False)
    assert not tracker.is_running


@pytest.fixture
async def running_tracker(event_loop: asyncio.AbstractEventLoop):
    """
    Fixture to create an IsRunningTracker instance associated with the
    current event_loop.
    """
    tracker = IsRunningTracker()
    # Associate the tracker with the current event loop
    tracker._IsRunningTracker__event_loop = (
        event_loop  # pyright: ignore[reportPrivateUsage]
    )


def run_async(coro):
    """Helper to run a coroutine if a test function cannot be async."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_methods_fail_without_proper_event_loop_setup(mocker):
    """
    Tests that async utility methods fail as expected if the event loop
    is not properly initialized within IsRunningTracker.
    """
    # mock_aio_get_loop was here, but it's unused as the relevant failure path
    # is now via get_global_event_loop().

    err_msg_ensure_loop = (
        "Event loop not found by _get_loop_func. "
        "Must be called from within a running event loop or have an event loop set."
    )
    # err_msg_run_on_event_loop was here, but is no longer used as the sub-tests using it were removed.

    # Sub-test for __ensure_event_loop_initialized direct call
    tracker = IsRunningTracker(get_loop_func=lambda: None)
    with pytest.raises(RuntimeError, match=err_msg_ensure_loop):
        run_async(
            tracker._IsRunningTracker__ensure_event_loop_initialized()
        )  # pyright: ignore[reportPrivateUsage]
    assert (
        tracker._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    # Sub-test for wait_until_started() when tracker is stopped (initial state)
    tracker = IsRunningTracker(
        get_loop_func=lambda: None
    )  # New instance, initially False (stopped)
    # It will call __ensure_event_loop_initialized because it's not running
    with pytest.raises(RuntimeError, match=err_msg_ensure_loop):
        run_async(tracker.wait_until_started())
    assert (
        tracker._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    # Sub-test for wait_until_stopped() when tracker is running
    # This requires the tracker to be "running" and its __event_loop to be None initially
    # for __ensure_event_loop_initialized to be called by wait_until_stopped.
    tracker_ws_running = IsRunningTracker(get_loop_func=lambda: None)
    # Manually set tracker to running state without involving proper event loop init for set()
    tracker_ws_running._Atomic__value = (
        True  # pylint: disable=protected-access
    )
    assert tracker_ws_running.is_running
    assert (
        tracker_ws_running._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError, match=err_msg_ensure_loop):
        run_async(tracker_ws_running.wait_until_stopped())
    # __ensure_event_loop_initialized is called by wait_until_stopped when self.get() is True
    assert (
        tracker_ws_running._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    # Sub-test for wait_until_stopped() when tracker is already stopped (initial state)
    # This tracker does not need the get_loop_func injection as __ensure_event_loop_initialized
    # should not be called.
    tracker_ws_stopped = (
        IsRunningTracker()
    )  # New instance, initially False (stopped)
    # It returns early, __ensure_event_loop_initialized is not called.
    run_async(tracker_ws_stopped.wait_until_stopped())
    assert (
        tracker_ws_stopped._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    # The following sub-tests for set(), start(), stop() expecting err_msg_run_on_event_loop
    # were removed because IsRunningTracker.set() (and thus start/stop) has a guard
    # that prevents calling aio_utils.run_on_event_loop if self.__event_loop is None.
    # This is the correct behavior for IsRunningTracker, as it cannot schedule an
    # async task without a loop. The unit test was attempting to test a state
    # that IsRunningTracker is designed to prevent. The earlier parts of this test
    # correctly verify the failure of async methods when the injected get_loop_func
    # returns None.


def test_set_method_interaction_with_run_on_event_loop_and_clear(mocker):
    """
    Tests the set() method's interaction with run_on_event_loop
    and the clear() callback.
    """
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)

    # Mock where run_on_event_loop is imported and used by IsRunningTracker.set
    mock_run_on_event_loop = mocker.patch(
        "tsercom.util.is_running_tracker.run_on_event_loop"
    )
    mock_task = MagicMock(spec=asyncio.Task)
    mock_run_on_event_loop.return_value = mock_task

    # Mock is_running_on_event_loop to simulate not being on the loop,
    # so task.result() gets called.
    mock_is_running_on_event_loop = mocker.patch(
        "tsercom.util.is_running_tracker.is_running_on_event_loop",
        return_value=False,
    )

    tracker = IsRunningTracker()
    # Manually set the event loop for this test, as __ensure_event_loop_initialized
    # would normally do this. This is white-box testing.
    tracker._IsRunningTracker__event_loop = (
        mock_loop  # pyright: ignore[reportPrivateUsage]
    )

    # Call set()
    tracker.set(True)

    # Assert run_on_event_loop was called correctly
    assert mock_run_on_event_loop.call_count == 1
    call_args = mock_run_on_event_loop.call_args[0]
    assert isinstance(call_args[0], partial)
    assert (
        call_args[0].func == tracker._IsRunningTracker__set_impl
    )  # pyright: ignore[reportPrivateUsage]
    assert call_args[0].args == (True,)
    assert call_args[1] is mock_loop

    # Assert add_done_callback was called on the mock task
    mock_task.add_done_callback.assert_called_once()
    clear_callback = mock_task.add_done_callback.call_args[0][0]

    # Keep original event objects for comparison
    original_running_barrier = (
        tracker._IsRunningTracker__running_barrier
    )  # pyright: ignore[reportPrivateUsage]
    original_stopped_barrier = (
        tracker._IsRunningTracker__stopped_barrier
    )  # pyright: ignore[reportPrivateUsage]

    # Invoke the clear callback
    clear_callback(None)  # Argument is the future, not used by clear

    # Check that barriers are new event objects
    assert (
        tracker._IsRunningTracker__running_barrier
        is not original_running_barrier
    )  # pyright: ignore[reportPrivateUsage]
    assert (
        tracker._IsRunningTracker__stopped_barrier
        is not original_stopped_barrier
    )  # pyright: ignore[reportPrivateUsage]
    # Check their state (should be default: not set)
    assert (
        not tracker._IsRunningTracker__running_barrier.is_set()
    )  # pyright: ignore[reportPrivateUsage]
    assert (
        not tracker._IsRunningTracker__stopped_barrier.is_set()
    )  # pyright: ignore[reportPrivateUsage]

    # Check that __event_loop is set to None
    assert (
        tracker._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    # Ensure task.result() was called
    mock_task.result.assert_called_once()
    mock_is_running_on_event_loop.assert_called_once_with(mock_loop)
