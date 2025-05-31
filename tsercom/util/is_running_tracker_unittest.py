"""Tests for IsRunningTracker."""

import asyncio
import threading
import time

import pytest

from tsercom.util.is_running_tracker import IsRunningTracker
from tsercom.threading.aio import aio_utils


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


from functools import partial
from unittest.mock import MagicMock


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
    # Mock get_running_loop_or_none where it's imported by IsRunningTracker
    mock_get_loop = mocker.patch(
        "tsercom.util.is_running_tracker.get_running_loop_or_none",
        return_value=None,
    )
    # This mock is for aio_utils.run_on_event_loop's internal call
    mock_aio_get_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.get_running_loop_or_none",
        return_value=None,
    )

    err_msg_ensure_loop = "Must be called from within a running event loop or have an event loop set."
    err_msg_run_on_event_loop = (
        "No event loop is running, and no event loop was previously set."
    )

    # Sub-test for __ensure_event_loop_initialized direct call
    tracker = IsRunningTracker()
    with pytest.raises(AssertionError, match=err_msg_ensure_loop):
        run_async(
            tracker._IsRunningTracker__ensure_event_loop_initialized()
        )  # pyright: ignore[reportPrivateUsage]
    mock_get_loop.assert_called_once()
    assert (
        tracker._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]
    mock_get_loop.reset_mock()

    # Sub-test for wait_until_started() when tracker is stopped (initial state)
    tracker = IsRunningTracker()  # New instance, initially False (stopped)
    # It will call __ensure_event_loop_initialized because it's not running
    with pytest.raises(AssertionError, match=err_msg_ensure_loop):
        run_async(tracker.wait_until_started())
    if not tracker.get():  # Should be true
        mock_get_loop.assert_called_once()
    assert (
        tracker._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]
    mock_get_loop.reset_mock()

    # Sub-test for wait_until_stopped() when tracker is running
    # This requires the tracker to be "running" and its __event_loop to be None initially
    # for __ensure_event_loop_initialized to be called by wait_until_stopped.
    tracker_ws_running = IsRunningTracker()
    # Manually set tracker to running state without involving proper event loop init for set()
    tracker_ws_running._Atomic__value = (
        True  # pylint: disable=protected-access
    )
    assert tracker_ws_running.is_running
    assert (
        tracker_ws_running._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(AssertionError, match=err_msg_ensure_loop):
        run_async(tracker_ws_running.wait_until_stopped())
    # __ensure_event_loop_initialized is called by wait_until_stopped when self.get() is True
    mock_get_loop.assert_called_once()
    assert (
        tracker_ws_running._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]
    mock_get_loop.reset_mock()

    # Sub-test for wait_until_stopped() when tracker is already stopped (initial state)
    tracker_ws_stopped = (
        IsRunningTracker()
    )  # New instance, initially False (stopped)
    # It returns early, __ensure_event_loop_initialized is not called.
    run_async(tracker_ws_stopped.wait_until_stopped())
    mock_get_loop.assert_not_called()  # Not called because it returns early
    assert (
        tracker_ws_stopped._IsRunningTracker__event_loop is None
    )  # pyright: ignore[reportPrivateUsage]

    # --- Sub-tests for set(), start(), stop() that use run_on_event_loop ---
    # These expect run_on_event_loop to fail because its internal call to
    # aio_utils.get_running_loop_or_none (mocked by mock_aio_get_loop) returns None.

    tracker_set = IsRunningTracker()
    with pytest.raises(AssertionError, match=err_msg_run_on_event_loop):
        # tracker.set() is sync, but the coroutine it schedules via run_on_event_loop will fail.
        # We need to execute that coroutine.
        # The actual IsRunningTracker.set() is synchronous and won't raise here directly.
        # The error comes from task.result() if not on loop, or from the loop itself.
        # For this test, we assume we are "not on the loop" when calling set,
        # so task.result() would be called.
        # To simulate this, we'd need to mock run_on_event_loop more deeply or
        # test the behavior of the scheduled partial(self.__set_impl, value)
        # when run_on_event_loop itself can't find a loop.
        # The simplest is to check if set() fails if run_on_event_loop fails.
        mocker.patch(
            "tsercom.util.is_running_tracker.is_running_on_event_loop",
            return_value=False,
        )
        tracker_set.set(
            True
        )  # This should call task.result() internally and raise
    mock_aio_get_loop.assert_called_once()  # run_on_event_loop calls get_running_loop_or_none
    mock_aio_get_loop.reset_mock()

    tracker_start = IsRunningTracker()
    with pytest.raises(AssertionError, match=err_msg_run_on_event_loop):
        mocker.patch(
            "tsercom.util.is_running_tracker.is_running_on_event_loop",
            return_value=False,
        )
        tracker_start.start()
    mock_aio_get_loop.assert_called_once()
    mock_aio_get_loop.reset_mock()

    tracker_stop = IsRunningTracker()
    # Manually set to running so stop actually tries to do async work
    tracker_stop._Atomic__value = True  # pylint: disable=protected-access
    tracker_stop._IsRunningTracker__event_loop = MagicMock(
        spec=asyncio.AbstractEventLoop
    )  # pyright: ignore[reportPrivateUsage]
    # Temporarily allow aio_utils.get_running_loop_or_none to return this mock loop
    # so that the initial part of set() in stop() can find a loop.
    mock_aio_get_loop.return_value = (
        tracker_stop._IsRunningTracker__event_loop
    )  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(AssertionError, match=err_msg_run_on_event_loop):
        mocker.patch(
            "tsercom.util.is_running_tracker.is_running_on_event_loop",
            return_value=False,
        )
        # Now, when stop() calls set(False), make sure aio_utils.get_running_loop_or_none returns None again
        # This is tricky because the loop is captured by `run_on_event_loop` at the time of the call.
        # The current mock setup for mock_aio_get_loop will make run_on_event_loop (inside set)
        # get None when it calls get_running_loop_or_none.

        # The most direct way to test stop() failing if the loop disappears:
        # 1. Tracker is running, has an associated loop.
        # 2. Mock aio_utils.get_running_loop_or_none to NOW return None.
        # 3. Call stop(). The run_on_event_loop inside set() should pick up the None.
        mock_aio_get_loop.return_value = (
            None  # Critical change for this specific call path
        )
        tracker_stop.stop()  # which calls set(False)
    # It should be called when run_on_event_loop (called by set) tries to get a loop.
    mock_aio_get_loop.assert_called()  # Assert it was called (at least once, due to prior return_value change)


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
