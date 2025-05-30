import pytest
import functools

from tsercom.data.data_timeout_tracker import DataTimeoutTracker

# Assuming aio_utils are used as imported in data_timeout_tracker.py
# from tsercom.threading.aio.aio_utils import run_on_event_loop, is_running_on_event_loop


# --- Fixtures ---


@pytest.fixture
def mock_run_on_event_loop(mocker):
    # Patch where it's imported and used by DataTimeoutTracker
    return mocker.patch("tsercom.data.data_timeout_tracker.run_on_event_loop")


@pytest.fixture
def mock_asyncio_sleep(mocker):
    # Patch asyncio.sleep where it's used by DataTimeoutTracker
    # pytest-asyncio will make this an AsyncMock if not specified otherwise
    return mocker.patch("tsercom.data.data_timeout_tracker.asyncio.sleep")


@pytest.fixture
def mock_is_running_on_event_loop(mocker):
    # Patch where it's imported and used by DataTimeoutTracker
    mock = mocker.patch(
        "tsercom.data.data_timeout_tracker.is_running_on_event_loop"
    )
    mock.return_value = (
        True  # Default to True for most async method internal checks
    )
    return mock


@pytest.fixture
def mock_tracked_object(mocker):
    return mocker.create_autospec(DataTimeoutTracker.Tracked, instance=True)


# Custom exception to break the loop after one iteration
class BreakLoop(Exception):
    pass


# --- Test Cases ---


# 1. __init__
def test_init_default_timeout():
    tracker = DataTimeoutTracker()
    assert tracker._DataTimeoutTracker__timeout_seconds == 60
    assert tracker._DataTimeoutTracker__tracked_list == []


def test_init_custom_timeout():
    custom_timeout = 30
    tracker = DataTimeoutTracker(timeout_seconds=custom_timeout)
    assert tracker._DataTimeoutTracker__timeout_seconds == custom_timeout
    assert tracker._DataTimeoutTracker__tracked_list == []


# 2. register() and __register_impl()
@pytest.mark.asyncio
async def test_register_calls_run_on_event_loop(
    mock_run_on_event_loop, mock_tracked_object
):
    tracker = DataTimeoutTracker()
    tracker.register(mock_tracked_object)

    mock_run_on_event_loop.assert_called_once()
    call_args = mock_run_on_event_loop.call_args[0]
    assert isinstance(call_args[0], functools.partial)

    partial_func = call_args[0]
    assert partial_func.func == tracker._DataTimeoutTracker__register_impl
    assert partial_func.args == (mock_tracked_object,)


@pytest.mark.asyncio
async def test_register_impl_adds_to_list(
    mock_is_running_on_event_loop, mock_tracked_object
):
    tracker = DataTimeoutTracker()
    assert len(tracker._DataTimeoutTracker__tracked_list) == 0

    await tracker._DataTimeoutTracker__register_impl(mock_tracked_object)

    mock_is_running_on_event_loop.assert_called_once()
    assert len(tracker._DataTimeoutTracker__tracked_list) == 1
    assert mock_tracked_object in tracker._DataTimeoutTracker__tracked_list


@pytest.mark.asyncio
async def test_register_impl_asserts_if_not_on_event_loop(
    mock_is_running_on_event_loop, mock_tracked_object
):
    mock_is_running_on_event_loop.return_value = False
    tracker = DataTimeoutTracker()
    with pytest.raises(AssertionError):
        await tracker._DataTimeoutTracker__register_impl(mock_tracked_object)


# 3. start()
def test_start_calls_run_on_event_loop(mock_run_on_event_loop):
    tracker = DataTimeoutTracker()
    tracker.start()
    mock_run_on_event_loop.assert_called_once_with(
        tracker._DataTimeoutTracker__execute_periodically
    )


# 4. Logic of __execute_periodically() (Controlled Simulation)


@pytest.mark.asyncio
async def test_execute_periodically_calls_sleep_and_on_triggered(
    mock_asyncio_sleep, mock_tracked_object, mock_is_running_on_event_loop
):
    test_timeout = 42
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)
    await tracker._DataTimeoutTracker__register_impl(mock_tracked_object)

    # Manually start the tracker's running state for direct test of __execute_periodically
    tracker._DataTimeoutTracker__is_running.start()

    def stop_and_raise_breakloop_on_triggered(*args, **kwargs):
        tracker._DataTimeoutTracker__is_running.stop()
        raise BreakLoop("Break from _on_triggered")

    mock_tracked_object._on_triggered.side_effect = (
        stop_and_raise_breakloop_on_triggered
    )

    # __execute_periodically will catch BreakLoop from _on_triggered and log it.
    # The loop terminates because __is_running is set to False by the side effect.
    await tracker._DataTimeoutTracker__execute_periodically()
    # No explicit stop needed here as the side effect handles it.

    mock_asyncio_sleep.assert_called_once_with(test_timeout)
    mock_tracked_object._on_triggered.assert_called_once_with(test_timeout)


@pytest.mark.asyncio
async def test_execute_periodically_multiple_objects(
    mock_asyncio_sleep, mocker, mock_is_running_on_event_loop
):
    test_timeout = 35
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)

    tracked_obj1 = mocker.create_autospec(
        DataTimeoutTracker.Tracked, instance=True
    )
    tracked_obj2 = mocker.create_autospec(
        DataTimeoutTracker.Tracked, instance=True
    )

    await tracker._DataTimeoutTracker__register_impl(tracked_obj1)
    await tracker._DataTimeoutTracker__register_impl(tracked_obj2)

    # Manually start the tracker's running state
    tracker._DataTimeoutTracker__is_running.start()

    def stop_and_raise_breakloop_on_obj2_triggered(*args, **kwargs):
        tracker._DataTimeoutTracker__is_running.stop()
        raise BreakLoop("Break from obj2._on_triggered")

    tracked_obj2._on_triggered.side_effect = (
        stop_and_raise_breakloop_on_obj2_triggered
    )

    # __execute_periodically will catch BreakLoop from _on_triggered and log it.
    await tracker._DataTimeoutTracker__execute_periodically()
    # No explicit stop needed here.

    mock_asyncio_sleep.assert_called_once_with(test_timeout)
    tracked_obj1._on_triggered.assert_called_once_with(test_timeout)
    tracked_obj2._on_triggered.assert_called_once_with(test_timeout)


@pytest.mark.asyncio
async def test_execute_periodically_no_tracked_objects(mock_asyncio_sleep):
    test_timeout = 15
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)

    # Manually start the tracker's running state
    tracker._DataTimeoutTracker__is_running.start()

    def stop_and_raise_breakloop_on_sleep(*args, **kwargs):
        # This exception should propagate out of __execute_periodically
        # because the try-except in SUT only covers _on_triggered.
        # tracker._DataTimeoutTracker__is_running.stop() # Stop to ensure loop termination if BreakLoop was caught
        raise BreakLoop("Break from sleep")

    mock_asyncio_sleep.side_effect = stop_and_raise_breakloop_on_sleep

    with pytest.raises(BreakLoop, match="Break from sleep"):
        await tracker._DataTimeoutTracker__execute_periodically()

    # Explicitly stop if BreakLoop didn't also stop it via side effect,
    # or if the test needs to ensure it's stopped regardless.
    # Given stop_and_raise_breakloop_on_sleep doesn't stop it, we might need it here if loop could continue.
    # However, if BreakLoop propagates, __is_running.stop() might not be strictly necessary for this test's assertions.
    # For safety and consistency, ensuring it's stopped if it was started:
    if tracker._DataTimeoutTracker__is_running.get():
        tracker._DataTimeoutTracker__is_running.stop()

    mock_asyncio_sleep.assert_called_once_with(test_timeout)


@pytest.mark.asyncio
async def test_execute_periodically_handles_exception_in_on_triggered_and_continues(
    mock_asyncio_sleep, mocker, mock_is_running_on_event_loop
):
    test_timeout = 25
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)

    tracked_obj1 = mocker.create_autospec(
        DataTimeoutTracker.Tracked, instance=True
    )
    tracked_obj2 = mocker.create_autospec(
        DataTimeoutTracker.Tracked, instance=True
    )  # Will raise error
    tracked_obj3 = mocker.create_autospec(
        DataTimeoutTracker.Tracked, instance=True
    )  # Should still be called

    tracked_obj2._on_triggered.side_effect = RuntimeError(
        "Test error in _on_triggered"
    )
    # Make the last object's _on_triggered raise BreakLoop to exit the test loop
    tracked_obj3._on_triggered.side_effect = BreakLoop

    await tracker._DataTimeoutTracker__register_impl(tracked_obj1)
    await tracker._DataTimeoutTracker__register_impl(tracked_obj2)
    await tracker._DataTimeoutTracker__register_impl(tracked_obj3)

    # Manually start the tracker's running state
    tracker._DataTimeoutTracker__is_running.start()

    def stop_and_raise_breakloop_on_obj3_triggered(*args, **kwargs):
        tracker._DataTimeoutTracker__is_running.stop()
        raise BreakLoop("Break from obj3._on_triggered")

    tracked_obj2._on_triggered.side_effect = (
        RuntimeError(  # This will be caught and logged by SUT
            "Test error in _on_triggered"
        )
    )
    tracked_obj3._on_triggered.side_effect = (
        stop_and_raise_breakloop_on_obj3_triggered
    )

    # __execute_periodically will catch RuntimeError and BreakLoop. Loop terminates via side effect.
    await tracker._DataTimeoutTracker__execute_periodically()
    # No explicit stop needed here.

    mock_asyncio_sleep.assert_called_once_with(test_timeout)
    tracked_obj1._on_triggered.assert_called_once_with(test_timeout)
    tracked_obj2._on_triggered.assert_called_once_with(
        test_timeout
    )  # Error is caught, loop continues
    tracked_obj3._on_triggered.assert_called_once_with(
        test_timeout
    )  # Should be called now

    # The SUT now catches exceptions from _on_triggered and continues,
    # so all assertions about calls should hold.
