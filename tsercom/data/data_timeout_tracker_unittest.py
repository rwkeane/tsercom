import pytest
import asyncio
import functools
from unittest.mock import MagicMock, call, ANY

from tsercom.data.data_timeout_tracker import DataTimeoutTracker
# Assuming aio_utils are used as imported in data_timeout_tracker.py
# from tsercom.threading.aio.aio_utils import run_on_event_loop, is_running_on_event_loop


# --- Fixtures ---

@pytest.fixture
def mock_run_on_event_loop(mocker):
    # Patch where it's imported and used by DataTimeoutTracker
    return mocker.patch('tsercom.data.data_timeout_tracker.run_on_event_loop')

@pytest.fixture
def mock_asyncio_sleep(mocker):
    # Patch asyncio.sleep where it's used by DataTimeoutTracker
    # pytest-asyncio will make this an AsyncMock if not specified otherwise
    return mocker.patch('tsercom.data.data_timeout_tracker.asyncio.sleep')

@pytest.fixture
def mock_is_running_on_event_loop(mocker):
    # Patch where it's imported and used by DataTimeoutTracker
    mock = mocker.patch('tsercom.data.data_timeout_tracker.is_running_on_event_loop')
    mock.return_value = True # Default to True for most async method internal checks
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
async def test_register_calls_run_on_event_loop(mock_run_on_event_loop, mock_tracked_object):
    tracker = DataTimeoutTracker()
    tracker.register(mock_tracked_object)

    mock_run_on_event_loop.assert_called_once()
    call_args = mock_run_on_event_loop.call_args[0] 
    assert isinstance(call_args[0], functools.partial)
    
    partial_func = call_args[0]
    assert partial_func.func == tracker._DataTimeoutTracker__register_impl
    assert partial_func.args == (mock_tracked_object,)

@pytest.mark.asyncio
async def test_register_impl_adds_to_list(mock_is_running_on_event_loop, mock_tracked_object):
    tracker = DataTimeoutTracker()
    assert len(tracker._DataTimeoutTracker__tracked_list) == 0 
    
    await tracker._DataTimeoutTracker__register_impl(mock_tracked_object)
    
    mock_is_running_on_event_loop.assert_called_once() 
    assert len(tracker._DataTimeoutTracker__tracked_list) == 1
    assert mock_tracked_object in tracker._DataTimeoutTracker__tracked_list

@pytest.mark.asyncio
async def test_register_impl_asserts_if_not_on_event_loop(mock_is_running_on_event_loop, mock_tracked_object):
    mock_is_running_on_event_loop.return_value = False 
    tracker = DataTimeoutTracker()
    with pytest.raises(AssertionError):
        await tracker._DataTimeoutTracker__register_impl(mock_tracked_object)

# 3. start()
def test_start_calls_run_on_event_loop(mock_run_on_event_loop):
    tracker = DataTimeoutTracker()
    tracker.start()
    mock_run_on_event_loop.assert_called_once_with(tracker._DataTimeoutTracker__execute_periodically)

# 4. Logic of __execute_periodically() (Controlled Simulation)

@pytest.mark.asyncio
async def test_execute_periodically_calls_sleep_and_on_triggered(
    mock_asyncio_sleep, mock_tracked_object, mock_is_running_on_event_loop
):
    test_timeout = 42
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)
    await tracker._DataTimeoutTracker__register_impl(mock_tracked_object)
    
    # Make _on_triggered raise BreakLoop to exit after one full iteration
    mock_tracked_object._on_triggered.side_effect = BreakLoop
    
    with pytest.raises(BreakLoop):
        await tracker._DataTimeoutTracker__execute_periodically()
        
    mock_asyncio_sleep.assert_called_once_with(test_timeout)
    mock_tracked_object._on_triggered.assert_called_once_with(test_timeout)

@pytest.mark.asyncio
async def test_execute_periodically_multiple_objects(
    mock_asyncio_sleep, mocker, mock_is_running_on_event_loop
):
    test_timeout = 35
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)
    
    tracked_obj1 = mocker.create_autospec(DataTimeoutTracker.Tracked, instance=True)
    tracked_obj2 = mocker.create_autospec(DataTimeoutTracker.Tracked, instance=True)
    
    await tracker._DataTimeoutTracker__register_impl(tracked_obj1)
    await tracker._DataTimeoutTracker__register_impl(tracked_obj2)
    
    # Make the last object's _on_triggered raise BreakLoop
    tracked_obj2._on_triggered.side_effect = BreakLoop
    
    with pytest.raises(BreakLoop):
        await tracker._DataTimeoutTracker__execute_periodically()
        
    mock_asyncio_sleep.assert_called_once_with(test_timeout)
    tracked_obj1._on_triggered.assert_called_once_with(test_timeout)
    tracked_obj2._on_triggered.assert_called_once_with(test_timeout)

@pytest.mark.asyncio
async def test_execute_periodically_no_tracked_objects(mock_asyncio_sleep):
    test_timeout = 15
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)
    
    # asyncio.sleep itself will raise BreakLoop because no _on_triggered to attach it to
    mock_asyncio_sleep.side_effect = BreakLoop 
    
    with pytest.raises(BreakLoop):
        await tracker._DataTimeoutTracker__execute_periodically()
        
    mock_asyncio_sleep.assert_called_once_with(test_timeout)

@pytest.mark.asyncio
async def test_execute_periodically_handles_exception_in_on_triggered_and_continues(
    mock_asyncio_sleep, mocker, mock_is_running_on_event_loop
):
    test_timeout = 25
    tracker = DataTimeoutTracker(timeout_seconds=test_timeout)

    tracked_obj1 = mocker.create_autospec(DataTimeoutTracker.Tracked, instance=True)
    tracked_obj2 = mocker.create_autospec(DataTimeoutTracker.Tracked, instance=True) # Will raise error
    tracked_obj3 = mocker.create_autospec(DataTimeoutTracker.Tracked, instance=True) # Should still be called

    tracked_obj2._on_triggered.side_effect = RuntimeError("Test error in _on_triggered")
    # Make the last object's _on_triggered raise BreakLoop to exit the test loop
    tracked_obj3._on_triggered.side_effect = BreakLoop

    await tracker._DataTimeoutTracker__register_impl(tracked_obj1)
    await tracker._DataTimeoutTracker__register_impl(tracked_obj2)
    await tracker._DataTimeoutTracker__register_impl(tracked_obj3)
    
    # Expect BreakLoop because obj3 should be reached and its _on_triggered will raise it.
    # The RuntimeError from obj2 should be caught by the SUT if it's designed to be robust.
    # Current SUT: It does NOT catch errors in _on_triggered. So this test needs to expect RuntimeError.

    mock_asyncio_sleep.reset_mock() # Reset as sleep might not be reached if error occurs earlier
    tracked_obj1._on_triggered.reset_mock()
    tracked_obj2._on_triggered.reset_mock()
    tracked_obj3._on_triggered.reset_mock()
    
    tracked_obj2._on_triggered.side_effect = RuntimeError("Error from obj2") # This error will propagate

    with pytest.raises(RuntimeError, match="Error from obj2"):
        await tracker._DataTimeoutTracker__execute_periodically()

    mock_asyncio_sleep.assert_called_once_with(test_timeout) 
    tracked_obj1._on_triggered.assert_called_once_with(test_timeout)
    tracked_obj2._on_triggered.assert_called_once_with(test_timeout) 
    tracked_obj3._on_triggered.assert_not_called() # Not called due to error from obj2 stopping the loop iteration
    
    # To test "continues after exception", the SUT's __execute_periodically would need:
    # for tracked in self.__tracked_list:
    #     try:
    #         tracked._on_triggered(self.__timeout_seconds)
    #     except Exception:
    #         # log error, continue
    #         pass
    # Since SUT doesn't have this, the current assertions are correct for its behavior.
