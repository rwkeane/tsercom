import asyncio
import pytest
import threading
import time
from typing import Any, Coroutine, Optional, Tuple, Dict, List

from tsercom.threading.aio import aio_utils
from tsercom.threading.aio import global_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher # For global_event_loop setup

# --- Helper Coroutines ---

async def simple_coro(value: Any, delay: float = 0) -> Any:
    """A simple coroutine that returns a value after a delay."""
    if delay > 0:
        await asyncio.sleep(delay)
    return value

async def failing_coro(message: str = "Coroutine failed", delay: float = 0) -> None:
    """A coroutine that raises an exception after a delay."""
    if delay > 0:
        await asyncio.sleep(delay)
    raise ValueError(message)

async def coro_with_args_kwargs(
    pos_arg1: Any, pos_arg2: Any, *, kw_arg1: Any, kw_arg2: Any
) -> Dict[str, Any]:
    """A coroutine that returns its arguments."""
    return {
        "pos_arg1": pos_arg1,
        "pos_arg2": pos_arg2,
        "kw_arg1": kw_arg1,
        "kw_arg2": kw_arg2,
    }

async def coro_check_current_loop() -> asyncio.AbstractEventLoop:
    """Coroutine that returns the loop it's running on."""
    return asyncio.get_running_loop()


# --- Fixtures ---

@pytest.fixture
def new_event_loop() -> asyncio.AbstractEventLoop:
    """Creates a new event loop and closes it after the test."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        if not loop.is_closed():
            # Cancel all tasks on the loop before closing
            # From Python 3.9, loop.shutdown_default_executor() and loop.shutdown_asyncgens() exist
            # For broader compatibility, we can iterate and cancel tasks manually if needed,
            # but usually just closing is fine for simple test cases if tasks complete.
            # If tasks might be pending, more robust cleanup is needed.
            # For these tests, tasks are generally awaited or short-lived.
            loop.call_soon_threadsafe(loop.stop)
            # Give a chance for the loop to stop if it's running in a thread (not typical for this fixture)
            # time.sleep(0.01) 
            loop.close()


@pytest.fixture
def managed_global_loop():
    """Sets up and tears down the tsercom global event loop for a test."""
    watcher = ThreadWatcher() # Required by create_tsercom_event_loop_from_watcher
    try:
        # In case a previous test left a global loop uncleared (should not happen with good tests)
        if global_event_loop.is_global_event_loop_set():
            global_event_loop.clear_tsercom_event_loop()

        global_event_loop.create_tsercom_event_loop_from_watcher(watcher)
        yield global_event_loop.get_global_event_loop()
    finally:
        if global_event_loop.is_global_event_loop_set():
            global_event_loop.clear_tsercom_event_loop()
        # Ensure watcher's resources are also cleaned if it holds any (e.g. threads)
        # ThreadWatcher itself doesn't explicitly start threads unless its methods are called.


@pytest.fixture(autouse=True)
def ensure_no_global_loop_unless_managed(request):
    """Ensures that tests not explicitly using a global loop fixture start clean."""
    if "managed_global_loop" not in request.fixturenames:
        if global_event_loop.is_global_event_loop_set():
            global_event_loop.clear_tsercom_event_loop()
    yield
    if "managed_global_loop" not in request.fixturenames:
        if global_event_loop.is_global_event_loop_set():
            # This would indicate a test leaked a global loop
            global_event_loop.clear_tsercom_event_loop()
            # Consider raising an error here if strict about test isolation
            # raise RuntimeError("Test leaked a global event loop setting.")


# --- Tests for get_running_loop_or_none() ---

def test_get_running_loop_or_none_outside_loop():
    """Test get_running_loop_or_none() when called from outside an event loop."""
    assert aio_utils.get_running_loop_or_none() is None

@pytest.mark.asyncio
async def test_get_running_loop_or_none_inside_loop():
    """Test get_running_loop_or_none() when called from within a running event loop."""
    current_loop = asyncio.get_running_loop()
    assert aio_utils.get_running_loop_or_none() is current_loop


# --- Tests for is_running_on_event_loop() ---

def test_is_running_on_event_loop_outside_loop_no_specific():
    """Test is_running_on_event_loop(None) outside a loop."""
    assert not aio_utils.is_running_on_event_loop(None)

@pytest.mark.asyncio
async def test_is_running_on_event_loop_inside_loop_no_specific():
    """Test is_running_on_event_loop(None) inside a loop."""
    assert aio_utils.is_running_on_event_loop(None)

@pytest.mark.asyncio
async def test_is_running_on_event_loop_inside_loop_specific_match():
    """Test is_running_on_event_loop(specific_loop) inside that specific loop."""
    current_loop = asyncio.get_running_loop()
    assert aio_utils.is_running_on_event_loop(current_loop)

@pytest.mark.asyncio
async def test_is_running_on_event_loop_inside_loop_specific_mismatch(new_event_loop):
    """Test is_running_on_event_loop(specific_loop) inside a *different* loop."""
    # This test needs to ensure the context is running on pytest-asyncio's loop,
    # while checking against `new_event_loop` which is different.
    current_loop = asyncio.get_running_loop()
    assert current_loop is not new_event_loop 
    assert not aio_utils.is_running_on_event_loop(new_event_loop)

def test_is_running_on_event_loop_outside_loop_specific(new_event_loop):
    """Test is_running_on_event_loop(specific_loop) outside any loop."""
    assert not aio_utils.is_running_on_event_loop(new_event_loop)


# --- Tests for run_on_event_loop() ---

# Helper to run a future and get its result in the current thread
def await_future_in_thread(future: asyncio.Future, timeout: float = 1.0) -> Any:
    # This is a simplified way to get future result from a different thread's loop.
    # More robust mechanisms might be needed for complex scenarios.
    # asyncio.Future.result() can block, but it needs the loop to be running.
    # A common pattern is to use a threading.Event for signaling.
    event = threading.Event()
    result = None
    exc = None

    def callback(fut):
        nonlocal result, exc
        try:
            result = fut.result()
        except Exception as e:
            exc = e
        finally:
            event.set()
            
    # Assuming the loop that `future` belongs to is running in another thread (e.g. global_loop)
    future._loop.call_soon_threadsafe(future.add_done_callback, callback) # type: ignore

    if not event.wait(timeout):
        raise TimeoutError("Timeout waiting for future to complete")
    
    if exc:
        raise exc
    return result


# 1. With Global Loop
def test_run_on_event_loop_with_global_loop_success(managed_global_loop):
    """Test run_on_event_loop with a global loop: successful coroutine."""
    expected_value = "global_success"
    future = aio_utils.run_on_event_loop(simple_coro, value=expected_value, delay=0.01)
    
    result = await_future_in_thread(future, timeout=0.5)
    assert result == expected_value

    # Check it ran on the global loop
    loop_check_future = aio_utils.run_on_event_loop(coro_check_current_loop)
    loop_it_ran_on = await_future_in_thread(loop_check_future, timeout=0.5)
    assert loop_it_ran_on is managed_global_loop

def test_run_on_event_loop_with_global_loop_args_kwargs(managed_global_loop):
    """Test run_on_event_loop with global loop: argument passing."""
    pos_args: Tuple[Any, ...] = ("val1", 100)
    kw_args: Dict[str, Any] = {"kw_arg1": "val2", "kw_arg2": 200}
    expected_dict = {
        "pos_arg1": pos_args[0], "pos_arg2": pos_args[1],
        "kw_arg1": kw_args["kw_arg1"], "kw_arg2": kw_args["kw_arg2"]
    }

    future = aio_utils.run_on_event_loop(coro_with_args_kwargs, *pos_args, **kw_args)
    result = await_future_in_thread(future, timeout=0.5)
    assert result == expected_dict

def test_run_on_event_loop_with_global_loop_exception(managed_global_loop):
    """Test run_on_event_loop with global loop: exception propagation."""
    error_message = "Global loop failure"
    future = aio_utils.run_on_event_loop(failing_coro, message=error_message, delay=0.01)

    with pytest.raises(ValueError, match=error_message):
        await_future_in_thread(future, timeout=0.5)


# 2. With Specified Loop
def test_run_on_event_loop_with_specified_loop(new_event_loop):
    """Test run_on_event_loop with an explicitly specified event loop."""
    # Need to run this loop in a separate thread for run_coroutine_threadsafe to work
    loop_thread = threading.Thread(target=new_event_loop.run_forever, daemon=True)
    loop_thread.start()
    
    # Wait for loop to start
    while not new_event_loop.is_running():
        time.sleep(0.001)

    expected_value = "specified_loop_success"
    future = aio_utils.run_on_event_loop(simple_coro, new_event_loop, value=expected_value, delay=0.01)
    
    result = await_future_in_thread(future, timeout=0.5)
    assert result == expected_value

    # Check it ran on the specified loop
    loop_check_future = aio_utils.run_on_event_loop(coro_check_current_loop, new_event_loop)
    loop_it_ran_on = await_future_in_thread(loop_check_future, timeout=0.5)
    assert loop_it_ran_on is new_event_loop
    
    # Clean up the loop thread
    new_event_loop.call_soon_threadsafe(new_event_loop.stop)
    loop_thread.join(timeout=0.5)
    # The fixture will close it

def test_run_on_event_loop_with_specified_loop_exception(new_event_loop):
    """Test run_on_event_loop with specified loop: exception propagation."""
    loop_thread = threading.Thread(target=new_event_loop.run_forever, daemon=True)
    loop_thread.start()
    while not new_event_loop.is_running():
        time.sleep(0.001)

    error_message = "Specified loop failure"
    future = aio_utils.run_on_event_loop(failing_coro, new_event_loop, message=error_message, delay=0.01)

    with pytest.raises(ValueError, match=error_message):
        await_future_in_thread(future, timeout=0.5)
        
    new_event_loop.call_soon_threadsafe(new_event_loop.stop)
    loop_thread.join(timeout=0.5)


# 3. Error Case (No Global Loop, No Specified Loop)
def test_run_on_event_loop_no_global_or_specified_loop():
    """Test run_on_event_loop raises RuntimeError if no loop is available."""
    # Ensure no global loop is set (autouse fixture should handle this, but double-check)
    if global_event_loop.is_global_event_loop_set():
        global_event_loop.clear_tsercom_event_loop()
    
    with pytest.raises(RuntimeError, match="ERROR: tsercom global event loop not set!"):
        aio_utils.run_on_event_loop(simple_coro, value="test")

# --- Edge cases for global loop setup ---

def test_run_on_event_loop_global_loop_set_then_cleared_then_error():
    """Test behavior if global loop was set then cleared."""
    watcher = ThreadWatcher()
    global_event_loop.create_tsercom_event_loop_from_watcher(watcher)
    assert global_event_loop.is_global_event_loop_set()
    
    # Clear it
    global_event_loop.clear_tsercom_event_loop()
    assert not global_event_loop.is_global_event_loop_set()

    with pytest.raises(RuntimeError, match="ERROR: tsercom global event loop not set!"):
        aio_utils.run_on_event_loop(simple_coro, value="test")

def test_set_tsercom_event_loop_manually(new_event_loop):
    """Test run_on_event_loop when global loop is set manually via set_tsercom_event_loop."""
    loop_thread = threading.Thread(target=new_event_loop.run_forever, daemon=True)
    loop_thread.start()
    while not new_event_loop.is_running():
        time.sleep(0.001)

    try:
        global_event_loop.set_tsercom_event_loop(new_event_loop)
        assert global_event_loop.get_global_event_loop() is new_event_loop

        expected_value = "manual_global_set_success"
        future = aio_utils.run_on_event_loop(simple_coro, value=expected_value, delay=0.01)
        result = await_future_in_thread(future, timeout=0.5)
        assert result == expected_value
    finally:
        if global_event_loop.is_global_event_loop_set():
            # set_tsercom_event_loop doesn't store the factory, so clear_tsercom_event_loop won't stop it.
            # We need to stop it manually.
            new_event_loop.call_soon_threadsafe(new_event_loop.stop)
            global_event_loop.clear_tsercom_event_loop() # Clears the global variable
        
        loop_thread.join(timeout=0.5)
