import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import functools # For functools.partial
from collections import deque # For kMaxResponses verification

from tsercom.threading.async_poller import AsyncPoller
# Import the module to be patched
import tsercom.threading.aio.aio_utils as aio_utils_to_patch

# kMaxResponses from async_poller.py, assuming it's 30
K_MAX_RESPONSES = 30

@pytest.fixture
async def mock_aio_utils(monkeypatch):
    """
    Mocks aio_utils functions used by AsyncPoller.
    run_on_event_loop is mocked to execute the passed partial (which wraps a sync method)
    by directly calling it. Since __set_results_available is synchronous, we don't need
    to make the mock_run_on_event_loop itself async or await the partial.
    """
    
    # Mock for run_on_event_loop:
    # The target method __set_results_available is synchronous.
    # run_on_event_loop in SUT is called without await, returning a Future.
    # Our mock will execute the partial immediately and return a completed Future.
    mock_run_on_event_loop_sync_exec = MagicMock(name="mock_run_on_event_loop_sync_exec")

    def simplified_run_on_loop_side_effect(partial_func, loop=None, *args, **kwargs):
        print(f"MOCKED run_on_event_loop CALLED with partial: {partial_func}")
        # partial_func is functools.partial(self._AsyncPoller__set_results_available)
        # This is a synchronous method.
        partial_func() # Execute the synchronous method
        print(f"  Partial function {getattr(partial_func, 'func', 'N/A').__name__} executed.")
        
        # Return a completed Future, as the original does
        f = asyncio.Future()
        # Ensure the future is associated with the correct loop for awaiters
        try:
            current_loop = asyncio.get_running_loop()
            if not current_loop.is_closed():
                 asyncio.ensure_future(f, loop=current_loop)
        except RuntimeError: # pragma: no cover
            pass # No loop running or other issue
        f.set_result(None)
        return f
        
    mock_run_on_event_loop_sync_exec.side_effect = simplified_run_on_loop_side_effect

    # Mock for get_running_loop_or_none:
    mock_get_running_loop = MagicMock(name="mock_get_running_loop_or_none")
    # Default: return the current loop the test is running on
    mock_get_running_loop.return_value = asyncio.get_running_loop() 

    # Mock for is_running_on_event_loop:
    mock_is_on_loop = MagicMock(name="mock_is_running_on_event_loop")
    # Default: pretend we are always on the "correct" loop initially
    mock_is_on_loop.return_value = True 

    monkeypatch.setattr(aio_utils_to_patch, "run_on_event_loop", mock_run_on_event_loop_sync_exec)
    monkeypatch.setattr(aio_utils_to_patch, "get_running_loop_or_none", mock_get_running_loop)
    monkeypatch.setattr(aio_utils_to_patch, "is_running_on_event_loop", mock_is_on_loop)
    
    print("Patched aio_utils methods for AsyncPoller tests.")
    yield {
        "run_on_event_loop": mock_run_on_event_loop_sync_exec,
        "get_running_loop_or_none": mock_get_running_loop,
        "is_running_on_event_loop": mock_is_on_loop
    }
    print("Unpatched aio_utils methods.")


@pytest.mark.asyncio
class TestAsyncPoller:

    @pytest.fixture(autouse=True)
    def _ensure_aio_utils_mocked(self, mock_aio_utils):
        # This fixture ensures mock_aio_utils is activated for every test in this class
        pass

    async def test_on_available_and_wait_instance_single_item(self):
        print("\n--- Test: test_on_available_and_wait_instance_single_item ---")
        poller = AsyncPoller[str](name="TestPollerSingle")
        item1 = "item_one"

        poller.on_available(item1)
        print(f"  Item '{item1}' made available.")
        
        result = await poller.wait_instance()
        print(f"  wait_instance() returned: {result}")
        
        assert result == [item1]
        assert len(poller) == 0, "Poller should be empty after wait_instance"
        print("--- Test: test_on_available_and_wait_instance_single_item finished ---")

    async def test_wait_instance_blocks_until_on_available(self):
        print("\n--- Test: test_wait_instance_blocks_until_on_available ---")
        poller = AsyncPoller[str](name="TestPollerBlocks")
        item1 = "item_blocker"

        wait_task = asyncio.create_task(poller.wait_instance())
        
        # Give a moment for wait_task to actually start and block on the event
        await asyncio.sleep(0.01) 
        assert not wait_task.done(), "wait_instance should be blocked before item is available"
        print("  wait_instance confirmed blocked.")

        poller.on_available(item1)
        print(f"  Item '{item1}' made available.")
        
        # Now wait_task should complete
        result = await asyncio.wait_for(wait_task, timeout=1.0) # Wait with timeout
        print(f"  wait_instance task completed with result: {result}")
        
        assert result == [item1]
        assert len(poller) == 0
        print("--- Test: test_wait_instance_blocks_until_on_available finished ---")

    async def test_multiple_items_retrieved_in_order(self):
        print("\n--- Test: test_multiple_items_retrieved_in_order ---")
        poller = AsyncPoller[int](name="TestPollerMultiOrder")
        item1, item2, item3 = 1, 2, 3

        poller.on_available(item1)
        poller.on_available(item2)
        poller.on_available(item3)
        print("  Items 1, 2, 3 made available.")
        assert len(poller) == 3

        result = await poller.wait_instance()
        print(f"  wait_instance() returned: {result}")
        
        assert result == [item1, item2, item3]
        assert len(poller) == 0
        print("--- Test: test_multiple_items_retrieved_in_order finished ---")

    async def test_queue_limit_kMaxResponses(self):
        print("\n--- Test: test_queue_limit_kMaxResponses ---")
        poller = AsyncPoller[int](name="TestPollerLimit")
        
        # Add more items than K_MAX_RESPONSES
        num_items_to_add = K_MAX_RESPONSES + 5 
        items_added = list(range(num_items_to_add))
        for i in items_added:
            poller.on_available(i)
        print(f"  Added {num_items_to_add} items.")
        
        assert len(poller) == K_MAX_RESPONSES, f"Poller length should be capped at {K_MAX_RESPONSES}"
        print(f"  Poller length confirmed as {len(poller)}.")

        result = await poller.wait_instance()
        print(f"  wait_instance() returned {len(result)} items.")
        
        assert len(result) == K_MAX_RESPONSES
        # The poller uses a deque with append (right) and trimming from left if over limit.
        # So, the result should be the *last* K_MAX_RESPONSES items.
        expected_items = items_added[-K_MAX_RESPONSES:]
        assert result == expected_items, "Poller did not return the last K_MAX_RESPONSES items"
        assert len(poller) == 0
        print("--- Test: test_queue_limit_kMaxResponses finished ---")

    async def test_flush_clears_queue(self):
        print("\n--- Test: test_flush_clears_queue ---")
        poller = AsyncPoller[str](name="TestPollerFlush")
        poller.on_available("item_a")
        poller.on_available("item_b")
        assert len(poller) == 2
        print("  Items added, poller length is 2.")

        poller.flush()
        print("  Poller flushed.")
        
        assert len(poller) == 0, "Poller should be empty after flush"

        # Verify wait_instance blocks (or times out)
        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01) # Give task a chance to run
        assert not wait_task.done(), "wait_instance should block after flush"
        print("  wait_instance confirmed blocked after flush.")
        
        # Clean up the task to avoid "never awaited" warnings
        wait_task.cancel()
        try:
            await wait_task
        except asyncio.CancelledError:
            print("  wait_task cancelled as expected.")
        print("--- Test: test_flush_clears_queue finished ---")

    async def test_len_accurate(self):
        print("\n--- Test: test_len_accurate ---")
        poller = AsyncPoller[float](name="TestPollerLen")
        assert len(poller) == 0
        
        poller.on_available(1.0)
        assert len(poller) == 1
        
        poller.on_available(2.0)
        assert len(poller) == 2
        print(f"  Poller length is {len(poller)} after adding 2 items.")

        await poller.wait_instance()
        assert len(poller) == 0, "Poller should be empty after wait_instance"
        print("  Poller length is 0 after wait_instance.")
        print("--- Test: test_len_accurate finished ---")

    async def test_async_iterator(self):
        print("\n--- Test: test_async_iterator ---")
        poller = AsyncPoller[str](name="TestPollerAsyncIter")
        item_x = "item_x"
        item_y = "item_y"

        # Scenario 1: Items available before starting iteration
        poller.on_available(item_x)
        poller.on_available(item_y)
        print(f"  Items '{item_x}', '{item_y}' made available.")

        collected_via_iter = []
        # Start a task that will add more items while iterating
        async def add_more_items_later():
            await asyncio.sleep(0.02) # Ensure iterator is likely waiting
            print("  Adding item_z during iteration.")
            poller.on_available("item_z")
            await asyncio.sleep(0.02) # Allow iterator to pick it up
            print("  Adding item_w during iteration.")
            poller.on_available("item_w") 
            # To terminate the iterator for this test, we might need to signal it
            # or rely on a limited number of iterations if the poller stops producing.
            # For now, we'll add a known number of items and expect that many yields.
            # The current AsyncPoller's iterator doesn't have an explicit stop condition other than no items.
            # If wait_instance blocks indefinitely, the async for loop will also block.
            # For testing, we'll rely on items being processed.
            # To make it finite for the test, we can call flush or stop (if poller had stop).
            # Or, we can limit the number of iterations in the test's async for loop.
            # For now, let's assume two yields are expected based on two separate calls to on_available after wait_instance.
            # The current implementation of wait_instance returns ALL items.
            # So, the iterator will yield lists of items available since the last yield.

        iter_count = 0
        async for item_list in poller:
            print(f"  Async iterator yielded: {item_list}")
            collected_via_iter.append(item_list)
            iter_count += 1
            if iter_count == 1: # First yield should get item_x, item_y
                assert item_list == [item_x, item_y]
                # Start task to add more items *after* this first yield
                asyncio.create_task(add_more_items_later())
            elif iter_count == 2: # Second yield should get item_z
                 assert item_list == ["item_z"]
            elif iter_count == 3: # Third yield should get item_w
                 assert item_list == ["item_w"]
                 break # Exit loop for test finiteness
            if iter_count > 3: # Failsafe
                 pytest.fail("Iterator yielded more times than expected") # pragma: no cover

        assert iter_count == 3, "Iterator did not yield the expected number of times"
        # Expected structure: [[item_x, item_y], [item_z], [item_w]] - based on how on_available triggers event
        expected_full_collection = [[item_x, item_y], ["item_z"], ["item_w"]]
        assert collected_via_iter == expected_full_collection
        print(f"  Collected via iterator: {collected_via_iter}")
        print("--- Test: test_async_iterator finished ---")

    async def test_event_loop_property_set(self, mock_aio_utils):
        print("\n--- Test: test_event_loop_property_set ---")
        poller = AsyncPoller[int](name="TestPollerEventLoop")
        
        assert poller.event_loop is None, "Event loop should initially be None"
        print("  event_loop is None initially.")

        # Call wait_instance once to set the event loop
        # Need to provide an item or it will block indefinitely
        poller.on_available(100)
        await poller.wait_instance()
        print("  wait_instance called once.")

        assert poller.event_loop is not None, "Event loop should be set after wait_instance"
        assert poller.event_loop is asyncio.get_running_loop(), "Event loop not set to current loop"
        # Check if get_running_loop_or_none was used if poller's loop was initially None
        mock_aio_utils["get_running_loop_or_none"].assert_called() # Called by __ensure_event_set
        print(f"  poller.event_loop is now {poller.event_loop}.")
        print("--- Test: test_event_loop_property_set finished ---")

    async def test_run_on_event_loop_called_for_set_results_available(self, mock_aio_utils):
        print("\n--- Test: test_run_on_event_loop_called_for_set_results_available ---")
        poller = AsyncPoller[str](name="TestPollerRunOnLoop")
        item_signal = "signal_item"

        # First, call wait_instance to set the poller's __event_loop and __is_loop_running
        poller.on_available("initial_item_for_setup")
        await poller.wait_instance()
        print("  Initial wait_instance completed to set up loop state.")
        
        # Reset mock for run_on_event_loop to check specific call by subsequent on_available
        mock_aio_utils["run_on_event_loop"].reset_mock()
        
        # Configure is_running_on_event_loop to return False to trigger run_on_event_loop
        mock_aio_utils["is_running_on_event_loop"].return_value = False
        print("  is_running_on_event_loop mocked to return False.")

        poller.on_available(item_signal)
        print(f"  Item '{item_signal}' made available (should trigger run_on_event_loop).")

        # Assert run_on_event_loop was called
        mock_aio_utils["run_on_event_loop"].assert_called_once()
        # Check the arguments of the call to run_on_event_loop
        # First arg (args[0]) is the partial function
        # Second arg (args[1]) is the event loop instance
        call_args_list = mock_aio_utils["run_on_event_loop"].call_args_list
        assert len(call_args_list) == 1
        
        partial_arg = call_args_list[0][0][0] # The functools.partial object
        loop_arg = call_args_list[0][0][1]     # The event loop
        
        assert isinstance(partial_arg, functools.partial), "run_on_event_loop not called with a partial"
        assert partial_arg.func.__name__ == "_AsyncPoller__set_results_available", \
            "run_on_event_loop not called with __set_results_available"
        assert loop_arg is poller.event_loop, "run_on_event_loop not called with the poller's event loop"
        
        print("  run_on_event_loop assertions passed.")
        print("--- Test: test_run_on_event_loop_called_for_set_results_available finished ---")

```

**Summary of Implementation:**
1.  **Imports**: Added necessary modules.
2.  **`K_MAX_RESPONSES`**: Defined for test clarity.
3.  **`mock_aio_utils` Fixture**:
    *   Patches `run_on_event_loop`, `get_running_loop_or_none`, and `is_running_on_event_loop` in `tsercom.threading.aio.aio_utils` (the source module, which `AsyncPoller` then imports from).
    *   The mock for `run_on_event_loop` (`mock_run_on_event_loop_sync_exec`) is designed to directly execute the synchronous `_AsyncPoller__set_results_available` method (when wrapped by `functools.partial`) and return a completed `Future`. This is based on the assumption that `__set_results_available` is synchronous.
4.  **`TestAsyncPoller` Class**:
    *   Uses an `autouse` fixture `_ensure_aio_utils_mocked` to activate `mock_aio_utils` for all tests.
    *   **`test_on_available_and_wait_instance_single_item`**: Verifies basic add and retrieve.
    *   **`test_wait_instance_blocks_until_on_available`**: Checks that `wait_instance` blocks until an item is available using `asyncio.sleep` and task state.
    *   **`test_multiple_items_retrieved_in_order`**: Checks multiple items are retrieved in FIFO order and queue is cleared.
    *   **`test_queue_limit_kMaxResponses`**: Verifies the queue size limit (30) and that the last 30 items are kept.
    *   **`test_flush_clears_queue`**: Checks `flush()` empties the queue and `wait_instance` subsequently blocks.
    *   **`test_len_accurate`**: Verifies `len(poller)` reports correct size.
    *   **`test_async_iterator`**: Tests `async for item_list in poller:`. It verifies that items added dynamically are picked up. The current `AsyncPoller` yields all available items in one list per `wait_instance` call.
    *   **`test_event_loop_property_set`**: Confirms `poller.event_loop` is set after the first `wait_instance`.
    *   **`test_run_on_event_loop_called_for_set_results_available`**: Checks that `run_on_event_loop` (the mock) is called by `on_available` when `__is_loop_running` is true and the current loop is different (simulated by setting `is_running_on_event_loop` mock to `False`).

This suite of tests covers the specified methods and scenarios. The mocking strategy for `aio_utils` is central to enabling these tests.The test file `tsercom/threading/async_poller_unittest.py` has been written.
It includes:
- A fixture `mock_aio_utils` that patches `run_on_event_loop`, `get_running_loop_or_none`, and `is_running_on_event_loop` from `tsercom.threading.aio.aio_utils`. The mock for `run_on_event_loop` executes the synchronous `_AsyncPoller__set_results_available` method (when wrapped by `functools.partial`) and returns a completed `Future`.
- A test class `TestAsyncPoller` with various test methods for `AsyncPoller`'s functionality:
    - `test_on_available_and_wait_instance_single_item`
    - `test_wait_instance_blocks_until_on_available`
    - `test_multiple_items_retrieved_in_order`
    - `test_queue_limit_kMaxResponses` (verifying queue truncation at 30 items)
    - `test_flush_clears_queue`
    - `test_len_accurate`
    - `test_async_iterator` (verifying `async for` behavior)
    - `test_event_loop_property_set`
    - `test_run_on_event_loop_called_for_set_results_available` (verifying `run_on_event_loop` is called under specific conditions)

The tests are designed to cover the methods and scenarios outlined in the subtask description.
I will now run these tests. The print statements within the tests and mocks will help diagnose any issues.
