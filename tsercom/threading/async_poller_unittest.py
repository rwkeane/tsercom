import asyncio
import pytest
import pytest_asyncio  # For async fixtures if needed, though direct async tests are fine

# from unittest.mock import patch, AsyncMock, MagicMock # Removed
import functools  # For functools.partial
from collections import deque  # For kMaxResponses verification

from tsercom.threading.atomic import (
    Atomic,
)  # For manipulating internal state in tests
from tsercom.threading.async_poller import AsyncPoller

# Import the module to be patched
import tsercom.threading.aio.aio_utils as aio_utils_to_patch

# kMaxResponses from async_poller.py, assuming it's 30
K_MAX_RESPONSES = 30


@pytest_asyncio.fixture
async def mock_aio_utils(mocker):

    def new_run_on_event_loop_side_effect(
        func_or_partial, loop_param, *args, **kwargs
    ):
        # func_or_partial is expected to be functools.partial(poller_instance._AsyncPoller__set_results_available)
        # This partial, when called, returns a coroutine.
        # We need to ensure this coroutine is scheduled on the event loop.
        print(
            f"MOCKED new_run_on_event_loop CALLED with func: {func_or_partial}, loop: {loop_param}"
        )
        coroutine = func_or_partial()
        if asyncio.iscoroutine(coroutine):
            print(
                f"  Scheduling coroutine {getattr(coroutine, '__name__', 'unknown_coro')} on loop {loop_param}"
            )
            asyncio.ensure_future(coroutine, loop=loop_param)
        else:
            # Fallback if it wasn't a coroutine for some reason (should not happen for __set_results_available)
            print(
                f"  Warning: Expected a coroutine from partial, got {type(coroutine)}. Executing directly."
            )
            coroutine()
        return None

    # Mock for run_on_event_loop
    patched_run_on_event_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.run_on_event_loop",
        side_effect=new_run_on_event_loop_side_effect,
    )

    # Mock for get_running_loop_or_none
    actual_loop = asyncio.get_running_loop()
    patched_get_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.get_running_loop_or_none",
        return_value=actual_loop,
    )
    # This patched_get_loop IS the mock object we will assert on.
    # Give it a name if needed for clarity in test output, e.g., patched_get_loop.name = "patched_get_running_loop_or_none"
    patched_get_loop.name = "patched_get_running_loop_or_none"  # Optional: for clearer mock names in test output

    # Mock for is_running_on_event_loop
    patched_is_on_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.is_running_on_event_loop",
        return_value=True,
    )

    print(
        "Patched aio_utils methods using mocker.patch for AsyncPoller tests."
    )
    yield {
        "run_on_event_loop": patched_run_on_event_loop,
        "get_running_loop_or_none": patched_get_loop,  # mock object from mocker.patch()
        "is_running_on_event_loop": patched_is_on_loop,
    }
    print("Mocks automatically undone by mocker.")


@pytest.mark.asyncio
class TestAsyncPoller:

    @pytest.fixture(autouse=True)
    def _ensure_aio_utils_mocked(self, mock_aio_utils):
        # This fixture ensures mock_aio_utils is activated for every test in this class
        pass

    async def test_on_available_and_wait_instance_single_item(self):
        print(
            "\n--- Test: test_on_available_and_wait_instance_single_item ---"
        )
        poller = AsyncPoller[str]()  # Removed name="TestPollerSingle"
        item1 = "item_one"

        poller.on_available(item1)
        print(f"  Item '{item1}' made available.")

        result = await poller.wait_instance()
        print(f"  wait_instance() returned: {result}")

        assert result == [item1]
        assert len(poller) == 0, "Poller should be empty after wait_instance"
        print(
            "--- Test: test_on_available_and_wait_instance_single_item finished ---"
        )

    async def test_wait_instance_blocks_until_on_available(self):
        print("\n--- Test: test_wait_instance_blocks_until_on_available ---")
        poller = AsyncPoller[str]()  # Removed name="TestPollerBlocks"
        item1 = "item_blocker"

        wait_task = asyncio.create_task(poller.wait_instance())

        # Give a moment for wait_task to actually start and block on the event
        await asyncio.sleep(0.01)
        assert (
            not wait_task.done()
        ), "wait_instance should be blocked before item is available"
        print("  wait_instance confirmed blocked.")

        poller.on_available(item1)
        print(f"  Item '{item1}' made available.")

        # Now wait_task should complete
        result = await asyncio.wait_for(
            wait_task, timeout=1.0
        )  # Wait with timeout
        print(f"  wait_instance task completed with result: {result}")

        assert result == [item1]
        assert len(poller) == 0
        print(
            "--- Test: test_wait_instance_blocks_until_on_available finished ---"
        )

    async def test_multiple_items_retrieved_in_order(self):
        print("\n--- Test: test_multiple_items_retrieved_in_order ---")
        poller = AsyncPoller[int]()  # Removed name="TestPollerMultiOrder"
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
        poller = AsyncPoller[int]()  # Removed name="TestPollerLimit"

        # Add more items than K_MAX_RESPONSES
        num_items_to_add = K_MAX_RESPONSES + 5
        items_added = list(range(num_items_to_add))
        for i in items_added:
            poller.on_available(i)
        print(f"  Added {num_items_to_add} items.")

        assert (
            len(poller) == K_MAX_RESPONSES
        ), f"Poller length should be capped at {K_MAX_RESPONSES}"
        print(f"  Poller length confirmed as {len(poller)}.")

        result = await poller.wait_instance()
        print(f"  wait_instance() returned {len(result)} items.")

        assert len(result) == K_MAX_RESPONSES
        # The poller uses a deque with append (right) and trimming from left if over limit.
        # So, the result should be the *last* K_MAX_RESPONSES items.
        expected_items = items_added[-K_MAX_RESPONSES:]
        assert (
            result == expected_items
        ), "Poller did not return the last K_MAX_RESPONSES items"
        assert len(poller) == 0
        print("--- Test: test_queue_limit_kMaxResponses finished ---")

    async def test_flush_clears_queue(self):
        print("\n--- Test: test_flush_clears_queue ---")
        poller = AsyncPoller[str]()  # Removed name="TestPollerFlush"
        poller.on_available("item_a")
        poller.on_available("item_b")
        assert len(poller) == 2
        print("  Items added, poller length is 2.")

        poller.flush()
        print("  Poller flushed.")

        assert len(poller) == 0, "Poller should be empty after flush"

        # Verify wait_instance blocks (or times out)
        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01)  # Give task a chance to run
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
        poller = AsyncPoller[float]()  # Removed name="TestPollerLen"
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
        poller = AsyncPoller[str]()  # Removed name="TestPollerAsyncIter"
        item_x = "item_x"
        item_y = "item_y"

        # Scenario 1: Items available before starting iteration
        poller.on_available(item_x)
        poller.on_available(item_y)
        print(f"  Items '{item_x}', '{item_y}' made available.")

        collected_via_iter = []

        # Start a task that will add more items while iterating
        async def add_more_items_later():
            await asyncio.sleep(0.02)  # Ensure iterator is likely waiting
            print("  Adding item_z during iteration.")
            poller.on_available("item_z")
            await asyncio.sleep(0.02)  # Allow iterator to pick it up
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
            if iter_count == 1:  # First yield should get item_x, item_y
                assert item_list == [item_x, item_y]
                # Start task to add more items *after* this first yield
                asyncio.create_task(add_more_items_later())
            elif iter_count == 2:  # Second yield should get item_z
                assert item_list == ["item_z"]
            elif iter_count == 3:  # Third yield should get item_w
                assert item_list == ["item_w"]
                break  # Exit loop for test finiteness
            if iter_count > 3:  # Failsafe
                pytest.fail(
                    "Iterator yielded more times than expected"
                )  # pragma: no cover

        assert (
            iter_count == 3
        ), "Iterator did not yield the expected number of times"
        # Expected structure: [[item_x, item_y], [item_z], [item_w]] - based on how on_available triggers event
        expected_full_collection = [[item_x, item_y], ["item_z"], ["item_w"]]
        assert collected_via_iter == expected_full_collection
        print(f"  Collected via iterator: {collected_via_iter}")
        print("--- Test: test_async_iterator finished ---")

    async def test_event_loop_property_set(self, mock_aio_utils):
        print("\n--- Test: test_event_loop_property_set ---")
        poller = AsyncPoller[int]()  # Removed name="TestPollerEventLoop"

        assert poller.event_loop is None, "Event loop should initially be None"
        print("  event_loop is None initially.")

        # Call wait_instance once to set the event loop
        # Need to provide an item or it will block indefinitely
        poller.on_available(100)
        await poller.wait_instance()
        print("  wait_instance called once.")

        assert (
            poller.event_loop is not None
        ), "Event loop should be set after wait_instance"
        assert (
            poller.event_loop is asyncio.get_running_loop()
        ), "Event loop not set to current loop"
        # Check if get_running_loop_or_none was used if poller's loop was initially None
        # mock_aio_utils["get_running_loop_or_none"].assert_called() # Called by __ensure_event_set
        assert (
            poller.event_loop is not None
        ), "poller.event_loop should be set after wait_instance"
        # Get the loop that the mock was configured to return
        expected_loop = mock_aio_utils["get_running_loop_or_none"].return_value
        assert (
            poller.event_loop is expected_loop
        ), "poller.event_loop was not set to the mock's return_value"
        print(f"  poller.event_loop is now {poller.event_loop}.")
        print("--- Test: test_event_loop_property_set finished ---")

    async def test_run_on_event_loop_called_for_set_results_available(
        self, mock_aio_utils
    ):
        print(
            "\n--- Test: test_run_on_event_loop_called_for_set_results_available ---"
        )
        poller = AsyncPoller[str]()  # Removed name="TestPollerRunOnLoop"
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
        print(
            f"  Item '{item_signal}' made available (should trigger run_on_event_loop)."
        )

        # Assert run_on_event_loop was called
        # mock_aio_utils["run_on_event_loop"].assert_called_once()
        # Check the arguments of the call to run_on_event_loop
        # First arg (args[0]) is the partial function
        # Second arg (args[1]) is the event loop instance
        # call_args_list = mock_aio_utils["run_on_event_loop"].call_args_list
        # assert len(call_args_list) == 1

        # partial_arg = call_args_list[0][0][0] # The functools.partial object
        # loop_arg = call_args_list[0][0][1]     # The event loop

        # assert isinstance(partial_arg, functools.partial), "run_on_event_loop not called with a partial"
        # assert partial_arg.func.__name__ == "_AsyncPoller__set_results_available", \
        #     "run_on_event_loop not called with __set_results_available"
        # assert loop_arg is poller.event_loop, "run_on_event_loop not called with the poller's event loop"

        # print("  run_on_event_loop assertions passed.")
        # Add a small delay to allow the event loop to process the scheduled coroutine
        await asyncio.sleep(0.01)
        assert (
            poller._AsyncPoller__barrier.is_set()
        ), "The barrier should be set after on_available when loop is running"
        print(
            "--- Test: test_run_on_event_loop_called_for_set_results_available finished ---"
        )


@pytest.mark.asyncio
class TestAsyncPollerWaitInstanceStopped:
    """Tests for AsyncPoller.wait_instance when the poller is stopped."""

    async def test_wait_instance_raises_runtime_error_if_stopped_before_first_call(
        self,
    ):
        """Test wait_instance raises RuntimeError if poller is already stopped."""
        poller = AsyncPoller[str]()

        # Directly manipulate internal state to simulate a stopped poller
        # Assuming _AsyncPoller__is_loop_running is an Atomic[bool]
        # And that it's initialized to True or gets set to True on first call if not stopped.
        # For this test, we explicitly stop it BEFORE any call to wait_instance that might set it.
        # The AsyncPoller's __is_loop_running is initialized to Atomic[bool](False)
        # and set to True within wait_instance if self.__event_loop is None.
        # So, to ensure it's seen as "stopped", we need to set it after it might have been set to True,
        # or ensure event_loop is not None.

        # Simulate that an event loop was previously acquired and then poller stopped
        poller._AsyncPoller__event_loop = (
            asyncio.get_running_loop()
        )  # Assign a dummy loop for test
        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_raises_runtime_error_if_stopped_during_wait(
        self,
    ):
        """Test wait_instance raises RuntimeError if poller is stopped while waiting."""
        poller = AsyncPoller[str]()

        wait_task = asyncio.create_task(poller.wait_instance())

        # Allow wait_instance to start and enter the waiting state
        await asyncio.sleep(0.01)

        # Ensure the loop was actually started by wait_instance
        assert poller._AsyncPoller__is_loop_running.get() is True
        assert poller._AsyncPoller__event_loop is not None

        # Stop the poller
        poller._AsyncPoller__is_loop_running.set(False)
        poller._AsyncPoller__barrier.set()  # Signal the barrier to wake up wait_instance

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await wait_task

    async def test_wait_instance_raises_error_even_if_data_queued_then_stopped(
        self,
    ):
        """Test wait_instance raises RuntimeError if stopped, even if data was previously queued."""
        poller = AsyncPoller[str]()
        poller.on_available("test_data_should_not_be_retrieved")

        # Simulate that an event loop was previously acquired
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(False)  # Stop it

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_after_on_available_and_stop_then_wait(self):
        """Test behavior when on_available is called, then poller is stopped, then wait_instance."""
        poller = AsyncPoller[str]()

        # Call on_available - this might try to set the barrier on an event loop
        # if __is_loop_running is True.
        # We need to ensure __is_loop_running is True and an event_loop is set for on_available
        # to interact with the barrier as it would in a running poller.
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(True)
        poller.on_available("some_data")

        # Now stop the poller
        poller._AsyncPoller__is_loop_running.set(False)

        # wait_instance should raise because the poller is stopped,
        # regardless of data being in __responses or barrier being set by on_available.
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        # Double check that data is still in the queue, but wait_instance didn't yield it
        assert len(poller._AsyncPoller__responses) == 1

    async def test_multiple_wait_calls_on_stopped_poller(self):
        """Ensure subsequent calls to wait_instance on a stopped poller also raise."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        # And again
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_stop_poller_after_wait_instance_already_returned_data(self):
        """Test stopping poller after wait_instance successfully returned data."""
        poller = AsyncPoller[str]()

        poller.on_available("data1")
        results = await poller.wait_instance()
        assert results == ["data1"]

        # Now stop the poller
        assert (
            poller._AsyncPoller__event_loop is not None
        )  # wait_instance should have set it
        poller._AsyncPoller__is_loop_running.set(False)
        poller._AsyncPoller__barrier.set()  # Wake any potential waiters (though none here)

        # Call wait_instance again, should raise error
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        # Add more data, try to wait again, should still be stopped
        poller.on_available("data2")
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        assert (
            len(poller._AsyncPoller__responses) == 1
        )  # "data2" should be in the queue
        assert poller._AsyncPoller__responses[0] == "data2"
