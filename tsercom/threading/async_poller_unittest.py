"""Tests for AsyncPoller."""

import asyncio
import pytest
import pytest_asyncio # For async fixtures if needed, though direct async tests are fine

from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.atomic import Atomic # For manipulating internal state in tests


@pytest.mark.asyncio
class TestAsyncPollerWaitInstanceStopped:
    """Tests for AsyncPoller.wait_instance when the poller is stopped."""

    async def test_wait_instance_raises_runtime_error_if_stopped_before_first_call(self):
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
        poller._AsyncPoller__event_loop = asyncio.get_running_loop() # Assign a dummy loop for test
        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_raises_runtime_error_if_stopped_during_wait(self):
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
        poller._AsyncPoller__barrier.set() # Signal the barrier to wake up wait_instance

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await wait_task
            
    async def test_wait_instance_raises_error_even_if_data_queued_then_stopped(self):
        """Test wait_instance raises RuntimeError if stopped, even if data was previously queued."""
        poller = AsyncPoller[str]()
        poller.on_available("test_data_should_not_be_retrieved")

        # Simulate that an event loop was previously acquired
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(False) # Stop it

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
        assert poller._AsyncPoller__event_loop is not None # wait_instance should have set it
        poller._AsyncPoller__is_loop_running.set(False)
        poller._AsyncPoller__barrier.set() # Wake any potential waiters (though none here)

        # Call wait_instance again, should raise error
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        # Add more data, try to wait again, should still be stopped
        poller.on_available("data2")
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()
        
        assert len(poller._AsyncPoller__responses) == 1 # "data2" should be in the queue
        assert poller._AsyncPoller__responses[0] == "data2"
