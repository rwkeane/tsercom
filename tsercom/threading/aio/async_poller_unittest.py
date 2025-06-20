"""Unit tests for the AsyncPoller class."""

import asyncio
import pytest
import pytest_asyncio

from tsercom.threading.aio.async_poller import AsyncPoller

# K_MAX_RESPONSES = 30 # This constant was removed from the test logic


from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    AsyncGenerator,
    List,
)  # Added List, AsyncGenerator
from unittest.mock import MagicMock


@pytest_asyncio.fixture
async def mock_aio_utils(
    mocker: MagicMock,
) -> AsyncGenerator[Dict[str, MagicMock], None]:
    """Mocks aio_utils used by AsyncPoller, patching them in the SUT's import scope."""

    def new_run_on_event_loop_side_effect(
        func_or_partial: Callable[[], Coroutine[Any, Any, Any]],
        loop_param: asyncio.AbstractEventLoop,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        coroutine = func_or_partial()
        if asyncio.iscoroutine(coroutine):
            asyncio.ensure_future(coroutine, loop=loop_param)
        else:
            coroutine()
        return None

    patched_run_on_event_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.run_on_event_loop",
        side_effect=new_run_on_event_loop_side_effect,
    )

    actual_loop = asyncio.get_running_loop()
    patched_get_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.get_running_loop_or_none",
        return_value=actual_loop,
    )
    patched_get_loop.name = "patched_get_running_loop_or_none"

    patched_is_on_loop = mocker.patch(
        "tsercom.threading.aio.aio_utils.is_running_on_event_loop",
        return_value=True,
    )

    yield {
        "run_on_event_loop": patched_run_on_event_loop,
        "get_running_loop_or_none": patched_get_loop,
        "is_running_on_event_loop": patched_is_on_loop,
    }


@pytest.mark.asyncio
class TestAsyncPoller:
    """Tests core functionality of the AsyncPoller."""

    @pytest.fixture(autouse=True)
    def _ensure_aio_utils_mocked(self, mock_aio_utils: Dict[str, MagicMock]) -> None:
        """Ensures mock_aio_utils fixture is activated for all tests in its class."""
        pass

    async def test_on_available_and_wait_instance_single_item(self) -> None:
        poller = AsyncPoller[str]()
        item1 = "item_one"

        poller.on_available(item1)
        result = await poller.wait_instance()

        assert result == [item1]
        assert len(poller) == 0

    async def test_wait_instance_blocks_until_on_available(self) -> None:
        poller = AsyncPoller[str]()
        item1 = "item_blocker"

        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        poller.on_available(item1)
        result = await asyncio.wait_for(wait_task, timeout=1.0)

        assert result == [item1]
        assert len(poller) == 0

    async def test_multiple_items_retrieved_in_order(self) -> None:
        poller = AsyncPoller[int](max_responses_queued=5)
        item1, item2, item3 = 1, 2, 3

        poller.on_available(item1)
        poller.on_available(item2)
        poller.on_available(item3)
        assert len(poller) == 3

        result = await poller.wait_instance()

        assert result == [item1, item2, item3]
        assert len(poller) == 0

    async def test_queue_limit_respected(self) -> None:
        poller = AsyncPoller[int](max_responses_queued=5)
        num_items_to_add = 5 + 3  # Test with 8 items for a limit of 5
        items_added = list(range(num_items_to_add))
        for i in items_added:
            poller.on_available(i)

        assert len(poller) == 5
        result = await poller.wait_instance()

        assert len(result) == 5
        expected_items = items_added[-5:]
        assert result == expected_items
        assert len(poller) == 0

    async def test_flush_clears_queue(self) -> None:
        poller = AsyncPoller[str]()
        poller.on_available("item_a")
        poller.on_available("item_b")
        assert len(poller) == 2

        poller.flush()
        assert len(poller) == 0

        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        wait_task.cancel()
        try:
            await wait_task
        except asyncio.CancelledError:
            pass

    async def test_len_accurate(self) -> None:
        poller = AsyncPoller[float]()
        assert len(poller) == 0

        poller.on_available(1.0)
        assert len(poller) == 1

        poller.on_available(2.0)
        assert len(poller) == 2

        await poller.wait_instance()
        assert len(poller) == 0

    async def test_async_iterator(self) -> None:
        poller = AsyncPoller[str]()
        item_x = "item_x"
        item_y = "item_y"

        poller.on_available(item_x)
        poller.on_available(item_y)

        collected_via_iter: List[List[str]] = []

        async def add_more_items_later() -> None:
            await asyncio.sleep(0.02)
            poller.on_available("item_z")
            await asyncio.sleep(0.02)
            poller.on_available("item_w")

        iter_count = 0
        async for (
            item_list
        ) in (
            poller
        ):  # The [misc] ignore might no longer be needed with newer mypy/iterator typing
            collected_via_iter.append(item_list)
            iter_count += 1
            if iter_count == 1:
                assert item_list == [item_x, item_y]
                asyncio.create_task(add_more_items_later())
            elif iter_count == 2:
                assert item_list == ["item_z"]
            elif iter_count == 3:
                assert item_list == ["item_w"]
                break
            if iter_count > 3:
                pytest.fail("Iterator yielded more times than expected")

        assert iter_count == 3
        expected_full_collection = [[item_x, item_y], ["item_z"], ["item_w"]]
        assert collected_via_iter == expected_full_collection

    async def test_event_loop_property_set(
        self, mock_aio_utils: Dict[str, MagicMock]
    ) -> None:
        poller = AsyncPoller[int](max_responses_queued=5)
        assert poller.event_loop is None

        poller.on_available(100)
        await poller.wait_instance()

        assert poller.event_loop is not None
        assert poller.event_loop is asyncio.get_running_loop()
        expected_loop = mock_aio_utils["get_running_loop_or_none"].return_value
        assert poller.event_loop is expected_loop

    async def test_run_on_event_loop_called_for_set_results_available(
        self, mock_aio_utils: Dict[str, MagicMock]
    ) -> None:
        poller = AsyncPoller[str]()
        item_signal = "signal_item"

        poller.on_available("initial_item_for_setup")
        await poller.wait_instance()

        mock_aio_utils["run_on_event_loop"].reset_mock()
        mock_aio_utils["is_running_on_event_loop"].return_value = False

        poller.on_available(item_signal)
        await asyncio.sleep(0.01)
        assert poller._AsyncPoller__barrier.is_set()  # type: ignore[attr-defined]

    async def test_queue_limit_none_is_unbounded(self) -> None:
        """Tests that max_responses_queued=None allows many items."""
        poller = AsyncPoller[int](max_responses_queued=None)
        num_items_to_add = 100  # A number larger than the old default
        items_added = list(range(num_items_to_add))
        for i in items_added:
            poller.on_available(i)

        assert len(poller) == num_items_to_add
        result = await poller.wait_instance()

        assert len(result) == num_items_to_add
        assert result == items_added
        assert len(poller) == 0


@pytest.mark.asyncio
class TestAsyncPollerWaitInstanceStopped:
    """Tests for AsyncPoller.wait_instance when the poller is stopped."""

    async def test_wait_instance_raises_runtime_error_if_stopped_before_first_call(
        self,
    ) -> None:
        """Test wait_instance raises RuntimeError if poller is already stopped."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
        # Simulate stop AFTER on_available but BEFORE first wait_instance
        poller._AsyncPoller__is_loop_running.set(False)  # type: ignore[attr-defined]
        # Ensure barrier is also set as stop() would do if loop was known
        poller._AsyncPoller__barrier.set()  # type: ignore[attr-defined] # stop() would set this if loop known

        # Poller is stopped and presumed empty, should raise RuntimeError immediately
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_raises_runtime_error_if_stopped_during_wait(
        self,
    ) -> None:
        """Test wait_instance raises RuntimeError if poller is stopped while waiting."""
        poller = AsyncPoller[str]()
        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01)

        assert poller._AsyncPoller__is_loop_running.get() is True  # type: ignore[attr-defined]
        assert poller._AsyncPoller__event_loop is not None  # type: ignore[attr-defined]

        poller.stop()
        await asyncio.sleep(0)  # Allow stop's scheduled tasks to run

        with pytest.raises(
            RuntimeError,
            match="AsyncPoller stopped while waiting for instance.",
        ):
            await wait_task

    async def test_wait_instance_raises_error_even_if_data_queued_then_stopped(
        self,
    ) -> None:
        """Test wait_instance raises RuntimeError if stopped, even if data was previously queued."""
        poller = AsyncPoller[str]()
        poller.on_available("test_data_should_not_be_retrieved")

        poller._AsyncPoller__event_loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
        poller._AsyncPoller__is_loop_running.set(False)  # type: ignore[attr-defined] # Poller is stopped
        # Ensure barrier is also set as stop() would do if loop was known
        poller._AsyncPoller__barrier.set()  # type: ignore[attr-defined]

        # First call should now return the data due to the drain-on-stop logic
        results = await poller.wait_instance()
        assert results == ["test_data_should_not_be_retrieved"]
        assert len(poller) == 0

        # Subsequent calls should raise RuntimeError
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_after_on_available_and_stop_then_wait(self) -> None:
        """Test behavior when on_available is called, then poller is stopped, then wait_instance."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
        poller._AsyncPoller__is_loop_running.set(True)  # type: ignore[attr-defined]
        poller.on_available("some_data")

        poller._AsyncPoller__is_loop_running.set(False)  # type: ignore[attr-defined]
        # Ensure barrier is also set as stop() would do
        poller._AsyncPoller__barrier.set()  # type: ignore[attr-defined]

        # First call should now return the data
        results = await poller.wait_instance()
        assert results == ["some_data"]
        assert len(poller) == 0  # Item should be consumed

        # Subsequent calls should raise RuntimeError
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_multiple_wait_calls_on_stopped_poller(self) -> None:
        """Ensure subsequent calls to wait_instance on a stopped poller also raise."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()  # type: ignore[attr-defined]
        poller._AsyncPoller__is_loop_running.set(False)  # type: ignore[attr-defined]

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_stop_poller_after_wait_instance_already_returned_data(self) -> None:
        """Test stopping poller after wait_instance successfully returned data."""
        poller = AsyncPoller[str]()

        poller.on_available("data1")
        results = await poller.wait_instance()
        assert results == ["data1"]

        assert poller._AsyncPoller__event_loop is not None  # type: ignore[attr-defined]
        poller._AsyncPoller__is_loop_running.set(False)  # type: ignore[attr-defined]
        poller._AsyncPoller__barrier.set()  # type: ignore[attr-defined] # Simulate stop() setting the barrier

        # This call should still raise RuntimeError as no new data was added before stop
        # and the previous data was consumed. Or, if the barrier being set by stop
        # could be misconstrued as new data if responses is empty, this might need thought.
        # The current drain logic: if stopped and responses empty, it raises.
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        poller.on_available("data2")  # Item added after poller is logically stopped
        # but before a wait_instance call that would consume it.

        # The next call to wait_instance should retrieve "data2" due to drain-on-stop logic
        results2 = await poller.wait_instance()
        assert results2 == ["data2"]
        assert len(poller) == 0

        # And now, subsequent calls should fail as it's stopped and empty.
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()
