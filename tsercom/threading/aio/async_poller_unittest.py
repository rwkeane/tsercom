"""Unit tests for the AsyncPoller class."""

import asyncio
import pytest
import pytest_asyncio
import functools
from collections import deque

from tsercom.threading.atomic import Atomic
from tsercom.threading.aio.async_poller import AsyncPoller
import tsercom.threading.aio.aio_utils as aio_utils_to_patch

K_MAX_RESPONSES = 30


@pytest_asyncio.fixture
async def mock_aio_utils(mocker):
    """Mocks aio_utils used by AsyncPoller, patching them in the SUT's import scope."""

    def new_run_on_event_loop_side_effect(
        func_or_partial, loop_param, *args, **kwargs
    ):
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
    def _ensure_aio_utils_mocked(self, mock_aio_utils):
        """Ensures mock_aio_utils fixture is activated for all tests in its class."""
        pass

    async def test_on_available_and_wait_instance_single_item(self):
        poller = AsyncPoller[str]()
        item1 = "item_one"

        poller.on_available(item1)
        result = await poller.wait_instance()

        assert result == [item1]
        assert len(poller) == 0

    async def test_wait_instance_blocks_until_on_available(self):
        poller = AsyncPoller[str]()
        item1 = "item_blocker"

        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01)
        assert not wait_task.done()

        poller.on_available(item1)
        result = await asyncio.wait_for(wait_task, timeout=1.0)

        assert result == [item1]
        assert len(poller) == 0

    async def test_multiple_items_retrieved_in_order(self):
        poller = AsyncPoller[int]()
        item1, item2, item3 = 1, 2, 3

        poller.on_available(item1)
        poller.on_available(item2)
        poller.on_available(item3)
        assert len(poller) == 3

        result = await poller.wait_instance()

        assert result == [item1, item2, item3]
        assert len(poller) == 0

    async def test_queue_limit_kMaxResponses(self):
        poller = AsyncPoller[int]()
        num_items_to_add = K_MAX_RESPONSES + 5
        items_added = list(range(num_items_to_add))
        for i in items_added:
            poller.on_available(i)

        assert len(poller) == K_MAX_RESPONSES
        result = await poller.wait_instance()

        assert len(result) == K_MAX_RESPONSES
        expected_items = items_added[-K_MAX_RESPONSES:]
        assert result == expected_items
        assert len(poller) == 0

    async def test_flush_clears_queue(self):
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

    async def test_len_accurate(self):
        poller = AsyncPoller[float]()
        assert len(poller) == 0

        poller.on_available(1.0)
        assert len(poller) == 1

        poller.on_available(2.0)
        assert len(poller) == 2

        await poller.wait_instance()
        assert len(poller) == 0

    async def test_async_iterator(self):
        poller = AsyncPoller[str]()
        item_x = "item_x"
        item_y = "item_y"

        poller.on_available(item_x)
        poller.on_available(item_y)

        collected_via_iter = []

        async def add_more_items_later():
            await asyncio.sleep(0.02)
            poller.on_available("item_z")
            await asyncio.sleep(0.02)
            poller.on_available("item_w")

        iter_count = 0
        async for item_list in poller:
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

    async def test_event_loop_property_set(self, mock_aio_utils):
        poller = AsyncPoller[int]()
        assert poller.event_loop is None

        poller.on_available(100)
        await poller.wait_instance()

        assert poller.event_loop is not None
        assert poller.event_loop is asyncio.get_running_loop()
        expected_loop = mock_aio_utils["get_running_loop_or_none"].return_value
        assert poller.event_loop is expected_loop

    async def test_run_on_event_loop_called_for_set_results_available(
        self, mock_aio_utils
    ):
        poller = AsyncPoller[str]()
        item_signal = "signal_item"

        poller.on_available("initial_item_for_setup")
        await poller.wait_instance()

        mock_aio_utils["run_on_event_loop"].reset_mock()
        mock_aio_utils["is_running_on_event_loop"].return_value = False

        poller.on_available(item_signal)
        await asyncio.sleep(0.01)
        assert poller._AsyncPoller__barrier.is_set()


@pytest.mark.asyncio
class TestAsyncPollerWaitInstanceStopped:
    """Tests for AsyncPoller.wait_instance when the poller is stopped."""

    async def test_wait_instance_raises_runtime_error_if_stopped_before_first_call(
        self,
    ):
        """Test wait_instance raises RuntimeError if poller is already stopped."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_raises_runtime_error_if_stopped_during_wait(
        self,
    ):
        """Test wait_instance raises RuntimeError if poller is stopped while waiting."""
        poller = AsyncPoller[str]()
        wait_task = asyncio.create_task(poller.wait_instance())
        await asyncio.sleep(0.01)

        assert poller._AsyncPoller__is_loop_running.get() is True
        assert poller._AsyncPoller__event_loop is not None

        poller._AsyncPoller__is_loop_running.set(False)
        poller._AsyncPoller__barrier.set()

        with pytest.raises(
            RuntimeError,
            match="AsyncPoller stopped while waiting for instance.",
        ):
            await wait_task

    async def test_wait_instance_raises_error_even_if_data_queued_then_stopped(
        self,
    ):
        """Test wait_instance raises RuntimeError if stopped, even if data was previously queued."""
        poller = AsyncPoller[str]()
        poller.on_available("test_data_should_not_be_retrieved")

        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_wait_instance_after_on_available_and_stop_then_wait(self):
        """Test behavior when on_available is called, then poller is stopped, then wait_instance."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(True)
        poller.on_available("some_data")

        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        assert len(poller._AsyncPoller__responses) == 1

    async def test_multiple_wait_calls_on_stopped_poller(self):
        """Ensure subsequent calls to wait_instance on a stopped poller also raise."""
        poller = AsyncPoller[str]()
        poller._AsyncPoller__event_loop = asyncio.get_running_loop()
        poller._AsyncPoller__is_loop_running.set(False)

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

    async def test_stop_poller_after_wait_instance_already_returned_data(self):
        """Test stopping poller after wait_instance successfully returned data."""
        poller = AsyncPoller[str]()

        poller.on_available("data1")
        results = await poller.wait_instance()
        assert results == ["data1"]

        assert poller._AsyncPoller__event_loop is not None
        poller._AsyncPoller__is_loop_running.set(False)
        poller._AsyncPoller__barrier.set()

        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        poller.on_available("data2")
        with pytest.raises(RuntimeError, match="AsyncPoller is stopped"):
            await poller.wait_instance()

        assert len(poller._AsyncPoller__responses) == 1
        assert poller._AsyncPoller__responses[0] == "data2"
