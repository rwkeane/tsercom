import datetime
import torch
import pytest  # Using pytest conventions
from typing import List, Tuple  # For type hints

from tsercom.data.tensor_demuxer import TensorDemuxer  # Absolute import

# Helper type for captured calls by the mock client
CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        # Store a clone of the tensor to avoid issues if the demuxer modifies it later (though it shouldn't)
        self.calls.append((tensor.clone(), timestamp))
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(self) -> CapturedTensorChange | None:
        return self.calls[-1] if self.calls else None

    def get_all_calls_summary(
        self,
    ) -> List[Tuple[List[float], datetime.datetime]]:
        """Returns a summary of calls with tensor data as list of floats."""
        return [(t.tolist(), ts) for t, ts in self.calls]


@pytest.fixture
async def mock_client() -> (
    MockTensorDemuxerClient
):  # Keep async if tests are async
    return MockTensorDemuxerClient()


@pytest.fixture
async def demuxer(
    mock_client: MockTensorDemuxerClient,  # This is the coroutine for mock_client
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:  # Return a tuple
    actual_mock_client = await mock_client  # Await it here
    demuxer_instance = TensorDemuxer(
        client=actual_mock_client, tensor_length=4, data_timeout_seconds=60.0
    )
    return demuxer_instance, actual_mock_client  # Return both


@pytest.fixture
async def demuxer_short_timeout(
    mock_client: MockTensorDemuxerClient,  # This is the coroutine for mock_client
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:  # Return a tuple
    actual_mock_client = await mock_client  # Await it here
    demuxer_instance = TensorDemuxer(
        client=actual_mock_client, tensor_length=4, data_timeout_seconds=0.1
    )
    return demuxer_instance, actual_mock_client  # Return both


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)  # T0 + 10s
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)  # T1 + 10s
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)  # T2 + 10s


def test_constructor_validations():
    mock_cli = MockTensorDemuxerClient()  # Sync for this test
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorDemuxer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorDemuxer(client=mock_cli, tensor_length=1, data_timeout_seconds=0)


@pytest.mark.asyncio
async def test_first_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    d, mc = await demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1)

    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([5.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1


@pytest.mark.asyncio
async def test_sequential_updates_same_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    d, mc = await demuxer
    await d.on_update_received(
        tensor_index=1, value=10.0, timestamp=T1
    )  # Call 1
    await d.on_update_received(
        tensor_index=3, value=40.0, timestamp=T1
    )  # Call 2

    assert mc.call_count == 2
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([0.0, 10.0, 0.0, 40.0])
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1

    # Check intermediate call if necessary
    first_call_tensor, _ = mc.calls[0]
    expected_first_tensor = torch.tensor([0.0, 10.0, 0.0, 0.0])
    assert torch.equal(first_call_tensor, expected_first_tensor)


@pytest.mark.asyncio
async def test_updates_different_timestamps(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    d, mc = await demuxer
    await d.on_update_received(
        tensor_index=0, value=5.0, timestamp=T1
    )  # Call 1 (T1)
    mc.clear_calls()  # Focus on next call

    await d.on_update_received(
        tensor_index=0, value=15.0, timestamp=T2
    )  # Call 1 (T2)

    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor_t2, ts_t2 = last_call

    expected_tensor_t2 = torch.tensor([15.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor_t2, expected_tensor_t2)
    assert ts_t2 == T2

    # Ensure T1 state is still there internally (though not directly tested via client here)
    assert T1 in d._tensor_states
    assert torch.equal(
        d._tensor_states[T1], torch.tensor([5.0, 0.0, 0.0, 0.0])
    )


@pytest.mark.asyncio
async def test_out_of_order_scenario_from_prompt(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    d, mc = await demuxer
    # 1. Receive (index=1, value=10.0, timestamp=T2)
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T2)
    assert mc.call_count == 1
    call1_tensor, call1_ts = mc.calls[0]
    assert torch.equal(call1_tensor, torch.tensor([0.0, 10.0, 0.0, 0.0]))
    assert call1_ts == T2

    # 2. Receive (index=3, value=40.0, timestamp=T2)
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T2)
    assert mc.call_count == 2
    call2_tensor, call2_ts = mc.calls[1]
    assert torch.equal(call2_tensor, torch.tensor([0.0, 10.0, 0.0, 40.0]))
    assert call2_ts == T2

    # 3. Receive out-of-order (index=1, value=99.0, timestamp=T1) where T1 < T2
    await d.on_update_received(tensor_index=1, value=99.0, timestamp=T1)
    assert mc.call_count == 3
    call3_tensor, call3_ts = mc.calls[2]
    assert torch.equal(call3_tensor, torch.tensor([0.0, 99.0, 0.0, 0.0]))
    assert call3_ts == T1
    # Check T2 state is unaffected
    assert torch.equal(
        d._tensor_states[T2], torch.tensor([0.0, 10.0, 0.0, 40.0])
    )

    # 4. Receive another update (index=2, value=88.0, timestamp=T1)
    await d.on_update_received(tensor_index=2, value=88.0, timestamp=T1)
    assert mc.call_count == 4
    call4_tensor, call4_ts = mc.calls[3]
    assert torch.equal(call4_tensor, torch.tensor([0.0, 99.0, 88.0, 0.0]))
    assert call4_ts == T1

    # Verify all calls if needed for exact sequence and content
    all_calls = mc.get_all_calls_summary()
    expected_all_calls = [
        ([0.0, 10.0, 0.0, 0.0], T2),
        ([0.0, 10.0, 0.0, 40.0], T2),
        ([0.0, 99.0, 0.0, 0.0], T1),
        ([0.0, 99.0, 88.0, 0.0], T1),
    ]
    assert all_calls == expected_all_calls


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    dmx_instance, mc = await demuxer_short_timeout  # alias

    # Update at T0
    await dmx_instance.on_update_received(
        tensor_index=0, value=1.0, timestamp=T0
    )  # T0 = 12:00:00
    assert mc.call_count == 1
    assert T0 in dmx_instance._tensor_states
    mc.clear_calls()

    # Update at T2 (T0 is 20s ago, timeout is 0.1s)
    # When T2 update arrives, latest_update_timestamp becomes T2.
    # Cleanup threshold becomes T2 - 0.1s. T0 is much older.
    await dmx_instance.on_update_received(
        tensor_index=0, value=2.0, timestamp=T2
    )  # T2 = 12:00:20

    assert T0 not in dmx_instance._tensor_states  # T0 should be timed out
    assert T2 in dmx_instance._tensor_states
    assert mc.call_count == 1  # Only for T2 update
    tensor_t2, ts_t2 = mc.get_last_call()
    assert torch.equal(tensor_t2, torch.tensor([2.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2

    # Update for T1 (T1 = 12:00:10). Current latest is T2 (12:00:20). Timeout 0.1s.
    # Cutoff is T2 - 0.1s = 12:00:19.9. T1 (12:00:10) is older than this cutoff.
    # So this update for T1 should be ignored.
    await dmx_instance.on_update_received(tensor_index=0, value=3.0, timestamp=T1)
    assert mc.call_count == 1  # No new call
    assert T1 not in dmx_instance._tensor_states  # T1 should not be created


@pytest.mark.asyncio
async def test_index_out_of_bounds(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    d, mc = await demuxer
    # Tensor length is 4, so valid indices are 0, 1, 2, 3
    await d.on_update_received(tensor_index=4, value=1.0, timestamp=T1)
    assert mc.call_count == 0  # No call for out-of-bounds

    await d.on_update_received(tensor_index=-1, value=1.0, timestamp=T1)
    assert mc.call_count == 0  # No call for out-of-bounds

    # Ensure no state was created for T1 due to bad indices
    assert T1 not in d._tensor_states


@pytest.mark.asyncio
async def test_update_no_value_change(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    d, mc = await demuxer
    await d.on_update_received(
        tensor_index=0, value=5.0, timestamp=T1
    )  # Initial call
    assert mc.call_count == 1

    # Send the same value again
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1)
    assert (
        mc.call_count == 1
    )  # No new call because value didn't change

    # Send a different value to confirm it then calls
    await d.on_update_received(tensor_index=0, value=6.0, timestamp=T1)
    assert mc.call_count == 2

    last_call_tensor, _ = mc.get_last_call()
    assert torch.equal(last_call_tensor, torch.tensor([6.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_timeout_behavior_cleanup_order(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient]
):
    dmx_instance, mc = await demuxer_short_timeout  # alias for short lines, timeout = 0.1s

    # T1 arrives: 12:00:10
    await dmx_instance.on_update_received(0, 1.0, T1)
    assert T1 in dmx_instance._tensor_states
    assert dmx_instance._latest_update_timestamp == T1
    mc.clear_calls()

    # T0 arrives (out of order): 12:00:00
    # latest_update_timestamp is still T1. Cutoff for cleanup is T1 - 0.1s = 12:00:09.9
    # T0 (12:00:00) is older than this cutoff.
    # So, when T0 arrives, _cleanup_old_data() is called. _latest_update_timestamp is T1.
    # The update for T0 itself should be checked against this T1-based cutoff.
    # This means T0 data should be ignored.
    await dmx_instance.on_update_received(0, 2.0, T0)
    assert (
        T0 not in dmx_instance._tensor_states
    )  # T0 should be considered stale on arrival
    assert mc.call_count == 0  # No client call for T0

    # T2 arrives: 12:00:20
    # latest_update_timestamp becomes T2. Cutoff T2 - 0.1s = 12:00:19.9
    # T1 (12:00:10) is older. So T1 state is cleaned.
    await dmx_instance.on_update_received(0, 3.0, T2)
    assert T1 not in dmx_instance._tensor_states  # T1 should now be gone
    assert T2 in dmx_instance._tensor_states
    assert mc.call_count == 1  # For T2 update
    tensor_t2, ts_t2 = mc.get_last_call()
    assert torch.equal(tensor_t2, torch.tensor([3.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2
