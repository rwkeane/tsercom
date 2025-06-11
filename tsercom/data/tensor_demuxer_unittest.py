import datetime
import torch
import pytest
from typing import List, Tuple, Sequence # Added Sequence

from tsercom.data.tensor_demuxer import (
    TensorDemuxer,
    TensorIndex # Import TensorIndex
)

# Helper type for captured calls by the mock client
CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor.clone(), timestamp)) # Store a clone
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(self) -> CapturedTensorChange | None:
        return self.calls[-1] if self.calls else None

    def get_latest_tensor_for_ts(self, timestamp: datetime.datetime) -> torch.Tensor | None:
        for i in range(len(self.calls) - 1, -1, -1):
            tensor, ts = self.calls[i]
            if ts == timestamp:
                return tensor
        return None

    def get_all_calls_summary_nd(
        self,
    ) -> List[Tuple[List, datetime.datetime]]: # For N-D tensors, tolist() will be nested
        """Returns a summary of calls with N-D tensor data as nested list of floats."""
        return [(t.tolist(), ts) for t, ts in self.calls]


@pytest.fixture
def default_tensor_shape() -> Sequence[int]:
    return (2, 2) # Using a simple 2x2 N-D shape for tests

@pytest.fixture
def mock_client() -> MockTensorDemuxerClient: # No longer async fixture
    return MockTensorDemuxerClient()

@pytest.fixture
def demuxer(
    mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]
) -> TensorDemuxer: # Return only demuxer
    return TensorDemuxer(
        client=mock_client, tensor_shape=default_tensor_shape, data_timeout_seconds=60.0
    )

@pytest.fixture
def demuxer_1d(mock_client: MockTensorDemuxerClient) -> TensorDemuxer:
    return TensorDemuxer(
        client=mock_client, tensor_shape=(4,), data_timeout_seconds=60.0
    )


@pytest.fixture
def demuxer_short_timeout(
    mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]
) -> TensorDemuxer:
    return TensorDemuxer(
        client=mock_client, tensor_shape=default_tensor_shape, data_timeout_seconds=0.1
    )


# Timestamps
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)

def _create_tensor_nd(shape: Sequence[int], start_val: float = 0.0, increment: float = 1.0) -> torch.Tensor:
    num_elements = torch.prod(torch.tensor(shape)).item()
    return (torch.arange(num_elements) * increment + start_val).reshape(shape)

def _get_all_indices_nd(shape: Sequence[int]) -> List[TensorIndex]:
    import itertools
    if not shape: return []
    return list(itertools.product(*[range(s) for s in shape]))


@pytest.mark.asyncio
async def test_constructor_validations_nd(mock_client: MockTensorDemuxerClient):
    with pytest.raises(ValueError, match="Tensor shape must be a non-empty sequence of positive integers."):
        TensorDemuxer(client=mock_client, tensor_shape=())
    with pytest.raises(ValueError, match="Tensor shape must be a non-empty sequence of positive integers."):
        TensorDemuxer(client=mock_client, tensor_shape=(0, 2))
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorDemuxer(client=mock_client, tensor_shape=(1,1), data_timeout_seconds=-1)


@pytest.mark.asyncio
async def test_first_update_nd(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    update_idx = (0, 0)
    update_val = 5.0
    await demuxer.on_update_received(tensor_index=update_idx, value=update_val, timestamp=T1)

    assert mock_client.call_count == 1
    last_call = mock_client.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.zeros(default_tensor_shape)
    expected_tensor[update_idx] = update_val
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1


@pytest.mark.asyncio
async def test_sequential_updates_same_timestamp_nd(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    idx1, val1 = (0, 1), 10.0
    idx2, val2 = (1, 0), 40.0

    await demuxer.on_update_received(tensor_index=idx1, value=val1, timestamp=T1)
    await demuxer.on_update_received(tensor_index=idx2, value=val2, timestamp=T1)

    assert mock_client.call_count == 2
    last_call = mock_client.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.zeros(default_tensor_shape)
    expected_tensor[idx1] = val1
    expected_tensor[idx2] = val2
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1

    first_call_tensor, _ = mock_client.calls[0]
    expected_first_tensor = torch.zeros(default_tensor_shape)
    expected_first_tensor[idx1] = val1
    assert torch.equal(first_call_tensor, expected_first_tensor)


@pytest.mark.asyncio
async def test_updates_different_timestamps_nd(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    idx_t1, val_t1 = (0,0), 5.0
    await demuxer.on_update_received(tensor_index=idx_t1, value=val_t1, timestamp=T1)
    mock_client.clear_calls()

    idx_t2, val_t2 = (0,0), 15.0
    await demuxer.on_update_received(tensor_index=idx_t2, value=val_t2, timestamp=T2)

    assert mock_client.call_count == 1
    tensor_t2_client, ts_t2_client = mock_client.get_last_call()

    expected_tensor_t1_state = torch.zeros(default_tensor_shape)
    expected_tensor_t1_state[idx_t1] = val_t1

    expected_tensor_t2_state = expected_tensor_t1_state.clone()
    expected_tensor_t2_state[idx_t2] = val_t2

    assert torch.equal(tensor_t2_client, expected_tensor_t2_state)
    assert ts_t2_client == T2

    internal_t1_state = demuxer._tensor_states[0][1]
    assert torch.equal(internal_t1_state, expected_tensor_t1_state)
    internal_t2_state = demuxer._tensor_states[1][1]
    assert torch.equal(internal_t2_state, expected_tensor_t2_state)


@pytest.mark.asyncio
async def test_out_of_order_nd_cascade(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    idx_t2_1, val_t2_1 = (0,1), 10.0
    await demuxer.on_update_received(idx_t2_1, val_t2_1, T2)

    idx_t2_2, val_t2_2 = (1,1), 40.0
    await demuxer.on_update_received(idx_t2_2, val_t2_2, T2)
    t2_state_after_direct_updates = torch.zeros(default_tensor_shape)
    t2_state_after_direct_updates[idx_t2_1] = val_t2_1
    t2_state_after_direct_updates[idx_t2_2] = val_t2_2

    mock_client.clear_calls()

    idx_t1_1, val_t1_1 = (0,1), 99.0
    await demuxer.on_update_received(idx_t1_1, val_t1_1, T1)

    t1_expected_state = torch.zeros(default_tensor_shape)
    t1_expected_state[idx_t1_1] = val_t1_1

    t2_expected_state_after_cascade = t1_expected_state.clone()
    t2_expected_state_after_cascade[idx_t2_1] = val_t2_1
    t2_expected_state_after_cascade[idx_t2_2] = val_t2_2

    assert mock_client.call_count == 2

    call_t1 = mock_client.calls[0]
    assert torch.equal(call_t1[0], t1_expected_state)
    assert call_t1[1] == T1

    call_t2_cascade = mock_client.calls[1]
    assert torch.equal(call_t2_cascade[0], t2_expected_state_after_cascade)
    assert call_t2_cascade[1] == T2

    assert torch.equal(demuxer._tensor_states[0][1], t1_expected_state)
    assert torch.equal(demuxer._tensor_states[1][1], t2_expected_state_after_cascade)


@pytest.mark.asyncio
async def test_data_timeout_nd(demuxer_short_timeout: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    dmx = demuxer_short_timeout
    idx_t0, val_t0 = (0,0), 1.0
    await dmx.on_update_received(idx_t0, val_t0, T0)
    assert len(dmx._tensor_states) == 1
    mock_client.clear_calls()

    idx_t2, val_t2 = (0,0), 2.0
    await dmx.on_update_received(idx_t2, val_t2, T2)

    assert len(dmx._tensor_states) == 1
    assert dmx._tensor_states[0][0] == T2

    assert mock_client.call_count == 1
    tensor_t2_client, ts_t2_client = mock_client.get_last_call()
    expected_t2_tensor = torch.zeros(default_tensor_shape)
    expected_t2_tensor[idx_t2] = val_t2
    assert torch.equal(tensor_t2_client, expected_t2_tensor)
    assert ts_t2_client == T2

    idx_t1, val_t1 = (0,0), 3.0
    await dmx.on_update_received(idx_t1, val_t1, T1)
    assert len(dmx._tensor_states) == 1
    assert dmx._tensor_states[0][0] == T2
    assert mock_client.call_count == 1


@pytest.mark.asyncio
async def test_index_out_of_bounds_nd(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    bad_indices = [(2,0), (0,2), (-1,0), (0,-1), (0,0,0), (0,)] # Assuming default (2,2)
    if default_tensor_shape == (2,2): # Adjust if shape changes
      pass # bad_indices is fine
    elif default_tensor_shape == (2,3):
        bad_indices = [(2,0), (0,3), (-1,0), (0,-1), (0,0,0), (0,)]
    # Add more specific bad indices based on default_tensor_shape if needed for robustness

    for bad_idx in bad_indices:
        await demuxer.on_update_received(tensor_index=bad_idx, value=1.0, timestamp=T1)
        assert mock_client.call_count == 0
    assert len(demuxer._tensor_states) == 0


@pytest.mark.asyncio
async def test_update_no_value_change_nd(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    idx, val = (0,0), 5.0
    await demuxer.on_update_received(idx, val, T1)
    assert mock_client.call_count == 1

    await demuxer.on_update_received(idx, val, T1)
    assert mock_client.call_count == 1

    await demuxer.on_update_received(idx, val + 1.0, T1)
    assert mock_client.call_count == 2

    last_call_tensor, _ = mock_client.get_last_call()
    expected_tensor = torch.zeros(default_tensor_shape)
    expected_tensor[idx] = val + 1.0
    assert torch.equal(last_call_tensor, expected_tensor)


@pytest.mark.asyncio
async def test_get_tensor_at_nd(demuxer: TensorDemuxer, mock_client: MockTensorDemuxerClient, default_tensor_shape: Sequence[int]):
    idx_t1, val_t1 = (0,0), 1.0
    idx_t2_1, val_t2_1 = (0,1), 2.0
    idx_t2_2, val_t2_2 = (1,0), 3.0

    await demuxer.on_update_received(idx_t1, val_t1, T1)
    expected_t1_state = torch.zeros(default_tensor_shape)
    expected_t1_state[idx_t1] = val_t1

    await demuxer.on_update_received(idx_t2_1, val_t2_1, T2)
    await demuxer.on_update_received(idx_t2_2, val_t2_2, T2)
    expected_t2_state = expected_t1_state.clone()
    expected_t2_state[idx_t2_1] = val_t2_1
    expected_t2_state[idx_t2_2] = val_t2_2

    mock_client.clear_calls()

    retrieved_t1 = demuxer.get_tensor_at(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, expected_t1_state)
    if default_tensor_shape != (0,) and torch.prod(torch.tensor(default_tensor_shape)).item() > 0 : # Avoid id check for empty tensor
      assert id(retrieved_t1) != id(demuxer._tensor_states[0][1])

    retrieved_t2 = demuxer.get_tensor_at(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, expected_t2_state)

    retrieved_t0 = demuxer.get_tensor_at(T0)
    assert retrieved_t0 is None

    assert mock_client.call_count == 0


@pytest.mark.asyncio
async def test_1d_tensor_demuxing(demuxer_1d: TensorDemuxer, mock_client: MockTensorDemuxerClient):
    shape_1d = (4,)

    await demuxer_1d.on_update_received(tensor_index=(0,), value=10.0, timestamp=T1)
    await demuxer_1d.on_update_received(tensor_index=(2,), value=30.0, timestamp=T1)

    assert mock_client.call_count == 2
    tensor_t1_client, _ = mock_client.get_last_call()
    expected_t1 = torch.tensor([10.0, 0.0, 30.0, 0.0])
    assert torch.equal(tensor_t1_client, expected_t1)

    retrieved_t1 = demuxer_1d.get_tensor_at(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, expected_t1)
    mock_client.clear_calls()

    await demuxer_1d.on_update_received(tensor_index=(1,), value=5.0, timestamp=T0)

    assert mock_client.call_count == 2

    call_t0_data = mock_client.calls[0]
    assert torch.equal(call_t0_data[0], torch.tensor([0.0, 5.0, 0.0, 0.0]))
    assert call_t0_data[1] == T0

    call_t1_cascade_data = mock_client.calls[1]
    assert torch.equal(call_t1_cascade_data[0], torch.tensor([10.0, 5.0, 30.0, 0.0]))
    assert call_t1_cascade_data[1] == T1

