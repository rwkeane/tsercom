"""Unit tests for the SmoothedTensorDemuxer."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pytest
import torch

from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy
from tsercom.data.tensor.linear_interpolation_strategy import LinearInterpolationStrategy
from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # For TensorDemuxer.Client


# Helper to get current UTC time
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# A concrete, minimal Client for testing, conforming to TensorDemuxer.Client
class MockDemuxerClient(TensorDemuxer.Client):
    def __init__(self) -> None:
        self.received_tensors: List[Tuple[torch.Tensor, datetime]] = []
        self.call_log: List[Tuple[str, torch.Tensor, datetime]] = []

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime
    ) -> None:
        # Method name from TensorDemuxer.Client
        self.received_tensors.append((tensor.clone(), timestamp))
        self.call_log.append(("on_tensor_changed", tensor.clone(), timestamp))

    def clear_received_tensors(self) -> None:
        self.received_tensors = []
        self.call_log = []


@pytest.fixture
def mock_client() -> MockDemuxerClient:  # Use the new MockDemuxerClient
    return MockDemuxerClient()


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest.fixture
def tensor_shape_2d() -> Tuple[int, int]:
    return (2, 2)  # Example 2D shape


@pytest.fixture
def tensor_shape_1d() -> Tuple[int]:
    return (4,)  # Example 1D shape for cascading test


@pytest.mark.asyncio
async def test_demuxer_initialization(
    linear_strategy: LinearInterpolationStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_1d: Tuple[int],
) -> None:
    """Test basic initialization of SmoothedTensorDemuxer."""
    demuxer = SmoothedTensorDemuxer(
        smoothing_strategy=linear_strategy,
        client=mock_client,
        output_interval=0.1,
        tensor_shape=tensor_shape_1d,
    )
    assert demuxer.smoothing_strategy == linear_strategy
    assert demuxer._client == mock_client  # _client is from base
    assert demuxer.output_interval == 0.1
    assert demuxer._SmoothedTensorDemuxer__tensor_shape == tensor_shape_1d  # type: ignore[attr-defined]
    assert demuxer._lock is not None  # _lock inherited from TensorDemuxer

    # Test context manager behavior
    async with demuxer:
        assert demuxer._SmoothedTensorDemuxer__worker_task is not None  # type: ignore[attr-defined]
        assert not demuxer._SmoothedTensorDemuxer__worker_task.done()  # type: ignore[attr-defined]
    assert (
        demuxer._SmoothedTensorDemuxer__worker_task is None  # type: ignore[attr-defined]
        or demuxer._SmoothedTensorDemuxer__worker_task.done()  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_on_update_received_nd_indices(
    linear_strategy: SmoothingStrategy, # Test against ABC
    mock_client: MockDemuxerClient,
    tensor_shape_2d: Tuple[int, int],
) -> None:
    """Test on_update_received with N-D indices, sorting, and buffering."""
    demuxer = SmoothedTensorDemuxer(
        linear_strategy, # Pass the concrete instance for the test
        mock_client,
        0.1,
        tensor_shape=tensor_shape_2d,
        buffer_size=2,
    )

    ts1 = _utcnow()
    ts2 = ts1 + timedelta(seconds=1)
    ts3 = ts1 + timedelta(seconds=0.5)  # Out of order

    idx0_0 = (0, 0)
    await demuxer.on_update_received(idx0_0, 10.0, ts1)
    await demuxer.on_update_received(idx0_0, 20.0, ts2)
    await demuxer.on_update_received(idx0_0, 15.0, ts3)  # Inserted in between

    keyframes_idx0_0 = demuxer._SmoothedTensorDemuxer__per_index_keyframes[  # type: ignore[attr-defined]
        idx0_0
    ]
    assert len(keyframes_idx0_0) == 2  # Buffer size is 2
    assert keyframes_idx0_0[0] == (ts3, 15.0)  # Sorted: ts3 (0.5s), ts2 (1s)
    assert keyframes_idx0_0[1] == (ts2, 20.0)

    idx1_1 = (1, 1)
    await demuxer.on_update_received(idx1_1, 1.0, ts1)
    await demuxer.on_update_received(idx1_1, 2.0, ts2)
    keyframes_idx1_1 = demuxer._SmoothedTensorDemuxer__per_index_keyframes[  # type: ignore[attr-defined]
        idx1_1
    ]
    assert len(keyframes_idx1_1) == 2
    assert keyframes_idx1_1[0] == (ts1,1.0,)
    assert keyframes_idx1_1[1] == (ts2, 2.0)

    await demuxer.on_update_received((0,0,0), 100.0, ts1)
    assert (0,0,0) not in demuxer._SmoothedTensorDemuxer__per_index_keyframes  # type: ignore[attr-defined]

    await demuxer.on_update_received((2,0), 100.0, ts1)
    assert (2,0) not in demuxer._SmoothedTensorDemuxer__per_index_keyframes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_interpolation_worker_produces_nd_output(
    linear_strategy: LinearInterpolationStrategy, # Use concrete for fixture
    mock_client: MockDemuxerClient,
    tensor_shape_2d: Tuple[int, int],
) -> None:
    """Test worker produces N-D tensors at intervals."""
    output_interval = 0.05
    demuxer = SmoothedTensorDemuxer(
        linear_strategy, mock_client, output_interval, tensor_shape=tensor_shape_2d,
    )

    ts1 = _utcnow()
    await demuxer.on_update_received((0,0), 10.0, ts1)
    await demuxer.on_update_received((0,1), 11.0, ts1)
    await demuxer.on_update_received((1,0), 12.0, ts1)
    await demuxer.on_update_received((1,1), 13.0, ts1)
    await demuxer.on_update_received((0,0), 20.0, ts1 + timedelta(seconds=0.2))

    async with demuxer:
        await asyncio.sleep(output_interval * 3.5)

    assert len(mock_client.received_tensors) >= 2
    for tensor, ts in mock_client.received_tensors:
        assert tensor.shape == tensor_shape_2d
        assert isinstance(tensor, torch.Tensor)

    first_tensor_val_0_0 = mock_client.received_tensors[0][0][0,0].item()
    last_tensor_val_0_0 = mock_client.received_tensors[-1][0][0,0].item()
    assert first_tensor_val_0_0 >= 10.0
    if len(mock_client.received_tensors) > 1:
        assert last_tensor_val_0_0 > first_tensor_val_0_0 or last_tensor_val_0_0 == 20.0

    if mock_client.received_tensors:
        reference_tensor = mock_client.received_tensors[0][0]
        assert reference_tensor[0,1].item() == 11.0
        assert reference_tensor[1,0].item() == 12.0
        assert reference_tensor[1,1].item() == 13.0


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_cascading_interpolation_scenario_1d(
    linear_strategy: LinearInterpolationStrategy, # Use concrete for fixture
    mock_client: MockDemuxerClient,
    tensor_shape_1d: Tuple[int],
) -> None:
    """Tests CRITICAL per-index cascading interpolation."""
    output_interval = 0.01
    demuxer = SmoothedTensorDemuxer(
        linear_strategy, mock_client, output_interval, tensor_shape=tensor_shape_1d,
    )
    time_A = _utcnow(); time_B = time_A + timedelta(seconds=0.2); time_C = time_A + timedelta(seconds=0.4); time_D = time_A + timedelta(seconds=0.6)
    val_A0,val_A1,val_A2,val_A3=10.,11.,12.,13.; val_D0,val_D1,val_D2,val_D3=50.,51.,52.,53.
    val_B0,val_B1=20.,21.; val_C2,val_C3=32.,33.
    for i,v in enumerate([val_A0,val_A1,val_A2,val_A3]): await demuxer.on_update_received((i,), v, time_A)
    for i,v in enumerate([val_D0,val_D1,val_D2,val_D3]): await demuxer.on_update_received((i,), v, time_D)
    await demuxer.on_update_received((0,), val_B0, time_B); await demuxer.on_update_received((1,), val_B1, time_B)
    await demuxer.on_update_received((2,), val_C2, time_C); await demuxer.on_update_received((3,), val_C3, time_C)
    T_interp_exact = time_A + timedelta(seconds=0.3)
    async with demuxer: await asyncio.sleep(0.5)
    assert len(mock_client.received_tensors) > 0
    closest_tensor_info = min(mock_client.received_tensors, key=lambda x: abs((x[1] - T_interp_exact).total_seconds()))
    assert abs((closest_tensor_info[1] - T_interp_exact).total_seconds()) <= output_interval*1.5
    output_tensor, actual_T_interp = closest_tensor_info
    # Expected for idx 0,1 (B->D)
    span_BD=(time_D-time_B).total_seconds(); ratio_01=max(0.,min(1.,(actual_T_interp-time_B).total_seconds()/span_BD if span_BD else 0))
    exp_B0D=val_B0+ratio_01*(val_D0-val_B0) if actual_T_interp>=time_B else val_B0; exp_B1D=val_B1+ratio_01*(val_D1-val_B1) if actual_T_interp>=time_B else val_B1
    if actual_T_interp>time_D: exp_B0D,exp_B1D=val_D0,val_D1
    # Expected for idx 2,3 (A->C then C->D)
    exp_A2C, exp_A3C = val_A2, val_A3
    if actual_T_interp < time_A: pass
    elif actual_T_interp <= time_C: span_AC=(time_C-time_A).total_seconds(); ratio_23AC=max(0.,min(1.,(actual_T_interp-time_A).total_seconds()/span_AC if span_AC else 0)); exp_A2C=val_A2+ratio_23AC*(val_C2-val_A2); exp_A3C=val_A3+ratio_23AC*(val_C3-val_A3)
    elif actual_T_interp <= time_D: span_CD=(time_D-time_C).total_seconds(); ratio_23CD=max(0.,min(1.,(actual_T_interp-time_C).total_seconds()/span_CD if span_CD else 0)); exp_A2C=val_C2+ratio_23CD*(val_D2-val_C2); exp_A3C=val_C3+ratio_23CD*(val_D3-val_C3)
    else: exp_A2C,exp_A3C=val_D2,val_D3
    torch.testing.assert_close(output_tensor[0].item(),exp_B0D,rtol=1e-1,atol=1e-1); torch.testing.assert_close(output_tensor[1].item(),exp_B1D,rtol=1e-1,atol=1e-1)
    torch.testing.assert_close(output_tensor[2].item(),exp_A2C,rtol=1e-1,atol=1e-1); torch.testing.assert_close(output_tensor[3].item(),exp_A3C,rtol=1e-1,atol=1e-1)

@pytest.mark.asyncio
async def test_no_keyframes_output_nan_if_shape_known(linear_strategy: SmoothingStrategy, mock_client: MockDemuxerClient, tensor_shape_1d:Tuple[int])->None:
    demuxer=SmoothedTensorDemuxer(linear_strategy,mock_client,0.01,tensor_shape=tensor_shape_1d)
    async with demuxer:await asyncio.sleep(0.025)
    assert len(mock_client.received_tensors)>=1
    for t,_ in mock_client.received_tensors: assert t.shape==tensor_shape_1d and torch.isnan(t).all()

@pytest.mark.asyncio
async def test_one_keyframe_output_constant_nd(linear_strategy:LinearInterpolationStrategy, mock_client:MockDemuxerClient, tensor_shape_2d:Tuple[int,int])->None:
    demuxer=SmoothedTensorDemuxer(linear_strategy,mock_client,0.01,tensor_shape=tensor_shape_2d)
    kf_time=_utcnow(); vals={(0,0):1.,(0,1):2.,(1,0):3.,(1,1):4.}
    for i,v in vals.items():await demuxer.on_update_received(i,v,kf_time)
    async with demuxer: await asyncio.sleep(0.035)
    assert len(mock_client.received_tensors)>=2
    exp_t=torch.tensor([[vals[(0,0)],vals[(0,1)]],[vals[(1,0)],vals[(1,1)]]],dtype=torch.float32)
    for t,_ in mock_client.received_tensors: assert t.shape==tensor_shape_2d and torch.testing.assert_close(t,exp_t)

@pytest.mark.asyncio
async def test_worker_stop_and_restart(linear_strategy:LinearInterpolationStrategy,mock_client:MockDemuxerClient,tensor_shape_1d:Tuple[int])->None:
    demuxer=SmoothedTensorDemuxer(linear_strategy,mock_client,0.01,tensor_shape=tensor_shape_1d); await demuxer.on_update_received((0,),1.,_utcnow())
    demuxer.start(); await asyncio.sleep(0.05)
    t1=demuxer._SmoothedTensorDemuxer__worker_task # type: ignore[attr-defined]
    assert t1 and not t1.done(); n1=len(mock_client.received_tensors); assert n1>0
    await demuxer.stop(); assert demuxer._SmoothedTensorDemuxer__worker_task is None or demuxer._SmoothedTensorDemuxer__worker_task.done() # type: ignore[attr-defined]
    await asyncio.sleep(0.05); assert len(mock_client.received_tensors)==n1
    mock_client.clear_received_tensors(); demuxer.start(); await demuxer.on_update_received((0,),2.,_utcnow()+timedelta(seconds=0.1)); await asyncio.sleep(0.05)
    t2=demuxer._SmoothedTensorDemuxer__worker_task # type: ignore[attr-defined]
    assert t2 and not t2.done(); assert len(mock_client.received_tensors)>0
    await demuxer.stop()
