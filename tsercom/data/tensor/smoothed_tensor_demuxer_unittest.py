# pylint: disable=missing-class-docstring, missing-function-docstring, protected-access, too-many-lines
# pylint: disable=too-many-arguments, too-many-locals, too-many-statements

import asyncio
import datetime
import itertools
import math
from typing import List, Tuple, Any, Optional, AsyncGenerator, Callable, Coroutine
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch
from _pytest.logging import LogCaptureFixture # For caplog type

from tsercom.data.tensor.smoothing_strategy import (
    SmoothingStrategy,
)
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)
from tsercom.data.tensor.smoothed_tensor_demuxer import (
    SmoothedTensorDemuxer,
)
from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)


# --- Helper Functions ---
BASE_TIME = datetime.datetime(
    2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
)


def ts_at(
    seconds_offset: float, base_time: datetime.datetime = BASE_TIME
) -> datetime.datetime:
    return base_time + datetime.timedelta(seconds=seconds_offset)


def assert_tensors_equal(
    t1: Optional[torch.Tensor], t2: Optional[torch.Tensor], message: str = ""
) -> None:
    if t1 is None and t2 is None:
        return
    if t1 is None or t2 is None:
        pytest.fail(
            f"{message} One tensor is None and the other is not. t1: {t1}, t2: {t2}"
        )

    assert (
        t1.shape == t2.shape
    ), f"{message} Tensor shapes differ: {t1.shape} vs {t2.shape}"
    assert torch.allclose(
        t1, t2
    ), f"{message} Tensor values differ: \n{t1} \nvs \n{t2}"


# --- Fake Client ---
class FakeClient(TensorDemuxer.Client):
    def __init__(self) -> None:
        self.received_tensors: List[Tuple[torch.Tensor, datetime.datetime]] = (
            []
        )
        self.call_log: List[Tuple[str, Any]] = []
        self.lock = asyncio.Lock()

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        async with self.lock:
            cloned_tensor = tensor.clone()
            self.received_tensors.append((cloned_tensor, timestamp))
            self.call_log.append(
                (
                    "on_tensor_changed",
                    {
                        "timestamp": timestamp,
                        "tensor_values": cloned_tensor.tolist(),
                    },
                )
            )

    def clear(self) -> None:
        self.received_tensors.clear()
        self.call_log.clear()

    @property
    def last_payload(self) -> Optional[Tuple[torch.Tensor, datetime.datetime]]:
        if not self.received_tensors:
            return None
        return self.received_tensors[-1]

    def get_payloads_sorted_by_time(
        self,
    ) -> List[Tuple[torch.Tensor, datetime.datetime]]:
        return sorted(self.received_tensors, key=lambda x: x[1])


# --- Fixtures ---
@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest_asyncio.fixture
async def fake_client() -> FakeClient:
    return FakeClient()


@pytest_asyncio.fixture
async def demuxer_1d_default(
    fake_client: FakeClient, linear_strategy: LinearInterpolationStrategy
) -> AsyncGenerator[SmoothedTensorDemuxer, None]:
    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_shape=(3,),
        smoothing_strategy=linear_strategy,
        smoothing_period_seconds=0.1,
    )
    await demuxer.start()
    yield demuxer
    await demuxer.close()


@pytest_asyncio.fixture
async def demuxer_factory(
    fake_client: FakeClient, linear_strategy: LinearInterpolationStrategy
) -> AsyncGenerator[Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], None]:
    demuxers_to_clean: List[SmoothedTensorDemuxer] = []

    async def _factory(
        shape: Tuple[int, ...],
        period: float = 0.1,
        strategy: Optional[SmoothingStrategy] = None,
    ) -> SmoothedTensorDemuxer:
        strat = strategy if strategy else linear_strategy
        dmx = SmoothedTensorDemuxer(
            client=fake_client,
            tensor_shape=shape,
            smoothing_strategy=strat,
            smoothing_period_seconds=period,
        )
        await dmx.start()
        demuxers_to_clean.append(dmx)
        return dmx

    yield _factory

    for d in demuxers_to_clean:
        await d.close()


# --- Test Classes ---

class TestSmoothedTensorDemuxer:
    @pytest.mark.asyncio
    async def test_initialization(
        self,
        fake_client: FakeClient,
        linear_strategy: LinearInterpolationStrategy,
    ) -> None:
        shape = (2, 2)
        period = 0.05
        demuxer = SmoothedTensorDemuxer(
            fake_client, shape, linear_strategy, period
        )
        assert demuxer._SmoothedTensorDemuxer__client == fake_client  # type: ignore[attr-defined]
        assert demuxer._SmoothedTensorDemuxer__tensor_shape == shape
        assert (
            demuxer._SmoothedTensorDemuxer__smoothing_strategy
            == linear_strategy
        )
        assert (
            demuxer._SmoothedTensorDemuxer__smoothing_period_seconds == period
        )
        assert demuxer._SmoothedTensorDemuxer__per_index_keyframes == {}
        assert demuxer._SmoothedTensorDemuxer__interpolation_loop_task is None

        expected_indices = list(
            itertools.product(range(shape[0]), range(shape[1]))
        )
        assert demuxer._SmoothedTensorDemuxer__all_indices == expected_indices
        assert (
            demuxer._tensor_total_elements
            == math.prod(shape)
        )

    @pytest.mark.asyncio
    async def test_initialization_invalid_params(
        self,
        fake_client: FakeClient,
        linear_strategy: LinearInterpolationStrategy,
    ) -> None:
        with pytest.raises(ValueError, match="Tensor shape cannot be empty"):
            SmoothedTensorDemuxer(fake_client, (), linear_strategy, 0.1)
        with pytest.raises(
            ValueError, match="All tensor dimensions must be positive"
        ):
            SmoothedTensorDemuxer(fake_client, (2, 0), linear_strategy, 0.1)
        with pytest.raises(
            ValueError, match="Smoothing period must be positive"
        ):
            SmoothedTensorDemuxer(fake_client, (2,), linear_strategy, 0)
        with pytest.raises(
            TypeError,
            match="smoothing_strategy must be an instance of SmoothingStrategy",
        ):
            SmoothedTensorDemuxer(fake_client, (2,), MagicMock(), 0.1)

    @pytest.mark.asyncio
    async def test_start_stop_worker(
        self, demuxer_1d_default: SmoothedTensorDemuxer
    ) -> None:
        assert (
            demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task
            is not None
        )
        assert (
            not demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task.done()
        )
        await demuxer_1d_default.close()
        assert (
            demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task
            is None
        )
        await demuxer_1d_default.start()
        assert (
            demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task
            is not None
        )
        assert (
            not demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task.done()
        )
        task_id = id(
            demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task
        )
        await demuxer_1d_default.start()
        assert (
            id(
                demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task
            )
            == task_id
        )

    @pytest.mark.asyncio
    async def test_on_update_received_stores_keyframes_1d(
        self, demuxer_1d_default: SmoothedTensorDemuxer
    ) -> None:
        dmx = demuxer_1d_default
        idx0, idx1 = (0,), (1,)
        await dmx.on_update_received(idx0, 10.0, ts_at(0))
        await dmx.on_update_received(idx0, 11.0, ts_at(1))
        await dmx.on_update_received(idx1, 20.0, ts_at(0.5))
        keyframes = dmx._SmoothedTensorDemuxer__per_index_keyframes
        assert keyframes[idx0] == [(ts_at(0), 10.0), (ts_at(1), 11.0)]
        assert keyframes[idx1] == [(ts_at(0.5), 20.0)]
        await dmx.on_update_received(idx0, 10.5, ts_at(0))
        assert keyframes[idx0] == [(ts_at(0), 10.5), (ts_at(1), 11.0)]
        await dmx.on_update_received(idx0, 10.2, ts_at(0.5))
        assert keyframes[idx0] == [
            (ts_at(0), 10.5),
            (ts_at(0.5), 10.2),
            (ts_at(1), 11.0),
        ]

    @pytest.mark.asyncio
    async def test_on_update_received_invalid_index(
        self, demuxer_1d_default: SmoothedTensorDemuxer, caplog: LogCaptureFixture
    ) -> None:
        await demuxer_1d_default.on_update_received((3,), 10.0, ts_at(0))
        assert not demuxer_1d_default._SmoothedTensorDemuxer__per_index_keyframes
        assert "Index (3,) out of bounds for shape (3,)" in caplog.text
        await demuxer_1d_default.on_update_received((0,0), 10.0, ts_at(0))
        assert not demuxer_1d_default._SmoothedTensorDemuxer__per_index_keyframes
        assert "Invalid index dimension (0, 0) for shape (3,)" in caplog.text

    @pytest.mark.asyncio
    async def test_simple_interpolation_one_index(
        self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient
    ) -> None:
        dmx = await demuxer_factory(shape=(1,), period=0.1)
        idx = (0,)
        await dmx.on_update_received(idx, 0.0, ts_at(0))
        await dmx.on_update_received(idx, 20.0, ts_at(2.0))
        await asyncio.sleep(2.0 + 0.5)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) >= 20
        assert payloads[0][1] == ts_at(0.1)
        assert_tensors_equal(payloads[0][0], torch.tensor([1.0]))
        found_mid = any(p[1] == ts_at(1.0) and torch.allclose(p[0], torch.tensor([10.0])) for p in payloads)
        assert found_mid, "Midpoint interpolation at t=1.0 not found"
        found_end_kf = any(p[1] == ts_at(2.0) and torch.allclose(p[0], torch.tensor([20.0])) for p in payloads)
        assert found_end_kf, "End keyframe interpolation at t=2.0 not found"
        found_after_end = any(p[1] == ts_at(2.1) and torch.allclose(p[0], torch.tensor([20.0])) for p in payloads)
        assert found_after_end, "Interpolation after last keyframe (holding value) not found"

    @pytest.mark.asyncio
    async def test_interpolation_multiple_indices_independent(
        self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient
    ) -> None:
        dmx = await demuxer_factory(shape=(2,), period=0.1)
        idx0, idx1 = (0,), (1,)
        await dmx.on_update_received(idx0, 0.0, ts_at(0))
        await dmx.on_update_received(idx0, 10.0, ts_at(1.0))
        await dmx.on_update_received(idx1, 100.0, ts_at(0.5))
        await dmx.on_update_received(idx1, 200.0, ts_at(1.5))
        await asyncio.sleep(1.5 + 0.5)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 10
        target_ts_to_check = ts_at(0.6)
        found_target_ts = any(p[1] == target_ts_to_check and torch.allclose(p[0], torch.tensor([6.0, 110.0])) for p in payloads)
        assert found_target_ts, f"Did not find interpolated tensor at {target_ts_to_check}"
        target_ts_to_check_2 = ts_at(1.2)
        found_target_ts_2 = any(p[1] == target_ts_to_check_2 and torch.allclose(p[0], torch.tensor([10.0, 170.0])) for p in payloads)
        assert found_target_ts_2, f"Did not find interpolated tensor at {target_ts_to_check_2}"

    @pytest.mark.asyncio
    async def test_no_keyframes_emits_zeros_or_nothing(
        self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient
    ) -> None:
        dmx = await demuxer_factory(shape=(2,), period=0.1)
        await asyncio.sleep(0.5)
        assert not fake_client.received_tensors
        await dmx.on_update_received((0,), 10.0, ts_at(0))
        await dmx.on_update_received((0,), 20.0, ts_at(1.0))
        fake_client.clear()
        await asyncio.sleep(0.5)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0
        first_payload_tensor, _ = payloads[0]
        assert first_payload_tensor[1].item() == 0.0
        assert first_payload_tensor[0].item() != 0.0

    @pytest.mark.asyncio
    async def test_critical_cascading_scenario(
        self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient
    ) -> None:
        dmx = await demuxer_factory(shape=(4,), period=0.1)
        idx = [(i,) for i in range(4)]
        tA, tB, tC, tD = ts_at(0), ts_at(1.0), ts_at(2.0), ts_at(3.0)
        vals_A = [0.0, 10.0, 20.0, 30.0]
        for i in range(4): await dmx.on_update_received(idx[i], vals_A[i], tA)
        vals_D = [30.0, 40.0, 50.0, 60.0]
        for i in range(4): await dmx.on_update_received(idx[i], vals_D[i], tD)
        vals_B = [5.0, 15.0]
        await dmx.on_update_received(idx[0], vals_B[0], tB)
        await dmx.on_update_received(idx[1], vals_B[1], tB)
        vals_C = [28.0, 38.0]
        await dmx.on_update_received(idx[2], vals_C[0], tC)
        await dmx.on_update_received(idx[3], vals_C[1], tC)
        await asyncio.sleep(3.0 + 0.5)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0, "No tensors were emitted"
        target_t_interp = ts_at(1.5)
        found_target_payload = False
        for tensor, ts_val in payloads:
            if ts_val == target_t_interp:
                found_target_payload = True
                expected_values = torch.tensor([11.25, 21.25, 26.0, 36.0], dtype=torch.float32)
                assert_tensors_equal(tensor, expected_values, message=f"Tensor at {target_t_interp}")
                break
        assert found_target_payload, f"Payload for target T_interp {target_t_interp} not found."

    @pytest.mark.asyncio
    async def test_worker_stops_and_restarts(
        self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient
    ) -> None:
        dmx = await demuxer_factory(shape=(1,), period=0.05)
        await dmx.on_update_received((0,), 0.0, ts_at(0))
        await dmx.on_update_received((0,), 10.0, ts_at(0.2))
        await asyncio.sleep(0.3)
        assert len(fake_client.received_tensors) > 0
        await dmx.close()
        assert dmx._SmoothedTensorDemuxer__interpolation_loop_task is None or dmx._SmoothedTensorDemuxer__interpolation_loop_task.done() # type: ignore[attr-defined]
        fake_client.clear()
        await dmx.on_update_received((0,), 20.0, ts_at(0.4))
        await asyncio.sleep(0.2)
        assert not fake_client.received_tensors
        await dmx.start()
        await asyncio.sleep(0.5)
        assert len(fake_client.received_tensors) > 0
        target_t_interp = ts_at(0.3)
        found_target = any(p[1] == target_t_interp and torch.allclose(p[0], torch.tensor([15.0])) for p in fake_client.get_payloads_sorted_by_time())
        assert found_target, f"Interpolation after restart for {target_t_interp} failed or not found."

    @pytest.mark.asyncio
    async def test_generate_all_indices_static_method(self) -> None:
        assert SmoothedTensorDemuxer._generate_all_indices(()) == []
        assert SmoothedTensorDemuxer._generate_all_indices((3,)) == [(0,), (1,), (2,)]
        assert SmoothedTensorDemuxer._generate_all_indices((2,2)) == [(0,0), (0,1), (1,0), (1,1)]
        assert SmoothedTensorDemuxer._generate_all_indices((1,2,1)) == [(0,0,0), (0,1,0)]

    @pytest.mark.asyncio
    async def test_interpolation_with_2d_tensor(
        self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient
    ) -> None:
        dmx = await demuxer_factory(shape=(2,1), period=0.1)
        idx_0_0, idx_1_0 = (0,0), (1,0)
        await dmx.on_update_received(idx_0_0, 0.0, ts_at(0))
        await dmx.on_update_received(idx_0_0, 10.0, ts_at(1.0))
        await dmx.on_update_received(idx_1_0, 100.0, ts_at(0))
        await dmx.on_update_received(idx_1_0, 120.0, ts_at(1.0))
        await asyncio.sleep(0.5 + 0.2)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0
        expected_tensor = torch.tensor([[5.0], [110.0]], dtype=torch.float32)
        target_t_interp = ts_at(0.5)
        found_target = any(p[1] == target_t_interp and torch.allclose(p[0], expected_tensor) for p in payloads)
        assert found_target, f"Did not find tensor for {target_t_interp} with 2D shape."

    @pytest.mark.asyncio
    async def test_behavior_with_very_small_smoothing_period(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient) -> None:
        period = 0.01
        dmx = await demuxer_factory(shape=(1,), period=period)
        idx = (0,)
        await dmx.on_update_received(idx, 0.0, ts_at(0))
        await dmx.on_update_received(idx, 10.0, ts_at(0.1))
        await asyncio.sleep(0.1 + 0.05)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) >= 8
        found_target_ts = False
        if payloads:
            first_payload_ts = payloads[0][1]
            assert abs((first_payload_ts - ts_at(0.01)).total_seconds()) < period, \
                f"First emitted point {first_payload_ts} not close to expected {ts_at(0.01)}"
            assert_tensors_equal(payloads[0][0], torch.tensor([1.0]), message="Tensor at first emission")
            found_target_ts = True
        assert found_target_ts, "No data emitted or first point check failed with small period."

    @pytest.mark.asyncio
    async def test_behavior_with_large_smoothing_period(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient) -> None:
        period = 1.0
        dmx = await demuxer_factory(shape=(1,), period=period)
        idx = (0,)
        await dmx.on_update_received(idx, 0.0, ts_at(0))
        await dmx.on_update_received(idx, 10.0, ts_at(0.5))
        await asyncio.sleep(period + 0.2)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) == 1, "Expected exactly one emission with large period"
        tensor_val, ts_val = payloads[0]
        assert ts_val == ts_at(1.0)
        assert_tensors_equal(tensor_val, torch.tensor([10.0]), message="Tensor at t=1.0 (extrapolated)")

    @pytest.mark.asyncio
    async def test_rapid_on_update_received_calls(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient) -> None:
        dmx = await demuxer_factory(shape=(1,), period=0.1)
        idx = (0,)

        num_updates = 20 # Ends at t=0.19, val=19.0
        for i in range(num_updates):
            time = ts_at(i * 0.01)
            value = float(i)
            await dmx.on_update_received(idx, value, time)

        await asyncio.sleep(0.2 + 0.2) # Wait for interpolation to catch up

        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0, "No data emitted"

        # Expect first emission at t0 (0.0) + period (0.1) = ts_at(0.1)
        # Value at t=0.1: keyframes are (0.0,0)...(0.1,10.0)...(0.19,19.0)
        # Interpolation strategy with keyframes up to (0.1,10) and req_ts=0.1 should yield 10.0
        exact_target_ts = ts_at(0.1)
        expected_value_tensor = torch.tensor([10.0])

        found_exact_target = False
        for tensor, ts_val in payloads:
            if ts_val == exact_target_ts:
                assert_tensors_equal(tensor, expected_value_tensor, message=f"Tensor at {exact_target_ts}")
                found_exact_target = True
                break
        assert found_exact_target, f"Did not find expected emission at {exact_target_ts} (expected value {expected_value_tensor.item()}). Payloads: {payloads}"

    @pytest.mark.asyncio
    async def test_update_timestamp_far_past(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient) -> None:
        dmx = await demuxer_factory(shape=(1,), period=0.1)
        idx = (0,)
        await dmx.on_update_received(idx, 10.0, ts_at(1.0))
        await dmx.on_update_received(idx, 20.0, ts_at(2.0))
        await asyncio.sleep(0.2)
        fake_client.clear()
        await dmx.on_update_received(idx, 0.0, ts_at(0.0))
        await asyncio.sleep(0.1 + 0.05)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0, "No data emitted after past update"
        first_tensor, first_ts = payloads[0]
        assert first_ts == ts_at(0.1)
        assert_tensors_equal(first_tensor, torch.tensor([1.0]))

    @pytest.mark.asyncio
    async def test_update_timestamp_far_future(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient) -> None:
        dmx = await demuxer_factory(shape=(1,), period=0.1)
        idx = (0,)

        await dmx.on_update_received(idx, 0.0, ts_at(0))
        await dmx.on_update_received(idx, 10.0, ts_at(1.0)) # Original data: (0,0) to (1,10)

        # Let some initial interpolation happen
        await asyncio.sleep(0.25) # e.g., emits for t=0.1 (val 1), t=0.2 (val 2)
        initial_payload_count = len(fake_client.received_tensors)
        assert initial_payload_count > 0, "No initial emissions"
        fake_client.clear()

        # Add update far in the future
        await dmx.on_update_received(idx, 100.0, ts_at(10.0))

        # Wait significantly longer to ensure worker has many cycles
        # to emit points for the original segment if logic is correct.
        # Target is ts_at(0.5) (value 5.0)
        await asyncio.sleep(0.6) # Should be enough for t=0.3, 0.4, 0.5, etc.

        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0, "No data emitted after future update"

        target_ts_to_check = ts_at(0.5)
        expected_value_at_target = 5.0 # Interpolated from (0,0) and (1,10)

        found_target = False
        for tensor, ts_val in payloads:
            if ts_val == target_ts_to_check:
                assert_tensors_equal(tensor, torch.tensor([expected_value_at_target]),
                                     message=f"Tensor at {target_ts_to_check}")
                found_target = True
                break
        assert found_target, f"Interpolation for {target_ts_to_check} (value {expected_value_at_target}) not found or incorrect after future update. Payloads: {payloads}"

        # Also check that interpolation towards the far future point eventually occurs
        fake_client.clear()
        await asyncio.sleep(10.0) # Very long sleep to pass t=1.0 and go towards t=10.0

        payloads_after_long_sleep = fake_client.get_payloads_sorted_by_time()
        assert len(payloads_after_long_sleep) > 0, "No emissions after long sleep towards future point"

        # Check for a point interpolating between (1.0, 10.0) and (10.0, 100.0)
        # e.g., ts_at(9.9). Value = 10.0 + (100.0-10.0) * ( (9.9-1.0) / (10.0-1.0) ) = 99.0
        future_target_ts = ts_at(9.9)
        expected_future_value = 99.0
        found_future_target = False
        for tensor, ts_val in payloads_after_long_sleep:
            # Allow for slight timing deviations by checking a range or closest point if exact fails
            if abs((ts_val - future_target_ts).total_seconds()) < (dmx._SmoothedTensorDemuxer__smoothing_period_seconds / 2.0): # type: ignore[attr-defined]
                # Recalculate expected for actual ts_val to be robust
                # For simplicity, if it's close to 9.9, we expect value close to 99.0
                assert abs(tensor.item() - expected_future_value) < 1.0,                     f"Value {tensor.item()} at {ts_val} not close to expected {expected_future_value} for target {future_target_ts}"
                found_future_target = True
                break
        assert found_future_target, f"Interpolation towards far future point {future_target_ts} failed. Payloads: {payloads_after_long_sleep}"

    @pytest.mark.asyncio
    async def test_strategy_raises_exception(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient, caplog: LogCaptureFixture) -> None:
        class MockErrorStrategy(SmoothingStrategy):
            def interpolate_series(self, kf: List[Tuple[datetime.datetime, Any]], ts_req: List[datetime.datetime]) -> List[Any]: # Fully fixed signature
                raise ValueError("Test strategy error")
        dmx = await demuxer_factory(shape=(1,), strategy=MockErrorStrategy(), period=0.05)
        idx = (0,)
        await dmx.on_update_received(idx, 0.0, ts_at(0))
        await dmx.on_update_received(idx, 10.0, ts_at(0.1))
        await asyncio.sleep(0.1)
        assert not fake_client.received_tensors, "No tensors should be emitted if strategy fails"
        assert "Test strategy error" in caplog.text
        assert "Strategy error for index (0,)" in caplog.text

    @pytest.mark.asyncio
    async def test_reinitialize_index_after_clearing_keyframes(self, demuxer_factory: Callable[..., Coroutine[Any, Any, SmoothedTensorDemuxer]], fake_client: FakeClient) -> None:
        dmx = await demuxer_factory(shape=(1,), period=0.1)
        idx_to_clear = (0,)
        await dmx.on_update_received(idx_to_clear, 10.0, ts_at(0))
        await dmx.on_update_received(idx_to_clear, 20.0, ts_at(1.0))
        await asyncio.sleep(0.2)
        assert len(fake_client.received_tensors) > 0
        fake_client.clear()
        async with dmx._SmoothedTensorDemuxer__keyframe_lock: # type: ignore[attr-defined]
            if idx_to_clear in dmx._SmoothedTensorDemuxer__per_index_keyframes: # type: ignore[attr-defined]
                dmx._SmoothedTensorDemuxer__per_index_keyframes[idx_to_clear].clear() # type: ignore[attr-defined]
        await dmx.on_update_received(idx_to_clear, 100.0, ts_at(2.0))
        await dmx.on_update_received(idx_to_clear, 110.0, ts_at(3.0))
        await asyncio.sleep(0.1 + 0.05)
        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0, "No data emitted after re-initializing index"
        first_tensor, first_ts = payloads[0]
        assert first_ts >= ts_at(2.1)
        expected_val_at_first_ts = 100.0 + (110.0 - 100.0) * (
            (first_ts - ts_at(2.0)).total_seconds()
            / (ts_at(3.0) - ts_at(2.0)).total_seconds()
        )
        assert abs(first_tensor.item() - expected_val_at_first_ts) < 1e-5
