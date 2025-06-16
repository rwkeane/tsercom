# pylint: disable=missing-class-docstring, missing-function-docstring, protected-access, too-many-lines
# pylint: disable=too-many-arguments, too-many-locals, too-many-statements

import asyncio
import datetime
import itertools
import math
from typing import List, Tuple, Any, Optional  # Added Dict
from unittest.mock import MagicMock  # Added MagicMock

import pytest
import pytest_asyncio
import torch

from tsercom.data.tensor.smoothing_strategy import (  # Corrected filename
    SmoothingStrategy,
    # Numeric, # Not used in this file
)
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)  # New import
from tsercom.data.tensor.smoothed_tensor_demuxer import (
    SmoothedTensorDemuxer,
)
from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # For Client definition


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
):
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
    def __init__(self):
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

    def clear(self):
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
) -> SmoothedTensorDemuxer:
    # 1D tensor, length 3, period 1s
    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_shape=(3,),
        smoothing_strategy=linear_strategy,
        smoothing_period_seconds=0.1,  # Use a shorter period for faster test feedback
    )
    await demuxer.start()
    yield demuxer
    await demuxer.close()


@pytest_asyncio.fixture
async def demuxer_factory(
    fake_client: FakeClient, linear_strategy: LinearInterpolationStrategy
):
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
# TestLinearInterpolationStrategy class removed as it's now in its own file.


class TestSmoothedTensorDemuxer:
    @pytest.mark.asyncio
    async def test_initialization(
        self,
        fake_client: FakeClient,
        linear_strategy: LinearInterpolationStrategy,
    ):
        shape = (2, 2)
        period = 0.05
        demuxer = SmoothedTensorDemuxer(
            fake_client, shape, linear_strategy, period
        )
        assert demuxer._SmoothedTensorDemuxer__client == fake_client
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
            demuxer._tensor_total_elements  # Corrected attribute access
            == math.prod(shape)
        )

    @pytest.mark.asyncio
    async def test_initialization_invalid_params(
        self,
        fake_client: FakeClient,
        linear_strategy: LinearInterpolationStrategy,
    ):
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
            SmoothedTensorDemuxer(fake_client, (2,), MagicMock(), 0.1)  # type: ignore

    @pytest.mark.asyncio
    async def test_start_stop_worker(
        self, demuxer_1d_default: SmoothedTensorDemuxer
    ):
        # Started by fixture
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
        )  # Cleared by close

        # Restart
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

        await demuxer_1d_default.start()  # Start again, should be idempotent
        assert (
            id(
                demuxer_1d_default._SmoothedTensorDemuxer__interpolation_loop_task
            )
            == task_id
        )

    @pytest.mark.asyncio
    async def test_on_update_received_stores_keyframes_1d(
        self, demuxer_1d_default: SmoothedTensorDemuxer
    ):
        dmx = demuxer_1d_default  # shape (3,)
        idx0, idx1 = (0,), (1,)

        await dmx.on_update_received(idx0, 10.0, ts_at(0))
        await dmx.on_update_received(idx0, 11.0, ts_at(1))
        await dmx.on_update_received(idx1, 20.0, ts_at(0.5))

        keyframes = dmx._SmoothedTensorDemuxer__per_index_keyframes
        assert idx0 in keyframes
        assert idx1 in keyframes
        assert keyframes[idx0] == [(ts_at(0), 10.0), (ts_at(1), 11.0)]
        assert keyframes[idx1] == [(ts_at(0.5), 20.0)]

        # Update existing timestamp for idx0
        await dmx.on_update_received(idx0, 10.5, ts_at(0))
        assert keyframes[idx0] == [(ts_at(0), 10.5), (ts_at(1), 11.0)]

        # Add out-of-order for idx0
        await dmx.on_update_received(idx0, 10.2, ts_at(0.5))
        assert keyframes[idx0] == [
            (ts_at(0), 10.5),
            (ts_at(0.5), 10.2),
            (ts_at(1), 11.0),
        ]

    @pytest.mark.asyncio
    async def test_on_update_received_invalid_index(
        self, demuxer_1d_default: SmoothedTensorDemuxer, caplog
    ):
        # demuxer_1d_default has shape (3,). Valid indices: (0,), (1,), (2,)
        await demuxer_1d_default.on_update_received(
            (3,), 10.0, ts_at(0)
        )  # Out of bounds
        assert (
            not demuxer_1d_default._SmoothedTensorDemuxer__per_index_keyframes
        )
        assert "Index (3,) out of bounds for shape (3,)" in caplog.text

        await demuxer_1d_default.on_update_received(
            (0, 0), 10.0, ts_at(0)
        )  # Wrong dimension
        assert (
            not demuxer_1d_default._SmoothedTensorDemuxer__per_index_keyframes
        )
        assert "Invalid index dimension (0, 0) for shape (3,)" in caplog.text

    @pytest.mark.asyncio
    async def test_simple_interpolation_one_index(
        self, demuxer_factory: Any, fake_client: FakeClient
    ):
        dmx = await demuxer_factory(
            shape=(1,), period=0.1
        )  # Single element tensor
        idx = (0,)

        await dmx.on_update_received(idx, 0.0, ts_at(0))
        await dmx.on_update_received(
            idx, 20.0, ts_at(2.0)
        )  # Kf0 at t=0, Kf1 at t=2

        # Worker period is 0.1s.
        # _get_next_emission_timestamp logic:
        #   last_emitted is None. min_overall_kf_ts = ts_at(0).
        #   next_emission = ts_at(0) + 0.1s = ts_at(0.1).
        # Worker loop:
        # T_interp = ts_at(0.1). Value = 0.0 + (20-0)* (0.1/2.0) = 1.0. Tensor [1.0]
        # T_interp = ts_at(0.2). Value = 0.0 + (20-0)* (0.2/2.0) = 2.0. Tensor [2.0]
        # ...
        # T_interp = ts_at(1.9). Value = 0.0 + (20-0)* (1.9/2.0) = 19.0. Tensor [19.0]
        # T_interp = ts_at(2.0). Value = 20.0. Tensor [20.0] (exact match with keyframe)
        # Next T_interp would be ts_at(2.1).
        # max_ts_overall is ts_at(2.0).
        # t_interp (2.1) > max_ts_overall (2.0) + period (0.1) is FALSE (2.1 not > 2.1)
        # So it will try to interpolate for ts_at(2.1). LinearInterp will hold last value (20.0)

        # Wait for enough cycles for interpolation up to and slightly beyond last keyframe
        # Total time for 20 points (0.1 to 2.0) is 2.0s. Add buffer.
        await asyncio.sleep(2.0 + 0.5)  # Sleep for 2.5s

        payloads = fake_client.get_payloads_sorted_by_time()

        # Expected number of points: from ts_at(0.1) to ts_at(2.0) inclusive = 20 points
        # And potentially one more at ts_at(2.1) holding value 20.0.
        # The worker's `t_interp > max_ts_overall + self.__smoothing_period_seconds` check
        # means if t_interp = 2.1, max_ts_overall = 2.0, period = 0.1
        # 2.1 > 2.0 + 0.1  (i.e. 2.1 > 2.1) is false. So it will attempt to interpolate.
        # LinearInterpolationStrategy for t_interp=2.1 (after last keyframe at 2.0) will give value 20.0.

        assert len(payloads) >= 20  # Should be around 20-21 points

        # Check first point
        assert payloads[0][1] == ts_at(0.1)
        assert_tensors_equal(payloads[0][0], torch.tensor([1.0]))

        # Check point at t=1.0 (midpoint)
        found_mid = False
        for tensor, ts_val in payloads:
            if ts_val == ts_at(1.0):
                assert_tensors_equal(tensor, torch.tensor([10.0]))
                found_mid = True
                break
        assert found_mid, "Midpoint interpolation at t=1.0 not found"

        # Check point at t=2.0 (exact keyframe)
        found_end_kf = False
        for tensor, ts_val in payloads:
            if ts_val == ts_at(2.0):
                assert_tensors_equal(tensor, torch.tensor([20.0]))
                found_end_kf = True
                break
        assert found_end_kf, "End keyframe interpolation at t=2.0 not found"

        # Check point after last keyframe (e.g. t=2.1)
        found_after_end = False
        for tensor, ts_val in payloads:
            if ts_val == ts_at(2.1):
                assert_tensors_equal(
                    tensor, torch.tensor([20.0])
                )  # Should hold last value
                found_after_end = True
                break
        assert (
            found_after_end
        ), "Interpolation after last keyframe (holding value) not found"

    @pytest.mark.asyncio
    async def test_interpolation_multiple_indices_independent(
        self, demuxer_factory: Any, fake_client: FakeClient
    ):
        dmx = await demuxer_factory(
            shape=(2,), period=0.1
        )  # 1D tensor, 2 elements
        idx0, idx1 = (0,), (1,)

        # Index 0: 0 at t=0, 10 at t=1
        await dmx.on_update_received(idx0, 0.0, ts_at(0))
        await dmx.on_update_received(idx0, 10.0, ts_at(1.0))

        # Index 1: 100 at t=0.5, 200 at t=1.5
        await dmx.on_update_received(idx1, 100.0, ts_at(0.5))
        await dmx.on_update_received(idx1, 200.0, ts_at(1.5))

        # Wait for interpolation to occur for some time
        await asyncio.sleep(1.5 + 0.5)  # Sleep for 2.0s

        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 10  # Expect a number of points

        # Check specific timestamp, e.g., t=0.6
        # Worker emits at 0.1, 0.2, ..., 0.6
        # At t=0.6:
        #   Idx0: between (0,0) and (1,10). Value = 0 + (10-0)*(0.6/1.0) = 6.0
        #   Idx1: between (0.5,100) and (1.5,200). Value = 100 + (200-100)*( (0.6-0.5) / (1.5-0.5) )
        #         = 100 + 100 * (0.1/1.0) = 100 + 10 = 110.0
        # Expected tensor: [6.0, 110.0]

        found_target_ts = False
        target_ts_to_check = ts_at(0.6)
        for tensor, ts_val in payloads:
            if ts_val == target_ts_to_check:
                assert_tensors_equal(tensor, torch.tensor([6.0, 110.0]))
                found_target_ts = True
                break
        assert (
            found_target_ts
        ), f"Did not find interpolated tensor at {target_ts_to_check}"

        # Check another timestamp, e.g., t=1.2
        # At t=1.2:
        #   Idx0: After last keyframe (1.0, 10.0). Value = 10.0
        #   Idx1: between (0.5,100) and (1.5,200). Value = 100 + (200-100)*( (1.2-0.5) / (1.5-0.5) )
        #         = 100 + 100 * (0.7/1.0) = 100 + 70 = 170.0
        # Expected tensor: [10.0, 170.0]
        found_target_ts_2 = False
        target_ts_to_check_2 = ts_at(1.2)
        for tensor, ts_val in payloads:
            if ts_val == target_ts_to_check_2:
                assert_tensors_equal(tensor, torch.tensor([10.0, 170.0]))
                found_target_ts_2 = True
                break
        assert (
            found_target_ts_2
        ), f"Did not find interpolated tensor at {target_ts_to_check_2}"

    @pytest.mark.asyncio
    async def test_no_keyframes_emits_zeros_or_nothing(
        self, demuxer_factory: Any, fake_client: FakeClient
    ):
        # If no keyframes at all, worker should ideally not emit, or emit zeros if that's desired.
        # Current worker logic: if no keyframes, _get_next_emission_timestamp bases on current_time.
        # Then in worker, if __per_index_keyframes is empty, it logs and sleeps.
        # If an index has no keyframes, it gets default 0.0.
        dmx = await demuxer_factory(shape=(2,), period=0.1)

        await asyncio.sleep(0.5)  # Let worker run a few times
        assert (
            not fake_client.received_tensors
        )  # Should not emit if no data ever received

        # Add data for one index, but not the other
        await dmx.on_update_received((0,), 10.0, ts_at(0))
        await dmx.on_update_received((0,), 20.0, ts_at(1.0))

        fake_client.clear()
        await asyncio.sleep(0.5)

        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0

        # Example: check first emitted tensor. Should be like [val_for_idx0, 0.0]
        # T_interp = ts_at(0.1). Idx0 val = 1.0. Idx1 val = 0.0 (default as no kfs)
        # Expected: [1.0, 0.0]
        first_payload_tensor, first_payload_ts = payloads[0]
        # The exact first timestamp depends on when the worker ran relative to data input.
        # We expect index 0 to be interpolated, index 1 to be 0.0
        assert first_payload_tensor[1].item() == 0.0
        assert (
            first_payload_tensor[0].item() != 0.0
        )  # Should be some interpolated value for idx0

    @pytest.mark.asyncio
    async def test_critical_cascading_scenario(
        self, demuxer_factory: Any, fake_client: FakeClient
    ):
        # Shape (4,), period 0.1s for faster feedback
        dmx = await demuxer_factory(shape=(4,), period=0.1)

        idx = [(i,) for i in range(4)]  # idx[0]=(0,), idx[1]=(1,), ...

        # Timestamps A, B, C, D
        tA = ts_at(0)
        tB = ts_at(1.0)
        tC = ts_at(2.0)
        tD = ts_at(3.0)

        # 1. Keyframes at A: A0, A1, A2, A3 for all indices
        vals_A = [0.0, 10.0, 20.0, 30.0]
        for i in range(4):
            await dmx.on_update_received(idx[i], vals_A[i], tA)

        # 2. Keyframes at D: D0, D1, D2, D3 for all indices
        vals_D = [30.0, 40.0, 50.0, 60.0]  # D is 3s after A
        for i in range(4):
            await dmx.on_update_received(idx[i], vals_D[i], tD)

        # 3. Keyframes at B (out-of-order for some indices' timelines): B0, B1 for idx0, idx1
        vals_B = [5.0, 15.0]  # B is 1s after A
        await dmx.on_update_received(idx[0], vals_B[0], tB)
        await dmx.on_update_received(idx[1], vals_B[1], tB)

        # 4. Keyframes at C (out-of-order for some indices' timelines): C2, C3 for idx2, idx3
        vals_C = [28.0, 38.0]  # C is 2s after A
        await dmx.on_update_received(idx[2], vals_C[0], tC)
        await dmx.on_update_received(idx[3], vals_C[1], tC)

        # Let worker run. We need to check a T_interp between B and C.
        # E.g., T_interp = ts_at(1.5) (0.5s after B, 0.5s before C)
        # Period is 0.1s. Worker runs at 0.1, 0.2 ... 1.0(B) ... 1.5 ... 2.0(C) ... 3.0(D)
        await asyncio.sleep(
            3.0 + 0.5
        )  # Wait for interpolations up to D and a bit beyond.

        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0, "No tensors were emitted"

        target_t_interp = ts_at(1.5)
        found_target_payload = False

        for tensor, ts_val in payloads:
            if ts_val == target_t_interp:
                found_target_payload = True
                # At T_interp = 1.5s:
                # Idx0: interpolates B0 (5.0 at tB=1s) towards D0 (30.0 at tD=3s)
                #       Path B->D is 2s long (from 1s to 3s). T_interp is 0.5s into this path.
                #       Value = 5.0 + (30.0 - 5.0) * (0.5 / 2.0) = 5.0 + 25.0 * 0.25 = 5.0 + 6.25 = 11.25
                # Idx1: interpolates B1 (15.0 at tB=1s) towards D1 (40.0 at tD=3s)
                #       Value = 15.0 + (40.0 - 15.0) * (0.5 / 2.0) = 15.0 + 25.0 * 0.25 = 15.0 + 6.25 = 21.25
                # Idx2: interpolates A2 (20.0 at tA=0s) towards C2 (28.0 at tC=2s)
                #       Path A->C is 2s long (from 0s to 2s). T_interp is 1.5s into this path.
                #       Value = 20.0 + (28.0 - 20.0) * (1.5 / 2.0) = 20.0 + 8.0 * 0.75 = 20.0 + 6.0 = 26.0
                # Idx3: interpolates A3 (30.0 at tA=0s) towards C3 (38.0 at tC=2s)
                #       Value = 30.0 + (38.0 - 30.0) * (1.5 / 2.0) = 30.0 + 8.0 * 0.75 = 30.0 + 6.0 = 36.0

                expected_values = torch.tensor(
                    [11.25, 21.25, 26.0, 36.0], dtype=torch.float32
                )
                assert_tensors_equal(
                    tensor,
                    expected_values,
                    message=f"Tensor at {target_t_interp}",
                )
                break

        assert (
            found_target_payload
        ), f"Payload for target T_interp {target_t_interp} not found."

    @pytest.mark.asyncio
    async def test_worker_stops_and_restarts(
        self, demuxer_factory: Any, fake_client: FakeClient
    ):
        dmx = await demuxer_factory(shape=(1,), period=0.05)  # Short period
        await dmx.on_update_received((0,), 0.0, ts_at(0))
        await dmx.on_update_received((0,), 10.0, ts_at(0.2))

        await asyncio.sleep(0.3)  # Let it emit a few points
        payload_count_before_stop = len(fake_client.received_tensors)
        assert payload_count_before_stop > 0

        await dmx.close()  # Stop the worker
        # Ensure worker task is gone or done
        assert (
            dmx._SmoothedTensorDemuxer__interpolation_loop_task is None
            or dmx._SmoothedTensorDemuxer__interpolation_loop_task.done()
        )

        fake_client.clear()  # Clear previous payloads

        # Try sending more data while stopped - should be stored but not emitted
        await dmx.on_update_received((0,), 20.0, ts_at(0.4))
        await asyncio.sleep(0.2)  # Give time for emission if it were running
        assert (
            not fake_client.received_tensors
        )  # No new tensors should be emitted

        # Restart the worker
        await dmx.start()
        await asyncio.sleep(0.5)  # Let it run again (increased sleep)

        payload_count_after_restart = len(fake_client.received_tensors)
        assert payload_count_after_restart > 0

        # Check if it interpolated using the data point received while stopped
        # T_interp e.g. ts_at(0.3) (0.1s after (0.2,10), 0.1s before (0.4,20))
        # Value = 10 + (20-10) * (0.1/0.2) = 10 + 10 * 0.5 = 15
        target_t_interp = ts_at(0.3)
        found_target = False
        for tensor, ts_val in fake_client.get_payloads_sorted_by_time():
            if ts_val == target_t_interp:
                assert_tensors_equal(tensor, torch.tensor([15.0]))
                found_target = True
                break
        assert (
            found_target
        ), f"Interpolation after restart for {target_t_interp} failed or not found."

    @pytest.mark.asyncio
    async def test_generate_all_indices_static_method(self):
        assert (
            SmoothedTensorDemuxer._generate_all_indices(()) == []
        )  # Empty shape
        assert SmoothedTensorDemuxer._generate_all_indices((3,)) == [
            (0,),
            (1,),
            (2,),
        ]
        assert SmoothedTensorDemuxer._generate_all_indices((2, 2)) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]
        assert SmoothedTensorDemuxer._generate_all_indices((1, 2, 1)) == [
            (0, 0, 0),
            (0, 1, 0),
        ]

    @pytest.mark.asyncio
    async def test_interpolation_with_2d_tensor(
        self, demuxer_factory: Any, fake_client: FakeClient
    ):
        dmx = await demuxer_factory(
            shape=(2, 1), period=0.1
        )  # Shape (2,1) -> indices (0,0), (1,0)
        idx_0_0 = (0, 0)
        idx_1_0 = (1, 0)

        await dmx.on_update_received(idx_0_0, 0.0, ts_at(0))
        await dmx.on_update_received(
            idx_0_0, 10.0, ts_at(1.0)
        )  # (0,0) from 0 to 10 over 1s

        await dmx.on_update_received(idx_1_0, 100.0, ts_at(0))
        await dmx.on_update_received(
            idx_1_0, 120.0, ts_at(1.0)
        )  # (1,0) from 100 to 120 over 1s

        await asyncio.sleep(0.5 + 0.2)  # Interpolate up to t=0.5

        payloads = fake_client.get_payloads_sorted_by_time()
        assert len(payloads) > 0

        # Check t_interp = ts_at(0.5)
        # Idx (0,0): 0 + (10-0)*(0.5/1.0) = 5.0
        # Idx (1,0): 100 + (120-100)*(0.5/1.0) = 110.0
        # Expected tensor: [[5.0], [110.0]]
        expected_tensor = torch.tensor([[5.0], [110.0]], dtype=torch.float32)

        target_t_interp = ts_at(0.5)
        found_target = False
        for tensor, ts_val in payloads:
            if ts_val == target_t_interp:
                assert_tensors_equal(tensor, expected_tensor)
                found_target = True
                break
        assert (
            found_target
        ), f"Did not find tensor for {target_t_interp} with 2D shape."
