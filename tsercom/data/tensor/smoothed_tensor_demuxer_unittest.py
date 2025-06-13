# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=protected-access # For accessing _sm_client, _keyframes etc. in tests
# pylint: disable=too-many-arguments

import asyncio
import datetime
from typing import List, Tuple, Any, Optional  # Optional was missing

# from unittest.mock import AsyncMock  # For mocking the client - Removed
import pytest_asyncio  # Added

import pytest
import torch

# Absolute imports as required
from tsercom.data.tensor.tensor_demuxer import TensorDemuxer
from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer


# Helper for creating timestamps
def ts_at(
    seconds_offset: float, base_time: Optional[datetime.datetime] = None
) -> datetime.datetime:
    if base_time is None:
        base_time = datetime.datetime(
            2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
        )
    return base_time + datetime.timedelta(seconds=seconds_offset)


class FakeClient(TensorDemuxer.Client):
    """A fake client to capture outputs from SmoothedTensorDemuxer."""

    def __init__(self):
        self.received_tensors: List[Tuple[torch.Tensor, datetime.datetime]] = (
            []
        )
        self.call_log: List[Tuple[str, Any]] = (
            []
        )  # Logs calls for detailed verification

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.received_tensors.append((tensor.clone(), timestamp))
        self.call_log.append(
            (
                "on_tensor_changed",
                {"timestamp": timestamp, "tensor_shape": tensor.shape},
            )
        )
        # print(f"FakeClient received tensor at {timestamp} with shape {tensor.shape} and values {tensor.tolist()}")

    def clear(self):
        self.received_tensors.clear()
        self.call_log.clear()

    @property
    def last_tensor_payload(
        self,
    ) -> Optional[Tuple[torch.Tensor, datetime.datetime]]:
        if not self.received_tensors:
            return None
        return self.received_tensors[-1]


@pytest_asyncio.fixture  # Changed
async def fake_client() -> FakeClient:
    return FakeClient()


@pytest_asyncio.fixture  # Changed
async def sm_demuxer(fake_client: FakeClient) -> SmoothedTensorDemuxer:
    # Default tensor_length=3, smoothing_period=1.0s
    demuxer = SmoothedTensorDemuxer(
        client=fake_client, tensor_length=3, smoothing_period_seconds=1.0
    )
    await demuxer.start()  # Start the interpolation worker
    yield demuxer
    await demuxer.close()  # Ensure cleanup


@pytest_asyncio.fixture  # Changed
async def sm_demuxer_custom(fake_client: FakeClient):
    """Fixture to create a demuxer with custom params passed via a factory function."""
    demuxers_to_clean = []

    async def _factory(
        tensor_length: int, smoothing_period_seconds: float
    ) -> SmoothedTensorDemuxer:
        demuxer = SmoothedTensorDemuxer(
            client=fake_client,
            tensor_length=tensor_length,
            smoothing_period_seconds=smoothing_period_seconds,
        )
        await demuxer.start()
        demuxers_to_clean.append(demuxer)
        return demuxer

    yield _factory

    for d in demuxers_to_clean:
        await d.close()


class TestSmoothedTensorDemuxer:
    """
    Unit tests for the SmoothedTensorDemuxer class.
    """

    @pytest.mark.asyncio
    async def test_initialization(self, fake_client: FakeClient):
        tensor_length = 5
        smoothing_period = 0.5
        demuxer = SmoothedTensorDemuxer(
            client=fake_client,
            tensor_length=tensor_length,
            smoothing_period_seconds=smoothing_period,
        )
        assert demuxer._sm_client == fake_client
        assert demuxer._tensor_length == tensor_length  # Accessed via property
        assert demuxer._smoothing_period_seconds == smoothing_period
        assert not demuxer._keyframes
        assert not demuxer._keyframe_explicit_updates
        assert (
            demuxer._interpolation_loop_task is None
        )  # Not started until start() is called
        await demuxer.start()
        assert demuxer._interpolation_loop_task is not None
        await demuxer.close()
        # After close, task should be None or done
        assert (
            demuxer._interpolation_loop_task is None
            or demuxer._interpolation_loop_task.done()
        )

    @pytest.mark.asyncio
    async def test_start_and_close_idempotency(
        self, sm_demuxer: SmoothedTensorDemuxer
    ):
        # Initial start is done by fixture
        assert sm_demuxer._interpolation_loop_task is not None
        task_id_1 = id(sm_demuxer._interpolation_loop_task)

        await sm_demuxer.start()  # Should not recreate task if already running
        assert id(sm_demuxer._interpolation_loop_task) == task_id_1

        await sm_demuxer.close()
        task_after_close = sm_demuxer._interpolation_loop_task
        assert task_after_close is None or task_after_close.done()

        await sm_demuxer.close()  # Should be safe to call close multiple times
        task_after_double_close = sm_demuxer._interpolation_loop_task
        assert (
            task_after_double_close is None or task_after_double_close.done()
        )

        await sm_demuxer.start()  # Should be able to restart after close
        assert sm_demuxer._interpolation_loop_task is not None
        assert sm_demuxer._interpolation_loop_task.done() is False

    # More tests will be added here in subsequent steps for:
    # - on_update_received (real keyframe creation)
    # - Simple interpolation
    # - Insufficient keyframes for interpolation
    # - Cascading scenario (out-of-order real data)
    # - Edge cases (e.g. smoothing_period = 0)

    @pytest.mark.asyncio
    async def test_simple_interpolation_two_keyframes(
        self, sm_demuxer: SmoothedTensorDemuxer, fake_client: FakeClient
    ):
        # sm_demuxer: tensor_length=3, smoothing_period_seconds=1.0
        t0 = ts_at(0)
        # Keyframe 2 is 2.0 seconds after t0, so one synthetic point should be generated at t0 + 1.0s
        t_kf2 = ts_at(2.0)

        # Real keyframe 1 at T0
        await sm_demuxer.on_update_received(
            tensor_index=0, value=0.0, timestamp=t0
        )
        await sm_demuxer.on_update_received(
            tensor_index=1, value=0.0, timestamp=t0
        )
        await sm_demuxer.on_update_received(
            tensor_index=2, value=0.0, timestamp=t0
        )
        # Keyframe state at t0: [0,0,0]

        # Real keyframe 2 at T_KF2
        await sm_demuxer.on_update_received(
            tensor_index=0, value=20.0, timestamp=t_kf2
        )
        await sm_demuxer.on_update_received(
            tensor_index=1, value=40.0, timestamp=t_kf2
        )
        await sm_demuxer.on_update_received(
            tensor_index=2, value=60.0, timestamp=t_kf2
        )
        # Keyframe state at t_kf2: [20,40,60]

        # Wait for interpolation to occur.
        # The interpolation worker sleeps for `smoothing_period / N` (e.g., 0.2s if N=5).
        # The first synthetic point is expected at t0 + smoothing_period (i.e., t0 + 1.0s).
        # To be safe, wait a bit longer than one smoothing period.
        await asyncio.sleep(
            sm_demuxer._smoothing_period_seconds + 0.5
        )  # Wait for T1 synthetic to be generated

        assert (
            len(fake_client.received_tensors) >= 1
        ), "Should have received at least one synthetic tensor"

        synthetic_tensor, synthetic_ts = fake_client.received_tensors[0]
        expected_synthetic_ts = t0 + datetime.timedelta(
            seconds=sm_demuxer._smoothing_period_seconds
        )

        assert (
            synthetic_ts == expected_synthetic_ts
        ), f"Expected synthetic timestamp {expected_synthetic_ts}, got {synthetic_ts}"

        # Expected tensor at t1 is halfway between [0,0,0] (at t0) and [20,40,60] (at t_kf2=t0+2s)
        # So, at t0+1s, it should be [10,20,30]
        expected_tensor_values = torch.tensor([10.0, 20.0, 30.0])
        assert torch.allclose(
            synthetic_tensor, expected_tensor_values
        ), f"Expected tensor {expected_tensor_values.tolist()}, got {synthetic_tensor.tolist()}"

        # For this specific setup, we expect only one synthetic data point to be generated
        # because the next one (at t0+2s) would coincide with a real keyframe or be beyond range.
        # The _interpolation_worker logic might try to emit at t0+2s, but if t1 (t0+1s) was the last emission,
        # next is t0+2s. If t0+2s is a real keyframe, it should skip or handle it.
        # The current _interpolation_worker logic: if next_emission_timestamp == t1 (a real keyframe's time),
        # it advances next_emission_timestamp by another period.
        # So if kf at t0, kf at t2, period 1s:
        # 1. last_emitted=None. next_emission = t0+1s. Interpolate for t0+1s. Set last_emitted = t0+1s.
        # 2. last_emitted=t0+1s. next_emission = t0+2s. This is t_kf2.
        #    Logic: `if t1 == next_emission_timestamp: next_emission_timestamp = t1 + period`.
        #    Here, t1 would be t_kf2. next_emission_timestamp becomes t_kf2 + 1s = t0+3s.
        #    This t0+3s is > last_kf_ts (t_kf2). So loop passes.
        # Thus, only one emission is correct.
        assert (
            len(fake_client.received_tensors) == 1
        ), "Expected only one synthetic tensor for this setup"

    @pytest.mark.asyncio
    async def test_no_interpolation_lt_two_keyframes(
        self, sm_demuxer: SmoothedTensorDemuxer, fake_client: FakeClient
    ):
        t0 = ts_at(0)
        # Only one keyframe
        await sm_demuxer.on_update_received(
            tensor_index=0, value=1.0, timestamp=t0
        )

        # Wait for a duration longer than the smoothing period
        await asyncio.sleep(
            sm_demuxer._smoothing_period_seconds * 2
        )  # e.g., 2 seconds

        assert (
            not fake_client.received_tensors
        ), "No synthetic tensors should be emitted with less than two keyframes"

    @pytest.mark.asyncio
    async def test_interpolation_multiple_points(
        self, sm_demuxer_custom: Any, fake_client: FakeClient
    ):
        # Use custom demuxer for specific timing control
        period = 0.5  # seconds
        tensor_len = 2
        demuxer = await sm_demuxer_custom(
            tensor_length=tensor_len, smoothing_period_seconds=period
        )

        t0 = ts_at(0)
        t_kf_end = ts_at(2.0)  # End keyframe is 2s after t0

        # Keyframe at t0: [0, 0]
        await demuxer.on_update_received(0, 0.0, t0)
        await demuxer.on_update_received(1, 0.0, t0)

        # Keyframe at t_kf_end: [20, 40]
        await demuxer.on_update_received(0, 20.0, t_kf_end)
        await demuxer.on_update_received(1, 40.0, t_kf_end)

        # Expected synthetic points at:
        # t0 + 0.5s = ts_at(0.5) -> [5, 10]
        # t0 + 1.0s = ts_at(1.0) -> [10, 20]
        # t0 + 1.5s = ts_at(1.5) -> [15, 30]
        # t0 + 2.0s is the real keyframe t_kf_end. Interpolation should stop before/at it.
        # The worker logic will attempt t0+2.0s, see it's a real keyframe, advance target to t0+2.5s, which is > t_kf_end.

        await asyncio.sleep(
            period * 5
        )  # Wait long enough for 3 points (1.5s) + buffer (1.0s) = 2.5s

        assert (
            len(fake_client.received_tensors) == 3
        ), "Expected 3 synthetic tensors"

        expected_timestamps = [ts_at(0.5), ts_at(1.0), ts_at(1.5)]
        expected_values = [
            torch.tensor([5.0, 10.0]),
            torch.tensor([10.0, 20.0]),
            torch.tensor([15.0, 30.0]),
        ]

        # Sort received by timestamp just in case of slight async reordering (unlikely for this)
        received_sorted = sorted(
            fake_client.received_tensors, key=lambda x: x[1]
        )

        for i in range(3):
            synth_tensor, synth_ts = received_sorted[i]
            assert (
                synth_ts == expected_timestamps[i]
            ), f"Point {i}: Timestamp mismatch"
            assert torch.allclose(
                synth_tensor, expected_values[i]
            ), f"Point {i}: Tensor value mismatch"

    @pytest.mark.asyncio
    async def test_cascading_recalculation_out_of_order_real_data(
        self, sm_demuxer_custom: Any, fake_client: FakeClient
    ):
        # Use a custom demuxer: tensor_length=2, smoothing_period_seconds=1.0
        # This makes values easier to track: T=[v0, v1]
        period = 1.0
        tensor_len = 2
        demuxer = await sm_demuxer_custom(
            tensor_length=tensor_len, smoothing_period_seconds=period
        )

        # Timestamps
        t0 = ts_at(0)  # Base time
        t1 = ts_at(1.0)  # For first synthetic point
        t2 = ts_at(
            2.0
        )  # For out-of-order real, OR second synthetic if no out-of-order
        t3 = ts_at(3.0)  # For third synthetic point
        t4 = ts_at(4.0)  # For second real keyframe

        # 1. Initial Setup: Real data for T0=[0,0] and T4=[40,400]
        await demuxer.on_update_received(0, 0.0, t0)  # T0_real = [0, ?]
        await demuxer.on_update_received(1, 0.0, t0)  # T0_real = [0, 0]

        await demuxer.on_update_received(0, 40.0, t4)  # T4_real = [40, ?]
        await demuxer.on_update_received(1, 400.0, t4)  # T4_real = [40, 400]

        # 2. Initial Smoothing: Allow time for synthetic T1, T2, T3 to be generated.
        # Expected:
        # T1_synth (at t1=1s): interpolated between T0=[0,0] and T4=[40,400] -> [10, 100]
        # T2_synth (at t2=2s): interpolated between T0=[0,0] and T4=[40,400] -> [20, 200]
        # T3_synth (at t3=3s): interpolated between T0=[0,0] and T4=[40,400] -> [30, 300]

        await asyncio.sleep(
            period * 3 + 0.5
        )  # Wait for T1, T2, T3 (3.0s) + buffer (0.5s)

        assert (
            len(fake_client.received_tensors) == 3
        ), "Expected 3 initial synthetic tensors"

        # Verify initial synthetic points (sort by timestamp for safety)
        received_initial_sorted = sorted(
            fake_client.received_tensors, key=lambda x: x[1]
        )

        # Check T1_synth
        assert received_initial_sorted[0][1] == t1
        assert torch.allclose(
            received_initial_sorted[0][0], torch.tensor([10.0, 100.0])
        )
        # Check T2_synth
        assert received_initial_sorted[1][1] == t2
        assert torch.allclose(
            received_initial_sorted[1][0], torch.tensor([20.0, 200.0])
        )
        # Check T3_synth
        assert received_initial_sorted[2][1] == t3
        assert torch.allclose(
            received_initial_sorted[2][0], torch.tensor([30.0, 300.0])
        )

        fake_client.clear()  # Clear received tensors before next phase

        # 3. Out-of-Order Keyframe: A new *real* data point arrives at T2_real=[99,0]
        # This T2_real should replace the previously synthetic T2.
        # And it should cause T3_synth to be recalculated.
        t2_real_val_idx0 = 99.0
        t2_real_val_idx1 = 0.0
        await demuxer.on_update_received(
            0, t2_real_val_idx0, t2
        )  # T2_real = [99,?]
        await demuxer.on_update_received(
            1, t2_real_val_idx1, t2
        )  # T2_real = [99,0]

        # When on_update_received processes (t2, [99,0]), it should:
        # - Store this new real keyframe.
        # - Reset demuxer._last_synthetic_emitted_at to t1 (timestamp of keyframe before t2).
        #   Or more precisely, to the timestamp of the keyframe just before the modified one.
        #   In this case, T0 is at t0. T2_real is new. The keyframe before T2_real is T0_real.
        #   So, _last_synthetic_emitted_at should effectively reset to t0, or None if t2 was first.
        #   The logic in on_update_received: `self._last_synthetic_emitted_at = self._keyframes[insertion_point - 1][0]`
        #   Keyframes: (t0, v0), (t2, v_new_real), (t4, v4). Insertion point for t2 is 1. So keyframes[0][0] = t0.
        #   This means the worker will next try to emit for t0 + period = t1.

        # 4. Cascading Recalculation:
        # Worker runs.
        # - It might try to re-emit T1_synth. If T1 is before _last_synthetic_emitted_at after reset, it proceeds.
        #   If _last_synthetic_emitted_at became t0:
        #   Next synthetic target is t0 + period = t1. Interpolation for t1 is now T0_real to T2_real.
        #   T1_recalc: between T0=[0,0] and T2_real=[99,0] -> at t1: [49.5, 0]

        # - Then it will try to emit T2_synth. Target t0 + 2*period = t2.
        #   This is a real keyframe T2_real. Worker should skip direct emission or advance.
        #   If it advances, next target is t2 + period = t3.

        # - Then it will try to emit T3_synth. Target t3.
        #   Interpolation for t3 is now T2_real=[99,0] and T4_real=[40,400].
        #   T3_recalc: at t3 (which is 1s after t2, and 1s before t4).
        #   So, halfway between T2_real=[99,0] and T4_real=[40,400].
        #   Value = [99,0] + ([40,400] - [99,0]) * 0.5
        #         = [99,0] + [-59, 400] * 0.5
        #         = [99,0] + [-29.5, 200] = [69.5, 200]

        # Wait for the recalculations.
        # The _last_synthetic_emitted_at was reset to t0 (effectively).
        # So, T1, T2(skip), T3 need to be processed. This is 3 periods.
        await asyncio.sleep(period * 3 + 0.5)

        assert (
            len(fake_client.received_tensors) >= 2
        ), "Expected at least T1_recalc and T3_recalc"

        recalc_sorted = sorted(
            fake_client.received_tensors, key=lambda x: x[1]
        )

        # Check T1_recalc
        # Interpolated between T0_real [0,0] (at t0) and T2_real [99,0] (at t2)
        # T1_recalc is at t1 (halfway)
        assert recalc_sorted[0][1] == t1
        assert torch.allclose(recalc_sorted[0][0], torch.tensor([49.5, 0.0]))

        # Check T3_recalc
        # Interpolated between T2_real [99,0] (at t2) and T4_real [40,400] (at t4)
        # T3_recalc is at t3 (halfway)
        assert recalc_sorted[1][1] == t3
        assert torch.allclose(recalc_sorted[1][0], torch.tensor([69.5, 200.0]))

        # Ensure no other points were emitted, especially not an old T2_synth or old T3_synth
        assert len(fake_client.received_tensors) == 2

    @pytest.mark.asyncio
    async def test_smoothing_period_zero(
        self, fake_client: FakeClient
    ):  # Removed sm_demuxer_custom as it's not used for direct __init__ test
        # With smoothing_period_seconds = 0, the behavior might be to pass through data
        # as fast as possible, or simply not interpolate and only emit real keyframes
        # if they were directly pushed (which they are not, on_update_received stores them).
        # The current _interpolation_worker sleeps for 0.1s if period is 0.
        # It will try to interpolate between available keyframes.
        # Let's define expected behavior: it should still interpolate, but the 'period'
        # for stepping from _last_synthetic_emitted_at is problematic.
        # The _interpolation_worker adds 0 to timestamps if period is 0.
        # This will cause it to try to emit at the same timestamp repeatedly.
        # This needs clarification or a specific design choice for period=0.
        #
        # Based on current code: `datetime.timedelta(seconds=0)` is valid.
        # `_interpolation_worker` will calculate:
        # `next_emission_timestamp = self._last_synthetic_emitted_at + datetime.timedelta(seconds=0)`
        # This means it will try to emit at `self._last_synthetic_emitted_at` over and over.
        # If `next_emission_timestamp == t1` (a real keyframe), it advances by period (0), so still t1.
        # This will likely lead to an infinite loop or rapid re-emission if not handled.
        #
        # The ValueError in __init__ for smoothing_period_seconds <= 0 prevents this.
        # So, this test should actually check for that ValueError.

        tensor_len = 2

        # Test __init__ directly to ensure ValueError is raised for period = 0
        with pytest.raises(
            ValueError, match="Smoothing period must be positive"
        ):
            direct_demuxer = SmoothedTensorDemuxer(
                client=fake_client,  # Need a client instance
                tensor_length=tensor_len,
                smoothing_period_seconds=0.0,
            )
            # await direct_demuxer.start() # Not reached
            # await direct_demuxer.close() # Not reached

        # Test __init__ for negative period
        with pytest.raises(
            ValueError, match="Smoothing period must be positive"
        ):
            direct_demuxer_neg = SmoothedTensorDemuxer(
                client=fake_client,
                tensor_length=tensor_len,
                smoothing_period_seconds=-1.0,
            )

    @pytest.mark.asyncio
    async def test_on_update_received_creates_new_keyframes(
        self, sm_demuxer: SmoothedTensorDemuxer
    ):
        # sm_demuxer fixture has tensor_length = 3
        t0 = ts_at(0)
        t1 = ts_at(1)

        await sm_demuxer.on_update_received(
            tensor_index=0, value=1.0, timestamp=t0
        )
        await sm_demuxer.on_update_received(
            tensor_index=1, value=2.0, timestamp=t0
        )

        # Check state of t0 before adding t1
        async with sm_demuxer._keyframe_lock:
            assert len(sm_demuxer._keyframes) == 1
            kf0_ts_check, kf0_tensor_check = sm_demuxer._keyframes[0]
            kf0_explicit_updates_check = sm_demuxer._keyframe_explicit_updates[
                0
            ]
            assert kf0_ts_check == t0
            assert torch.equal(kf0_tensor_check, torch.tensor([1.0, 2.0, 0.0]))
            assert sorted(kf0_explicit_updates_check) == sorted(
                [(0, 1.0), (1, 2.0)]
            )

        # Add a new keyframe at t1
        await sm_demuxer.on_update_received(
            tensor_index=0, value=10.0, timestamp=t1
        )

        async with sm_demuxer._keyframe_lock:
            assert len(sm_demuxer._keyframes) == 2
            assert len(sm_demuxer._keyframe_explicit_updates) == 2

            # Check first keyframe (t0) - should be unchanged
            kf0_ts, kf0_tensor = sm_demuxer._keyframes[0]
            kf0_updates = sm_demuxer._keyframe_explicit_updates[0]
            assert kf0_ts == t0
            assert torch.equal(kf0_tensor, torch.tensor([1.0, 2.0, 0.0]))
            assert sorted(kf0_updates) == sorted([(0, 1.0), (1, 2.0)])

            # Check second keyframe (t1)
            kf1_ts, kf1_tensor = sm_demuxer._keyframes[1]
            kf1_updates = sm_demuxer._keyframe_explicit_updates[1]
            assert kf1_ts == t1
            # kf1 should inherit from kf0, then apply its own update
            expected_kf1_tensor = (
                kf0_tensor.clone()
            )  # Starts as [1.0, 2.0, 0.0]
            expected_kf1_tensor[0] = 10.0  # Update for t1
            assert torch.equal(
                kf1_tensor, expected_kf1_tensor
            )  # Expected: [10.0, 2.0, 0.0]
            assert sorted(kf1_updates) == sorted([(0, 10.0)])

    @pytest.mark.asyncio
    async def test_on_update_received_updates_existing_keyframe(
        self, sm_demuxer: SmoothedTensorDemuxer
    ):
        t0 = ts_at(0)
        await sm_demuxer.on_update_received(
            tensor_index=0, value=1.0, timestamp=t0
        )
        await sm_demuxer.on_update_received(
            tensor_index=1, value=2.0, timestamp=t0
        )

        # Update existing timestamp t0
        await sm_demuxer.on_update_received(
            tensor_index=0, value=1.5, timestamp=t0
        )  # Change value for index 0
        await sm_demuxer.on_update_received(
            tensor_index=2, value=3.0, timestamp=t0
        )  # Add value for index 2

        async with sm_demuxer._keyframe_lock:
            assert (
                len(sm_demuxer._keyframes) == 1
            )  # Still only one keyframe timestamp
            assert len(sm_demuxer._keyframe_explicit_updates) == 1

            kf0_ts, kf0_tensor = sm_demuxer._keyframes[0]
            kf0_updates = sm_demuxer._keyframe_explicit_updates[0]

            assert kf0_ts == t0
            assert torch.equal(kf0_tensor, torch.tensor([1.5, 2.0, 3.0]))
            # Check that the explicit updates list reflects all changes
            assert sorted(kf0_updates) == sorted(
                [(0, 1.5), (1, 2.0), (2, 3.0)]
            )

    @pytest.mark.asyncio
    async def test_on_update_received_out_of_bounds_index(
        self, sm_demuxer: SmoothedTensorDemuxer, caplog
    ):
        # sm_demuxer has tensor_length = 3
        t0 = ts_at(0)
        # Index 3 is out of bounds for tensor_length 3 (valid indices 0, 1, 2)
        await sm_demuxer.on_update_received(
            tensor_index=3, value=100.0, timestamp=t0
        )

        async with sm_demuxer._keyframe_lock:
            assert not sm_demuxer._keyframes  # No keyframe should be created
            assert not sm_demuxer._keyframe_explicit_updates

        # Check for log message (implementation of on_update_received should log this)
        assert any(
            "Invalid tensor_index" in record.message
            and record.levelname in ["WARNING", "ERROR"]
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_on_update_received_internal_cascade(
        self, sm_demuxer: SmoothedTensorDemuxer
    ):
        # Test cascading update within _keyframes, similar to base TensorDemuxer
        # sm_demuxer has tensor_length = 3
        t0 = ts_at(0)
        t1 = ts_at(1)
        t2 = ts_at(2)

        # Keyframe at t0
        await sm_demuxer.on_update_received(
            tensor_index=0, value=1.0, timestamp=t0
        )  # [1,0,0]
        # Keyframe at t1, inherits from t0
        await sm_demuxer.on_update_received(
            tensor_index=1, value=2.0, timestamp=t1
        )  # t1 state: [1,2,0]
        # Keyframe at t2, inherits from t1
        await sm_demuxer.on_update_received(
            tensor_index=2, value=3.0, timestamp=t2
        )  # t2 state: [1,2,3]

        async with sm_demuxer._keyframe_lock:  # Initial state check
            assert torch.equal(
                sm_demuxer._keyframes[0][1], torch.tensor([1.0, 0.0, 0.0])
            )
            assert torch.equal(
                sm_demuxer._keyframes[1][1], torch.tensor([1.0, 2.0, 0.0])
            )
            assert torch.equal(
                sm_demuxer._keyframes[2][1], torch.tensor([1.0, 2.0, 3.0])
            )

        # Now, update t0 out-of-order, which should cascade through t1 and t2's states
        await sm_demuxer.on_update_received(
            tensor_index=0, value=10.0, timestamp=t0
        )  # t0 new state: [10,0,0]

        async with sm_demuxer._keyframe_lock:  # Check state after cascade
            # Check t0
            assert torch.equal(
                sm_demuxer._keyframes[0][1], torch.tensor([10.0, 0.0, 0.0])
            )
            # Check t1: should re-inherit from new t0, then apply its own explicit update (1, 2.0)
            # Expected t1: [10.0, 0.0, 0.0] (from new t0) -> apply (1, 2.0) -> [10.0, 2.0, 0.0]
            assert torch.equal(
                sm_demuxer._keyframes[1][1], torch.tensor([10.0, 2.0, 0.0])
            )
            # Check t2: should re-inherit from new t1, then apply its own explicit update (2, 3.0)
            # Expected t2: [10.0, 2.0, 0.0] (from new t1) -> apply (2, 3.0) -> [10.0, 2.0, 3.0]
            assert torch.equal(
                sm_demuxer._keyframes[2][1], torch.tensor([10.0, 2.0, 3.0])
            )
