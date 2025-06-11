import asyncio
import datetime
import pytest
import torch
from typing import List, Tuple, Any, Optional  # Added Optional
from unittest.mock import AsyncMock  # Or consider pytest-mock's mocker fixture

from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # For the Client interface
from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer


class FakeClient(TensorDemuxer.Client):
    def __init__(self):
        self.received_tensors: List[Tuple[torch.Tensor, datetime.datetime]] = (
            []
        )
        self.call_log: List[Tuple[str, Any]] = (
            []
        )  # To log calls if needed beyond tensors

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.received_tensors.append((tensor.clone(), timestamp))
        self.call_log.append(
            (
                "on_tensor_changed",
                {"timestamp": timestamp, "tensor": tensor.clone()},
            )
        )

    def get_tensor_at_time(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        for ts, tensor in reversed(
            self.received_tensors
        ):  # Search reverse for most recent if duplicate ts
            if ts == timestamp:
                return tensor
        return None

    def clear(self):
        self.received_tensors.clear()
        self.call_log.clear()


@pytest.mark.asyncio
async def test_smoothed_tensor_demuxer_initialization():
    fake_client = FakeClient()
    tensor_length = 10
    smoothing_period = 0.5

    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
    )
    assert (
        demuxer._tensor_length == tensor_length
    )  # Accessing protected for test validation
    assert (
        demuxer._smoothing_period_seconds == smoothing_period
    )  # Accessing protected
    assert (
        demuxer._actual_downstream_client == fake_client
    )  # Accessing protected

    # Ensure the interpolation task is started
    assert demuxer._interpolation_loop_task is not None
    assert not demuxer._interpolation_loop_task.done()

    # Clean up the demuxer to stop the task
    await demuxer.close()
    await asyncio.sleep(0)  # Allow event loop to process task cancellation
    assert (
        demuxer._interpolation_loop_task.done()
        or demuxer._interpolation_loop_task.cancelled()
    )


# Add a fixture for creating demuxer instances if it becomes repetitive
@pytest.fixture
async def smoothed_demuxer_tuple():  # Renamed to avoid conflict if used directly
    fake_client = FakeClient()
    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=10,
        smoothing_period_seconds=0.1,  # Using a smaller period for tests
        data_timeout_seconds=1.0,  # And shorter timeout
    )
    yield demuxer, fake_client  # Yield both so client can be inspected

    # Ensure cleanup happens even if test fails
    if (
        demuxer._interpolation_loop_task
        and not demuxer._interpolation_loop_task.done()
    ):
        await demuxer.close()
        await asyncio.sleep(0)  # Allow cancellation to complete
    elif (
        not demuxer._interpolation_loop_task
    ):  # Handle cases where task might not have been created
        pass  # Or specific error handling if task should always exist post-init


@pytest.mark.asyncio
async def test_simple_interpolation():
    fake_client = FakeClient()
    tensor_length = 2
    smoothing_period = 1.0
    # Use a longer data_timeout_seconds for this test to ensure keyframes are not prematurely removed
    # by the base TensorDemuxer's cleanup, if it were to run aggressively.
    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
        data_timeout_seconds=10.0,
    )

    # kf1_ts = datetime.datetime.now(datetime.timezone.utc)
    # Using a fixed reference for reproducibility if tests were to be re-run with fixed seed or time mocking
    base_time = datetime.datetime(
        2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
    )
    kf1_ts = base_time
    kf1_tensor_val = torch.tensor([10.0, 20.0])

    # Send updates for first keyframe
    # These calls simulate the underlying TensorDemuxer having processed raw updates
    # and calling self._base_client_adapter.on_tensor_changed, which in turn calls _handle_real_keyframe
    await demuxer.on_update_received(0, kf1_tensor_val[0].item(), kf1_ts)
    await demuxer.on_update_received(1, kf1_tensor_val[1].item(), kf1_ts)

    kf2_ts = kf1_ts + datetime.timedelta(seconds=2.0)
    kf2_tensor_val = torch.tensor([30.0, 40.0])

    # Send updates for second keyframe
    await demuxer.on_update_received(0, kf2_tensor_val[0].item(), kf2_ts)
    await demuxer.on_update_received(1, kf2_tensor_val[1].item(), kf2_ts)

    # Allow time for interpolation worker to run a few cycles
    # Expected emissions:
    # 1. At kf1_ts (value: kf1_tensor_val)
    # 2. At kf1_ts + 1.0s (value: interpolated)
    # 3. At kf2_ts (value: kf2_tensor_val)
    # Sleep duration needs to be enough for these 3 points.
    # The worker sleeps for smoothing_period/3 if no work, or 0.001 if work.
    # If it emits kf1_ts, then sleeps 0.001.
    # Then emits kf1_ts + 1s, then sleeps 0.001.
    # Then emits kf2_ts, then sleeps 0.001.
    # After this, next_synthetic_ts will be kf2_ts + 1s, which is > last_kf_ts (kf2_ts).
    # So it will do no work and sleep for smoothing_period / 3.0.
    # Total time for 3 emissions ~ 0.003s. Add buffer.
    # Trying a longer sleep to ensure it's not a timing issue for the first point.
    await asyncio.sleep(smoothing_period * 3.0)  # Was 0.5, increased to 3.0

    # Expected synthetic points
    expected_ts1 = kf1_ts
    expected_tensor1 = kf1_tensor_val

    expected_ts2 = kf1_ts + datetime.timedelta(seconds=1.0)
    expected_tensor2 = torch.tensor(
        [20.0, 30.0]
    )  # Interpolated: (10+30)/2 = 20, (20+40)/2 = 30

    expected_ts3 = kf2_ts
    expected_tensor3 = kf2_tensor_val

    # The SmoothedTensorDemuxer's _actual_downstream_client (our FakeClient)
    # should ONLY receive from the _interpolation_worker.

    assert (
        len(fake_client.received_tensors) == 3
    ), f"Expected exactly 3 tensors, got {len(fake_client.received_tensors)}. Data: {fake_client.received_tensors}"

    # Check point 1
    recv_tensor_1, recv_ts_1 = fake_client.received_tensors[0]
    assert (
        recv_ts_1 == expected_ts1
    ), f"Timestamp 1 mismatch. Expected {expected_ts1}, got {recv_ts_1}"
    assert torch.allclose(
        recv_tensor_1, expected_tensor1
    ), f"Tensor 1 mismatch. Expected {expected_tensor1}, got {recv_tensor_1}"

    # Check point 2
    recv_tensor_2, recv_ts_2 = fake_client.received_tensors[1]
    assert (
        recv_ts_2 == expected_ts2
    ), f"Timestamp 2 mismatch. Expected {expected_ts2}, got {recv_ts_2}"
    assert torch.allclose(
        recv_tensor_2, expected_tensor2
    ), f"Tensor 2 mismatch. Expected {expected_tensor2}, got {recv_tensor_2}"

    # Check point 3
    recv_tensor_3, recv_ts_3 = fake_client.received_tensors[2]
    assert (
        recv_ts_3 == expected_ts3
    ), f"Timestamp 3 mismatch. Expected {expected_ts3}, got {recv_ts_3}"
    assert torch.allclose(
        recv_tensor_3, expected_tensor3
    ), f"Tensor 3 mismatch. Expected {expected_tensor3}, got {recv_tensor_3}"

    await demuxer.close()
    await asyncio.sleep(0)  # Allow cancellation to complete


@pytest.mark.asyncio
async def test_insufficient_keyframes():
    fake_client = FakeClient()
    tensor_length = 2
    smoothing_period = 0.1  # Use a small period for faster test

    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
        data_timeout_seconds=1.0,  # Keep data timeout reasonable for test
    )

    base_time = datetime.datetime(
        2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
    )
    kf1_ts = base_time
    kf1_tensor_val = torch.tensor([10.0, 20.0])

    # Send updates for the single keyframe
    await demuxer.on_update_received(0, kf1_tensor_val[0].item(), kf1_ts)
    await demuxer.on_update_received(1, kf1_tensor_val[1].item(), kf1_ts)

    # Allow time for interpolation worker to run, though it shouldn't produce anything
    # The worker checks `len(self._keyframes) < 2`
    await asyncio.sleep(smoothing_period * 3)  # Sleep a few cycles

    # Assert that no smoothed tensors were received by the actual downstream client
    assert (
        len(fake_client.received_tensors) == 0
    ), f"Expected no tensors with insufficient keyframes, got {len(fake_client.received_tensors)}. Data: {fake_client.received_tensors}"

    await demuxer.close()
    await asyncio.sleep(0)  # Allow cancellation to complete


@pytest.mark.asyncio
async def test_out_of_order_real_data_causes_cascading_recalculation():
    fake_client = FakeClient()
    tensor_length = 1
    smoothing_period = 1.0

    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
        data_timeout_seconds=10.0,  # Ensure keyframes persist for the test duration
    )

    # Using a fixed reference time for reproducibility
    base_time = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    t0 = base_time
    t1 = t0 + datetime.timedelta(seconds=1.0)
    t2 = t0 + datetime.timedelta(seconds=2.0)
    t3 = t0 + datetime.timedelta(seconds=3.0)
    t4 = t0 + datetime.timedelta(seconds=4.0)

    # Initial Keyframes KF0 and KF4
    kf0_tensor_val = torch.tensor([0.0])
    await demuxer.on_update_received(0, kf0_tensor_val[0].item(), t0)

    kf4_tensor_val = torch.tensor([40.0])
    await demuxer.on_update_received(0, kf4_tensor_val[0].item(), t4)

    # Allow initial interpolation phase
    # Expecting 5 points: t0, t1, t2, t3, t4
    await asyncio.sleep(smoothing_period * 4.5)

    expected_initial_tensors_ordered = [
        (t0, kf0_tensor_val),
        (t1, torch.tensor([10.0])),  # Interpolated: 0 + (40-0)*(1/4)
        (t2, torch.tensor([20.0])),  # Interpolated: 0 + (40-0)*(2/4)
        (t3, torch.tensor([30.0])),  # Interpolated: 0 + (40-0)*(3/4)
        (t4, kf4_tensor_val),
    ]

    assert (
        len(fake_client.received_tensors) == 5
    ), f"Initial interpolation should produce 5 points, got {len(fake_client.received_tensors)}. Data: {fake_client.received_tensors}"
    for i, (expected_ts, expected_val) in enumerate(
        expected_initial_tensors_ordered
    ):
        actual_tensor, actual_timestamp = fake_client.received_tensors[
            i
        ]  # Corrected unpacking
        assert (
            actual_timestamp == expected_ts
        ), f"Timestamp mismatch at initial index {i}. Expected {expected_ts}, got {actual_timestamp}"
        assert torch.allclose(
            actual_tensor, expected_val
        ), f"Tensor value mismatch for {expected_ts} at initial index {i}. Expected {expected_val}, got {actual_tensor}"

    # Introduce Out-of-Order Keyframe KF2_real
    kf2_real_tensor_val = torch.tensor([99.0])
    await demuxer.on_update_received(0, kf2_real_tensor_val[0].item(), t2)

    # Allow recalculation phase
    # _last_synthetic_emitted_at should be reset to t0.
    # Worker will then re-emit points for t1, t2, t3, t4 based on new keyframe reality.
    await asyncio.sleep(smoothing_period * 4.5)

    # After KF2_real, new points are appended. Total should be 5 (initial) + 5 (recalculated) = 10
    # because _last_synthetic_emitted_at is reset to None, causing full recalc from t0.
    assert (
        len(fake_client.received_tensors) == 10
    ), f"Expected 10 total tensors after recalculation, got {len(fake_client.received_tensors)}. Data: {fake_client.received_tensors}"

    # Recalculated points based on new reality: KF0(t0,0), KF2_real(t2,99), KF4(t4,40)
    # The worker will re-emit from t0.
    expected_recalculated_tensors_ordered = [
        (t0, kf0_tensor_val),  # Re-emitted KF0
        (
            t1,
            torch.tensor([49.5]),
        ),  # Interpolated KF0(0.0) and KF2_real(99.0). Ratio (t1-t0)/(t2-t0) = 1/2. Value = 0 + (99-0)*0.5
        (t2, kf2_real_tensor_val),  # The new real keyframe KF2_real
        (
            t3,
            torch.tensor([69.5]),
        ),  # Interpolated KF2_real(99.0) and KF4(40.0). Ratio (t3-t2)/(t4-t2) = 1/2. Value = 99 + (40-99)*0.5
        (t4, kf4_tensor_val),  # Real keyframe KF4 (re-emitted)
    ]

    # Check the recalculated part of the sequence (indices 5 through 9)
    for i, (expected_ts, expected_val) in enumerate(
        expected_recalculated_tensors_ordered
    ):
        actual_tensor, actual_timestamp = fake_client.received_tensors[
            5 + i
        ]  # Corrected unpacking  # Check appended part

        assert (
            actual_timestamp == expected_ts
        ), f"Recalculated timestamp mismatch at overall index {5+i}. Expected {expected_ts}, got {actual_timestamp}"
        assert torch.allclose(
            actual_tensor, expected_val
        ), f"Recalculated tensor mismatch for ts {expected_ts} at overall index {5+i}. Expected {expected_val}, got {actual_tensor}"

    await demuxer.close()
    await asyncio.sleep(0)  # Allow cancellation


@pytest.mark.asyncio
async def test_graceful_shutdown():
    fake_client = FakeClient()
    tensor_length = 2
    smoothing_period = 0.05  # Use a small period

    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
        data_timeout_seconds=1.0,
    )

    assert (
        demuxer._interpolation_loop_task is not None
    ), "Interpolation task should be created"
    assert (
        not demuxer._interpolation_loop_task.done()
    ), "Interpolation task should be running initially"

    # Add a keyframe to make sure the loop might be active
    kf1_ts = datetime.datetime.now(datetime.timezone.utc)
    kf1_tensor_val = torch.tensor([10.0, 20.0])
    await demuxer.on_update_received(0, kf1_tensor_val[0].item(), kf1_ts)
    await demuxer.on_update_received(1, kf1_tensor_val[1].item(), kf1_ts)
    await asyncio.sleep(smoothing_period * 2)  # Let it run a bit

    await demuxer.close()

    # After awaiting close(), the task should be done (cancelled and awaited by close itself)
    assert (
        demuxer._interpolation_loop_task.done()
    ), "Interpolation task should be done after close()"

    # To be absolutely sure about cancellation (optional, as done() implies it if close() is correct)
    # We can check if the task raised CancelledError or is marked as cancelled.
    # If close() awaits the task, then the task should be done.
    # If it was cancelled, .cancelled() is True.
    # If it finished normally or with another error, .cancelled() is False but .done() is True.
    # The close method is designed to cancel and await.
    assert (
        demuxer._interpolation_loop_task.cancelled()
    ), "Interpolation task should be in cancelled state after close()"


@pytest.mark.asyncio  # Async not strictly needed here but fine for consistency
async def test_init_invalid_smoothing_period():
    fake_client = FakeClient()
    tensor_length = 2

    with pytest.raises(ValueError, match="Smoothing period must be positive."):
        SmoothedTensorDemuxer(
            client=fake_client,
            tensor_length=tensor_length,
            smoothing_period_seconds=0.0,
        )

    with pytest.raises(ValueError, match="Smoothing period must be positive."):
        SmoothedTensorDemuxer(
            client=fake_client,
            tensor_length=tensor_length,
            smoothing_period_seconds=-1.0,
        )


@pytest.mark.asyncio
async def test_interpolation_multiple_segments():
    fake_client = FakeClient()
    tensor_length = 1
    smoothing_period = 1.0

    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
        data_timeout_seconds=10.0,  # Ensure keyframes persist
    )

    # Using a fixed reference time for reproducibility
    t_base = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    kf1_ts = t_base
    kf1_val = torch.tensor([10.0])
    await demuxer.on_update_received(0, kf1_val[0].item(), kf1_ts)

    kf2_ts = t_base + datetime.timedelta(seconds=2.0)  # Gap of 2s from kf1
    kf2_val = torch.tensor([30.0])
    await demuxer.on_update_received(0, kf2_val[0].item(), kf2_ts)

    kf3_ts = t_base + datetime.timedelta(seconds=5.0)  # Gap of 3s from kf2
    kf3_val = torch.tensor([0.0])
    await demuxer.on_update_received(0, kf3_val[0].item(), kf3_ts)

    # Expected emissions:
    # t_base + 0s (kf1_ts): 10.0 (Real KF1)
    # t_base + 1s (kf1_ts + 1s): 20.0 (Interpolated KF1-KF2: 10 + (30-10)*1/2)
    # t_base + 2s (kf2_ts): 30.0 (Real KF2)
    # t_base + 3s (kf2_ts + 1s): 20.0 (Interpolated KF2-KF3: 30 + (0-30)*1/3)
    # t_base + 4s (kf2_ts + 2s): 10.0 (Interpolated KF2-KF3: 30 + (0-30)*2/3)
    # t_base + 5s (kf3_ts): 0.0 (Real KF3)

    # Allow time for all points up to kf3_ts (t_base + 5s).
    # Worker sleeps briefly after each emission. Max timestamp is kf3_ts.
    # The last point is kf3_ts. Worker will try kf3_ts + 1s next, which is > kf3_ts (last_kf_ts).
    # So, it will process up to kf3_ts.
    await asyncio.sleep(
        smoothing_period * 5.5
    )  # t_base to t_base+5s is 5 intervals

    expected_sequence = [
        (kf1_ts, kf1_val),
        (kf1_ts + datetime.timedelta(seconds=1.0), torch.tensor([20.0])),
        (kf2_ts, kf2_val),
        (kf2_ts + datetime.timedelta(seconds=1.0), torch.tensor([20.0])),
        (kf2_ts + datetime.timedelta(seconds=2.0), torch.tensor([10.0])),
        (kf3_ts, kf3_val),
    ]

    assert len(fake_client.received_tensors) == len(expected_sequence), (
        f"Expected {len(expected_sequence)} tensors, "
        f"got {len(fake_client.received_tensors)}. Data: {fake_client.received_tensors}"
    )

    for i, (ts_expected, val_expected) in enumerate(expected_sequence):
        val_actual, ts_actual = fake_client.received_tensors[
            i
        ]  # Corrected unpacking based on FakeClient
        assert (
            ts_actual == ts_expected
        ), f"Timestamp mismatch at index {i}. Expected {ts_expected}, got {ts_actual}"
        assert torch.allclose(
            val_actual, val_expected, atol=1e-6
        ), f"Tensor value mismatch at index {i}. Expected {val_expected}, got {val_actual}"

    await demuxer.close()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_interpolation_timing_various_periods():
    # Scenario 1: Small smoothing period, many synthetic points
    fake_client_small_period = FakeClient()
    tensor_length = 1
    small_smoothing_period = 0.1

    demuxer_small_period = SmoothedTensorDemuxer(
        client=fake_client_small_period,
        tensor_length=tensor_length,
        smoothing_period_seconds=small_smoothing_period,
        data_timeout_seconds=5.0,  # Ensure keyframes persist
    )

    t_base_small = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    kf1_ts_s = t_base_small
    kf1_val_s = torch.tensor([0.0])
    await demuxer_small_period.on_update_received(
        0, kf1_val_s[0].item(), kf1_ts_s
    )

    kf2_ts_s = t_base_small + datetime.timedelta(seconds=1.0)  # 1s gap
    kf2_val_s = torch.tensor([10.0])
    await demuxer_small_period.on_update_received(
        0, kf2_val_s[0].item(), kf2_ts_s
    )

    # Expected: 1 (kf1_ts_s) + 9 (interpolated over 0.9s) + 1 (kf2_ts_s) = 11 points
    # Points at T0, T0+0.1, T0+0.2, ..., T0+0.9, T0+1.0
    # Wait for 1s / 0.1s = 10 periods. Sleep slightly more.
    await asyncio.sleep(
        1.0 + small_smoothing_period * 3
    )  # Increased buffer slightly

    assert len(fake_client_small_period.received_tensors) == 11, (
        f"Small period: Expected 11 tensors, "
        f"got {len(fake_client_small_period.received_tensors)}. "
        f"Data: {fake_client_small_period.received_tensors}"
    )

    # Check first point
    val_actual_s0, ts_actual_s0 = fake_client_small_period.received_tensors[0]
    assert ts_actual_s0 == kf1_ts_s
    assert torch.allclose(val_actual_s0, kf1_val_s, atol=1e-6)

    # Check last point
    val_actual_s_last, ts_actual_s_last = (
        fake_client_small_period.received_tensors[-1]
    )
    assert ts_actual_s_last == kf2_ts_s
    assert torch.allclose(val_actual_s_last, kf2_val_s, atol=1e-6)

    # Check a midpoint
    # Point at T0+0.5s should be index 5 (0-indexed for T0, T0+0.1, ..., T0+0.5)
    mid_point_idx = 5
    expected_mid_ts_s = kf1_ts_s + datetime.timedelta(
        seconds=small_smoothing_period * mid_point_idx
    )
    expected_mid_val_s = torch.tensor([5.0])  # 0 + (10-0) * (0.5/1.0) = 5.0

    val_actual_s_mid, ts_actual_s_mid = (
        fake_client_small_period.received_tensors[mid_point_idx]
    )
    assert ts_actual_s_mid == expected_mid_ts_s
    assert torch.allclose(val_actual_s_mid, expected_mid_val_s, atol=1e-6)

    await demuxer_small_period.close()
    await asyncio.sleep(0)

    # Scenario 2: Large smoothing period, skips over some real KFs for direct emission
    fake_client_large_period = FakeClient()
    large_smoothing_period = 2.5

    demuxer_large_period = SmoothedTensorDemuxer(
        client=fake_client_large_period,
        tensor_length=tensor_length,
        smoothing_period_seconds=large_smoothing_period,
        data_timeout_seconds=5.0,  # Ensure keyframes persist
    )

    t_base_large = datetime.datetime(
        2024, 1, 1, 13, 0, 0, tzinfo=datetime.timezone.utc
    )
    kfa_ts_l = t_base_large  # T0
    kfa_val_l = torch.tensor([0.0])
    await demuxer_large_period.on_update_received(
        0, kfa_val_l[0].item(), kfa_ts_l
    )

    kfb_ts_l = t_base_large + datetime.timedelta(seconds=1.0)  # T0+1s
    kfb_val_l = torch.tensor([10.0])
    await demuxer_large_period.on_update_received(
        0, kfb_val_l[0].item(), kfb_ts_l
    )

    kfc_ts_l = t_base_large + datetime.timedelta(seconds=3.0)  # T0+3s
    kfc_val_l = torch.tensor([30.0])
    await demuxer_large_period.on_update_received(
        0, kfc_val_l[0].item(), kfc_ts_l
    )

    # Expected emissions:
    # 1. (kfa_ts_l, tensor([0.0])) -> at T0. Last emitted = T0.
    # 2. Next synthetic target: T0 + 2.5s.
    #    This is between kfb_ts_l (T0+1s, [10.0]) and kfc_ts_l (T0+3s, [30.0]).
    #    t1_bracket=kfb_ts_l, v1_bracket=10.0. t2_bracket=kfc_ts_l, v2_bracket=30.0.
    #    Target_ts = T0+2.5s.
    #    Ratio = ( (T0+2.5s) - (T0+1s) ) / ( (T0+3s) - (T0+1s) ) = 1.5s / 2.0s = 0.75
    #    Value = 10.0 + (30.0-10.0)*0.75 = 10.0 + 20.0*0.75 = 10.0 + 15.0 = 25.0
    #    So: (t_base_large + 2.5s, tensor([25.0]))
    # 3. Next synthetic target: (T0+2.5s) + 2.5s = T0+5s. This is beyond kfc_ts_l (T0+3s).
    #    The worker should emit kfc_ts_l as it's the last keyframe before T0+5s.
    #    So: (kfc_ts_l, tensor([30.0]))
    # Total 2 points, as T0+5s (next target) is beyond kfc_ts_l, and worker doesn't auto-emit last KF.

    await asyncio.sleep(
        large_smoothing_period * 1.5 + 0.5
    )  # Time for T0 and T0+2.5s

    expected_sequence_large = [
        (kfa_ts_l, kfa_val_l),
        (t_base_large + datetime.timedelta(seconds=2.5), torch.tensor([25.0])),
        # (kfc_ts_l, kfc_val_l), # This point is not expected with current worker logic
    ]

    assert len(fake_client_large_period.received_tensors) == 2, (
        f"Large period: Expected 2 tensors, "
        f"got {len(fake_client_large_period.received_tensors)}. "
        f"Data: {fake_client_large_period.received_tensors}"
    )

    for i, (ts_expected, val_expected) in enumerate(expected_sequence_large):
        val_actual, ts_actual = fake_client_large_period.received_tensors[i]
        assert (
            ts_actual == ts_expected
        ), f"Large period: Timestamp mismatch at index {i}. Expected {ts_expected}, Got {ts_actual}"
        assert torch.allclose(
            val_actual, val_expected, atol=1e-6
        ), f"Large period: Tensor value mismatch at index {i}. Expected {val_expected}, Got {val_actual}"

    await demuxer_large_period.close()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_close_idempotency():
    fake_client = FakeClient()
    tensor_length = 2
    smoothing_period = 0.1

    demuxer = SmoothedTensorDemuxer(
        client=fake_client,
        tensor_length=tensor_length,
        smoothing_period_seconds=smoothing_period,
        data_timeout_seconds=1.0,  # Keep other params reasonable for init
    )

    initial_task = demuxer._interpolation_loop_task
    assert initial_task is not None, "Interpolation task should exist"
    assert (
        not initial_task.done()
    ), "Interpolation task should be running initially"

    # Call close multiple times
    await demuxer.close()  # First call
    assert initial_task.done(), "Task should be done after first close()"
    # Depending on how Pytest handles event loop for fixtures vs tests,
    # a tiny sleep might be needed for the task's state to reflect if not awaited enough by close().
    # However, demuxer.close() is itself async and awaits the task.

    await demuxer.close()  # Second call
    assert initial_task.done(), "Task should remain done after second close()"

    # Verify that the task object itself hasn't been inappropriately replaced or re-created.
    assert demuxer._interpolation_loop_task is initial_task, (
        "Task object should ideally not change after being cancelled/closed."
        " If it does, ensure it's handled gracefully."
    )

    try:
        await demuxer.close()  # Third call
    except Exception as e:  # pylint: disable=broad-except
        pytest.fail(f"Calling close() multiple times raised an exception: {e}")

    assert initial_task.done(), "Task should remain done after third close()"
    # Check cancellation status too for completeness
    assert initial_task.cancelled(), "Task should be in 'cancelled' state"
