import asyncio
import datetime
import pytest
import torch
from typing import List, Tuple, Any, Optional  # Added Optional
from unittest.mock import AsyncMock  # Or consider pytest-mock's mocker fixture

from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # For the Client interface
from tsercom.data.smoothed_tensor_demuxer import SmoothedTensorDemuxer


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
    await asyncio.sleep(
        smoothing_period * 3.0  # Was 0.5, increased to 3.0
    )

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
        actual_tensor, actual_timestamp = fake_client.received_tensors[i]  # Corrected unpacking
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
        (t0, kf0_tensor_val), # Re-emitted KF0
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
        actual_tensor, actual_timestamp = fake_client.received_tensors[  # Corrected unpacking
            5 + i
        ]  # Check appended part

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


# TODO: Add more tests for functionality:
# - Test keyframe handling (_handle_real_keyframe) specifically for out-of-order updates and their impact on recalculation.
# - Test interpolation with more complex scenarios (e.g., >2 keyframes, varying time gaps).
# - Behavior when not enough keyframes (should not produce anything).
# - Correct timing of synthetic tensors when smoothing_period is very small or very large relative to keyframe gaps.
# - Test on_update_received more directly if possible, or ensure coverage through other tests.
# - Test cleanup of old keyframes if SmoothedTensorDemuxer implements its own logic beyond base class.
#   (Currently, it relies on base class for data timeout of *its* view of tensors, which are keyframes)
# - Test error conditions (e.g., invalid smoothing_period during init - already partly covered by init test).
# - Test the close() method more thoroughly (idempotency, behavior if called multiple times, interaction with ongoing processing).
# - Test scenario where a keyframe is updated (same timestamp, different tensor value).
# - Test scenario with only one keyframe (no interpolation should occur).
