# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for SmoothedTensorProvider."""

import asyncio
import datetime
from typing import List, Tuple

import torch
import pytest  # For pytest.mark.asyncio if not already standard

# Assuming SmoothedTensorProvider is in a module named 'smoothed_tensor_provider'
# and that the module is in the same directory or accessible via PYTHONPATH.
from tsercom.data.smoothed_tensor_provider import SmoothedTensorProvider

# Pytest mark for async functions
pytestmark = pytest.mark.asyncio


class FakeSmoothedTensorClient(SmoothedTensorProvider.Client):
    def __init__(self):
        self.received_tensors: List[Tuple[torch.Tensor, datetime.datetime]] = (
            []
        )
        self.call_count: int = 0

    async def on_smoothed_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """Stores received tensors and their timestamps."""
        self.received_tensors.append(
            (tensor.clone(), timestamp)
        )  # Clone tensor to store a snapshot
        self.call_count += 1

    def clear(self):
        """Clears the recorded data."""
        self.received_tensors = []
        self.call_count = 0


# Helper for creating specific timestamps for tests
def T(
    seconds_offset: float,
    base_time: datetime.datetime = datetime.datetime(2023, 1, 1, 0, 0, 0),
) -> datetime.datetime:
    return base_time + datetime.timedelta(seconds=seconds_offset)


async def test_example_scenario_interpolation(mocker):
    """Tests the primary example scenario from the issue description."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )

    await provider.on_tensor_changed(torch.tensor([4.0, 2.0, 1.0]), T(0))
    assert fake_client.call_count == 0  # No output yet

    await provider.on_tensor_changed(torch.tensor([0.0, 0.0, 0.0]), T(4))

    # Allow the interpolation task to run.
    # In a real scenario, asyncio.create_task schedules it. For testing,
    # we might need a small sleep if the task doesn't complete instantly
    # or if it yields control with asyncio.sleep(0) internally.
    # The _perform_interpolation has asyncio.sleep(0)
    await asyncio.sleep(0.01)  # Give time for the interpolation task to run

    assert fake_client.call_count == 3

    expected_tensors = [
        (torch.tensor([3.0, 1.5, 0.75]), T(1)),
        (torch.tensor([2.0, 1.0, 0.50]), T(2)),
        (torch.tensor([1.0, 0.5, 0.25]), T(3)),
    ]

    assert len(fake_client.received_tensors) == len(expected_tensors)
    for i, (received_tensor, received_ts) in enumerate(
        fake_client.received_tensors
    ):
        expected_tensor, expected_ts = expected_tensors[i]
        assert torch.allclose(
            received_tensor, expected_tensor, atol=1e-6
        ), f"Tensor {i} mismatch"
        assert received_ts == expected_ts, f"Timestamp {i} mismatch"

    await provider.cancel()  # Clean up task


async def test_no_interpolation_if_period_too_large():
    """Tests that no tensors are emitted if smoothing period is larger than the interval."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=5.0
    )

    await provider.on_tensor_changed(torch.tensor([1.0]), T(0))
    await provider.on_tensor_changed(
        torch.tensor([2.0]), T(4)
    )  # Interval is 4s, period is 5s
    await asyncio.sleep(0.01)

    assert fake_client.call_count == 0
    await provider.cancel()


async def test_first_data_point_no_output():
    """Tests that no output is generated after only one data point."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )

    await provider.on_tensor_changed(torch.tensor([1.0]), T(0))
    await asyncio.sleep(0.01)  # Should not trigger anything

    assert fake_client.call_count == 0
    await provider.cancel()


async def test_data_points_closer_than_smoothing_period():
    """Tests no interpolation if data points are closer than the smoothing period."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )

    await provider.on_tensor_changed(torch.tensor([1.0]), T(0))
    await provider.on_tensor_changed(
        torch.tensor([2.0]), T(0.5)
    )  # 0.5s interval, 1s period
    await asyncio.sleep(0.01)

    assert fake_client.call_count == 0
    await provider.cancel()


async def test_multiple_interpolation_segments():
    """Tests correct interpolation over multiple sequential data segments."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )

    # Segment 1: T(0) to T(2)
    await provider.on_tensor_changed(torch.tensor([0.0]), T(0))
    await provider.on_tensor_changed(torch.tensor([2.0]), T(2))
    await asyncio.sleep(0.01)  # Allow interpolation for segment 1

    assert fake_client.call_count == 1
    assert torch.allclose(
        fake_client.received_tensors[0][0], torch.tensor([1.0])
    )
    assert fake_client.received_tensors[0][1] == T(1)

    # Segment 2: T(2) to T(5)
    await provider.on_tensor_changed(torch.tensor([5.0]), T(5))
    await asyncio.sleep(0.01)  # Allow interpolation for segment 2

    # Total calls should be 1 (from seg1) + 2 (from seg2 for T(3), T(4)) = 3
    assert fake_client.call_count == 3

    # Check tensors from segment 2
    # expected: T(2) [2.0] , T(5) [5.0] -> T(3) [3.0], T(4) [4.0]
    assert torch.allclose(
        fake_client.received_tensors[1][0], torch.tensor([3.0])
    )  # (2 + (5-2)*(1/3))
    assert fake_client.received_tensors[1][1] == T(3)

    assert torch.allclose(
        fake_client.received_tensors[2][0], torch.tensor([4.0])
    )  # (2 + (5-2)*(2/3))
    assert fake_client.received_tensors[2][1] == T(4)

    await provider.cancel()


async def test_interpolation_task_cancellation_on_new_data(mocker):
    """Tests that an ongoing interpolation task is cancelled if new data arrives."""
    fake_client = FakeSmoothedTensorClient()
    # Use a longer smoothing period to make it easier to interrupt
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=0.1
    )

    # Mock asyncio.sleep to control the execution of _perform_interpolation
    # We want the first interpolation to start but not finish before the next point arrives.
    mocker.spy(asyncio, "sleep")

    # Send first two points to start interpolation
    # Segment 1: T(0) [0.0] to T(10.0) [100.0], period 0.1. Expect 99 points if uninterrupted.
    await provider.on_tensor_changed(torch.tensor([0.0]), T(0))
    await provider.on_tensor_changed(torch.tensor([100.0]), T(10.0)) # Increased duration and end value

    # Yield control once to allow the interpolation task to start and potentially run a bit.
    await asyncio.sleep(0.005)

    count_before_new_data = fake_client.call_count
    assert count_before_new_data > 0  # Make sure some interpolation happened
    assert count_before_new_data < 99  # Make sure it didn't complete fully (99 points for 10s duration)

    # Introduce new data to cancel the first segment's interpolation.
    # The new on_tensor_changed call will make (T(10.0), tensor([100.0])) the first point (ts1, t1)
    # for the new interpolation, and (T(10.5), tensor([105.0])) the second point (ts2, t2).
    await provider.on_tensor_changed(torch.tensor([105.0]), T(10.5))
    await asyncio.sleep(0.01)  # Let the new interpolation run for a bit.

    # Expected points from Segment 2 (T(10.0) to T(10.5), period 0.1): T(10.1), T(10.2), T(10.3), T(10.4). Values: 101, 102, 103, 104.
    expected_points_from_segment2 = 4
    assert fake_client.call_count == count_before_new_data + expected_points_from_segment2

    # Check the first tensor of the *second* segment.
    # Its index in received_tensors is count_before_new_data.
    # Value should be 100.0 + (105.0-100.0) * (0.1/0.5) = 100 + 5 * 0.2 = 101.0
    assert torch.allclose(fake_client.received_tensors[count_before_new_data][0], torch.tensor([101.0]))
    assert fake_client.received_tensors[count_before_new_data][1] == T(10.1)

    await provider.cancel()


async def test_constructor_invalid_smoothing_period():
    """Tests that the constructor raises ValueError for non-positive smoothing_period_seconds."""
    fake_client = FakeSmoothedTensorClient()
    with pytest.raises(
        ValueError, match="smoothing_period_seconds must be positive"
    ):
        SmoothedTensorProvider(
            client=fake_client, smoothing_period_seconds=0.0
        )
    with pytest.raises(
        ValueError, match="smoothing_period_seconds must be positive"
    ):
        SmoothedTensorProvider(
            client=fake_client, smoothing_period_seconds=-1.0
        )


async def test_timestamps_not_strictly_increasing():
    """Tests behavior when a new tensor arrives with a timestamp not after the previous one."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )

    await provider.on_tensor_changed(torch.tensor([1.0]), T(1))
    assert provider._last_timestamp_1 == T(
        1
    )  # Internal check for understanding

    await provider.on_tensor_changed(torch.tensor([2.0]), T(3))
    assert provider._last_timestamp_1 == T(1)
    assert provider._last_tensor_1 is not None and torch.allclose(
        provider._last_tensor_1, torch.tensor([1.0])
    )
    assert provider._last_timestamp_2 == T(3)
    assert provider._last_tensor_2 is not None and torch.allclose(
        provider._last_tensor_2, torch.tensor([2.0])
    )
    await asyncio.sleep(
        0.01
    )  # Let interpolation run for T(1) to T(3) -> T(2) [1.5]
    assert fake_client.call_count == 1
    assert torch.allclose(
        fake_client.received_tensors[0][0], torch.tensor([1.5])
    )
    assert fake_client.received_tensors[0][1] == T(2)

    fake_client.clear()

    # Send a point with timestamp equal to _last_timestamp_1 (which is T(3) now due to shift)
    # Current state: ts1=T(3), t1=[2.0] ; ts2=None, t2=None (after interpolation completed and state shifted)
    # No, this is wrong. After on_tensor_changed(T(3)), the state is:
    # _last_timestamp_1 = T(1), _last_tensor_1 = [1.0]
    # _last_timestamp_2 = T(3), _last_tensor_2 = [2.0]
    # The interpolation task for (T(1),[1.0]) and (T(3),[2.0]) is created.
    # Now, send a new point. This new point becomes the new _last_timestamp_2/_last_tensor_2.
    # The old _last_timestamp_2/_last_tensor_2 becomes the new _last_timestamp_1/_last_tensor_1.

    # So, before sending T(2.5):
    # provider._last_timestamp_1 == T(1)
    # provider._last_tensor_1 == tensor([1.0])
    # provider._last_timestamp_2 == T(3)
    # provider._last_tensor_2 == tensor([2.0])

    # Send a new point T(2.5) which is < T(3) (the current _last_timestamp_2)
    # This new point (T(2.5), [3.0]) becomes the new P2.
    # The old P2 (T(3), [2.0]) becomes the new P1.
    # So, P1=(T(3),[2.0]), P2=(T(2.5),[3.0]). This is non-monotonic for P1->P2.
    # The code should handle this by setting P1 to the new point and clearing P2.
    await provider.on_tensor_changed(torch.tensor([3.0]), T(2.5))
    await asyncio.sleep(0.01)

    # The existing interpolation (for T(1)-T(3)) should have been cancelled.
    # The new state should be P1=(T(2.5),[3.0]), P2=None.
    # So, no new interpolation should run.
    # Client was cleared, so call_count should be 0.
    assert fake_client.call_count == 0  # No new points after the T(2.5) update

    assert provider._last_timestamp_1 == T(2.5)
    assert torch.allclose(provider._last_tensor_1, torch.tensor([3.0]))
    assert provider._last_timestamp_2 is None
    assert provider._last_tensor_2 is None

    # Now send another point to start a new valid interpolation
    await provider.on_tensor_changed(
        torch.tensor([5.0]), T(4.5)
    )  # P1=T(2.5),[3.0], P2=T(4.5),[5.0]
    await asyncio.sleep(0.01)  # Interpolates for T(3.5) -> [4.0]

    assert fake_client.call_count == 1  # One new point from the new segment
    assert torch.allclose(
        fake_client.received_tensors[0][0], torch.tensor([4.0])  # Index 0 as it's the first after clear
    )
    assert fake_client.received_tensors[0][1] == T(3.5) # Index 0 for timestamp as well

    await provider.cancel()


async def test_interpolation_exact_end_time():
    """Tests scenario where last interpolation point is exactly at ts2."""
    fake_client = FakeSmoothedTensorClient()
    # ts1=0, ts2=2, period=1. Should emit at t=1. Should NOT emit at t=2.
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )
    await provider.on_tensor_changed(torch.tensor([0.0]), T(0))
    await provider.on_tensor_changed(torch.tensor([2.0]), T(2))
    await asyncio.sleep(0.01)

    assert fake_client.call_count == 1
    assert torch.allclose(
        fake_client.received_tensors[0][0], torch.tensor([1.0])
    )
    assert fake_client.received_tensors[0][1] == T(1)
    await provider.cancel()


async def test_zero_duration_between_points():
    """Tests scenario where ts1 and ts2 are identical."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=1.0
    )
    await provider.on_tensor_changed(torch.tensor([0.0]), T(0))
    await provider.on_tensor_changed(
        torch.tensor([2.0]), T(0)
    )  # Same timestamp
    await asyncio.sleep(0.01)

    assert fake_client.call_count == 0  # No interpolation for zero duration
    # Provider state should be: P1=T(0),[2.0], P2=None because new point T(0) is not > old P1 T(0)
    assert provider._last_timestamp_1 == T(0)
    assert provider._last_tensor_1 is not None and torch.allclose(
        provider._last_tensor_1, torch.tensor([2.0])
    )
    assert provider._last_timestamp_2 is None
    assert provider._last_tensor_2 is None
    await provider.cancel()


async def test_cancel_method_stops_interpolation(mocker):
    """Tests that the cancel method stops an ongoing interpolation."""
    fake_client = FakeSmoothedTensorClient()
    provider = SmoothedTensorProvider(
        client=fake_client, smoothing_period_seconds=0.01
    )  # Frequent updates

    # Spy on the client's on_smoothed_tensor
    on_smoothed_tensor_spy = mocker.spy(fake_client, "on_smoothed_tensor")

    await provider.on_tensor_changed(torch.tensor([0.0]), T(0))
    await provider.on_tensor_changed(
        torch.tensor([1.0]), T(1.0)
    )  # Should generate many points

    # Let interpolation run for a very short moment
    await asyncio.sleep(0.02)

    call_count_before_cancel = on_smoothed_tensor_spy.call_count
    assert call_count_before_cancel > 0  # Ensure some points were emitted

    await provider.cancel()  # Cancel the provider

    # Wait a bit more to ensure no more calls happen after cancellation
    await asyncio.sleep(0.05)

    assert (
        on_smoothed_tensor_spy.call_count == call_count_before_cancel
    )  # No new calls
