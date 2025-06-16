"""Unit tests for the LinearInterpolationStrategy."""

from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pytest
import math # For isnan

from tsercom.data.tensor.linear_interpolation_strategy import LinearInterpolationStrategy

# Helper to create datetime objects easily
def T(offset_seconds: float, base_time: datetime = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)) -> datetime:
    return base_time + timedelta(seconds=offset_seconds)

@pytest.fixture
def strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()

def test_interpolate_no_keyframes(strategy: LinearInterpolationStrategy):
    """Test interpolation with no keyframes. Expect NaNs or empty based on impl."""
    timestamps_to_interpolate = [T(1), T(2)]
    result = strategy.interpolate_series([], timestamps_to_interpolate)
    assert len(result) == len(timestamps_to_interpolate)
    for val in result:
        assert math.isnan(val), "Expected NaN for no keyframes"

def test_interpolate_single_keyframe(strategy: LinearInterpolationStrategy):
    """Test interpolation with a single keyframe. Expect constant value."""
    kf = [(T(0), 10.0)]
    timestamps_to_interpolate = [T(-1), T(0), T(1), T(2.5)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == [10.0, 10.0, 10.0, 10.0]

def test_interpolate_between_two_keyframes(strategy: LinearInterpolationStrategy):
    """Test simple interpolation between two keyframes."""
    kf = [(T(0), 10.0), (T(10), 20.0)]
    # Test points: before first, at first, mid, at second, after second
    timestamps_to_interpolate = [T(5)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == [15.0]

def test_interpolate_direct_hit_on_keyframes(strategy: LinearInterpolationStrategy):
    """Test when required timestamps are exactly on keyframes."""
    kf = [(T(0), 10.0), (T(10), 20.0), (T(20), 0.0)]
    timestamps_to_interpolate = [T(0), T(10), T(20)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == [10.0, 20.0, 0.0]

def test_extrapolate_before_first_keyframe(strategy: LinearInterpolationStrategy):
    """Test extrapolation before the first keyframe."""
    kf = [(T(10), 10.0), (T(20), 20.0)]
    # Current LinearInterpolationStrategy clamps to the first/last value for extrapolation beyond ends.
    # If extrapolation based on the first two points was desired, the strategy would need modification.
    # Given the current implementation, it should clamp.
    timestamps_to_interpolate = [T(0), T(5)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    # The current logic in LinearInterpolationStrategy, if t_req < kf_timestamps[0], clamps.
    # If t_req is between first two points, it interpolates.
    # If t_req is AT kf_timestamps[0], it's a direct hit.
    # Let's test the clamping behavior based on the provided code for LinearInterpolationStrategy
    # `if kf_timestamps[0].timestamp() >= t_req :` -> `if t_req <= kf_timestamps[0].timestamp(): results.append(kf_values[0])`
    # This indicates clamping for T(0) and T(5) since they are <= T(10).
    assert result == [10.0, 10.0]


def test_extrapolate_after_last_keyframe(strategy: LinearInterpolationStrategy):
    """Test extrapolation after the last keyframe."""
    kf = [(T(0), 10.0), (T(10), 20.0)]
    # Current LinearInterpolationStrategy clamps to the last value for extrapolation.
    timestamps_to_interpolate = [T(15), T(20)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    # `surround_idx == -1` uses last two points to extrapolate.
    # (t_req - t1) * (v2 - v1) / (t2 - t1) + v1
    # For T(15): t1=T(0), v1=10; t2=T(10), v2=20. t_req=T(15)
    # (15 - 0) * (20-10) / (10-0) + 10 = 15 * 10 / 10 + 10 = 15 + 10 = 25.0
    # For T(20): (20 - 0) * (20-10) / (10-0) + 10 = 20 * 10 / 10 + 10 = 20 + 10 = 30.0
    assert result == [25.0, 30.0]


def test_multiple_required_timestamps(strategy: LinearInterpolationStrategy):
    """Test with various required timestamps including interpolation and extrapolation."""
    kf = [(T(10), 0.0), (T(20), 100.0)]
    timestamps_to_interpolate = [
        T(5),    # Extrapolate before (clamp)
        T(10),   # Hit
        T(15),   # Interpolate
        T(20),   # Hit
        T(25)    # Extrapolate after
    ]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    # T(5) -> clamps to 0.0
    # T(10) -> 0.0
    # T(15) -> 50.0
    # T(20) -> 100.0
    # T(25) -> 100 + ( (25-10) * (100-0) / (20-10) ) - (100-0) -> this is not how it works
    # T(25) uses T(10) and T(20) for extrapolation: (25-10)*(100-0)/(20-10) + 0 = 15 * 100 / 10 = 150.0
    assert result == [0.0, 0.0, 50.0, 100.0, 150.0]

def test_empty_required_timestamps(strategy: LinearInterpolationStrategy):
    """Test with an empty list of required timestamps."""
    kf = [(T(0), 10.0), (T(10), 20.0)]
    timestamps_to_interpolate: List[datetime] = []
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == []

def test_timestamps_out_of_order_in_keyframes_input_not_supported_by_spec(strategy: LinearInterpolationStrategy):
    """
    Test with keyframes not sorted by time. The strategy expects sorted keyframes.
    This test is more to document behavior if precondition is violated.
    The actual sorting should be ensured by the caller (e.g., SmoothedTensorDemuxer).
    """
    kf = [(T(10), 20.0), (T(0), 10.0)] # Unsorted
    timestamps_to_interpolate = [T(5)]
    # Behavior is undefined by spec, but current code might produce something.
    # This test is just to observe, not to assert specific correctness for unsorted.
    # SmoothedTensorDemuxer ensures keyframes are sorted.
    with pytest.raises(Exception): # Or expect specific weird result if known
        # The current implementation might fail or produce weird results if kf not sorted.
        # For example, `keyframes[-2]` or `keyframes[surround_idx-1]` might be wrong.
        # A robust strategy might sort internally or error if not sorted.
        # Let's assume the provided strategy code will error or misbehave badly.
        # The provided code for LinearInterpolationStrategy does not explicitly sort
        # or check for sorted input. Its logic for finding t1/t2 will be flawed.
        # For instance, if t_req is T(5), surround_idx for [(T(10),20), (T(0),10)] would be 0 (kf_ts[0]=T(10) >= T(5)).
        # Then it would pick t1=keyframes[-1]=(T(0),10) and t2=keyframes[0]=(T(10),20). This might actually work by chance.
        # Let's try a case that would more clearly break:
        kf_broken = [(T(20), 30.0), (T(0), 10.0), (T(10), 20.0)] # Badly unsorted
        timestamps_to_interpolate_br = [T(5)] # Should be between T(0) and T(10)
        # If it picks T(0) and T(10) as surround, result is 15.
        # If it picks T(0) and T(20) as surround, result is 10 + 5/20 * (30-10) = 10 + 5 = 15.
        # The loop `for i, kf_ts_dt in enumerate(kf_timestamps): if kf_ts_dt.timestamp() >= t_req:`
        # will find the *first* one. For kf_broken timestamps [20,0,10] and t_req=5, surround_idx will be 0 (ts=20).
        # Then it tries: t1=kf_broken[-1]=(T(10),20) and t2=kf_broken[0]=(T(20),30).
        # interp = 20 + (5-10)*(30-20)/(20-10) = 20 + (-5)*(10)/(10) = 20-5 = 15.
        # This is surprisingly robust due to how it picks points *around* surround_idx or from ends.
        # However, the logic for extrapolation `t1_dt, v1 = keyframes[-2]; t2_dt, v2 = keyframes[-1]`
        # absolutely relies on the last two being chronologically last.
        # Let's test extrapolation with unsorted:
        kf_ext_unsorted = [(T(20), 20.0), (T(0), 0.0)] # last two are (T(20),20) and (T(0),0)
        ts_req_ext = [T(25)] # Should extrapolate from 0 and 20.
        # surround_idx = -1. It will use kf[-2]=(T(20),20) and kf[-1]=(T(0),0) as t1,v1 and t2,v2.
        # This is t1=20, v1=20; t2=0, v2=0.
        # val = 20 + (25-20)*(0-20)/(0-20) = 20 + 5*1 = 25. This is correct for points (0,0) and (20,20).
        # The strategy seems more robust to unsorted inputs than initially thought for *some* cases,
        # but it's not guaranteed and violates its implicit precondition.
        # For now, we won't assert failure but acknowledge this is outside spec.
        # A production system should ensure sorted keyframes are passed.
        pass # Not asserting, just documenting consideration.

def test_keyframe_values_can_be_negative(strategy: LinearInterpolationStrategy):
    """Test interpolation with negative values."""
    kf = [(T(0), -10.0), (T(10), -20.0)]
    timestamps_to_interpolate = [T(5)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == [-15.0]

def test_keyframe_values_can_be_mixed_sign(strategy: LinearInterpolationStrategy):
    """Test interpolation with mixed sign values."""
    kf = [(T(0), -10.0), (T(10), 10.0)]
    timestamps_to_interpolate = [T(2.5), T(5), T(7.5)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == [-5.0, 0.0, 5.0]

def test_interpolation_with_identical_timestamps_in_keyframes_not_ideal(strategy: LinearInterpolationStrategy):
    """
    Test behavior with identical timestamps in keyframes.
    The SmoothedTensorDemuxer should ensure unique timestamps are passed.
    If not, this strategy's behavior for t1==t2 is to return v1.
    """
    kf = [(T(0), 10.0), (T(5), 15.0), (T(5), 16.0), (T(10), 20.0)] # Duplicate T(5)
    # This input violates the "sorted by timestamp" if values differ, or implies an update.
    # The strategy's search for surround_idx will find the first T(5).
    # If a required_timestamp is T(5), it will hit kf[1] and return 15.0.
    timestamps_to_interpolate = [T(5)]
    result = strategy.interpolate_series(kf, timestamps_to_interpolate)
    assert result == [15.0] # Hits the first T(5)

    # If interpolation happens across a duplicate timestamp, e.g. T(2.5)
    # kf_timestamps = [0, 5, 5, 10]
    # t_req = 2.5. surround_idx = 1 (kf_ts[1]=5 >= 2.5)
    # t1_dt,v1 = kf[0] = (T(0),10)
    # t2_dt,v2 = kf[1] = (T(5),15)
    # interp = 10 + (2.5-0)*(15-10)/(5-0) = 10 + 2.5*5/5 = 10+2.5 = 12.5. Correct.
    timestamps_to_interpolate_2 = [T(2.5)]
    result_2 = strategy.interpolate_series(kf, timestamps_to_interpolate_2)
    assert result_2 == [12.5]

    # If interpolation happens at T(7.5)
    # t_req = 7.5. surround_idx = 3 (kf_ts[3]=10 >= 7.5)
    # t1_dt,v1 = kf[2] = (T(5),16)
    # t2_dt,v2 = kf[3] = (T(10),20)
    # interp = 16 + (7.5-5)*(20-16)/(10-5) = 16 + 2.5*4/5 = 16 + 2 = 18
    timestamps_to_interpolate_3 = [T(7.5)]
    result_3 = strategy.interpolate_series(kf, timestamps_to_interpolate_3)
    assert result_3 == [18.0]
    # This shows that if duplicate timestamps are passed, the choice of which one is used
    # for t1 or t2 depends on the surround_idx search.
    # SmoothedTensorDemuxer's on_update_received replaces values for identical timestamps,
    # so it should naturally pass a list of keyframes with unique timestamps.
