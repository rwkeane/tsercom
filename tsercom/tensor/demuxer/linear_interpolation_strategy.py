import torch
from typing import Optional # Added import

from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    Implements linear interpolation for a series of data points using torch.Tensor.
    """

    def __init__(
        self,
        max_extrapolation_seconds: Optional[float] = None,
        max_interpolation_gap_seconds: Optional[float] = None,
    ):
        super().__init__()
        self.max_extrapolation_seconds = max_extrapolation_seconds
        self.max_interpolation_gap_seconds = max_interpolation_gap_seconds

    def interpolate_series(
        self,
        timestamps: torch.Tensor,  # float tensor, e.g., Unix timestamps
        values: torch.Tensor,  # float tensor
        required_timestamps: torch.Tensor,  # float tensor
    ) -> torch.Tensor:
        """
        Performs linear interpolation using torch.Tensor operations.

        Timestamps are expected to be numerical (e.g., Unix timestamps as float).

        - If a required timestamp is before the first keyframe, the value of the
          first keyframe is returned (extrapolation).
        - If a required timestamp is after the last keyframe, the value of the
          last keyframe is returned (extrapolation).
        - If a required timestamp exactly matches a keyframe, the value of that
          keyframe is returned.
        - If there are no keyframes, a tensor of NaNs is returned.
        - If there is only one keyframe, its value is returned for all required
          timestamps.

        Args:
            timestamps: A 1D torch.Tensor of numerical timestamps, sorted ascending.
            values: A 1D torch.Tensor of corresponding values. Must be the same
                    length as `timestamps`.
            required_timestamps: A 1D torch.Tensor of numerical timestamps for which
                                 values are needed.

        Returns:
            A 1D torch.Tensor of interpolated values (float), corresponding to each
            `required_timestamp`. Values for which interpolation is not possible
            (e.g., no keyframes) will be NaN.
        """
        if timestamps.numel() == 0:
            return torch.full_like(
                required_timestamps, float("nan"), dtype=values.dtype
            )

        if timestamps.numel() == 1:
            return torch.full_like(
                required_timestamps, values[0].item(), dtype=values.dtype
            )

        # Find insertion points for required_timestamps in the keyframe timestamps
        # 'right=True' means if required_ts == key_ts, idx will be such that key_ts[idx-1] == required_ts
        # We want to find key_times[idx-1] <= required_ts < key_times[idx]
        # searchsorted returns idx where element would be inserted to maintain order.
        # So, for a value req_ts, key_times[idx-1] is the largest key_time <= req_ts (if idx > 0)
        # and key_times[idx] is the smallest key_time > req_ts (if idx < len(key_times))

        # Clamp required_timestamps to the range of keyframe timestamps for easier indexing
        # This simplifies extrapolation logic by mapping out-of-bound points to the boundary indices

        # Find indices of the keyframes that are to the right of each required_timestamp
        # `right=False` (default) means `timestamps[i-1] < v <= timestamps[i]`
        # `right=True` means `timestamps[i-1] <= v < timestamps[i]` (if v is in timestamps)
        # For `idx = torch.searchsorted(timestamps, clamped_required_timestamps)`:
        # if `clamped_required_timestamps[k] == timestamps[j]`, then `idx[k] == j`.
        # We need `idx_right` to point to the *upper* bound of the interval.

        # Ensure idx_right is at least 1 for safety, so idx_left (idx_right - 1) is valid.
        # For values exactly matching timestamps[0], idx_right could be 0.
        # For values matching timestamps[j], idx_right becomes j.
        # We want values between timestamps[idx_left] and timestamps[idx_right_actual].
        # So, if required_ts is timestamps[0], idx_right is 0. We want values[0].
        # If required_ts is timestamps[-1], idx_right is len(timestamps)-1. We want values[-1].

        # Initialize result tensor
        interpolated_values = torch.empty_like(required_timestamps, dtype=values.dtype)

        # Handle extrapolation for points before the first keyframe
        before_mask = required_timestamps < timestamps[0]
        interpolated_values[before_mask] = values[0]

        # Handle extrapolation for points after the last keyframe
        after_mask = required_timestamps > timestamps[-1]
        interpolated_values[after_mask] = values[-1]

        # Handle points within the range (including exact matches on boundaries)
        # `within_mask` will cover points between first and last keyframe, inclusive.
        within_mask = ~before_mask & ~after_mask

        if torch.any(within_mask):
            # Process only the relevant subset of required_timestamps
            current_req_ts = required_timestamps[within_mask]

            # Find right and left indices for interpolation for these 'within' points
            # `torch.searchsorted` returns the index where an element should be inserted
            # to maintain order.
            # `idx_right_interp` will be the index of the first timestamp > current_req_ts
            # or len(timestamps) if all timestamps are <= current_req_ts

            # `idx_left_interp` will be idx_right_interp - 1
            # These are the indices into the original 'timestamps' and 'values' tensors.

            # Clamp indices to be valid for 'timestamps' and 'values'
            # This is important for required_timestamps that exactly match a keyframe timestamp.
            # If current_req_ts = timestamps[0], idx_right_interp might be 0 (if right=False) or 1 (if right=True).
            # If right=True, idx_right_interp=0 if current_req_ts < timestamps[0] (already handled by before_mask)
            # or if current_req_ts == timestamps[0] and timestamps[0] is the only element.
            # We need to be careful.

            # Let's re-evaluate indexing for robustness.
            # For each current_req_ts:
            #   Find i such that timestamps[i] <= current_req_ts < timestamps[i+1]
            #   t1 = timestamps[i], v1 = values[i]
            #   t2 = timestamps[i+1], v2 = values[i+1]

            # `torch.searchsorted(timestamps, current_req_ts, right=True)` gives insert position `k`.
            # So, `timestamps[k-1]` is the largest timestamp <= `current_req_ts`. This is `t1`.
            # And `timestamps[k]` is the smallest timestamp > `current_req_ts` (or `timestamps[k-1]` if it's an exact match and `k` points to it). This is `t2`.

            t2_idx = torch.searchsorted(timestamps, current_req_ts, right=True)
            # Ensure t2_idx is not 0 for current_req_ts = timestamps[0] unless it's the only point (already handled)
            # Ensure t2_idx is not len(timestamps) for current_req_ts = timestamps[-1]
            t2_idx = torch.clamp(t2_idx, 1, timestamps.numel() - 1)
            t1_idx = t2_idx - 1

            # Handle cases where current_req_ts is an exact match with a keyframe timestamp
            # In this scenario, t1_idx points to the keyframe, and t2_idx might point to the same or next.
            # If timestamps[t1_idx] == current_req_ts, then proportion is 0.
            # If timestamps[t2_idx] == current_req_ts, then proportion is 1 (if t1_idx != t2_idx).

            t1 = timestamps[t1_idx]
            v1 = values[t1_idx]
            t2 = timestamps[t2_idx]
            v2 = values[t2_idx]

            # Avoid division by zero if t1 == t2 (e.g. duplicate timestamps in keyframes, or at boundaries)
            # If t1 == t2, implies v1 should be used (or v2, they should be same if data is consistent)
            denominator = t2 - t1
            # Create a mask for where denominator is zero
            zero_denom_mask = denominator == 0

            # Calculate proportion, default to 0 where denominator is zero (will use v1)
            proportion = torch.zeros_like(current_req_ts, dtype=values.dtype)
            # Calculate proportion only where denominator is non-zero
            calculated_proportion = (
                current_req_ts[~zero_denom_mask] - t1[~zero_denom_mask]
            ) / denominator[~zero_denom_mask]
            proportion[~zero_denom_mask] = calculated_proportion.to(proportion.dtype)

            # Handle cases where proportion might be slightly out of [0,1] due to float precision
            proportion = torch.clamp(proportion, 0.0, 1.0)

            interp_vals_within = v1 + proportion * (v2 - v1)
            interpolated_values[within_mask] = interp_vals_within

            # Special handling for exact matches:
            # If current_req_ts is an exact match with timestamps[t1_idx], value should be values[t1_idx]
            # If current_req_ts is an exact match with timestamps[t2_idx], value should be values[t2_idx]
            # The interpolation formula `v1 + proportion * (v2 - v1)` handles this naturally:
            # If current_req_ts == t1, proportion = 0, result = v1.
            # If current_req_ts == t2, proportion = 1, result = v2.
            # This is correct.

        return interpolated_values
