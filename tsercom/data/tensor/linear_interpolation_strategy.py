import torch

# Removed bisect, datetime, List, Tuple, Union

from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    Implements linear interpolation for a series of data points using torch.Tensor.
    """

    def interpolate_series(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        required_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs linear interpolation using tensor operations.

        Timestamps are expected to be numerical (e.g., POSIX timestamps as float64).
        Keyframe timestamps must be sorted.

        - If a required timestamp is before the first keyframe, the value of the
          first keyframe is returned (extrapolation).
        - If a required timestamp is after the last keyframe, the value of the
          last keyframe is returned (extrapolation).
        - If a required timestamp exactly matches a keyframe, the value of that
          keyframe is returned.
        - If there are no keyframes, a tensor of NaNs is returned for all required_timestamps.
        - If there is only one keyframe, its value is returned for all required timestamps.

        Args:
            timestamps: A 1D tensor of sorted keyframe timestamps.
            values: A 1D tensor of keyframe values, same length as 'timestamps'.
            required_timestamps: A 1D tensor of timestamps for which values are needed.

        Returns:
            A 1D tensor of interpolated float values, corresponding to each
            required_timestamp. Values will be of the same dtype as 'values' tensor.
        """
        if timestamps.numel() == 0:
            return torch.full_like(
                required_timestamps, float("nan"), dtype=values.dtype
            )

        if timestamps.numel() == 1:
            return torch.full_like(
                required_timestamps, values[0].item(), dtype=values.dtype
            )

        # Ensure inputs are float for calculations, especially timestamps for division
        # Output dtype should match input 'values' dtype.
        output_dtype = values.dtype
        timestamps_float = timestamps.to(torch.float64)
        values_float = values.to(
            torch.float64
        )  # Promote for internal calculation precision
        required_timestamps_float = required_timestamps.to(torch.float64)

        interpolated_results = torch.empty_like(
            required_timestamps_float, dtype=output_dtype
        )

        # Handle extrapolation for timestamps before the first keyframe
        before_mask = required_timestamps_float <= timestamps_float[0]
        if torch.any(before_mask):
            interpolated_results[before_mask] = values_float[0].to(output_dtype)

        # Handle extrapolation for timestamps after the last keyframe
        after_mask = required_timestamps_float >= timestamps_float[-1]
        if torch.any(after_mask):
            interpolated_results[after_mask] = values_float[-1].to(output_dtype)

        # Handle interpolation for timestamps between the first and last keyframes
        # Create a mask for values that are neither before the first nor after the last.
        # These are the candidates for actual interpolation.
        between_mask = ~(before_mask | after_mask)

        # If there are no timestamps strictly between the keyframe range, we're done.
        if not torch.any(between_mask):
            return interpolated_results.to(output_dtype)

        # Work only with the subset of required_timestamps that are between keyframes
        relevant_req_ts = required_timestamps_float[between_mask]

        # Find insertion points for relevant_req_ts in timestamps
        # 'right=False' makes it behave like bisect_left (idx such that all ts[:idx] < val)
        # For interpolation, we usually need the index of the keyframe *before* and *after*
        # searchsorted returns the index idx such that timestamps[idx-1] < req_ts <= timestamps[idx]
        indices = torch.searchsorted(
            timestamps_float, relevant_req_ts, right=False
        )

        # Ensure indices are within valid bounds for t1, t2 access
        # Clamp indices to avoid going out of bounds for t1 (idx-1) and t2 (idx)
        # If indices[i] is 0, it means relevant_req_ts[i] is <= timestamps_float[0].
        # This should have been caught by 'before_mask', but as a safeguard:
        indices = torch.clamp(indices, 1, timestamps_float.numel() - 1)

        t1 = timestamps_float[indices - 1]
        v1 = values_float[indices - 1]
        t2 = timestamps_float[indices]
        v2 = values_float[indices]

        # Handle cases where required_timestamp exactly matches a keyframe timestamp
        # If relevant_req_ts[i] == t1[i] (i.e., timestamps_float[indices[i]-1]), use v1[i].
        # If relevant_req_ts[i] == t2[i] (i.e., timestamps_float[indices[i]]), use v2[i].
        # searchsorted with right=False means if req_ts == timestamps[k], idx = k. So t2 is the match.
        exact_match_t2_mask = relevant_req_ts == t2
        if torch.any(exact_match_t2_mask):
            # Apply to the subset of interpolated_results indicated by between_mask
            # And then further by exact_match_t2_mask (which is relative to relevant_req_ts)
            temp_exact_match_full_mask = torch.zeros_like(
                required_timestamps_float, dtype=torch.bool
            )
            temp_exact_match_full_mask[between_mask] = exact_match_t2_mask
            interpolated_results[temp_exact_match_full_mask] = v2[
                exact_match_t2_mask
            ].to(output_dtype)

        # For actual interpolation: req_ts is between t1 and t2 (and not equal to t2)
        interp_needed_mask = ~exact_match_t2_mask

        # If all "between" points were exact matches to t2, nothing left to interpolate.
        if not torch.any(interp_needed_mask):
            # This check is important. If all points within 'between_mask' were exact matches,
            # then interp_needed_mask would be all False. Accessing tensors with an all-False mask
            # for assignment can be tricky or lead to empty tensors.
            # We've already filled exact matches, so we can return.
            return interpolated_results.to(output_dtype)

        # Filter to only those that need actual interpolation
        _t1 = t1[interp_needed_mask]
        _v1 = v1[interp_needed_mask]
        _t2 = t2[interp_needed_mask]
        _v2 = v2[interp_needed_mask]
        _relevant_req_ts_interp = relevant_req_ts[interp_needed_mask]

        # Calculate proportion, guard against division by zero if t2 == t1
        time_diff = _t2 - _t1
        # Where time_diff is zero, proportion should effectively make result v1 (or v2).
        # If t1=t2, and req_ts is between them (only if req_ts=t1=t2),
        # it should have been caught by exact match. If somehow not, default to v1.
        proportion = torch.zeros_like(time_diff)
        valid_time_diff_mask = (
            time_diff > 1e-9
        )  # Using a small epsilon for float comparison

        proportion[valid_time_diff_mask] = (
            _relevant_req_ts_interp[valid_time_diff_mask]
            - _t1[valid_time_diff_mask]
        ) / time_diff[valid_time_diff_mask]

        # If time_diff is not > epsilon (i.e., t1 approx t2), default to v1.
        # The values for !valid_time_diff_mask in interpolated_results[between_mask][interp_needed_mask]
        # will be set to _v1 values.
        current_interp_values = _v1 + proportion * (_v2 - _v1)

        # Fill in the results for the points that needed interpolation
        # Create a combined mask to update interpolated_results:
        # It starts with 'between_mask', then is filtered by 'interp_needed_mask'.
        # A direct way: create a full-size mask for interp_needed points
        final_interp_mask = torch.zeros_like(
            required_timestamps_float, dtype=torch.bool
        )
        final_interp_mask[between_mask] = interp_needed_mask
        if torch.any(final_interp_mask):
            interpolated_results[final_interp_mask] = current_interp_values.to(
                output_dtype
            )

        return interpolated_results.to(output_dtype)
