import torch

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
            )  # .item() is correct here for full_like

        output_dtype = values.dtype
        timestamps_float = timestamps.to(torch.float64)
        values_float = values.to(torch.float64)
        required_timestamps_float = required_timestamps.to(torch.float64)

        interpolated_results = torch.empty_like(
            required_timestamps_float, dtype=output_dtype
        )

        before_mask = required_timestamps_float <= timestamps_float[0]
        if torch.any(before_mask):  # Restored guard
            interpolated_results[before_mask] = values_float[0].to(
                output_dtype
            )  # Restored .to(output_dtype)

        after_mask = required_timestamps_float >= timestamps_float[-1]
        if torch.any(after_mask):  # Restored guard
            interpolated_results[after_mask] = values_float[-1].to(
                output_dtype
            )  # Restored .to(output_dtype)

        between_mask = ~(before_mask | after_mask)

        if not torch.any(between_mask):
            return interpolated_results.to(
                output_dtype
            )  # Ensure final cast if returning early

        relevant_req_ts = required_timestamps_float[between_mask]
        indices = torch.searchsorted(
            timestamps_float, relevant_req_ts, right=False
        )
        indices = torch.clamp(indices, 1, timestamps_float.numel() - 1)

        t1 = timestamps_float[indices - 1]
        v1 = values_float[indices - 1]
        t2 = timestamps_float[indices]
        v2 = values_float[indices]

        exact_match_t2_mask = relevant_req_ts == t2
        # temp_exact_match_full_mask logic from previous successful version
        if torch.any(exact_match_t2_mask):
            temp_exact_match_full_mask = torch.zeros_like(
                required_timestamps_float, dtype=torch.bool
            )
            temp_exact_match_full_mask[between_mask] = exact_match_t2_mask
            interpolated_results[temp_exact_match_full_mask] = v2[
                exact_match_t2_mask
            ].to(output_dtype)

        interp_needed_mask = ~exact_match_t2_mask

        if not torch.any(interp_needed_mask):
            return interpolated_results.to(output_dtype)  # Ensure final cast

        _t1 = t1[interp_needed_mask]
        _v1 = v1[interp_needed_mask]
        _t2 = t2[interp_needed_mask]
        _v2 = v2[interp_needed_mask]
        _relevant_req_ts_interp = relevant_req_ts[interp_needed_mask]

        time_diff = _t2 - _t1
        # Ensure proportion is float64 for precision with timestamp differences
        proportion = torch.zeros_like(time_diff, dtype=torch.float64)

        valid_time_diff_mask = time_diff > 1e-9  # Epsilon for float comparison

        # Perform division only where time_diff is valid
        proportion[valid_time_diff_mask] = (
            _relevant_req_ts_interp[valid_time_diff_mask]
            - _t1[valid_time_diff_mask]
        ) / time_diff[valid_time_diff_mask]

        current_interp_values = _v1 + proportion * (
            _v2 - _v1
        )  # Calculation in float64

        # Create full-size mask for final assignment
        final_interp_mask = torch.zeros_like(
            required_timestamps_float, dtype=torch.bool
        )
        final_interp_mask[between_mask] = interp_needed_mask
        if torch.any(final_interp_mask):  # Guard assignment
            interpolated_results[final_interp_mask] = current_interp_values.to(
                output_dtype
            )

        return interpolated_results.to(output_dtype)  # Ensure final cast
