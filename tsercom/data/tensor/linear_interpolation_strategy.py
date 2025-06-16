import torch

from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy


class LinearInterpolationStrategy(SmoothingStrategy):
    """
    Implements linear interpolation for a series of data points using PyTorch tensors.
    """

    def interpolate_series(
        self,
        timestamps: torch.Tensor,
        values: torch.Tensor,
        required_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs linear interpolation using tensor operations.

        Args:
            timestamps: torch.Tensor: A 1D tensor of Unix timestamps (float64), sorted.
            values: torch.Tensor: A 1D tensor of corresponding values (float32).
            required_timestamps: torch.Tensor: A 1D tensor of Unix timestamps (float64)
                               for which values are needed.

        Returns:
            torch.Tensor: A 1D tensor of interpolated values (float32), same size as
                          `required_timestamps`. Contains `float('nan')` where
                          interpolation is not possible (e.g. no keyframes).
                          Extrapolates using the first/last keyframe value if
                          required timestamps are outside the keyframe range.
        """
        if (
            not isinstance(timestamps, torch.Tensor)
            or not isinstance(values, torch.Tensor)
            or not isinstance(required_timestamps, torch.Tensor)
        ):
            raise TypeError("All inputs must be torch.Tensor objects.")
        if (
            timestamps.ndim != 1
            or values.ndim != 1
            or required_timestamps.ndim != 1
        ):
            raise ValueError("All input tensors must be 1D.")
        if (
            timestamps.dtype != torch.float64
            or values.dtype != torch.float32
            or required_timestamps.dtype != torch.float64
        ):
            raise ValueError(
                "Input tensor dtypes do not match expected: timestamps (float64), values (float32), required_timestamps (float64)."
            )

        if timestamps.numel() == 0:
            return torch.full_like(
                required_timestamps, float("nan"), dtype=torch.float32
            )

        if timestamps.numel() == 1:
            return torch.full_like(
                required_timestamps, values[0].item(), dtype=torch.float32
            )

        # Initialize results with NaN, output dtype should be float32
        interpolated_results = torch.full_like(
            required_timestamps, float("nan"), dtype=torch.float32
        )

        # Extrapolation for timestamps before the first keyframe
        before_mask = required_timestamps <= timestamps[0]
        interpolated_results[before_mask] = values[0].item()

        # Extrapolation for timestamps after the last keyframe
        after_mask = required_timestamps >= timestamps[-1]
        interpolated_results[after_mask] = values[-1].item()

        # Interpolation for timestamps between the first and last keyframes
        # Ensure strict inequality for interpolation range to avoid issues at boundaries already handled by extrapolation
        interp_mask = (required_timestamps > timestamps[0]) & (
            required_timestamps < timestamps[-1]
        )

        if torch.any(interp_mask):
            relevant_req_ts = required_timestamps[interp_mask]

            # Find indices of right neighbors for each required timestamp in the interpolation range
            # searchsorted will find the index where an element could be inserted to maintain order.
            # For a timestamp ts_req, if ts_req is between timestamps[i] and timestamps[i+1],
            # searchsorted(timestamps, ts_req) would give i+1 (if side='left')
            # or i+1 (if side='right' and ts_req > timestamps[i]).
            # We want right_indices such that timestamps[left_indices] <= relevant_req_ts < timestamps[right_indices]
            # So, side='right' is appropriate here.
            right_indices = torch.searchsorted(
                timestamps, relevant_req_ts, side="right"
            )

            # Clamp right_indices to prevent going out of bounds if relevant_req_ts is equal to timestamps[-1]
            # This should ideally not happen due to strict interp_mask, but good for safety.
            right_indices = torch.clamp(
                right_indices, 1, timestamps.numel() - 1
            )
            left_indices = right_indices - 1

            # Ensure left_indices are valid (>=0)
            # This should also be guaranteed by interp_mask and clamping, but defensive check.
            left_indices = torch.clamp(left_indices, 0, timestamps.numel() - 2)

            t1 = timestamps[left_indices]
            v1 = values[left_indices]
            t2 = timestamps[right_indices]
            v2 = values[right_indices]

            dt = t2 - t1
            dv = v2 - v1

            # Initialize interpolated values for this segment with v1
            # This correctly handles cases where dt == 0 (t1 == t2) by assigning v1.
            interp_values_segment = v1.clone().to(torch.float32)

            # Mask for points where linear interpolation is possible (dt > 0)
            is_linear_interp_possible = dt > 0

            if torch.any(is_linear_interp_possible):
                # Calculate proportion only for points where dt > 0
                # Ensure relevant_req_ts, t1, dt, v1, dv are indexed by is_linear_interp_possible for broadcasting
                prop_ts = relevant_req_ts[is_linear_interp_possible]
                prop_t1 = t1[is_linear_interp_possible]
                prop_dt = dt[is_linear_interp_possible]
                prop_v1 = v1[is_linear_interp_possible]
                prop_dv = dv[is_linear_interp_possible]

                proportion = (prop_ts - prop_t1) / prop_dt
                interp_values_segment[is_linear_interp_possible] = (
                    prop_v1 + proportion * prop_dv
                ).to(torch.float32)  # Ensure result is float32

            interpolated_results[interp_mask] = interp_values_segment

        # Refinement for exact matches:
        # This step ensures that if a required_timestamp exactly matches a keyframe timestamp,
        # the exact keyframe value is used. This can sometimes be important due to floating point precision.
        # Note: torch.searchsorted with side='left' gives the index `i` such that all e in a[:i] have e < v,
        # and all e in a[i:] have e >= v.
        # So if timestamps[indices_of_exact_matches] == required_timestamps, it's an exact match.

        # Find potential indices for exact matches
        indices_of_exact_matches = torch.searchsorted(
            timestamps, required_timestamps, side="left"
        )

        # Clamp indices to be valid for `timestamps` tensor to avoid out-of-bounds errors
        indices_of_exact_matches.clamp_(0, timestamps.numel() - 1)

        # Create a mask for actual exact matches
        # Need to check bounds for indices_of_exact_matches before indexing timestamps
        # This check ensures we only use valid indices from searchsorted
        # valid_indices_mask = indices_of_exact_matches < timestamps.numel() # Unused variable

        # Further, ensure that timestamps at these clamped indices actually match
        # This needs careful handling if required_timestamps can be outside the range of timestamps
        # The extrapolation steps should have handled those already.
        # We only care about exact matches within the original range of timestamps.

        # Create a mask for valid exact matches
        # exact_match_mask = torch.zeros_like( # Unused variable
        #     required_timestamps, dtype=torch.bool
        # )

        # Check only if there are elements to avoid errors on empty tensors
        if timestamps.numel() > 0 and required_timestamps.numel() > 0:
            # Get timestamps that correspond to searchsorted indices
            candidate_timestamps = timestamps[indices_of_exact_matches]
            # Identify where these candidate timestamps are exactly equal to required_timestamps
            actual_exact_matches = candidate_timestamps == required_timestamps
            # Apply this mask to interpolated_results
            if torch.any(actual_exact_matches):
                interpolated_results[actual_exact_matches] = values[
                    indices_of_exact_matches[actual_exact_matches]
                ].to(torch.float32)

        return interpolated_results.to(torch.float32)
