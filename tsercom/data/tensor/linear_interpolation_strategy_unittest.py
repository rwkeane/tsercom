# pylint: disable=missing-class-docstring, missing-function-docstring, protected-access
# pylint: disable=too-many-arguments, too-many-locals, too-many-statements

import datetime

import pytest

# import pytest_asyncio # Not needed if tests are not async

from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)

# Numeric is defined in smoothing_strategy, but tests here deal with concrete values directly
# For helper functions if they were to use Numeric:
# from tsercom.data.tensor.smoothing_strategy import Numeric


# --- Helper Functions (copied from smoothed_tensor_demuxer_unittest.py if needed) ---
BASE_TIME = datetime.datetime(
    2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
)


def ts_at(
    seconds_offset: float, base_time: datetime.datetime = BASE_TIME
) -> datetime.datetime:
    return base_time + datetime.timedelta(seconds=seconds_offset)


# --- Fixtures (copied/adapted from smoothed_tensor_demuxer_unittest.py) ---
@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


# --- Test Class (copied from smoothed_tensor_demuxer_unittest.py) ---
class TestLinearInterpolationStrategy:
    def test_empty_keyframes_raises_error(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        with pytest.raises(
            ValueError, match="Cannot interpolate with no keyframes"
        ):
            linear_strategy.interpolate_series([], [ts_at(1)])

    def test_empty_required_timestamps(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        assert linear_strategy.interpolate_series([(ts_at(0), 1.0)], []) == []

    def test_before_first_keyframe(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(1), 10.0), (ts_at(3), 30.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(0)]) == [
            10.0
        ]
        assert linear_strategy.interpolate_series(
            keyframes, [ts_at(-1), ts_at(0.5)]
        ) == [10.0, 10.0]

    def test_after_last_keyframe(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(1), 10.0), (ts_at(3), 30.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(4)]) == [
            30.0
        ]
        assert linear_strategy.interpolate_series(
            keyframes, [ts_at(3.5), ts_at(5)]
        ) == [30.0, 30.0]

    def test_exact_match_keyframe(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(1), 10.0), (ts_at(3), 30.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(1)]) == [
            10.0
        ]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(3)]) == [
            30.0
        ]
        assert linear_strategy.interpolate_series(
            keyframes, [ts_at(1), ts_at(3)]
        ) == [10.0, 30.0]

    def test_interpolation_single_point(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(0), 0.0), (ts_at(2), 20.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(1)]) == [
            10.0
        ]

    def test_interpolation_multiple_points(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(0), 0.0), (ts_at(4), 40.0)]
        req_ts = [ts_at(1), ts_at(2), ts_at(3)]
        expected_values = [10.0, 20.0, 30.0]
        assert (
            linear_strategy.interpolate_series(keyframes, req_ts)
            == expected_values
        )

    def test_interpolation_at_exact_first_keyframe(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(0), 0.0), (ts_at(1), 10.0), (ts_at(2), 20.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(0)]) == [
            0.0
        ]

    def test_interpolation_at_exact_last_keyframe(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(0), 0.0), (ts_at(1), 10.0), (ts_at(2), 20.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(2)]) == [
            20.0
        ]

    def test_keyframes_with_non_monotonic_values(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        # Timestamps are monotonic, but values go up then down
        keyframes = [(ts_at(0), 0.0), (ts_at(1), 20.0), (ts_at(2), 10.0)]
        # Interpolate at t=0.5 (between 0,0 and 1,20) -> should be 10.0
        # Interpolate at t=1.5 (between 1,20 and 2,10) -> should be 15.0
        assert linear_strategy.interpolate_series(keyframes, [ts_at(0.5)]) == [
            10.0
        ]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(1.5)]) == [
            15.0
        ]

    def test_timestamps_with_microsecond_precision(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        t0 = datetime.datetime(
            2024, 1, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc
        )
        t1 = t0 + datetime.timedelta(microseconds=100)
        t_half = t0 + datetime.timedelta(microseconds=50)
        keyframes = [(t0, 0.0), (t1, 100.0)]
        assert linear_strategy.interpolate_series(keyframes, [t_half]) == [
            50.0
        ]

        t_other = t0 + datetime.timedelta(microseconds=20)
        assert linear_strategy.interpolate_series(keyframes, [t_other]) == [
            20.0
        ]

    def test_interpolation_with_only_two_keyframes_explicit(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(5), 50.0), (ts_at(10), 100.0)]
        # Before
        assert linear_strategy.interpolate_series(keyframes, [ts_at(0)]) == [
            50.0
        ]
        # Exact start
        assert linear_strategy.interpolate_series(keyframes, [ts_at(5)]) == [
            50.0
        ]
        # Middle
        assert linear_strategy.interpolate_series(keyframes, [ts_at(7.5)]) == [
            75.0
        ]
        # Exact end
        assert linear_strategy.interpolate_series(keyframes, [ts_at(10)]) == [
            100.0
        ]
        # After
        assert linear_strategy.interpolate_series(keyframes, [ts_at(15)]) == [
            100.0
        ]

    def test_identical_timestamps_different_values_interpolation_behavior(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        kf = [
            (ts_at(1), 10.0),  # (A)
            (ts_at(1), 20.0),  # (B)
            (ts_at(3), 30.0),  # (C)
        ]
        assert linear_strategy.interpolate_series(kf, [ts_at(1)]) == [10.0]
        assert linear_strategy.interpolate_series(kf, [ts_at(2)]) == [25.0]

        kf2 = [
            (ts_at(0), 0.0),  # (X)
            (ts_at(1), 10.0),  # (A)
            (ts_at(1), 20.0),  # (B)
            (ts_at(3), 30.0),  # (C)
        ]
        assert linear_strategy.interpolate_series(kf2, [ts_at(1)]) == [10.0]
        assert linear_strategy.interpolate_series(kf2, [ts_at(2)]) == [25.0]

    def test_mixed_cases(self, linear_strategy: LinearInterpolationStrategy) -> None:
        keyframes = [(ts_at(10), 100.0), (ts_at(20), 200.0)]
        req_ts = [ts_at(5), ts_at(10), ts_at(15), ts_at(20), ts_at(25)]
        expected_values = [100.0, 100.0, 150.0, 200.0, 200.0]
        assert (
            linear_strategy.interpolate_series(keyframes, req_ts)
            == expected_values
        )

    def test_identical_timestamps_in_keyframes(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(1), 10.0), (ts_at(1), 15.0), (ts_at(3), 30.0)]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(1)]) == [
            10.0
        ]
        assert linear_strategy.interpolate_series(keyframes, [ts_at(2)]) == [
            22.5
        ]

    def test_single_keyframe(
        self, linear_strategy: LinearInterpolationStrategy
    ) -> None:
        keyframes = [(ts_at(1), 10.0)]
        req_ts = [ts_at(0), ts_at(1), ts_at(2)]
        expected_values = [10.0, 10.0, 10.0]
        assert (
            linear_strategy.interpolate_series(keyframes, req_ts)
            == expected_values
        )
