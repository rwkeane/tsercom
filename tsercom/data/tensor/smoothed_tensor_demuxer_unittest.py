import asyncio
from datetime import datetime as real_datetime, timedelta, timezone
from typing import (
    List,  # Keep List for MockClient.pushes
    Tuple,
    Optional,
    AsyncGenerator,
    Any,  # For MockClient type
)
import abc

import torch
import numpy as np  # Keep numpy for np.ndindex
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


class TestClientInterface(abc.ABC):
    @abc.abstractmethod
    async def push_tensor_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: real_datetime
    ) -> None:
        pass


class MockClient(TestClientInterface):
    def __init__(self) -> None:
        self.pushes: List[Tuple[str, torch.Tensor, real_datetime]] = []
        self.last_pushed_tensor: Optional[torch.Tensor] = None
        self.last_pushed_timestamp: Optional[real_datetime] = None

    async def push_tensor_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: real_datetime
    ) -> None:
        if not isinstance(data, torch.Tensor):
            pytest.fail(
                f"MockClient received data of type {type(data)}, expected torch.Tensor"
            )
        self.pushes.append((tensor_name, data.clone(), timestamp))
        self.last_pushed_tensor = data.clone()
        self.last_pushed_timestamp = timestamp

    def clear_pushes(self) -> None:
        self.pushes = []
        self.last_pushed_tensor = None
        self.last_pushed_timestamp = None


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest.fixture
def default_dtype() -> torch.dtype:
    return torch.float32


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    default_dtype: torch.dtype,
) -> AsyncGenerator[SmoothedTensorDemuxer, None]:
    demuxer_instance = SmoothedTensorDemuxer(
        tensor_name="test_tensor",
        tensor_shape=(2,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=10,
        align_output_timestamps=False,
        name="TestSmoothedDemuxer",
        fill_value=float("nan"),
        default_dtype=default_dtype,
    )
    yield demuxer_instance
    if (
        demuxer_instance._interpolation_worker_task is not None
        and not demuxer_instance._interpolation_worker_task.done()
    ):
        await demuxer_instance.stop()


@pytest.mark.asyncio
async def test_initialization(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    default_dtype: torch.dtype,
) -> None:
    tensor_name = "init_tensor"
    tensor_shape = (3, 3)
    output_interval = 1.0
    demuxer_instance = SmoothedTensorDemuxer(
        tensor_name=tensor_name,
        tensor_shape=tensor_shape,
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=output_interval,
        default_dtype=default_dtype,
    )
    assert demuxer_instance.tensor_name == tensor_name
    assert demuxer_instance.get_tensor_shape() == tensor_shape
    assert demuxer_instance._output_interval_seconds == output_interval
    assert demuxer_instance.get_default_dtype() == default_dtype


@pytest.mark.asyncio
async def test_on_update_received_adds_keyframes(
    demuxer: SmoothedTensorDemuxer, default_dtype: torch.dtype
) -> None:
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = real_datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    index0 = (0,)
    await demuxer.on_update_received(index0, 10.0, ts2_dt)
    await demuxer.on_update_received(
        index0, 5.0, ts1_dt
    )  # Insert older timestamp

    async with demuxer._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer._SmoothedTensorDemuxer__per_index_keyframes[index0]
        )  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() == 2
    assert values_tensor.numel() == 2
    assert timestamps_tensor.dtype == torch.float64
    assert values_tensor.dtype == default_dtype

    # Keyframes should be sorted by timestamp
    assert timestamps_tensor[0].item() == pytest.approx(ts1_dt.timestamp())
    assert values_tensor[0].item() == pytest.approx(5.0)
    assert timestamps_tensor[1].item() == pytest.approx(ts2_dt.timestamp())
    assert values_tensor[1].item() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_on_update_received_respects_history_limit(
    linear_strategy: LinearInterpolationStrategy,
    mock_client: MockClient,
    default_dtype: torch.dtype,
) -> None:
    history_limit = 3
    demuxer_limited_fixture = SmoothedTensorDemuxer(
        tensor_name="limited_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
        default_dtype=default_dtype,
    )
    index = (0,)
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(history_limit + 2):  # Add 5 keyframes
        ts_dt = base_ts_dt + timedelta(seconds=i)
        await demuxer_limited_fixture.on_update_received(
            index, float(i), ts_dt
        )

    async with demuxer_limited_fixture._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer_limited_fixture._SmoothedTensorDemuxer__per_index_keyframes[
                index
            ]
        )  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() == history_limit
    assert values_tensor.numel() == history_limit

    # Expected values are 2.0, 3.0, 4.0 if history_limit is 3 and 5 items (0,1,2,3,4) were added
    # First element should be value of item (history_limit + 2 - history_limit) = 2
    assert values_tensor[0].item() == pytest.approx(
        float(history_limit + 2 - history_limit)
    )
    # Last element should be value of item (history_limit + 2 - 1) = 4
    assert values_tensor[-1].item() == pytest.approx(
        float(history_limit + 2 - 1)
    )


@pytest.mark.asyncio
async def test_interpolation_worker_simple_case(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._output_interval_seconds = 0.05  # Shorten for faster test
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)

    await demuxer.on_update_received((0,), 10.0, ts1_dt)
    await demuxer.on_update_received((0,), 20.0, ts2_dt)
    await demuxer.on_update_received((1,), 100.0, ts1_dt)
    await demuxer.on_update_received((1,), 200.0, ts2_dt)

    mock_client.clear_pushes()
    current_time_for_mock = [
        ts1_dt
    ]  # Use a list to allow modification within closure

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "MockedDateTime":
            dt_to_return = current_time_for_mock[0]
            return dt_to_return.replace(tzinfo=tz) if tz else dt_to_return  # type: ignore

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime_for_mocking",
        MockedDateTime,
    )

    await demuxer.start()

    # Simulate time passing for worker cycles
    # Cycle 1
    current_time_for_mock[0] = (
        ts1_dt  # Initial time for worker to calculate first sleep
    )
    await asyncio.sleep(
        0.005
    )  # Brief sleep to let worker start and calculate initial sleep
    current_time_for_mock[0] = ts1_dt + timedelta(
        seconds=0.05
    )  # Advance time to first push point
    await asyncio.sleep(0.05 + 0.02)  # Worker sleep (0.05) + buffer

    # Cycle 2
    current_time_for_mock[0] = ts1_dt + timedelta(seconds=0.10)
    await asyncio.sleep(0.05 + 0.02)

    # Cycle 3
    current_time_for_mock[0] = ts1_dt + timedelta(seconds=0.15)
    await asyncio.sleep(0.05 + 0.02)

    await demuxer.stop()

    assert (
        len(mock_client.pushes) >= 1
    ), f"Worker should have pushed. Pushes: {mock_client.pushes}"

    found_relevant_push = False
    for _, data_tensor, push_ts_dt in mock_client.pushes:
        expected_push_times_dt = [
            ts1_dt + timedelta(seconds=0.05),
            ts1_dt + timedelta(seconds=0.10),
            ts1_dt + timedelta(seconds=0.15),
        ]
        if not any(
            abs((push_ts_dt - ept).total_seconds()) < 0.01
            for ept in expected_push_times_dt
        ):
            continue

        if ts1_dt < push_ts_dt < ts2_dt:
            val0 = data_tensor[0].item()
            val1 = data_tensor[1].item()
            time_ratio = (push_ts_dt - ts1_dt).total_seconds() / (
                ts2_dt - ts1_dt
            ).total_seconds()

            expected_val_0 = 10.0 + (20.0 - 10.0) * time_ratio
            expected_val_1 = 100.0 + (200.0 - 100.0) * time_ratio

            assert val0 == pytest.approx(expected_val_0, abs=1e-5)
            assert val1 == pytest.approx(expected_val_1, abs=1e-5)
            found_relevant_push = True

    assert (
        found_relevant_push
    ), f"No relevant interpolated tensor found. Pushes: {mock_client.pushes}"


@pytest.mark.asyncio
async def test_critical_cascading_interpolation_scenario(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    default_dtype: torch.dtype,
) -> None:
    demuxer_cascade = SmoothedTensorDemuxer(
        tensor_name="cascade_tensor",
        tensor_shape=(4,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        max_keyframe_history_per_index=10,
        align_output_timestamps=False,
        name="CascadeDemuxer",
        fill_value=float("nan"),
        default_dtype=default_dtype,
    )
    time_A_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    time_B_dt = time_A_dt + timedelta(seconds=2)
    time_C_dt = time_A_dt + timedelta(seconds=4)
    time_D_dt = time_A_dt + timedelta(seconds=6)

    await demuxer_cascade.on_update_received((0,), 10.0, time_A_dt)
    await demuxer_cascade.on_update_received((1,), 20.0, time_A_dt)
    await demuxer_cascade.on_update_received((2,), 30.0, time_A_dt)
    await demuxer_cascade.on_update_received((3,), 40.0, time_A_dt)

    await demuxer_cascade.on_update_received((0,), 100.0, time_D_dt)
    await demuxer_cascade.on_update_received((1,), 200.0, time_D_dt)
    await demuxer_cascade.on_update_received((2,), 300.0, time_D_dt)
    await demuxer_cascade.on_update_received((3,), 400.0, time_D_dt)

    await demuxer_cascade.on_update_received((0,), 50.0, time_B_dt)
    await demuxer_cascade.on_update_received((1,), 60.0, time_B_dt)
    await demuxer_cascade.on_update_received((2,), 70.0, time_C_dt)
    await demuxer_cascade.on_update_received((3,), 80.0, time_C_dt)

    time_interp_target_dt = time_A_dt + timedelta(
        seconds=3
    )  # Target for manual check
    output_tensor_manual = torch.full(
        demuxer_cascade.get_tensor_shape(), float("nan"), dtype=default_dtype
    )

    required_ts_tensor = torch.tensor(
        [time_interp_target_dt.timestamp()], dtype=torch.float64
    )

    async with demuxer_cascade._keyframes_lock:
        for i in range(demuxer_cascade.get_tensor_shape()[0]):
            idx = (i,)
            keyframe_data_tuple = demuxer_cascade._SmoothedTensorDemuxer__per_index_keyframes.get(
                idx
            )  # type: ignore [attr-defined]
            if keyframe_data_tuple:
                ts_tensor, val_tensor = keyframe_data_tuple
                if ts_tensor.numel() > 0:
                    interpolated_value_tensor = (
                        linear_strategy.interpolate_series(
                            ts_tensor, val_tensor, required_ts_tensor
                        )
                    )
                    if (
                        interpolated_value_tensor.numel() > 0
                        and not torch.isnan(interpolated_value_tensor[0])
                    ):
                        output_tensor_manual[idx] = interpolated_value_tensor[
                            0
                        ].item()

    # Expected values based on problem description:
    # Index 0: Keyframes (A,10), (B,50), (D,100). Target time_A+3s (B+1s). Interpolates between (B,50) and (D,100).
    #          B is at 2s, D is at 6s. Target is 3s. (3-2)/(6-2) = 1/4 = 0.25.  50 + 0.25 * (100-50) = 50 + 12.5 = 62.5
    # Index 1: Keyframes (A,20), (B,60), (D,200). Target time_A+3s (B+1s). Interpolates between (B,60) and (D,200).
    #          60 + 0.25 * (200-60) = 60 + 0.25*140 = 60 + 35 = 95.0
    # Index 2: Keyframes (A,30), (C,70), (D,300). Target time_A+3s. Interpolates between (A,30) and (C,70).
    #          A is at 0s, C is at 4s. Target is 3s. (3-0)/(4-0) = 3/4 = 0.75. 30 + 0.75 * (70-30) = 30 + 30 = 60.0
    # Index 3: Keyframes (A,40), (C,80), (D,400). Target time_A+3s. Interpolates between (A,40) and (C,80).
    #          40 + 0.75 * (80-40) = 40 + 30 = 70.0
    assert output_tensor_manual[0].item() == pytest.approx(62.5)
    assert output_tensor_manual[1].item() == pytest.approx(95.0)
    assert output_tensor_manual[2].item() == pytest.approx(60.0)
    assert output_tensor_manual[3].item() == pytest.approx(70.0)


@pytest.mark.asyncio
async def test_start_stop_worker(demuxer: SmoothedTensorDemuxer) -> None:
    assert demuxer._interpolation_worker_task is None
    await demuxer.start()
    assert demuxer._interpolation_worker_task is not None
    assert not demuxer._interpolation_worker_task.done()
    await asyncio.sleep(0.01)  # Allow task to run
    await demuxer.stop()
    # After stop, task should be None or done
    assert (
        demuxer._interpolation_worker_task is None
        or demuxer._interpolation_worker_task.done()
    )


@pytest.mark.asyncio
async def test_process_external_update_decomposes_tensor(
    demuxer: SmoothedTensorDemuxer, default_dtype: torch.dtype
) -> None:
    ts_dt = real_datetime(2023, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    full_tensor_data = torch.tensor([55.0, 66.0], dtype=default_dtype)

    await demuxer.process_external_update(
        demuxer.tensor_name, full_tensor_data, ts_dt
    )

    async with demuxer._keyframes_lock:
        ts_tensor0, val_tensor0 = (
            demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))
        )  # type: ignore [attr-defined]
        ts_tensor1, val_tensor1 = (
            demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((1,))
        )  # type: ignore [attr-defined]

    assert ts_tensor0 is not None and ts_tensor0.numel() == 1
    assert val_tensor0 is not None and val_tensor0.numel() == 1
    assert ts_tensor0[0].item() == pytest.approx(ts_dt.timestamp())
    assert val_tensor0[0].item() == pytest.approx(55.0)

    assert ts_tensor1 is not None and ts_tensor1.numel() == 1
    assert val_tensor1 is not None and val_tensor1.numel() == 1
    assert ts_tensor1[0].item() == pytest.approx(ts_dt.timestamp())
    assert val_tensor1[0].item() == pytest.approx(66.0)


@pytest.mark.asyncio
async def test_empty_keyframes_output_fill_value(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
    default_dtype: torch.dtype,
) -> None:
    demuxer._output_interval_seconds = 0.05
    fixed_keyframe_time_dt = real_datetime(
        2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
    )
    # Only update index (0,), index (1,) will have no keyframes
    await demuxer.on_update_received((0,), 10.0, fixed_keyframe_time_dt)

    current_time_for_mock = [fixed_keyframe_time_dt]

    class MockedDateTimeEmpty(real_datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "MockedDateTimeEmpty":
            dt = current_time_for_mock[0]
            return dt.replace(tzinfo=tz) if tz else dt  # type: ignore

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime_for_mocking",
        MockedDateTimeEmpty,
    )

    await demuxer.start()
    # Let worker run twice to ensure a push
    current_time_for_mock[0] = (
        fixed_keyframe_time_dt  # Set current time for first sleep calculation
    )
    await asyncio.sleep(0.005)  # let worker calculate sleep
    current_time_for_mock[0] = fixed_keyframe_time_dt + timedelta(
        seconds=demuxer._output_interval_seconds
    )  # Advance to push time
    await asyncio.sleep(
        demuxer._output_interval_seconds + 0.02
    )  # Wait for push

    await demuxer.stop()

    assert len(mock_client.pushes) > 0, "Worker should have pushed tensors"
    last_tensor = mock_client.last_pushed_tensor
    assert last_tensor is not None
    assert last_tensor.shape == demuxer.get_tensor_shape()
    assert last_tensor.dtype == default_dtype

    # Index (0) should have an extrapolated value (10.0)
    assert not torch.isnan(
        last_tensor[0]
    ).item(), "Index (0) should have a real value"
    assert last_tensor[0].item() == pytest.approx(10.0)
    # Index (1) should be fill_value (NaN in this test setup)
    assert torch.isnan(
        last_tensor[1]
    ).item(), "Index (1) should be NaN (fill_value)"


@pytest.mark.asyncio
async def test_align_output_timestamps_true(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    default_dtype: torch.dtype,
):
    output_interval = 1.0
    # Start time is 0.5s into a 1.0s interval slot
    start_time_dt = real_datetime(
        2023, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc
    )

    demuxer_aligned = SmoothedTensorDemuxer(
        tensor_name="aligned_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=output_interval,
        align_output_timestamps=True,
        name="AlignedDemuxer",
        default_dtype=default_dtype,
    )
    # Add a keyframe before the start time to ensure interpolation happens
    kf_ts_dt = start_time_dt - timedelta(
        seconds=0.2
    )  # 2023-01-01 00:00:00.300Z
    await demuxer_aligned.on_update_received((0,), 10.0, kf_ts_dt)

    current_time_for_mock = [start_time_dt]

    class MockedDateTimeAlign(real_datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "MockedDateTimeAlign":
            dt = current_time_for_mock[0]
            return dt.replace(tzinfo=tz) if tz else dt  # type: ignore

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime_for_mocking",
        MockedDateTimeAlign,
    )

    # Expected push times calculation:
    # Worker starts, current_loop_start_time (mocked as start_time_dt) = ...0.500Z
    # 1. _last_pushed_timestamp = _get_next_aligned_timestamp(start_time_dt = ...0.500Z)
    #    _get_next_aligned_timestamp for 0.5 with interval 1.0 gives next slot start: 1.000Z
    #    So, _last_pushed_timestamp = ...1.000Z
    # 2. First next_output_dt (target for push):
    #    next_output_dt_unaligned = _last_pushed_timestamp + interval = ...1.000Z + 1.0s = ...2.000Z
    #    next_output_dt_aligned = _get_next_aligned_timestamp(next_output_dt_unaligned = ...2.000Z) = ...2.000Z
    #    (Note: the problem description's manual calculation was slightly off here, it should align to the *start* of the next interval if already aligned)
    #    Corrected logic for worker for first push:
    #    _last_pushed_timestamp initialized to first *aligned* slot >= current time for first cycle, e.g. 2023-01-01 00:00:01.000Z
    #    Then target for push is _last_pushed_timestamp + interval = 2023-01-01 00:00:02.000Z
    #    If this is then aligned, it remains 2023-01-01 00:00:02.000Z

    # Let's trace the worker logic for the first actual push:
    # _last_pushed_timestamp is None initially.
    # current_loop_start_time = start_time_dt (...0.500Z)
    # _last_pushed_timestamp = _get_next_aligned_timestamp(start_time_dt)
    #   _get_next_aligned_timestamp(0.5s, interval=1s) -> ceil(0.5/1)*1 = 1.0s. next_slot_start_seconds = 1.0
    #   This is > 0.5, so _last_pushed_timestamp becomes datetime for 1.0s (2023-01-01 00:00:01.000Z)
    # next_output_dt = _last_pushed_timestamp (1.0s) + interval (1.0s) = 2.0s (2023-01-01 00:00:02.000Z)
    # next_output_dt = _get_next_aligned_timestamp(2.0s) -> still 2.0s
    # So, first push is at 3.0s (2023-01-01 00:00:03.000Z) - Corrected
    expected_first_push_ts = real_datetime(
        2023, 1, 1, 0, 0, 3, 0, tzinfo=timezone.utc
    )
    sleep_duration1 = (
        expected_first_push_ts - start_time_dt
    ).total_seconds()  # 3.0 - 0.5 = 2.5s - Corrected

    # For second push:
    # _last_pushed_timestamp is now 3.0s (expected_first_push_ts)
    # current_loop_start_time is mocked to expected_first_push_ts (3.0s)
    # next_output_dt = _last_pushed_timestamp (3.0s) + interval (1.0s) = 4.0s
    # next_output_dt = _get_next_aligned_timestamp(4.0s) -> becomes 5.0s
    expected_second_push_ts = real_datetime(
        2023, 1, 1, 0, 0, 5, 0, tzinfo=timezone.utc
    )
    sleep_duration2 = (
        expected_second_push_ts - expected_first_push_ts
    ).total_seconds()  # 3.0 - 2.0 = 1.0s

    await demuxer_aligned.start()

    current_time_for_mock[0] = start_time_dt  # Set for worker's first loop
    await asyncio.sleep(0.01)  # let worker calculate first sleep
    current_time_for_mock[0] = (
        expected_first_push_ts  # Advance time to when first push should happen
    )
    await asyncio.sleep(sleep_duration1 + 0.02)

    current_time_for_mock[0] = (
        expected_second_push_ts  # Advance time for second push
    )
    await asyncio.sleep(sleep_duration2 + 0.02)

    await demuxer_aligned.stop()

    pushed_timestamps = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) >= 1
    ), f"Demuxer should have pushed. Pushes: {pushed_timestamps}"

    assert mock_client.pushes[0][2] == expected_first_push_ts
    assert mock_client.pushes[0][2].timestamp() % output_interval == 0.0

    if len(mock_client.pushes) > 1:
        assert mock_client.pushes[1][2] == expected_second_push_ts
        assert mock_client.pushes[1][2].timestamp() % output_interval == 0.0


@pytest.mark.asyncio
async def test_small_max_keyframe_history(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    default_dtype: torch.dtype,
):
    history_limit = 1
    demuxer_small_hist = SmoothedTensorDemuxer(
        tensor_name="small_hist_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        max_keyframe_history_per_index=history_limit,
        default_dtype=default_dtype,
    )

    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)
    ts3_dt = ts1_dt + timedelta(seconds=2)

    await demuxer_small_hist.on_update_received((0,), 10.0, ts1_dt)
    await demuxer_small_hist.on_update_received(
        (0,), 20.0, ts2_dt
    )  # This makes keyframes [(ts2,20)]
    await demuxer_small_hist.on_update_received(
        (0,), 30.0, ts3_dt
    )  # This makes keyframes [(ts3,30)]

    current_time_for_mock = [ts3_dt]  # Current time is at the last keyframe

    class MockedDateTimeSmallHist(real_datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "MockedDateTimeSmallHist":
            dt = current_time_for_mock[0]
            return dt.replace(tzinfo=tz) if tz else dt  # type: ignore

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime_for_mocking",
        MockedDateTimeSmallHist,
    )

    mock_client.clear_pushes()
    await demuxer_small_hist.start()

    # Worker's first current_loop_start_time = ts3_dt
    # _last_pushed_timestamp will be ts3_dt
    # First next_output_timestamp = ts3_dt + 0.05s
    # Worker sleeps for 0.05s
    current_time_for_mock[0] = ts3_dt + timedelta(
        seconds=demuxer_small_hist._output_interval_seconds
    )
    await asyncio.sleep(0.05 + 0.02)  # Wait for worker sleep and push

    await demuxer_small_hist.stop()
    assert len(mock_client.pushes) > 0
    for _, tensor_data, _ in mock_client.pushes:
        # Since only (ts3, 30.0) is in history, it will extrapolate 30.0
        assert tensor_data[0].item() == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_2d_tensor_shape(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    default_dtype: torch.dtype,
):
    demuxer_2d = SmoothedTensorDemuxer(
        tensor_name="2d_tensor",
        tensor_shape=(2, 2),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        default_dtype=default_dtype,
    )

    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)

    await demuxer_2d.on_update_received((0, 0), 10.0, ts1_dt)
    await demuxer_2d.on_update_received((0, 1), 20.0, ts1_dt)
    await demuxer_2d.on_update_received((1, 0), 30.0, ts1_dt)
    await demuxer_2d.on_update_received((1, 1), 40.0, ts1_dt)
    await demuxer_2d.on_update_received(
        (0, 0), 15.0, ts2_dt
    )  # Update (0,0) at ts2

    current_time_for_mock_2d = [ts1_dt]

    class MockedDateTime2D(real_datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "MockedDateTime2D":
            dt = current_time_for_mock_2d[0]
            return dt.replace(tzinfo=tz) if tz else dt  # type: ignore

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime_for_mocking",
        MockedDateTime2D,
    )

    await demuxer_2d.start()
    expected_push_time = ts1_dt + timedelta(
        seconds=demuxer_2d._output_interval_seconds
    )

    current_time_for_mock_2d[0] = ts1_dt  # Set time for first sleep calc
    await asyncio.sleep(0.01)  # let worker calculate sleep
    current_time_for_mock_2d[0] = (
        expected_push_time  # Advance time to push point
    )
    await asyncio.sleep(0.05 + 0.02)  # Wait for worker to push

    await demuxer_2d.stop()

    assert len(mock_client.pushes) > 0, f"Pushes: {mock_client.pushes}"
    found_interp_frame = False
    for _, tensor_data, push_ts in mock_client.pushes:
        if (
            abs((push_ts - expected_push_time).total_seconds())
            < demuxer_2d._output_interval_seconds * 0.5
        ):
            assert tensor_data.shape == (2, 2)
            time_ratio = (push_ts - ts1_dt).total_seconds() / (
                ts2_dt - ts1_dt
            ).total_seconds()
            time_ratio = max(
                0.0, min(1.0, time_ratio)
            )  # Clamp ratio for safety

            expected_00 = (
                10.0 + (15.0 - 10.0) * time_ratio
            )  # (0,0) interpolates
            assert tensor_data[0, 0].item() == pytest.approx(
                expected_00, abs=1e-5
            )
            assert tensor_data[0, 1].item() == pytest.approx(
                20.0
            )  # Extrapolates ts1 value
            assert tensor_data[1, 0].item() == pytest.approx(
                30.0
            )  # Extrapolates ts1 value
            assert tensor_data[1, 1].item() == pytest.approx(
                40.0
            )  # Extrapolates ts1 value
            found_interp_frame = True
            break
    assert (
        found_interp_frame
    ), f"No suitable interpolated frame found. Expected around {expected_push_time}. Pushes: {mock_client.pushes}"


@pytest.mark.asyncio
async def test_significantly_out_of_order_updates(
    demuxer: SmoothedTensorDemuxer, default_dtype: torch.dtype
) -> None:
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Add some initial keyframes far in the "future"
    for i in range(5):
        await demuxer.on_update_received(
            (0,), float(i + 20), base_ts_dt + timedelta(seconds=i + 20)
        )

    # Add an old keyframe
    old_ts_dt = base_ts_dt + timedelta(seconds=1)
    await demuxer.on_update_received((0,), 1.0, old_ts_dt)

    async with demuxer._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer._SmoothedTensorDemuxer__per_index_keyframes[(0,)]
        )  # type: ignore [attr-defined]

    # Check if the old keyframe (1.0 at old_ts_dt) is now the first
    assert timestamps_tensor[0].item() == pytest.approx(old_ts_dt.timestamp())
    assert values_tensor[0].item() == pytest.approx(1.0)

    # Add more keyframes, some of which should cause pruning of the old_ts_dt if history limit is hit
    # Demuxer fixture has history_limit=10
    # Current keyframes: (old_ts,1.0) + 5 from initial loop = 6 keyframes
    # Add 5 more, total 11. One should be pruned.
    for i in range(5):  # Add 5 more, from base_ts+30s to base_ts+34s
        await demuxer.on_update_received(
            (0,), float(i + 30), base_ts_dt + timedelta(seconds=i + 30)
        )

    async with demuxer._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer._SmoothedTensorDemuxer__per_index_keyframes[(0,)]
        )  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() <= demuxer._max_keyframe_history_per_index

    # The first keyframe should now be one of the ones from the first loop,
    # as (old_ts, 1.0) should have been pruned.
    # Original 5: (20s,20), (21s,21), (22s,22), (23s,23), (24s,24)
    # Added old: (1s,1)
    # Sorted: (1s,1), (20s,20), ..., (24s,24)
    # Added 5 more: (30s,30) ... (34s,34)
    # Total 11. Pruned to 10. Smallest timestamp (1s,1.0) is pruned.
    # First element should be (base_ts + 20s, 20.0)
    expected_first_ts_after_prune = base_ts_dt + timedelta(seconds=20)
    assert timestamps_tensor[0].item() == pytest.approx(
        expected_first_ts_after_prune.timestamp()
    )
    assert values_tensor[0].item() == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_on_update_received_updates_existing_timestamp_value(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
) -> None:
    """
    Tests that if on_update_received is called for a timestamp that already
    exists for a given index, the value is updated in place in the tensor.
    """
    demuxer_instance = SmoothedTensorDemuxer(
        tensor_name="test_update_existing",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.01,  # Short interval for quick testing if worker runs
        max_keyframe_history_per_index=10,
        default_dtype=torch.float32,
    )

    ts_datetime = real_datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    index0 = (0,)

    # Initial update
    await demuxer_instance.on_update_received(index0, 10.0, ts_datetime)

    # Verify initial state
    async with demuxer_instance._keyframes_lock:
        # Accessing private member for test validation
        timestamps_tensor, values_tensor = (
            demuxer_instance._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )
    assert timestamps_tensor.numel() == 1
    assert values_tensor.numel() == 1
    assert abs(timestamps_tensor[0].item() - ts_datetime.timestamp()) < 1e-9
    assert abs(values_tensor[0].item() - 10.0) < 1e-9

    # Second update for the exact same timestamp, with a new value
    await demuxer_instance.on_update_received(index0, 20.0, ts_datetime)

    async with demuxer_instance._keyframes_lock:
        timestamps_tensor_updated, values_tensor_updated = (
            demuxer_instance._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )

    assert (
        timestamps_tensor_updated.numel() == 1
    ), "Timestamp count should remain 1"
    assert values_tensor_updated.numel() == 1, "Value count should remain 1"
    assert (
        abs(timestamps_tensor_updated[0].item() - ts_datetime.timestamp())
        < 1e-9
    ), "Timestamp should be unchanged"
    assert (
        abs(values_tensor_updated[0].item() - 20.0) < 1e-9
    ), "Value should be updated to 20.0"

    # Third update, same timestamp, same value as previous (should not change anything)
    await demuxer_instance.on_update_received(index0, 20.0, ts_datetime)
    async with demuxer_instance._keyframes_lock:
        timestamps_tensor_final, values_tensor_final = (
            demuxer_instance._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )
    assert (
        abs(values_tensor_final[0].item() - 20.0) < 1e-9
    ), "Value should still be 20.0"
    assert timestamps_tensor_final.numel() == 1


@pytest.mark.asyncio
async def test_default_dtype_and_fill_value_interaction(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
) -> None:
    """
    Tests how default_dtype and fill_value (NaN) interact, especially in the output_tensor
    of the interpolation worker.
    """
    test_dtypes = [torch.float32, torch.float64]
    if hasattr(torch, "bfloat16"):  # Conditionally add bfloat16 if supported
        test_dtypes.append(torch.bfloat16)

    for dtype_to_test in test_dtypes:
        mock_client.clear_pushes()
        demuxer_instance = SmoothedTensorDemuxer(
            tensor_name=f"test_dtype_{dtype_to_test}",
            tensor_shape=(2,),  # One index with data, one without
            output_client=mock_client,
            smoothing_strategy=linear_strategy,
            output_interval_seconds=0.05,
            max_keyframe_history_per_index=10,
            fill_value=float("nan"),  # Explicitly float('nan')
            default_dtype=dtype_to_test,
            align_output_timestamps=False,  # Simplify timing for mock
        )

        # Provide one keyframe for index (0,)
        kf_time = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        await demuxer_instance.on_update_received((0,), 100.0, kf_time)

        # Mock datetime.now() for predictable worker execution
        # Worker will want to push at kf_time + output_interval
        # Let's set "now" to be that exact push time.
        push_time = kf_time + timedelta(
            seconds=demuxer_instance._output_interval_seconds
        )

        current_time_for_mock = [push_time]  # datetime.now() will return this

        class MockedDateTime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                dt_to_return = current_time_for_mock[0]
                return dt_to_return.replace(tzinfo=tz) if tz else dt_to_return

        MockedDateTime.timedelta = timedelta  # Patch timedelta as well
        MockedDateTime.timezone = timezone

        mocker.patch(
            "tsercom.data.tensor.smoothed_tensor_demuxer.datetime_for_mocking",
            MockedDateTime,
        )

        await demuxer_instance.start()
        await asyncio.sleep(
            0.01
        )  # Allow worker to start and schedule first push

        # Advance "time" to allow the worker to complete one cycle
        # The worker sleeps until next_output_dt. If now() is already next_output_dt, sleep is minimal.
        current_time_for_mock[0] = push_time
        await asyncio.sleep(
            demuxer_instance._output_interval_seconds + 0.02
        )  # Wait for push

        await demuxer_instance.stop()

        assert (
            len(mock_client.pushes) >= 1
        ), f"Should have pushed for dtype {dtype_to_test}"
        _, output_tensor, _ = mock_client.pushes[
            -1
        ]  # Get the last pushed tensor

        assert (
            output_tensor.dtype == dtype_to_test
        ), f"Output tensor dtype mismatch for {dtype_to_test}"

        # Index (0,) should have extrapolated value 100.0
        if dtype_to_test == torch.bfloat16:
            assert (
                abs(output_tensor[0].item() - 100.0) < 0.1
            )  # Looser tolerance for bfloat16
        else:
            assert abs(output_tensor[0].item() - 100.0) < 1e-9

        # Index (1,) should be fill_value (NaN)
        assert torch.isnan(
            output_tensor[1]
        ).item(), f"Index (1,) should be NaN for dtype {dtype_to_test}"


@pytest.mark.asyncio
async def test_max_keyframe_history_one(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
) -> None:
    """Tests pruning logic with max_keyframe_history_per_index = 1."""
    demuxer_hist_one = SmoothedTensorDemuxer(
        tensor_name="test_hist_one",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,  # Not critical for this test's direct assertions
        max_keyframe_history_per_index=1,  # Critical setting
        default_dtype=torch.float64,  # Use float64 for precise value checks
    )

    index0 = (0,)
    ts1 = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = real_datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    ts3 = real_datetime(2023, 1, 1, 0, 0, 2, tzinfo=timezone.utc)

    # Add first keyframe
    await demuxer_hist_one.on_update_received(index0, 10.0, ts1)
    async with demuxer_hist_one._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer_hist_one._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )
    assert timestamps_tensor.numel() == 1
    assert abs(values_tensor[0].item() - 10.0) < 1e-9

    # Add second keyframe, should replace the first (ts2 > ts1)
    await demuxer_hist_one.on_update_received(index0, 20.0, ts2)
    async with demuxer_hist_one._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer_hist_one._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )
    assert timestamps_tensor.numel() == 1
    assert abs(timestamps_tensor[0].item() - ts2.timestamp()) < 1e-9
    assert abs(values_tensor[0].item() - 20.0) < 1e-9

    # Add third keyframe (ts3 > ts2), should replace the second
    await demuxer_hist_one.on_update_received(index0, 30.0, ts3)
    async with demuxer_hist_one._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer_hist_one._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )
    assert timestamps_tensor.numel() == 1
    assert abs(timestamps_tensor[0].item() - ts3.timestamp()) < 1e-9
    assert abs(values_tensor[0].item() - 30.0) < 1e-9

    # Add an older keyframe (ts1 < ts3), should be discarded as history is full with ts3
    await demuxer_hist_one.on_update_received(
        index0, 5.0, ts1
    )  # This is ts1 < ts3
    async with demuxer_hist_one._keyframes_lock:
        timestamps_tensor, values_tensor = (
            demuxer_hist_one._SmoothedTensorDemuxer__per_index_keyframes[
                index0
            ]
        )
    assert timestamps_tensor.numel() == 1
    assert (
        abs(timestamps_tensor[0].item() - ts3.timestamp()) < 1e-9
    ), "Old timestamp should not have replaced newer one"
    assert abs(values_tensor[0].item() - 30.0) < 1e-9
