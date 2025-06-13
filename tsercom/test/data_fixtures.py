import datetime
import torch
import pytest
import pytest_asyncio
from typing import List, Tuple

from tsercom.data.tensor.tensor_demuxer import TensorDemuxer

# Helper type for captured calls by the mock client
CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor.clone(), timestamp))
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(self) -> CapturedTensorChange | None:
        return self.calls[-1] if self.calls else None

    def get_latest_tensor_for_ts(
        self, timestamp: datetime.datetime
    ) -> torch.Tensor | None:
        for i in range(len(self.calls) - 1, -1, -1):
            tensor, ts = self.calls[i]
            if ts == timestamp:
                return tensor
        return None

    def get_all_calls_summary(
        self,
    ) -> List[Tuple[List[float], datetime.datetime]]:
        return [(t.tolist(), ts) for t, ts in self.calls]


@pytest_asyncio.fixture
async def mock_client() -> MockTensorDemuxerClient:
    return MockTensorDemuxerClient()


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    actual_mock_client = mock_client
    demuxer_instance = TensorDemuxer(
        client=actual_mock_client, tensor_length=4, data_timeout_seconds=60.0
    )
    return demuxer_instance, actual_mock_client


@pytest_asyncio.fixture
async def demuxer_short_timeout(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    actual_mock_client = mock_client
    demuxer_instance = TensorDemuxer(
        client=actual_mock_client, tensor_length=4, data_timeout_seconds=0.1
    )
    return demuxer_instance, actual_mock_client
