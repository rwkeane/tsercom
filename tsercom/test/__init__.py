# tsercom - Test Utilities
# Allows "from tsercom.test import ..." for centralized fixtures and helpers.

from tsercom.test.loop_fixtures import manage_tsercom_loop, clear_loop_fixture
from tsercom.test.runtime_fixtures import (
    FakeData,
    FakeEvent,
    FakeRuntime,
    FakeRuntimeInitializer,
    ErrorThrowingRuntime,
    ErrorThrowingRuntimeInitializer,
    FaultyCreateRuntimeInitializer,
    BroadcastTestFakeRuntime,
    BroadcastTestFakeRuntimeInitializer,
)
from tsercom.test.discovery_fixtures import (
    DiscoveryTestClient,
    SelectiveDiscoveryClient,
    UpdateTestClient,
    MultiDiscoveryTestClient,
)
from tsercom.test.data_fixtures import (
    MockTensorDemuxerClient,
    mock_client,
    demuxer,
    demuxer_short_timeout,
)

__all__ = [
    # From loop_fixtures
    "manage_tsercom_loop",
    "clear_loop_fixture",
    # From runtime_fixtures
    "FakeData",
    "FakeEvent",
    "FakeRuntime",
    "FakeRuntimeInitializer",
    "ErrorThrowingRuntime",
    "ErrorThrowingRuntimeInitializer",
    "FaultyCreateRuntimeInitializer",
    "BroadcastTestFakeRuntime",
    "BroadcastTestFakeRuntimeInitializer",
    # From discovery_fixtures
    "DiscoveryTestClient",
    "SelectiveDiscoveryClient",
    "UpdateTestClient",
    "MultiDiscoveryTestClient",
    # From data_fixtures
    "MockTensorDemuxerClient",
    "mock_client",
    "demuxer",
    "demuxer_short_timeout",
]
