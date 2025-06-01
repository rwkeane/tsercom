"""Tests for FakeTimeSyncClient."""

import pytest
import threading
import time

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.fake_time_sync_client import FakeTimeSyncClient
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)


@pytest.fixture
def mock_watcher(mocker) -> ThreadWatcher:
    """Fixture for a mocked ThreadWatcher."""
    return mocker.MagicMock(spec=ThreadWatcher)


@pytest.fixture
def client(mock_watcher: ThreadWatcher) -> FakeTimeSyncClient:
    """Fixture for a FakeTimeSyncClient instance."""
    # Use dummy values for server_ip and ntp_port as they are not used by FakeTimeSyncClient
    return FakeTimeSyncClient(mock_watcher, "127.0.0.1", 123)


def test_init(client: FakeTimeSyncClient):
    """Tests basic initialization of FakeTimeSyncClient."""
    assert not client.is_running()
    assert (
        not client._FakeTimeSyncClient__start_barrier.is_set()
    )  # pyright: ignore[reportPrivateUsage]
    assert (
        len(client._FakeTimeSyncClient__time_offsets) == 0
    )  # pyright: ignore[reportPrivateUsage]


def test_start_async(client: FakeTimeSyncClient):
    """Tests the start_async method."""
    client.start_async()
    assert client.is_running()
    assert (
        client._FakeTimeSyncClient__start_barrier.is_set()
    )  # pyright: ignore[reportPrivateUsage]
    # Compare content by converting deque to list
    assert list(client._FakeTimeSyncClient__time_offsets) == [
        0.0
    ]  # pyright: ignore[reportPrivateUsage]

    # Test calling start_async() again when already running
    with pytest.raises(AssertionError, match="Client is already running."):
        client.start_async()

    # Cleanup
    client.stop()


def test_stop(client: FakeTimeSyncClient):
    """Tests the stop method."""
    client.start_async()  # Must be running to stop
    assert client.is_running()

    client.stop()
    assert not client.is_running()
    assert (
        not client._FakeTimeSyncClient__start_barrier.is_set()
    )  # pyright: ignore[reportPrivateUsage]

    # Test calling stop() again when already stopped (should be a no-op)
    try:
        client.stop()
    except Exception as e:
        pytest.fail(
            f"Calling stop() on an already stopped client raised an error: {e}"
        )


def test_is_running(client: FakeTimeSyncClient):
    """Tests the is_running method reflects the correct state."""
    assert not client.is_running()  # Initial state
    client.start_async()
    assert client.is_running()
    client.stop()
    assert not client.is_running()


def test_get_offset_seconds_blocks_if_not_started(client: FakeTimeSyncClient):
    """Tests that get_offset_seconds blocks until the client is started."""
    result = []  # To store result from the thread

    def target():
        result.append(client.get_offset_seconds())

    thread = threading.Thread(target=target)
    thread.start()

    # Assert the thread is alive (blocked) for a short period
    time.sleep(0.1)  # Give some time for the thread to block on the barrier
    assert thread.is_alive()
    assert not result  # Result list should be empty as thread is blocked

    # Start the client to unblock the thread
    client.start_async()
    thread.join(timeout=1.0)  # Wait for the thread to complete

    assert not thread.is_alive()  # Thread should have finished
    assert result == [0.0]

    # Cleanup
    client.stop()


def test_get_offset_seconds_returns_offset_after_started(
    client: FakeTimeSyncClient,
):
    """Tests that get_offset_seconds returns the offset once started."""
    client.start_async()
    assert client.get_offset_seconds() == 0.0
    # Cleanup
    client.stop()


def test_get_offset_seconds_deque_empty_assertion(client: FakeTimeSyncClient):
    """
    Tests assertion if the time_offsets deque is empty after start
    (an unlikely edge case for FakeTimeSyncClient unless manipulated).
    """
    client.start_async()
    # Manually clear the deque after start_async populated it
    client._FakeTimeSyncClient__time_offsets.clear()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(
        AssertionError,
        match="Time offsets deque should not be empty after start.",
    ):
        client.get_offset_seconds()

    # Cleanup
    client.stop()


def test_get_synchronized_clock(client: FakeTimeSyncClient):
    """Tests the get_synchronized_clock method."""
    sync_clock = client.get_synchronized_clock()
    assert isinstance(sync_clock, ClientSynchronizedClock)
    # Check if the client instance within the clock is the FakeTimeSyncClient itself
    assert (
        sync_clock._ClientSynchronizedClock__client is client
    )  # pyright: ignore[reportPrivateUsage]
