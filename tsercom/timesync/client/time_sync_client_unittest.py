"""Tests for TimeSyncClient."""

import pytest
import time
import threading
from unittest import mock

import ntplib  # For NTPException

from tsercom.timesync.client.time_sync_client import TimeSyncClient
from tsercom.threading.thread_watcher import ThreadWatcher


@pytest.fixture
def mock_thread_watcher_fixture():
    watcher = mock.Mock(spec=ThreadWatcher)
    # If create_tracked_thread is called, make it just run the target directly
    # for simplicity in testing the client's logic, not the watcher's.
    # However, for thread joining, we might need a real thread or a more sophisticated mock.
    # For now, let's assume direct execution or a simple thread.

    def create_thread_side_effect(target, *args, **kwargs):
        # For tests that need the loop to run and then stop, a real thread is better.
        thread = threading.Thread(
            target=target, args=args, kwargs=kwargs, daemon=True
        )
        return thread

    watcher.create_tracked_thread = mock.Mock(
        side_effect=create_thread_side_effect
    )
    return watcher


class TestTimeSyncClientNTPFailures:
    """Tests TimeSyncClient behavior when NTP requests consistently fail."""

    @mock.patch("tsercom.timesync.client.time_sync_client.ntplib.NTPClient")
    def test_start_barrier_not_set_and_no_offsets_if_ntp_fails_consistently(
        self, MockNTPClient, mock_thread_watcher_fixture
    ):
        """
        Test that __start_barrier is not set and no fake offsets are added
        if NTP requests always fail.
        """
        # Configure NTPClient mock to always raise NTPException on request
        mock_ntp_instance = MockNTPClient.return_value
        mock_ntp_instance.request.side_effect = ntplib.NTPException(
            "Mock NTP failure"
        )

        client = TimeSyncClient(mock_thread_watcher_fixture, "dummy_server_ip")
        client.start_async()

        # Allow some time for the sync loop to run and attempt NTP requests.
        # The loop has a 3-second sleep, so we need to wait less than that to catch
        # it before it would successfully get an offset (if it could).
        # We also need to ensure it tries at least once.
        # To make this test faster and more deterministic, we could mock time.sleep
        # in __run_sync_loop, but that's more involved.
        # For now, assume the first request attempt happens quickly.
        time.sleep(0.1)  # Give a little time for the loop to start and try.

        assert (
            not client._TimeSyncClient__start_barrier.is_set()
        ), "Start barrier should not be set if NTP requests fail."
        assert (
            len(client._TimeSyncClient__time_offsets) == 0
        ), "Time offsets should be empty if NTP requests fail."

        client.stop()  # Clean up the client's thread
        # Wait for the thread to actually join
        if client._TimeSyncClient__sync_loop_thread is not None:
            client._TimeSyncClient__sync_loop_thread.join(timeout=1.0)

    @mock.patch("tsercom.timesync.client.time_sync_client.ntplib.NTPClient")
    @mock.patch(
        "tsercom.timesync.client.time_sync_client.time.sleep"
    )  # Mock sleep to speed up test
    def test_get_offset_seconds_blocks_when_barrier_not_set(
        self, mock_time_sleep, MockNTPClient, mock_thread_watcher_fixture
    ):
        """
        Test that get_offset_seconds blocks (does not return quickly)
        if the start barrier is not set due to NTP failures.
        """
        mock_ntp_instance = MockNTPClient.return_value
        mock_ntp_instance.request.side_effect = ntplib.NTPException(
            "Mock NTP failure"
        )

        # Make time.sleep do nothing to speed up the loop for testing
        mock_time_sleep.return_value = None

        client = TimeSyncClient(mock_thread_watcher_fixture, "dummy_server_ip")
        client.start_async()

        # Let the client loop run a few times (mocked sleep makes this fast)
        for _ in range(5):
            if client._TimeSyncClient__start_barrier.is_set():
                break  # Should not happen in this test
            time.sleep(
                0.001
            )  # Tiny sleep to allow context switching if loop is very tight

        assert (
            not client._TimeSyncClient__start_barrier.is_set()
        ), "Pre-condition: Start barrier should not be set."
        assert (
            len(client._TimeSyncClient__time_offsets) == 0
        ), "Pre-condition: Time offsets should be empty."

        # Call get_offset_seconds in a separate thread
        offset_result = []
        exception_result = []

        def target_get_offset():
            try:
                offset = client.get_offset_seconds()
                offset_result.append(offset)
            except Exception as e:
                exception_result.append(e)

        thread = threading.Thread(target=target_get_offset)
        thread.daemon = True  # So it doesn't block test exit
        thread.start()

        # Check that it's blocking
        thread.join(timeout=0.2)  # Give it a moment to potentially return
        assert thread.is_alive(), "get_offset_seconds should be blocking."

        # Clean up
        client.stop()
        thread.join(timeout=1.0)  # Now it should exit after client.stop()

        assert (
            not offset_result
        ), "get_offset_seconds should not have returned a value."
        # Depending on how stop() affects a blocked get_offset_seconds,
        # an exception might be raised (e.g., AssertionError if __time_offsets is empty).
        # This test primarily cares that it blocked *before* stop().
        if exception_result:
            # This is acceptable if stop() causes the wait on barrier to break and then an assert fails
            assert isinstance(
                exception_result[0], AssertionError
            ) or isinstance(
                exception_result[0], RuntimeError
            ), f"Unexpected exception: {exception_result[0]}"
            print(
                f"get_offset_seconds raised {type(exception_result[0])} after stop, which is acceptable."
            )

        # Final check on barrier state after stop
        # Depending on stop logic, barrier might be set or not. Not the primary focus.
        # print(f"Barrier state after stop: {client._TimeSyncClient__start_barrier.is_set()}")

    @mock.patch("tsercom.timesync.client.time_sync_client.ntplib.NTPClient")
    def test_barrier_set_and_offset_available_on_successful_ntp_request(
        self, MockNTPClient, mock_thread_watcher_fixture
    ):
        """Test that the barrier is set and offset is available after a successful NTP request."""
        mock_ntp_instance = MockNTPClient.return_value
        mock_response = mock.Mock()
        mock_response.offset = 0.123  # Example offset
        mock_ntp_instance.request.return_value = mock_response

        client = TimeSyncClient(mock_thread_watcher_fixture, "dummy_server_ip")
        client.start_async()

        # Wait for the barrier to be set. This might take up to kOffsetFrequencySeconds (3s)
        # in the worst case in the original code. For a test, we want it faster.
        # We can wait on the barrier itself with a timeout.
        barrier_set = client._TimeSyncClient__start_barrier.wait(
            timeout=1.0
        )  # Adjust timeout as needed

        assert (
            barrier_set
        ), "Start barrier should be set after successful NTP request."
        assert (
            len(client._TimeSyncClient__time_offsets) > 0
        ), "Time offsets should not be empty after successful NTP request."
        assert client._TimeSyncClient__time_offsets[0] == 0.123

        # Check get_offset_seconds returns the value
        assert client.get_offset_seconds() == 0.123

        client.stop()
        if client._TimeSyncClient__sync_loop_thread is not None:
            client._TimeSyncClient__sync_loop_thread.join(timeout=1.0)
