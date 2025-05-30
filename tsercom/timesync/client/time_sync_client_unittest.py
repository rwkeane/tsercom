"""Tests for TimeSyncClient."""

import pytest
import time
import threading
from unittest.mock import MagicMock

import ntplib

from tsercom.timesync.client.time_sync_client import TimeSyncClient
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.constants import kNtpVersion
from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)


@pytest.fixture
def mock_thread_watcher_fixture(mocker):
    watcher = mocker.MagicMock(spec=ThreadWatcher)

    def create_thread_side_effect(target, *args, **kwargs):
        thread = threading.Thread(
            target=target, args=args, kwargs=kwargs, daemon=True
        )
        return thread

    watcher.create_tracked_thread = mocker.MagicMock(
        side_effect=create_thread_side_effect
    )
    return watcher


@pytest.fixture
def mock_ntp_client_fixture(mocker):
    """Fixture to mock ntplib.NTPClient."""
    MockNTPClient = mocker.patch(
        "tsercom.timesync.client.time_sync_client.ntplib.NTPClient"
    )
    mock_instance = MockNTPClient.return_value
    # Ensure that the mock_instance.request itself is a mock if it's going to be called
    if not isinstance(mock_instance.request, MagicMock):
        mock_instance.request = MagicMock()
    return mock_instance


@pytest.fixture
def mock_time_sleep_fixture(mocker):
    """Fixture to mock time.sleep to prevent actual sleeping in tests."""
    return mocker.patch(
        "tsercom.timesync.client.time_sync_client.time.sleep",
        return_value=None,
    )


class TestTimeSyncClientNTPFailures:
    """Tests TimeSyncClient behavior when NTP requests consistently fail."""

    @pytest.mark.usefixtures(
        "mock_ntp_client_fixture", "mock_time_sleep_fixture"
    )
    def test_start_barrier_not_set_and_no_offsets_if_ntp_fails_consistently(
        self, mock_thread_watcher_fixture, mock_ntp_client_fixture
    ):
        mock_ntp_client_fixture.request.side_effect = ntplib.NTPException(
            "Mock NTP failure"
        )
        client = TimeSyncClient(mock_thread_watcher_fixture, "dummy_server_ip")
        client.start_async()
        time.sleep(0.01)

        assert (
            not client._TimeSyncClient__start_barrier.is_set()
        ), "Start barrier should not be set if NTP requests fail."
        assert (
            len(client._TimeSyncClient__time_offsets) == 0
        ), "Time offsets should be empty if NTP requests fail."

        client.stop()
        sync_thread = getattr(
            client, "_TimeSyncClient__sync_loop_thread", None
        )
        if sync_thread is not None and sync_thread.is_alive():
            sync_thread.join(timeout=0.5)

    @pytest.mark.usefixtures(
        "mock_ntp_client_fixture", "mock_time_sleep_fixture"
    )
    def test_get_offset_seconds_blocks_when_barrier_not_set(
        self, mock_thread_watcher_fixture, mock_ntp_client_fixture
    ):
        mock_ntp_client_fixture.request.side_effect = ntplib.NTPException(
            "Mock NTP failure"
        )
        client = TimeSyncClient(mock_thread_watcher_fixture, "dummy_server_ip")
        client.start_async()
        time.sleep(0.01)

        assert (
            not client._TimeSyncClient__start_barrier.is_set()
        ), "Pre-condition: Start barrier should not be set."

        offset_result = []
        exception_result = []

        def target_get_offset():
            try:
                offset = client.get_offset_seconds()
                offset_result.append(offset)
            except Exception as e:
                exception_result.append(e)

        thread = threading.Thread(target=target_get_offset, daemon=True)
        thread.start()
        thread.join(timeout=0.2)
        assert thread.is_alive(), "get_offset_seconds should be blocking."

        client.stop()
        thread.join(timeout=0.5)
        assert (
            not offset_result
        ), "get_offset_seconds should not have returned a value."
        if exception_result:
            assert isinstance(
                exception_result[0], (AssertionError, RuntimeError)
            ), f"Unexpected exception: {exception_result[0]}"

    @pytest.mark.usefixtures(
        "mock_ntp_client_fixture", "mock_time_sleep_fixture"
    )
    def test_barrier_set_and_offset_available_on_successful_ntp_request(
        self, mock_thread_watcher_fixture, mock_ntp_client_fixture
    ):
        mock_response = MagicMock()
        mock_response.offset = 0.123
        mock_ntp_client_fixture.request.return_value = mock_response
        client = TimeSyncClient(mock_thread_watcher_fixture, "dummy_server_ip")
        client.start_async()
        barrier_set = client._TimeSyncClient__start_barrier.wait(timeout=1.0)
        assert (
            barrier_set
        ), "Start barrier should be set after successful NTP request."
        assert (
            len(client._TimeSyncClient__time_offsets) > 0
        ), "Time offsets should not be empty after successful NTP request."
        assert client._TimeSyncClient__time_offsets[0] == 0.123
        assert client.get_offset_seconds() == 0.123
        client.stop()
        sync_thread = getattr(
            client, "_TimeSyncClient__sync_loop_thread", None
        )
        if sync_thread is not None and sync_thread.is_alive():
            sync_thread.join(timeout=0.5)


# --- New Test Class for Operations ---
@pytest.mark.usefixtures(
    "mock_thread_watcher_fixture",
    "mock_ntp_client_fixture",
    "mock_time_sleep_fixture",
)
class TestTimeSyncClientOperations:
    """Tests for general operations of TimeSyncClient."""

    @pytest.fixture
    def client(self, mock_thread_watcher_fixture) -> TimeSyncClient:
        client_instance = TimeSyncClient(
            mock_thread_watcher_fixture, "pool.ntp.org", ntp_port=123
        )
        yield client_instance
        if client_instance.is_running():
            client_instance.stop()
        sync_thread = getattr(
            client_instance, "_TimeSyncClient__sync_loop_thread", None
        )
        if sync_thread and sync_thread.is_alive():
            sync_thread.join(timeout=1.0)

    def test_offset_averaging(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        offsets = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_ntp_client_fixture.request.side_effect = [
            MagicMock(offset=o) for o in offsets
        ]
        client.start_async()
        barrier_set = client._TimeSyncClient__start_barrier.wait(timeout=1.0)
        assert barrier_set, "Start barrier was not set after starting client."

        max_wait_cycles = len(offsets) + 20
        cycles = 0
        while (
            mock_ntp_client_fixture.request.call_count < len(offsets)
            and cycles < max_wait_cycles
        ):
            time.sleep(0.001)
            cycles += 1
        assert mock_ntp_client_fixture.request.call_count >= len(
            offsets
        ), f"NTP request not called enough. Expected: {len(offsets)}, Got: {mock_ntp_client_fixture.request.call_count}"
        expected_average = sum(offsets) / len(offsets)
        assert client.get_offset_seconds() == pytest.approx(expected_average)
        assert list(client._TimeSyncClient__time_offsets) == offsets

    def test_max_offset_count(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        # Note: kMaxOffsetCount is a local variable within TimeSyncClient.__run_sync_loop.
        # It's hardcoded to 10 in the source. We test against this known value.
        # Direct patching of this local variable to a different value (e.g., 3) from
        # a test is not straightforward without modifying the source code structure.
        k_max_offset_count_from_source = 10
        num_offsets_to_send = (
            12  # Send more than kMaxOffsetCount to test truncation
        )

        offsets = [0.1 * (i + 1) for i in range(num_offsets_to_send)]
        mock_ntp_client_fixture.request.side_effect = [
            MagicMock(offset=o) for o in offsets
        ]
        client.start_async()
        barrier_set = client._TimeSyncClient__start_barrier.wait(timeout=1.0)
        assert barrier_set, "Start barrier was not set."

        max_wait_cycles = num_offsets_to_send + 20
        cycles = 0
        while (
            mock_ntp_client_fixture.request.call_count < num_offsets_to_send
            and cycles < max_wait_cycles
        ):
            time.sleep(0.001)
            cycles += 1
        assert (
            mock_ntp_client_fixture.request.call_count >= num_offsets_to_send
        ), f"NTP request not called enough. Expected: {num_offsets_to_send}, Got: {mock_ntp_client_fixture.request.call_count}"
        expected_stored_offsets = offsets[-k_max_offset_count_from_source:]
        assert (
            list(client._TimeSyncClient__time_offsets)
            == expected_stored_offsets
        )
        expected_average = sum(expected_stored_offsets) / len(
            expected_stored_offsets
        )
        assert client.get_offset_seconds() == pytest.approx(expected_average)

    def test_stop_terminates_sync_loop(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        mock_ntp_client_fixture.request.return_value = MagicMock(offset=0.1)
        client.start_async()
        sync_loop_thread = client._TimeSyncClient__sync_loop_thread
        assert sync_loop_thread is not None and sync_loop_thread.is_alive()
        client.stop()
        sync_loop_thread.join(timeout=1.0)
        assert not sync_loop_thread.is_alive()

    def test_get_offset_after_stop(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        mock_ntp_client_fixture.request.return_value = MagicMock(offset=0.123)
        client.start_async()
        barrier_set = client._TimeSyncClient__start_barrier.wait(timeout=1.0)
        assert barrier_set, "Start barrier was not set."

        max_wait_cycles = 5
        cycles = 0
        while (
            mock_ntp_client_fixture.request.call_count < 1
            and cycles < max_wait_cycles
        ):
            time.sleep(0.001)
            cycles += 1
        assert (
            mock_ntp_client_fixture.request.call_count >= 1
        ), "NTP request was not called."

        first_offset = client.get_offset_seconds()
        assert first_offset == pytest.approx(0.123)
        client.stop()
        assert client.get_offset_seconds() == pytest.approx(first_offset)
        assert client._TimeSyncClient__start_barrier.is_set()

    def test_assertion_error_in_loop_calls_watcher_and_stops_loop(
        self,
        client: TimeSyncClient,
        mock_ntp_client_fixture,
        mock_thread_watcher_fixture,
    ):
        test_exception = AssertionError("Test assertion in loop")
        mock_ntp_client_fixture.request.side_effect = test_exception
        client.start_async()
        max_wait_cycles = 10
        cycles = 0
        while (
            mock_thread_watcher_fixture.on_exception_seen.call_count == 0
            and cycles < max_wait_cycles
        ):
            time.sleep(0.01)
            cycles += 1
        mock_thread_watcher_fixture.on_exception_seen.assert_called_once_with(
            test_exception
        )
        sync_loop_thread = client._TimeSyncClient__sync_loop_thread
        if sync_loop_thread:
            sync_loop_thread.join(timeout=1.0)
            assert (
                not sync_loop_thread.is_alive()
            ), "Sync loop thread should be dead."
        assert (
            client.is_running()
        ), "is_running is True as loop crash doesn't call client.stop()."
        client.stop()
        assert not client.is_running()

    def test_general_exception_in_loop_logs_and_continues(
        self, client: TimeSyncClient, mock_ntp_client_fixture, mocker
    ):
        mock_logging_error = mocker.patch("logging.error")
        mock_ok_response = MagicMock(offset=0.25)
        test_value_error = ValueError("Test value error in loop")

        def request_side_effect_with_default(*args, **kwargs):
            if mock_ntp_client_fixture.request.call_count == 1:
                raise test_value_error
            return mock_ok_response

        mock_ntp_client_fixture.request.side_effect = (
            request_side_effect_with_default
        )

        client.start_async()
        barrier_set = client._TimeSyncClient__start_barrier.wait(timeout=1.0)
        assert (
            barrier_set
        ), "Start barrier was not set, successful response not processed."

        expected_log_message = f"Error during NTP sync: {test_value_error}"
        call_found = any(
            expected_log_message == call_arg[0][0]
            for call_arg in mock_logging_error.call_args_list
        )
        assert (
            call_found
        ), f"Specific ValueError log ('{expected_log_message}') not found. Logs: {mock_logging_error.call_args_list}"

        specific_error_logs_count = sum(
            1
            for call_arg in mock_logging_error.call_args_list
            if expected_log_message == call_arg[0][0]
        )
        assert (
            specific_error_logs_count == 1
        ), f"Expected 1 log for the specific ValueError, got {specific_error_logs_count}. Logs: {mock_logging_error.call_args_list}"

        assert client.is_running()
        assert client.get_offset_seconds() == pytest.approx(0.25)
        sync_loop_thread = client._TimeSyncClient__sync_loop_thread
        assert sync_loop_thread and sync_loop_thread.is_alive()

    def test_successful_request_after_initial_failures(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        # Explicitly reset the mock for the request method to ensure clean state
        mock_ntp_client_fixture.request.reset_mock(
            return_value=True, side_effect=True
        )

        mock_ok_response = MagicMock(offset=0.333)
        # Test with one failure, then one success
        mock_ok_response = MagicMock(
            offset=0.333
        )  # Ensure mock_ok_response is defined
        second_call_done_event = threading.Event()

        call_number = 0

        def request_side_effect_limited(*args, **kwargs):
            nonlocal call_number
            call_number += 1
            if call_number == 1:
                raise ntplib.NTPException("fail1")
            elif call_number == 2:
                second_call_done_event.set()
                return mock_ok_response
            else:
                # This will be hit on the 3rd call onwards.
                client.stop()  # Reinstate stop to prevent runaway calls
                raise ntplib.NTPException(
                    f"Called more than expected (call_number: {call_number}). Client should have stopped."
                )

        mock_ntp_client_fixture.request.side_effect = (
            request_side_effect_limited
        )

        client.start_async()
        # Allow some time for the sync thread to start and make the first (failing) call.
        # This is a real sleep, not affected by the mock_time_sleep_fixture.
        time.sleep(0.05)

        # Wait for the second call's side effect to signal completion
        assert second_call_done_event.wait(
            timeout=2.0
        ), "The second NTP call (successful one) did not complete as expected."

        # Wait for the client to process the successful response and set the barrier
        # This ensures that the effects of the successful call are visible before stopping.
        barrier_set_successfully = client._TimeSyncClient__start_barrier.wait(
            timeout=1.0
        )  # Using a 1s timeout
        assert (
            barrier_set_successfully
        ), "The start barrier was not set by the client within 1.0s after the successful NTP response."

        client.stop()
        sync_thread = client._TimeSyncClient__sync_loop_thread
        if sync_thread and sync_thread.is_alive():
            sync_thread.join(timeout=1.0)

        assert (
            client._TimeSyncClient__start_barrier.is_set()
        ), "Start barrier was not set."

        assert list(client._TimeSyncClient__time_offsets) == [
            0.333
        ], f"Expected offsets list [0.333], got: {list(client._TimeSyncClient__time_offsets)}"

        assert client.get_offset_seconds() == pytest.approx(
            0.333
        ), "Offset is not the expected 0.333."

        # Total calls should be 2 (one failure, one success), but allow 3 due to potential race condition with mocked sleep
        assert (
            mock_ntp_client_fixture.request.call_count <= 3
        ), f"Expected 2 or 3 NTP requests. Got {mock_ntp_client_fixture.request.call_count}"

        assert (
            not client.is_running()
        ), "Client should not be running after stop()."

    def test_start_async_already_running(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        mock_ntp_client_fixture.request.return_value = MagicMock(offset=0.1)
        client.start_async()
        assert client.is_running()
        with pytest.raises(
            AssertionError, match="TimeSyncClient is already running."
        ):
            client.start_async()

    def test_get_synchronized_clock(self, client: TimeSyncClient):
        sync_clock = client.get_synchronized_clock()
        assert isinstance(sync_clock, ClientSynchronizedClock)
        assert sync_clock._ClientSynchronizedClock__client is client

    def test_is_running(self, client: TimeSyncClient, mock_ntp_client_fixture):
        assert not client.is_running()
        mock_ntp_client_fixture.request.return_value = MagicMock(offset=0.1)
        client.start_async()
        assert client.is_running()
        client.stop()
        assert not client.is_running()

    def test_ntp_request_uses_configured_version_and_params(
        self, client: TimeSyncClient, mock_ntp_client_fixture
    ):
        mock_ntp_client_fixture.request.return_value = MagicMock(offset=0.1)

        # Reset mock before start to isolate calls for this test
        mock_ntp_client_fixture.request.reset_mock()

        client.start_async()
        barrier_set = client._TimeSyncClient__start_barrier.wait(timeout=1.0)
        assert barrier_set, "Start barrier was not set."

        server_ip_used = client._TimeSyncClient__server_ip
        ntp_port_used = client._TimeSyncClient__ntp_port

        # Check the first call (or any subsequent call, as parameters should be consistent)
        mock_ntp_client_fixture.request.assert_any_call(
            server_ip_used,  # host is positional
            version=kNtpVersion,
            port=ntp_port_used,
        )
