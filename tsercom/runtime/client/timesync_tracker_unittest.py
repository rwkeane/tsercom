"""Tests for TimeSyncTracker."""

import pytest

from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.client.time_sync_client import TimeSyncClient
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


class TestTimeSyncTracker:
    """Tests for the TimeSyncTracker class."""

    @pytest.fixture
    def mock_thread_watcher(self, mocker):
        """Provides a mock ThreadWatcher instance."""
        return mocker.MagicMock(spec=ThreadWatcher)

    @pytest.fixture
    def mock_time_sync_client_class(self, mocker):
        """Patches TimeSyncClient and provides the mock class and its instance."""
        mock_class = mocker.patch(
            "tsercom.runtime.client.timesync_tracker.TimeSyncClient",
            autospec=True,
        )
        mock_instance = mock_class.return_value
        mock_instance.get_synchronized_clock.return_value = mocker.MagicMock(
            spec=SynchronizedClock
        )
        yield mock_class

    def test_on_connect_new_ip(
        self, mock_thread_watcher, mock_time_sync_client_class, mocker
    ):
        """Test on_connect when a new IP address is encountered."""
        tracker = TimeSyncTracker(mock_thread_watcher)
        test_ip = "192.168.1.100"

        actual_client_instance_created = (
            mock_time_sync_client_class.return_value
        )
        mock_synchronized_clock = (
            actual_client_instance_created.get_synchronized_clock.return_value
        )

        returned_clock = tracker.on_connect(test_ip)

        mock_time_sync_client_class.assert_called_once_with(
            mock_thread_watcher, test_ip
        )
        actual_client_instance_created.start_async.assert_called_once()
        assert (
            tracker._TimeSyncTracker__map[test_ip][1]
            == actual_client_instance_created
        )
        assert tracker._TimeSyncTracker__map[test_ip][0] == 1
        actual_client_instance_created.get_synchronized_clock.assert_called_once()
        assert returned_clock == mock_synchronized_clock

    def test_on_connect_existing_ip(
        self, mock_thread_watcher, mock_time_sync_client_class, mocker
    ):
        """Test on_connect when an IP address is already being tracked."""
        tracker = TimeSyncTracker(mock_thread_watcher)
        test_ip = "192.168.1.101"

        original_client_instance = mock_time_sync_client_class.return_value
        mock_synchronized_clock = (
            original_client_instance.get_synchronized_clock.return_value
        )

        tracker.on_connect(test_ip)

        mock_time_sync_client_class.assert_called_once_with(
            mock_thread_watcher, test_ip
        )
        original_client_instance.start_async.assert_called_once()
        original_client_instance.get_synchronized_clock.assert_called_once()

        returned_clock_second = tracker.on_connect(test_ip)

        mock_time_sync_client_class.assert_called_once_with(
            mock_thread_watcher, test_ip
        )
        original_client_instance.start_async.assert_called_once()
        assert (
            tracker._TimeSyncTracker__map[test_ip][1]
            == original_client_instance
        )
        assert tracker._TimeSyncTracker__map[test_ip][0] == 2
        assert original_client_instance.get_synchronized_clock.call_count == 2
        assert returned_clock_second == mock_synchronized_clock

    def test_on_disconnect_count_greater_than_one(
        self, mock_thread_watcher, mock_time_sync_client_class, mocker
    ):
        """Tests on_disconnect when ref count > 1; client should not be stopped."""
        tracker = TimeSyncTracker(mock_thread_watcher)
        test_ip = "192.168.1.102"
        client_instance = mock_time_sync_client_class.return_value

        tracker.on_connect(test_ip)
        tracker.on_connect(test_ip)
        assert tracker._TimeSyncTracker__map[test_ip][0] == 2

        tracker.on_disconnect(test_ip)

        assert tracker._TimeSyncTracker__map[test_ip][0] == 1
        client_instance.stop.assert_not_called()
        assert test_ip in tracker._TimeSyncTracker__map

    def test_on_disconnect_count_equals_one(
        self, mock_thread_watcher, mock_time_sync_client_class, mocker
    ):
        """Tests on_disconnect when ref count is 1; client should be stopped."""
        tracker = TimeSyncTracker(mock_thread_watcher)
        test_ip = "192.168.1.103"
        client_instance = mock_time_sync_client_class.return_value

        tracker.on_connect(test_ip)
        assert tracker._TimeSyncTracker__map[test_ip][0] == 1

        tracker.on_disconnect(test_ip)

        assert test_ip not in tracker._TimeSyncTracker__map
        client_instance.stop.assert_called_once()

    def test_on_disconnect_non_existent_ip(self, mock_thread_watcher, mocker):
        """Test on_disconnect with a non-existent IP address."""
        tracker = TimeSyncTracker(mock_thread_watcher)
        test_ip = "192.168.1.104"

        with pytest.raises(
            KeyError, match=f"IP address '{test_ip}' not found"
        ):
            tracker.on_disconnect(test_ip)

    def test_on_connect_different_ips(
        self, mock_thread_watcher, mock_time_sync_client_class, mocker
    ):
        """Tests that different IPs result in different TimeSyncClient instances."""
        tracker = TimeSyncTracker(mock_thread_watcher)
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"

        mock_client_instance1 = mocker.MagicMock(spec=TimeSyncClient)
        mock_clock1 = mocker.MagicMock(spec=SynchronizedClock)
        mock_client_instance1.get_synchronized_clock.return_value = mock_clock1

        mock_client_instance2 = mocker.MagicMock(spec=TimeSyncClient)
        mock_clock2 = mocker.MagicMock(spec=SynchronizedClock)
        mock_client_instance2.get_synchronized_clock.return_value = mock_clock2

        mock_time_sync_client_class.side_effect = [
            mock_client_instance1,
            mock_client_instance2,
        ]

        returned_clock1 = tracker.on_connect(ip1)
        mock_time_sync_client_class.assert_any_call(mock_thread_watcher, ip1)
        mock_client_instance1.start_async.assert_called_once()
        assert tracker._TimeSyncTracker__map[ip1][1] == mock_client_instance1
        assert tracker._TimeSyncTracker__map[ip1][0] == 1
        assert returned_clock1 == mock_clock1

        returned_clock2 = tracker.on_connect(ip2)
        mock_time_sync_client_class.assert_any_call(mock_thread_watcher, ip2)
        mock_client_instance2.start_async.assert_called_once()
        assert tracker._TimeSyncTracker__map[ip2][1] == mock_client_instance2
        assert tracker._TimeSyncTracker__map[ip2][0] == 1
        assert returned_clock2 == mock_clock2

        assert len(tracker._TimeSyncTracker__map) == 2
        assert mock_time_sync_client_class.call_count == 2
