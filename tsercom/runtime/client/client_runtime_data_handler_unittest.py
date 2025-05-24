"""Tests for ClientRuntimeDataHandler."""

import pytest
from unittest import mock

from tsercom.runtime.client.client_runtime_data_handler import (
    ClientRuntimeDataHandler,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.data.remote_data_reader import (
    RemoteDataReader,
)  # For type hint if needed by mock
from tsercom.threading.async_poller import AsyncPoller
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


class TestClientRuntimeDataHandler:
    """Tests for the ClientRuntimeDataHandler class."""

    @pytest.fixture
    def mock_thread_watcher(self):
        return mock.Mock(spec=ThreadWatcher)

    @pytest.fixture
    def mock_data_reader(self):
        return mock.Mock(spec=RemoteDataReader)

    @pytest.fixture
    def mock_event_source_poller(self):
        return mock.Mock(spec=AsyncPoller)

    @pytest.fixture
    def mock_time_sync_tracker_instance(self):
        return mock.Mock(spec=TimeSyncTracker)

    @pytest.fixture
    def mock_id_tracker_instance(self):
        return mock.Mock(spec=IdTracker)

    @pytest.fixture
    def mock_endpoint_data_processor(self):
        return mock.Mock(spec=EndpointDataProcessor)

    @pytest.fixture
    def handler(
        self,
        mock_thread_watcher,
        mock_data_reader,
        mock_event_source_poller,
        mock_time_sync_tracker_instance,
        mock_id_tracker_instance,
    ):
        with (
            mock.patch(
                "tsercom.runtime.client.client_runtime_data_handler.TimeSyncTracker",
                return_value=mock_time_sync_tracker_instance,
            ) as mock_ts_tracker_class,
            mock.patch(
                "tsercom.runtime.client.client_runtime_data_handler.IdTracker",
                return_value=mock_id_tracker_instance,
            ) as mock_id_tracker_class,
        ):

            handler_instance = ClientRuntimeDataHandler(
                thread_watcher=mock_thread_watcher,
                data_reader=mock_data_reader,
                event_source=mock_event_source_poller,
            )
            handler_instance._mock_ts_tracker_class = mock_ts_tracker_class
            handler_instance._mock_id_tracker_class = mock_id_tracker_class
            handler_instance._mock_time_sync_tracker_instance = (
                mock_time_sync_tracker_instance
            )
            handler_instance._mock_id_tracker_instance = (
                mock_id_tracker_instance
            )
            return handler_instance

    def test_init(
        self,
        handler,
        mock_thread_watcher,
        mock_data_reader,
        mock_event_source_poller,
    ):
        handler._mock_ts_tracker_class.assert_called_once_with(
            mock_thread_watcher
        )
        handler._mock_id_tracker_class.assert_called_once()

        assert (
            handler._ClientRuntimeDataHandler__clock_tracker
            == handler._mock_time_sync_tracker_instance
        )
        assert (
            handler._ClientRuntimeDataHandler__id_tracker
            == handler._mock_id_tracker_instance
        )

        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert (
            handler._RuntimeDataHandlerBase__event_source
            == mock_event_source_poller
        )

    def test_register_caller(self, handler, mock_endpoint_data_processor):
        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345

        mock_synchronized_clock = mock.Mock(spec=SynchronizedClock)
        handler._mock_time_sync_tracker_instance.on_connect.return_value = (
            mock_synchronized_clock
        )

        with mock.patch.object(
            handler,
            "_create_data_processor",
            return_value=mock_endpoint_data_processor,
        ) as mock_create_dp:
            returned_processor = handler._register_caller(
                mock_caller_id, mock_endpoint, mock_port
            )

        handler._mock_id_tracker_instance.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        handler._mock_time_sync_tracker_instance.on_connect.assert_called_once_with(
            mock_endpoint
        )
        mock_create_dp.assert_called_once_with(
            mock_caller_id, mock_synchronized_clock
        )
        assert returned_processor == mock_endpoint_data_processor

    def test_unregister_caller(self, handler):
        mock_caller_id = CallerIdentifier.random()
        mock_address = "192.168.1.200"
        mock_port = 54321

        handler._mock_id_tracker_instance.get.return_value = (
            mock_address,
            mock_port,
        )

        with pytest.raises(AssertionError) as excinfo:
            handler._unregister_caller(mock_caller_id)

        handler._mock_id_tracker_instance.get.assert_called_once_with(
            mock_caller_id
        )

        # self.__clock_tracker.on_disconnect(address) is NOT called due to the assert False
        handler._mock_time_sync_tracker_instance.on_disconnect.assert_not_called()

        assert "Find out if I should be keeping or deleting these?" in str(
            excinfo.value
        )

    # Remove the old test_unregister_caller as its testing an assert False
    # The new tests will cover the updated logic.

    def test_unregister_caller_valid_id(self, handler):
        """Test _unregister_caller with a valid and existing caller_id."""
        mock_caller_id = CallerIdentifier.random()
        mock_address = "192.168.1.100"
        mock_port = 12345
        
        # Configure mocks
        handler._mock_id_tracker_instance.try_get.return_value = (mock_address, mock_port)
        # remove() should return True if successful, though _unregister_caller doesn't check this return
        handler._mock_id_tracker_instance.remove.return_value = True 

        result = handler._unregister_caller(mock_caller_id)

        assert result is True
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(mock_caller_id)
        handler._mock_id_tracker_instance.remove.assert_called_once_with(mock_caller_id)
        handler._mock_time_sync_tracker_instance.on_disconnect.assert_called_once_with(mock_address)

    def test_unregister_caller_invalid_id_not_found(self, handler):
        """Test _unregister_caller with a non-existent caller_id."""
        mock_caller_id = CallerIdentifier.random()
        
        # Configure mocks
        handler._mock_id_tracker_instance.try_get.return_value = None

        # Patch logging.warning for this specific test
        with mock.patch("tsercom.runtime.client.client_runtime_data_handler.logging") as mock_logging:
            result = handler._unregister_caller(mock_caller_id)

        assert result is False
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(mock_caller_id)
        handler._mock_id_tracker_instance.remove.assert_not_called()
        handler._mock_time_sync_tracker_instance.on_disconnect.assert_not_called()
        mock_logging.warning.assert_called_once_with(
            f"Attempted to unregister non-existent caller_id: {mock_caller_id}"
        )
        
    def test_try_get_caller_id(self, handler):
        mock_endpoint = "10.0.0.1"
        mock_port = 8080
        expected_caller_id = CallerIdentifier.random()

        handler._mock_id_tracker_instance.try_get.return_value = (
            expected_caller_id
        )

        returned_caller_id = handler._try_get_caller_id(
            mock_endpoint, mock_port
        )

        handler._mock_id_tracker_instance.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id == expected_caller_id

    def test_try_get_caller_id_not_found(self, handler):
        mock_endpoint = "10.0.0.2"
        mock_port = 8081

        handler._mock_id_tracker_instance.try_get.return_value = None

        returned_caller_id = handler._try_get_caller_id(
            mock_endpoint, mock_port
        )

        handler._mock_id_tracker_instance.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id is None


# Removed final syntax error.
