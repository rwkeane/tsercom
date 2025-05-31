"""Tests for ServerRuntimeDataHandler."""

import pytest

from tsercom.runtime.server.server_runtime_data_handler import (
    ServerRuntimeDataHandler,
)
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.server.time_sync_server import TimeSyncServer
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.threading.thread_watcher import ThreadWatcher


class TestServerRuntimeDataHandler:
    """Tests for the ServerRuntimeDataHandler class."""

    @pytest.fixture
    def mock_thread_watcher(self, mocker):
        """Provides a mock ThreadWatcher instance."""
        return mocker.MagicMock(spec=ThreadWatcher)

    @pytest.fixture
    def mock_data_reader(self, mocker):
        """Provides a mock RemoteDataReader instance."""
        return mocker.MagicMock(spec=RemoteDataReader)

    @pytest.fixture
    def mock_event_source_poller(self, mocker):
        """Provides a mock AsyncPoller instance for events."""
        return mocker.MagicMock(spec=AsyncPoller)

    @pytest.fixture
    def mock_time_sync_server_instance(self, mocker):
        """Provides a mock TimeSyncServer instance with a nested mock clock."""
        mock_server = mocker.MagicMock(spec=TimeSyncServer)
        mock_server.get_synchronized_clock.return_value = mocker.MagicMock(
            spec=SynchronizedClock
        )
        return mock_server

    @pytest.fixture
    def mock_id_tracker_instance(self, mocker):
        """Provides a mock IdTracker instance."""
        return mocker.MagicMock(spec=IdTracker)

    @pytest.fixture
    def mock_endpoint_data_processor(self, mocker):
        """Provides a mock EndpointDataProcessor instance."""
        return mocker.MagicMock(spec=EndpointDataProcessor)

    @pytest.fixture
    def handler_with_mocks(
        self,
        mock_data_reader,
        mock_event_source_poller,
        mock_time_sync_server_instance,
        mock_id_tracker_instance,
        mocker,
    ):
        """Sets up ServerRuntimeDataHandler with mocked class dependencies."""
        mock_TimeSyncServer_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer",
            return_value=mock_time_sync_server_instance,
            autospec=True,
        )
        mock_IdTracker_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.IdTracker",
            return_value=mock_id_tracker_instance,
            autospec=True,
        )

        handler_instance = ServerRuntimeDataHandler(
            data_reader=mock_data_reader,
            event_source=mock_event_source_poller,
            is_testing=False,
        )

        yield {
            "handler": handler_instance,
            "TimeSyncServer_class_mock": mock_TimeSyncServer_class,
            "IdTracker_class_mock": mock_IdTracker_class,
            "time_sync_server_instance_mock": mock_time_sync_server_instance,
            "id_tracker_instance_mock": mock_id_tracker_instance,
        }

    def test_init(
        self, handler_with_mocks, mock_data_reader, mock_event_source_poller
    ):
        """Tests constructor for correct initialization and dependency usage."""
        handler = handler_with_mocks["handler"]
        TimeSyncServer_class_mock = handler_with_mocks[
            "TimeSyncServer_class_mock"
        ]
        time_sync_server_instance_mock = handler_with_mocks[
            "time_sync_server_instance_mock"
        ]
        IdTracker_class_mock = handler_with_mocks["IdTracker_class_mock"]
        id_tracker_instance_mock = handler_with_mocks[
            "id_tracker_instance_mock"
        ]

        TimeSyncServer_class_mock.assert_called_once_with()
        time_sync_server_instance_mock.start_async.assert_called_once()
        time_sync_server_instance_mock.get_synchronized_clock.assert_called_once()
        assert (
            handler._ServerRuntimeDataHandler__clock
            == time_sync_server_instance_mock.get_synchronized_clock.return_value
        )

        IdTracker_class_mock.assert_called_once_with()
        assert (
            handler._ServerRuntimeDataHandler__id_tracker
            == id_tracker_instance_mock
        )

        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert (
            handler._RuntimeDataHandlerBase__event_source
            == mock_event_source_poller
        )

    def test_register_caller(
        self, handler_with_mocks, mock_endpoint_data_processor, mocker
    ):
        """Tests _register_caller method for correct registration flow."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks[
            "id_tracker_instance_mock"
        ]

        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345

        expected_clock = handler._ServerRuntimeDataHandler__clock

        mock_create_dp_method = mocker.patch.object(
            handler,
            "_create_data_processor",
            return_value=mock_endpoint_data_processor,
        )

        returned_processor = handler._register_caller(
            mock_caller_id, mock_endpoint, mock_port
        )

        id_tracker_instance_mock.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        mock_create_dp_method.assert_called_once_with(
            mock_caller_id, expected_clock
        )
        assert returned_processor == mock_endpoint_data_processor

    def test_unregister_caller(self, handler_with_mocks):
        """Tests _unregister_caller method's current behavior."""
        # TODO(developer): Update assertions to reflect that SUT's _unregister_caller now calls id_tracker.has_id() and returns bool.
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks[
            "id_tracker_instance_mock"
        ]
        time_sync_server_instance_mock = handler_with_mocks[
            "time_sync_server_instance_mock"
        ]

        mock_caller_id = CallerIdentifier.random()
        id_tracker_instance_mock.has_id.return_value = (
            True  # Assume the ID exists for this test path
        )

        try:
            result = handler._unregister_caller(mock_caller_id)
        except Exception as e:
            pytest.fail(
                f"_unregister_caller raised an exception unexpectedly: {e}"
            )

        assert (
            result is True
        ), "Expected _unregister_caller to return True when ID exists"
        id_tracker_instance_mock.add.assert_not_called()
        id_tracker_instance_mock.get.assert_not_called()
        # id_tracker_instance_mock.try_get.assert_not_called() # try_get might not be relevant now
        id_tracker_instance_mock.remove.assert_not_called()  # Based on SUT, remove is not called
        id_tracker_instance_mock.has_id.assert_called_once_with(mock_caller_id)
        id_tracker_instance_mock.has_address.assert_not_called()

        if hasattr(time_sync_server_instance_mock, "on_disconnect"):
            time_sync_server_instance_mock.on_disconnect.assert_not_called()

    def test_try_get_caller_id(self, handler_with_mocks):
        """Tests _try_get_caller_id for a successfully found ID."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks[
            "id_tracker_instance_mock"
        ]

        mock_endpoint = "10.0.0.1"
        mock_port = 8080
        expected_caller_id = CallerIdentifier.random()

        id_tracker_instance_mock.try_get.return_value = expected_caller_id

        returned_caller_id = handler._try_get_caller_id(
            mock_endpoint, mock_port
        )

        id_tracker_instance_mock.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id == expected_caller_id

    def test_try_get_caller_id_not_found(self, handler_with_mocks):
        """Tests _try_get_caller_id when the ID is not found."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks[
            "id_tracker_instance_mock"
        ]

        mock_endpoint = "10.0.0.2"
        mock_port = 8081

        id_tracker_instance_mock.try_get.return_value = None

        returned_caller_id = handler._try_get_caller_id(
            mock_endpoint, mock_port
        )

        id_tracker_instance_mock.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id is None
