"""Tests for ServerRuntimeDataHandler."""

import pytest
from unittest import mock

from tsercom.runtime.server.server_runtime_data_handler import ServerRuntimeDataHandler
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.annotated_instance import AnnotatedInstance # For type hint
from tsercom.data.serializable_annotated_instance import SerializableAnnotatedInstance # For type hint
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
    def mock_thread_watcher(self):
        return mock.Mock(spec=ThreadWatcher)

    @pytest.fixture
    def mock_data_reader(self):
        return mock.Mock(spec=RemoteDataReader)

    @pytest.fixture
    def mock_event_source_poller(self):
        return mock.Mock(spec=AsyncPoller)

    @pytest.fixture
    def mock_time_sync_server_instance(self):
        mock_server = mock.Mock(spec=TimeSyncServer)
        mock_server.get_synchronized_clock.return_value = mock.Mock(spec=SynchronizedClock)
        return mock_server

    @pytest.fixture
    def mock_id_tracker_instance(self):
        return mock.Mock(spec=IdTracker) 
    
    @pytest.fixture
    def mock_endpoint_data_processor(self):
        return mock.Mock(spec=EndpointDataProcessor)

    @pytest.fixture
    def handler(self, mock_data_reader, mock_event_source_poller, 
                mock_time_sync_server_instance, mock_id_tracker_instance):
        with mock.patch('tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer', return_value=mock_time_sync_server_instance) as mock_ts_server_class, \
             mock.patch('tsercom.runtime.server.server_runtime_data_handler.IdTracker', return_value=mock_id_tracker_instance) as mock_id_tracker_class:
            
            handler_instance = ServerRuntimeDataHandler(
                data_reader=mock_data_reader,
                event_source=mock_event_source_poller
            )
            handler_instance._mock_ts_server_class = mock_ts_server_class
            handler_instance._mock_id_tracker_class = mock_id_tracker_class
            handler_instance._mock_time_sync_server_instance = mock_time_sync_server_instance
            handler_instance._mock_id_tracker_instance = mock_id_tracker_instance
            return handler_instance

    def test_init(self, handler, mock_data_reader, mock_event_source_poller): 
        handler._mock_ts_server_class.assert_called_once_with() 
        handler._mock_time_sync_server_instance.start_async.assert_called_once()
        handler._mock_time_sync_server_instance.get_synchronized_clock.assert_called_once()
        assert handler._ServerRuntimeDataHandler__clock == handler._mock_time_sync_server_instance.get_synchronized_clock.return_value

        handler._mock_id_tracker_class.assert_called_once()
        assert handler._ServerRuntimeDataHandler__id_tracker == handler._mock_id_tracker_instance
        
        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert handler._RuntimeDataHandlerBase__event_source == mock_event_source_poller

    def test_register_caller(self, handler, mock_endpoint_data_processor):
        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345
        
        expected_clock = handler._ServerRuntimeDataHandler__clock

        with mock.patch.object(handler, '_create_data_processor', return_value=mock_endpoint_data_processor) as mock_create_dp:
            returned_processor = handler._register_caller(mock_caller_id, mock_endpoint, mock_port)

        handler._mock_id_tracker_instance.add.assert_called_once_with(mock_caller_id, mock_endpoint, mock_port)
        mock_create_dp.assert_called_once_with(mock_caller_id, expected_clock)
        assert returned_processor == mock_endpoint_data_processor
        
        if hasattr(handler._mock_time_sync_server_instance, 'on_connect'): 
            handler._mock_time_sync_server_instance.on_connect.assert_not_called()


    def test_unregister_caller(self, handler):
        mock_caller_id = CallerIdentifier.random()
        
        try:
            handler._unregister_caller(mock_caller_id)
        except Exception as e:
            pytest.fail(f"_unregister_caller raised an exception unexpectedly: {e}")

        handler._mock_id_tracker_instance.add.assert_not_called()
        handler._mock_id_tracker_instance.get.assert_not_called()
        handler._mock_id_tracker_instance.try_get.assert_not_called()
        handler._mock_id_tracker_instance.has_id.assert_not_called()
        handler._mock_id_tracker_instance.has_address.assert_not_called()

        if hasattr(handler._mock_time_sync_server_instance, 'on_disconnect'): 
             handler._mock_time_sync_server_instance.on_disconnect.assert_not_called()


    def test_try_get_caller_id(self, handler):
        mock_endpoint = "10.0.0.1"
        mock_port = 8080
        expected_caller_id = CallerIdentifier.random()
        
        handler._mock_id_tracker_instance.try_get.return_value = expected_caller_id
        
        returned_caller_id = handler._try_get_caller_id(mock_endpoint, mock_port)
        
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(mock_endpoint, mock_port)
        assert returned_caller_id == expected_caller_id

    def test_try_get_caller_id_not_found(self, handler):
        mock_endpoint = "10.0.0.2"
        mock_port = 8081
        
        handler._mock_id_tracker_instance.try_get.return_value = None 
        
        returned_caller_id = handler._try_get_caller_id(mock_endpoint, mock_port)
        
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(mock_endpoint, mock_port)
        assert returned_caller_id is None
# Removed final syntax error.
