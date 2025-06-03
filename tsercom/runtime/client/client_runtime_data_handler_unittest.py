"""Tests for ClientRuntimeDataHandler."""

import pytest
from unittest.mock import patch, MagicMock, ANY

from tsercom.runtime.client.client_runtime_data_handler import (
    ClientRuntimeDataHandler,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.caller_processor_registry import CallerProcessorRegistry # Added
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
# For mocking the nested __ConcreteDataProcessor
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase


class TestClientRuntimeDataHandler:
    """Tests for the ClientRuntimeDataHandler class."""

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
    def mock_time_sync_tracker_instance(self, mocker):
        """Provides a mock TimeSyncTracker instance."""
        tracker = mocker.MagicMock(spec=TimeSyncTracker)
        # Mock the methods that will be called by SUT
        tracker.on_connect = mocker.MagicMock()
        tracker.on_disconnect = mocker.MagicMock()
        tracker.get_clock_for_caller_id = mocker.MagicMock()
        tracker.get_clock_for_endpoint = mocker.MagicMock()
        return tracker

    # mock_id_tracker_instance is removed as IdTracker is created internally

    @pytest.fixture
    def mock_endpoint_data_processor(self, mocker):
        """Provides a mock EndpointDataProcessor instance."""
        return mocker.MagicMock(spec=EndpointDataProcessor)

    @pytest.fixture
    def handler_with_mocks( # Renamed from handler_and_class_mocks for consistency
        self,
        mock_thread_watcher,
        mock_data_reader,
        mock_event_source_poller,
        mock_time_sync_tracker_instance, # This specific instance will be returned by the patched class
        mocker,
    ):
        """Sets up handler instance with mocked class dependencies."""

        mock_id_tracker_instance = mocker.MagicMock(spec=IdTracker)
        mock_IdTracker_class = mocker.patch(
            "tsercom.runtime.client.client_runtime_data_handler.IdTracker",
            return_value=mock_id_tracker_instance,
            autospec=True,
        )

        mock_processor_registry_instance = mocker.MagicMock(spec=CallerProcessorRegistry)
        mock_CallerProcessorRegistry_class = mocker.patch(
            "tsercom.runtime.client.client_runtime_data_handler.CallerProcessorRegistry",
            return_value=mock_processor_registry_instance,
            autospec=True
        )

        mock_TimeSyncTracker_class = mocker.patch(
            "tsercom.runtime.client.client_runtime_data_handler.TimeSyncTracker",
            return_value=mock_time_sync_tracker_instance, # Ensure the specific instance is returned
            autospec=True,
        )

        with patch("asyncio.create_task") as mock_async_create_task:
            handler_instance = ClientRuntimeDataHandler(
                thread_watcher=mock_thread_watcher,
                data_reader=mock_data_reader,
                event_source=mock_event_source_poller,
                # id_tracker is no longer a constructor argument
            )

        # Store mocks for easy access in tests if needed, though patching classes is often cleaner
        # handler_instance._mock_time_sync_tracker_instance = mock_time_sync_tracker_instance
        # handler_instance._mock_id_tracker_instance = mock_id_tracker_instance

        yield {
            "handler": handler_instance,
            "TimeSyncTracker_class_mock": mock_TimeSyncTracker_class,
            "IdTracker_class_mock": mock_IdTracker_class,
            "CallerProcessorRegistry_class_mock": mock_CallerProcessorRegistry_class,
            "time_sync_tracker_instance_mock": mock_time_sync_tracker_instance, # Instance returned by TimeSyncTracker()
            "id_tracker_instance_mock": mock_id_tracker_instance, # Instance returned by IdTracker()
            "processor_registry_instance_mock": mock_processor_registry_instance,
            "async_create_task_mock": mock_async_create_task
        }

    def test_init_and_processor_factory( # Renamed and expanded
        self,
        handler_with_mocks,
        mock_thread_watcher,
        mock_data_reader,
        mock_event_source_poller,
        mocker,
    ):
        """Tests __init__ for correct instantiation, dependency setup, and processor_factory logic."""
        handler = handler_with_mocks["handler"]
        TimeSyncTracker_class_mock = handler_with_mocks["TimeSyncTracker_class_mock"]
        IdTracker_class_mock = handler_with_mocks["IdTracker_class_mock"]
        CallerProcessorRegistry_class_mock = handler_with_mocks["CallerProcessorRegistry_class_mock"]

        time_sync_tracker_instance_mock = handler_with_mocks["time_sync_tracker_instance_mock"]
        id_tracker_instance_mock = handler_with_mocks["id_tracker_instance_mock"]


        TimeSyncTracker_class_mock.assert_called_once_with(
            mock_thread_watcher, is_testing=False # Assuming default is_testing=False
        )
        IdTracker_class_mock.assert_called_once_with()

        assert handler._ClientRuntimeDataHandler__clock_tracker == time_sync_tracker_instance_mock
        assert handler._ClientRuntimeDataHandler__id_tracker == id_tracker_instance_mock

        # Test CallerProcessorRegistry factory
        CallerProcessorRegistry_class_mock.assert_called_once()
        factory_arg = CallerProcessorRegistry_class_mock.call_args[1]['processor_factory']
        assert callable(factory_arg)

        mock_test_caller_id = CallerIdentifier.random()
        mock_clock_for_factory = mocker.MagicMock(spec=SynchronizedClock)
        time_sync_tracker_instance_mock.get_clock_for_caller_id.return_value = mock_clock_for_factory

        mock_concrete_processor = mocker.MagicMock(spec=RuntimeDataHandlerBase._RuntimeDataHandlerBase__ConcreteDataProcessor)
        mock_internal_poller = mocker.MagicMock(spec=AsyncPoller)
        setattr(mock_concrete_processor, '_RuntimeDataHandlerBase__ConcreteDataProcessor__internal_poller', mock_internal_poller)
        mocker.patch.object(handler, '_create_data_processor', return_value=mock_concrete_processor)

        returned_poller = factory_arg(mock_test_caller_id)

        time_sync_tracker_instance_mock.get_clock_for_caller_id.assert_called_once_with(mock_test_caller_id)
        handler._create_data_processor.assert_called_once_with(mock_test_caller_id, mock_clock_for_factory)
        assert returned_poller is mock_internal_poller

        # Base class init checks
        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert handler._RuntimeDataHandlerBase__event_source == mock_event_source_poller
        assert handler._id_tracker == id_tracker_instance_mock # Passed to base
        assert handler._RuntimeDataHandlerBase__processor_registry == handler_with_mocks["processor_registry_instance_mock"]
        handler_with_mocks["async_create_task_mock"].assert_called_once()


    def test_register_caller( # Updated test
        self, handler_with_mocks, mocker
    ):
        """Tests _register_caller for address mapping, clock setup, and returning ConcreteDataProcessor."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler._ClientRuntimeDataHandler__id_tracker
        time_sync_tracker_instance_mock = handler._ClientRuntimeDataHandler__clock_tracker

        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345

        mock_clock_for_endpoint = mocker.MagicMock(spec=SynchronizedClock)
        time_sync_tracker_instance_mock.get_clock_for_endpoint.return_value = mock_clock_for_endpoint

        mock_created_concrete_processor = mocker.MagicMock(spec=RuntimeDataHandlerBase._RuntimeDataHandlerBase__ConcreteDataProcessor)
        mocker.patch.object(handler, '_create_data_processor', return_value=mock_created_concrete_processor)

        returned_processor = handler._register_caller(
            mock_caller_id, mock_endpoint, mock_port
        )

        id_tracker_instance_mock.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        time_sync_tracker_instance_mock.on_connect.assert_called_once_with(
            mock_endpoint, mock_caller_id # Ensure caller_id is passed
        )
        time_sync_tracker_instance_mock.get_clock_for_endpoint.assert_called_once_with(mock_endpoint)
        handler._create_data_processor.assert_called_once_with(
            mock_caller_id, mock_clock_for_endpoint
        )
        assert returned_processor is mock_created_concrete_processor


    def test_unregister_caller_valid_id(self, handler_with_mocks): # Updated to use correct mocks
        """Test _unregister_caller with a valid and existing caller_id."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler._ClientRuntimeDataHandler__id_tracker
        time_sync_tracker_instance_mock = handler._ClientRuntimeDataHandler__clock_tracker

        mock_caller_id = CallerIdentifier.random()
        mock_address = "192.168.1.100"
        mock_port = 12345

        id_tracker_instance_mock.try_get.return_value = (mock_address, mock_port)
        id_tracker_instance_mock.remove.return_value = True # Assume remove is successful

        result = handler._unregister_caller(mock_caller_id)

        assert result is True
        id_tracker_instance_mock.try_get.assert_called_once_with(mock_caller_id)
        id_tracker_instance_mock.remove.assert_called_once_with(mock_caller_id)
        time_sync_tracker_instance_mock.on_disconnect.assert_called_once_with(mock_address)


    def test_unregister_caller_invalid_id_not_found( # Updated to use correct mocks
        self, handler_with_mocks, mocker
    ):
        """Test _unregister_caller with a non-existent caller_id."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler._ClientRuntimeDataHandler__id_tracker
        time_sync_tracker_instance_mock = handler._ClientRuntimeDataHandler__clock_tracker
        mock_caller_id = CallerIdentifier.random()

        id_tracker_instance_mock.try_get.return_value = None

        # Patch logging inside the SUT module
        mock_logging_module = mocker.patch("tsercom.runtime.client.client_runtime_data_handler.logging")

        result = handler._unregister_caller(mock_caller_id)

        assert result is False
        id_tracker_instance_mock.try_get.assert_called_once_with(mock_caller_id)
        id_tracker_instance_mock.remove.assert_not_called()
        time_sync_tracker_instance_mock.on_disconnect.assert_not_called()
        mock_logging_module.warning.assert_called_once_with(
            f"Attempted to unregister non-existent caller_id: {mock_caller_id}"
        )


    def test_try_get_caller_id(self, handler_with_mocks): # Updated to use correct mocks
