"""Tests for ServerRuntimeDataHandler."""

import pytest
from unittest.mock import patch, MagicMock, ANY  # Added ANY

from tsercom.runtime.server.server_runtime_data_handler import (
    ServerRuntimeDataHandler,
)
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.server.time_sync_server import TimeSyncServer
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.caller_processor_registry import (
    CallerProcessorRegistry,
)  # Added
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.threading.thread_watcher import ThreadWatcher

# For mocking the nested __ConcreteDataProcessor
from tsercom.runtime.runtime_data_handler_base import RuntimeDataHandlerBase


class TestServerRuntimeDataHandler:
    """Tests for the ServerRuntimeDataHandler class."""

    @pytest.fixture
    def mock_thread_watcher(self, mocker):
        """Provides a mock ThreadWatcher instance."""
        return mocker.MagicMock(spec=ThreadWatcher)  # No change

    @pytest.fixture
    def mock_data_reader(self, mocker):
        """Provides a mock RemoteDataReader instance."""
        return mocker.MagicMock(spec=RemoteDataReader)  # No change

    @pytest.fixture
    def mock_event_source_poller(self, mocker):
        """Provides a mock AsyncPoller instance for events."""
        return mocker.MagicMock(spec=AsyncPoller)  # No change

    @pytest.fixture
    def mock_time_sync_server_instance(self, mocker):
        """Provides a mock TimeSyncServer instance with a nested mock clock."""
        mock_server = mocker.MagicMock(spec=TimeSyncServer)
        mock_server.get_synchronized_clock.return_value = mocker.MagicMock(
            spec=SynchronizedClock
        )
        return mock_server

    # mock_id_tracker_instance is removed as IdTracker is created internally now
    # mock_endpoint_data_processor can stay if used by other tests directly

    @pytest.fixture
    def mock_endpoint_data_processor(
        self, mocker
    ):  # No change, might be used by specific tests
        """Provides a mock EndpointDataProcessor instance."""
        return mocker.MagicMock(spec=EndpointDataProcessor)

    @pytest.fixture
    def handler_with_mocks(
        self,
        mock_data_reader,
        mock_event_source_poller,
        mock_time_sync_server_instance,
        # mock_id_tracker_instance, # Removed
        mocker,
    ):
        """Sets up ServerRuntimeDataHandler with mocked class dependencies (IdTracker, CallerProcessorRegistry)."""

        # Mock IdTracker constructor
        mock_id_tracker_instance = mocker.MagicMock(spec=IdTracker)
        mock_IdTracker_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.IdTracker",
            return_value=mock_id_tracker_instance,
            autospec=True,
        )

        # Mock CallerProcessorRegistry constructor
        mock_processor_registry_instance = mocker.MagicMock(
            spec=CallerProcessorRegistry
        )
        mock_CallerProcessorRegistry_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.CallerProcessorRegistry",
            return_value=mock_processor_registry_instance,
            autospec=True,
        )

        # Mock TimeSyncServer constructor (already done in original test)
        mock_TimeSyncServer_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer",
            return_value=mock_time_sync_server_instance,
            autospec=True,
        )

        # Patch asyncio.create_task from RuntimeDataHandlerBase's __init__
        with patch("asyncio.create_task") as mock_async_create_task:
            handler_instance = ServerRuntimeDataHandler(
                data_reader=mock_data_reader,
                event_source=mock_event_source_poller,
                # id_tracker is no longer a constructor argument
                is_testing=False,
            )

        yield {
            "handler": handler_instance,
            "TimeSyncServer_class_mock": mock_TimeSyncServer_class,
            "IdTracker_class_mock": mock_IdTracker_class,
            "CallerProcessorRegistry_class_mock": mock_CallerProcessorRegistry_class,
            "time_sync_server_instance_mock": mock_time_sync_server_instance,
            "id_tracker_instance_mock": mock_id_tracker_instance,  # This is the instance returned by the patched IdTracker()
            "processor_registry_instance_mock": mock_processor_registry_instance,
            "async_create_task_mock": mock_async_create_task,
        }

    def test_init_and_processor_factory(  # Renamed and expanded
        self,
        handler_with_mocks,
        mock_data_reader,
        mock_event_source_poller,
        mocker,
    ):
        """Tests constructor, internal IdTracker/CallerProcessorRegistry creation, and processor_factory logic."""
        handler = handler_with_mocks["handler"]
        TimeSyncServer_class_mock = handler_with_mocks[
            "TimeSyncServer_class_mock"
        ]
        time_sync_server_instance_mock = handler_with_mocks[
            "time_sync_server_instance_mock"
        ]
        IdTracker_class_mock = handler_with_mocks["IdTracker_class_mock"]
        CallerProcessorRegistry_class_mock = handler_with_mocks[
            "CallerProcessorRegistry_class_mock"
        ]

        # Assertions for TimeSyncServer and IdTracker creation (internal)
        TimeSyncServer_class_mock.assert_called_once_with()
        time_sync_server_instance_mock.start_async.assert_called_once()
        time_sync_server_instance_mock.get_synchronized_clock.assert_called_once()
        assert (
            handler._ServerRuntimeDataHandler__clock
            == time_sync_server_instance_mock.get_synchronized_clock.return_value
        )

        IdTracker_class_mock.assert_called_once_with()
        # Check if the handler's __id_tracker is the one from the mocked constructor
        assert (
            handler._ServerRuntimeDataHandler__id_tracker
            == IdTracker_class_mock.return_value
        )

        # Assert CallerProcessorRegistry creation and capture the factory
        CallerProcessorRegistry_class_mock.assert_called_once()
        factory_arg = CallerProcessorRegistry_class_mock.call_args[1][
            "processor_factory"
        ]
        assert callable(factory_arg)

        # Test the captured factory
        mock_test_caller_id = CallerIdentifier.random()

        # Mock what _create_data_processor would return: a __ConcreteDataProcessor instance
        # This mock needs to have the __internal_poller attribute (mangled name)
        mock_concrete_processor = mocker.MagicMock(
            spec=RuntimeDataHandlerBase._RuntimeDataHandlerBase__ConcreteDataProcessor
        )
        mock_internal_poller = mocker.MagicMock(spec=AsyncPoller)
        # Set the mangled name attribute on the mock
        setattr(
            mock_concrete_processor,
            "_RuntimeDataHandlerBase__ConcreteDataProcessor__internal_poller",
            mock_internal_poller,
        )

        # Patch handler's _create_data_processor to control its output for the factory test
        mocker.patch.object(
            handler,
            "_create_data_processor",
            return_value=mock_concrete_processor,
        )

        # Call the factory
        returned_poller = factory_arg(mock_test_caller_id)

        # Assert handler._create_data_processor was called by the factory
        handler._create_data_processor.assert_called_once_with(
            mock_test_caller_id, handler._ServerRuntimeDataHandler__clock
        )
        # Assert the factory returned the internal poller
        assert returned_poller is mock_internal_poller

        # Assert super().__init__ was called correctly (indirectly via checking base class attributes)
        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert handler._event_source == mock_event_source_poller # Corrected attribute access
        # Check that the IdTracker instance created by ServerRuntimeDataHandler was passed to base
        assert handler._id_tracker == IdTracker_class_mock.return_value
        # Check that the CallerProcessorRegistry instance was passed to base
        assert (
            handler._RuntimeDataHandlerBase__processor_registry
            == CallerProcessorRegistry_class_mock.return_value
        )

        # Check that RuntimeDataHandlerBase.__init__ called asyncio.create_task
        handler_with_mocks["async_create_task_mock"].assert_called_once()

    def test_register_caller(
        self,
        handler_with_mocks,
        mocker,  # mock_endpoint_data_processor no longer needed here directly
    ):
        """Tests _register_caller method for correct address mapping and returning ConcreteDataProcessor."""
        handler = handler_with_mocks["handler"]
        # This is the IdTracker instance created *by* ServerRuntimeDataHandler
        id_tracker_instance_mock = (
            handler._ServerRuntimeDataHandler__id_tracker
        )

        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345

        # Mock the _create_data_processor method to return a specific mock EDP
        mock_created_concrete_processor = mocker.MagicMock(
            spec=RuntimeDataHandlerBase._RuntimeDataHandlerBase__ConcreteDataProcessor
        )
        mocker.patch.object(
            handler,
            "_create_data_processor",
            return_value=mock_created_concrete_processor,
        )

        returned_processor = handler._register_caller(
            mock_caller_id, mock_endpoint, mock_port
        )

        # Assert address mapping is done on its own IdTracker
        id_tracker_instance_mock.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        # Assert _create_data_processor was called
        handler._create_data_processor.assert_called_once_with(
            mock_caller_id, handler._ServerRuntimeDataHandler__clock
        )
        # Assert it returns the created concrete processor
        assert returned_processor is mock_created_concrete_processor

    def test_unregister_caller(
        self, handler_with_mocks
    ):  # No change to SUT logic, so test remains same
        """Tests _unregister_caller method's current behavior (no-op)."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = (
            handler._ServerRuntimeDataHandler__id_tracker
        )

        mock_caller_id = CallerIdentifier.random()

        result = handler._unregister_caller(mock_caller_id)

        assert result is False, "Expected _unregister_caller to return False"
        # Ensure no IdTracker modification methods were called
        id_tracker_instance_mock.remove.assert_not_called()

    def test_try_get_caller_id(
        self, handler_with_mocks
    ):  # Ensure correct IdTracker is used
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
