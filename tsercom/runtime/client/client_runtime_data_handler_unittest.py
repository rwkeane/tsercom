"""Tests for ClientRuntimeDataHandler."""

import pytest
import asyncio
import unittest.mock
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)

from tsercom.runtime.client.client_runtime_data_handler import (
    ClientRuntimeDataHandler,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.runtime.client.timesync_tracker import TimeSyncTracker
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


class TestClientRuntimeDataHandler:
    """Tests for the ClientRuntimeDataHandler class."""

    @pytest.fixture(autouse=True)
    def manage_event_loop(self):
        """Ensures a global event loop is set for tsercom for each test."""
        loop = None
        try:
            # Try to get existing loop, or create new if none for current context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("existing loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Only set if not already set (by conftest.py or another fixture)
            initial_loop_was_set_by_this_fixture = False
            if not is_global_event_loop_set():
                set_tsercom_event_loop(loop)
                initial_loop_was_set_by_this_fixture = True

            yield loop
        finally:
            if initial_loop_was_set_by_this_fixture:
                clear_tsercom_event_loop()
            # Loop closing logic can remain if tests in this file might create loops
            # for non-tsercom-global purposes, but usually pytest-asyncio handles its own loop well.
            if (
                loop
                and not getattr(loop, "_default_loop", False)
                and not loop.is_closed()
            ):
                if (
                    loop.is_running()
                ):  # This check might be problematic if loop is from higher scope
                    pass  # Avoid stopping pytest-asyncio's loop here; it manages its own lifecycle.
                # loop.close() # Generally, don't close pytest-asyncio's loop.

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
        return mocker.MagicMock(spec=TimeSyncTracker)

    @pytest.fixture
    def mock_id_tracker_instance(self, mocker):
        """Provides a mock IdTracker instance."""
        return mocker.MagicMock(spec=IdTracker)

    @pytest.fixture
    def mock_endpoint_data_processor(self, mocker):
        """Provides a mock EndpointDataProcessor instance."""
        return mocker.MagicMock(spec=EndpointDataProcessor)

    @pytest.fixture
    def handler_and_class_mocks(
        self,
        mock_thread_watcher,
        mock_data_reader,
        mock_event_source_poller,
        mock_time_sync_tracker_instance,
        mock_id_tracker_instance,
        mocker,
    ):
        """Sets up handler instance with mocked class dependencies (TimeSyncTracker, IdTracker)."""
        mock_TimeSyncTracker_class = mocker.patch(
            "tsercom.runtime.client.client_runtime_data_handler.TimeSyncTracker",
            return_value=mock_time_sync_tracker_instance,
            autospec=True,
        )
        mock_id_tracker_init = mocker.patch(
            "tsercom.runtime.id_tracker.IdTracker.__init__",
            return_value=None,  # __init__ should return None
        )

        handler_instance = ClientRuntimeDataHandler(
            thread_watcher=mock_thread_watcher,
            data_reader=mock_data_reader,
            event_source=mock_event_source_poller,
        )
        # Force set the __id_tracker to our mock instance
        handler_instance._RuntimeDataHandlerBase__id_tracker = (
            mock_id_tracker_instance
        )
        handler_instance._mock_time_sync_tracker_instance = (
            mock_time_sync_tracker_instance
        )
        # This direct assignment is for tests to access the mock_id_tracker_instance easily
        # The actual IdTracker instance used by the handler is set above.
        handler_instance._mock_id_tracker_instance = mock_id_tracker_instance

        yield {
            "handler": handler_instance,
            "TimeSyncTracker_class_mock": mock_TimeSyncTracker_class,
            "id_tracker_init_mock": mock_id_tracker_init,
            "time_sync_tracker_instance_mock": mock_time_sync_tracker_instance,
            "id_tracker_instance_mock": mock_id_tracker_instance,
        }

    def test_init(
        self,
        handler_and_class_mocks,
        mock_thread_watcher,
        mock_data_reader,
        mock_event_source_poller,
    ):
        """Tests the __init__ method for correct instantiation and dependency setup."""
        handler = handler_and_class_mocks["handler"]
        TimeSyncTracker_class_mock = handler_and_class_mocks[
            "TimeSyncTracker_class_mock"
        ]
        id_tracker_init_mock = handler_and_class_mocks["id_tracker_init_mock"]
        time_sync_tracker_instance_mock = handler_and_class_mocks[
            "time_sync_tracker_instance_mock"
        ]
        id_tracker_instance_mock = handler_and_class_mocks[
            "id_tracker_instance_mock"
        ]

        TimeSyncTracker_class_mock.assert_called_once_with(
            mock_thread_watcher, is_testing=False
        )
        # Check that IdTracker.__init__ was called (it's called by RuntimeDataHandlerBase)
        id_tracker_init_mock.assert_called_once_with(unittest.mock.ANY)

        # Check that our mock_id_tracker_instance is now the one used by the handler
        assert (
            handler._RuntimeDataHandlerBase__id_tracker  # Corrected attribute
            == id_tracker_instance_mock
        )
        assert (
            handler._ClientRuntimeDataHandler__clock_tracker
            == time_sync_tracker_instance_mock
        )
        # The following assertion is redundant due to direct assignment but confirms internal state
        assert (
            handler._RuntimeDataHandlerBase__id_tracker  # Corrected attribute
            == id_tracker_instance_mock
        )

        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert (
            handler._RuntimeDataHandlerBase__event_source
            == mock_event_source_poller
        )

    @pytest.mark.asyncio
    async def test_register_caller(
        self, handler_and_class_mocks, mock_endpoint_data_processor, mocker
    ):
        """Tests the _register_caller method for correct registration flow."""
        handler = handler_and_class_mocks["handler"]
        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345

        mock_synchronized_clock = mocker.MagicMock(spec=SynchronizedClock)
        handler._mock_time_sync_tracker_instance.on_connect.return_value = (
            mock_synchronized_clock
        )

        mock_create_dp_method = mocker.patch.object(
            handler,
            "_create_data_processor",
            return_value=mock_endpoint_data_processor,
        )

        returned_processor = await handler._register_caller(
            mock_caller_id, mock_endpoint, mock_port
        )

        handler._mock_id_tracker_instance.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        handler._mock_time_sync_tracker_instance.on_connect.assert_called_once_with(
            mock_endpoint
        )
        mock_create_dp_method.assert_called_once_with(
            mock_caller_id, mock_synchronized_clock
        )
        assert returned_processor == mock_endpoint_data_processor

    @pytest.mark.asyncio
    async def test_unregister_caller_valid_id(
        self, handler_and_class_mocks, mocker
    ):  # Added mocker
        """Test _unregister_caller with a valid and existing caller_id."""
        handler = handler_and_class_mocks["handler"]
        mock_caller_id = CallerIdentifier.random()
        mock_address = "192.168.1.100"
        mock_port = 12345

        handler._mock_id_tracker_instance.try_get.return_value = (
            mock_address,
            mock_port,
            mocker.MagicMock(),  # Added third element for poller
        )
        handler._mock_id_tracker_instance.remove.return_value = True

        result = await handler._unregister_caller(mock_caller_id)

        assert result is True
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(
            mock_caller_id
        )
        handler._mock_id_tracker_instance.remove.assert_called_once_with(
            mock_caller_id
        )
        handler._mock_time_sync_tracker_instance.on_disconnect.assert_called_once_with(
            mock_address
        )

    @pytest.mark.asyncio
    async def test_unregister_caller_invalid_id_not_found(
        self, handler_and_class_mocks, mocker
    ):
        """Test _unregister_caller with a non-existent caller_id."""
        handler = handler_and_class_mocks["handler"]
        mock_caller_id = CallerIdentifier.random()

        handler._mock_id_tracker_instance.try_get.return_value = None

        mock_logging_module = mocker.patch(
            "tsercom.runtime.client.client_runtime_data_handler.logging"
        )

        result = await handler._unregister_caller(mock_caller_id)

        assert result is False
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(
            mock_caller_id
        )
        handler._mock_id_tracker_instance.remove.assert_not_called()
        handler._mock_time_sync_tracker_instance.on_disconnect.assert_not_called()
        mock_logging_module.warning.assert_called_once_with(
            "Attempted to unregister non-existent caller_id: %s",
            mock_caller_id,
        )

    def test_try_get_caller_id(
        self, handler_and_class_mocks, mocker
    ):  # Added mocker
        """Tests _try_get_caller_id for a successfully found ID."""
        handler = handler_and_class_mocks["handler"]
        mock_endpoint = "10.0.0.1"
        mock_port = 8080
        expected_caller_id = CallerIdentifier.random()

        # _try_get_caller_id in base class now expects a 2-tuple from id_tracker.try_get
        # but returns only the first element (CallerIdentifier)
        handler._mock_id_tracker_instance.try_get.return_value = (
            expected_caller_id,
            mocker.MagicMock(),  # Mock for the TrackedDataT part
        )

        returned_caller_id = handler._try_get_caller_id(
            mock_endpoint, mock_port
        )

        handler._mock_id_tracker_instance.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id == expected_caller_id

    def test_try_get_caller_id_not_found(self, handler_and_class_mocks):
        """Tests _try_get_caller_id when the ID is not found."""
        handler = handler_and_class_mocks["handler"]
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

    @pytest.mark.asyncio
    async def test_unregister_caller_id_tracker_remove_fails(
        self, handler_and_class_mocks, mocker
    ):
        """
        Tests _unregister_caller when IdTracker.try_get finds the caller,
        but IdTracker.remove returns False.
        """
        handler = handler_and_class_mocks["handler"]
        mock_caller_id = CallerIdentifier.random()
        mock_address = "192.168.1.101"
        mock_port = 54321

        # Configure IdTracker.try_get to return a found caller
        handler._mock_id_tracker_instance.try_get.return_value = (
            mock_address,
            mock_port,
            mocker.MagicMock(),  # Mock for the poller part of the tuple
        )
        # Configure IdTracker.remove to return False
        handler._mock_id_tracker_instance.remove.return_value = False

        # Patch logging
        mock_logging_module = mocker.patch(
            "tsercom.runtime.client.client_runtime_data_handler.logging"
        )

        result = await handler._unregister_caller(mock_caller_id)

        # Assertions
        assert result is False
        handler._mock_id_tracker_instance.try_get.assert_called_once_with(
            mock_caller_id
        )
        handler._mock_id_tracker_instance.remove.assert_called_once_with(
            mock_caller_id
        )
        mock_logging_module.warning.assert_called_once_with(
            "Failed to remove caller_id %s from IdTracker, though it was initially found. "
            "Skipping clock_tracker.on_disconnect.",
            mock_caller_id,
        )
        # Assert that TimeSyncTracker.on_disconnect was NOT called
        handler._mock_time_sync_tracker_instance.on_disconnect.assert_not_called()
