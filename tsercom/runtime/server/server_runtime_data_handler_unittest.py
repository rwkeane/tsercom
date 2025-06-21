"""Tests for ServerRuntimeDataHandler."""

import pytest
import pytest_asyncio  # Added import
import asyncio
import unittest.mock
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop,
    clear_tsercom_event_loop,
)

from tsercom.runtime.server.server_runtime_data_handler import (
    ServerRuntimeDataHandler,
)
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.server.time_sync_server import TimeSyncServer
from tsercom.runtime.id_tracker import IdTracker
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.threading.thread_watcher import ThreadWatcher


class TestServerRuntimeDataHandler:
    """Tests for the ServerRuntimeDataHandler class."""

    # Removed _disabled_manage_event_loop fixture.
    # Assuming pytest-asyncio handles loop setup, and tsercom's global loop
    # will be managed by a central conftest.py fixture or tests will ensure
    # asyncio.get_running_loop() is sufficient for SUT.
    # If RuntimeDataHandlerBase needs explicit global loop setting,
    # a new fixture like manage_tsercom_global_loop (using pytest-asyncio's event_loop)
    # should be added, ideally in a conftest.py.

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

    @pytest_asyncio.fixture
    async def handler_with_mocks(
        self,
        mock_data_reader,
        mock_event_source_poller,
        mock_time_sync_server_instance,  # This mock will be used by ServerRuntimeDataHandler
        mock_id_tracker_instance,
        mocker,
    ):
        """Sets up ServerRuntimeDataHandler with mocked class dependencies."""
        # Ensure stop_async is an AsyncMock on the time_sync_server_instance
        # This is called by ServerRuntimeDataHandler.stop_async()
        mock_time_sync_server_instance.stop_async = mocker.AsyncMock()

        # If ServerRuntimeDataHandler.stop_async() calls super().async_close(),
        # and we want to prevent the real RuntimeDataHandlerBase.async_close()
        # from running (e.g. to avoid its real __dispatch_task logic),
        # then we should mock RuntimeDataHandlerBase.async_close.
        mock_base_async_close = mocker.patch(
            "tsercom.runtime.runtime_data_handler_base.RuntimeDataHandlerBase.async_close",
            new_callable=mocker.AsyncMock,
        )

        mock_TimeSyncServer_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer",
            return_value=mock_time_sync_server_instance,  # This is the mock instance
            autospec=True,
        )
        mock_id_tracker_init = mocker.patch(
            "tsercom.runtime.id_tracker.IdTracker.__init__",
            return_value=None,
        )

        handler_instance = ServerRuntimeDataHandler(
            data_reader=mock_data_reader,
            event_source=mock_event_source_poller,
            is_testing=False,
        )
        # Force set the __id_tracker to our mock instance
        handler_instance._RuntimeDataHandlerBase__id_tracker = mock_id_tracker_instance

        yield {
            "handler": handler_instance,
            "TimeSyncServer_class_mock": mock_TimeSyncServer_class,
            "id_tracker_init_mock": mock_id_tracker_init,
            "time_sync_server_instance_mock": mock_time_sync_server_instance,
            "id_tracker_instance_mock": mock_id_tracker_instance,
            "base_async_close_mock": mock_base_async_close,  # Pass the new mock
        }

        # Teardown
        try:
            # This will call the real ServerRuntimeDataHandler.stop_async()
            # which should call mock_time_sync_server_instance.stop_async() (AsyncMock)
            # and super().async_close() (which is now mock_base_async_close, an AsyncMock)
            await handler_instance.stop_async()
        except Exception as e:
            print(f"Error during handler_with_mocks teardown: {e}")

    @pytest.mark.asyncio
    async def test_init(
        self, handler_with_mocks, mock_data_reader, mock_event_source_poller
    ):
        """Tests constructor for correct initialization and dependency usage."""
        handler = handler_with_mocks["handler"]
        TimeSyncServer_class_mock = handler_with_mocks["TimeSyncServer_class_mock"]
        time_sync_server_instance_mock = handler_with_mocks[
            "time_sync_server_instance_mock"
        ]
        id_tracker_init_mock = handler_with_mocks["id_tracker_init_mock"]
        id_tracker_instance_mock = handler_with_mocks["id_tracker_instance_mock"]

        TimeSyncServer_class_mock.assert_called_once_with()
        time_sync_server_instance_mock.start_async.assert_called_once()
        time_sync_server_instance_mock.get_synchronized_clock.assert_called_once()
        assert (
            handler._ServerRuntimeDataHandler__clock
            == time_sync_server_instance_mock.get_synchronized_clock.return_value
        )

        # Check that IdTracker.__init__ was called (it's called by RuntimeDataHandlerBase)
        id_tracker_init_mock.assert_called_once_with(unittest.mock.ANY)

        # Check that our mock_id_tracker_instance is now the one used by the handler
        assert (
            handler._RuntimeDataHandlerBase__id_tracker  # Corrected attribute
            == id_tracker_instance_mock
        )
        # The following assertion is redundant due to direct assignment but confirms internal state
        assert (
            handler._RuntimeDataHandlerBase__id_tracker  # Corrected attribute
            == id_tracker_instance_mock
        )

        assert handler._RuntimeDataHandlerBase__data_reader == mock_data_reader
        assert handler._RuntimeDataHandlerBase__event_source == mock_event_source_poller

    @pytest.mark.asyncio
    async def test_register_caller(
        self,
        handler_with_mocks,
        mock_endpoint_data_processor,
        mocker,  # handler_with_mocks is an async fixture
    ):
        """Tests _register_caller method for correct registration flow."""
        # Correctly consume the async fixture
        hwm_data = handler_with_mocks  # In pytest-asyncio, the fixture result is directly awaitable if it's a coroutine,
        # or directly usable if it's a regular value from an async fixture.
        # If the async fixture `yields`, it's an async generator.
        # The previous change made it `async def` that `yields`.

        # If handler_with_mocks is an async generator (due to yield), we'd iterate:
        # async for hwm_data_item in handler_with_mocks:
        #    hwm_data = hwm_data_item # Assuming it yields only once
        #    break
        # However, pytest-asyncio might resolve the single yield from an async fixture directly.
        # Let's assume direct result for now, if it's an `async def` fixture that `returns` or `yields once`.
        # The typical pattern for an async fixture that yields once is that the test receives the yielded value.

        # The fixture `handler_with_mocks` yields a dictionary.
        # Pytest should handle awaiting the async fixture and providing the yielded value.
        handler = hwm_data["handler"]
        id_tracker_instance_mock = hwm_data["id_tracker_instance_mock"]

        mock_caller_id = CallerIdentifier.random()
        mock_endpoint = "192.168.1.100"
        mock_port = 12345

        expected_clock = handler._ServerRuntimeDataHandler__clock

        mock_create_dp_method = mocker.patch.object(
            handler,
            "_create_data_processor",
            return_value=mock_endpoint_data_processor,
        )

        returned_processor = await handler._register_caller(
            mock_caller_id, mock_endpoint, mock_port
        )

        id_tracker_instance_mock.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        mock_create_dp_method.assert_called_once_with(mock_caller_id, expected_clock)
        assert returned_processor == mock_endpoint_data_processor

    @pytest.mark.asyncio
    async def test_unregister_caller(self, handler_with_mocks):
        """Tests _unregister_caller method's current behavior."""
        # TODO(developer): Update assertions to reflect that SUT's _unregister_caller now calls id_tracker.has_id() and returns bool.
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks["id_tracker_instance_mock"]
        time_sync_server_instance_mock = handler_with_mocks[
            "time_sync_server_instance_mock"
        ]

        mock_caller_id = CallerIdentifier.random()
        # Do not set id_tracker_instance_mock.has_id.return_value if has_id is not expected to be called.

        try:
            result = await handler._unregister_caller(mock_caller_id)
        except Exception as e:
            pytest.fail(f"_unregister_caller raised an exception unexpectedly: {e}")

        # Assuming the method now simply returns False and does not interact with id_tracker.
        assert (
            result is False
        ), "Expected _unregister_caller to return False (current understanding of its behavior)"
        id_tracker_instance_mock.add.assert_not_called()
        id_tracker_instance_mock.get.assert_not_called()
        # id_tracker_instance_mock.try_get.assert_not_called()
        id_tracker_instance_mock.remove.assert_not_called()  # Verify remove is NOT called
        id_tracker_instance_mock.has_id.assert_not_called()  # Verify has_id is NOT called
        id_tracker_instance_mock.has_address.assert_not_called()

        if hasattr(time_sync_server_instance_mock, "on_disconnect"):
            time_sync_server_instance_mock.on_disconnect.assert_not_called()

    def test_try_get_caller_id(self, handler_with_mocks, mocker):  # Added mocker
        """Tests _try_get_caller_id for a successfully found ID."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks["id_tracker_instance_mock"]

        mock_endpoint = "10.0.0.1"
        mock_port = 8080
        expected_caller_id = CallerIdentifier.random()

        id_tracker_instance_mock.try_get.return_value = (
            expected_caller_id,
            mocker.MagicMock(),  # Mock for the TrackedDataT part
        )

        returned_caller_id = handler._try_get_caller_id(mock_endpoint, mock_port)

        id_tracker_instance_mock.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id == expected_caller_id

    def test_try_get_caller_id_not_found(self, handler_with_mocks):
        """Tests _try_get_caller_id when the ID is not found."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks["id_tracker_instance_mock"]

        mock_endpoint = "10.0.0.2"
        mock_port = 8081

        id_tracker_instance_mock.try_get.return_value = None

        returned_caller_id = handler._try_get_caller_id(mock_endpoint, mock_port)

        id_tracker_instance_mock.try_get.assert_called_once_with(
            mock_endpoint, mock_port
        )
        assert returned_caller_id is None

    def test_init_is_testing_true(
        self, mock_data_reader, mock_event_source_poller, mocker
    ):
        """Tests __init__ with is_testing=True."""
        mock_FakeSynchronizedClock_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.FakeSynchronizedClock"
        )
        mock_fake_clock_instance = mocker.MagicMock(spec=SynchronizedClock)
        mock_FakeSynchronizedClock_class.return_value = mock_fake_clock_instance

        mock_TimeSyncServer_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer"
        )

        # Mock IdTracker.__init__ as it's called in the base class constructor
        mocker.patch("tsercom.runtime.id_tracker.IdTracker.__init__", return_value=None)

        handler = ServerRuntimeDataHandler(
            data_reader=mock_data_reader,
            event_source=mock_event_source_poller,
            is_testing=True,  # Key for this test
        )

        mock_FakeSynchronizedClock_class.assert_called_once_with()
        mock_TimeSyncServer_class.assert_not_called()
        assert handler._ServerRuntimeDataHandler__clock is mock_fake_clock_instance
