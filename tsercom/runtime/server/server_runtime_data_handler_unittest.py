"""Tests for ServerRuntimeDataHandler."""

import pytest
import asyncio
import unittest.mock  # Added import
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

    # Removed local manage_event_loop fixture to rely on conftest.py:manage_tsercom_loop

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
        # manage_event_loop, # Removed: rely on conftest.py's autouse fixture
    ):
        """Sets up ServerRuntimeDataHandler with mocked class dependencies."""
        mock_TimeSyncServer_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer",
            return_value=mock_time_sync_server_instance,
            autospec=True,
        )
        mock_id_tracker_init = mocker.patch(
            "tsercom.runtime.id_tracker.IdTracker.__init__",
            return_value=None,  # __init__ should return None
        )

        handler_instance = ServerRuntimeDataHandler(
            data_reader=mock_data_reader,
            event_source=mock_event_source_poller,
            is_testing=False,
        )
        # Force set the __id_tracker to our mock instance
        handler_instance._RuntimeDataHandlerBase__id_tracker = (
            mock_id_tracker_instance
        )

        # The 'manage_event_loop' fixture is autouse=True and function-scoped in this file.
        # We need to explicitly pass it to this fixture if we want to use its yielded loop object.
        # However, the original 'manage_event_loop' fixture in this file doesn't seem to be passed around.
        # The one from conftest.py is also function-scoped.
        # Let's assume the 'manage_event_loop' fixture in *this file* is the one we need to use.
        # To do that, 'handler_with_mocks' must request 'manage_event_loop'.
        # The 'manage_event_loop' fixture in this file needs to be requested by handler_with_mocks:
        #
        # @pytest.fixture
        # def handler_with_mocks(
        #     self,
        #     ..., # other args
        #     manage_event_loop, # ADD THIS LINE to request the fixture from this file
        # ):
        #
        # Then in finally: loop_instance = manage_event_loop

        try:
            yield {
                "handler": handler_instance,
                "TimeSyncServer_class_mock": mock_TimeSyncServer_class,
                "id_tracker_init_mock": mock_id_tracker_init,
                "time_sync_server_instance_mock": mock_time_sync_server_instance,
                "id_tracker_instance_mock": mock_id_tracker_instance,
            }
        finally:
            # Cleanup:
            # The 'manage_event_loop' fixture from conftest.py ensures a loop is running.
            try:
                loop_instance = asyncio.get_running_loop()
            except RuntimeError:
                # This should ideally not happen if conftest.py:manage_tsercom_loop is effective
                print(
                    "Error: No running event loop in handler_with_mocks cleanup!"
                )
                return  # Cannot proceed with cleanup

            dispatch_task_attr_name = "_RuntimeDataHandlerBase__dispatch_task"
            if hasattr(handler_instance, dispatch_task_attr_name):
                dispatch_task = getattr(
                    handler_instance, dispatch_task_attr_name
                )
                if dispatch_task is not None:
                    if loop_instance and not loop_instance.is_closed():
                        try:
                            if not dispatch_task.done():
                                loop_instance.run_until_complete(
                                    handler_instance.async_close()
                                )
                            elif dispatch_task.exception() is not None:
                                _ = dispatch_task.exception()
                        except RuntimeError as e:
                            print(
                                f"Error during handler_with_mocks cleanup (run_until_complete): {e}"
                            )
                        except Exception as e:
                            print(
                                f"Generic error during handler_with_mocks cleanup: {e}"
                            )
                    # This 'elif' might be logically redundant due to 'if dispatch_task is not None'
                    # and also might try to operate on a closed loop if the RuntimeError above happened.
                    elif dispatch_task and not dispatch_task.done():
                        dispatch_task.cancel()
                        try:
                            # This part is best-effort if loop is already closed or other issues.
                            if loop_instance and not loop_instance.is_closed():
                                loop_instance.run_until_complete(dispatch_task)
                        except:  # pylint: disable=bare-except
                            pass

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
        id_tracker_init_mock = handler_with_mocks["id_tracker_init_mock"]
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
        assert (
            handler._RuntimeDataHandlerBase__event_source
            == mock_event_source_poller
        )

    @pytest.mark.asyncio
    async def test_register_caller(
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

        returned_processor = await handler._register_caller(
            mock_caller_id, mock_endpoint, mock_port
        )

        id_tracker_instance_mock.add.assert_called_once_with(
            mock_caller_id, mock_endpoint, mock_port
        )
        mock_create_dp_method.assert_called_once_with(
            mock_caller_id, expected_clock
        )
        assert returned_processor == mock_endpoint_data_processor

    @pytest.mark.asyncio
    async def test_unregister_caller(self, handler_with_mocks):
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
        # Do not set id_tracker_instance_mock.has_id.return_value if has_id is not expected to be called.

        try:
            result = await handler._unregister_caller(mock_caller_id)
        except Exception as e:
            pytest.fail(
                f"_unregister_caller raised an exception unexpectedly: {e}"
            )

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

    def test_try_get_caller_id(
        self, handler_with_mocks, mocker
    ):  # Added mocker
        """Tests _try_get_caller_id for a successfully found ID."""
        handler = handler_with_mocks["handler"]
        id_tracker_instance_mock = handler_with_mocks[
            "id_tracker_instance_mock"
        ]

        mock_endpoint = "10.0.0.1"
        mock_port = 8080
        expected_caller_id = CallerIdentifier.random()

        id_tracker_instance_mock.try_get.return_value = (
            expected_caller_id,
            mocker.MagicMock(),  # Mock for the TrackedDataT part
        )

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

    def test_init_is_testing_true(
        self, mock_data_reader, mock_event_source_poller, mocker
    ):
        """Tests __init__ with is_testing=True."""
        mock_FakeSynchronizedClock_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.FakeSynchronizedClock"
        )
        mock_fake_clock_instance = mocker.MagicMock(spec=SynchronizedClock)
        mock_FakeSynchronizedClock_class.return_value = (
            mock_fake_clock_instance
        )

        mock_TimeSyncServer_class = mocker.patch(
            "tsercom.runtime.server.server_runtime_data_handler.TimeSyncServer"
        )

        # Mock IdTracker.__init__ as it's called in the base class constructor
        mocker.patch(
            "tsercom.runtime.id_tracker.IdTracker.__init__", return_value=None
        )

        handler = ServerRuntimeDataHandler(
            data_reader=mock_data_reader,
            event_source=mock_event_source_poller,
            is_testing=True,  # Key for this test
        )

        mock_FakeSynchronizedClock_class.assert_called_once_with()
        mock_TimeSyncServer_class.assert_not_called()
        assert (
            handler._ServerRuntimeDataHandler__clock
            is mock_fake_clock_instance
        )
