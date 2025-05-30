import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import Future
import asyncio
from multiprocessing import Process  # For spec in ProcessCreator mock

from tsercom.api.runtime_manager import RuntimeManager, RuntimeFuturePopulator
from tsercom.api.runtime_manager_helpers import (
    ProcessCreator,
    SplitErrorWatcherSourceFactory,
)
from tsercom.api.local_process.local_runtime_factory_factory import (
    LocalRuntimeFactoryFactory,
)
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.api.runtime_handle import RuntimeHandle


@pytest.fixture
def mock_thread_watcher(mocker):
    mock = mocker.MagicMock(spec=ThreadWatcher)
    # Mock the create_tracked_thread_pool_executor to return a mock executor
    mock.create_tracked_thread_pool_executor.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_local_rff(mocker):
    return mocker.MagicMock(spec=LocalRuntimeFactoryFactory)


@pytest.fixture
def mock_split_rff(mocker):
    return mocker.MagicMock(spec=SplitRuntimeFactoryFactory)


@pytest.fixture
def mock_process_creator(mocker):
    mock = mocker.MagicMock(spec=ProcessCreator)
    mock.create_process.return_value = mocker.MagicMock(spec=Process)
    return mock


@pytest.fixture
def mock_split_ewsf(mocker):
    mock = mocker.MagicMock(spec=SplitErrorWatcherSourceFactory)
    mock.create.return_value = mocker.MagicMock(
        spec=SplitProcessErrorWatcherSource
    )
    return mock


@pytest.fixture
def manager_with_mocks(
    mock_thread_watcher,
    mock_local_rff,
    mock_split_rff,
    mock_process_creator,
    mock_split_ewsf,
):
    return RuntimeManager(
        thread_watcher=mock_thread_watcher,
        local_runtime_factory_factory=mock_local_rff,
        split_runtime_factory_factory=mock_split_rff,
        process_creator=mock_process_creator,
        split_error_watcher_source_factory=mock_split_ewsf,
    )


@pytest.fixture
def mock_runtime_initializer(mocker):
    return mocker.MagicMock(spec=RuntimeInitializer)


class TestRuntimeManager:
    def test_initialization_with_no_arguments(self, mocker):
        """Test RuntimeManager() with no arguments: Ensure internal dependencies are created."""
        # We can't easily isinstance check the thread pools inside the default factories,
        # so we'll patch the factory constructors themselves to verify they are called.
        mock_tw = mocker.patch(
            "tsercom.api.runtime_manager.ThreadWatcher", autospec=True
        )
        mock_lff_constructor = mocker.patch(
            "tsercom.api.runtime_manager.LocalRuntimeFactoryFactory",
            autospec=True,
        )
        mock_sff_constructor = mocker.patch(
            "tsercom.api.runtime_manager.SplitRuntimeFactoryFactory",
            autospec=True,
        )
        mock_pc_constructor = mocker.patch(
            "tsercom.api.runtime_manager.ProcessCreator", autospec=True
        )
        mock_sewsf_constructor = mocker.patch(
            "tsercom.api.runtime_manager.SplitErrorWatcherSourceFactory",
            autospec=True,
        )

        # Mock the create_tracked_thread_pool_executor on the ThreadWatcher instance
        # that will be created by RuntimeManager
        mock_thread_watcher_instance = mock_tw.return_value
        mock_thread_pool = mocker.MagicMock()
        mock_thread_watcher_instance.create_tracked_thread_pool_executor.return_value = mock_thread_pool

        manager = RuntimeManager(is_testing=True)

        mock_tw.assert_called_once()
        # Check that factories are initialized with a thread pool from the created ThreadWatcher
        mock_lff_constructor.assert_called_once_with(mock_thread_pool)
        mock_sff_constructor.assert_called_once_with(
            mock_thread_pool, mock_thread_watcher_instance
        )
        mock_pc_constructor.assert_called_once()
        mock_sewsf_constructor.assert_called_once()

        assert manager._RuntimeManager__is_testing is True
        assert (
            manager._RuntimeManager__thread_watcher
            is mock_thread_watcher_instance
        )
        assert (
            manager._RuntimeManager__local_runtime_factory_factory
            is mock_lff_constructor.return_value
        )
        assert (
            manager._RuntimeManager__split_runtime_factory_factory
            is mock_sff_constructor.return_value
        )
        assert (
            manager._RuntimeManager__process_creator
            is mock_pc_constructor.return_value
        )
        assert (
            manager._RuntimeManager__split_error_watcher_source_factory
            is mock_sewsf_constructor.return_value
        )

    def test_initialization_with_all_dependencies_mocked(
        self,
        manager_with_mocks,  # This fixture provides a manager initialized with mocks
        mock_thread_watcher,
        mock_local_rff,
        mock_split_rff,
        mock_process_creator,
        mock_split_ewsf,
    ):
        """Test RuntimeManager() with all dependencies mocked."""
        assert (
            manager_with_mocks._RuntimeManager__thread_watcher
            is mock_thread_watcher
        )
        assert (
            manager_with_mocks._RuntimeManager__local_runtime_factory_factory
            is mock_local_rff
        )
        assert (
            manager_with_mocks._RuntimeManager__split_runtime_factory_factory
            is mock_split_rff
        )
        assert (
            manager_with_mocks._RuntimeManager__process_creator
            is mock_process_creator
        )
        assert (
            manager_with_mocks._RuntimeManager__split_error_watcher_source_factory
            is mock_split_ewsf
        )
        assert (
            not manager_with_mocks._RuntimeManager__is_testing
        )  # Default is_testing is False

    def test_register_runtime_initializer_successful(
        self, manager_with_mocks, mock_runtime_initializer
    ):
        """Successful registration: Check internal __initializers list size."""
        assert len(manager_with_mocks._RuntimeManager__initializers) == 0
        future_handle = manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )
        assert len(manager_with_mocks._RuntimeManager__initializers) == 1
        assert isinstance(future_handle, Future)
        pair = manager_with_mocks._RuntimeManager__initializers[0]
        assert pair.initializer is mock_runtime_initializer
        assert pair.handle_future is future_handle

    def test_register_runtime_initializer_after_start_raises_error(
        self, manager_with_mocks, mock_runtime_initializer, mocker
    ):
        """Registration after start: Use pytest.raises(RuntimeError)."""
        # Mock has_started to return True
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        with pytest.raises(
            RuntimeError,
            match="Cannot register runtime initializer after the manager has started.",
        ):
            manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )

    @patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
    @patch("tsercom.runtime.runtime_main.initialize_runtimes")
    def test_start_in_process(
        self,
        mock_initialize_runtimes,
        mock_set_tsercom_event_loop,
        manager_with_mocks,
        mock_local_rff,
        mock_thread_watcher,
        mock_runtime_initializer,
    ):
        """Test start_in_process verifies calls and state changes."""
        loop = asyncio.new_event_loop()
        manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )

        mock_factory_instance = MagicMock()
        mock_local_rff.create_factory.return_value = mock_factory_instance

        manager_with_mocks.start_in_process(loop)

        mock_set_tsercom_event_loop.assert_called_once_with(loop)
        # For start_in_process, __error_watcher is not set; errors are handled by __thread_watcher.
        assert manager_with_mocks._RuntimeManager__error_watcher is None
        assert (
            manager_with_mocks._RuntimeManager__thread_watcher
            is mock_thread_watcher
        )

        # Check that create_factory was called on the local_rff
        # The client passed should be a RuntimeFuturePopulator
        assert mock_local_rff.create_factory.call_count == 1
        args, kwargs = mock_local_rff.create_factory.call_args
        assert isinstance(args[0], RuntimeFuturePopulator)  # client
        assert args[1] is mock_runtime_initializer  # initializer

        mock_initialize_runtimes.assert_called_once_with(
            mock_thread_watcher, [mock_factory_instance], is_testing=False
        )
        assert manager_with_mocks.has_started is True

        with pytest.raises(
            RuntimeError, match="RuntimeManager has already been started."
        ):
            manager_with_mocks.start_in_process(loop)

    @patch("tsercom.api.runtime_manager.get_running_loop_or_none")
    def test_start_in_process_async_no_loop_raises_error(
        self, mock_get_running_loop, manager_with_mocks
    ):
        mock_get_running_loop.return_value = None
        with pytest.raises(
            RuntimeError,
            match="Could not determine the current running event loop",
        ):
            asyncio.run(manager_with_mocks.start_in_process_async())

    @patch("tsercom.api.runtime_manager.get_running_loop_or_none")
    @patch.object(RuntimeManager, "start_in_process")
    def test_start_in_process_async_successful(
        self,
        mock_start_in_process_sync,
        mock_get_running_loop,
        manager_with_mocks,
        mock_runtime_initializer,
    ):
        loop = asyncio.new_event_loop()
        mock_get_running_loop.return_value = loop

        # Ensure future is completed for result retrieval
        future_handle = manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )
        mock_handle = MagicMock(spec=RuntimeHandle)
        future_handle.set_result(mock_handle)

        returned_value = asyncio.run(
            manager_with_mocks.start_in_process_async()
        )

        mock_start_in_process_sync.assert_called_once_with(loop)
        assert returned_value is None
        # Verify that the future was indeed populated, which is the primary
        # way to get the handle now.
        assert future_handle.result(timeout=0) == mock_handle

    @patch(
        "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
    )
    @patch("tsercom.api.runtime_manager.create_multiprocess_queues")
    @patch(
        "tsercom.runtime.runtime_main.remote_process_main"
    )  # Mock the target of the process
    def test_start_out_of_process(
        self,
        mock_remote_process_main,
        mock_create_mp_queues,
        mock_create_tsercom_loop,
        manager_with_mocks,
        mock_split_rff,
        mock_split_ewsf,  # Fixture for the factory
        mock_thread_watcher,
        mock_process_creator,
        mock_runtime_initializer,
    ):
        """Test start_out_of_process verifies calls and state changes."""
        mock_error_sink, mock_error_source_queue = MagicMock(), MagicMock()
        mock_create_mp_queues.return_value = (
            mock_error_sink,
            mock_error_source_queue,
        )

        # This is the mock for SplitProcessErrorWatcherSource instance
        mock_error_watcher_source_instance = (
            mock_split_ewsf.create.return_value
        )

        manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )
        mock_factory_instance = MagicMock()
        mock_split_rff.create_factory.return_value = mock_factory_instance

        # Mock the process object returned by the creator
        mock_process_instance = (
            mock_process_creator.create_process.return_value
        )

        manager_with_mocks.start_out_of_process(start_as_daemon=True)

        mock_create_tsercom_loop.assert_called_once_with(mock_thread_watcher)
        mock_create_mp_queues.assert_called_once()

        # Verify the factory for SplitProcessErrorWatcherSource was used
        mock_split_ewsf.create.assert_called_once_with(
            mock_thread_watcher, mock_error_source_queue
        )
        mock_error_watcher_source_instance.start.assert_called_once()
        assert (
            manager_with_mocks._RuntimeManager__error_watcher
            is mock_error_watcher_source_instance
        )

        # Check that create_factory was called on the split_rff
        assert mock_split_rff.create_factory.call_count == 1
        args, kwargs = mock_split_rff.create_factory.call_args
        assert isinstance(args[0], RuntimeFuturePopulator)  # client
        assert args[1] is mock_runtime_initializer  # initializer

        # Check process creation
        mock_process_creator.create_process.assert_called_once()
        call_args = mock_process_creator.create_process.call_args
        assert call_args[1]["daemon"] is True  # start_as_daemon=True
        # Check that the target for the process is a partial of remote_process_main
        # and that it contains the correct arguments
        target_partial = call_args[1]["target"]
        assert target_partial.func is mock_remote_process_main
        assert target_partial.args[0] == [mock_factory_instance]  # factories
        assert target_partial.args[1] is mock_error_sink  # error_sink
        assert (
            target_partial.keywords["is_testing"] is False
        )  # manager default

        mock_process_instance.start.assert_called_once()
        assert manager_with_mocks.has_started is True

        with pytest.raises(
            RuntimeError, match="RuntimeManager has already been started."
        ):
            manager_with_mocks.start_out_of_process()

    def test_start_out_of_process_is_testing_daemon(
        self, mocker, manager_with_mocks
    ):
        """Test is_testing=True makes process daemonic in start_out_of_process."""
        manager_with_mocks._RuntimeManager__is_testing = (
            True  # Set is_testing to True
        )

        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mocker.patch(
            "tsercom.api.runtime_manager.create_multiprocess_queues",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        mock_process_creator = (
            manager_with_mocks._RuntimeManager__process_creator
        )  # Get the mock from manager

        manager_with_mocks.start_out_of_process(
            start_as_daemon=False
        )  # Explicitly False

        mock_process_creator.create_process.assert_called_once()
        call_args = mock_process_creator.create_process.call_args
        assert (
            call_args[1]["daemon"] is True
        )  # Should be True due to is_testing=True

    def test_run_until_exception_not_started(self, manager_with_mocks):
        """Test RuntimeError if manager not started for run_until_exception."""
        with pytest.raises(
            RuntimeError, match="RuntimeManager has not been started."
        ):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_error_watcher_none(
        self, manager_with_mocks, mocker
    ):
        """Test RuntimeError if __error_watcher is None but manager started."""
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        # manager_with_mocks._RuntimeManager__error_watcher = None # No longer relevant for this check
        manager_with_mocks._RuntimeManager__thread_watcher = (
            None  # This should trigger the error
        )
        with pytest.raises(
            RuntimeError,
            match="Error watcher is not available. Ensure the RuntimeManager has been properly started.",
        ):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_calls_thread_watcher(  # Renamed
        self, manager_with_mocks, mock_thread_watcher, mocker
    ):
        """Verify calls to mock_thread_watcher.run_until_exception()."""  # Simplified
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        # __error_watcher state doesn't influence the call to __thread_watcher here
        manager_with_mocks._RuntimeManager__thread_watcher = (
            mock_thread_watcher
        )
        manager_with_mocks.run_until_exception()
        mock_thread_watcher.run_until_exception.assert_called_once()

    def test_check_for_exception_not_started(
        self, manager_with_mocks, mock_thread_watcher
    ):
        """Test check_for_exception does nothing if not started."""
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_not_called()

    def test_check_for_exception_error_watcher_none(
        self, manager_with_mocks, mocker
    ):
        """Test RuntimeError if __error_watcher is None but manager started for check_for_exception."""
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        # Only __thread_watcher needs to be None to cause the error now
        manager_with_mocks._RuntimeManager__thread_watcher = None
        with pytest.raises(
            RuntimeError,
            match="Error watcher is not available. Ensure the RuntimeManager has been properly started.",  # Updated match
        ):
            manager_with_mocks.check_for_exception()

    def test_check_for_exception_calls_thread_watcher(  # Renamed
        self, manager_with_mocks, mock_thread_watcher, mocker
    ):
        """Verify calls to mock_thread_watcher.check_for_exception()."""  # Docstring updated & simplified
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = (
            mock_thread_watcher
        )
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_called_once()

    def test_runtime_future_populator_indirectly(
        self, manager_with_mocks, mock_local_rff, mock_runtime_initializer
    ):
        """
        Ensure that when factory.create_factory is called, the client argument
        is a RuntimeFuturePopulator and that its future gets completed.
        """
        loop = asyncio.new_event_loop()  # For start_in_process
        future_handle = manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )

        mock_created_handle = MagicMock(spec=RuntimeHandle)

        # Mock create_factory to simulate the behavior of a real factory
        # by calling _on_handle_ready on the client it receives.
        def mock_create_factory_impl(client, initializer):
            assert isinstance(client, RuntimeFuturePopulator)
            assert initializer is mock_runtime_initializer
            # Simulate handle creation and notify client
            client._on_handle_ready(mock_created_handle)
            return MagicMock()  # Return a mock factory instance

        mock_local_rff.create_factory.side_effect = mock_create_factory_impl

        with (
            patch("tsercom.api.runtime_manager.set_tsercom_event_loop"),
            patch("tsercom.runtime.runtime_main.initialize_runtimes"),
        ):
            manager_with_mocks.start_in_process(loop)

        assert future_handle.done()
        assert future_handle.result(timeout=0) is mock_created_handle
        mock_local_rff.create_factory.assert_called_once()

    def test_start_out_of_process_process_creation_fails(
        self, manager_with_mocks, mock_process_creator, mocker
    ):
        """Test that if process_creator.create_process returns None, process.start() is not called."""
        mock_process_creator.create_process.return_value = (
            None  # Simulate creation failure
        )

        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mocker.patch(
            "tsercom.api.runtime_manager.create_multiprocess_queues",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")

        manager_with_mocks.start_out_of_process()

        # mock_process_instance.start() should not be called if mock_process_instance is None
        # This is implicitly tested as mock_process_creator.create_process.return_value is None,
        # so there's no .start attribute on None. If it were called, an AttributeError would occur.
        # We can also check that the mocked Process object (if one was accidentally configured on the
        # process_creator mock for return) didn't have its start() called.
        # However, the fixture for mock_process_creator already sets create_process.return_value
        # to a MagicMock(spec=Process). So we need to ensure *that specific mock's* start isn't called.

        # If create_process returned an actual mock Process, that mock's start method would be checked.
        # But since it returns None, we verify that the code handles it gracefully.
        # The main check is that no error occurs and the program proceeds.
        # We can assert that the internal __process attribute is None.
        assert manager_with_mocks._RuntimeManager__process is None

        # And ensure that the .start() method of the *mocked Process object that could have been returned*
        # was not called.
        # So we check the mock that *would* have been returned if create_process didn't return None
        # This is a bit tricky. Let's re-evaluate.
        # The intent is: if self.__process_creator.create_process returns None, then self.__process is None,
        # and self.__process.start() is not called.

        # The mock_process_creator fixture does this:
        #   mock = mocker.MagicMock(spec=ProcessCreator)
        #   mock.create_process.return_value = mocker.MagicMock(spec=Process)
        # So, to test the failure case, we need to override this for *this specific test*:

        failing_process_creator = mocker.MagicMock(spec=ProcessCreator)
        failing_process_creator.create_process.return_value = (
            None  # Simulate creation failure
        )

        # Re-initialize manager with this specific failing mock
        manager_with_failing_pc = RuntimeManager(
            thread_watcher=manager_with_mocks._RuntimeManager__thread_watcher,
            local_runtime_factory_factory=manager_with_mocks._RuntimeManager__local_runtime_factory_factory,
            split_runtime_factory_factory=manager_with_mocks._RuntimeManager__split_runtime_factory_factory,
            process_creator=failing_process_creator,  # Use the failing one
            split_error_watcher_source_factory=manager_with_mocks._RuntimeManager__split_error_watcher_source_factory,
        )

        manager_with_failing_pc.start_out_of_process()
        assert manager_with_failing_pc._RuntimeManager__process is None
        # No Process.start() should have been called on the None object.
        # The test passes if no AttributeError is raised.
