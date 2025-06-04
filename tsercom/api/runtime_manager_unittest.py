import pytest
from unittest.mock import (
    MagicMock,
    patch,
    PropertyMock,
    AsyncMock,
)  # Import AsyncMock
from concurrent.futures import Future
import asyncio
from multiprocessing import Process  # For spec in ProcessCreator mock
import functools  # For checking functools.partial

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

# Import for managing global event loop in tests
import tsercom.threading.aio.global_event_loop as gev_loop

# Import for auth_config


async def dummy_coroutine_for_test():
    """A simple coroutine that does nothing, for mocking awaitables."""
    pass


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
    mock_init = mocker.MagicMock(spec=RuntimeInitializer)
    # Default auth_config to None to prevent ChannelFactorySelector ValueError
    mock_init.auth_config = None
    return mock_init


class TestRuntimeManager:
    def test_initialization_with_no_arguments(self, mocker):
        """Test RuntimeManager() with no arguments: Ensure internal dependencies are created."""
        mock_tw = mocker.patch(
            "tsercom.api.runtime_manager.ThreadWatcher", autospec=True
        )
        mock_lff_class = mocker.patch(
            "tsercom.api.runtime_manager.LocalRuntimeFactoryFactory"
        )
        mock_lff_constructor_callable = mocker.MagicMock(
            name="LFF_ConstructorProxy"
        )
        mock_lff_class.__getitem__.return_value = mock_lff_constructor_callable
        mock_sff_class = mocker.patch(
            "tsercom.api.runtime_manager.SplitRuntimeFactoryFactory"
        )
        mock_sff_constructor_callable = mocker.MagicMock(
            name="SFF_ConstructorProxy"
        )
        mock_sff_class.__getitem__.return_value = mock_sff_constructor_callable
        mock_pc_constructor = mocker.patch(
            "tsercom.api.runtime_manager.ProcessCreator", autospec=True
        )
        mock_sewsf_constructor = mocker.patch(
            "tsercom.api.runtime_manager.SplitErrorWatcherSourceFactory",
            autospec=True,
        )
        mock_thread_watcher_instance = mock_tw.return_value
        mock_thread_pool = mocker.MagicMock()
        mock_thread_watcher_instance.create_tracked_thread_pool_executor.return_value = (
            mock_thread_pool
        )

        manager = RuntimeManager(is_testing=True)

        mock_tw.assert_called_once()
        mock_lff_constructor_callable.assert_called_once_with(mock_thread_pool)
        mock_sff_constructor_callable.assert_called_once_with(
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
            is mock_lff_constructor_callable.return_value
        )
        assert (
            manager._RuntimeManager__split_runtime_factory_factory
            is mock_sff_constructor_callable.return_value
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
        manager_with_mocks,
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
        assert not manager_with_mocks._RuntimeManager__is_testing

    def test_register_runtime_initializer_successful(
        self, manager_with_mocks, mock_runtime_initializer
    ):
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
    @patch(
        "tsercom.api.runtime_manager.initialize_runtimes"
    )  # Patched where it's imported in runtime_manager
    def test_start_in_process(
        self,
        mock_initialize_runtimes_in_manager_scope,
        mock_set_tsercom_event_loop_in_manager,
        manager_with_mocks,
        mock_local_rff,
        mock_thread_watcher,
        mock_runtime_initializer,
    ):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gev_loop.set_tsercom_event_loop(loop)

            manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )

            mock_factory_instance = MagicMock()
            mock_factory_instance.auth_config = None

            mock_runtime_on_factory = MagicMock()
            mock_runtime_on_factory.start_async = AsyncMock(
                return_value=dummy_coroutine_for_test()
            )
            mock_factory_instance.create.return_value = mock_runtime_on_factory

            mock_local_rff.create_factory.return_value = mock_factory_instance

            manager_with_mocks.start_in_process(loop)

            mock_set_tsercom_event_loop_in_manager.assert_called_once_with(loop)
            assert manager_with_mocks._RuntimeManager__error_watcher is None
            assert (
                manager_with_mocks._RuntimeManager__thread_watcher
                is mock_thread_watcher
            )
            assert mock_local_rff.create_factory.call_count == 1
            args, kwargs = mock_local_rff.create_factory.call_args
            assert isinstance(args[0], RuntimeFuturePopulator)
            assert args[1] is mock_runtime_initializer

            mock_initialize_runtimes_in_manager_scope.assert_called_once_with(
                mock_thread_watcher, [mock_factory_instance], is_testing=False
            )
            assert manager_with_mocks.has_started is True
            with pytest.raises(
                RuntimeError, match="RuntimeManager has already been started."
            ):
                manager_with_mocks.start_in_process(loop)
        finally:
            gev_loop.clear_tsercom_event_loop()
            loop.close()

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
        asyncio.set_event_loop(loop)
        mock_get_running_loop.return_value = loop
        try:
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
            assert future_handle.result(timeout=0) == mock_handle
        finally:
            if (
                gev_loop.is_global_event_loop_set()
                and gev_loop.get_global_event_loop() == loop
            ):
                gev_loop.clear_tsercom_event_loop()
            loop.close()

    @patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
    @patch("tsercom.api.runtime_manager.create_multiprocess_queues")
    @patch("tsercom.api.runtime_manager.remote_process_main")
    def test_start_out_of_process(
        self,
        mock_remote_process_main_in_manager_scope,
        mock_create_mp_queues,
        mock_create_tsercom_loop,
        manager_with_mocks,
        mock_split_rff,
        mock_split_ewsf,
        mock_thread_watcher,
        mock_process_creator,
        mock_runtime_initializer,
    ):
        mock_error_sink, mock_error_source_queue = MagicMock(), MagicMock()
        mock_create_mp_queues.return_value = (
            mock_error_sink,
            mock_error_source_queue,
        )
        mock_error_watcher_source_instance = mock_split_ewsf.create.return_value
        manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )

        mock_factory_instance = MagicMock()
        mock_factory_instance.auth_config = None
        mock_split_rff.create_factory.return_value = mock_factory_instance

        mock_process_instance = mock_process_creator.create_process.return_value

        manager_with_mocks.start_out_of_process(start_as_daemon=True)

        mock_create_tsercom_loop.assert_called_once_with(mock_thread_watcher)
        mock_create_mp_queues.assert_called_once()
        mock_split_ewsf.create.assert_called_once_with(
            mock_thread_watcher, mock_error_source_queue
        )
        mock_error_watcher_source_instance.start.assert_called_once()
        assert (
            manager_with_mocks._RuntimeManager__error_watcher
            is mock_error_watcher_source_instance
        )
        assert mock_split_rff.create_factory.call_count == 1
        args, kwargs = mock_split_rff.create_factory.call_args
        assert isinstance(args[0], RuntimeFuturePopulator)
        assert args[1] is mock_runtime_initializer
        mock_process_creator.create_process.assert_called_once()
        call_args = mock_process_creator.create_process.call_args
        assert call_args[1]["daemon"] is True

        target_partial = call_args[1]["target"]
        assert isinstance(target_partial, functools.partial)
        assert target_partial.func is mock_remote_process_main_in_manager_scope

        assert target_partial.args[0] == [mock_factory_instance]
        assert target_partial.args[1] is mock_error_sink
        assert target_partial.keywords["is_testing"] is False
        mock_process_instance.start.assert_called_once()
        assert manager_with_mocks.has_started is True
        with pytest.raises(
            RuntimeError, match="RuntimeManager has already been started."
        ):
            manager_with_mocks.start_out_of_process()

    def test_start_out_of_process_is_testing_daemon(
        self, mocker, manager_with_mocks
    ):
        manager_with_mocks._RuntimeManager__is_testing = True
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mocker.patch(
            "tsercom.api.runtime_manager.create_multiprocess_queues",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch("tsercom.api.runtime_manager.remote_process_main")
        mock_process_creator = (
            manager_with_mocks._RuntimeManager__process_creator
        )
        manager_with_mocks.start_out_of_process(start_as_daemon=False)
        mock_process_creator.create_process.assert_called_once()
        call_args = mock_process_creator.create_process.call_args
        assert call_args[1]["daemon"] is True

    def test_run_until_exception_not_started(self, manager_with_mocks):
        with pytest.raises(
            RuntimeError, match="RuntimeManager has not been started."
        ):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_error_watcher_none(
        self, manager_with_mocks, mocker
    ):
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = None
        with pytest.raises(
            RuntimeError,
            match="Error watcher is not available. Ensure the RuntimeManager has been properly started.",
        ):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_calls_thread_watcher(
        self, manager_with_mocks, mock_thread_watcher, mocker
    ):
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = mock_thread_watcher
        manager_with_mocks.run_until_exception()
        mock_thread_watcher.run_until_exception.assert_called_once()

    def test_check_for_exception_not_started(
        self, manager_with_mocks, mock_thread_watcher
    ):
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_not_called()

    def test_check_for_exception_error_watcher_none(
        self, manager_with_mocks, mocker
    ):
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = None
        with pytest.raises(
            RuntimeError,
            match="Error watcher is not available. Ensure the RuntimeManager has been properly started.",
        ):
            manager_with_mocks.check_for_exception()

    def test_check_for_exception_calls_thread_watcher(
        self, manager_with_mocks, mock_thread_watcher, mocker
    ):
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = mock_thread_watcher
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_called_once()

    @patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
    @patch(
        "tsercom.api.runtime_manager.initialize_runtimes"
    )  # Patched where it's imported in runtime_manager
    def test_runtime_future_populator_indirectly(
        self,
        mock_initialize_runtimes_in_manager_scope,
        mock_set_tsercom_event_loop_in_manager,
        manager_with_mocks,
        mock_local_rff,
        mock_runtime_initializer,
    ):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gev_loop.set_tsercom_event_loop(loop)

            future_handle = manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )
            mock_created_handle = MagicMock(spec=RuntimeHandle)

            def mock_create_factory_impl(client, initializer):
                assert isinstance(client, RuntimeFuturePopulator)
                assert initializer is mock_runtime_initializer
                client._on_handle_ready(mock_created_handle)

                factory_mock = MagicMock()
                factory_mock.auth_config = None

                mock_runtime_on_factory = MagicMock()
                mock_runtime_on_factory.start_async = AsyncMock(
                    return_value=dummy_coroutine_for_test()
                )
                factory_mock.create.return_value = mock_runtime_on_factory
                return factory_mock

            mock_local_rff.create_factory.side_effect = mock_create_factory_impl

            manager_with_mocks.start_in_process(loop)

            assert future_handle.done()
            assert future_handle.result(timeout=0) is mock_created_handle
            mock_local_rff.create_factory.assert_called_once()
            mock_set_tsercom_event_loop_in_manager.assert_called_once_with(loop)
            mock_initialize_runtimes_in_manager_scope.assert_called_once()
        finally:
            gev_loop.clear_tsercom_event_loop()
            loop.close()

    def test_start_out_of_process_process_creation_fails(
        self, manager_with_mocks, mock_process_creator, mocker
    ):
        mock_process_creator.create_process.return_value = None
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mocker.patch(
            "tsercom.api.runtime_manager.create_multiprocess_queues",
            return_value=(MagicMock(), MagicMock()),
        )
        mocker.patch("tsercom.api.runtime_manager.remote_process_main")

        manager_with_mocks.start_out_of_process()
        assert manager_with_mocks._RuntimeManager__process is None

        failing_process_creator = mocker.MagicMock(spec=ProcessCreator)
        failing_process_creator.create_process.return_value = None
        manager_with_failing_pc = RuntimeManager(
            thread_watcher=manager_with_mocks._RuntimeManager__thread_watcher,
            local_runtime_factory_factory=manager_with_mocks._RuntimeManager__local_runtime_factory_factory,
            split_runtime_factory_factory=manager_with_mocks._RuntimeManager__split_runtime_factory_factory,
            process_creator=failing_process_creator,
            split_error_watcher_source_factory=manager_with_mocks._RuntimeManager__split_error_watcher_source_factory,
        )
        manager_with_failing_pc.start_out_of_process()
        assert manager_with_failing_pc._RuntimeManager__process is None
