import multiprocessing
import pytest
from unittest.mock import (
    MagicMock as UMMagicMock,
    patch,
    PropertyMock,
    AsyncMock,
)
from concurrent.futures import Future
import asyncio
from multiprocessing import Process
import functools
from typing import (
    Any,
    Generator,
    List,
    Optional
)

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
from tsercom.runtime.runtime_factory import RuntimeFactory

import tsercom.threading.aio.global_event_loop as gev_loop


async def dummy_coroutine_for_test() -> None:
    """A simple coroutine that does nothing, for mocking awaitables."""
    pass


@pytest.fixture
def mock_thread_watcher(mocker: Any) -> ThreadWatcher:
    mock = mocker.MagicMock(spec=ThreadWatcher)
    mock.create_tracked_thread_pool_executor.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_local_rff(mocker: Any) -> LocalRuntimeFactoryFactory:
    mock = mocker.MagicMock(spec=LocalRuntimeFactoryFactory)
    # Setup default factory instance for start_in_process
    mock_factory_instance = UMMagicMock(spec=RuntimeFactory)
    mock_factory_instance._mp_context = None
    mock_factory_instance.auth_config = None
    mock.create_factory.return_value = mock_factory_instance
    return mock


@pytest.fixture
def mock_split_rff(mocker: Any) -> SplitRuntimeFactoryFactory:
    mock = mocker.MagicMock(spec=SplitRuntimeFactoryFactory)
    mock_factory_instance = UMMagicMock(spec=RuntimeFactory)
    mock_factory_instance._mp_context = None
    mock_factory_instance.auth_config = None
    mock.create_factory.return_value = mock_factory_instance
    return mock


@pytest.fixture
def mock_process_creator(mocker: Any) -> ProcessCreator:
    mock = mocker.MagicMock(spec=ProcessCreator)
    mock.create_process.return_value = mocker.MagicMock(spec=Process)
    return mock


@pytest.fixture
def mock_split_ewsf(mocker: Any) -> SplitErrorWatcherSourceFactory:
    mock = mocker.MagicMock(spec=SplitErrorWatcherSourceFactory)
    mock.create.return_value = mocker.MagicMock(spec=SplitProcessErrorWatcherSource)
    return mock


@pytest.fixture
def manager_with_mocks(
    mock_thread_watcher: ThreadWatcher,
    mock_local_rff: LocalRuntimeFactoryFactory,
    mock_split_rff: SplitRuntimeFactoryFactory,
    mock_process_creator: ProcessCreator,
    mock_split_ewsf: SplitErrorWatcherSourceFactory,
) -> RuntimeManager[Any, Any]:
    return RuntimeManager[Any, Any](
        thread_watcher=mock_thread_watcher,
        local_runtime_factory_factory=mock_local_rff,
        split_runtime_factory_factory=mock_split_rff,
        process_creator=mock_process_creator,
        split_error_watcher_source_factory=mock_split_ewsf,
    )


@pytest.fixture
def mock_runtime_initializer(mocker: Any) -> RuntimeInitializer[Any, Any]:
    mock_init = mocker.MagicMock(spec=RuntimeInitializer)
    mock_init.auth_config = None
    return mock_init


class TestRuntimeManager:
    def test_initialization_with_no_arguments(self, mocker: Any) -> None:
        mock_tw = mocker.patch("tsercom.api.runtime_manager.ThreadWatcher", autospec=True)
        mock_lff_init = mocker.patch("tsercom.api.local_process.local_runtime_factory_factory.LocalRuntimeFactoryFactory.__init__", return_value=None, autospec=True)
        mock_sff_init = mocker.patch("tsercom.api.split_process.split_runtime_factory_factory.SplitRuntimeFactoryFactory.__init__", return_value=None, autospec=True)
        mock_pc_constructor = mocker.patch("tsercom.api.runtime_manager.ProcessCreator", autospec=True)
        mock_sewsf_constructor = mocker.patch("tsercom.api.runtime_manager.SplitErrorWatcherSourceFactory", autospec=True)
        mock_thread_watcher_instance = mock_tw.return_value
        mock_thread_pool = UMMagicMock()
        mock_thread_watcher_instance.create_tracked_thread_pool_executor.return_value = mock_thread_pool
        manager: RuntimeManager[Any, Any] = RuntimeManager(is_testing=True)
        mock_tw.assert_called_once()
        mock_lff_init.assert_called_once_with(mocker.ANY, mock_thread_pool)
        mock_sff_init.assert_called_once_with(mocker.ANY, mock_thread_pool, mock_thread_watcher_instance)
        mock_pc_constructor.assert_called_once()
        mock_sewsf_constructor.assert_called_once()
        assert manager._RuntimeManager__is_testing is True
        assert manager._RuntimeManager__thread_watcher is mock_thread_watcher_instance
        assert isinstance(manager._RuntimeManager__local_runtime_factory_factory, LocalRuntimeFactoryFactory)
        assert isinstance(manager._RuntimeManager__split_runtime_factory_factory, SplitRuntimeFactoryFactory)
        assert manager._RuntimeManager__process_creator is mock_pc_constructor.return_value
        assert manager._RuntimeManager__split_error_watcher_source_factory is mock_sewsf_constructor.return_value

    def test_initialization_with_all_dependencies_mocked(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_thread_watcher: ThreadWatcher,
        mock_local_rff: LocalRuntimeFactoryFactory,
        mock_split_rff: SplitRuntimeFactoryFactory,
        mock_process_creator: ProcessCreator,
        mock_split_ewsf: SplitErrorWatcherSourceFactory,
    ) -> None:
        assert manager_with_mocks._RuntimeManager__thread_watcher is mock_thread_watcher
        assert manager_with_mocks._RuntimeManager__local_runtime_factory_factory is mock_local_rff
        assert manager_with_mocks._RuntimeManager__split_runtime_factory_factory is mock_split_rff
        assert manager_with_mocks._RuntimeManager__process_creator is mock_process_creator
        assert manager_with_mocks._RuntimeManager__split_error_watcher_source_factory is mock_split_ewsf
        assert not manager_with_mocks._RuntimeManager__is_testing

    def test_register_runtime_initializer_successful(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: RuntimeInitializer[Any, Any],
    ) -> None:
        assert len(manager_with_mocks._RuntimeManager__initializers) == 0
        future_handle = manager_with_mocks.register_runtime_initializer(mock_runtime_initializer)
        assert len(manager_with_mocks._RuntimeManager__initializers) == 1
        assert isinstance(future_handle, Future)
        pair = manager_with_mocks._RuntimeManager__initializers[0]
        assert pair.initializer is mock_runtime_initializer
        assert pair.handle_future is future_handle

    def test_register_runtime_initializer_after_start_raises_error(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: RuntimeInitializer[Any, Any],
        mocker: Any,
    ) -> None:
        mocker.patch.object(RuntimeManager, "has_started", new_callable=PropertyMock, return_value=True)
        with pytest.raises(RuntimeError, match="Cannot register runtime initializer after the manager has started."):
            manager_with_mocks.register_runtime_initializer(mock_runtime_initializer)

    @patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
    @patch("tsercom.runtime.runtime_main.initialize_runtimes")
    def test_start_in_process(
        self,
        mock_initialize_runtimes: UMMagicMock,
        mock_set_tsercom_event_loop: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_local_rff: LocalRuntimeFactoryFactory,
        mock_thread_watcher: ThreadWatcher,
        mock_runtime_initializer: RuntimeInitializer[Any, Any],
    ) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gev_loop.set_tsercom_event_loop(loop)
            manager_with_mocks.register_runtime_initializer(mock_runtime_initializer)
            mock_factory_instance = mock_local_rff.create_factory.return_value
            manager_with_mocks.start_in_process(loop)
            mock_set_tsercom_event_loop.assert_called_once_with(loop)
            assert manager_with_mocks._RuntimeManager__error_watcher is None
            assert manager_with_mocks._RuntimeManager__thread_watcher is mock_thread_watcher
            mock_local_rff.create_factory.assert_called_once()
            args, _ = mock_local_rff.create_factory.call_args
            assert isinstance(args[0], RuntimeFuturePopulator)
            assert args[1] is mock_runtime_initializer
            mock_initialize_runtimes.assert_called_once_with(mock_thread_watcher, [mock_factory_instance], is_testing=False)
            assert manager_with_mocks.has_started is True
            with pytest.raises(RuntimeError, match="RuntimeManager has already been started."):
                manager_with_mocks.start_in_process(loop)
        finally:
            gev_loop.clear_tsercom_event_loop()
            loop.close()

    @patch("tsercom.api.runtime_manager.get_running_loop_or_none")
    def test_start_in_process_async_no_loop_raises_error(
        self, mock_get_running_loop: UMMagicMock, manager_with_mocks: RuntimeManager[Any, Any]
    ) -> None:
        mock_get_running_loop.return_value = None
        with pytest.raises(RuntimeError, match="Could not determine the current running event loop"):
            asyncio.run(manager_with_mocks.start_in_process_async())

    @patch("tsercom.api.runtime_manager.get_running_loop_or_none")
    @patch.object(RuntimeManager, "start_in_process")
    def test_start_in_process_async_successful(
        self,
        mock_start_in_process_sync: UMMagicMock,
        mock_get_running_loop: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: RuntimeInitializer[Any, Any],
    ) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mock_get_running_loop.return_value = loop
        try:
            future_handle = manager_with_mocks.register_runtime_initializer(mock_runtime_initializer)
            mock_handle = UMMagicMock(spec=RuntimeHandle)
            future_handle.set_result(mock_handle)
            asyncio.run(manager_with_mocks.start_in_process_async())
            mock_start_in_process_sync.assert_called_once_with(loop)
            assert future_handle.result(timeout=0) == mock_handle
        finally:
            if gev_loop.is_global_event_loop_set() and gev_loop.get_global_event_loop() == loop:
                gev_loop.clear_tsercom_event_loop()
            loop.close()

    @patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
    @patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
    @patch("tsercom.runtime.runtime_main.remote_process_main")
    @patch("multiprocessing.context.SpawnContext.Process")
    def test_start_out_of_process(
        self,
        MockSpawnProcess: UMMagicMock,
        mock_remote_process_main: UMMagicMock,
        MockDQFactory: UMMagicMock,
        mock_create_loop: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any], # manager_with_mocks uses the fixture mock_split_rff
        # mock_split_rff: SplitRuntimeFactoryFactory, # Do not take from fixture, create locally
        mock_split_ewsf: SplitErrorWatcherSourceFactory,
        mock_thread_watcher: ThreadWatcher,
        mocker: Any,
    ) -> None:
        # Create a local mock for SplitRuntimeFactoryFactory for this test
        local_mock_split_rff = mocker.MagicMock(spec=SplitRuntimeFactoryFactory)
        local_mock_factory_instance = UMMagicMock(spec=RuntimeFactory)
        local_mock_factory_instance._mp_context = None
        local_mock_factory_instance.auth_config = None
        local_mock_split_rff.create_factory.return_value = local_mock_factory_instance

        # Replace the one in manager_with_mocks with our local one
        manager_with_mocks._RuntimeManager__split_runtime_factory_factory = local_mock_split_rff

        picklable_error_sink = multiprocessing.Queue()
        mock_error_source_queue = UMMagicMock()
        MockDQFactory.__getitem__.return_value = MockDQFactory
        MockDQFactory.return_value.create_queues.return_value = (picklable_error_sink, mock_error_source_queue)
        mock_error_watcher_instance = mock_split_ewsf.create.return_value

        local_mock_runtime_initializer = mocker.MagicMock(spec=RuntimeInitializer)
        local_mock_runtime_initializer.auth_config = None

        manager_with_mocks._RuntimeManager__initializers = []
        manager_with_mocks.register_runtime_initializer(local_mock_runtime_initializer)

        mock_created_process = MockSpawnProcess.return_value

        # local_mock_split_rff.create_factory is already fresh as it's local to this test
        # No need to reset specifically unless it was called before start_out_of_process

        manager_with_mocks.start_out_of_process(start_as_daemon=True)

        mock_create_loop.assert_called_once_with(mock_thread_watcher)
        MockDQFactory.__getitem__.assert_called_once_with(Exception)
        MockDQFactory.return_value.create_queues.assert_called_once_with()
        mock_split_ewsf.create.assert_called_once_with(mock_thread_watcher, mock_error_source_queue)
        mock_error_watcher_instance.start.assert_called_once()

        local_mock_split_rff.create_factory.assert_called_once()
        args_factory, _ = local_mock_split_rff.create_factory.call_args
        assert args_factory[1] is local_mock_runtime_initializer

        MockSpawnProcess.assert_called_once()
        call_args = MockSpawnProcess.call_args
        assert call_args[1]["daemon"] is True
        target_partial = call_args[1]["target"]
        assert isinstance(target_partial, functools.partial)
        assert target_partial.func is mock_remote_process_main
        assert target_partial.args[0] == [mock_factory_instance]
        assert target_partial.args[1] is picklable_error_sink
        assert target_partial.keywords["is_testing"] is False
        mock_created_process.start.assert_called_once()
        assert manager_with_mocks.has_started is True
        with pytest.raises(RuntimeError, match="RuntimeManager has already been started."):
            manager_with_mocks.start_out_of_process()

    @patch("multiprocessing.context.SpawnContext.Process")
    def test_start_out_of_process_is_testing_daemon(
        self, MockSpawnProcess: UMMagicMock, mocker: Any, manager_with_mocks: RuntimeManager[Any, Any]
    ) -> None:
        manager_with_mocks._RuntimeManager__is_testing = True
        mocker.patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
        mock_mp_factory_class_mock = mocker.patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
        mock_mp_factory_class_mock.__getitem__.return_value = mock_mp_factory_class_mock
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        picklable_error_sink = multiprocessing.Queue()
        mock_mp_factory_instance.create_queues.return_value = (picklable_error_sink, UMMagicMock())
        mock_split_rff = manager_with_mocks._RuntimeManager__split_runtime_factory_factory
        # create_factory is already configured by the mock_split_rff fixture
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        mock_created_process = MockSpawnProcess.return_value

        manager_with_mocks.start_out_of_process(start_as_daemon=False)

        MockSpawnProcess.assert_called_once()
        call_args = MockSpawnProcess.call_args
        assert call_args[1]["daemon"] is True
        target_partial = call_args[1]["target"]
        assert target_partial.args[1] is picklable_error_sink
        mock_created_process.start.assert_called_once()

    @patch("multiprocessing.context.SpawnContext.Process")
    def test_start_out_of_process_process_creation_fails(
        self, MockSpawnProcess: UMMagicMock, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        MockSpawnProcess.return_value = None
        mocker.patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
        mock_mp_factory_class_mock = mocker.patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
        mock_mp_factory_class_mock.__getitem__.return_value = mock_mp_factory_class_mock
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        picklable_error_sink = multiprocessing.Queue()
        mock_mp_factory_instance.create_queues.return_value = (picklable_error_sink, UMMagicMock())
        mock_split_rff = manager_with_mocks._RuntimeManager__split_runtime_factory_factory
        # create_factory is already configured by the mock_split_rff fixture
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")

        manager_with_mocks.start_out_of_process()
        assert manager_with_mocks._RuntimeManager__process is None

    def test_register_after_start_in_process(
        self, manager_with_mocks: RuntimeManager[Any, Any], mock_runtime_initializer: RuntimeInitializer[Any, Any], mocker: Any
    ) -> None:
        mock_loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mocker.patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
        mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")
        manager_with_mocks.start_in_process(mock_loop)
        with pytest.raises(RuntimeError, match="Cannot register runtime initializer after the manager has started."):
            manager_with_mocks.register_runtime_initializer(mock_runtime_initializer)

    @patch("multiprocessing.context.SpawnContext.Process")
    def test_register_after_start_out_of_process(
        self, MockSpawnProcess: UMMagicMock, manager_with_mocks: RuntimeManager[Any, Any], mock_runtime_initializer: RuntimeInitializer[Any, Any], mocker: Any
    ) -> None:
        mocker.patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
        mock_mp_factory_class_mock = mocker.patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
        mock_mp_factory_class_mock.__getitem__.return_value = mock_mp_factory_class_mock
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        picklable_error_sink = multiprocessing.Queue()
        mock_mp_factory_instance.create_queues.return_value = (picklable_error_sink, UMMagicMock())
        mock_split_rff = manager_with_mocks._RuntimeManager__split_runtime_factory_factory
        # create_factory is already configured by the mock_split_rff fixture
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        MockSpawnProcess.return_value.is_alive.return_value = True

        manager_with_mocks.start_out_of_process()
        with pytest.raises(RuntimeError, match="Cannot register runtime initializer after the manager has started."):
            manager_with_mocks.register_runtime_initializer(mock_runtime_initializer)

    def test_start_in_process_multiple_times(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mock_loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mocker.patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
        mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")
        manager_with_mocks.start_in_process(mock_loop)
        with pytest.raises(RuntimeError, match="RuntimeManager has already been started."):
            manager_with_mocks.start_in_process(mock_loop)

    @patch("multiprocessing.context.SpawnContext.Process")
    def test_start_out_of_process_multiple_times(
        self, MockSpawnProcess: UMMagicMock, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mocker.patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
        mock_mp_factory_class_mock = mocker.patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
        mock_mp_factory_class_mock.__getitem__.return_value = mock_mp_factory_class_mock
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        picklable_error_sink = multiprocessing.Queue()
        mock_mp_factory_instance.create_queues.return_value = (picklable_error_sink, UMMagicMock())
        mock_split_rff = manager_with_mocks._RuntimeManager__split_runtime_factory_factory
        # create_factory is already configured by the mock_split_rff fixture
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        MockSpawnProcess.return_value.is_alive.return_value = True

        manager_with_mocks.start_out_of_process()
        with pytest.raises(RuntimeError, match="RuntimeManager has already been started."):
            manager_with_mocks.start_out_of_process()

    @patch("multiprocessing.context.SpawnContext.Process")
    def test_shutdown_terminates_process(
        self, MockSpawnProcess: UMMagicMock, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mock_created_process = MockSpawnProcess.return_value
        mock_created_process.is_alive.return_value = True
        mocker.patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
        mock_mp_factory_class_mock = mocker.patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
        mock_mp_factory_class_mock.__getitem__.return_value = mock_mp_factory_class_mock
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        picklable_error_sink = multiprocessing.Queue()
        mock_mp_factory_instance.create_queues.return_value = (picklable_error_sink, UMMagicMock())
        mock_split_rff = manager_with_mocks._RuntimeManager__split_runtime_factory_factory
        # create_factory is already configured by the mock_split_rff fixture
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        mock_error_watcher = manager_with_mocks._RuntimeManager__split_error_watcher_source_factory.create.return_value

        manager_with_mocks.start_out_of_process()
        assert manager_with_mocks._RuntimeManager__process is mock_created_process
        manager_with_mocks.shutdown()
        mock_created_process.kill.assert_called_once()
        mock_created_process.join.assert_called_once()
        mock_error_watcher.stop.assert_called_once()

    @patch("multiprocessing.context.SpawnContext.Process")
    def test_shutdown_stops_error_watcher(
        self, MockSpawnProcess: UMMagicMock, manager_with_mocks: RuntimeManager[Any, Any], mock_split_ewsf: SplitErrorWatcherSourceFactory, mocker: Any
    ) -> None:
        mock_error_watcher = mock_split_ewsf.create.return_value
        mocker.patch("tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher")
        mock_mp_factory_class_mock = mocker.patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
        mock_mp_factory_class_mock.__getitem__.return_value = mock_mp_factory_class_mock
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        picklable_error_sink = multiprocessing.Queue()
        mock_mp_factory_instance.create_queues.return_value = (picklable_error_sink, UMMagicMock())
        mock_split_rff = manager_with_mocks._RuntimeManager__split_runtime_factory_factory
        # create_factory is already configured by the mock_split_rff fixture
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        mock_created_process = MockSpawnProcess.return_value
        mock_created_process.is_alive.return_value = False
        manager_with_mocks.start_out_of_process()
        manager_with_mocks.shutdown()
        mock_error_watcher.stop.assert_called_once()

    def test_run_until_exception_not_started(
        self, manager_with_mocks: RuntimeManager[Any, Any]
    ) -> None:
        with pytest.raises(RuntimeError, match="RuntimeManager has not been started."):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_error_watcher_none(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mocker.patch.object(RuntimeManager, "has_started", new_callable=PropertyMock, return_value=True)
        manager_with_mocks._RuntimeManager__thread_watcher = None
        with pytest.raises(RuntimeError, match="Internal ThreadWatcher is None"):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_calls_thread_watcher(
        self, manager_with_mocks: RuntimeManager[Any, Any], mock_thread_watcher: ThreadWatcher, mocker: Any
    ) -> None:
        mocker.patch.object(RuntimeManager, "has_started", new_callable=PropertyMock, return_value=True)
        manager_with_mocks._RuntimeManager__thread_watcher = mock_thread_watcher
        manager_with_mocks.run_until_exception()
        mock_thread_watcher.run_until_exception.assert_called_once()

    def test_check_for_exception_not_started(
        self, manager_with_mocks: RuntimeManager[Any, Any], mock_thread_watcher: ThreadWatcher
    ) -> None:
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_not_called()

    def test_check_for_exception_error_watcher_none(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mocker.patch.object(RuntimeManager, "has_started", new_callable=PropertyMock, return_value=True)
        manager_with_mocks._RuntimeManager__thread_watcher = None
        with pytest.raises(RuntimeError, match="Internal ThreadWatcher is None"):
            manager_with_mocks.check_for_exception()

    def test_check_for_exception_calls_thread_watcher(
        self, manager_with_mocks: RuntimeManager[Any, Any], mock_thread_watcher: ThreadWatcher, mocker: Any
    ) -> None:
        mocker.patch.object(RuntimeManager, "has_started", new_callable=PropertyMock, return_value=True)
        manager_with_mocks._RuntimeManager__thread_watcher = mock_thread_watcher
        manager_with_mocks.check_for_exception() # Corrected typo
        mock_thread_watcher.check_for_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_in_process_async_no_loop(self, mocker: Any) -> None:
        manager: RuntimeManager[Any, Any] = RuntimeManager()
        mocker.patch("tsercom.api.runtime_manager.get_running_loop_or_none", return_value=None)
        with pytest.raises(RuntimeError, match="Could not determine the current running event loop"):
            await manager.start_in_process_async()

    def test_run_until_exception_before_start(self) -> None:
        manager: RuntimeManager[Any, Any] = RuntimeManager()
        manager._RuntimeManager__thread_watcher = UMMagicMock(spec=ThreadWatcher)
        with pytest.raises(RuntimeError, match="RuntimeManager has not been started."):
            manager.run_until_exception()

    def test_check_for_exception_before_start(self, mocker: Any) -> None:
        mock_tw = mocker.patch("tsercom.api.runtime_manager.ThreadWatcher")
        manager: RuntimeManager[Any, Any] = RuntimeManager(thread_watcher=mock_tw)
        mocker.patch.object(RuntimeManager, "has_started", new_callable=PropertyMock, return_value=False)
        try:
            manager.check_for_exception()
        except RuntimeError:
            pytest.fail("check_for_exception raised RuntimeError unexpectedly before start")
        mock_tw.check_for_exception.assert_not_called()
