import pytest
from unittest.mock import (
    MagicMock as UMMagicMock,  # unittest.mock.MagicMock
    patch,
    PropertyMock,
    AsyncMock,
)
from concurrent.futures import Future
import asyncio
from multiprocessing import Process  # For spec in ProcessCreator mock
import functools  # For checking functools.partial
from typing import (
    Any,
    Generator,
)  # Using Generator for pytest fixtures if they yield

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

import tsercom.threading.aio.global_event_loop as gev_loop


async def dummy_coroutine_for_test() -> None:
    """A simple coroutine that does nothing, for mocking awaitables."""
    pass


@pytest.fixture
def mock_thread_watcher(mocker: Any) -> Any:  # Changed to Any
    mock = mocker.MagicMock(spec=ThreadWatcher)
    # Mock the create_tracked_thread_pool_executor to return a mock executor
    mock.create_tracked_thread_pool_executor.return_value = mocker.MagicMock()
    return mock


@pytest.fixture
def mock_local_rff(mocker: Any) -> Any:  # Changed to Any
    return mocker.MagicMock(spec=LocalRuntimeFactoryFactory)


@pytest.fixture
def mock_split_rff(mocker: Any) -> Any:  # Changed to Any
    return mocker.MagicMock(spec=SplitRuntimeFactoryFactory)


@pytest.fixture
def mock_process_creator(mocker: Any) -> Any:  # Changed to Any
    mock = mocker.MagicMock(spec=ProcessCreator)
    mock.create_process.return_value = mocker.MagicMock(spec=Process)
    return mock


@pytest.fixture
def mock_split_ewsf(mocker: Any) -> Any:  # Changed to Any
    mock = mocker.MagicMock(spec=SplitErrorWatcherSourceFactory)
    mock.create.return_value = mocker.MagicMock(
        spec=SplitProcessErrorWatcherSource
    )
    return mock


@pytest.fixture
def manager_with_mocks(
    mock_thread_watcher: UMMagicMock,
    mock_local_rff: UMMagicMock,
    mock_split_rff: UMMagicMock,
    mock_process_creator: UMMagicMock,
    mock_split_ewsf: UMMagicMock,
) -> RuntimeManager[Any, Any]:
    return RuntimeManager[Any, Any](  # Explicitly generic for tests
        thread_watcher=mock_thread_watcher,
        local_runtime_factory_factory=mock_local_rff,
        split_runtime_factory_factory=mock_split_rff,
        process_creator=mock_process_creator,
        split_error_watcher_source_factory=mock_split_ewsf,
    )


@pytest.fixture
def mock_runtime_initializer(mocker: Any) -> Any:  # Changed to Any
    mock_init = mocker.MagicMock(spec=RuntimeInitializer)
    # Default auth_config to None to prevent ChannelFactorySelector ValueError
    mock_init.auth_config = None
    return mock_init


class TestRuntimeManager:
    def test_initialization_with_no_arguments(self, mocker: Any) -> None:
        """Test RuntimeManager() with no arguments: Ensure internal dependencies are created."""
        mock_tw = mocker.patch(
            "tsercom.api.runtime_manager.ThreadWatcher", autospec=True
        )
        mock_lff_init = mocker.patch(
            "tsercom.api.local_process.local_runtime_factory_factory.LocalRuntimeFactoryFactory.__init__",
            return_value=None,
            autospec=True,
        )
        mock_sff_init = mocker.patch(
            "tsercom.api.split_process.split_runtime_factory_factory.SplitRuntimeFactoryFactory.__init__",
            return_value=None,
            autospec=True,
        )
        mock_pc_constructor = mocker.patch(
            "tsercom.api.runtime_manager.ProcessCreator", autospec=True
        )
        mock_sewsf_constructor = mocker.patch(
            "tsercom.api.runtime_manager.SplitErrorWatcherSourceFactory",
            autospec=True,
        )
        mock_thread_watcher_instance = mock_tw.return_value
        mock_thread_pool = UMMagicMock()
        mock_thread_watcher_instance.create_tracked_thread_pool_executor.return_value = (
            mock_thread_pool
        )

        manager: RuntimeManager[Any, Any] = RuntimeManager(is_testing=True)

        mock_tw.assert_called_once()
        mock_lff_init.assert_called_once_with(mocker.ANY, mock_thread_pool)
        mock_sff_init.assert_called_once_with(
            mocker.ANY, mock_thread_pool, mock_thread_watcher_instance
        )
        mock_pc_constructor.assert_called_once()
        mock_sewsf_constructor.assert_called_once()
        assert manager._RuntimeManager__is_testing is True  # type: ignore[attr-defined]
        assert (
            manager._RuntimeManager__thread_watcher  # type: ignore[attr-defined]
            is mock_thread_watcher_instance
        )
        # Corrected assertions for the new mocking strategy (patching __init__)
        assert isinstance(
            manager._RuntimeManager__local_runtime_factory_factory,  # type: ignore[attr-defined]
            LocalRuntimeFactoryFactory,
        )
        assert isinstance(
            manager._RuntimeManager__split_runtime_factory_factory,  # type: ignore[attr-defined]
            SplitRuntimeFactoryFactory,
        )
        assert (
            manager._RuntimeManager__process_creator  # type: ignore[attr-defined]
            is mock_pc_constructor.return_value
        )
        assert (
            manager._RuntimeManager__split_error_watcher_source_factory  # type: ignore[attr-defined]
            is mock_sewsf_constructor.return_value
        )

    def test_initialization_with_all_dependencies_mocked(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_thread_watcher: UMMagicMock,
        mock_local_rff: UMMagicMock,
        mock_split_rff: UMMagicMock,
        mock_process_creator: UMMagicMock,
        mock_split_ewsf: UMMagicMock,
    ) -> None:
        """Test RuntimeManager() with all dependencies mocked."""
        assert (
            manager_with_mocks._RuntimeManager__thread_watcher  # type: ignore[attr-defined]
            is mock_thread_watcher
        )
        assert (
            manager_with_mocks._RuntimeManager__local_runtime_factory_factory  # type: ignore[attr-defined]
            is mock_local_rff
        )
        assert (
            manager_with_mocks._RuntimeManager__split_runtime_factory_factory  # type: ignore[attr-defined]
            is mock_split_rff
        )
        assert (
            manager_with_mocks._RuntimeManager__process_creator  # type: ignore[attr-defined]
            is mock_process_creator
        )
        assert (
            manager_with_mocks._RuntimeManager__split_error_watcher_source_factory  # type: ignore[attr-defined]
            is mock_split_ewsf
        )
        assert not manager_with_mocks._RuntimeManager__is_testing  # type: ignore[attr-defined]

    def test_register_runtime_initializer_successful(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: UMMagicMock,
    ) -> None:
        assert len(manager_with_mocks._RuntimeManager__initializers) == 0  # type: ignore[attr-defined]
        future_handle = manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )
        assert len(manager_with_mocks._RuntimeManager__initializers) == 1  # type: ignore[attr-defined]
        assert isinstance(future_handle, Future)
        pair = manager_with_mocks._RuntimeManager__initializers[0]  # type: ignore[attr-defined]
        assert pair.initializer is mock_runtime_initializer
        assert pair.handle_future is future_handle

    def test_register_runtime_initializer_after_start_raises_error(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: UMMagicMock,
        mocker: Any,
    ) -> None:
        mocker.patch.object(
            RuntimeManager,  # Patching the class itself
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
        "tsercom.runtime.runtime_main.initialize_runtimes"  # Corrected target
    )
    def test_start_in_process(
        self,
        mock_initialize_runtimes_in_manager_scope: UMMagicMock,
        mock_set_tsercom_event_loop_in_manager: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_local_rff: UMMagicMock,
        mock_thread_watcher: UMMagicMock,
        mock_runtime_initializer: UMMagicMock,
    ) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gev_loop.set_tsercom_event_loop(loop)

            manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )

            mock_factory_instance = UMMagicMock()
            mock_factory_instance.auth_config = None

            mock_runtime_on_factory = UMMagicMock()
            # Use side_effect with the async function itself
            mock_runtime_on_factory.start_async = AsyncMock(
                side_effect=dummy_coroutine_for_test
            )
            mock_factory_instance.create.return_value = mock_runtime_on_factory

            mock_local_rff.create_factory.return_value = mock_factory_instance

            manager_with_mocks.start_in_process(loop)

            mock_set_tsercom_event_loop_in_manager.assert_called_once_with(
                loop
            )
            assert manager_with_mocks._RuntimeManager__error_watcher is None  # type: ignore[attr-defined]
            assert (
                manager_with_mocks._RuntimeManager__thread_watcher  # type: ignore[attr-defined]
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
        self,
        mock_get_running_loop: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
    ) -> None:
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
        mock_start_in_process_sync: UMMagicMock,
        mock_get_running_loop: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: UMMagicMock,
    ) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mock_get_running_loop.return_value = loop
        try:
            future_handle = manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )
            mock_handle = UMMagicMock(spec=RuntimeHandle)
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

    @patch(
        "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
    )
    @patch("tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory")
    @patch(
        "tsercom.runtime.runtime_main.remote_process_main"
    )  # Corrected target
    def test_start_out_of_process(
        self,
        mock_remote_process_main_in_manager_scope: UMMagicMock,
        MockDefaultMultiprocessQueueFactory: UMMagicMock,  # This is the mock for the class
        mock_create_tsercom_loop: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_split_rff: UMMagicMock,
        mock_split_ewsf: UMMagicMock,
        mock_thread_watcher: UMMagicMock,
        mock_process_creator: UMMagicMock,
        mock_runtime_initializer: UMMagicMock,
    ) -> None:
        mock_error_sink, mock_error_source_queue = UMMagicMock(), UMMagicMock()
        # This was mock_factory_instance, but it's shadowing the one defined below.
        # Renaming to avoid confusion, and using the correct mock from the decorator.

        # Configure the __getitem__ to return the class mock itself, so DefaultMultiprocessQueueFactory[Exception] still refers to MockDefaultMultiprocessQueueFactory
        MockDefaultMultiprocessQueueFactory.__getitem__.return_value = (
            MockDefaultMultiprocessQueueFactory
        )

        # This is the INSTANCE mock, returned when MockDefaultMultiprocessQueueFactory() is called
        instance_mock = MockDefaultMultiprocessQueueFactory.return_value
        instance_mock.create_queues.return_value = (
            mock_error_sink,
            mock_error_source_queue,
        )
        mock_error_watcher_source_instance = (
            mock_split_ewsf.create.return_value
        )
        manager_with_mocks.register_runtime_initializer(
            mock_runtime_initializer
        )

        # This mock_factory_instance is for the mock_split_rff.create_factory()
        mock_factory_instance_for_split_rff = UMMagicMock()
        mock_factory_instance_for_split_rff.auth_config = None
        mock_split_rff.create_factory.return_value = (
            mock_factory_instance_for_split_rff
        )

        mock_process_instance = (
            mock_process_creator.create_process.return_value
        )

        manager_with_mocks.start_out_of_process(start_as_daemon=True)

        mock_create_tsercom_loop.assert_called_once_with(mock_thread_watcher)
        # The class itself is called with [Exception] and then (), so __getitem__ then the instance call
        MockDefaultMultiprocessQueueFactory.__getitem__.assert_called_once_with(
            Exception
        )
        MockDefaultMultiprocessQueueFactory.assert_called_once_with()  # Called to get the instance
        instance_mock.create_queues.assert_called_once_with()
        mock_split_ewsf.create.assert_called_once_with(
            mock_thread_watcher, mock_error_source_queue
        )
        mock_error_watcher_source_instance.start.assert_called_once()
        assert (
            manager_with_mocks._RuntimeManager__error_watcher  # type: ignore[attr-defined]
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

        assert target_partial.args[0] == [
            mock_factory_instance_for_split_rff
        ]  # Corrected here
        assert target_partial.args[1] is mock_error_sink
        assert target_partial.keywords["is_testing"] is False
        mock_process_instance.start.assert_called_once()
        assert manager_with_mocks.has_started is True
        with pytest.raises(
            RuntimeError, match="RuntimeManager has already been started."
        ):
            manager_with_mocks.start_out_of_process()

    def test_start_out_of_process_is_testing_daemon(
        self, mocker: Any, manager_with_mocks: RuntimeManager[Any, Any]
    ) -> None:
        manager_with_mocks._RuntimeManager__is_testing = True  # type: ignore[attr-defined]
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mock_mp_factory_class_mock = mocker.patch(  # Renamed for clarity
            "tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory"
        )
        # Configure for Generic[Exception] access
        mock_mp_factory_class_mock.__getitem__.return_value = (
            mock_mp_factory_class_mock
        )

        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        mock_mp_factory_instance.create_queues.return_value = (
            UMMagicMock(),  # sink
            UMMagicMock(),  # source
        )
        mocker.patch(
            "tsercom.runtime.runtime_main.remote_process_main"
        )  # Corrected target
        mock_process_creator = (
            manager_with_mocks._RuntimeManager__process_creator  # type: ignore[attr-defined]
        )
        manager_with_mocks.start_out_of_process(start_as_daemon=False)
        mock_process_creator.create_process.assert_called_once()
        call_args = mock_process_creator.create_process.call_args
        assert call_args[1]["daemon"] is True

    def test_run_until_exception_not_started(
        self, manager_with_mocks: RuntimeManager[Any, Any]
    ) -> None:
        with pytest.raises(
            RuntimeError, match="RuntimeManager has not been started."
        ):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_error_watcher_none(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = None  # type: ignore[attr-defined]
        with pytest.raises(
            RuntimeError,
            match="Internal ThreadWatcher is None when checking for exceptions after start.",
        ):
            manager_with_mocks.run_until_exception()

    def test_run_until_exception_calls_thread_watcher(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_thread_watcher: UMMagicMock,
        mocker: Any,
    ) -> None:
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = (  # type: ignore[attr-defined]
            mock_thread_watcher
        )
        manager_with_mocks.run_until_exception()
        mock_thread_watcher.run_until_exception.assert_called_once()

    def test_check_for_exception_not_started(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_thread_watcher: UMMagicMock,
    ) -> None:
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_not_called()

    def test_check_for_exception_error_watcher_none(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = None  # type: ignore[attr-defined]
        with pytest.raises(
            RuntimeError,
            match="Internal ThreadWatcher is None when checking for exceptions after start.",
        ):
            manager_with_mocks.check_for_exception()

    def test_check_for_exception_calls_thread_watcher(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_thread_watcher: UMMagicMock,
        mocker: Any,
    ) -> None:
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=True,
        )
        manager_with_mocks._RuntimeManager__thread_watcher = (  # type: ignore[attr-defined]
            mock_thread_watcher
        )
        manager_with_mocks.check_for_exception()
        mock_thread_watcher.check_for_exception.assert_called_once()

    @patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
    @patch(
        "tsercom.runtime.runtime_main.initialize_runtimes"  # Corrected target
    )
    def test_runtime_future_populator_indirectly(
        self,
        mock_initialize_runtimes_in_manager_scope: UMMagicMock,
        mock_set_tsercom_event_loop_in_manager: UMMagicMock,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_local_rff: UMMagicMock,
        mock_runtime_initializer: UMMagicMock,
    ) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gev_loop.set_tsercom_event_loop(loop)

            future_handle = manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )
            mock_created_handle = UMMagicMock(spec=RuntimeHandle)

            def mock_create_factory_impl(
                client: RuntimeFuturePopulator[Any, Any],
                initializer: RuntimeInitializer[Any, Any],
            ) -> UMMagicMock:
                assert isinstance(client, RuntimeFuturePopulator)
                assert initializer is mock_runtime_initializer
                client._on_handle_ready(mock_created_handle)

                factory_mock = UMMagicMock()
                factory_mock.auth_config = None

                mock_runtime_on_factory = UMMagicMock()
                # Use side_effect with the async function itself
                mock_runtime_on_factory.start_async = AsyncMock(
                    side_effect=dummy_coroutine_for_test
                )
                factory_mock.create.return_value = mock_runtime_on_factory
                return factory_mock

            mock_local_rff.create_factory.side_effect = (
                mock_create_factory_impl
            )

            manager_with_mocks.start_in_process(loop)

            assert future_handle.done()
            assert future_handle.result(timeout=0) is mock_created_handle
            mock_local_rff.create_factory.assert_called_once()
            mock_set_tsercom_event_loop_in_manager.assert_called_once_with(
                loop
            )
            mock_initialize_runtimes_in_manager_scope.assert_called_once()
        finally:
            gev_loop.clear_tsercom_event_loop()
            loop.close()

    def test_start_out_of_process_process_creation_fails(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_process_creator: UMMagicMock,
        mocker: Any,
    ) -> None:
        mock_process_creator.create_process.return_value = None
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mock_mp_factory_class_mock = mocker.patch(
            "tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory"
        )
        mock_mp_factory_class_mock.__getitem__.return_value = (
            mock_mp_factory_class_mock
        )
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        mock_mp_factory_instance.create_queues.return_value = (
            UMMagicMock(),  # sink
            UMMagicMock(),  # source
        )
        mocker.patch(
            "tsercom.runtime.runtime_main.remote_process_main"
        )  # Corrected target

        manager_with_mocks.start_out_of_process()
        assert manager_with_mocks._RuntimeManager__process is None  # type: ignore[attr-defined]

        failing_process_creator = mocker.MagicMock(spec=ProcessCreator)
        failing_process_creator.create_process.return_value = None
        manager_with_failing_pc: RuntimeManager[Any, Any] = RuntimeManager(
            thread_watcher=manager_with_mocks._RuntimeManager__thread_watcher,  # type: ignore[attr-defined]
            local_runtime_factory_factory=manager_with_mocks._RuntimeManager__local_runtime_factory_factory,  # type: ignore[attr-defined]
            split_runtime_factory_factory=manager_with_mocks._RuntimeManager__split_runtime_factory_factory,  # type: ignore[attr-defined]
            process_creator=failing_process_creator,
            split_error_watcher_source_factory=manager_with_mocks._RuntimeManager__split_error_watcher_source_factory,  # type: ignore[attr-defined]
        )
        manager_with_failing_pc.start_out_of_process()
        assert manager_with_failing_pc._RuntimeManager__process is None  # type: ignore[attr-defined]

    def test_register_after_start_in_process(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: UMMagicMock,
        mocker: Any,
    ) -> None:
        """Tests registering an initializer after start_in_process."""
        # Mock AbstractEventLoop
        mock_loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mocker.patch(
            "tsercom.api.runtime_manager.set_tsercom_event_loop"
        )  # Prevent actual loop setting
        mocker.patch(
            "tsercom.runtime.runtime_main.initialize_runtimes"
        )  # Prevent runtime init

        manager_with_mocks.start_in_process(mock_loop)
        with pytest.raises(
            RuntimeError,
            match="Cannot register runtime initializer after the manager has started.",
        ):
            manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )

    def test_register_after_start_out_of_process(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_runtime_initializer: UMMagicMock,
        mocker: Any,
    ) -> None:
        """Tests registering an initializer after start_out_of_process."""
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mock_mp_factory_class_mock = mocker.patch(
            "tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory"
        )
        mock_mp_factory_class_mock.__getitem__.return_value = (
            mock_mp_factory_class_mock
        )
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        mock_mp_factory_instance.create_queues.return_value = (
            UMMagicMock(),  # sink
            UMMagicMock(),  # source
        )
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        # Ensure process is created and started
        mock_process = (
            manager_with_mocks._RuntimeManager__process_creator.create_process.return_value  # type: ignore[attr-defined]
        )
        mock_process.is_alive.return_value = True

        manager_with_mocks.start_out_of_process()
        with pytest.raises(
            RuntimeError,
            match="Cannot register runtime initializer after the manager has started.",
        ):
            manager_with_mocks.register_runtime_initializer(
                mock_runtime_initializer
            )

    def test_start_in_process_multiple_times(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        """Tests calling start_in_process multiple times."""
        mock_loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mocker.patch("tsercom.api.runtime_manager.set_tsercom_event_loop")
        mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")

        manager_with_mocks.start_in_process(mock_loop)  # First call
        with pytest.raises(
            RuntimeError, match="RuntimeManager has already been started."
        ):
            manager_with_mocks.start_in_process(mock_loop)  # Second call

    def test_start_out_of_process_multiple_times(
        self, manager_with_mocks: RuntimeManager[Any, Any], mocker: Any
    ) -> None:
        """Tests calling start_out_of_process multiple times."""
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mock_mp_factory_class_mock = mocker.patch(
            "tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory"
        )
        mock_mp_factory_class_mock.__getitem__.return_value = (
            mock_mp_factory_class_mock
        )
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        mock_mp_factory_instance.create_queues.return_value = (
            UMMagicMock(),  # sink
            UMMagicMock(),  # source
        )
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        mock_process = (
            manager_with_mocks._RuntimeManager__process_creator.create_process.return_value  # type: ignore[attr-defined]
        )
        mock_process.is_alive.return_value = True

        manager_with_mocks.start_out_of_process()  # First call
        with pytest.raises(
            RuntimeError, match="RuntimeManager has already been started."
        ):
            manager_with_mocks.start_out_of_process()  # Second call

    def test_shutdown_terminates_process(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_process_creator: UMMagicMock,
        mocker: Any,
    ) -> None:
        """Tests that shutdown terminates the process in out-of-process mode."""
        mock_process = mock_process_creator.create_process.return_value
        mock_process.is_alive.return_value = True  # Simulate live process

        # Setup for start_out_of_process to run
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mock_mp_factory_class_mock = mocker.patch(
            "tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory"
        )
        mock_mp_factory_class_mock.__getitem__.return_value = (
            mock_mp_factory_class_mock
        )
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        mock_mp_factory_instance.create_queues.return_value = (
            UMMagicMock(),  # sink
            UMMagicMock(),  # source
        )
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        # Mock the error watcher to prevent issues during shutdown
        mock_error_watcher = UMMagicMock(spec=SplitProcessErrorWatcherSource)
        manager_with_mocks._RuntimeManager__split_error_watcher_source_factory.create.return_value = (  # type: ignore[attr-defined]
            mock_error_watcher
        )

        manager_with_mocks.start_out_of_process()
        manager_with_mocks.shutdown()

        mock_process.kill.assert_called_once()
        mock_process.join.assert_called_once()

    def test_shutdown_stops_error_watcher(
        self,
        manager_with_mocks: RuntimeManager[Any, Any],
        mock_split_ewsf: UMMagicMock,
        mocker: Any,
    ) -> None:
        """Tests that shutdown stops the error watcher in out-of-process mode."""
        mock_error_watcher = mock_split_ewsf.create.return_value
        # manager_with_mocks._RuntimeManager__process = None # Simulate in-process or no process started yet
        # manager_with_mocks._RuntimeManager__error_watcher = mock_error_watcher # Assign watcher

        # Setup for start_out_of_process to run without creating a real process that hangs
        mocker.patch(
            "tsercom.api.runtime_manager.create_tsercom_event_loop_from_watcher"
        )
        mock_mp_factory_class_mock = mocker.patch(
            "tsercom.api.runtime_manager.DefaultMultiprocessQueueFactory"
        )
        mock_mp_factory_class_mock.__getitem__.return_value = (
            mock_mp_factory_class_mock
        )
        mock_mp_factory_instance = mock_mp_factory_class_mock.return_value
        mock_mp_factory_instance.create_queues.return_value = (
            UMMagicMock(),  # sink
            UMMagicMock(),  # source
        )
        mocker.patch("tsercom.runtime.runtime_main.remote_process_main")
        mock_process = (
            manager_with_mocks._RuntimeManager__process_creator.create_process.return_value  # type: ignore[attr-defined]
        )
        mock_process.is_alive.return_value = (
            False  # Process not alive, so kill/join won't be problematic
        )

        manager_with_mocks.start_out_of_process()  # This will set up the error watcher via mocks
        manager_with_mocks.shutdown()

        mock_error_watcher.stop.assert_called_once()

    # New tests to be added here

    @pytest.mark.asyncio
    async def test_start_in_process_async_no_loop(self, mocker: Any) -> None:
        """Tests start_in_process_async when no event loop is found."""
        manager: RuntimeManager[Any, Any] = RuntimeManager()  # Fresh instance
        mocker.patch(
            "tsercom.api.runtime_manager.get_running_loop_or_none",
            return_value=None,
        )
        with pytest.raises(
            RuntimeError,
            match="Could not determine the current running event loop for start_in_process_async.",  # Corrected message
        ):
            await manager.start_in_process_async()

    def test_run_until_exception_before_start(self) -> None:
        """Tests run_until_exception before the manager has started."""
        manager: RuntimeManager[Any, Any] = RuntimeManager()  # Fresh instance
        # Ensure ThreadWatcher is created for this test if not using manager_with_mocks
        manager._RuntimeManager__thread_watcher = UMMagicMock(spec=ThreadWatcher)  # type: ignore[attr-defined]
        with pytest.raises(
            RuntimeError, match="RuntimeManager has not been started."
        ):
            manager.run_until_exception()

    def test_check_for_exception_before_start(self, mocker: Any) -> None:
        """Tests check_for_exception before the manager has started. Should not raise."""
        # We need a ThreadWatcher instance, but it shouldn't be called.
        mock_tw = mocker.patch("tsercom.api.runtime_manager.ThreadWatcher")
        manager: RuntimeManager[Any, Any] = RuntimeManager(
            thread_watcher=mock_tw
        )

        # Explicitly ensure has_started is False
        mocker.patch.object(
            RuntimeManager,
            "has_started",
            new_callable=PropertyMock,
            return_value=False,
        )

        try:
            manager.check_for_exception()
        except RuntimeError:
            pytest.fail(
                "check_for_exception raised RuntimeError unexpectedly before start"
            )

        # Verify that the ThreadWatcher's method was not called
        assert (
            manager._RuntimeManager__thread_watcher is mock_tw  # type: ignore[attr-defined]
        )  # manager holds the class mock itself
        mock_tw.check_for_exception.assert_not_called()  # Methods are called on the class mock
