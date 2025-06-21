from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import mock
import queue  # For queue.Full exception

import pytest

from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.split_process.remote_runtime_factory import RemoteRuntimeFactory
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.runtime.runtime_config import ServiceType, RuntimeConfig
from tsercom.threading.thread_watcher import ThreadWatcher

# Note: DefaultMultiprocessQueueFactory and TorchMultiprocessQueueFactory are not
# directly mocked in most tests here anymore; instead, the queue_factory property is mocked.
# However, they are needed for spec in mocks and for the "real" test.
from multiprocessing.context import BaseContext as MPContextType
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,  # For spec
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import MultiprocessQueueSink


@pytest.fixture
def mock_mp_context_provider_fixture() -> mock.Mock:
    """Mocks the MultiprocessingContextProvider class lookup and its instantiation."""
    provider_instance_mock = mock.MagicMock()
    # This mock is returned when MultiprocessingContextProvider[Any] is called (instantiated)
    specialized_provider_callable_mock = mock.MagicMock(
        return_value=provider_instance_mock
    )
    # This mock replaces the MultiprocessingContextProvider class in the SUT's module
    class_mock = mock.MagicMock()
    # Configure MultiprocessingContextProvider[Any] to return the callable
    class_mock.__getitem__.return_value = specialized_provider_callable_mock

    with mock.patch(
        "tsercom.api.split_process.split_runtime_factory_factory.MultiprocessingContextProvider",
        class_mock,  # class_mock is now the stand-in for MultiprocessingContextProvider
    ):
        # When SplitRuntimeFactoryFactory.__init__ calls:
        # MultiprocessingContextProvider[Any](context_method="spawn")
        # 1. MultiprocessingContextProvider -> class_mock (our patch)
        # 2. class_mock[Any] -> specialized_provider_callable_mock
        # 3. specialized_provider_callable_mock(context_method="spawn") -> provider_instance_mock
        # So, SUT's self.__mp_context_provider becomes provider_instance_mock
        yield provider_instance_mock


@pytest.fixture
def split_runtime_factory_factory_instance(
    mock_mp_context_provider_fixture: mock.Mock,  # SUT will use this mocked provider instance
) -> SplitRuntimeFactoryFactory:
    """Instance of SUT with mocked MultiprocessingContextProvider."""
    return SplitRuntimeFactoryFactory(
        thread_pool=mock.MagicMock(spec=ThreadPoolExecutor),
        thread_watcher=mock.MagicMock(spec=ThreadWatcher),
    )


@pytest.fixture
def real_split_runtime_factory_factory() -> SplitRuntimeFactoryFactory:
    """Instance of SUT that uses real underlying context and queue factories."""
    return SplitRuntimeFactoryFactory(
        thread_pool=mock.MagicMock(spec=ThreadPoolExecutor),
        thread_watcher=mock.MagicMock(spec=ThreadWatcher),
    )


def test_create_pair_interaction_with_provider_and_factory(
    split_runtime_factory_factory_instance: SplitRuntimeFactoryFactory,
    mock_mp_context_provider_fixture: mock.Mock,  # This is the SUT's self.__mp_context_provider
) -> None:
    """
    Tests that _create_pair correctly uses the MultiprocessingContextProvider:
    - Fetches the context.
    - Fetches the queue_factory.
    - Calls create_queues on the queue_factory with correct IPC parameters.
    """
    mock_context_instance = mock.MagicMock(spec=MPContextType)
    context_prop_mock = mock.PropertyMock(return_value=mock_context_instance)
    type(mock_mp_context_provider_fixture).context = context_prop_mock

    mock_queue_factory_instance = mock.MagicMock(spec=MultiprocessQueueFactory)
    queue_factory_prop_mock = mock.PropertyMock(
        return_value=mock_queue_factory_instance
    )
    type(mock_mp_context_provider_fixture).queue_factory = queue_factory_prop_mock

    # Side effect for the three calls to create_queues
    mock_queue_factory_instance.create_queues.side_effect = [
        (mock.MagicMock(spec=MultiprocessQueueSink), mock.MagicMock()),  # Event Qs
        (mock.MagicMock(spec=MultiprocessQueueSink), mock.MagicMock()),  # Data Qs
        (mock.MagicMock(spec=MultiprocessQueueSink), mock.MagicMock()),  # Command Qs
    ]

    ipc_q_size = 20
    ipc_blocking = False
    initializer = mock.MagicMock(spec=RuntimeInitializer)
    initializer.max_ipc_queue_size = ipc_q_size
    initializer.is_ipc_blocking = ipc_blocking
    initializer.timeout_seconds = None
    initializer.data_aggregator_client = None
    initializer.service_type_enum = ServiceType.SERVER
    initializer.config = mock.MagicMock(spec=RuntimeConfig)

    runtime_handle, runtime_factory = (
        split_runtime_factory_factory_instance._create_pair(initializer)
    )

    # context_prop_mock.assert_called() # .context is not directly called by _create_pair
    queue_factory_prop_mock.assert_called()

    assert mock_queue_factory_instance.create_queues.call_count == 2
    calls = mock_queue_factory_instance.create_queues.call_args_list

    # Event queue call with initializer's params
    assert calls[0] == mock.call(
        max_ipc_queue_size=ipc_q_size, is_ipc_blocking=ipc_blocking
    )
    # Data queue call with initializer's params
    assert calls[1] == mock.call(
        max_ipc_queue_size=ipc_q_size, is_ipc_blocking=ipc_blocking
    )
    # Command queue is no longer created by this mocked factory.

    assert isinstance(runtime_handle, ShimRuntimeHandle)
    assert isinstance(runtime_factory, RemoteRuntimeFactory)


def test_factory_with_non_blocking_queue_is_lossy(
    real_split_runtime_factory_factory: SplitRuntimeFactoryFactory,
    mocker: Any,
) -> None:
    """
    Tests non-blocking queue behavior using a real factory setup.
    Ensures queue.Full is raised on the underlying multiprocessing.Queue.
    """
    # Patch _TORCH_AVAILABLE where MultiprocessingContextProvider checks it
    mocker.patch(
        "tsercom.threading.multiprocess.multiprocessing_context_provider._TORCH_AVAILABLE",
        False,
    )

    expected_max_size = 1
    initializer = mock.MagicMock(spec=RuntimeInitializer)
    initializer.max_ipc_queue_size = expected_max_size
    initializer.is_ipc_blocking = False
    initializer.timeout_seconds = None
    initializer.data_aggregator_client = None
    initializer.service_type_enum = ServiceType.SERVER
    initializer.config = mock.MagicMock(spec=RuntimeConfig)

    runtime_handle, _ = real_split_runtime_factory_factory._create_pair(initializer)

    event_sink_wrapper: MultiprocessQueueSink = runtime_handle._ShimRuntimeHandle__event_queue  # type: ignore
    underlying_mp_queue = event_sink_wrapper._MultiprocessQueueSink__queue  # type: ignore

    underlying_mp_queue.put_nowait("item1")

    with pytest.raises(queue.Full):
        underlying_mp_queue.put_nowait("item2")

    # Cleanup (basic)
    if hasattr(runtime_handle, "stop"):  # Attempt to stop the handle if possible
        try:
            runtime_handle.stop()
        except Exception:  # Broad catch as stop might depend on other running parts
            pass  # Test focus is on queue behavior

    # Underlying queues are managed by processes usually, direct close might be tricky
    # or handled by process termination. For this test, focus on queue behavior.Tool output for `overwrite_file_with_block`:
