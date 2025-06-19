import pytest
from unittest.mock import MagicMock

# Module to be tested & whose attributes will be patched
import tsercom.api.split_process.split_runtime_factory_factory as srff_module
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.runtime_factory_factory import (
    RuntimeFactoryFactory as BaseRuntimeFactoryFactory,
)

import torch
import multiprocessing as std_mp
import torch.multiprocessing as torch_mp
from typing import TypeVar, Generic, Any

from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


# --- Fake Classes for Dependencies & Patched Classes ---


class FakeThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.shutdown_called = False

    def shutdown(self, wait=True):
        self.shutdown_called = True


class FakeThreadWatcher:
    def __init__(self, name="FakeThreadWatcher"):
        self.name = name


class FakeRuntimeInitializer(RuntimeInitializer[str, str]):
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):
        super().__init__(
            service_type=service_type,
            data_aggregator_client=data_aggregator_client,
            timeout_seconds=timeout_seconds,
        )

    def create(self) -> Any:
        return MagicMock()


LocalDataTypeT = TypeVar("LocalDataTypeT")
LocalEventTypeT = TypeVar("LocalEventTypeT")


class GenericFakeRuntimeInitializer(
    RuntimeInitializer[LocalDataTypeT, LocalEventTypeT],
    Generic[LocalDataTypeT, LocalEventTypeT],
):
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):
        super().__init__(
            service_type=service_type,
            data_aggregator_client=data_aggregator_client,
            timeout_seconds=timeout_seconds,
        )

    def create(
        self, thread_watcher, data_handler, grpc_channel_factory
    ) -> Any:
        return MagicMock()


g_fake_remote_runtime_factory_instances = []
g_fake_remote_data_aggregator_instances = []
g_fake_shim_runtime_handle_instances = []


class FakeRemoteRuntimeFactory:
    __class_getitem__ = classmethod(lambda cls, item: cls)

    def __init__(
        self, initializer, event_source, data_reader_sink, command_source
    ):
        self.initializer = initializer
        self.event_source = event_source
        self.data_reader_sink = data_reader_sink
        self.command_source = command_source
        g_fake_remote_runtime_factory_instances.append(self)


class FakeRemoteDataAggregatorImpl:
    __class_getitem__ = classmethod(lambda cls, item: cls)

    def __init__(self, thread_pool, client, timeout=None):
        self.thread_pool = thread_pool
        self.client = client
        self.timeout = timeout
        g_fake_remote_data_aggregator_instances.append(self)


class FakeShimRuntimeHandle:
    __class_getitem__ = classmethod(lambda cls, item: cls)

    def __init__(
        self,
        thread_watcher,
        event_queue,
        data_queue,
        runtime_command_queue,
        data_aggregator,
        block=False,
    ):
        self.thread_watcher = thread_watcher
        self.event_queue = event_queue
        self.data_queue = data_queue
        self.runtime_command_queue = runtime_command_queue
        self.data_aggregator = data_aggregator
        self.block = block
        g_fake_shim_runtime_handle_instances.append(self)


class FakeRuntimeFactoryFactoryClient(BaseRuntimeFactoryFactory.Client):
    def __init__(self):
        self.handle_ready_called = False
        self.received_handle = None

    def _on_handle_ready(self, handle):
        self.handle_ready_called = True
        self.received_handle = handle


# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def clear_globals_and_mocks(mocker):
    global g_fake_remote_runtime_factory_instances, g_fake_remote_data_aggregator_instances, g_fake_shim_runtime_handle_instances
    g_fake_remote_runtime_factory_instances = []
    g_fake_remote_data_aggregator_instances = []
    g_fake_shim_runtime_handle_instances = []
    mocker.resetall()


@pytest.fixture
def fake_executor():
    return FakeThreadPoolExecutor()


@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_initializer():
    return FakeRuntimeInitializer()


@pytest.fixture
def fake_client():
    return FakeRuntimeFactoryFactoryClient()


std_mp_context = std_mp.get_context("spawn")
torch_mp_context = torch_mp.get_context("spawn")


@pytest.fixture
def mock_queue_factories(mocker):
    mock_default_init = mocker.patch.object(
        srff_module.DefaultMultiprocessQueueFactory,
        "__init__",
        return_value=None,
    )
    default_queues_results = []
    for _ in range(3):
        q = std_mp_context.Queue()
        default_queues_results.append(
            (MultiprocessQueueSink(q), MultiprocessQueueSource(q))
        )
    mock_default_create_queues = mocker.patch.object(
        srff_module.DefaultMultiprocessQueueFactory,
        "create_queues",
        side_effect=default_queues_results,
    )
    mock_torch_init = mocker.patch.object(
        srff_module.TorchMemcpyQueueFactory,
        "__init__",
        return_value=None,
    )
    torch_queues_results = []
    for _ in range(2):
        q = torch_mp_context.Queue()
        torch_queues_results.append(
            (MultiprocessQueueSink(q), MultiprocessQueueSource(q))
        )
    mock_torch_create_queues = mocker.patch.object(
        srff_module.TorchMemcpyQueueFactory,
        "create_queues",
        side_effect=torch_queues_results,
    )
    return {
        "default_init": mock_default_init,
        "default_create_queues": mock_default_create_queues,
        "torch_init": mock_torch_init,
        "torch_create_queues": mock_torch_create_queues,
        "_default_results_list": default_queues_results,
        "_torch_results_list": torch_queues_results,
    }


@pytest.fixture
def patch_other_dependencies(request, mocker):
    originals = {
        "RemoteRuntimeFactory": getattr(
            srff_module, "RemoteRuntimeFactory", None
        ),
        "RemoteDataAggregatorImpl": getattr(
            srff_module, "RemoteDataAggregatorImpl", None
        ),
        "ShimRuntimeHandle": getattr(srff_module, "ShimRuntimeHandle", None),
    }
    setattr(srff_module, "RemoteRuntimeFactory", FakeRemoteRuntimeFactory)
    setattr(
        srff_module, "RemoteDataAggregatorImpl", FakeRemoteDataAggregatorImpl
    )
    setattr(srff_module, "ShimRuntimeHandle", FakeShimRuntimeHandle)

    def cleanup():
        for attr, original_value in originals.items():
            if original_value:
                setattr(srff_module, attr, original_value)
            elif hasattr(srff_module, attr):
                delattr(srff_module, attr)

    request.addfinalizer(cleanup)


# --- Unit Tests ---
def test_create_factory_and_pair_logic_default_queues(
    fake_executor,
    fake_watcher,
    fake_initializer,
    fake_client,
    mock_queue_factories,
    patch_other_dependencies,
):
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    returned_factory = factory_factory.create_factory(
        fake_client, fake_initializer
    )

    mock_queue_factories["default_init"].assert_called()
    assert mock_queue_factories["default_create_queues"].call_count == 3
    mock_queue_factories["torch_init"].assert_not_called()
    assert mock_queue_factories["torch_create_queues"].call_count == 0

    assert len(g_fake_remote_runtime_factory_instances) == 1
    remote_factory = g_fake_remote_runtime_factory_instances[0]
    assert len(g_fake_shim_runtime_handle_instances) == 1
    shim_handle = g_fake_shim_runtime_handle_instances[0]

    event_sink_q = shim_handle.event_queue._MultiprocessQueueSink__queue
    event_source_q = (
        remote_factory.event_source._MultiprocessQueueSource__queue
    )
    assert isinstance(event_sink_q, type(std_mp_context.Queue()))
    assert (
        event_sink_q
        is mock_queue_factories["_default_results_list"][0][
            0
        ]._MultiprocessQueueSink__queue
    )
    assert (
        event_source_q
        is mock_queue_factories["_default_results_list"][0][
            1
        ]._MultiprocessQueueSource__queue
    )

    data_sink_q = remote_factory.data_reader_sink._MultiprocessQueueSink__queue
    data_source_q = shim_handle.data_queue._MultiprocessQueueSource__queue
    assert isinstance(data_sink_q, type(std_mp_context.Queue()))
    assert (
        data_sink_q
        is mock_queue_factories["_default_results_list"][1][
            0
        ]._MultiprocessQueueSink__queue
    )
    assert (
        data_source_q
        is mock_queue_factories["_default_results_list"][1][
            1
        ]._MultiprocessQueueSource__queue
    )

    cmd_sink_q = (
        shim_handle.runtime_command_queue._MultiprocessQueueSink__queue
    )
    cmd_source_q = (
        remote_factory.command_source._MultiprocessQueueSource__queue
    )
    assert isinstance(cmd_sink_q, type(std_mp_context.Queue()))
    assert (
        cmd_sink_q
        is mock_queue_factories["_default_results_list"][2][
            0
        ]._MultiprocessQueueSink__queue
    )
    assert (
        cmd_source_q
        is mock_queue_factories["_default_results_list"][2][
            1
        ]._MultiprocessQueueSource__queue
    )

    assert len(g_fake_remote_data_aggregator_instances) == 1
    aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert aggregator_instance.thread_pool is fake_executor
    assert (
        aggregator_instance.client == fake_initializer.data_aggregator_client
    )
    assert aggregator_instance.timeout == fake_initializer.timeout_seconds

    assert remote_factory.initializer is fake_initializer
    assert shim_handle.thread_watcher is fake_watcher
    assert shim_handle.data_aggregator is aggregator_instance
    assert returned_factory is remote_factory
    assert fake_client.handle_ready_called
    assert fake_client.received_handle is shim_handle


@pytest.mark.parametrize(
    "initializer_type, data_type, event_type, expected_torch_calls, expected_default_data_event_calls, expected_default_cmd_calls, expected_internal_q_type",
    [
        (
            GenericFakeRuntimeInitializer[torch.Tensor, str],
            torch.Tensor,
            str,
            1,
            0,
            1,
            type(torch_mp_context.Queue()),
        ),
        (
            GenericFakeRuntimeInitializer[str, torch.Tensor],
            str,
            torch.Tensor,
            1,
            0,
            1,
            type(torch_mp_context.Queue()),
        ),
        (
            GenericFakeRuntimeInitializer[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            1,
            0,
            1,
            type(torch_mp_context.Queue()),
        ),
        (
            GenericFakeRuntimeInitializer[str, int],
            str,
            int,
            0,
            1,
            1,
            type(std_mp_context.Queue()),
        ),
    ],
)
def test_dynamic_queue_selection(
    fake_executor,
    fake_watcher,
    mock_queue_factories,
    patch_other_dependencies,
    initializer_type,
    data_type,
    event_type,
    expected_torch_calls,
    expected_default_data_event_calls,
    expected_default_cmd_calls,
    expected_internal_q_type,
):
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )

    specific_initializer = initializer_type(data_aggregator_client=None)
    factory_factory._create_pair(specific_initializer)

    expected_default_init_calls = 0
    if expected_default_data_event_calls > 0:
        expected_default_init_calls += 1
    expected_default_init_calls += 1

    if expected_torch_calls > 0:
        mock_queue_factories["torch_init"].assert_called()
    else:
        mock_queue_factories["torch_init"].assert_not_called()

    if expected_default_data_event_calls > 0 or expected_default_cmd_calls > 0:
        mock_queue_factories["default_init"].assert_called()
    else:
        mock_queue_factories["default_init"].assert_not_called()

    assert mock_queue_factories["torch_create_queues"].call_count == (
        expected_torch_calls * 2
    )
    assert (
        mock_queue_factories["default_create_queues"].call_count
        == (expected_default_data_event_calls * 2) + expected_default_cmd_calls
    )

    assert len(g_fake_remote_runtime_factory_instances) == 1
    remote_factory = g_fake_remote_runtime_factory_instances[0]
    assert len(g_fake_shim_runtime_handle_instances) == 1
    shim_handle = g_fake_shim_runtime_handle_instances[0]

    event_sink_q = shim_handle.event_queue._MultiprocessQueueSink__queue
    assert isinstance(event_sink_q, expected_internal_q_type)

    data_source_q = shim_handle.data_queue._MultiprocessQueueSource__queue
    assert isinstance(data_source_q, expected_internal_q_type)

    cmd_sink_q = (
        shim_handle.runtime_command_queue._MultiprocessQueueSink__queue
    )
    assert isinstance(cmd_sink_q, type(std_mp_context.Queue()))


def test_init_method(fake_executor, fake_watcher):
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    assert (
        factory_factory._SplitRuntimeFactoryFactory__thread_pool
        is fake_executor
    )
    assert (
        factory_factory._SplitRuntimeFactoryFactory__thread_watcher
        is fake_watcher
    )


def test_create_pair_aggregator_no_timeout(
    fake_executor,
    fake_watcher,
    mocker,
    mock_queue_factories,
    patch_other_dependencies,
):
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    initializer_no_timeout = GenericFakeRuntimeInitializer[str, str](
        timeout_seconds=None
    )
    mock_aggregator_init = mocker.spy(
        srff_module.RemoteDataAggregatorImpl, "__init__"
    )
    factory_factory._create_pair(initializer_no_timeout)
    mock_aggregator_init.assert_called_once()
    assert mock_queue_factories["default_create_queues"].call_count == 3
    created_aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert created_aggregator_instance.timeout is None
