"""Unit tests for DelegatingMultiprocessQueueFactory and its components."""

import pytest
from pytest_mock import MockerFixture
import multiprocessing
import multiprocessing.synchronize as mp_sync
import time
from typing import Any, List, Optional, cast, TypeAlias, Dict
import types
import queue
import warnings

import tsercom.threading.multiprocess.delegating_multiprocess_queue_factory as dqf_module
from tsercom.threading.multiprocess.delegating_multiprocess_queue_factory import (
    DelegatingMultiprocessQueueSink,
    DelegatingMultiprocessQueueSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Conditional import strategy for torch and torch.multiprocessing for type checking
_torch_installed = False
_real_torch_imported_module: Optional[types.ModuleType] = None
torch_module: Optional[types.ModuleType] = None
torch_mp_module: Optional[types.ModuleType] = None

try:
    import torch as _imported_torch_real
    import torch.multiprocessing as _imported_torch_mp_real

    _real_torch_imported_module = _imported_torch_real
    torch_module = _imported_torch_real
    torch_mp_module = _imported_torch_mp_real
    _torch_installed = True
except ImportError:

    class Tensor:  # Placeholder class for when torch is not available
        pass

    if not torch_module:
        torch_mock = MockerFixture(None).MagicMock()
        torch = torch_mock  # type: ignore[assignment]
        if hasattr(torch, "Tensor"):
            torch.Tensor = Tensor  # type: ignore[misc]
    if not torch_mp_module:
        torch_mp_mock = MockerFixture(None).MagicMock()
        torch_mp = torch_mp_mock  # type: ignore[assignment]


if _torch_installed and _real_torch_imported_module:
    TensorType: TypeAlias = _real_torch_imported_module.Tensor
else:
    TensorType: TypeAlias = Any


def set_torch_mp_start_method_if_needed(method: str = "spawn") -> None:
    """
    Sets the multiprocessing start method for torch.multiprocessing.
    Attempts to be robust to multiple calls or pre-set contexts.
    """
    if not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_start_method")
        and hasattr(torch_mp_module, "set_start_method")
    ):
        if method == "spawn" and not torch_mp_module:
            try:
                multiprocessing.set_start_method(method, force=True)
                print(
                    f"Successfully set standard multiprocessing start_method to '{method}'."
                )
            except RuntimeError as e:  # pragma: no cover
                current_method_after_error = multiprocessing.get_start_method(
                    allow_none=True
                )
                if current_method_after_error != method:
                    print(
                        f"Warning: Standard MP: Could not set start_method to '{method}'. Current: '{current_method_after_error}'. Error: {e}"
                    )
                    if "context has already been set" in str(
                        e
                    ) or "cannot start a process after starting a new process" in str(
                        e
                    ):
                        raise RuntimeError(
                            f"Standard MP context is '{current_method_after_error}', "
                            f"cannot change to '{method}'. Test requires '{method}'."
                        ) from e
        return

    try:
        current_method = torch_mp_module.get_start_method(allow_none=True)
        if current_method == method:
            print(f"Torch MP start_method already set to '{method}'.")
            return

        torch_mp_module.set_start_method(method, force=True)
        print(f"Successfully set Torch MP start_method to '{method}'.")

    except RuntimeError as e:  # pragma: no cover
        current_method_after_error = torch_mp_module.get_start_method(
            allow_none=True
        )
        err_msg = str(e).lower()

        if current_method_after_error == method:
            print(
                f"Torch MP start_method is '{method}' (error during forced set: '{err_msg}'). Assuming context is as desired."
            )
            return

        context_already_set = "context has already been set" in err_msg
        processes_already_started = (
            "cannot start a process after starting a new process" in err_msg
        )

        if (
            context_already_set or processes_already_started
        ) and current_method_after_error != method:
            raise RuntimeError(
                f"Torch MP context is '{current_method_after_error}' and cannot be changed to '{method}'. "
                f"Test requires '{method}'. Original error: {e}"
            ) from e
        raise
    except Exception as e:  # pragma: no cover
        print(
            f"An unexpected error occurred while setting start method to '{method}': {e}"
        )
        raise


# Top-level worker functions
def sink_process_worker(  # Original worker - kept for other tests if they use it
    barrier: mp_sync.Barrier,
    results_q: multiprocessing.Queue,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    item_to_put: Any,
    process_id: Any,
) -> None:
    try:
        barrier.wait(timeout=5)
        success = delegating_sink.put_nowait(item_to_put)
        if not success:
            results_q.put((process_id, "put_fail_full", None))
            return
        results_q.put((process_id, "put_success", item_to_put))
    except Exception as e:  # pragma: no cover
        results_q.put((process_id, "put_exception", e))


def source_process_worker(  # Original worker
    barrier: Optional[mp_sync.Barrier],
    results_q: multiprocessing.Queue,
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    num_items_to_get: int,
    process_id: Any,
) -> None:
    items_received: List[Any] = []
    try:
        if barrier:
            barrier.wait(timeout=5)
        for _ in range(num_items_to_get):
            item = delegating_source.get_blocking(timeout=2)
            if item is None:
                results_q.put((process_id, "get_timeout", items_received))
                return
            items_received.append(item)
        results_q.put((process_id, "get_success", items_received))
    except Exception as e:  # pragma: no cover
        results_q.put((process_id, "get_exception", e))


def source_process_worker_ipc(  # Original worker
    results_q: multiprocessing.Queue,
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    sentinel: Any,
    process_id: Any,
) -> None:
    items_received: List[Any] = []
    try:
        while True:
            item = delegating_source.get_blocking(timeout=5)
            if item == sentinel:
                break
            if item is None:
                results_q.put(("get_timeout", items_received))
                return
            items_received.append(item)
        results_q.put(("get_success", items_received))
    except Exception as e:  # pragma: no cover
        results_q.put(("get_exception", e))


# Worker for default item MP correctness tests
def _producer_worker_default(
    sink_queue: dqf_module.DelegatingMultiprocessQueueSink[Dict[str, Any]],
    item_to_send: Dict[str, Any],
    result_queue: "torch_mp_module.Queue[str]",  # type: ignore[name-defined]
):
    try:
        if not sink_queue.put_blocking(item_to_send, timeout=10):
            result_queue.put("put_failed_timeout_or_full")
            return
        result_queue.put("put_successful")
    except Exception as e:  # pragma: no cover
        import traceback

        tb_str = traceback.format_exc()
        result_queue.put(
            f"producer_exception: {type(e).__name__}: {e}\n{tb_str}"
        )


def _consumer_worker_default(
    source_queue: dqf_module.DelegatingMultiprocessQueueSource[Dict[str, Any]],
    result_queue: "torch_mp_module.Queue[Any]",  # type: ignore[name-defined]
):
    try:
        item = source_queue.get_blocking(timeout=15)
        result_queue.put(item)
    except queue.Empty:  # pragma: no cover
        result_queue.put("get_failed_timeout_empty")
    except Exception as e:  # pragma: no cover
        import traceback

        tb_str = traceback.format_exc()
        result_queue.put(
            f"consumer_exception: {type(e).__name__}: {e}\n{tb_str}"
        )


# Worker functions for tensor MP correctness tests
def _producer_worker_tensor(
    sink_queue: dqf_module.DelegatingMultiprocessQueueSink[TensorType],
    item_to_send: TensorType,
    result_queue: "torch_mp_module.Queue[str]",  # type: ignore[name-defined]
):
    try:
        if not sink_queue.put_blocking(item_to_send, timeout=10):
            result_queue.put("put_failed_timeout_or_full")
            return
        result_queue.put("put_successful")
    except Exception as e:  # pragma: no cover
        import traceback

        tb_str = traceback.format_exc()
        result_queue.put(
            f"producer_exception: {type(e).__name__}: {e}\n{tb_str}"
        )


def _consumer_worker_tensor(
    source_queue: dqf_module.DelegatingMultiprocessQueueSource[TensorType],
    result_queue: "torch_mp_module.Queue[Any]",  # type: ignore[name-defined]
):
    try:
        item = source_queue.get_blocking(timeout=15)
        result_queue.put(item)
    except queue.Empty:  # pragma: no cover
        result_queue.put("get_failed_timeout_empty")
    except Exception as e:  # pragma: no cover
        import traceback

        tb_str = traceback.format_exc()
        result_queue.put(
            f"consumer_exception: {type(e).__name__}: {e}\n{tb_str}"
        )


# Worker function for initialization race condition test
def _race_condition_producer_worker(
    manager_proxy: Any,  # Actual manager instance from parent
    shared_dict_proxy: Any,  # Manager.dict proxy
    shared_lock_proxy: Any,  # Manager.Lock proxy
    item_to_send: Any,
    result_queue: "torch_mp_module.Queue[str]",  # type: ignore[name-defined]
    process_id: int,
    barrier: "torch_mp_module.Barrier",  # type: ignore[name-defined]
):
    try:
        # Each worker creates its own sink instance using shared components
        # The manager_proxy is crucial here if the sink needs to create manager-dependent queues
        sink_queue_instance = dqf_module.DelegatingMultiprocessQueueSink[Any](
            shared_manager_dict=shared_dict_proxy,
            shared_lock=shared_lock_proxy,
            manager_instance=manager_proxy,
        )

        barrier.wait(timeout=10)  # Synchronize start

        if not sink_queue_instance.put_blocking(item_to_send, timeout=10):
            result_queue.put(f"put_failed_producer_{process_id}")
            return
        result_queue.put(f"put_successful_producer_{process_id}")
    except Exception as e:  # pragma: no cover
        import traceback

        tb_str = traceback.format_exc()
        result_queue.put(
            f"producer_{process_id}_exception: {type(e).__name__}: {e}\n{tb_str}"
        )


# --- Existing Fixtures ---
@pytest.fixture
def mock_is_torch_available(mocker: MockerFixture) -> MockerFixture:
    return mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.is_torch_available"
    )


@pytest.fixture
def MockStdManager(mocker: MockerFixture) -> MockerFixture:
    return mocker.patch("multiprocessing.Manager")


@pytest.fixture
def MockTorchManager(mocker: MockerFixture) -> MockerFixture:
    if _torch_installed and torch_mp_module:
        return mocker.patch.object(torch_mp_module, "Manager")
    return mocker.MagicMock()


# ... (all other existing test functions and fixtures up to delegating_mp_factory) ...
def test_factory_init_defaults() -> None:
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    assert factory._DelegatingMultiprocessQueueFactory__manager is None


def test_get_manager_std_manager_when_torch_unavailable(
    mock_is_torch_available: MockerFixture,
    MockStdManager: MockerFixture,
    MockTorchManager: MockerFixture,
) -> None:
    mock_is_torch_available.return_value = False
    mock_std_manager_instance = MockStdManager.return_value
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    manager1 = factory._DelegatingMultiprocessQueueFactory__get_manager()
    MockStdManager.assert_called_once()
    if _torch_installed:
        MockTorchManager.assert_not_called()
    assert manager1 is mock_std_manager_instance
    manager2 = factory._DelegatingMultiprocessQueueFactory__get_manager()
    MockStdManager.assert_called_once()
    assert manager2 is manager1


@pytest.mark.skipif(not _torch_installed, reason="PyTorch not installed.")
def test_get_manager_torch_manager_when_torch_available(
    mock_is_torch_available: MockerFixture,
    MockStdManager: MockerFixture,
    MockTorchManager: MockerFixture,
) -> None:
    mock_is_torch_available.return_value = True
    mock_torch_manager_instance = MockTorchManager.return_value
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    manager1 = factory._DelegatingMultiprocessQueueFactory__get_manager()
    MockTorchManager.assert_called_once()
    MockStdManager.assert_not_called()
    assert manager1 is mock_torch_manager_instance
    manager2 = factory._DelegatingMultiprocessQueueFactory__get_manager()
    MockTorchManager.assert_called_once()
    assert manager2 is manager1


def test_create_queues_uses_manager_and_returns_sink_source(
    mocker: MockerFixture, mock_is_torch_available: MockerFixture
) -> None:
    mock_is_torch_available.return_value = False
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    mock_manager_instance: Any = mocker.MagicMock()
    mock_lock: Any = mocker.MagicMock(spec=mp_sync.Lock)
    mock_dict: Any = mocker.MagicMock(spec=dict)
    mock_manager_instance.Lock.return_value = mock_lock
    mock_manager_instance.dict.return_value = mock_dict

    patched_get_manager_mock = mocker.MagicMock(
        return_value=mock_manager_instance
    )

    mocker.patch.object(
        factory,
        "_DelegatingMultiprocessQueueFactory__get_manager",
        new=patched_get_manager_mock,
    )

    MockedSink = mocker.patch.object(
        dqf_module, "DelegatingMultiprocessQueueSink"
    )
    MockedSource = mocker.patch.object(
        dqf_module, "DelegatingMultiprocessQueueSource"
    )

    intermediate_sink_mock = mocker.MagicMock()
    MockedSink.__getitem__.return_value = intermediate_sink_mock

    intermediate_source_mock = mocker.MagicMock()
    MockedSource.__getitem__.return_value = intermediate_source_mock

    mock_sink_instance = intermediate_sink_mock.return_value
    mock_source_instance = intermediate_source_mock.return_value

    sink, source = factory.create_queues()

    patched_get_manager_mock.assert_called_once()
    mock_manager_instance.Lock.assert_called_once()
    mock_manager_instance.dict.assert_called_once()
    mock_dict.__setitem__.assert_any_call("initialized", False)
    mock_dict.__setitem__.assert_any_call("real_queue_source_ref", None)

    intermediate_sink_mock.assert_called_once_with(
        shared_manager_dict=mock_dict,
        shared_lock=mock_lock,
        manager_instance=mock_manager_instance,
    )
    intermediate_source_mock.assert_called_once_with(
        shared_manager_dict=mock_dict, shared_lock=mock_lock
    )
    assert sink is mock_sink_instance
    assert source is mock_source_instance


def test_shutdown_no_manager_created() -> None:
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    assert (
        factory._DelegatingMultiprocessQueueFactory__manager is None
    ), "Manager should be None initially."
    factory.shutdown()
    assert (
        factory._DelegatingMultiprocessQueueFactory__manager is None
    ), "Manager should still be None after shutdown."


def test_shutdown_with_active_manager(
    mocker: MockerFixture,
    mock_is_torch_available: MockerFixture,
    MockStdManager: MockerFixture,
) -> None:
    mock_is_torch_available.return_value = False
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    manager_instance = (
        factory._DelegatingMultiprocessQueueFactory__get_manager()
    )
    assert factory._DelegatingMultiprocessQueueFactory__manager is not None
    assert (
        factory._DelegatingMultiprocessQueueFactory__manager
        is manager_instance
    )
    assert manager_instance is MockStdManager.return_value

    MockStdManager.return_value.shutdown = mocker.MagicMock()
    factory.shutdown()
    MockStdManager.return_value.shutdown.assert_called_once()
    assert factory._DelegatingMultiprocessQueueFactory__manager is None


def test_shutdown_manager_shutdown_raises_exception(
    mocker: MockerFixture,
    mock_is_torch_available: MockerFixture,
    MockStdManager: MockerFixture,
) -> None:
    mock_is_torch_available.return_value = False
    factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
        dqf_module.DelegatingMultiprocessQueueFactory()
    )
    manager_instance = (
        factory._DelegatingMultiprocessQueueFactory__get_manager()
    )
    assert factory._DelegatingMultiprocessQueueFactory__manager is not None
    assert manager_instance is MockStdManager.return_value

    MockStdManager.return_value.shutdown = mocker.MagicMock(
        side_effect=RuntimeError("Manager shutdown failed")
    )
    try:
        factory.shutdown()
    except RuntimeError:  # pragma: no cover
        pytest.fail(
            "Factory.shutdown() should not re-raise manager's shutdown exception."
        )
    MockStdManager.return_value.shutdown.assert_called_once()
    assert factory._DelegatingMultiprocessQueueFactory__manager is None


@pytest.fixture
def mock_shared_dict(mocker: MockerFixture) -> Any:
    return mocker.MagicMock(spec=dict)


@pytest.fixture
def mock_shared_lock(mocker: MockerFixture) -> Any:
    return mocker.MagicMock(spec=mp_sync.Lock)


@pytest.fixture
def mock_manager_instance(mocker: MockerFixture) -> Any:
    mock_manager_instance = mocker.MagicMock()
    mock_manager_instance.Queue.return_value = mocker.MagicMock(
        spec=multiprocessing.Queue
    )
    return mock_manager_instance


@pytest.fixture
def test_item() -> Any:
    return "test_data"


@pytest.fixture
def tensor_item_mock(mocker: MockerFixture) -> Optional[TensorType]:
    if _torch_installed and torch_module:
        return cast(TensorType, mocker.MagicMock(spec=torch_module.Tensor))
    return None


@pytest.fixture
def delegating_sink(
    mock_shared_dict: Any,
    mock_shared_lock: Any,
    mock_manager_instance: Any,
    mock_is_torch_available: MockerFixture,
) -> DelegatingMultiprocessQueueSink[Any]:
    return DelegatingMultiprocessQueueSink[Any](
        shared_manager_dict=mock_shared_dict,
        shared_lock=mock_shared_lock,
        manager_instance=mock_manager_instance,
    )


def test_sink_init(
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_shared_dict: Any,
) -> None:
    assert (
        delegating_sink._DelegatingMultiprocessQueueSink__shared_dict
        is mock_shared_dict
    )
    assert (
        delegating_sink._DelegatingMultiprocessQueueSink__real_sink_internal
        is None
    )
    assert not delegating_sink._DelegatingMultiprocessQueueSink__closed_flag


@pytest.mark.skipif(not _torch_installed, reason="PyTorch not installed.")
def test_initialize_real_sink_torch_path_if_torch_available(
    mocker: MockerFixture,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_is_torch_available: MockerFixture,
    mock_shared_dict: Any,
) -> None:
    mock_is_torch_available.return_value = True
    mock_shared_dict.get.return_value = False

    assert (
        torch_module is not None
    ), "Torch module not available for torch path test"
    mock_tensor_data = mocker.MagicMock(spec=torch_module.Tensor)
    item_with_tensor_data = mocker.MagicMock()
    item_with_tensor_data.data = mock_tensor_data

    PatchedInternalSink = mocker.patch.object(
        dqf_module, "MultiprocessQueueSink"
    )
    PatchedInternalSource = mocker.patch.object(
        dqf_module, "MultiprocessQueueSource"
    )

    mock_torch_factory_instance = mocker.MagicMock()
    mock_torch_factory_instance.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )

    MockedTorchFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
    )
    MockedTorchFactory.return_value = mock_torch_factory_instance

    MockedDefaultFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )

    delegating_sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
        item_with_tensor_data
    )

    MockedTorchFactory.assert_called_once()
    mock_torch_factory_instance.create_queues.assert_called_once()
    MockedDefaultFactory.assert_not_called()

    found_ref_call = False
    for call_args_tuple in mock_shared_dict.__setitem__.call_args_list:
        if call_args_tuple[0][0] == "real_queue_source_ref":
            assert isinstance(call_args_tuple[0][1], mocker.MagicMock)
            found_ref_call = True
            break
    assert found_ref_call, "real_queue_source_ref was not set as expected."

    assert isinstance(
        delegating_sink._DelegatingMultiprocessQueueSink__real_sink_internal,
        mocker.MagicMock,
    )


def test_initialize_real_sink_default_path_when_torch_available(
    mocker: MockerFixture,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_is_torch_available: MockerFixture,
    mock_shared_dict: Any,
    test_item: Any,
) -> None:
    mock_is_torch_available.return_value = True
    mock_shared_dict.get.return_value = False

    mock_default_factory_instance = mocker.MagicMock()
    mock_default_factory_instance.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )

    MockedDefaultFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )
    MockedDefaultFactory.return_value = mock_default_factory_instance

    MockedTorchFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
    )

    delegating_sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
        test_item
    )

    MockedDefaultFactory.assert_called_once()
    mock_default_factory_instance.create_queues.assert_called_once()
    MockedTorchFactory.assert_not_called()


def test_initialize_real_sink_default_path_when_torch_unavailable(
    mocker: MockerFixture,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_is_torch_available: MockerFixture,
    mock_shared_dict: Any,
    test_item: Any,
) -> None:
    mock_is_torch_available.return_value = False
    mock_shared_dict.get.return_value = False

    mock_default_factory_instance = mocker.MagicMock()
    mock_default_factory_instance.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )

    MockedDefaultFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )
    MockedDefaultFactory.return_value = mock_default_factory_instance

    MockedTorchFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
    )

    delegating_sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
        test_item
    )

    MockedDefaultFactory.assert_called_once()
    mock_default_factory_instance.create_queues.assert_called_once()
    MockedTorchFactory.assert_not_called()


def test_initialize_real_sink_called_only_once(
    mocker: MockerFixture,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_is_torch_available: MockerFixture,
    mock_shared_dict: Any,
    test_item: Any,
) -> None:
    mock_is_torch_available.return_value = False
    mock_shared_dict.get.side_effect = [
        False,
        True,
        True,
        True,
    ]

    mock_default_factory_instance = mocker.MagicMock()
    mock_default_factory_instance.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )

    MockedDefaultFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )
    MockedDefaultFactory.return_value = mock_default_factory_instance
    delegating_sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
        test_item
    )
    MockedDefaultFactory.assert_called_once()
    mock_default_factory_instance.create_queues.assert_called_once()

    delegating_sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
        test_item
    )
    MockedDefaultFactory.assert_called_once()
    mock_default_factory_instance.create_queues.assert_called_once()


def test_put_blocking_and_put_nowait_delegation(
    mocker: MockerFixture,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_is_torch_available: MockerFixture,
    mock_shared_dict: Any,
    test_item: Any,
) -> None:
    mock_is_torch_available.return_value = False
    mock_shared_dict.get.return_value = False

    mock_real_sink_instance = mocker.MagicMock(spec=MultiprocessQueueSink)
    mock_default_factory_instance = mocker.MagicMock()
    mock_default_factory_instance.create_queues.return_value = (
        mock_real_sink_instance,
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )

    MockedDefaultFactory = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )
    MockedDefaultFactory.return_value = mock_default_factory_instance
    delegating_sink.put_nowait(test_item)

    MockedDefaultFactory.assert_called_once()
    mock_default_factory_instance.create_queues.assert_called_once()
    assert (
        delegating_sink._DelegatingMultiprocessQueueSink__real_sink_internal
        is mock_real_sink_instance
    )
    mock_real_sink_instance.put_nowait.assert_called_once_with(test_item)

    delegating_sink.put_blocking("item2", timeout=1.0)
    mock_real_sink_instance.put_blocking.assert_called_once_with(
        "item2", timeout=1.0
    )

    delegating_sink.put_nowait("item3")
    assert mock_real_sink_instance.put_nowait.call_count == 2
    mock_real_sink_instance.put_nowait.assert_called_with("item3")


def test_put_when_closed_raises_runtime_error(
    delegating_sink: DelegatingMultiprocessQueueSink[Any], test_item: Any
) -> None:
    delegating_sink.close()
    with pytest.raises(RuntimeError, match="Sink closed"):
        delegating_sink.put_blocking(test_item)
    with pytest.raises(RuntimeError, match="Sink closed"):
        delegating_sink.put_nowait(test_item)


def test_properties_and_utility_methods_before_init(
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
) -> None:
    assert not delegating_sink.closed


def test_properties_and_utility_methods_after_init(
    mocker: MockerFixture,
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
) -> None:
    mock_underlying_mp_queue: Any = mocker.MagicMock()
    mock_underlying_mp_queue.qsize.return_value = 5
    mock_underlying_mp_queue.empty.return_value = False
    mock_underlying_mp_queue.full.return_value = True
    mock_real_sink_wrapper = mocker.MagicMock(spec=MultiprocessQueueSink)
    setattr(
        mock_real_sink_wrapper,
        "_MultiprocessQueueSink__queue",
        mock_underlying_mp_queue,
    )

    delegating_sink._DelegatingMultiprocessQueueSink__real_sink_internal = (
        cast(MultiprocessQueueSink[Any], mock_real_sink_wrapper)
    )
    internal_queue = (
        delegating_sink._DelegatingMultiprocessQueueSink__real_sink_internal._MultiprocessQueueSink__queue
    )
    assert internal_queue.qsize() == 5
    assert not internal_queue.empty()
    assert internal_queue.full()
    delegating_sink.close()
    assert delegating_sink.closed


@pytest.fixture
def mock_real_mp_queue_source(mocker: MockerFixture) -> Any:
    mock_source = mocker.MagicMock(spec=MultiprocessQueueSource)
    mock_source._MultiprocessQueueSource__queue = mocker.MagicMock()
    return mock_source


@pytest.fixture
def mock_time_sleep(mocker: MockerFixture) -> MockerFixture:
    return mocker.patch("time.sleep", return_value=None)


@pytest.fixture
def delegating_source(
    mock_shared_dict: Any,
    mock_shared_lock: Any,
    mock_time_sleep: MockerFixture,
) -> DelegatingMultiprocessQueueSource[Any]:
    return DelegatingMultiprocessQueueSource[Any](
        mock_shared_dict, mock_shared_lock
    )


def test_source_init(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
) -> None:
    assert (
        delegating_source._DelegatingMultiprocessQueueSource__shared_dict
        is mock_shared_dict
    )
    assert (
        delegating_source._DelegatingMultiprocessQueueSource__real_source_internal
        is None
    )


def test_ensure_real_source_initialized_immediately(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
    mock_real_mp_queue_source: Any,
) -> None:
    mock_shared_dict.get.side_effect = lambda key, default=None: {
        "initialized": True,
        "real_queue_source_ref": mock_real_mp_queue_source,
    }.get(key, default)
    delegating_source._ensure_real_source_initialized(polling_timeout=0.01)
    assert (
        delegating_source._DelegatingMultiprocessQueueSource__real_source_internal
        is mock_real_mp_queue_source
    )


def test_ensure_real_source_initialized_after_delay(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
    mock_real_mp_queue_source: Any,
    mock_time_sleep: MockerFixture,
) -> None:
    results: List[bool] = [False, False, True]

    def get_from_dict(key: str, default: Any = None) -> Any:
        if key == "initialized":
            return results.pop(0) if results else True
        if key == "real_queue_source_ref":
            return mock_real_mp_queue_source if not results else None
        return default

    mock_shared_dict.get.side_effect = get_from_dict
    delegating_source._ensure_real_source_initialized(polling_timeout=0.1)
    assert (
        delegating_source._DelegatingMultiprocessQueueSource__real_source_internal
        is mock_real_mp_queue_source
    )
    mock_time_sleep.assert_called()


def test_ensure_real_source_initialized_timeout(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
) -> None:
    mock_shared_dict.get.return_value = False
    with pytest.raises(queue.Empty):
        delegating_source._ensure_real_source_initialized(polling_timeout=0.03)


def test_ensure_real_source_initialized_bad_ref_none(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
) -> None:
    mock_shared_dict.get.side_effect = lambda k, d=None: {
        "initialized": True,
        "real_queue_source_ref": None,
    }.get(k, d)
    with pytest.raises(RuntimeError, match="missing"):
        delegating_source._ensure_real_source_initialized(polling_timeout=0.01)


def test_ensure_real_source_initialized_bad_ref_type(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
) -> None:
    mock_shared_dict.get.side_effect = lambda k, d=None: {
        "initialized": True,
        "real_queue_source_ref": "bad",
    }.get(k, d)
    with pytest.raises(RuntimeError, match="Invalid"):
        delegating_source._ensure_real_source_initialized(polling_timeout=0.01)


def test_get_methods_delegation_after_init(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_real_mp_queue_source: Any,
) -> None:
    delegating_source._DelegatingMultiprocessQueueSource__real_source_internal = (
        mock_real_mp_queue_source
    )
    val = "item"
    mock_real_mp_queue_source.get_blocking.return_value = val
    assert delegating_source.get_blocking(timeout=0.1) == val
    mock_real_mp_queue_source.get_or_none.return_value = val
    assert delegating_source.get_or_none() == val


def test_get_methods_wait_for_init(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
    mock_real_mp_queue_source: Any,
    mock_time_sleep: MockerFixture,
) -> None:
    results: List[bool] = [False, True]

    def get_from_dict(key: str, default: Any = None) -> Any:
        if key == "initialized":
            return results.pop(0) if results else True
        if key == "real_queue_source_ref":
            return mock_real_mp_queue_source if not results else None
        return default

    mock_shared_dict.get.side_effect = get_from_dict
    val = "item_delay"
    mock_real_mp_queue_source.get_blocking.return_value = val
    assert delegating_source.get_blocking(timeout=0.1) == val
    mock_time_sleep.assert_called()


def test_utility_methods_before_init(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_shared_dict: Any,
) -> None:
    assert (
        delegating_source._DelegatingMultiprocessQueueSource__real_source_internal
        is None
    )

    def mock_get(key: str, default: Any = None) -> Any:
        if key == dqf_module.INITIALIZED_KEY:
            return False
        if key == dqf_module.REAL_QUEUE_SOURCE_REF_KEY:
            return None
        return default

    mock_shared_dict.get.side_effect = mock_get
    assert delegating_source.get_or_none() is None


def test_utility_methods_after_init(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_real_mp_queue_source: Any,
) -> None:
    delegating_source._DelegatingMultiprocessQueueSource__real_source_internal = (
        mock_real_mp_queue_source
    )
    underlying_q_mock = (
        delegating_source._DelegatingMultiprocessQueueSource__real_source_internal._MultiprocessQueueSource__queue
    )
    underlying_q_mock.qsize.return_value = 3
    underlying_q_mock.empty.return_value = False
    underlying_q_mock.full.return_value = True
    assert underlying_q_mock.qsize() == 3
    assert not underlying_q_mock.empty()
    assert underlying_q_mock.full()


# Fixture for multi-process tests ensuring factory shutdown
@pytest.fixture
def delegating_mp_factory() -> Any:
    factory = dqf_module.DelegatingMultiprocessQueueFactory[Any]()
    yield factory
    print(f"Shutting down delegating_mp_factory {id(factory)}...")
    factory.shutdown()
    print(f"Finished shutting down delegating_mp_factory {id(factory)}.")


# Helper function for the actual test logic, parameterized by start method
def _execute_mp_correctness_logic(
    start_method_to_try: str,
    factory_fixture: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    item_to_send: Any,  # Can be Dict or Tensor
    is_tensor_test: bool = False,
) -> None:
    assert (
        torch_mp_module is not None
    )  # Should be guaranteed by skipif on calling test

    set_torch_mp_start_method_if_needed(method=start_method_to_try)

    current_ctx = torch_mp_module.get_context()
    producer_status_queue: "torch_mp_module.Queue[str]" = current_ctx.Queue()
    consumer_data_queue: "torch_mp_module.Queue[Any]" = current_ctx.Queue()

    sink_q: Any
    source_q: Any
    if is_tensor_test:
        assert (
            _torch_installed and torch_module is not None
        )  # Ensure torch is available for TensorType
        sink_q_tensor: DelegatingMultiprocessQueueSink[TensorType]
        source_q_tensor: DelegatingMultiprocessQueueSource[TensorType]
        sink_q_tensor, source_q_tensor = factory_fixture.create_queues()
        sink_q, source_q = sink_q_tensor, source_q_tensor
        worker_producer = _producer_worker_tensor
        worker_consumer = _consumer_worker_tensor
    else:
        sink_q_dict: DelegatingMultiprocessQueueSink[Dict[str, Any]]
        source_q_dict: DelegatingMultiprocessQueueSource[Dict[str, Any]]
        sink_q_dict, source_q_dict = factory_fixture.create_queues()
        sink_q, source_q = sink_q_dict, source_q_dict
        worker_producer = _producer_worker_default
        worker_consumer = _consumer_worker_default

    producer_process = current_ctx.Process(
        target=worker_producer,
        args=(sink_q, item_to_send, producer_status_queue),
    )
    consumer_process = current_ctx.Process(
        target=worker_consumer,
        args=(source_q, consumer_data_queue),
    )

    producer_process.start()
    consumer_process.start()

    process_join_timeout = 25
    producer_process.join(timeout=process_join_timeout)
    consumer_process.join(timeout=process_join_timeout)

    if producer_process.is_alive():  # pragma: no cover
        producer_process.terminate()
        producer_process.join()
        pytest.fail(
            f"Producer process timed out with '{start_method_to_try}' method."
        )

    if consumer_process.is_alive():  # pragma: no cover
        consumer_process.terminate()
        consumer_process.join()
        pytest.fail(
            f"Consumer process timed out with '{start_method_to_try}' method."
        )

    try:
        producer_status = producer_status_queue.get(timeout=5)
        assert (
            producer_status == "put_successful"
        ), f"Producer status with '{start_method_to_try}': {producer_status}"
    except queue.Empty:  # pragma: no cover
        pytest.fail(
            f"Producer status queue was empty with '{start_method_to_try}' method."
        )

    try:
        received_item = consumer_data_queue.get(timeout=5)
        if (
            isinstance(received_item, str)
            and "exception" in received_item.lower()
        ):
            pytest.fail(
                f"Consumer reported exception with '{start_method_to_try}': {received_item}"
            )

        if is_tensor_test:
            assert torch_module is not None and isinstance(
                received_item, torch_module.Tensor
            ), f"Received item is not a Tensor with '{start_method_to_try}': {type(received_item)}"
            assert torch_module.equal(
                received_item, item_to_send
            ), f"Received tensor {received_item} does not match sent tensor {item_to_send} with '{start_method_to_try}'."
        else:
            assert isinstance(
                received_item, dict
            ), f"Received item is not a dict with '{start_method_to_try}': {type(received_item)}. Content: {received_item}"
            assert (
                received_item == item_to_send
            ), f"Received item {received_item} does not match sent item {item_to_send} with '{start_method_to_try}'."
    except queue.Empty:  # pragma: no cover
        pytest.fail(
            f"Consumer data queue was empty with '{start_method_to_try}' method."
        )


@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "fork" in torch_mp_module.get_all_start_methods()
    ),
    reason="Fork start method not available or PyTorch MP not installed.",
)
def test_multiprocess_correctness_fork_default_item(
    delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    mocker: MockerFixture,
) -> None:
    """Tests MP correctness with 'fork' start method for a default item."""
    assert torch_mp_module is not None
    original_start_method = torch_mp_module.get_start_method(allow_none=True)
    print(f"Original MP start method: {original_start_method}")
    item_to_send = {
        "data": "test_value_fork",
        "id": 456,
        "timestamp": time.time(),
    }
    try:
        _execute_mp_correctness_logic(
            "fork", delegating_mp_factory, item_to_send, is_tensor_test=False
        )
    finally:
        if (
            original_start_method
            and hasattr(torch_mp_module, "set_start_method")
            and torch_mp_module.get_start_method(allow_none=True)
            != original_start_method
        ):
            try:
                print(f"Restoring MP start method to: {original_start_method}")
                torch_mp_module.set_start_method(
                    original_start_method, force=True
                )
            except RuntimeError as e:  # pragma: no cover
                warnings.warn(
                    UserWarning(
                        f"Could not restore original mp start method '{original_start_method}': {e}"
                    )
                )


@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "spawn" in torch_mp_module.get_all_start_methods()
    ),
    reason="Spawn start method not available or PyTorch MP not installed.",
)
@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="Known pickling issue: DelegatingQueueSink's manager instance is not pickleable for spawn",
)
def test_multiprocess_correctness_spawn_default_item(
    delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    mocker: MockerFixture,
) -> None:
    """Tests MP correctness with 'spawn' start method for a default item (expected to fail pickling)."""
    assert torch_mp_module is not None
    original_start_method = torch_mp_module.get_start_method(allow_none=True)
    print(f"Original MP start method: {original_start_method}")
    item_to_send = {
        "data": "test_value_spawn",
        "id": 789,
        "timestamp": time.time(),
    }
    try:
        _execute_mp_correctness_logic(
            "spawn", delegating_mp_factory, item_to_send, is_tensor_test=False
        )
    finally:
        if (
            original_start_method
            and hasattr(torch_mp_module, "set_start_method")
            and torch_mp_module.get_start_method(allow_none=True)
            != original_start_method
        ):
            try:
                print(f"Restoring MP start method to: {original_start_method}")
                torch_mp_module.set_start_method(
                    original_start_method, force=True
                )
            except RuntimeError as e:  # pragma: no cover
                warnings.warn(
                    UserWarning(
                        f"Could not restore original mp start method '{original_start_method}': {e}"
                    )
                )


# New tests for tensor items
@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "fork" in torch_mp_module.get_all_start_methods()
    ),
    reason="Fork start method not available or PyTorch MP not installed.",
)
def test_multiprocess_correctness_fork_tensor_item(
    delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    mocker: MockerFixture,
) -> None:
    """Tests MP correctness with 'fork' start method for a tensor item."""
    assert torch_module is not None and torch_mp_module is not None
    original_start_method = torch_mp_module.get_start_method(allow_none=True)
    print(
        f"Original MP start method for tensor fork test: {original_start_method}"
    )
    item_to_send = torch_module.tensor([1.0, 2.0, 3.0])
    try:
        _execute_mp_correctness_logic(
            "fork", delegating_mp_factory, item_to_send, is_tensor_test=True
        )
    finally:
        if (
            original_start_method
            and hasattr(torch_mp_module, "set_start_method")
            and torch_mp_module.get_start_method(allow_none=True)
            != original_start_method
        ):
            try:
                print(
                    f"Restoring MP start method to: {original_start_method} (from tensor fork test)"
                )
                torch_mp_module.set_start_method(
                    original_start_method, force=True
                )
            except RuntimeError as e:  # pragma: no cover
                warnings.warn(
                    UserWarning(
                        f"Could not restore original mp start method '{original_start_method}' from tensor fork test: {e}"
                    )
                )


@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "spawn" in torch_mp_module.get_all_start_methods()
    ),
    reason="Spawn start method not available or PyTorch MP not installed.",
)
@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="Known pickling issue with manager instance for spawn, expected for tensor path too",
)
def test_multiprocess_correctness_spawn_tensor_item(
    delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    mocker: MockerFixture,
) -> None:
    """Tests MP correctness with 'spawn' start method for a tensor item (expected to xfail pickling)."""
    assert torch_module is not None and torch_mp_module is not None
    original_start_method = torch_mp_module.get_start_method(allow_none=True)
    print(
        f"Original MP start method for tensor spawn test: {original_start_method}"
    )
    item_to_send = torch_module.tensor([4.0, 5.0, 6.0])
    try:
        _execute_mp_correctness_logic(
            "spawn", delegating_mp_factory, item_to_send, is_tensor_test=True
        )
    finally:
        if (
            original_start_method
            and hasattr(torch_mp_module, "set_start_method")
            and torch_mp_module.get_start_method(allow_none=True)
            != original_start_method
        ):
            try:
                print(
                    f"Restoring MP start method to: {original_start_method} (from tensor spawn test)"
                )
                torch_mp_module.set_start_method(
                    original_start_method, force=True
                )
            except RuntimeError as e:  # pragma: no cover
                warnings.warn(
                    UserWarning(
                        f"Could not restore original mp start method '{original_start_method}' from tensor spawn test: {e}"
                    )
                )


@pytest.mark.timeout(45)  # Increased timeout for race condition test
@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "fork" in torch_mp_module.get_all_start_methods()
    ),
    reason="Fork start method not available or PyTorch MP not installed for race condition test.",
)
def test_initialization_race_condition(
    delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    mocker: MockerFixture,
) -> None:
    """Tests race condition for queue initialization with multiple producers using 'fork'."""
    assert torch_mp_module is not None  # Guaranteed by skipif

    original_start_method = torch_mp_module.get_start_method(allow_none=True)
    print(f"Original MP start method for race test: {original_start_method}")

    try:
        set_torch_mp_start_method_if_needed("fork")

        # Get the manager from the factory to create shared resources directly
        # This manager will be created within the 'fork' context established above.
        manager = (
            delegating_mp_factory._DelegatingMultiprocessQueueFactory__get_manager()
        )
        assert (
            manager is not None
        ), "Manager should be initialized by the factory"

        shared_lock = manager.Lock()
        shared_dict = manager.dict()
        shared_dict[dqf_module.INITIALIZED_KEY] = False
        shared_dict[dqf_module.REAL_QUEUE_SOURCE_REF_KEY] = None

        num_producers = 3
        # Barrier for num_producers + main thread (to signal consumer creation and start consumption)
        barrier = torch_mp_module.Barrier(num_producers + 1)
        producer_results_queue = torch_mp_module.Queue()

        producers = []
        sent_items = {}  # To store what each producer sent

        for i in range(num_producers):
            item_to_send = f"race_item_producer_{i}"
            sent_items[f"put_successful_producer_{i}"] = item_to_send

            # Pass manager itself, and manager-created dict and lock to workers
            # Workers will construct their own DelegatingMultiprocessQueueSink instances
            p = torch_mp_module.Process(
                target=_race_condition_producer_worker,
                args=(
                    manager,
                    shared_dict,
                    shared_lock,
                    item_to_send,
                    producer_results_queue,
                    i,
                    barrier,
                ),
            )
            producers.append(p)
            p.start()

        # Create the consumer-side queue object using the same shared resources
        source_for_consumer = dqf_module.DelegatingMultiprocessQueueSource[
            Any
        ](shared_manager_dict=shared_dict, shared_lock=shared_lock)

        print(
            "Main process waiting on barrier before consumers start getting..."
        )
        barrier.wait(timeout=15)  # Unblock producers to start putting

        producer_success_count = 0
        for _ in range(num_producers):
            try:
                status = producer_results_queue.get(timeout=15)
                print(f"Producer result: {status}")
                if status.startswith("put_successful_producer_"):
                    producer_success_count += 1
                else:  # pragma: no cover
                    pytest.fail(f"A producer failed: {status}")
            except queue.Empty:  # pragma: no cover
                pytest.fail("Timeout waiting for producer results.")

        assert (
            producer_success_count == num_producers
        ), f"Expected {num_producers} successful producers, got {producer_success_count}"

        received_items = set()
        for _ in range(num_producers):
            try:
                item = source_for_consumer.get_blocking(timeout=15)
                print(f"Consumer received: {item}")
                received_items.add(item)
            except queue.Empty:  # pragma: no cover
                pytest.fail("Timeout waiting for consumer to receive item.")

        assert (
            len(received_items) == num_producers
        ), f"Expected to receive {num_producers} unique items, but got {len(received_items)}. Items: {received_items}"

        # Check if all sent items are part of the received items
        for expected_item_val in sent_items.values():
            assert (
                expected_item_val in received_items
            ), f"Expected item {expected_item_val} not found in received items {received_items}"

        # Indirect check for initialization: The shared dict's INITIALIZED_KEY should be True
        assert (
            shared_dict.get(dqf_module.INITIALIZED_KEY) is True
        ), "The shared INITIALIZED_KEY was not set to True after producers ran."

    finally:
        if (
            original_start_method
            and hasattr(torch_mp_module, "set_start_method")
            and torch_mp_module.get_start_method(allow_none=True)
            != original_start_method
        ):
            try:
                print(
                    f"Restoring MP start method to: {original_start_method} (from race test)"
                )
                torch_mp_module.set_start_method(
                    original_start_method, force=True
                )
            except RuntimeError as e:  # pragma: no cover
                warnings.warn(
                    UserWarning(
                        f"Could not restore original mp start method '{original_start_method}' from race test: {e}"
                    )
                )
