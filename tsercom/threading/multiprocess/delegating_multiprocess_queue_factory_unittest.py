"""Unit tests for DelegatingMultiprocessQueueFactory and its components
with the new eager creation, lazy selection architecture."""

import pytest
from pytest_mock import MockerFixture
import multiprocessing
import time
from typing import Any, List, Optional, cast, TypeAlias, Dict
import types
import queue  # For queue.Empty

# Module under test
import tsercom.threading.multiprocess.delegating_multiprocess_queue_factory as dqf_module
from tsercom.threading.multiprocess.delegating_multiprocess_queue_factory import (
    DelegatingMultiprocessQueueSink,
    DelegatingMultiprocessQueueSource,
    DelegatingMultiprocessQueueFactory,
    USE_TORCH_QUEUE_MSG,
    USE_DEFAULT_QUEUE_MSG,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)


# Conditional import strategy for torch and torch.multiprocessing for type checking
_torch_installed = False
_real_torch_imported_module: Optional[types.ModuleType] = None
torch_module: Optional[types.ModuleType] = (
    None  # For tests to use torch.tensor etc.
)
torch_mp_module: Optional[types.ModuleType] = (
    None  # For torch.multiprocessing specific things
)

try:
    import torch as _imported_torch_real
    import torch.multiprocessing as _imported_torch_mp_real

    _real_torch_imported_module = _imported_torch_real
    torch_module = _imported_torch_real
    torch_mp_module = _imported_torch_mp_real
    _torch_installed = True
    TensorType: TypeAlias = _imported_torch_real.Tensor
except ImportError:
    TensorType: TypeAlias = Any  # Placeholder if torch not installed

    class MockTensor:
        pass

    if (
        not hasattr(globals(), "torch_module")
        or globals()["torch_module"] is None
    ):

        class FallbackTorchMock:
            Tensor = MockTensor

            def __getattr__(self, name):
                return FallbackTorchMock()

            def __call__(self, *args, **kwargs):
                return FallbackTorchMock()

        globals()["torch_module"] = FallbackTorchMock()  # type: ignore
        globals()["torch_mp_module"] = FallbackTorchMock()  # type: ignore


def get_mp_context():
    if (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_context")
    ):
        # Prefer torch.multiprocessing if available and configured
        try:
            # Check if a context is already set, which is a good sign it's usable
            # get_start_method will throw error if not initialized on some OS, so be careful
            # If get_context() itself works, it's a good sign.
            return torch_mp_module.get_context()
        except Exception:  # Fallback if torch.mp context isn't quite ready
            pass
    return multiprocessing.get_context()


def set_torch_mp_start_method_if_needed(
    method: str = "spawn", force: bool = True
) -> None:
    mp_context_to_use = multiprocessing
    context_name = "standard multiprocessing"
    can_use_torch_mp = False

    if (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_start_method")
        and hasattr(torch_mp_module, "set_start_method")
    ):
        # Check if torch.multiprocessing has already been initialized
        # If get_start_method doesn't raise an error and returns a value, it's likely initialized.
        try:
            # if torch_mp_module.get_start_method(allow_none=True) is not None or method == "fork": # fork can sometimes be set implicitly
            mp_context_to_use = torch_mp_module  # type: ignore
            context_name = "Torch multiprocessing"
            can_use_torch_mp = True
        except RuntimeError:  # Not initialized yet
            if (
                multiprocessing.get_start_method(allow_none=True) is None
            ):  # if std mp is also not set, try setting torch mp
                mp_context_to_use = torch_mp_module  # type: ignore
                context_name = "Torch multiprocessing"
                can_use_torch_mp = True
            else:  # std mp is set, torch mp is not, stick to std mp to avoid conflict
                print(
                    f"Torch MP context not available or std MP already set to {multiprocessing.get_start_method(allow_none=True)}. Using std MP for setting start method."
                )
                pass

    try:
        current_method = mp_context_to_use.get_start_method(allow_none=True)
        if current_method == method and not force:
            print(f"{context_name} start_method already set to '{method}'.")
            return

        # For "fork", if it's already the method, setting it again (even with force=True)
        # can cause "cannot start a process after starting a new process" if any process (e.g. manager) has started.
        # So, if current_method is already 'fork' and we want 'fork', just return.
        if current_method == method and method == "fork":
            print(
                f"{context_name} start_method already '{method}'. Skipping set_start_method."
            )
            return

        mp_context_to_use.set_start_method(method, force=force)  # type: ignore
        print(f"Successfully set {context_name} start_method to '{method}'.")

    except RuntimeError as e:
        current_method_after_error = mp_context_to_use.get_start_method(
            allow_none=True
        )
        if current_method_after_error == method:
            print(
                f"{context_name} start_method is '{method}' (error during forced set: '{e}'). Assuming context is as desired."
            )
            return

        err_str = str(e).lower()
        if (
            "context has already been set" in err_str
            or "cannot start a process after starting a new process" in err_str
        ):
            if current_method_after_error != method:
                # If we are trying to set to 'spawn' and it's already 'fork', this is a critical issue for tests needing 'spawn'.
                # If torch mp was desired but we fell back to std mp, and std mp is wrong, also critical.
                accept_current = False
                if (
                    can_use_torch_mp
                    and mp_context_to_use != torch_mp_module
                    and multiprocessing.get_start_method(allow_none=True)
                    == method
                ):
                    # Fell back to std mp, but std mp is already the desired method
                    print(
                        f"Standard MP context is already '{method}'. Using it."
                    )
                    accept_current = True

                if not accept_current:
                    raise RuntimeError(
                        f"{context_name} context is '{current_method_after_error}' and cannot be changed to '{method}'. "
                        f"Test requires '{method}'. Original error: {e}"
                    ) from e
            else:  # context already set, and it's the one we want
                print(
                    f"{context_name} context was already '{method}'. Proceeding."
                )
        else:  # Other runtime error
            raise
    except Exception as e:
        current_method_final = mp_context_to_use.get_start_method(
            allow_none=True
        )
        if current_method_final != method:  # Final check
            raise RuntimeError(
                f"Failed to set {context_name} start method to '{method}'. Current: '{current_method_final}'. Error: {e}"
            ) from e


@pytest.fixture
def mock_default_queue_factory(mocker: MockerFixture) -> MockerFixture:
    mock = mocker.MagicMock(spec=DefaultMultiprocessQueueFactory)
    mock.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )
    return mock


@pytest.fixture
def mock_torch_queue_factory(mocker: MockerFixture) -> MockerFixture:
    if not _torch_installed:
        # Return a simple mock if torch isn't there, it shouldn't be used anyway by factory
        dummy_mock = mocker.MagicMock()
        dummy_mock.create_queues.return_value = (
            mocker.MagicMock(),
            mocker.MagicMock(),
        )
        return dummy_mock

    mock = mocker.MagicMock(spec=TorchMultiprocessQueueFactory)
    mock.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )
    return mock


@pytest.fixture
def delegating_factory_instance(
    mocker: MockerFixture,
    mock_default_queue_factory: MockerFixture,
    mock_torch_queue_factory: MockerFixture,
) -> DelegatingMultiprocessQueueFactory[Any]:
    mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory",
        return_value=mock_default_queue_factory,
    )
    # Only patch TorchMultiprocessQueueFactory if torch is considered installed by the test environment
    if _torch_installed:
        mocker.patch(
            "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory",
            return_value=mock_torch_queue_factory,
        )

    factory = DelegatingMultiprocessQueueFactory[Any]()
    yield factory
    factory.shutdown()


@pytest.fixture
def default_item() -> Dict[str, Any]:
    return {
        "type": "default",
        "data": "some_data",
        "id": 123,
        "timestamp": time.time(),
    }


@pytest.fixture
def tensor_item(mocker: MockerFixture) -> Optional[TensorType]:
    if not (_torch_installed and torch_module):
        return None
    return torch_module.tensor([1.0, 2.0, 3.0], dtype=torch_module.float32)


def test_factory_init_eager_creation(
    mocker: MockerFixture,
) -> None:
    MockedDefaultFactoryCons = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )
    MockedTorchFactoryCons = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
    )

    mock_default_factory_inst = MockedDefaultFactoryCons.return_value
    mock_torch_factory_inst = MockedTorchFactoryCons.return_value

    mock_default_factory_inst.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )
    mock_torch_factory_inst.create_queues.return_value = (
        mocker.MagicMock(spec=MultiprocessQueueSink),
        mocker.MagicMock(spec=MultiprocessQueueSource),
    )

    original_torch_available = dqf_module._TORCH_AVAILABLE
    dqf_module._TORCH_AVAILABLE = (
        True  # Test the path where torch is considered available
    )

    factory = DelegatingMultiprocessQueueFactory[Any]()

    MockedDefaultFactoryCons.assert_called_once()
    mock_default_factory_inst.create_queues.assert_called_once()

    # If _TORCH_AVAILABLE is true, Torch factory should be created.
    # This relies on the actual _torch_installed at import time of dqf_module for the real code path,
    # but here we are overriding dqf_module._TORCH_AVAILABLE for the test.
    if dqf_module._TORCH_AVAILABLE:  # Check the overridden value
        MockedTorchFactoryCons.assert_called_once()
        mock_torch_factory_inst.create_queues.assert_called_once()
    else:  # This case won't run due to override above, but good for completeness
        MockedTorchFactoryCons.assert_not_called()
        mock_torch_factory_inst.create_queues.assert_not_called()

    dqf_module._TORCH_AVAILABLE = (
        False  # Test the path where torch is considered unavailable
    )
    factory_no_torch = DelegatingMultiprocessQueueFactory[Any]()
    # Default factory is called again because it's a new instance of DelegatingMultiprocessQueueFactory
    # The mock counts calls on the class, not per instance of DelegatingMultiprocessQueueFactory
    assert (
        MockedDefaultFactoryCons.call_count == 2
    )  # Called for `factory` and `factory_no_torch`

    # Torch factory should not be called additionally if _TORCH_AVAILABLE is false
    # Its call count should remain 1 (from the first factory instance when _TORCH_AVAILABLE was True)
    assert MockedTorchFactoryCons.call_count == 1

    dqf_module._TORCH_AVAILABLE = original_torch_available  # Restore
    factory.shutdown()
    factory_no_torch.shutdown()


def test_factory_create_queues_returns_delegating_wrappers(
    delegating_factory_instance: DelegatingMultiprocessQueueFactory[Any],
) -> None:
    sink, source = delegating_factory_instance.create_queues()
    assert isinstance(sink, DelegatingMultiprocessQueueSink)
    assert isinstance(source, DelegatingMultiprocessQueueSource)


def test_factory_shutdown_calls_underlying_shutdowns(
    mocker: MockerFixture,
) -> None:
    MockedDefaultFactoryCons = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    )
    MockedTorchFactoryCons = mocker.patch(
        "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
    )

    mock_default_factory_inst = MockedDefaultFactoryCons.return_value
    mock_torch_factory_inst = MockedTorchFactoryCons.return_value

    mock_default_factory_inst.create_queues.return_value = (
        mocker.MagicMock(),
        mocker.MagicMock(),
    )
    mock_torch_factory_inst.create_queues.return_value = (
        mocker.MagicMock(),
        mocker.MagicMock(),
    )

    original_torch_available = dqf_module._TORCH_AVAILABLE
    dqf_module._TORCH_AVAILABLE = (
        True  # Assume torch available for this test unit
    )

    factory = DelegatingMultiprocessQueueFactory[Any]()

    mock_default_factory_inst.shutdown = mocker.MagicMock()
    mock_torch_factory_inst.shutdown = mocker.MagicMock()

    factory.shutdown()

    mock_default_factory_inst.shutdown.assert_called_once()
    if (
        dqf_module._TORCH_AVAILABLE
    ):  # Based on the mocked value during factory creation
        mock_torch_factory_inst.shutdown.assert_called_once()

    dqf_module._TORCH_AVAILABLE = original_torch_available


@pytest.fixture
def mock_underlying_default_sink(
    mocker: MockerFixture,
) -> MultiprocessQueueSink[Any]:
    return mocker.MagicMock(spec=MultiprocessQueueSink)


@pytest.fixture
def mock_underlying_torch_sink(
    mocker: MockerFixture,
) -> Optional[MultiprocessQueueSink[Any]]:
    if not _torch_installed:
        return None
    return mocker.MagicMock(spec=MultiprocessQueueSink)


@pytest.fixture
def delegating_sink(
    mock_underlying_default_sink: MultiprocessQueueSink[Any],
    mock_underlying_torch_sink: Optional[MultiprocessQueueSink[Any]],
) -> DelegatingMultiprocessQueueSink[Any]:
    return DelegatingMultiprocessQueueSink[Any](
        default_sink=mock_underlying_default_sink,
        torch_sink=mock_underlying_torch_sink,
    )


def test_sink_put_default_item(
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_underlying_default_sink: MultiprocessQueueSink[Any],
    mock_underlying_torch_sink: Optional[MultiprocessQueueSink[Any]],
    default_item: Dict[str, Any],
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        dqf_module, "_TORCH_AVAILABLE", False
    )  # Ensure default path chosen

    delegating_sink.put_nowait(default_item)

    mock_underlying_default_sink.put_nowait.assert_any_call(
        USE_DEFAULT_QUEUE_MSG
    )
    mock_underlying_default_sink.put_nowait.assert_any_call(default_item)

    if mock_underlying_torch_sink:
        mock_underlying_torch_sink.put_nowait.assert_not_called()

    another_default_item = {"data": "another"}
    delegating_sink.put_nowait(another_default_item)
    # Calls: 1 for USE_DEFAULT_QUEUE_MSG, 1 for default_item, 1 for another_default_item
    assert mock_underlying_default_sink.put_nowait.call_count == 3
    mock_underlying_default_sink.put_nowait.assert_any_call(
        another_default_item
    )


@pytest.mark.skipif(
    not (_torch_installed and torch_module),
    reason="PyTorch not installed/available.",
)
def test_sink_put_tensor_item(
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_underlying_default_sink: MultiprocessQueueSink[Any],
    mock_underlying_torch_sink: MultiprocessQueueSink[Any],
    tensor_item: TensorType,
    mocker: MockerFixture,
) -> None:
    assert mock_underlying_torch_sink is not None, (
        "Torch sink mock should be present for this test"
    )
    assert tensor_item is not None, "Tensor item fixture failed"

    mocker.patch.object(dqf_module, "_TORCH_AVAILABLE", True)
    mocker.patch.object(dqf_module, "_torch_tensor_type", torch_module.Tensor)

    delegating_sink.put_nowait(tensor_item)

    mock_underlying_default_sink.put_nowait.assert_called_once_with(
        USE_TORCH_QUEUE_MSG
    )
    mock_underlying_torch_sink.put_nowait.assert_called_once_with(tensor_item)

    another_tensor = torch_module.tensor([4.0])  # type: ignore
    delegating_sink.put_nowait(another_tensor)

    # Check call with another_tensor manually due to tensor comparison issues with assert_any_call
    found_another_tensor_call = False
    for call_args in mock_underlying_torch_sink.put_nowait.call_args_list:
        args, _ = call_args
        if len(args) == 1 and torch_module.equal(args[0], another_tensor):
            found_another_tensor_call = True
            break
    assert found_another_tensor_call, (
        "put_nowait was not called with another_tensor"
    )

    assert mock_underlying_torch_sink.put_nowait.call_count == 2
    assert mock_underlying_default_sink.put_nowait.call_count == 1


def test_sink_put_blocking(
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    mock_underlying_default_sink: MultiprocessQueueSink[Any],
    default_item: Dict[str, Any],
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(dqf_module, "_TORCH_AVAILABLE", False)
    delegating_sink.put_blocking(default_item, timeout=1.0)
    mock_underlying_default_sink.put_blocking.assert_any_call(
        default_item, timeout=1.0
    )
    mock_underlying_default_sink.put_nowait.assert_called_once_with(
        USE_DEFAULT_QUEUE_MSG
    )


def test_sink_closed_property_and_method(
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
) -> None:
    assert not delegating_sink.closed
    delegating_sink.close()
    assert delegating_sink.closed
    with pytest.raises(RuntimeError, match="Sink closed"):
        delegating_sink.put_nowait("item")


@pytest.fixture
def mock_underlying_default_source(
    mocker: MockerFixture,
) -> MultiprocessQueueSource[Any]:
    return mocker.MagicMock(spec=MultiprocessQueueSource)


@pytest.fixture
def mock_underlying_torch_source(
    mocker: MockerFixture,
) -> Optional[MultiprocessQueueSource[Any]]:
    if not _torch_installed:
        return None
    return mocker.MagicMock(spec=MultiprocessQueueSource)


@pytest.fixture
def delegating_source(
    mock_underlying_default_source: MultiprocessQueueSource[Any],
    mock_underlying_torch_source: Optional[MultiprocessQueueSource[Any]],
) -> DelegatingMultiprocessQueueSource[Any]:
    return DelegatingMultiprocessQueueSource[Any](
        default_source=mock_underlying_default_source,
        torch_source=mock_underlying_torch_source,
    )


def test_source_get_default_path(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_underlying_default_source: MultiprocessQueueSource[Any],
    mock_underlying_torch_source: Optional[MultiprocessQueueSource[Any]],
    default_item: Dict[str, Any],
) -> None:
    another_item = "another_item_str"
    mock_underlying_default_source.get_blocking.side_effect = [
        USE_DEFAULT_QUEUE_MSG,  # For coordination
        default_item,  # First actual item
        another_item,  # Second actual item
    ]

    item1 = delegating_source.get_blocking(timeout=0.1)
    assert item1 == default_item

    # get_blocking calls: 1 for coord, 1 for item1
    assert mock_underlying_default_source.get_blocking.call_count == 2
    # Check that the first call was for coordination (timeout passed through)
    # and the second call was for the data item (timeout passed through)
    mock_underlying_default_source.get_blocking.assert_any_call(timeout=0.1)

    item2 = delegating_source.get_blocking(timeout=0.1)
    assert item2 == another_item
    assert (
        mock_underlying_default_source.get_blocking.call_count == 3
    )  # Total calls

    if mock_underlying_torch_source:
        mock_underlying_torch_source.get_blocking.assert_not_called()


@pytest.mark.skipif(
    not (_torch_installed and torch_module),
    reason="PyTorch not installed/available.",
)
def test_source_get_tensor_path(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_underlying_default_source: MultiprocessQueueSource[Any],
    mock_underlying_torch_source: MultiprocessQueueSource[Any],
    tensor_item: TensorType,
) -> None:
    assert mock_underlying_torch_source is not None
    assert tensor_item is not None

    another_tensor = torch_module.tensor([4.0])  # type: ignore

    mock_underlying_default_source.get_blocking.return_value = (
        USE_TORCH_QUEUE_MSG
    )
    mock_underlying_torch_source.get_blocking.side_effect = [
        tensor_item,
        another_tensor,
    ]

    item1 = delegating_source.get_blocking(timeout=0.1)
    assert item1 is tensor_item

    mock_underlying_default_source.get_blocking.assert_called_once_with(
        timeout=0.1
    )
    mock_underlying_torch_source.get_blocking.assert_called_once_with(
        timeout=0.1
    )

    item2 = delegating_source.get_blocking(timeout=0.1)
    assert item2 is another_tensor
    assert mock_underlying_default_source.get_blocking.call_count == 1
    assert mock_underlying_torch_source.get_blocking.call_count == 2


def test_source_get_or_none(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_underlying_default_source: MultiprocessQueueSource[Any],
    default_item: Dict[str, Any],
) -> None:
    # Simulate coordination message on default_source.get_blocking(timeout=0.001)
    mock_underlying_default_source.get_blocking.return_value = (
        USE_DEFAULT_QUEUE_MSG
    )
    # Simulate actual data item on default_source.get_or_none()
    mock_underlying_default_source.get_or_none.return_value = default_item

    item = delegating_source.get_or_none()
    assert item == default_item

    mock_underlying_default_source.get_blocking.assert_called_once_with(
        timeout=0.001
    )
    mock_underlying_default_source.get_or_none.assert_called_once()


def test_source_get_blocking_timeout_on_coordination(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_underlying_default_source: MultiprocessQueueSource[Any],
) -> None:
    mock_underlying_default_source.get_blocking.side_effect = queue.Empty
    item = delegating_source.get_blocking(timeout=0.01)
    assert item is None  # Should return None if coordination times out
    mock_underlying_default_source.get_blocking.assert_called_once_with(
        timeout=0.01
    )


def test_source_get_blocking_timeout_on_data(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_underlying_default_source: MultiprocessQueueSource[Any],
) -> None:
    # First call to get_blocking (coordination) is successful
    # Second call to get_blocking (data) results in None (simulating timeout)
    mock_underlying_default_source.get_blocking.side_effect = [
        USE_DEFAULT_QUEUE_MSG,
        None,
    ]

    item = delegating_source.get_blocking(timeout=0.01)
    assert item is None

    assert mock_underlying_default_source.get_blocking.call_count == 2
    # First call for coord, second for data, both with the specified timeout
    mock_underlying_default_source.get_blocking.assert_any_call(timeout=0.01)


def test_source_invalid_coordination_message(
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    mock_underlying_default_source: MultiprocessQueueSource[Any],
) -> None:
    mock_underlying_default_source.get_blocking.return_value = (
        "INVALID_MESSAGE"
    )
    with pytest.raises(
        RuntimeError,
        match="Invalid coordination message received: INVALID_MESSAGE",
    ):
        delegating_source.get_blocking(timeout=0.01)


@pytest.fixture
def mp_delegating_factory(
    request,
) -> Any:  # request is a built-in pytest fixture
    # Attempt to set start_method based on marker or test name if needed,
    # otherwise ensure it's suitable for most tests (e.g. 'fork' on linux, 'spawn' elsewhere or if specified)
    # For simplicity, we'll often rely on tests calling set_torch_mp_start_method_if_needed themselves.

    # Clean up previous contexts if any - this can be tricky
    # One robust way is to ensure each test that uses multiprocessing runs in a way
    # that it can define its context without interference.
    # Pytest runs tests in the same process, so MP context is shared unless care is taken.

    # A common pattern for MP tests is to have a fixture that sets the start method
    # for the duration of the test, then restores it.
    # However, `set_start_method` can often only be called once.

    factory = dqf_module.DelegatingMultiprocessQueueFactory[Any]()
    yield factory
    print(f"Shutting down mp_delegating_factory {id(factory)}...")
    factory.shutdown()  # Essential to release manager resources
    print(f"Finished shutting down mp_delegating_factory {id(factory)}.")


def _mp_producer_worker(
    sink_queue: DelegatingMultiprocessQueueSink[Any],
    items_to_send: List[Any],
    result_queue_mp: queue.Queue,
    worker_id: int,
) -> None:
    try:
        print(f"Producer {worker_id} starting, items: {len(items_to_send)}")
        for i, item in enumerate(items_to_send):
            print(f"Producer {worker_id} putting item {i}: {type(item)}")
            put_success = sink_queue.put_blocking(
                item, timeout=20
            )  # Increased timeout
            if not put_success:
                result_queue_mp.put(
                    f"producer_{worker_id}_put_failed_item_{i}"
                )
                return
            print(f"Producer {worker_id} successfully put item {i}")
        result_queue_mp.put(f"producer_{worker_id}_success")
        print(f"Producer {worker_id} finished normally.")
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        result_queue_mp.put(
            f"producer_{worker_id}_exception: {type(e).__name__}: {e}\n{tb_str}"
        )
        print(f"Producer {worker_id} CRASHED: {e}\n{tb_str}")


def _mp_consumer_worker(
    source_queue: DelegatingMultiprocessQueueSource[Any],
    num_items_expected: int,
    result_queue_mp: queue.Queue,
    worker_id: int,
) -> None:
    received_items: List[Any] = []
    print(
        f"Consumer {worker_id} starting, expecting {num_items_expected} items."
    )
    try:
        for i in range(num_items_expected):
            print(f"Consumer {worker_id} getting item {i}...")
            item = source_queue.get_blocking(timeout=25)  # Increased timeout
            if item is None:
                result_queue_mp.put(
                    f"consumer_{worker_id}_get_timed_out_after_{len(received_items)}_items"
                )
                return
            print(f"Consumer {worker_id} got item {i}: {type(item)}")
            received_items.append(item)
        result_queue_mp.put(received_items)
        print(
            f"Consumer {worker_id} finished normally after receiving {len(received_items)} items."
        )
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        result_queue_mp.put(
            f"consumer_{worker_id}_exception: {type(e).__name__}: {e}\n{tb_str}"
        )
        print(f"Consumer {worker_id} CRASHED: {e}\n{tb_str}")


def _run_mp_test_logic(
    mp_start_method: str,
    # factory: DelegatingMultiprocessQueueFactory[Any], # Factory will be created inside
    items_to_send: List[Any],
    is_tensor_test: bool,
    process_id_suffix: str = "",
    mocker_fixture: Optional[
        MockerFixture
    ] = None,  # For tests needing to mock _TORCH_AVAILABLE
    force_torch_unavailable: bool = False,
) -> None:
    # This is critical: set start method *before* creating any MP objects like Queues for results.
    # `force=True` is important if other tests might have set it.
    set_torch_mp_start_method_if_needed(mp_start_method, force=True)

    # Corrected logic for setting PyTorch sharing strategy
    if _torch_installed and hasattr(torch_mp_module, "set_sharing_strategy"):
        desired_strategy = (
            "file_system"  # Default for fork, and now for spawn too
        )
        # if mp_start_method == 'spawn':
        #     desired_strategy = 'file_descriptor' # Keeping this commented
        try:
            current_strategy = torch_mp_module.get_sharing_strategy()
            if current_strategy != desired_strategy:
                torch_mp_module.set_sharing_strategy(desired_strategy)
                print(
                    f"INFO: Set PyTorch sharing strategy to '{desired_strategy}' for {mp_start_method} test: {process_id_suffix}"
                )
        except RuntimeError as e:  # pylint: disable=broad-except
            print(
                f"WARNING: Could not set PyTorch sharing strategy to '{desired_strategy}' for {mp_start_method} test {process_id_suffix}. Error: {e}"
            )

    # Get context *after* setting start method
    ctx = get_mp_context()

    original_torch_available_state = None
    original_torch_tensor_type_state = None

    if force_torch_unavailable and mocker_fixture:
        # This path is specifically for test_factory_behavior_torch_unavailable_e2e
        original_torch_available_state = dqf_module._TORCH_AVAILABLE
        original_torch_tensor_type_state = dqf_module._torch_tensor_type
        # We are not just mocking the global; we are ensuring the factory instance
        # itself is created while the module-level _TORCH_AVAILABLE is False.
        # This requires patching the module attribute before DelegatingMultiprocessQueueFactory is instantiated.
        mocker_fixture.patch.object(dqf_module, "_TORCH_AVAILABLE", False)
        mocker_fixture.patch.object(dqf_module, "_torch_tensor_type", None)

    factory_to_test = dqf_module.DelegatingMultiprocessQueueFactory[Any]()

    # These queues are for results/status from child processes back to the main test process.
    # They must be created by the context derived *after* set_start_method.
    producer_result_q_mp = ctx.Queue()
    consumer_result_q_mp = ctx.Queue()

    # These are the queues under test, created by the factory.
    sink, source = factory_to_test.create_queues()

    producer_process = ctx.Process(
        target=_mp_producer_worker,
        args=(
            sink,
            items_to_send,
            producer_result_q_mp,
            f"p{process_id_suffix}",
        ),
        name=f"TestProducer-{mp_start_method}-{process_id_suffix}",
    )
    consumer_process = ctx.Process(
        target=_mp_consumer_worker,
        args=(
            source,
            len(items_to_send),
            consumer_result_q_mp,
            f"c{process_id_suffix}",
        ),
        name=f"TestConsumer-{mp_start_method}-{process_id_suffix}",
    )

    print(
        f"Starting producer for {mp_start_method} test ({process_id_suffix})..."
    )
    producer_process.start()
    print(
        f"Starting consumer for {mp_start_method} test ({process_id_suffix})..."
    )
    consumer_process.start()

    # Wait for producer to finish and check its status
    # Timeout should be generous for MP tests.
    join_timeout_producer = 35
    join_timeout_consumer = (
        40  # Consumer might wait longer if producer is slow
    )

    print(f"Waiting for producer status ({process_id_suffix})...")
    producer_status = producer_result_q_mp.get(timeout=join_timeout_producer)
    assert producer_status == f"producer_p{process_id_suffix}_success", (
        f"Producer failed for '{mp_start_method}' ({process_id_suffix}): {producer_status}"
    )
    print(f"Producer finished successfully ({process_id_suffix}).")

    # Wait for consumer to finish and check its results
    print(f"Waiting for consumer results ({process_id_suffix})...")
    consumer_output = consumer_result_q_mp.get(
        timeout=join_timeout_consumer
    )  # Consumer might take longer

    # Add detailed logging for consumer output type
    print(
        f"Consumer output type for '{mp_start_method}' ({process_id_suffix}): {type(consumer_output)}"
    )
    print(
        f"Consumer output content for '{mp_start_method}' ({process_id_suffix}): {consumer_output}"
    )

    assert isinstance(consumer_output, list), (
        f"Consumer failed or returned wrong type for '{mp_start_method}' ({process_id_suffix}): {consumer_output}"
    )
    print(
        f"Consumer returned list of {len(consumer_output)} items ({process_id_suffix})."
    )

    received_items: List[Any] = cast(List[Any], consumer_output)
    assert len(received_items) == len(items_to_send), (
        f"Item count mismatch for '{mp_start_method}' ({process_id_suffix}): sent {len(items_to_send)}, got {len(received_items)}"
    )

    for i, (sent_item, received_item) in enumerate(
        zip(items_to_send, received_items)
    ):
        if (
            _torch_installed
            and torch_module
            and isinstance(sent_item, torch_module.Tensor)
        ):
            assert isinstance(received_item, torch_module.Tensor), (
                f"Item {i} was expected to be a Tensor but received {type(received_item)} for '{mp_start_method}' ({process_id_suffix})"
            )
            assert torch_module.equal(sent_item, received_item), (
                f"Item {i} (Tensor) mismatch for '{mp_start_method}' ({process_id_suffix}): Sent {sent_item}, Got {received_item}"
            )
        elif isinstance(sent_item, dict) and isinstance(received_item, dict):
            assert sent_item == received_item, (
                f"Item {i} (dict) mismatch for '{mp_start_method}' ({process_id_suffix}): Sent {sent_item}, Got {received_item}"
            )
        else:
            # General comparison for other types (e.g. simple strings, numbers, or if one is Tensor and other not)
            assert sent_item == received_item, (
                f"Item {i} mismatch for '{mp_start_method}' ({process_id_suffix}): Sent {sent_item} (type {type(sent_item)}), Got {received_item} (type {type(received_item)})"
            )

    print(f"Joining producer process ({process_id_suffix})...")
    producer_process.join(
        timeout=10
    )  # Should be quick as it already reported success
    print(f"Joining consumer process ({process_id_suffix})...")
    consumer_process.join(timeout=10)

    if producer_process.is_alive():
        producer_process.terminate()
        producer_process.join()
        pytest.fail(
            f"Producer process timed out post-completion for '{mp_start_method}' ({process_id_suffix})."
        )
    if consumer_process.is_alive():
        consumer_process.terminate()
        consumer_process.join()
        pytest.fail(
            f"Consumer process timed out post-completion for '{mp_start_method}' ({process_id_suffix})."
        )

    factory_to_test.shutdown()  # Shutdown the factory created in this function

    if force_torch_unavailable and mocker_fixture:
        # Restore the original state if it was changed
        if original_torch_available_state is not None:
            dqf_module._TORCH_AVAILABLE = original_torch_available_state
        if (
            original_torch_tensor_type_state is not None
        ):  # Should always be true if first is
            dqf_module._torch_tensor_type = original_torch_tensor_type_state
        # It's generally safer to restore via the mocker if that's how it was set for the test's scope
        # but since we patched module-level, direct assignment is how we undo if not using mocker.stopall()
        # For specific object patches, mocker.stopall() or context manager `with patch:` is better.
        # Here, we are controlling the module state for the duration of this specific logic.

    print(
        f"MP test logic completed for '{mp_start_method}' ({process_id_suffix})."
    )


# Determine if 'fork' and 'spawn' are available for MP tests
_mp_methods = (
    multiprocessing.get_all_start_methods()
    if hasattr(multiprocessing, "get_all_start_methods")
    else []
)
if not _mp_methods and multiprocessing.get_start_method(
    allow_none=True
):  # fallback for older python
    _mp_methods = [multiprocessing.get_start_method(allow_none=True)]  # type: ignore

_torch_mp_methods = []
if (
    _torch_installed
    and torch_mp_module
    and hasattr(torch_mp_module, "get_all_start_methods")
):
    _torch_mp_methods = torch_mp_module.get_all_start_methods()

FORK_AVAILABLE = "fork" in _mp_methods or "fork" in _torch_mp_methods
SPAWN_AVAILABLE = "spawn" in _mp_methods or "spawn" in _torch_mp_methods


# --- "fork" start method tests ---
@pytest.mark.skipif(not FORK_AVAILABLE, reason="Fork method not available.")
def test_mp_correctness_fork_default_item_first(
    default_item,
):  # Removed mp_delegating_factory
    items = [default_item, {"data": "data2_fork", "id": 2}]
    _run_mp_test_logic(
        "fork",
        items,  # Pass 'items' positionally
        is_tensor_test=False,
        process_id_suffix="fork_def",
    )


@pytest.mark.skipif(
    not (FORK_AVAILABLE and _torch_installed and torch_module),
    reason="Fork or PyTorch for tensor not available.",
)
def test_mp_correctness_fork_tensor_item_first(
    tensor_item,
):  # Removed mp_delegating_factory
    assert tensor_item is not None, "Tensor item fixture failed for fork test"
    items = [
        tensor_item,
        torch_module.tensor([9.9, 8.8], dtype=torch_module.float32),
    ]  # type: ignore
    _run_mp_test_logic(
        "fork",
        items,  # Pass 'items' positionally
        is_tensor_test=True,
        process_id_suffix="fork_tensor",
    )


@pytest.mark.skipif(
    not (FORK_AVAILABLE and _torch_installed and torch_module),
    reason="Fork or PyTorch for mixed test not available.",
)
def test_mp_correctness_fork_mixed_items_default_first(
    default_item, tensor_item
):  # Removed mp_delegating_factory
    assert tensor_item is not None
    items = [default_item, tensor_item]
    _run_mp_test_logic(
        "fork",
        items,  # Pass 'items' positionally
        is_tensor_test=False,
        process_id_suffix="fork_mix_def",
    )


# --- "spawn" start method tests ---
@pytest.mark.skipif(not SPAWN_AVAILABLE, reason="Spawn method not available.")
def test_mp_correctness_spawn_default_item_first(
    default_item,
):  # Removed mp_delegating_factory
    items = [default_item, {"data": "data_spawn", "id": 3}]
    _run_mp_test_logic(
        "spawn",
        items,  # Pass 'items' positionally
        is_tensor_test=False,
        process_id_suffix="spawn_def",
    )


@pytest.mark.skipif(
    not (SPAWN_AVAILABLE and _torch_installed and torch_module),
    reason="Spawn or PyTorch for tensor not available.",
)
def test_mp_correctness_spawn_tensor_item_first(
    tensor_item,
):  # Removed mp_delegating_factory
    assert tensor_item is not None
    items = [
        tensor_item,
        torch_module.tensor([-1.0, -2.0], dtype=torch_module.float32),
    ]  # type: ignore
    _run_mp_test_logic(
        "spawn",
        items,  # Pass 'items' positionally
        is_tensor_test=True,
        process_id_suffix="spawn_tensor",
    )


@pytest.mark.skipif(
    not (SPAWN_AVAILABLE and _torch_installed and torch_module),
    reason="Spawn or PyTorch for mixed test not available.",
)
def test_mp_correctness_spawn_mixed_items_tensor_first(
    tensor_item, default_item
):  # Removed mp_delegating_factory
    assert tensor_item is not None
    items = [tensor_item, default_item]
    _run_mp_test_logic(
        "spawn",
        items,  # Pass 'items' positionally
        is_tensor_test=True,
        process_id_suffix="spawn_mix_tensor",
    )


def test_factory_behavior_torch_unavailable_e2e(
    mocker: MockerFixture,
    default_item,  # Removed mp_delegating_factory
):
    # This test ensures that even if torch was installed in the environment,
    # if the factory *thinks* torch is unavailable (due to internal _TORCH_AVAILABLE flag),
    # it correctly uses only the default path for an end-to-end MP scenario.

    items_to_send = [default_item, {"type": "another_default", "value": 42}]
    # Determine a start method that is likely to be available for this E2E test
    # Prefer spawn if available, otherwise fork, as spawn is often more stringent.
    current_start_method = "spawn" if SPAWN_AVAILABLE else "fork"
    if (
        not SPAWN_AVAILABLE and not FORK_AVAILABLE
    ):  # Should not happen if tests run
        pytest.skip(
            "No suitable MP start method (spawn or fork) available for E2E test."
        )

    _run_mp_test_logic(
        current_start_method,
        items_to_send,  # Pass 'items_to_send' positionally
        is_tensor_test=False,
        process_id_suffix="notorch_e2e",
        mocker_fixture=mocker,  # Pass the test's mocker
        force_torch_unavailable=True,
    )
    # Old direct test logic removed as _run_mp_test_logic now covers it.
    # The original_module_torch_available restoration is handled by _run_mp_test_logic
    # original_module_torch_tensor_type = dqf_module._torch_tensor_type # This line is unused

    # Mock the module's view of torch availability
    mocker.patch.object(dqf_module, "_TORCH_AVAILABLE", False)
    mocker.patch.object(dqf_module, "_torch_tensor_type", None)

    # We need a new factory instance that is created *while* _TORCH_AVAILABLE is mocked as False.
    # The mp_delegating_factory fixture might have already created its instance.
    # So, we create one manually here for this specific test condition.

    # Mock the factory constructors that DelegatingMultiprocessQueueFactory calls
    # This mocking is now implicitly handled by _run_mp_test_logic if force_torch_unavailable is True
    # We just need to ensure that the internal mocks in _run_mp_test_logic do not interfere with this.
    # The current _run_mp_test_logic structure for force_torch_unavailable uses mocker.patch.object,
    # which is fine.
    pass  # Placeholder as the main logic is moved to _run_mp_test_logic


# --- Minimal Test Case for torch.multiprocessing.Queue with 'spawn' ---


def _minimal_producer_worker(
    tmp_queue: Any,  # Should be torch.multiprocessing.Queue
    tensor_to_send: TensorType,
    result_q: Any,  # Standard ctx.Queue for status
):
    try:
        print(f"[MinimalProducerMODIFIED] Original tensor: {tensor_to_send}")

        # Ensure contiguity and then share memory
        if not tensor_to_send.is_contiguous():
            print(
                "[MinimalProducerMODIFIED] Tensor not contiguous. Making it contiguous."
            )
            tensor_to_send = tensor_to_send.contiguous()

        print("[MinimalProducerMODIFIED] Calling share_memory_() on tensor.")
        tensor_to_send.share_memory_()  # Explicitly share memory
        print(
            f"[MinimalProducerMODIFIED] Putting tensor after share_memory_(): {tensor_to_send}"
        )
        tmp_queue.put(tensor_to_send)
        print("[MinimalProducerMODIFIED] Tensor put successfully.")
        result_q.put("producer_success")
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        print(
            f"[MinimalProducerMODIFIED] EXCEPTION: {e}\n{tb_str}"
        )  # Python f-string will handle newline
        result_q.put(f"producer_exception: {type(e).__name__}: {e}")


def _minimal_consumer_worker(
    tmp_queue: Any,  # Should be torch.multiprocessing.Queue
    result_q: Any,  # Standard ctx.Queue for received item or error
):
    try:
        print("[MinimalConsumer] Getting tensor...")
        received_tensor = tmp_queue.get(timeout=20)
        print(f"[MinimalConsumer] Tensor received: {received_tensor}")
        result_q.put(received_tensor)
    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        print(
            f"[MinimalConsumer] EXCEPTION: {e}\n{tb_str}"
        )  # Python f-string will handle newline
        result_q.put(f"consumer_exception: {type(e).__name__}: {e}")


@pytest.mark.skipif(
    not (
        SPAWN_AVAILABLE
        and _torch_installed
        and torch_module
        and torch_mp_module
    ),
    reason="Spawn or PyTorch not available for minimal MP Queue test.",
)
def test_minimal_torch_mp_queue_spawn_tensor(tensor_item: TensorType):
    """Tests torch.multiprocessing.Queue directly with 'spawn' and a tensor."""
    assert tensor_item is not None, "Tensor item fixture failed"
    assert torch_mp_module is not None, "torch_mp_module not available"

    print("\n--- Starting test_minimal_torch_mp_queue_spawn_tensor ---")

    # Crucially, set start method to spawn BEFORE getting context or creating queues.
    set_torch_mp_start_method_if_needed("spawn", force=True)

    # Also explicitly set sharing strategy to file_system as it's often more robust
    try:
        current_strategy = torch_mp_module.get_sharing_strategy()
        if current_strategy != "file_system":
            torch_mp_module.set_sharing_strategy("file_system")
            print(
                f"INFO [MinimalTest]: Set PyTorch sharing strategy to 'file_system'. Was: {current_strategy}"
            )
        else:
            print(
                "INFO [MinimalTest]: PyTorch sharing strategy already 'file_system'."
            )
    except Exception as e:  # Broad except for safety in test setup
        print(
            f"WARNING [MinimalTest]: Could not set PyTorch sharing strategy to 'file_system': {e}"
        )

    ctx = torch_mp_module.get_context(
        "spawn"
    )  # Ensure spawn context from torch

    torch_specific_queue = ctx.Queue()
    producer_status_q = ctx.Queue()
    consumer_data_q = ctx.Queue()

    sent_tensor = tensor_item  # Use the fixture

    producer_proc = ctx.Process(
        target=_minimal_producer_worker,
        args=(torch_specific_queue, sent_tensor, producer_status_q),
    )
    consumer_proc = ctx.Process(
        target=_minimal_consumer_worker,
        args=(torch_specific_queue, consumer_data_q),
    )

    print("[MinimalTest] Starting producer...")
    producer_proc.start()
    print("[MinimalTest] Starting consumer...")
    consumer_proc.start()

    producer_timeout = 30
    consumer_timeout = 30

    print("[MinimalTest] Waiting for producer status...")
    producer_status = producer_status_q.get(timeout=producer_timeout)
    assert producer_status == "producer_success", (
        f"Minimal producer failed: {producer_status}"
    )
    print(f"[MinimalTest] Producer status: {producer_status}")

    print("[MinimalTest] Waiting for consumer data...")
    received_data = consumer_data_q.get(timeout=consumer_timeout)

    assert isinstance(received_data, torch_module.Tensor), (
        f"Minimal consumer received non-tensor data: {type(received_data)}, content: {received_data}"
    )
    print(f"[MinimalTest] Consumer received tensor: {received_data}")

    assert torch_module.equal(sent_tensor, received_data), (
        f"Minimal test tensor mismatch: Sent {sent_tensor}, Got {received_data}"
    )

    print("[MinimalTest] Joining producer...")
    producer_proc.join(timeout=10)
    print("[MinimalTest] Joining consumer...")
    consumer_proc.join(timeout=10)

    if producer_proc.is_alive():
        producer_proc.terminate()
        pytest.fail("Minimal producer process timed out post-completion.")
    if consumer_proc.is_alive():
        consumer_proc.terminate()
        pytest.fail("Minimal consumer process timed out post-completion.")

    print("--- Finished test_minimal_torch_mp_queue_spawn_tensor ---")
