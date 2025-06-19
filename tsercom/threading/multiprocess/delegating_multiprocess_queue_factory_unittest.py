"""Unit tests for DelegatingMultiprocessQueueFactory and its components."""

import pytest
from pytest_mock import MockerFixture
import multiprocessing
import time
from typing import Any, Optional, cast, TypeAlias
from unittest.mock import MagicMock
import types
import queue

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

    class Tensor:
        pass

    if not torch_module:
        torch_mock = MockerFixture(None).MagicMock()
        torch = torch_mock
        if hasattr(torch, "Tensor"):
            torch.Tensor = Tensor
    if not torch_mp_module:
        torch_mp_mock = MockerFixture(None).MagicMock()
        torch_mp = torch_mp_mock

if _torch_installed and _real_torch_imported_module:
    TensorType: TypeAlias = _real_torch_imported_module.Tensor
else:
    TensorType: TypeAlias = Any


def set_torch_mp_start_method_if_needed(method: str = "spawn") -> None:
    if not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_start_method")
        and hasattr(torch_mp_module, "set_start_method")
    ):
        if method == "spawn" and not torch_mp_module:
            try:
                multiprocessing.set_start_method(method, force=True)
            except RuntimeError as e:
                current_method_after_error = multiprocessing.get_start_method(
                    allow_none=True
                )
                if current_method_after_error != method and (
                    "context has already been set" in str(e)
                    or "cannot start a process after starting a new process"
                    in str(e)
                ):
                    raise RuntimeError(
                        f"Standard MP context is '{current_method_after_error}', cannot change to '{method}'. Test requires '{method}'."
                    ) from e
        return
    try:
        current_method = torch_mp_module.get_start_method(allow_none=True)
        if current_method == method:
            return
        torch_mp_module.set_start_method(method, force=True)
    except RuntimeError as e:
        current_method_after_error = torch_mp_module.get_start_method(
            allow_none=True
        )
        err_msg = str(e).lower()
        if current_method_after_error == method:
            return
        if (
            "context has already been set" in err_msg
            or "cannot start a process after starting a new process" in err_msg
        ) and current_method_after_error != method:
            raise RuntimeError(
                f"Torch MP context is '{current_method_after_error}' and cannot be changed to '{method}'. Test requires '{method}'. Original error: {e}"
            ) from e
        raise
    except Exception:
        raise


@pytest.fixture
def mock_default_sink(mocker: MockerFixture) -> MagicMock:
    mock = mocker.MagicMock(spec=MultiprocessQueueSink)
    mock.max_queue_size = 10
    return mock


@pytest.fixture
def mock_torch_sink(mocker: MockerFixture) -> MagicMock:
    mock = mocker.MagicMock(spec=MultiprocessQueueSink)
    mock.max_queue_size = 10
    return mock


@pytest.fixture
def mock_default_source(mocker: MockerFixture) -> MagicMock:
    mock = mocker.MagicMock(spec=MultiprocessQueueSource)
    mock.max_queue_size = 10
    return mock


@pytest.fixture
def mock_torch_source(mocker: MockerFixture) -> MagicMock:
    mock = mocker.MagicMock(spec=MultiprocessQueueSource)
    mock.max_queue_size = 10
    return mock


@pytest.fixture
def delegating_mp_factory() -> (
    dqf_module.DelegatingMultiprocessQueueFactory[Any]
):
    factory = dqf_module.DelegatingMultiprocessQueueFactory[Any]()
    yield factory


@pytest.fixture
def delegating_sink(
    mocker: MockerFixture,
    mock_default_sink: MagicMock,
    request: pytest.FixtureRequest,
) -> DelegatingMultiprocessQueueSink[Any]:
    mock_torch_sink_opt = request.param if hasattr(request, "param") else None
    return DelegatingMultiprocessQueueSink[Any](
        default_queue_sink=mock_default_sink,
        torch_queue_sink=mock_torch_sink_opt,
    )


delegating_sink_parametrized = pytest.mark.parametrize(
    "delegating_sink",
    [None, pytest.lazy_fixture("mock_torch_sink")],
    indirect=True,
    ids=["no_torch_sink", "with_torch_sink"],
)


@pytest.fixture
def delegating_source(
    mocker: MockerFixture,
    mock_default_source: MagicMock,
    request: pytest.FixtureRequest,
) -> DelegatingMultiprocessQueueSource[Any]:
    mock_torch_source_opt = (
        request.param if hasattr(request, "param") else None
    )
    return DelegatingMultiprocessQueueSource[Any](
        default_queue_source=mock_default_source,
        torch_queue_source=mock_torch_source_opt,
    )


delegating_source_parametrized = pytest.mark.parametrize(
    "delegating_source",
    [None, pytest.lazy_fixture("mock_torch_source")],
    indirect=True,
    ids=["no_torch_source", "with_torch_source"],
)


@pytest.fixture
def test_item() -> Any:
    return "test_data"


@pytest.fixture
def tensor_item(mocker: MockerFixture) -> Optional[TensorType]:
    if _torch_installed and torch_module:
        return torch_module.tensor([1.0, 2.0])
    return cast(
        TensorType,
        mocker.MagicMock(spec=TensorType if TensorType is not Any else None),
    )


class TestDelegatingMultiprocessQueueFactory:
    def test_factory_init(self) -> None:
        factory = dqf_module.DelegatingMultiprocessQueueFactory[Any]()
        assert factory.max_queue_size == dqf_module.DEFAULT_MAX_QUEUE_SIZE
        factory_custom = dqf_module.DelegatingMultiprocessQueueFactory[Any](
            max_queue_size=50
        )
        assert factory_custom.max_queue_size == 50

    def test_create_queues_returns_delegating_sink_and_source(
        self,
        mocker: MockerFixture,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
    ) -> None:
        mock_default_factory_cls = mocker.patch(
            f"{dqf_module.__name__}.DefaultMultiprocessQueueFactory"
        )
        mock_torch_factory_cls = mocker.patch(
            f"{dqf_module.__name__}.TorchMultiprocessQueueFactory"
        )
        mock_std_manager_cls = mocker.patch(
            f"{dqf_module.__name__}.multiprocessing.Manager"
        )
        mock_torch_mp_module_patch = mocker.patch(
            f"{dqf_module.__name__}._torch_mp_module"
        )
        if _torch_installed and torch_mp_module:
            mock_torch_mp_module_patch.Manager = mocker.MagicMock()

        mock_dsi, mock_dso = mocker.MagicMock(
            spec=MultiprocessQueueSink
        ), mocker.MagicMock(spec=MultiprocessQueueSource)
        mock_tsi, mock_tso = mocker.MagicMock(
            spec=MultiprocessQueueSink
        ), mocker.MagicMock(spec=MultiprocessQueueSource)
        mock_default_factory_cls.return_value.create_queues.return_value = (
            mock_dsi,
            mock_dso,
        )
        mock_torch_factory_cls.return_value.create_queues.return_value = (
            mock_tsi,
            mock_tso,
        )
        mock_std_manager_inst = mock_std_manager_cls.return_value
        mock_torch_manager_inst = (
            mock_torch_mp_module_patch.Manager.return_value
            if hasattr(mock_torch_mp_module_patch, "Manager")
            else mocker.MagicMock()
        )

        delegating_mp_factory.max_queue_size = 30
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            sink, source = delegating_mp_factory.create_queues()
            mock_std_manager_cls.assert_called_once()
            mock_default_factory_cls.assert_called_once_with(
                max_queue_size=30, manager=mock_std_manager_inst
            )
            mock_default_factory_cls.return_value.create_queues.assert_called_once()
            if hasattr(mock_torch_mp_module_patch, "Manager"):
                mock_torch_mp_module_patch.Manager.assert_called_once()
                mock_torch_factory_cls.assert_called_once_with(
                    max_queue_size=30, manager=mock_torch_manager_inst
                )
                mock_torch_factory_cls.return_value.create_queues.assert_called_once()
            assert isinstance(
                sink, DelegatingMultiprocessQueueSink
            ) and isinstance(source, DelegatingMultiprocessQueueSource)
            assert sink._DelegatingMultiprocessQueueSink__default_queue_sink is mock_dsi  # type: ignore [attr-defined]
            assert sink._DelegatingMultiprocessQueueSink__torch_queue_sink is mock_tsi  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__default_queue_source is mock_dso  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__torch_queue_source is mock_tso  # type: ignore [attr-defined]

        mock_std_manager_cls.reset_mock()
        mock_default_factory_cls.reset_mock()
        mock_default_factory_cls.return_value.create_queues.reset_mock()
        if hasattr(mock_torch_mp_module_patch, "Manager"):
            mock_torch_mp_module_patch.Manager.reset_mock()
        mock_torch_factory_cls.reset_mock()
        mock_torch_factory_cls.return_value.create_queues.reset_mock()

        delegating_mp_factory.max_queue_size = 20
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=False
        ):
            sink, source = delegating_mp_factory.create_queues()
            mock_std_manager_cls.assert_called_once()
            mock_default_factory_cls.assert_called_once_with(
                max_queue_size=20, manager=mock_std_manager_inst
            )
            mock_default_factory_cls.return_value.create_queues.assert_called_once()
            if hasattr(mock_torch_mp_module_patch, "Manager"):
                mock_torch_mp_module_patch.Manager.assert_not_called()
            mock_torch_factory_cls.assert_not_called()
            assert isinstance(
                sink, DelegatingMultiprocessQueueSink
            ) and isinstance(source, DelegatingMultiprocessQueueSource)
            assert sink._DelegatingMultiprocessQueueSink__default_queue_sink is mock_dsi  # type: ignore [attr-defined]
            assert sink._DelegatingMultiprocessQueueSink__torch_queue_sink is None  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__default_queue_source is mock_dso  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__torch_queue_source is None  # type: ignore [attr-defined]


@delegating_sink_parametrized
class TestDelegatingMultiprocessQueueSink:
    def test_sink_init(
        self,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
    ) -> None:
        assert delegating_sink._DelegatingMultiprocessQueueSink__default_queue_sink is mock_default_sink  # type: ignore [attr-defined]
        if delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink is not None:  # type: ignore [attr-defined]
            assert isinstance(delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink, MagicMock)  # type: ignore [attr-defined]
        else:
            assert delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink is None  # type: ignore [attr-defined]
        assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is None  # type: ignore [attr-defined]
        assert not delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]
        assert not delegating_sink.closed

    @pytest.mark.skipif(not _torch_installed, reason="PyTorch not installed.")
    def test_sink_put_first_item_uses_torch_path(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
        tensor_item: TensorType,
    ) -> None:
        if (
            delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink
            is None
        ):
            pytest.skip("Torch sink not provided.")  # type: ignore [attr-defined]
        mock_torch_sink_ref = delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink  # type: ignore [attr-defined]
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            if not (
                _torch_installed
                and torch_module
                and isinstance(tensor_item, torch_module.Tensor)
            ):
                mocker.patch("isinstance", lambda obj, type_info: True if type_info is torch_module.Tensor else isinstance(obj, type_info))  # type: ignore [undefined-variable]
            mock_default_sink.put_blocking.return_value = True
            mock_torch_sink_ref.put_blocking.return_value = True  # type: ignore
            delegating_sink.put_blocking(tensor_item, timeout=1.0)
            mock_default_sink.put_blocking.assert_called_once_with(
                "USE_TORCH", 1.0
            )
            mock_torch_sink_ref.put_blocking.assert_called_once_with(tensor_item, timeout=1.0)  # type: ignore
            assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is mock_torch_sink_ref  # type: ignore [attr-defined]
            assert delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]

    def test_sink_put_first_item_uses_default_path_non_tensor(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
        test_item: Any,
    ) -> None:
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            mock_default_sink.put_blocking.return_value = True
            if (
                delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink
            ):
                delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink.put_blocking.assert_not_called()  # type: ignore [attr-defined]
            delegating_sink.put_blocking(test_item, timeout=1.0)
            mock_default_sink.put_blocking.assert_has_calls(
                [
                    mocker.call("USE_DEFAULT", 1.0),
                    mocker.call(test_item, timeout=1.0),
                ]
            )
            assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is mock_default_sink  # type: ignore [attr-defined]
            assert delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]

    def test_sink_put_first_item_uses_default_path_torch_unavailable(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
        tensor_item: Optional[TensorType],
        test_item: Any,
    ) -> None:
        item_to_use = tensor_item if tensor_item is not None else test_item
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=False
        ):
            mock_default_sink.put_blocking.return_value = True
            if (
                delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink
            ):
                delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink.put_blocking.assert_not_called()  # type: ignore [attr-defined]
            delegating_sink.put_blocking(item_to_use, timeout=1.0)
            mock_default_sink.put_blocking.assert_has_calls(
                [
                    mocker.call("USE_DEFAULT", 1.0),
                    mocker.call(item_to_use, timeout=1.0),
                ]
            )
            assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is mock_default_sink  # type: ignore [attr-defined]
            assert delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]

    @pytest.mark.skipif(not _torch_installed, reason="PyTorch not available.")
    def test_sink_put_subsequent_item(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
        tensor_item: TensorType,
        test_item: Any,
    ) -> None:
        if (
            delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink
            is None
        ):
            pytest.skip("Torch sink not provided.")  # type: ignore [attr-defined]
        mock_torch_sink_ref = delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink  # type: ignore [attr-defined]
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            if not (
                _torch_installed
                and torch_module
                and isinstance(tensor_item, torch_module.Tensor)
            ):
                mocker.patch("isinstance", lambda obj, type_info: True if type_info is torch_module.Tensor and obj is tensor_item else isinstance(obj, type_info))  # type: ignore [undefined-variable]
            mock_default_sink.put_blocking.return_value = True
            mock_torch_sink_ref.put_blocking.return_value = True  # type: ignore
            delegating_sink.put_blocking(tensor_item)
            mock_default_sink.put_blocking.reset_mock()
            mock_torch_sink_ref.put_blocking.reset_mock()  # type: ignore
            delegating_sink.put_blocking(test_item, timeout=0.5)
            mock_default_sink.put_blocking.assert_not_called()
            mock_torch_sink_ref.put_blocking.assert_called_once_with(test_item, timeout=0.5)  # type: ignore

    def test_put_when_closed_raises_runtime_error(
        self,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        test_item: Any,
    ) -> None:
        delegating_sink.close()
        with pytest.raises(
            RuntimeError,
            match="Cannot put item on a closed DelegatingMultiprocessQueueSink.",
        ):
            delegating_sink.put_blocking(test_item)
        with pytest.raises(
            RuntimeError,
            match="Cannot put item on a closed DelegatingMultiprocessQueueSink.",
        ):
            delegating_sink.put_nowait(test_item)

    def test_sink_close_closes_underlying_sinks(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
    ) -> None:
        mock_handler = mocker.MagicMock()
        delegating_sink.on_close += mock_handler
        delegating_sink.close()
        mock_default_sink.close.assert_called_once()
        if delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink:
            delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink.close.assert_called_once()  # type: ignore [attr-defined]
        mock_handler.assert_called_once()
        assert delegating_sink.closed


@delegating_source_parametrized
class TestDelegatingMultiprocessQueueSource:
    def test_source_init(
        self,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
    ) -> None:
        assert delegating_source._DelegatingMultiprocessQueueSource__default_queue_source is mock_default_source  # type: ignore [attr-defined]
        if delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source is not None:  # type: ignore [attr-defined]
            assert isinstance(delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source, MagicMock)  # type: ignore [attr-defined]
        else:
            assert delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source is None  # type: ignore [attr-defined]
        assert delegating_source._DelegatingMultiprocessQueueSource__selected_queue_source is None  # type: ignore [attr-defined]
        assert not delegating_source.closed

    def test_source_get_first_item_selects_torch_path(
        self,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
        test_item: Any,
    ) -> None:
        if (
            delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source
            is None
        ):
            pytest.skip("Torch source not provided.")  # type: ignore [attr-defined]
        mock_torch_source_ref = delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source  # type: ignore [attr-defined]
        mock_default_source.get_blocking.return_value = "USE_TORCH"
        mock_torch_source_ref.get_blocking.return_value = test_item  # type: ignore
        item = delegating_source.get_blocking(timeout=1.0)
        mock_default_source.get_blocking.assert_called_once_with(timeout=1.0)
        mock_torch_source_ref.get_blocking.assert_called_once_with(timeout=1.0)  # type: ignore
        assert item == test_item
        assert delegating_source._DelegatingMultiprocessQueueSource__selected_queue_source is mock_torch_source_ref  # type: ignore [attr-defined]

    def test_source_get_first_item_selects_default_path(
        self,
        mocker: MockerFixture,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
        test_item: Any,
    ) -> None:
        mock_default_source.get_blocking.side_effect = [
            "USE_DEFAULT",
            test_item,
        ]
        item = delegating_source.get_blocking(timeout=1.0)
        mock_default_source.get_blocking.assert_has_calls(
            [mocker.call(timeout=1.0), mocker.call(timeout=1.0)]
        )
        assert item == test_item
        assert delegating_source._DelegatingMultiprocessQueueSource__selected_queue_source is mock_default_source  # type: ignore [attr-defined]
        if (
            delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source
        ):
            delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source.get_blocking.assert_not_called()  # type: ignore [attr-defined]

    def test_source_get_subsequent_item(
        self,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
        test_item: Any,
    ) -> None:
        mock_default_source.get_blocking.side_effect = [
            "USE_DEFAULT",
            "first_item",
            test_item,
        ]
        delegating_source.get_blocking(timeout=1.0)  # First get
        item_2 = delegating_source.get_blocking(timeout=0.5)
        assert mock_default_source.get_blocking.call_count == 3
        assert item_2 == test_item

    def test_source_close_closes_underlying_sources(
        self,
        mocker: MockerFixture,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
    ) -> None:
        mock_handler = mocker.MagicMock()
        delegating_source.on_close += mock_handler
        delegating_source.close()
        mock_default_source.close.assert_called_once()
        if (
            delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source
        ):
            delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source.close.assert_called_once()  # type: ignore [attr-defined]
        mock_handler.assert_called_once()
        assert delegating_source.closed


# --- Multiprocessing Tests ---
# Worker functions are defined globally
def _producer_worker_default_mp(
    sink_queue: DelegatingMultiprocessQueueSink[Any],
    item_to_send: Any,
    result_queue: Any,
):
    try:
        if not sink_queue.put_blocking(item_to_send, timeout=10):
            result_queue.put("put_failed_timeout_or_full")
            return
        result_queue.put("put_successful")
    except Exception as e:
        import traceback

        result_queue.put(
            f"producer_exception: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _consumer_worker_default_mp(
    source_queue: DelegatingMultiprocessQueueSource[Any], result_queue: Any
):
    try:
        item = source_queue.get_blocking(timeout=15)
        result_queue.put(item)
    except queue.Empty:
        result_queue.put("get_failed_timeout_empty")
    except Exception as e:
        import traceback

        result_queue.put(
            f"consumer_exception: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _producer_worker_tensor_mp(
    sink_queue: DelegatingMultiprocessQueueSink[Any],
    item_to_send: Any,
    result_queue: Any,
):
    try:
        if not sink_queue.put_blocking(item_to_send, timeout=10):
            result_queue.put("put_failed_timeout_or_full")
            return
        result_queue.put("put_successful")
    except Exception as e:
        import traceback

        result_queue.put(
            f"producer_exception: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _consumer_worker_tensor_mp(
    source_queue: DelegatingMultiprocessQueueSource[Any], result_queue: Any
):
    try:
        item = source_queue.get_blocking(timeout=15)
        result_queue.put(item)
    except queue.Empty:
        result_queue.put("get_failed_timeout_empty")
    except Exception as e:
        import traceback

        result_queue.put(
            f"consumer_exception: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _execute_mp_correctness_logic(
    start_method_to_try: str,
    factory_fixture: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    item_to_send: Any,
    is_tensor_test: bool = False,
) -> None:
    mp_context_to_use = multiprocessing
    if (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_context")
    ):
        set_torch_mp_start_method_if_needed(method=start_method_to_try)
        mp_context_to_use = torch_mp_module.get_context()

    producer_status_queue: Any = mp_context_to_use.Queue()
    consumer_data_queue: Any = mp_context_to_use.Queue()
    sink_q, source_q = factory_fixture.create_queues()

    worker_producer = (
        _producer_worker_tensor_mp
        if is_tensor_test
        else _producer_worker_default_mp
    )
    worker_consumer = (
        _consumer_worker_tensor_mp
        if is_tensor_test
        else _consumer_worker_default_mp
    )

    producer_process = mp_context_to_use.Process(
        target=worker_producer,
        args=(sink_q, item_to_send, producer_status_queue),
    )
    consumer_process = mp_context_to_use.Process(
        target=worker_consumer, args=(source_q, consumer_data_queue)
    )

    try:
        producer_process.start()
        consumer_process.start()
        producer_process.join(timeout=25)
        consumer_process.join(timeout=25)
        assert not producer_process.is_alive(), "Producer process timed out."
        assert not consumer_process.is_alive(), "Consumer process timed out."
        assert (
            producer_process.exitcode == 0
        ), f"Producer exited with code {producer_process.exitcode}"
        assert (
            consumer_process.exitcode == 0
        ), f"Consumer exited with code {consumer_process.exitcode}"

        producer_status = producer_status_queue.get(timeout=5)
        assert (
            producer_status == "put_successful"
        ), f"Producer status: {producer_status}"
        received_item = consumer_data_queue.get(timeout=5)
        if (
            isinstance(received_item, str)
            and "exception" in received_item.lower()
        ):
            pytest.fail(f"Consumer exception: {received_item}")

        if is_tensor_test:
            assert torch_module is not None and isinstance(
                received_item, torch_module.Tensor
            )
            assert torch_module.equal(received_item, item_to_send)
        else:  # Assuming default items are dicts as per original worker logic
            assert isinstance(received_item, dict)
            assert received_item == item_to_send
    finally:
        if hasattr(sink_q, "close"):
            sink_q.close()
        if hasattr(source_q, "close"):
            source_q.close()
        if hasattr(producer_status_queue, "close"):
            producer_status_queue.close()
        if hasattr(consumer_data_queue, "close"):
            consumer_data_queue.close()


@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "fork" in torch_mp_module.get_all_start_methods()
    ),
    reason="Fork/Torch MP not fully available.",
)
class TestMultiprocessCorrectnessFork:
    def test_multiprocess_correctness_fork_default_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
    ) -> None:
        item_to_send = {
            "data": "test_value_fork",
            "id": 456,
            "timestamp": time.time(),
        }
        _execute_mp_correctness_logic(
            "fork", delegating_mp_factory, item_to_send, is_tensor_test=False
        )

    def test_multiprocess_correctness_fork_tensor_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
    ) -> None:
        assert torch_module is not None, "Torch module required."
        item_to_send = torch_module.tensor([1.0, 2.0, 3.0])
        _execute_mp_correctness_logic(
            "fork", delegating_mp_factory, item_to_send, is_tensor_test=True
        )


@pytest.mark.skipif(
    not (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_all_start_methods")
        and "spawn" in torch_mp_module.get_all_start_methods()
    ),
    reason="Spawn/Torch MP not fully available.",
)
@pytest.mark.xfail(
    raises=RuntimeError,
    strict=True,
    reason="Pickling/context issues with spawn expected.",
)
class TestMultiprocessCorrectnessSpawn:
    def test_multiprocess_correctness_spawn_default_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
    ) -> None:
        item_to_send = {
            "data": "test_value_spawn",
            "id": 789,
            "timestamp": time.time(),
        }
        _execute_mp_correctness_logic(
            "spawn", delegating_mp_factory, item_to_send, is_tensor_test=False
        )

    def test_multiprocess_correctness_spawn_tensor_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
    ) -> None:
        assert torch_module is not None, "Torch module required."
        item_to_send = torch_module.tensor([4.0, 5.0, 6.0])
        _execute_mp_correctness_logic(
            "spawn", delegating_mp_factory, item_to_send, is_tensor_test=True
        )
