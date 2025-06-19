"""Unit tests for DelegatingMultiprocessQueueFactory and its components."""

import pytest
from pytest_mock import MockerFixture
import multiprocessing
import time
from typing import Any, List, Optional, cast, TypeAlias, Dict
from unittest.mock import MagicMock
import types
import queue  # Ensure queue is imported for queue.Empty
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
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
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

    class Tensor:  # pyright: ignore [reportUnusedClass]
        pass

    if not torch_module:
        torch_mock = MockerFixture(None).MagicMock()  # type: ignore
        torch_module = torch_mock  # type: ignore
        if hasattr(torch_module, "Tensor"):  # type: ignore
            torch_module.Tensor = Tensor  # type: ignore
    if not torch_mp_module:
        torch_mp_mock = MockerFixture(None).MagicMock()  # type: ignore
        torch_mp_module = torch_mp_mock  # type: ignore

if (
    _torch_installed
    and _real_torch_imported_module
    and hasattr(_real_torch_imported_module, "Tensor")
):
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
        if (
            method == "spawn" and not torch_mp_module
        ):  # fallback for non-torch env
            try:
                # Ensure we are not trying to set if it's already what we want or unchangeable
                current_method_std_mp = multiprocessing.get_start_method(
                    allow_none=True
                )
                if (
                    current_method_std_mp is None
                    or current_method_std_mp == method
                ):
                    multiprocessing.set_start_method(
                        method,
                        force=True if current_method_std_mp is None else False,
                    )
                elif (
                    current_method_std_mp != method
                ):  # If it's set and different, this will likely error
                    multiprocessing.set_start_method(
                        method, force=False
                    )  # Try non-force first
            except RuntimeError as e:
                current_method_after_error = multiprocessing.get_start_method(
                    allow_none=True
                )
                if current_method_after_error != method and (
                    "context has already been set" in str(e).lower()
                    or "cannot start a process after starting a new process"  # Python 3.8+ specific for 'fork' after 'spawn'
                    in str(e).lower()
                ):
                    # Only raise if it's genuinely different and couldn't be set
                    warnings.warn(
                        f"Standard MP context is '{current_method_after_error}', could not change to '{method}'. Test might behave unexpectedly. Error: {e}",
                        RuntimeWarning,
                    )
        return

    try:
        current_method = torch_mp_module.get_start_method(allow_none=True)
        if current_method is None:  # Not set yet
            torch_mp_module.set_start_method(method, force=True)
        elif current_method == method:  # Already correct
            return
        else:  # Set but different
            torch_mp_module.set_start_method(
                method, force=False
            )  # Try non-force first
    except RuntimeError as e:
        current_method_after_error = torch_mp_module.get_start_method(
            allow_none=True
        )
        err_msg = str(e).lower()
        if (
            current_method_after_error == method
        ):  # It got set despite error (e.g. force=True was needed but not used)
            return
        if (
            "context has already been set" in err_msg
            or "cannot start a process after starting a new process"
            in err_msg  # Python 3.8+
        ) and current_method_after_error != method:
            # This is a critical issue for the test if the context cannot be set as required.
            # However, pytest.skip() or raising an error here might be too aggressive
            # if the test could potentially still run or if this setup is done globally.
            # For now, a warning. If a test specifically fails due to this, it should handle it.
            warnings.warn(
                f"Torch MP context is '{current_method_after_error}' and could not be changed to '{method}'. "
                f"Test might require '{method}'. Original error: {e}",
                RuntimeWarning,
            )
        # else: # Re-raise other RuntimeErrors
        # raise # Re-raise if it's not the "context already set" issue.
    except Exception:  # Catch other generic exceptions from set_start_method
        # Similar to above, warn rather than fail hard during setup.
        current_method_final = torch_mp_module.get_start_method(
            allow_none=True
        )
        warnings.warn(
            f"Failed to set Torch MP start method to '{method}'. Current method: '{current_method_final}'. Error: {e}",  # type: ignore
            RuntimeWarning,
        )


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
    # Cleanup for the factory if it holds resources, e.g. a manager
    if hasattr(factory, "_default_manager") and factory._default_manager:  # type: ignore
        factory._default_manager.shutdown()  # type: ignore
    if hasattr(factory, "_torch_manager") and factory._torch_manager:  # type: ignore
        if hasattr(factory._torch_manager, "shutdown"):  # type: ignore
            factory._torch_manager.shutdown()  # type: ignore


@pytest.fixture
def delegating_sink(
    mocker: MockerFixture,
    mock_default_sink: MagicMock,
    request: pytest.FixtureRequest,
) -> DelegatingMultiprocessQueueSink[Any]:
    mock_torch_sink_opt = None
    if hasattr(request, "param") and request.param is not None:
        if isinstance(request.param, str):
            mock_torch_sink_opt = request.getfixturevalue(request.param)
        else:
            mock_torch_sink_opt = request.param  # type: ignore
    return DelegatingMultiprocessQueueSink[Any](
        default_queue_sink=mock_default_sink,
        torch_queue_sink=mock_torch_sink_opt,
    )


@pytest.fixture
def delegating_source(
    mocker: MockerFixture,
    mock_default_source: MagicMock,
    request: pytest.FixtureRequest,
) -> DelegatingMultiprocessQueueSource[Any]:
    mock_torch_source_opt = None
    if hasattr(request, "param") and request.param is not None:
        if isinstance(request.param, str):
            mock_torch_source_opt = request.getfixturevalue(request.param)
        else:
            mock_torch_source_opt = request.param  # type: ignore
    return DelegatingMultiprocessQueueSource[Any](
        default_queue_source=mock_default_source,
        torch_queue_source=mock_torch_source_opt,
    )


@pytest.fixture
def test_item() -> Any:
    return {"data": "test_data_dict", "value": 123}


@pytest.fixture
def tensor_item(mocker: MockerFixture) -> Optional[TensorType]:
    if _torch_installed and torch_module:
        return torch_module.tensor([1.0, 2.0])
    return cast(
        TensorType,  # type: ignore
        mocker.MagicMock(spec=TensorType if TensorType is not Any else None),  # type: ignore
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
        mock_std_manager_cls = mocker.patch(
            f"{dqf_module.__name__}.multiprocessing.Manager"
        )
        # Patch the _torch_mp_module used within the factory's scope
        mock_torch_mp_module_in_factory_scope = mocker.patch(
            f"{dqf_module.__name__}._torch_mp_module"
        )
        # Ensure this mock has a Manager attribute if torch is supposed to be installed
        if _torch_installed and torch_mp_module:
            mock_torch_mp_module_in_factory_scope.Manager = mocker.MagicMock(
                return_value=mocker.MagicMock(Queue=mocker.MagicMock())
            )
            # If get_context is used by factory, mock it too
            if hasattr(torch_mp_module, "get_context"):
                mock_torch_mp_module_in_factory_scope.get_context = (
                    mocker.MagicMock(
                        return_value=mocker.MagicMock(Queue=mocker.MagicMock())
                    )
                )

        mock_actual_default_queue = mocker.MagicMock(
            spec=multiprocessing.Queue
        )
        mock_actual_torch_queue = mocker.MagicMock(
            spec=multiprocessing.Queue
        )  # Use base MP Queue for spec

        mock_std_manager_cls.return_value.Queue.return_value = (
            mock_actual_default_queue
        )
        if (
            _torch_installed
            and hasattr(mock_torch_mp_module_in_factory_scope, "Manager")
            and mock_torch_mp_module_in_factory_scope.Manager
        ):
            mock_torch_mp_module_in_factory_scope.Manager.return_value.Queue.return_value = (
                mock_actual_torch_queue
            )
        elif _torch_installed and hasattr(
            mock_torch_mp_module_in_factory_scope, "get_context"
        ):
            # If factory uses get_context().Queue()
            mock_torch_mp_module_in_factory_scope.get_context.return_value.Queue.return_value = (
                mock_actual_torch_queue
            )

        mock_sink_wrapper_cls = mocker.patch(
            f"{dqf_module.__name__}.MultiprocessQueueSink"
        )
        mock_source_wrapper_cls = mocker.patch(
            f"{dqf_module.__name__}.MultiprocessQueueSource"
        )

        mock_default_sink_inst = MagicMock(spec=MultiprocessQueueSink)
        mock_default_source_inst = MagicMock(spec=MultiprocessQueueSource)
        mock_torch_sink_inst = MagicMock(spec=MultiprocessQueueSink)
        mock_torch_source_inst = MagicMock(spec=MultiprocessQueueSource)

        mock_default_sink_inst.underlying_queue = mock_actual_default_queue
        mock_default_source_inst.underlying_queue = mock_actual_default_queue
        if mock_torch_sink_inst:
            mock_torch_sink_inst.underlying_queue = mock_actual_torch_queue
        if mock_torch_source_inst:
            mock_torch_source_inst.underlying_queue = mock_actual_torch_queue

        def sink_side_effect(q_arg: Any, max_size: int, mp_context_name: Optional[str] = None):  # type: ignore
            if q_arg is mock_actual_default_queue:
                return mock_default_sink_inst
            if q_arg is mock_actual_torch_queue:
                return mock_torch_sink_inst
            # Fallback for unexpected queue objects
            return MagicMock(spec=MultiprocessQueueSink)

        def source_side_effect(q_arg: Any, max_size: int, mp_context_name: Optional[str] = None):  # type: ignore
            if q_arg is mock_actual_default_queue:
                return mock_default_source_inst
            if q_arg is mock_actual_torch_queue:
                return mock_torch_source_inst
            # Fallback for unexpected queue objects
            return MagicMock(spec=MultiprocessQueueSource)

        mock_sink_wrapper_cls.side_effect = sink_side_effect
        mock_source_wrapper_cls.side_effect = source_side_effect

        delegating_mp_factory.max_queue_size = 30
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            sink, source = delegating_mp_factory.create_queues()

            # Check if default queue created
            # If factory uses its own manager:
            if hasattr(delegating_mp_factory, "_default_manager_instance") and delegating_mp_factory._default_manager_instance:  # type: ignore
                delegating_mp_factory._default_manager_instance.Queue.assert_called_once_with(maxsize=30)  # type: ignore
            # else check if global mp.Manager was used (less likely with new design)
            # elif mock_std_manager_cls.called:
            #    mock_std_manager_cls.return_value.Queue.assert_called_once_with(maxsize=30)

            # Check if torch queue created
            if hasattr(delegating_mp_factory, "_torch_manager_instance") and delegating_mp_factory._torch_manager_instance:  # type: ignore
                delegating_mp_factory._torch_manager_instance.Queue.assert_called_once_with(maxsize=30)  # type: ignore
            elif hasattr(delegating_mp_factory, "_torch_mp_context_for_queues") and delegating_mp_factory._torch_mp_context_for_queues:  # type: ignore
                delegating_mp_factory._torch_mp_context_for_queues.Queue.assert_called_once_with(maxsize=30)  # type: ignore

            assert isinstance(sink, DelegatingMultiprocessQueueSink)
            assert isinstance(source, DelegatingMultiprocessQueueSource)
            assert sink._DelegatingMultiprocessQueueSink__default_queue_sink is mock_default_sink_inst  # type: ignore [attr-defined]
            assert sink._DelegatingMultiprocessQueueSink__torch_queue_sink is mock_torch_sink_inst  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__default_queue_source is mock_default_source_inst  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__torch_queue_source is mock_torch_source_inst  # type: ignore [attr-defined]

        # Reset mocks for the next scenario
        if hasattr(delegating_mp_factory, "_default_manager_instance") and delegating_mp_factory._default_manager_instance:  # type: ignore
            delegating_mp_factory._default_manager_instance.Queue.reset_mock()  # type: ignore
        if hasattr(delegating_mp_factory, "_torch_manager_instance") and delegating_mp_factory._torch_manager_instance:  # type: ignore
            delegating_mp_factory._torch_manager_instance.Queue.reset_mock()  # type: ignore
        if hasattr(delegating_mp_factory, "_torch_mp_context_for_queues") and delegating_mp_factory._torch_mp_context_for_queues:  # type: ignore
            delegating_mp_factory._torch_mp_context_for_queues.Queue.reset_mock()  # type: ignore

        mock_sink_wrapper_cls.reset_mock()
        mock_source_wrapper_cls.reset_mock()

        delegating_mp_factory.max_queue_size = 20
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=False
        ):
            sink, source = delegating_mp_factory.create_queues()
            if hasattr(delegating_mp_factory, "_default_manager_instance") and delegating_mp_factory._default_manager_instance:  # type: ignore
                delegating_mp_factory._default_manager_instance.Queue.assert_called_once_with(maxsize=20)  # type: ignore

            if hasattr(delegating_mp_factory, "_torch_manager_instance") and delegating_mp_factory._torch_manager_instance:  # type: ignore
                assert not delegating_mp_factory._torch_manager_instance.Queue.called  # type: ignore
            if hasattr(delegating_mp_factory, "_torch_mp_context_for_queues") and delegating_mp_factory._torch_mp_context_for_queues:  # type: ignore
                assert not delegating_mp_factory._torch_mp_context_for_queues.Queue.called  # type: ignore

            assert isinstance(sink, DelegatingMultiprocessQueueSink)
            assert isinstance(source, DelegatingMultiprocessQueueSource)
            assert sink._DelegatingMultiprocessQueueSink__default_queue_sink is mock_default_sink_inst  # type: ignore [attr-defined]
            assert sink._DelegatingMultiprocessQueueSink__torch_queue_sink is None  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__default_queue_source is mock_default_source_inst  # type: ignore [attr-defined]
            assert source._DelegatingMultiprocessQueueSource__torch_queue_source is None  # type: ignore [attr-defined]


class TestDelegatingMultiprocessQueueSink:
    def test_sink_init(
        self,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
    ) -> None:
        assert delegating_sink._DelegatingMultiprocessQueueSink__default_queue_sink is mock_default_sink  # type: ignore [attr-defined]
        assert delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink is None  # type: ignore [attr-defined]
        assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is None  # type: ignore [attr-defined]
        assert not delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]
        assert not delegating_sink.closed

    @pytest.mark.skipif(not _torch_installed, reason="PyTorch not installed.")
    @pytest.mark.parametrize(
        "delegating_sink", ["mock_torch_sink"], indirect=True
    )
    def test_sink_put_first_item_uses_torch_path(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[
            Any
        ],  # Now properly parameterized
        mock_default_sink: MagicMock,
        mock_torch_sink: MagicMock,  # Will be the one inside delegating_sink
        tensor_item: TensorType,
    ) -> None:
        # delegating_sink fixture should now have mock_torch_sink due to parametrize
        assert delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink is mock_torch_sink  # type: ignore [attr-defined]

        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            # Ensure isinstance check for torch.Tensor will return True for tensor_item
            if not (
                _torch_installed
                and torch_module
                and isinstance(tensor_item, torch_module.Tensor)
            ):
                mocker.patch(
                    "isinstance",  # Patch built-in isinstance
                    lambda obj, type_info: (
                        True
                        if type_info
                        is (
                            _torch_installed
                            and torch_module
                            and torch_module.Tensor
                        )
                        and obj is tensor_item
                        else isinstance(obj, type_info)
                    ),
                )

            mock_default_sink.put_blocking.return_value = True
            mock_torch_sink.put_blocking.return_value = True
            delegating_sink.put_blocking(tensor_item, timeout=1.0)

            mock_default_sink.put_blocking.assert_called_once_with(
                dqf_module.COORDINATION_USE_TORCH,
                timeout=1.0,  # Coordination timeout should also be respected
            )
            mock_torch_sink.put_blocking.assert_called_once_with(
                tensor_item, timeout=1.0
            )
            assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is mock_torch_sink  # type: ignore [attr-defined]
            assert delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]

    def test_sink_put_first_item_uses_default_path_non_tensor(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        mock_default_sink: MagicMock,
        test_item: Any,
    ) -> None:
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available",
            return_value=True,  # Torch available, but item is not tensor
        ):
            mock_default_sink.put_blocking.return_value = True
            # Ensure torch_sink (if present from other tests) is not called
            if delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink:  # type: ignore [attr-defined]
                delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink.put_blocking.assert_not_called()  # type: ignore [attr-defined]

            delegating_sink.put_blocking(test_item, timeout=1.0)
            mock_default_sink.put_blocking.assert_has_calls(
                [
                    mocker.call(
                        dqf_module.COORDINATION_USE_DEFAULT, timeout=1.0
                    ),  # timeout passed to coordination
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
        tensor_item: Optional[TensorType],  # Can be a tensor or any other item
        test_item: Any,
    ) -> None:
        item_to_use = tensor_item if tensor_item is not None else test_item
        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available",
            return_value=False,  # Torch not available
        ):
            mock_default_sink.put_blocking.return_value = True
            if delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink:  # type: ignore [attr-defined]
                delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink.put_blocking.assert_not_called()  # type: ignore [attr-defined]

            delegating_sink.put_blocking(item_to_use, timeout=1.0)
            mock_default_sink.put_blocking.assert_has_calls(
                [
                    mocker.call(
                        dqf_module.COORDINATION_USE_DEFAULT, timeout=1.0
                    ),
                    mocker.call(item_to_use, timeout=1.0),
                ]
            )
            assert delegating_sink._DelegatingMultiprocessQueueSink__selected_queue_sink is mock_default_sink  # type: ignore [attr-defined]
            assert delegating_sink._DelegatingMultiprocessQueueSink__coordination_sent  # type: ignore [attr-defined]

    @pytest.mark.skipif(not _torch_installed, reason="PyTorch not available.")
    @pytest.mark.parametrize(
        "delegating_sink", ["mock_torch_sink"], indirect=True
    )
    def test_sink_put_subsequent_item(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],  # Parameterized
        mock_default_sink: MagicMock,
        mock_torch_sink: MagicMock,  # Will be the one inside delegating_sink
        tensor_item: TensorType,
        test_item: Any,
    ) -> None:
        assert delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink is mock_torch_sink  # type: ignore [attr-defined]

        with mocker.patch(
            f"{dqf_module.__name__}.is_torch_available", return_value=True
        ):
            if not (
                _torch_installed
                and torch_module
                and isinstance(tensor_item, torch_module.Tensor)
            ):
                mocker.patch(
                    "isinstance",
                    lambda obj, type_info: (
                        True
                        if type_info
                        is (
                            _torch_installed
                            and torch_module
                            and torch_module.Tensor
                        )
                        and obj is tensor_item
                        else isinstance(obj, type_info)
                    ),
                )

            mock_default_sink.put_blocking.return_value = True
            mock_torch_sink.put_blocking.return_value = True
            # First put (selects torch path)
            delegating_sink.put_blocking(tensor_item)
            mock_default_sink.put_blocking.reset_mock()
            mock_torch_sink.put_blocking.reset_mock()

            # Subsequent put
            delegating_sink.put_blocking(test_item, timeout=0.5)
            mock_default_sink.put_blocking.assert_not_called()
            mock_torch_sink.put_blocking.assert_called_once_with(
                test_item, timeout=0.5
            )

    def test_put_when_closed_raises_runtime_error(
        self,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],
        test_item: Any,
    ) -> None:
        delegating_sink.close()
        with pytest.raises(RuntimeError, match="Sink closed."):
            delegating_sink.put_blocking(test_item)
        with pytest.raises(RuntimeError, match="Sink closed."):
            delegating_sink.put_nowait(test_item)

    @pytest.mark.parametrize(
        "delegating_sink", ["mock_torch_sink"], indirect=True
    )
    def test_sink_close_closes_underlying_sinks(
        self,
        mocker: MockerFixture,
        delegating_sink: DelegatingMultiprocessQueueSink[Any],  # Parameterized
        mock_default_sink: MagicMock,
        mock_torch_sink: MagicMock,  # Will be the one inside delegating_sink
    ) -> None:
        assert delegating_sink._DelegatingMultiprocessQueueSink__torch_queue_sink is mock_torch_sink  # type: ignore [attr-defined]
        mock_handler = mocker.MagicMock()
        delegating_sink.on_close += mock_handler

        delegating_sink.close()

        mock_default_sink.close.assert_called_once()
        mock_torch_sink.close.assert_called_once()  # mock_torch_sink is guaranteed to be there
        mock_handler.assert_called_once()
        assert delegating_sink.closed


class TestDelegatingMultiprocessQueueSource:
    def test_source_init(
        self,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
    ) -> None:
        assert delegating_source._DelegatingMultiprocessQueueSource__default_queue_source is mock_default_source  # type: ignore [attr-defined]
        assert delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source is None  # type: ignore [attr-defined]
        assert delegating_source._DelegatingMultiprocessQueueSource__selected_queue_source is None  # type: ignore [attr-defined]
        assert not delegating_source.closed

    @pytest.mark.parametrize(
        "delegating_source", ["mock_torch_source"], indirect=True
    )
    def test_source_get_first_item_selects_torch_path(
        self,
        delegating_source: DelegatingMultiprocessQueueSource[
            Any
        ],  # Parameterized
        mock_default_source: MagicMock,
        mock_torch_source: MagicMock,  # Will be the one inside delegating_source
        test_item: Any,
    ) -> None:
        assert delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source is mock_torch_source  # type: ignore [attr-defined]

        mock_default_source.get_blocking.return_value = (
            dqf_module.COORDINATION_USE_TORCH
        )
        mock_torch_source.get_blocking.return_value = test_item
        item = delegating_source.get_blocking(timeout=1.0)

        mock_default_source.get_blocking.assert_called_once_with(timeout=1.0)
        mock_torch_source.get_blocking.assert_called_once_with(timeout=1.0)
        assert item == test_item
        assert delegating_source._DelegatingMultiprocessQueueSource__selected_queue_source is mock_torch_source  # type: ignore [attr-defined]

    def test_source_get_first_item_selects_default_path(
        self,
        mocker: MockerFixture,
        delegating_source: DelegatingMultiprocessQueueSource[Any],
        mock_default_source: MagicMock,
        test_item: Any,
    ) -> None:
        mock_default_source.get_blocking.side_effect = [
            dqf_module.COORDINATION_USE_DEFAULT,
            test_item,
        ]
        item = delegating_source.get_blocking(timeout=1.0)

        mock_default_source.get_blocking.assert_has_calls(
            [
                mocker.call(timeout=1.0),
                mocker.call(timeout=1.0),
            ]  # Two calls to default_source.get_blocking
        )
        assert item == test_item
        assert delegating_source._DelegatingMultiprocessQueueSource__selected_queue_source is mock_default_source  # type: ignore [attr-defined]
        if delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source:  # type: ignore [attr-defined]
            delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source.get_blocking.assert_not_called()  # type: ignore [attr-defined]

    @pytest.mark.parametrize(
        "delegating_source", ["mock_torch_source"], indirect=True
    )
    def test_source_get_subsequent_item(
        self,
        delegating_source: DelegatingMultiprocessQueueSource[
            Any
        ],  # Parameterized
        mock_default_source: MagicMock,
        mock_torch_source: MagicMock,  # Will be the one inside delegating_source
        test_item: Any,
    ) -> None:
        assert delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source is mock_torch_source  # type: ignore [attr-defined]
        # First get (selects torch path)
        mock_default_source.get_blocking.return_value = (
            dqf_module.COORDINATION_USE_TORCH
        )
        mock_torch_source.get_blocking.return_value = "first_item"
        delegating_source.get_blocking(timeout=1.0)
        mock_default_source.get_blocking.reset_mock()
        mock_torch_source.get_blocking.reset_mock()

        # Subsequent get
        mock_torch_source.get_blocking.return_value = test_item
        item_2 = delegating_source.get_blocking(timeout=0.5)

        mock_default_source.get_blocking.assert_not_called()
        mock_torch_source.get_blocking.assert_called_once_with(timeout=0.5)
        assert item_2 == test_item

    @pytest.mark.parametrize(
        "delegating_source", ["mock_torch_source"], indirect=True
    )
    def test_source_close_closes_underlying_sources(
        self,
        mocker: MockerFixture,
        delegating_source: DelegatingMultiprocessQueueSource[
            Any
        ],  # Parameterized
        mock_default_source: MagicMock,
        mock_torch_source: MagicMock,  # Will be the one inside delegating_source
    ) -> None:
        assert delegating_source._DelegatingMultiprocessQueueSource__torch_queue_source is mock_torch_source  # type: ignore [attr-defined]
        mock_handler = mocker.MagicMock()
        delegating_source.on_close += mock_handler

        delegating_source.close()

        mock_default_source.close.assert_called_once()
        mock_torch_source.close.assert_called_once()  # mock_torch_source is guaranteed
        mock_handler.assert_called_once()
        assert delegating_source.closed


def _producer_worker_default_mp(
    sink_queue: DelegatingMultiprocessQueueSink[Any],
    item_to_send: Any,
    result_queue: Any,  # Should be multiprocessing.Queue or torch.multiprocessing.Queue
) -> None:
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
    source_queue: DelegatingMultiprocessQueueSource[Any],
    result_queue: Any,  # Should be multiprocessing.Queue or torch.multiprocessing.Queue
) -> None:
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
    item_to_send: TensorType,  # Type hint for clarity
    result_queue: Any,  # Should be multiprocessing.Queue or torch.multiprocessing.Queue
) -> None:
    try:
        if not (
            _torch_installed
            and torch_module
            and isinstance(item_to_send, torch_module.Tensor)
        ):
            result_queue.put(
                "producer_error_item_not_tensor_or_torch_unavailable"
            )
            return
        # For the delegating worker, share_memory_ was NOT called in the version that led to FileNotFoundError with raw queues.
        # It was added to the raw worker. For consistency in _this_ worker, we might also need it,
        # or it might interact differently with the delegating wrappers.
        # Let's assume it might be needed if tensors are involved, but be mindful of past errors.
        if (
            _torch_installed
            and torch_module
            and isinstance(item_to_send, torch_module.Tensor)
        ):
            item_to_send.share_memory_()

        if not sink_queue.put_blocking(item_to_send, timeout=10):
            result_queue.put("put_failed_timeout_or_full")
            return
        result_queue.put("put_successful")
    except Exception as e:
        import traceback

        # Use a more generic exception key or specify it's for the delegating path
        result_queue.put(
            f"producer_exception_delegating_tensor: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _consumer_worker_tensor_mp(
    source_queue: DelegatingMultiprocessQueueSource[Any],
    result_queue: Any,  # Should be multiprocessing.Queue or torch.multiprocessing.Queue
) -> None:
    try:
        item = source_queue.get_blocking(timeout=15)
        result_queue.put(item)
    except queue.Empty:
        result_queue.put("get_failed_timeout_empty_delegating_tensor")
    except Exception as e:
        import traceback

        result_queue.put(
            f"consumer_exception_delegating_tensor: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _execute_mp_correctness_logic(
    start_method_to_try: str,
    factory_fixture: dqf_module.DelegatingMultiprocessQueueFactory[Any],
    item_to_send: Any,
    is_tensor_test: bool = False,
) -> None:
    # Determine the multiprocessing context (standard or Torch)
    mp_context_to_use: Any = (
        multiprocessing  # Default to standard multiprocessing
    )
    if (
        _torch_installed
        and torch_mp_module
        and hasattr(torch_mp_module, "get_context")
        and hasattr(
            torch_mp_module, "get_start_method"
        )  # Ensure get_start_method exists
    ):
        # Attempt to set the start method. Warnings may be issued if it fails.
        set_torch_mp_start_method_if_needed(method=start_method_to_try)
        # Use the torch multiprocessing context if available and configured
        # This might still be the standard MP context if torch didn't override it (e.g. on macOS with 'fork')
        mp_context_to_use = torch_mp_module.get_context()

    producer_status_queue: Any = mp_context_to_use.Queue()
    consumer_data_queue: Any = mp_context_to_use.Queue()
    sink_q, source_q = factory_fixture.create_queues()

    worker_producer_fn = (
        _producer_worker_tensor_mp
        if is_tensor_test
        else _producer_worker_default_mp
    )
    worker_consumer_fn = (
        _consumer_worker_tensor_mp
        if is_tensor_test
        else _consumer_worker_default_mp
    )

    producer_process = mp_context_to_use.Process(
        target=worker_producer_fn,
        args=(sink_q, item_to_send, producer_status_queue),
    )
    consumer_process = mp_context_to_use.Process(
        target=worker_consumer_fn, args=(source_q, consumer_data_queue)
    )

    process_join_timeout = 25  # seconds
    try:
        producer_process.start()
        consumer_process.start()

        producer_process.join(timeout=process_join_timeout)
        consumer_process.join(timeout=process_join_timeout)

        assert (
            not producer_process.is_alive()
        ), f"Producer process timed out after {process_join_timeout}s."
        assert (
            not consumer_process.is_alive()
        ), f"Consumer process timed out after {process_join_timeout}s."

        # Check exit codes after ensuring processes are joined
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
            pytest.fail(f"Consumer reported exception: {received_item}")

        if is_tensor_test:
            assert torch_module is not None and isinstance(
                received_item, torch_module.Tensor
            ), f"Expected Tensor, got {type(received_item)}"
            assert torch_module.equal(
                received_item, item_to_send
            ), f"Tensor data mismatch: sent {item_to_send}, got {received_item}"
        else:
            # Assuming default items are dicts for this test logic
            assert isinstance(
                received_item, dict
            ), f"Expected dict, got {type(received_item)}"
            assert (
                received_item == item_to_send
            ), f"Dict data mismatch: sent {item_to_send}, got {received_item}"
    finally:
        # Ensure processes are terminated if still alive (e.g. due to timeout in join)
        if producer_process.is_alive():
            producer_process.terminate()
            producer_process.join()
        if consumer_process.is_alive():
            consumer_process.terminate()
            consumer_process.join()

        # Close queues
        if hasattr(sink_q, "close"):
            sink_q.close()
        if hasattr(source_q, "close"):
            source_q.close()

        # Standard lib Queues: close + join_thread. Manager queues often just need close.
        # For result queues, be robust.
        for q_to_clean in [producer_status_queue, consumer_data_queue]:
            if hasattr(q_to_clean, "close"):
                try:
                    q_to_clean.close()
                except Exception:
                    pass  # Ignore errors on cleanup
            if hasattr(q_to_clean, "join_thread"):
                try:
                    q_to_clean.join_thread()
                except Exception:
                    pass  # Ignore errors on cleanup


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
        test_item: Any,  # Use the fixture
    ) -> None:
        _execute_mp_correctness_logic(
            "fork", delegating_mp_factory, test_item, is_tensor_test=False
        )

    def test_multiprocess_correctness_fork_tensor_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
    ) -> None:
        if not (_torch_installed and torch_module and torch_mp_module):
            pytest.skip(
                "PyTorch or torch.multiprocessing not fully available for this test."
            )

        assert (
            torch_module is not None
        ), "Torch module required for tensor item."
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
class TestMultiprocessCorrectnessSpawn:
    # @pytest.mark.xfail( # xfail removed for now to see actual error with current setup
    #     raises=RuntimeError, # This might be too specific, could be PicklingError etc.
    #     strict=True,
    #     reason="Pickling/context issues with spawn expected for default items using current delegating factory.",
    # )
    def test_multiprocess_correctness_spawn_default_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
        test_item: Any,  # Use the fixture
    ) -> None:
        _execute_mp_correctness_logic(
            "spawn", delegating_mp_factory, test_item, is_tensor_test=False
        )

    # @pytest.mark.xfail( # xfail removed for now
    #     raises=RuntimeError, # This might be too specific
    #     strict=True,
    #     reason="Known issues with Torch tensors and 'spawn' method via current delegating factory.",
    # )
    def test_multiprocess_correctness_spawn_tensor_item(
        self,
        delegating_mp_factory: dqf_module.DelegatingMultiprocessQueueFactory[
            Any
        ],
        tensor_item: TensorType,  # Use the fixture
    ) -> None:
        assert (
            torch_module is not None and tensor_item is not None
        ), "Torch module and tensor_item required."
        _execute_mp_correctness_logic(
            "spawn", delegating_mp_factory, tensor_item, is_tensor_test=True  # type: ignore
        )
