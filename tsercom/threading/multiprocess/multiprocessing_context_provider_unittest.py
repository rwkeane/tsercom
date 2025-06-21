import sys
import multiprocessing
from unittest import mock  # Import the whole module
import importlib  # Added for reloading

import pytest

# Conditional import for torch and its types for testing
try:
    import torch  # noqa: F401
    import torch.multiprocessing as torch_mp  # noqa: F401

    # For type checking, get the specific context type if torch is available
    # Note: isinstance checks will use the actual runtime types.
    if hasattr(torch_mp, "get_context"):
        TorchContextType = type(torch_mp.get_context("spawn"))
    else:  # Older torch versions might not have get_context in the same way
        TorchContextType = object  # Fallback
    _TORCH_INSTALLED = True
except ImportError:
    _TORCH_INSTALLED = False
    TorchContextType = type("TorchContextType", (), {})  # Dummy type

StdContextType = type(multiprocessing.get_context("spawn"))


# Import the class to be tested.
# OriginalMultiprocessingContextProvider is used to get the module path
# and as the class to instantiate after its module's _TORCH_AVAILABLE is patched.
from tsercom.threading.multiprocess.multiprocessing_context_provider import (
    MultiprocessingContextProvider as OriginalMultiprocessingContextProvider,
)
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)


# def get_provider_module_for_testing(torch_available_mock_value: bool):
#     """
#     Helper to get the MultiprocessingContextProvider class from its module,
#     ensuring the module is reloaded with _TORCH_AVAILABLE patched.
#     This allows tests to simulate torch being available or unavailable.
#     """
#     # The module name where MultiprocessingContextProvider is defined.
#     provider_module_name = OriginalMultiprocessingContextProvider.__module__
#
#     # Patch the _TORCH_AVAILABLE flag within that module.
#     # The path to _TORCH_AVAILABLE should be absolute from the perspective of where `patch` looks.
#     # This with statement means the patch is only active during the reload.
#     with mock.patch(
#         f"{provider_module_name}._TORCH_AVAILABLE", torch_available_mock_value
#     ):
#         # Reload the module. This is crucial because the module might have already
#         # evaluated _TORCH_AVAILABLE at its import time. Reloading makes it re-evaluate.
#         reloaded_module = importlib.reload(sys.modules[provider_module_name])
# Obsolete helper function removed.

@pytest.mark.skipif(
    not _TORCH_INSTALLED,
    reason="PyTorch is not installed, skipping torch-specific tests",
)
def test_lazy_init_with_torch_available() -> None:
    """
    Tests lazy initialization when PyTorch is available.
    Context and factory should be created once and cached.
    """
    provider_module_path = OriginalMultiprocessingContextProvider.__module__
    with mock.patch(f"{provider_module_path}._TORCH_AVAILABLE", True):
        # Instantiate the original class; its methods will see the patched global.
        provider = OriginalMultiprocessingContextProvider(context_method="spawn")

        # Mock the actual context creation and factory instantiation to check call counts
        with (
            mock.patch("torch.multiprocessing.get_context") as mock_get_torch_ctx,
            mock.patch(  # Patch where the class is defined
                "tsercom.threading.multiprocess.torch_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
            ) as mock_torch_q_factory_class,
        ):

            mock_torch_ctx_instance = mock.MagicMock(spec=TorchContextType)
            mock_get_torch_ctx.return_value = mock_torch_ctx_instance

            mock_torch_q_factory_instance = mock.MagicMock(
                spec=TorchMultiprocessQueueFactory
            )
            mock_torch_q_factory_class.return_value = mock_torch_q_factory_instance

            # First access of context
            context1 = provider.context
            mock_get_torch_ctx.assert_called_once_with("spawn")
            assert context1 is mock_torch_ctx_instance

            # First access of factory
            factory1 = provider.queue_factory
            # mock_get_torch_ctx should still be called once (due to factory accessing context)
            mock_get_torch_ctx.assert_called_once()
            mock_torch_q_factory_class.assert_called_once_with(
                context=mock_torch_ctx_instance
            )
            assert factory1 is mock_torch_q_factory_instance

            # Second access
            context2 = provider.context
            factory2 = provider.queue_factory

            # Creation methods should still only have been called once
            mock_get_torch_ctx.assert_called_once()
            mock_torch_q_factory_class.assert_called_once()
            assert context2 is context1  # Check for cached instance
            assert factory2 is factory1  # Check for cached instance

            # get_context_and_factory no longer exists
            # mock_get_torch_ctx.assert_called_once()
            # mock_torch_q_factory_class.assert_called_once()


def test_lazy_init_with_torch_unavailable() -> None:
    """
    Tests lazy initialization when PyTorch is unavailable.
    Context and factory should be created once and cached.
    """
    provider_module_path = OriginalMultiprocessingContextProvider.__module__
    with mock.patch(f"{provider_module_path}._TORCH_AVAILABLE", False):
        # No reload needed here if the class methods refer to the global _TORCH_AVAILABLE
        # from their module scope directly.
        provider = OriginalMultiprocessingContextProvider(context_method="spawn")

        with (
            mock.patch("multiprocessing.get_context") as mock_get_std_ctx,
            mock.patch(
                "tsercom.threading.multiprocess.default_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
            ) as mock_std_q_factory_class,  # This path should be where it's looked up by the provider
        ):
            mock_std_ctx_instance = mock.MagicMock(spec=StdContextType)
            mock_get_std_ctx.return_value = mock_std_ctx_instance

            mock_std_q_factory_instance = mock.MagicMock(
                spec=DefaultMultiprocessQueueFactory
            )
            mock_std_q_factory_class.return_value = mock_std_q_factory_instance

            # First access
            context1 = provider.context
            mock_get_std_ctx.assert_called_once_with("spawn")
            assert context1 is mock_std_ctx_instance

            factory1 = provider.queue_factory
            mock_get_std_ctx.assert_called_once()
            mock_std_q_factory_class.assert_called_once_with(
                context=mock_std_ctx_instance
            )
            assert factory1 is mock_std_q_factory_instance

            # Second access
            context2 = provider.context
            factory2 = provider.queue_factory

            mock_get_std_ctx.assert_called_once()
            mock_std_q_factory_class.assert_called_once()
            assert context2 is context1
            assert factory2 is factory1


@pytest.mark.skipif(not _TORCH_INSTALLED, reason="PyTorch is not installed.")
def test_properties_return_correct_types_with_torch() -> None:
    """Test properties return correct actual types when torch is available."""
    provider_module_path = OriginalMultiprocessingContextProvider.__module__
    with mock.patch(f"{provider_module_path}._TORCH_AVAILABLE", True):
        # Ensure the test uses the Original class but its methods see the patched global
        provider = OriginalMultiprocessingContextProvider(context_method="spawn")
        context = provider.context
        factory = provider.queue_factory

        assert isinstance(context, TorchContextType)
        from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
            TorchMultiprocessQueueFactory as ActualTorchFactory,
        )

        assert isinstance(factory, ActualTorchFactory)
        assert factory._mp_context is context


def test_properties_return_correct_types_without_torch() -> None:
    """Test properties return correct actual types when torch is unavailable."""
    provider_module_path = OriginalMultiprocessingContextProvider.__module__
    with mock.patch(f"{provider_module_path}._TORCH_AVAILABLE", False):
        provider = OriginalMultiprocessingContextProvider(context_method="spawn")
        context = provider.context
        factory = provider.queue_factory

        assert isinstance(context, StdContextType)
        from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
            DefaultMultiprocessQueueFactory as ActualDefaultFactory,
        )

        assert isinstance(factory, ActualDefaultFactory)
        assert factory._mp_context is context


def test_different_context_methods() -> None:
    """Test that different context methods are respected (if supported by system)."""
    provider_module_path = OriginalMultiprocessingContextProvider.__module__

    # Test with torch (if available)
    if _TORCH_INSTALLED:
        with mock.patch(f"{provider_module_path}._TORCH_AVAILABLE", True):
            provider_torch_spawn = OriginalMultiprocessingContextProvider(
                context_method="spawn"
            )
            ctx_torch_spawn = provider_torch_spawn.context
            assert "spawn" in ctx_torch_spawn.__class__.__name__.lower()
            assert hasattr(ctx_torch_spawn, "Process")

    # Test without torch
    with mock.patch(f"{provider_module_path}._TORCH_AVAILABLE", False):
        provider_std_spawn = OriginalMultiprocessingContextProvider(
            context_method="spawn"
        )
        ctx_std_spawn = provider_std_spawn.context
        assert "spawn" in ctx_std_spawn.__class__.__name__.lower()
        assert hasattr(ctx_std_spawn, "Process")

        if sys.platform != "win32":  # 'fork' is not available on Windows
            provider_std_fork = OriginalMultiprocessingContextProvider(
                context_method="fork"
            )
            ctx_std_fork = provider_std_fork.context
            assert "fork" in ctx_std_fork.__class__.__name__.lower()
            assert hasattr(ctx_std_fork, "Process")
