from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Any  # Added Any
from unittest import mock
import multiprocessing  # Added for context object

import pytest

from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.split_process.remote_runtime_factory import RemoteRuntimeFactory
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.runtime.runtime_config import ServiceType  # Added import
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)

# MPContextType is expected to be a type like multiprocessing.context.BaseContext
# As MPContextType is not defined in multiprocessing_context_provider,
# we use the actual base class for multiprocessing contexts.
from multiprocessing.context import BaseContext as MPContextType


# Mock classes for dependencies
MockRuntimeInitializer = mock.Mock(spec=RuntimeInitializer)
MockThreadPoolExecutor = mock.Mock(spec=ThreadPoolExecutor)
MockThreadWatcher = mock.Mock(spec=ThreadWatcher)

# Mock context for testing (Queue factory mocks will be created per test)
MockStdContext = mock.Mock(spec=multiprocessing.get_context("spawn").__class__)
MockTorchContext = mock.Mock(
    spec=MPContextType
)  # Use the alias, or torch specific if definitely testing torch path


@pytest.fixture
def mock_mp_context_provider() -> Iterator[mock.Mock]: # Removed mocker, not used in this version
    """Fixture to mock MultiprocessingContextProvider, handling Generic[Any]."""
    patch_target = "tsercom.api.split_process.split_runtime_factory_factory.MultiprocessingContextProvider"

    # This is the mock instance we want SplitRuntimeFactoryFactory to use for self.__mp_context_provider
    mock_provider_instance_to_be_used_by_sut = mock.MagicMock()

    # This mock will represent the specialized type callable, e.g., MultiprocessingContextProvider[Any]
    # When this is called (instantiated), it should return mock_provider_instance_to_be_used_by_sut
    mock_specialized_provider_type_callable = mock.MagicMock(return_value=mock_provider_instance_to_be_used_by_sut)

    # This mock replaces the class name "MultiprocessingContextProvider" in the target module.
    # It needs to handle being subscripted (via __getitem__).
    mock_class_replacement = mock.MagicMock()
    mock_class_replacement.__getitem__.return_value = mock_specialized_provider_type_callable

    with mock.patch(patch_target, new=mock_class_replacement): # Use 'new' to replace with our preconfigured mock
        # The test will receive the instance that SUT will use.
        yield mock_provider_instance_to_be_used_by_sut


@pytest.fixture
def split_runtime_factory_factory_instance(
    mock_mp_context_provider: mock.Mock,
) -> SplitRuntimeFactoryFactory:  # Depends on the above fixture
    """Fixture to create a SplitRuntimeFactoryFactory instance with a mocked provider."""
    # The provider is already mocked by mock_mp_context_provider fixture
    return SplitRuntimeFactoryFactory(
        thread_pool=MockThreadPoolExecutor(),
        thread_watcher=MockThreadWatcher(),
    )


def test_create_pair_uses_default_context_and_factory(
    split_runtime_factory_factory_instance: SplitRuntimeFactoryFactory,
    mock_mp_context_provider: mock.Mock,
    mocker: Any, # Add mocker fixture
) -> None:
    """
    Tests that _create_pair uses the context and factory from the provider
    (simulating default/non-torch case).
    """
    # Configure the mock provider to return a standard context and default factory
    mock_std_context_instance = MockStdContext() # Use module-level mock context instance

    # Configure the mock provider to return a standard context and default factory
    mock_std_context_instance = MockStdContext() # Use module-level mock context instance

    # Configure the mock provider to return a standard context and default factory
    mock_std_context_instance = MockStdContext() # Use module-level mock context instance

    the_queue_factory_mock_instance_std = mock.MagicMock()

    type(mock_mp_context_provider).context = mock.PropertyMock(
        return_value=mock_std_context_instance
    )
    type(mock_mp_context_provider).queue_factory = mock.PropertyMock(
        return_value=the_queue_factory_mock_instance_std
    )

    # Directly patch the create_queues method on this instance using mocker
    # (mocker fixture is implicitly available in pytest test methods)
    mock_create_queues_std = mocker.patch.object(
        the_queue_factory_mock_instance_std,
        'create_queues',
        side_effect=[
            (mock.MagicMock(), mock.MagicMock()),
            (mock.MagicMock(), mock.MagicMock()),
        ]
    )

    # Setup mocks for DefaultMultiprocessQueueFactory for command queue

    mock_configured_cmd_instance = mock.Mock()
    mock_configured_cmd_instance.create_queues.return_value = (mock.Mock(), mock.Mock())

    # This mock represents the specialized class, e.g., DefaultMultiprocessQueueFactory[RuntimeCommand]
    # When it's instantiated with (context=...), it returns mock_configured_cmd_instance
    mock_specialized_class_callable = mock.Mock(
        return_value=mock_configured_cmd_instance
    )

    # Import the actual class to patch its __class_getitem__ method
    from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
        DefaultMultiprocessQueueFactory as ActualDQF,
    )

    with mock.patch.object(
        ActualDQF, "__class_getitem__", return_value=mock_specialized_class_callable
    ) as mock_cgetitem_method:
        initializer = MockRuntimeInitializer()
        initializer.timeout_seconds = None  # Simplify aggregator mocking
        initializer.data_aggregator_client = None
        initializer.service_type_enum = ServiceType.SERVER

        runtime_handle, runtime_factory = (
            split_runtime_factory_factory_instance._create_pair(initializer)
        )

        # Assert that the context and queue_factory properties were accessed
        assert mock_mp_context_provider.context
        assert mock_mp_context_provider.queue_factory

        # Assert that the factory instance from provider was used for event and data queues
        assert mock_create_queues_std.call_count == 2

        # Assert that DefaultMultiprocessQueueFactory was instantiated for command queue with the correct context
        # The call is DefaultMultiprocessQueueFactory[RuntimeCommand](coTorchMemcpyQueueFactoryinstance)
        # So, DefaultMultiprocessQueueFactory[RuntimeCommand] will effectively call the mocked __class_getitem__.
        # This returns mock_specialized_class_callable.
        # Then mock_specialized_class_callable(context=...) is called.
        from tsercom.api.runtime_command import RuntimeCommand  # Import for assertion

        mock_cgetitem_method.assert_called_with(RuntimeCommand)
        mock_specialized_class_callable.assert_called_once_with(
            context=mock_std_context_instance
        )
        mock_configured_cmd_instance.create_queues.assert_called_once()

        assert isinstance(runtime_handle, ShimRuntimeHandle)
        assert isinstance(runtime_factory, RemoteRuntimeFactory)
        # RemoteRuntimeFactory does not store _mp_context directly.
        # The usage of the context is verified by checking its use in DefaultMultiprocessQueueFactory.


def test_create_pair_uses_torch_context_and_factory(
    split_runtime_factory_factory_instance: SplitRuntimeFactoryFactory,
    mock_mp_context_provider: mock.Mock,
    mocker: Any, # Add mocker fixture
) -> None:
    """
    Tests that _create_pair uses the context and factory from the provider
    (simulating torch case).
    """
    # Configure the mock provider to return a torch context and torch factory
    mock_torch_context_instance = MockTorchContext() # Use module-level mock context instance

    # Create a specific mock instance for the queue factory to be returned by the provider
    the_queue_factory_mock_instance_torch = mock.MagicMock()
    # Assign a new mock to its create_queues attribute using mocker
    mock_create_queues_torch = mocker.patch.object(
        the_queue_factory_mock_instance_torch,
        'create_queues',
        side_effect=[
            (mock.MagicMock(), mock.MagicMock()),  # Event queues
            (mock.MagicMock(), mock.MagicMock()),  # Data queues
        ]
    )

    type(mock_mp_context_provider).context = mock.PropertyMock(
        return_value=mock_torch_context_instance
    )
    type(mock_mp_context_provider).queue_factory = mock.PropertyMock(
        return_value=the_queue_factory_mock_instance_torch
    )

    # Setup mocks for DefaultMultiprocessQueueFactory for command queue (similar to above)
    mock_configured_cmd_instance_torch = mock.Mock()
    mock_configured_cmd_instance_torch.create_queues.return_value = (
        mock.Mock(),
        mock.Mock(),
    )
    mock_specialized_class_callable_torch = mock.Mock(
        return_value=mock_configured_cmd_instance_torch
    )

    # Import the actual class to patch its __class_getitem__ method (it's the same class)
    from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
        DefaultMultiprocessQueueFactory as ActualDQF,
    )  # Ensure it's imported for this test too

    with mock.patch.object(
        ActualDQF,
        "__class_getitem__",
        return_value=mock_specialized_class_callable_torch,
    ) as mock_cgetitem_method_torch:
        initializer = MockRuntimeInitializer()
        initializer.timeout_seconds = None
        initializer.data_aggregator_client = None
        initializer.service_type_enum = ServiceType.SERVER

        runtime_handle, runtime_factory = (
            split_runtime_factory_factory_instance._create_pair(initializer)
        )

        # Assert that the context and queue_factory properties were accessed
        assert mock_mp_context_provider.context
        assert mock_mp_context_provider.queue_factory

        # Assert that the factory instance from provider was used for event and data queues
        assert mock_create_queues_torch.call_count == 2

        # Assert that DefaultMultiprocessQueueFactory was instantiated for command queue with the torch context
        from tsercom.api.runtime_command import RuntimeCommand  # Import for assertion

        mock_cgetitem_method_torch.assert_called_with(RuntimeCommand)
        mock_specialized_class_callable_torch.assert_called_once_with(
            context=mock_torch_context_instance
        )
        mock_configured_cmd_instance_torch.create_queues.assert_called_once()

        assert isinstance(runtime_handle, ShimRuntimeHandle)
        assert isinstance(runtime_factory, RemoteRuntimeFactory)
        # RemoteRuntimeFactory does not store _mp_context directly.
        # The usage of the context is verified by checking its use in DefaultMultiprocessQueueFactory.


# It might be useful to keep a test for the old logic if TORCH_IS_AVAILABLE was a factor,
# but now that's encapsulated in the provider. The tests above cover provider interaction.
# The original tests for SplitRuntimeFactoryFactory might have tested queue types based on
# TORCH_IS_AVAILABLE and data types. This is now tested in MultiprocessingContextProvider's tests.
# The critical part for SplitRuntimeFactoryFactory is that it *uses* the provider correctly.
