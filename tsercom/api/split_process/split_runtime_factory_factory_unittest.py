from concurrent.futures import ThreadPoolExecutor
from typing import Iterator  # Added Iterator
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
from tsercom.threading.multiprocess.multiprocessing_context_provider import (
    MPContextType,
)


# Mock classes for dependencies
MockRuntimeInitializer = mock.Mock(spec=RuntimeInitializer)
MockThreadPoolExecutor = mock.Mock(spec=ThreadPoolExecutor)
MockThreadWatcher = mock.Mock(spec=ThreadWatcher)

# Mock queue factories and context for testing
MockDefaultQueueFactory = mock.Mock(spec=DefaultMultiprocessQueueFactory)
MockTorchQueueFactory = mock.Mock(spec=TorchMultiprocessQueueFactory)
MockStdContext = mock.Mock(spec=multiprocessing.get_context("spawn").__class__)
MockTorchContext = mock.Mock(
    spec=MPContextType
)  # Use the alias, or torch specific if definitely testing torch path


@pytest.fixture
def mock_mp_context_provider() -> Iterator[mock.Mock]:
    """Fixture to mock MultiprocessingContextProvider."""
    with mock.patch(
        "tsercom.api.split_process.split_runtime_factory_factory.MultiprocessingContextProvider"
    ) as mock_provider_class:
        mock_provider_instance = mock_provider_class.return_value
        yield mock_provider_instance


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
) -> None:
    """
    Tests that _create_pair uses the context and factory from the provider
    (simulating default/non-torch case).
    """
    # Configure the mock provider to return a standard context and default factory
    mock_std_context_instance = MockStdContext()
    mock_default_q_factory_instance = MockDefaultQueueFactory()
    mock_mp_context_provider.get_context_and_factory.return_value = (
        mock_std_context_instance,
        mock_default_q_factory_instance,
    )

    # Mock the create_queues method for event, data
    mock_default_q_factory_instance.create_queues.side_effect = [
        (mock.Mock(), mock.Mock()),  # Event queues
        (mock.Mock(), mock.Mock()),  # Data queues
    ]

    # Mock the DefaultMultiprocessQueueFactory for command queue
    # This is tricky because it's instantiated inside _create_pair
    with mock.patch(
        "tsercom.threading.multiprocess.default_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    ) as mock_cmd_q_factory_class:

        # Configure the mock for when DefaultMultiprocessQueueFactory[RuntimeCommand](...) is called
        mock_configured_cmd_instance = mock.Mock()
        mock_configured_cmd_instance.create_queues.return_value = (
            mock.Mock(),
            mock.Mock(),
        )

        # Make DefaultMultiprocessQueueFactory[RuntimeCommand] return a mock that, when called, returns the configured instance
        mock_getitem_result = mock.Mock()
        mock_getitem_result.return_value = mock_configured_cmd_instance
        mock_cmd_q_factory_class.__getitem__.return_value = mock_getitem_result

        # This direct configuration on return_value is for DefaultMultiprocessQueueFactory() if ever called without __getitem__
        # However, the code uses __getitem__, so the above is more critical.
        # For safety, ensure the direct return_value also points to something reasonable or the same mock.
        mock_cmd_q_factory_class.return_value = mock_configured_cmd_instance

        initializer = MockRuntimeInitializer()
        initializer.timeout_seconds = None  # Simplify aggregator mocking
        initializer.data_aggregator_client = None
        initializer.service_type_enum = (
            ServiceType.SERVER
        )  # Configure service_type_enum (corrected indentation)

        runtime_handle, runtime_factory = (
            split_runtime_factory_factory_instance._create_pair(initializer)
        )

        mock_mp_context_provider.get_context_and_factory.assert_called_once()

        # Assert that the factory instance from provider was used for event and data queues
        assert mock_default_q_factory_instance.create_queues.call_count == 2

        # Assert that DefaultMultiprocessQueueFactory was instantiated for command queue with the correct context
        # The call is DefaultMultiprocessQueueFactory[RuntimeCommand](context=mock_std_context_instance)
        # So, __getitem__ is called with RuntimeCommand, then the result is called with context.
        from tsercom.api.runtime_command import RuntimeCommand  # Import for assertion

        mock_cmd_q_factory_class.__getitem__.assert_called_with(RuntimeCommand)
        mock_getitem_result.assert_called_once_with(context=mock_std_context_instance)
        mock_configured_cmd_instance.create_queues.assert_called_once()

        assert isinstance(runtime_handle, ShimRuntimeHandle)
        assert isinstance(runtime_factory, RemoteRuntimeFactory)
        # Check that the context was passed to RemoteRuntimeFactory
        assert runtime_factory._mp_context == mock_std_context_instance


def test_create_pair_uses_torch_context_and_factory(
    split_runtime_factory_factory_instance: SplitRuntimeFactoryFactory,
    mock_mp_context_provider: mock.Mock,
) -> None:
    """
    Tests that _create_pair uses the context and factory from the provider
    (simulating torch case).
    """
    # Configure the mock provider to return a torch context and torch factory
    mock_torch_context_instance = MockTorchContext()
    mock_torch_q_factory_instance = (
        MockTorchQueueFactory()
    )  # Use the specific mock for torch factory
    mock_mp_context_provider.get_context_and_factory.return_value = (
        mock_torch_context_instance,
        mock_torch_q_factory_instance,
    )

    # Mock the create_queues method for event, data
    mock_torch_q_factory_instance.create_queues.side_effect = [
        (mock.Mock(), mock.Mock()),  # Event queues
        (mock.Mock(), mock.Mock()),  # Data queues
    ]

    with mock.patch(
        "tsercom.threading.multiprocess.default_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
    ) as mock_cmd_q_factory_class:

        # Configure the mock for when DefaultMultiprocessQueueFactory[RuntimeCommand](...) is called
        mock_configured_cmd_instance = mock.Mock()
        mock_configured_cmd_instance.create_queues.return_value = (
            mock.Mock(),
            mock.Mock(),
        )

        # Make DefaultMultiprocessQueueFactory[RuntimeCommand] return a mock that, when called, returns the configured instance
        mock_getitem_result = mock.Mock()
        mock_getitem_result.return_value = mock_configured_cmd_instance
        mock_cmd_q_factory_class.__getitem__.return_value = mock_getitem_result

        mock_cmd_q_factory_class.return_value = (
            mock_configured_cmd_instance  # For safety, as in the other test
        )

        initializer = MockRuntimeInitializer()
        initializer.timeout_seconds = None
        initializer.data_aggregator_client = None
        initializer.service_type_enum = (
            ServiceType.SERVER
        )  # Configure service_type_enum (corrected indentation)

        runtime_handle, runtime_factory = (
            split_runtime_factory_factory_instance._create_pair(initializer)
        )

        mock_mp_context_provider.get_context_and_factory.assert_called_once()

        # Assert that the factory instance from provider was used for event and data queues
        assert mock_torch_q_factory_instance.create_queues.call_count == 2

        # Assert that DefaultMultiprocessQueueFactory was instantiated for command queue with the torch context
        from tsercom.api.runtime_command import RuntimeCommand  # Import for assertion

        mock_cmd_q_factory_class.__getitem__.assert_called_with(RuntimeCommand)
        mock_getitem_result.assert_called_once_with(context=mock_torch_context_instance)
        mock_configured_cmd_instance.create_queues.assert_called_once()

        assert isinstance(runtime_handle, ShimRuntimeHandle)
        assert isinstance(runtime_factory, RemoteRuntimeFactory)
        assert runtime_factory._mp_context == mock_torch_context_instance


# It might be useful to keep a test for the old logic if TORCH_IS_AVAILABLE was a factor,
# but now that's encapsulated in the provider. The tests above cover provider interaction.
# The original tests for SplitRuntimeFactoryFactory might have tested queue types based on
# TORCH_IS_AVAILABLE and data types. This is now tested in MultiprocessingContextProvider's tests.
# The critical part for SplitRuntimeFactoryFactory is that it *uses* the provider correctly.
