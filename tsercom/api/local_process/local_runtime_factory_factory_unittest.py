import pytest
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tsercom.api.local_process.local_runtime_factory_factory import (
    LocalRuntimeFactoryFactory,
)
from tsercom.api.local_process.local_runtime_factory import LocalRuntimeFactory
from tsercom.api.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.runtime.runtime_config import ServiceType


class FakeThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        # In a real fake, you might execute fn immediately or store it
        # For this test, we mostly care that it's passed around.
        pass

    def shutdown(self, wait=True):
        self._shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class FakeRuntimeInitializer:
    def __init__(
        self,
        service_type_str="Server",  # Changed parameter name for clarity
        data_aggregator_client=None,
        timeout_seconds=60,
        min_send_frequency_seconds: Optional[float] = None,
        auth_config=None,
    ):
        """Initializes a fake runtime initializer.

        Args:
            service_type_str: The service type string ("Server" or "Client").
            data_aggregator_client: Fake data aggregator client.
            timeout_seconds: Timeout in seconds.
            min_send_frequency_seconds: Minimum send frequency in seconds.
            auth_config: Fake auth configuration.
        """
        # Store the string, but also prepare the enum
        if service_type_str == "Server":
            self.__service_type_enum_val = ServiceType.SERVER
        elif service_type_str == "Client":
            self.__service_type_enum_val = ServiceType.CLIENT
        else:
            raise ValueError(f"Invalid service_type_str: {service_type_str}")

        # This is what RuntimeConfig would store if initialized directly with an enum
        self._RuntimeConfig__service_type = self.__service_type_enum_val

        self.data_aggregator_client = data_aggregator_client
        self.timeout_seconds = timeout_seconds
        self.auth_config = auth_config
        self.min_send_frequency_seconds = min_send_frequency_seconds

        # Attributes/methods that might be called by the class under test or its collaborators
        self.create_called = False
        self.create_args = None

    def create(self, thread_watcher, data_handler, grpc_channel_factory):
        self.create_called = True
        self.create_args = (thread_watcher, data_handler, grpc_channel_factory)
        # Return a dummy runtime object if needed by the actual create() chain
        return object()

    @property
    def service_type_enum(self):
        return self.__service_type_enum_val


@pytest.fixture
def fake_executor():
    return FakeThreadPoolExecutor(max_workers=1)


@pytest.fixture
def fake_initializer():
    return FakeRuntimeInitializer()


# Fixture to monkeypatch run_on_event_loop
@pytest.fixture
def patch_run_on_event_loop(request):
    # Path to the module and function to patch
    # Patching where it's *used* if 'from x import y' is used by the target module
    module_path = "tsercom.data.data_timeout_tracker"
    func_name = "run_on_event_loop"

    try:
        # Import the module
        import importlib

        target_module = importlib.import_module(module_path)

        # Store the original function if it exists
        original_func = getattr(target_module, func_name, None)

        # Create a fake function
        def fake_run_on_event_loop(coro, *args, **kwargs):
            # This fake does nothing, preventing the need for an actual event loop.
            pass

        # Apply the patch
        setattr(target_module, func_name, fake_run_on_event_loop)

        # Teardown: restore the original function after the test
        def restore():
            if original_func is not None:
                setattr(target_module, func_name, original_func)
            else:
                # If the function didn't exist before, remove our fake one
                delattr(target_module, func_name)

        request.addfinalizer(restore)
    except ImportError:
        # If the module itself doesn't exist, there's nothing to patch.
        # This might happen if the codebase structure changes.
        # Allow tests to proceed; they might fail for other reasons if this patch was critical.
        print(
            f"Warning: Module {module_path} not found for patching {func_name}."
        )
        pass


@pytest.fixture
def factory_factory(fake_executor):
    return LocalRuntimeFactoryFactory(thread_pool=fake_executor)


def test_create_pair_return_types(
    factory_factory, fake_initializer, patch_run_on_event_loop
):
    """Tests that _create_pair returns objects of the expected types."""
    wrapper, factory = factory_factory._create_pair(fake_initializer)
    assert isinstance(wrapper, RuntimeWrapper)
    assert isinstance(factory, LocalRuntimeFactory)


def test_create_pair_remote_data_aggregator_wiring(
    factory_factory, fake_initializer, fake_executor, patch_run_on_event_loop
):
    """Tests the creation and wiring of RemoteDataAggregatorImpl."""
    wrapper, factory = factory_factory._create_pair(fake_initializer)

    # Check LocalRuntimeFactory's data_reader
    assert isinstance(
        factory._LocalRuntimeFactory__data_reader, RemoteDataAggregatorImpl
    )
    aggregator_in_factory = factory._LocalRuntimeFactory__data_reader

    # Check RuntimeWrapper's data_reader (which is an aggregator)
    assert wrapper._RuntimeWrapper__aggregator is aggregator_in_factory

    # Verify RemoteDataAggregatorImpl initialization
    # To do this properly, we might need to make RemoteDataAggregatorImpl more inspectable
    # or use a more sophisticated fake if we can't access its init args.
    # For now, we check it got the thread_pool.
    # Accessing private attributes like _thread_pool is for testing purposes.
    assert (
        aggregator_in_factory._RemoteDataAggregatorImpl__thread_pool
        is fake_executor
    )  # Corrected attribute
    assert (
        aggregator_in_factory._RemoteDataAggregatorImpl__client
        is fake_initializer.data_aggregator_client
    )  # Corrected attribute
    # The RemoteDataAggregatorImpl takes timeout OR tracker.
    # LocalRuntimeFactoryFactory._create_pair passes timeout, so tracker is created internally.
    # We need to inspect the tracker's timeout if timeout is passed to RemoteDataAggregatorImpl
    # For now, let's assume the direct timeout field on the tracker if it exists or check the tracker itself.
    # The current RemoteDataAggregatorImpl.__init__ creates a tracker if timeout is given.
    # Let's verify the tracker's timeout.
    assert (
        aggregator_in_factory._RemoteDataAggregatorImpl__tracker._DataTimeoutTracker__timeout_seconds
        == fake_initializer.timeout_seconds
    )


def test_create_pair_async_poller_wiring(
    factory_factory, fake_initializer, patch_run_on_event_loop
):
    """Tests the creation and wiring of AsyncPoller."""
    wrapper, factory = factory_factory._create_pair(fake_initializer)

    # Check LocalRuntimeFactory's event_poller
    assert isinstance(factory._LocalRuntimeFactory__event_poller, AsyncPoller)
    poller_in_factory = factory._LocalRuntimeFactory__event_poller

    # Check RuntimeWrapper's event_poller
    assert wrapper._RuntimeWrapper__event_poller is poller_in_factory


def test_create_pair_runtime_command_bridge_wiring(
    factory_factory, fake_initializer, patch_run_on_event_loop
):
    """Tests the creation and wiring of RuntimeCommandBridge."""
    wrapper, factory = factory_factory._create_pair(fake_initializer)

    # Check LocalRuntimeFactory's bridge
    assert isinstance(
        factory._LocalRuntimeFactory__bridge, RuntimeCommandBridge
    )
    bridge_in_factory = factory._LocalRuntimeFactory__bridge

    # Check RuntimeWrapper's bridge
    assert wrapper._RuntimeWrapper__bridge is bridge_in_factory


def test_create_pair_local_runtime_factory_initializer(
    factory_factory, fake_initializer, patch_run_on_event_loop
):
    """Tests that LocalRuntimeFactory is created with the correct initializer."""
    _wrapper, factory = factory_factory._create_pair(fake_initializer)
    assert factory._LocalRuntimeFactory__initializer is fake_initializer


# Further tests could verify RuntimeWrapper arguments if necessary,
# but they are largely covered by checking the shared instances of poller, aggregator, and bridge.


# Test for __init__ of LocalRuntimeFactoryFactory itself
def test_local_runtime_factory_factory_init():
    fake_exec = FakeThreadPoolExecutor()
    factory_fac = LocalRuntimeFactoryFactory(thread_pool=fake_exec)
    assert (
        factory_fac._LocalRuntimeFactoryFactory__thread_pool is fake_exec
    )  # Corrected attribute

    real_exec = ThreadPoolExecutor(max_workers=1)
    factory_fac_real_exec = LocalRuntimeFactoryFactory(thread_pool=real_exec)
    assert (
        factory_fac_real_exec._LocalRuntimeFactoryFactory__thread_pool
        is real_exec
    )
    real_exec.shutdown()


# Fake client for RuntimeFactoryFactory
class FakeRuntimeFactoryFactoryClient(LocalRuntimeFactoryFactory.Client):
    def __init__(self):
        self.handle_ready_called = False
        self.received_handle = None

    def _on_handle_ready(self, handle):
        self.handle_ready_called = True
        self.received_handle = handle


# Test that the _create_pair method is correctly called via the public create_factory interface
def test_create_factory_calls_create_pair(
    factory_factory, fake_initializer, patch_run_on_event_loop
):
    """Tests that create_factory calls _create_pair and wires client."""
    fake_client = FakeRuntimeFactoryFactoryClient()

    # Call the public method
    factory = factory_factory.create_factory(fake_client, fake_initializer)

    # Retrieve the handle that was passed to the client
    assert fake_client.handle_ready_called
    handle = fake_client.received_handle

    assert isinstance(handle, RuntimeWrapper)  # wrapper is the handle
    assert isinstance(factory, LocalRuntimeFactory)
    assert factory._LocalRuntimeFactory__initializer is fake_initializer
    assert (
        handle._RuntimeWrapper__aggregator
        is factory._LocalRuntimeFactory__data_reader
    )  # Corrected attribute
    assert (
        handle._RuntimeWrapper__event_poller
        is factory._LocalRuntimeFactory__event_poller
    )
    assert (
        handle._RuntimeWrapper__bridge is factory._LocalRuntimeFactory__bridge
    )

    # Check that the initializer passed to LocalRuntimeFactory has the correct RuntimeConfig attributes
    # based on the fake_initializer passed to _create_pair.
    # LocalRuntimeFactory's __init__ calls super().__init__(other_config=initializer)
    # This sets up RuntimeConfig attributes on the LocalRuntimeFactory instance itself.
    # Compare with the actual enum value
    # factory._RuntimeConfig__service_type is already the enum set by RuntimeConfig.__init__
    # fake_initializer.service_type_enum provides the enum from the fake
    assert (
        factory._RuntimeConfig__service_type
        == fake_initializer.service_type_enum
    )
    assert (
        factory.data_aggregator_client
        == fake_initializer.data_aggregator_client
    )  # property access
    assert (
        factory.timeout_seconds == fake_initializer.timeout_seconds
    )  # property access


def test_create_pair_aggregator_no_timeout(
    factory_factory, mocker, patch_run_on_event_loop
):
    """
    Tests that RemoteDataAggregatorImpl is initialized without a timeout
    when the initializer's timeout_seconds is None.
    """
    # Use the existing FakeRuntimeInitializer from the test file
    initializer_no_timeout = FakeRuntimeInitializer(timeout_seconds=None)

    # Spy on RemoteDataAggregatorImpl.__init__
    # The path needs to be where it's imported and used by LocalRuntimeFactoryFactory
    mock_aggregator_init = mocker.spy(RemoteDataAggregatorImpl, "__init__")

    # Call the method under test
    factory_factory._create_pair(initializer_no_timeout)

    # Assert that RemoteDataAggregatorImpl.__init__ was called
    mock_aggregator_init.assert_called_once()

    # Inspect the arguments passed to RemoteDataAggregatorImpl.__init__
    # call_args is a tuple (args, kwargs) or Call object
    # We expect __init__(self, thread_pool, client=None, timeout=None, tracker=None)
    # In _create_pair, client comes from initializer.data_aggregator_client
    # and thread_pool from self.__thread_pool.

    # Get the keyword arguments passed to __init__
    # The first argument to spy is `self`, so we look at `call_args[0][0]` for self,
    # `call_args[0][1]` for thread_pool, etc. if positional.
    # Or, more robustly, check kwargs if they are used.

    # The constructor of RemoteDataAggregatorImpl is:
    # def __init__(self, thread_pool: ThreadPoolExecutor, client: Optional[Client] = None,
    #              timeout: Optional[int] = None, tracker: Optional[DataTimeoutTracker] = None):

    # LocalRuntimeFactoryFactory._create_pair calls it like this when timeout is None:
    # RemoteDataAggregatorImpl(self.__thread_pool, initializer.data_aggregator_client)
    # or with timeout:
    # RemoteDataAggregatorImpl(self.__thread_pool, initializer.data_aggregator_client, timeout=initializer.timeout_seconds)

    args_list = mock_aggregator_init.call_args_list
    assert len(args_list) == 1
    call_args = args_list[0]

    # Check positional arguments (self, thread_pool, client)
    # args[0] is self (the RemoteDataAggregatorImpl instance)
    # args[1] should be the thread_pool from factory_factory
    assert (
        call_args.args[1]
        is factory_factory._LocalRuntimeFactoryFactory__thread_pool
    )

    # Check client argument (it's initializer_no_timeout.data_aggregator_client, which is None by default for FakeRuntimeInitializer)
    client_arg_from_call = None
    if len(call_args.args) > 2:  # Check if passed positionally
        client_arg_from_call = call_args.args[2]
    elif "client" in call_args.kwargs:  # Check if passed as keyword
        client_arg_from_call = call_args.kwargs["client"]

    assert (
        client_arg_from_call is initializer_no_timeout.data_aggregator_client
    )

    # Check keyword arguments
    # When timeout is None in initializer, _create_pair does not pass 'timeout' or 'tracker' kwargs.
    # So, kwargs should be empty or not contain 'timeout'.
    # The default for 'timeout' in RemoteDataAggregatorImpl.__init__ is None.
    assert (
        "timeout" not in call_args.kwargs
    ), "Timeout kwarg should not be present when initializer.timeout_seconds is None"
    assert (
        "tracker" not in call_args.kwargs
    ), "Tracker kwarg should not be present initially"

    # Alternatively, if it always passed timeout=None:
    # assert call_args.kwargs.get('timeout') is None
