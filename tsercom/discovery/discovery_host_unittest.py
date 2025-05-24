import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import functools # For functools.partial

from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.caller_id.caller_identifier import CallerIdentifier 
from tsercom.discovery.service_info import ServiceInfo 
# Import the actual InstanceListener for type hinting and spec for mocks
from tsercom.discovery.mdns.instance_listener import InstanceListener as ActualInstanceListener
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set
)

# 1. Global Event Loop Management Fixture
@pytest.fixture(autouse=True)
def manage_tsercom_global_event_loop_fixture(request):
    if not is_global_event_loop_set():
        set_tsercom_event_loop_to_current_thread()
    def finalizer():
        clear_tsercom_event_loop()
    request.addfinalizer(finalizer)

# 2. Mock for DiscoveryHost.Client
@pytest.fixture
def mock_discovery_host_client_fixture():
    client = AsyncMock(spec=DiscoveryHost.Client, name="MockDiscoveryHostClient")
    client._on_service_added = AsyncMock(name="client_on_service_added_method")
    return client

# 3. Mock for the actual InstanceListener class
@pytest.fixture
def mock_actual_instance_listener_fixture():
    # Patch the InstanceListener where DiscoveryHost would import it from if creating it.
    # This is tsercom.discovery.mdns.instance_listener.InstanceListener
    with patch("tsercom.discovery.mdns.instance_listener.InstanceListener", autospec=True) as MockListenerClass:
        mock_listener_instance = AsyncMock(spec=ActualInstanceListener, name="MockedActualInstanceListenerInstance")
        mock_listener_instance.start = AsyncMock(name="actual_listener_start_method")
        MockListenerClass.return_value = mock_listener_instance
        yield MockListenerClass, mock_listener_instance

# 4. Mock for CallerIdentifier.random (for _on_service_added tests)
@pytest.fixture
def mock_caller_identifier_random_fixture():
    with patch.object(CallerIdentifier, 'random', autospec=True) as mock_random:
        # Ensure each call to random returns a new, distinct mock object for easier verification
        mock_random.side_effect = lambda: MagicMock(spec=CallerIdentifier, name=f"RandomCallerIdInstance_{mock_random.call_count}")
        yield mock_random

# 5. The mock for run_on_event_loop
async def mock_run_on_event_loop_replacement(func_partial_producing_coro, event_loop=None, *args, **kwargs):
    """
    Mock for run_on_event_loop. Assumes func_partial_producing_coro returns a coroutine.
    This mock will be patched into tsercom.threading.aio.aio_utils.run_on_event_loop.
    """
    print(f"MOCK_RUN_ON_EVENT_LOOP_REPLACEMENT CALLED with func_partial: {func_partial_producing_coro}")
    
    # func_partial_producing_coro is expected to be functools.partial(self.__start_discovery_impl, client_arg)
    # or functools.partial(self.__stop_discovery_impl)
    # When called, it should produce a coroutine if the wrapped method (__start_discovery_impl) is async.
    coro = func_partial_producing_coro()
    
    if not asyncio.iscoroutine(coro):
        print(f"  WARNING: mock_run_on_event_loop_replacement received a non-coroutine: {type(coro)}."
              " This suggests the underlying method (e.g., __start_discovery_impl) might be synchronous"
              " or not correctly defined as async.")
        # If it's synchronous, we can't await it. Any async calls within it won't be awaited by this mock.
    else:
        print(f"  Awaiting coroutine from partial: {coro}")
        await coro # This executes the coroutine, e.g., __start_discovery_impl
        print(f"  Coroutine awaited successfully: {coro}")

    # The original run_on_event_loop returns a Future. We mimic this.
    # The result of this future is generally not used by DiscoveryHost.
    f = asyncio.Future()
    try:
        current_loop = asyncio.get_running_loop()
        if not current_loop.is_closed(): # Important check
            asyncio.ensure_future(f, loop=current_loop)
    except RuntimeError: # pragma: no cover
        # This might happen if the loop is closed or not available, e.g., during complex teardowns.
        pass 
    f.set_result(None) 
    return f

# 6. Test for start_discovery
@pytest.mark.asyncio
async def test_start_discovery_with_proper_patching(
    mock_actual_instance_listener_fixture, 
    mock_discovery_host_client_fixture 
):
    print("--- Starting test_start_discovery_with_proper_patching ---")
    MockListenerClass, mock_listener_instance = mock_actual_instance_listener_fixture
    mock_dh_client = mock_discovery_host_client_fixture

    # Patch run_on_event_loop at its source module: tsercom.threading.aio.aio_utils
    with patch("tsercom.threading.aio.aio_utils.run_on_event_loop", new=mock_run_on_event_loop_replacement) as mock_run_patch_obj:
        print(f"  Patch applied for tsercom.threading.aio.aio_utils.run_on_event_loop. Mock object: {mock_run_patch_obj}")

        SERVICE_TYPE = "_test_service_type_final_v3._tcp.local."
        # DiscoveryHost will try to import InstanceListener from tsercom.discovery.mdns.instance_listener
        # which is what mock_actual_instance_listener_fixture patches.
        host = DiscoveryHost(service_type=SERVICE_TYPE)
        print(f"  DiscoveryHost instance created with service_type: {SERVICE_TYPE}")

        # DiscoveryHost.start_discovery is 'async def'.
        # It calls run_on_event_loop (now our async mock_run_on_event_loop_replacement)
        # without an internal await. So, start_discovery implicitly returns the coroutine from our mock.
        # Thus, we await it here.
        await host.start_discovery(mock_dh_client)
        print("  Call to host.start_discovery has been awaited.")

        print("  Checking assertions for start_discovery...")
        mock_run_patch_obj.assert_called_once()
        print("  Assertion: mock_run_patch_obj.assert_called_once() - PASSED")

        MockListenerClass.assert_called_once_with(service_type=SERVICE_TYPE)
        print(f"  Assertion: MockListenerClass.assert_called_once_with(service_type='{SERVICE_TYPE}') - PASSED")
        
        # This assertion depends on __start_discovery_impl awaiting listener_instance.start()
        # If __start_discovery_impl is not awaiting it (a SUT issue), this might still pass if the call occurs.
        mock_listener_instance.start.assert_called_once_with(host)
        print(f"  Assertion: mock_listener_instance.start.assert_called_once_with(host_instance) - PASSED")
        
        assert host._DiscoveryHost__client is mock_dh_client
        print(f"  Assertion: host._DiscoveryHost__client is mock_dh_client - PASSED")
        
        print("--- test_start_discovery_with_proper_patching finished successfully ---")

# 7. Test for _on_service_added
@pytest.mark.asyncio
async def test_on_service_added_behavior(
    mock_actual_instance_listener_fixture, 
    mock_discovery_host_client_fixture,
    mock_caller_identifier_random_fixture
):
    print("--- Starting test_on_service_added_behavior ---")
    _, _ = mock_actual_instance_listener_fixture # Needed for start_discovery
    mock_dh_client = mock_discovery_host_client_fixture
    mock_random_caller_id_gen = mock_caller_identifier_random_fixture

    with patch("tsercom.threading.aio.aio_utils.run_on_event_loop", new=mock_run_on_event_loop_replacement) as mock_run_patch_obj_for_add:
        print(f"  Patch for run_on_event_loop in _on_service_added test. Mock: {mock_run_patch_obj_for_add}")

        SERVICE_TYPE_ADD_TEST = "_test_add_service_final_v3._tcp.local."
        host = DiscoveryHost(service_type=SERVICE_TYPE_ADD_TEST)
        print(f"  DiscoveryHost created for _on_service_added test with type: {SERVICE_TYPE_ADD_TEST}")
        
        await host.start_discovery(mock_dh_client)
        print("  host.start_discovery called for _on_service_added test setup.")

        service_info = ServiceInfo(
            name="TestSvcAddedV3", port=3333, addresses=["10.3.3.3"], 
            mdns_name="TestSvcAddedV3._test_add_service_final_v3._tcp.local."
        )
        print(f"  ServiceInfo created: {service_info.name}")

        await host._on_service_added(service_info)
        print(f"  host._on_service_added() called with {service_info.name}")

        print("  Checking assertions for _on_service_added...")
        mock_random_caller_id_gen.assert_called_once()
        generated_id_instance = mock_random_caller_id_gen.return_value 
        print("  Assertion: mock_random_caller_id_gen.assert_called_once() - PASSED")

        mock_dh_client._on_service_added.assert_called_once_with(service_info, generated_id_instance)
        print("  Assertion: mock_dh_client._on_service_added called with correct args - PASSED")

        assert host._known_services[service_info.mdns_name] is generated_id_instance
        print("  Assertion: CallerIdentifier cached correctly - PASSED")

        # Test caching
        mock_random_caller_id_gen.reset_mock()
        mock_dh_client._on_service_added.reset_mock()
        updated_service_info = ServiceInfo(
            name="TestSvcAddedV3Updated", port=4444, addresses=["10.4.4.4"], 
            mdns_name="TestSvcAddedV3._test_add_service_final_v3._tcp.local." # SAME mdns_name
        )
        await host._on_service_added(updated_service_info)
        mock_random_caller_id_gen.assert_not_called() 
        print("  Assertion: mock_random_caller_id_gen.assert_not_called() for cached service - PASSED")
        mock_dh_client._on_service_added.assert_called_once_with(updated_service_info, generated_id_instance)
        print("  Assertion: mock_dh_client._on_service_added called with updated info and CACHED ID - PASSED")
        print("--- test_on_service_added_behavior finished successfully ---")
