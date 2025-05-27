import asyncio
import pytest
import functools  # For functools.partial

from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo

# Import the actual InstanceListener for type hinting and spec for mocks
from tsercom.discovery.mdns.instance_listener import (
    InstanceListener as ActualInstanceListener,
)
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
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
def mock_discovery_host_client_fixture(mocker):
    client = mocker.AsyncMock(
        spec=DiscoveryHost.Client, name="MockDiscoveryHostClient"
    )
    client._on_service_added = mocker.AsyncMock(
        name="client_on_service_added_method"
    )
    return client


# 3. Mock for the actual InstanceListener class
@pytest.fixture
def mock_actual_instance_listener_fixture(mocker):
    # Patch InstanceListener where it's looked up by DiscoveryHost
    MockListenerClass = mocker.patch(
        "tsercom.discovery.discovery_host.InstanceListener",
        autospec=ActualInstanceListener,  # Use original for spec
    )
    mock_listener_instance = mocker.AsyncMock(
        spec=ActualInstanceListener, name="MockedActualInstanceListenerInstance"
    )
    # InstanceListener.start is expected by one of the assertions.
    # We create a spy mock that will be called by our side_effect.
    actual_listener_start_method_spy = mocker.AsyncMock(name="actual_listener_start_method_spy")

    async def start_side_effect(*args, **kwargs):
        print(f"DEBUG: mock_listener_instance.start (via side_effect) CALLED with args: {args}, kwargs: {kwargs}")
        try:
            # Forward the call to the spy mock
            result = await actual_listener_start_method_spy(*args, **kwargs)
            print(f"DEBUG: actual_listener_start_method_spy returned: {result}")
            return result
        except Exception as e:
            print(f"DEBUG: Exception in start_side_effect calling spy: {e!r}")
            raise

    # This is the mock attribute that the SUT will call
    mock_listener_instance.start = mocker.AsyncMock(
        name="start_method_wrapper_on_instance", 
        side_effect=start_side_effect
    )
    
    MockListenerClass.return_value = mock_listener_instance
    # Yield the spy for assertion
    yield MockListenerClass, mock_listener_instance, actual_listener_start_method_spy


# 4. Mock for CallerIdentifier.random (for _on_service_added tests)
@pytest.fixture
def mock_caller_identifier_random_fixture(mocker):
    # Create the specific mock instance that will be returned by CallerIdentifier.random()
    expected_caller_id_instance = mocker.MagicMock(
        spec=CallerIdentifier, name="FixedRandomCallerIdInstance"
    )
    mock_random_method = mocker.patch.object(
        CallerIdentifier, "random", autospec=True, return_value=expected_caller_id_instance
    )
    # Yield the mock method itself AND the instance it will return
    yield mock_random_method, expected_caller_id_instance


# Global list to store tasks created by the mock run_on_event_loop
_created_tasks_for_run_on_event_loop = []

# 5. The mock for run_on_event_loop - Synchronous version that stores tasks
def mock_run_on_event_loop_replacement_v3(
    func_partial_producing_coro, event_loop=None, *args, **kwargs
):
    global _created_tasks_for_run_on_event_loop
    print(
        f"MOCK_RUN_ON_EVENT_LOOP_REPLACEMENT_V3 CALLED with func_partial: {func_partial_producing_coro}"
    )
    coro = func_partial_producing_coro()

    if not asyncio.iscoroutine(coro):
        print(
            f"  WARNING: MOCK_RUN_ON_EVENT_LOOP_REPLACEMENT_V3 received a non-coroutine: {type(coro)}."
        )
        f = asyncio.Future()
        f.set_result(None)
        return f

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        _created_tasks_for_run_on_event_loop.append(task) # Store the task
        print(f"  Scheduled coroutine {coro} on loop {loop} as task {task}. Stored task.")
        
        # The original run_on_event_loop returns a future.
        # The test will await the task directly, so this future's result isn't critical.
        f = asyncio.Future(loop=loop)
        f.set_result(None) 
        return f
    except RuntimeError: 
        print(
            "  MOCK_RUN_ON_EVENT_LOOP_REPLACEMENT_V3: No running event loop. Cannot schedule coro."
        )
        f = asyncio.Future()
        f.set_result(None)
        return f

# 6. Test for start_discovery
@pytest.mark.asyncio
async def test_start_discovery_with_proper_patching(
    mock_actual_instance_listener_fixture, # This fixture now yields a 3-tuple
    mock_discovery_host_client_fixture,
    mocker,
):
    print("--- Starting test_start_discovery_with_proper_patching ---")
    MockListenerClass, mock_listener_instance, actual_listener_start_method_spy = (
        mock_actual_instance_listener_fixture
    )
    mock_dh_client = mock_discovery_host_client_fixture

    # Patch run_on_event_loop where it is looked up by DiscoveryHost
    mock_run_patch_obj = mocker.patch(
        "tsercom.discovery.discovery_host.run_on_event_loop"
    )
    mock_run_patch_obj.side_effect = mock_run_on_event_loop_replacement_v3
    print(
        f"  Patch applied for tsercom.discovery.discovery_host.run_on_event_loop. Mock object: {mock_run_patch_obj}, Side effect set to v3."
    )
    _created_tasks_for_run_on_event_loop.clear() # Clear before use

    SERVICE_TYPE = "_test_service_type_final_v3._tcp.local."
    # DiscoveryHost will try to import InstanceListener from tsercom.discovery.mdns.instance_listener
    # which is what mock_actual_instance_listener_fixture patches.
    host = DiscoveryHost(service_type=SERVICE_TYPE)
    print(
        f"  DiscoveryHost instance created with service_type: {SERVICE_TYPE}"
    )

    # DiscoveryHost.start_discovery is synchronous and schedules __start_discovery_impl.
    # The mocked run_on_event_loop_replacement handles the execution of __start_discovery_impl.
    # host.start_discovery() is sync but schedules __start_discovery_impl via the mocked run_on_event_loop.
    # The mock run_on_event_loop (mock_run_patch_obj) returns a future. We need to await this
    # future to ensure __start_discovery_impl completes before we check its side effects.
    host.start_discovery(mock_dh_client)
    print("  Call to host.start_discovery has been made.")
    
    # Await the future returned by our mock_run_on_event_loop_replacement_sync
    # This ensures that the scheduled __start_discovery_impl coroutine has finished.
    assert mock_run_patch_obj.called, "run_on_event_loop mock was not called"
    
    # Await all tasks created by our mock
    # Use `len()` directly as it's standard Python
    print(f"  Awaiting {len(_created_tasks_for_run_on_event_loop)} tasks created by mock_run_on_event_loop...")
    for task_idx, task in enumerate(_created_tasks_for_run_on_event_loop):
        print(f"    Awaiting task {task_idx}: {task}")
        await task
        print(f"    Task {task_idx} completed.")
    _created_tasks_for_run_on_event_loop.clear()
    print("  All created tasks awaited and list cleared.")

    print("  Checking assertions for start_discovery...")
    mock_run_patch_obj.assert_called_once() # This should still be true
    print("  Assertion: mock_run_patch_obj.assert_called_once() - PASSED")

    # Corrected assertion: InstanceListener is called with (DiscoveryHost_instance, service_type)
    MockListenerClass.assert_called_once_with(host, SERVICE_TYPE)
    print(
        f"  Assertion: MockListenerClass.assert_called_once_with(host, '{SERVICE_TYPE}') - PASSED"
    )

    # This assertion depends on __start_discovery_impl awaiting listener_instance.start(host)
    # We assert against the spy mock now.
    actual_listener_start_method_spy.assert_called_once_with(host)
    print(
        f"  Assertion: actual_listener_start_method_spy.assert_called_once_with(host_instance) - PASSED"
    )

    assert host._DiscoveryHost__client is mock_dh_client
    print(
        f"  Assertion: host._DiscoveryHost__client is mock_dh_client - PASSED"
    )

    # Ensure the underlying async method was called by our mock
    # This relies on the mock_run_on_event_loop_replacement correctly calling the partial.
    # We can check if the __discoverer was set up, which happens in __start_discovery_impl.
    assert host._DiscoveryHost__discoverer is not None # Check if __start_discovery_impl was run
    print(
        "  Assertion: host._DiscoveryHost__discoverer is not None (implies __start_discovery_impl ran) - PASSED"
    )

    print(
        "--- test_start_discovery_with_proper_patching finished successfully ---"
    )


# 7. Test for _on_service_added
@pytest.mark.asyncio
async def test_on_service_added_behavior(
    mock_actual_instance_listener_fixture,
    mock_discovery_host_client_fixture,
    mock_caller_identifier_random_fixture, # This now yields a tuple
    mocker,
):
    print("--- Starting test_on_service_added_behavior ---")
    _, _ = mock_actual_instance_listener_fixture  # Needed for start_discovery
    mock_dh_client = mock_discovery_host_client_fixture
    # mock_caller_identifier_random_fixture now yields (mock_method, expected_instance)
    mock_random_method, expected_caller_id = mock_caller_identifier_random_fixture

    # Patch run_on_event_loop for this test's context as well
    mock_run_patch_obj_for_add = mocker.patch(
        "tsercom.discovery.discovery_host.run_on_event_loop"
    )
    mock_run_patch_obj_for_add.side_effect = mock_run_on_event_loop_replacement_v3
    print(
        f"  Patch for run_on_event_loop in _on_service_added test. Mock: {mock_run_patch_obj_for_add}, Side effect set to v3."
    )
    _created_tasks_for_run_on_event_loop.clear() # Clear before use

    SERVICE_TYPE_ADD_TEST = "_test_add_service_final_v3._tcp.local."
    host = DiscoveryHost(service_type=SERVICE_TYPE_ADD_TEST)
    print(
        f"  DiscoveryHost created for _on_service_added test with type: {SERVICE_TYPE_ADD_TEST}"
    )

    host.start_discovery(mock_dh_client) 
    print("  host.start_discovery called for _on_service_added test setup.")

    # Await the future from the mock run_on_event_loop to ensure __start_discovery_impl runs
    assert mock_run_patch_obj_for_add.called
    # Await all tasks created by our mock for the setup phase
    # Use `len()` directly
    print(f"  Awaiting {len(_created_tasks_for_run_on_event_loop)} tasks for setup...")
    for task_idx, task in enumerate(_created_tasks_for_run_on_event_loop):
        print(f"    Awaiting setup task {task_idx}: {task}")
        await task
        print(f"    Setup task {task_idx} completed.")
    _created_tasks_for_run_on_event_loop.clear()
    print("  All setup tasks awaited and list cleared.")

    # Ensure the underlying async method was called by our mock
    assert host._DiscoveryHost__discoverer is not None # Check if __start_discovery_impl was run
    print(
        "  Assertion: host._DiscoveryHost__discoverer is not None (implies __start_discovery_impl ran for setup) - PASSED"
    )

    service_info = ServiceInfo(
        name="TestSvcAddedV3",
        port=3333,
        addresses=["10.3.3.3"],
        mdns_name="TestSvcAddedV3._test_add_service_final_v3._tcp.local.",
    )
    print(f"  ServiceInfo created: {service_info.name}")

    await host._on_service_added(service_info)
    print(f"  host._on_service_added() called with {service_info.name}")

    print("  Checking assertions for _on_service_added...")
    mock_random_method.assert_called_once() # Check the method was called
    # generated_id_instance is now expected_caller_id from the fixture
    print(
        "  Assertion: mock_random_method.assert_called_once() - PASSED"
    )

    mock_dh_client._on_service_added.assert_called_once_with(
        service_info, expected_caller_id # Use the expected instance from the fixture
    )
    print(
        "  Assertion: mock_dh_client._on_service_added called with correct args - PASSED"
    )

    assert (
        host._DiscoveryHost__caller_id_map[service_info.mdns_name] is expected_caller_id
    )
    print("  Assertion: CallerIdentifier cached correctly - PASSED")

    # Test caching
    mock_random_method.reset_mock() # Reset the method mock
    mock_dh_client._on_service_added.reset_mock()
    updated_service_info = ServiceInfo(
        name="TestSvcAddedV3Updated",
        port=4444,
        addresses=["10.4.4.4"],
        mdns_name="TestSvcAddedV3._test_add_service_final_v3._tcp.local.",  # SAME mdns_name
    )
    await host._on_service_added(updated_service_info)
    mock_random_method.assert_not_called() # random should not be called again
    print(
        "  Assertion: mock_random_method.assert_not_called() for cached service - PASSED"
    )
    mock_dh_client._on_service_added.assert_called_once_with(
        updated_service_info, expected_caller_id # Should still use the cached (original) ID
    )
    print(
        "  Assertion: mock_dh_client._on_service_added called with updated info and CACHED ID - PASSED"
    )
    print("--- test_on_service_added_behavior finished successfully ---")
