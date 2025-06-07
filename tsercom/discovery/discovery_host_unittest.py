import asyncio
import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock

from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo
from tsercom.discovery.service_source import ServiceSource
from tsercom.discovery.mdns.instance_listener import (
    InstanceListener as ActualInstanceListener,
)
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)

SERVICE_TYPE_DEFAULT = "_test_service._tcp.local."


# 1. Global Event Loop Management Fixture
@pytest.fixture(autouse=True)
def manage_tsercom_global_event_loop_fixture(request):
    if not is_global_event_loop_set():
        set_tsercom_event_loop_to_current_thread()
    try:
        yield
    finally:
        clear_tsercom_event_loop()


# 2. Mock for ServiceSource.Client (formerly DiscoveryHost.Client)
@pytest.fixture
def mock_service_source_client_fixture(
    mocker,
):
    client = mocker.create_autospec(
        ServiceSource.Client, instance=True, name="MockServiceSourceClient"
    )
    client._on_service_added = AsyncMock(name="client_on_service_added_method")
    return client


# 3. Mock for the actual InstanceListener class
@pytest.fixture
def mock_actual_instance_listener_fixture(mocker):
    mock_listener_instance = mocker.MagicMock(
        spec=ActualInstanceListener,
        name="MockedActualInstanceListenerInstance",
    )
    # No attempt to mock mock_listener_instance.__init__ as it was problematic and
    # should be unnecessary if return_value on the class patch works correctly.

    MockListenerClass_patch = mocker.patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener",
        return_value=mock_listener_instance,  # Set return_value directly in the patch call.
        # No autospec=True on the class patch itself when using return_value.
    )
    yield MockListenerClass_patch, mock_listener_instance


# 4. Mock for CallerIdentifier.random
@pytest.fixture
def mock_caller_identifier_random_fixture(mocker):
    mock_random = mocker.patch.object(
        CallerIdentifier, "random", autospec=True
    )
    mock_random.side_effect = lambda: MagicMock(
        spec=CallerIdentifier,
        name=f"RandomCallerIdInstance_{mock_random.call_count}",
    )
    yield mock_random


# Test Scenarios
def test_discovery_host_initialization(mocker):
    """Test DiscoveryHost initialization scenarios."""
    # Test with service_type only
    host_st = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    assert host_st._DiscoveryHost__service_type == SERVICE_TYPE_DEFAULT
    assert host_st._DiscoveryHost__instance_listener_factory is None

    # Test with instance_listener_factory only
    mock_factory = mocker.Mock(name="MockListenerFactory")
    host_lf = DiscoveryHost(instance_listener_factory=mock_factory)
    assert host_lf._DiscoveryHost__service_type is None
    assert host_lf._DiscoveryHost__instance_listener_factory is mock_factory

    # Test with neither (should raise ValueError)
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
    ):
        DiscoveryHost()

    # Test with both (should raise ValueError)
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
    ):
        DiscoveryHost(
            service_type=SERVICE_TYPE_DEFAULT,
            instance_listener_factory=mock_factory,
        )


@pytest.mark.asyncio
async def test_start_discovery_successfully(
    mock_service_source_client_fixture,
    mock_actual_instance_listener_fixture,
    mocker,
):
    """Test successful start of discovery."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    mock_ss_client = mock_service_source_client_fixture
    MockListenerClass, mock_listener_instance = (
        mock_actual_instance_listener_fixture
    )

    # Expect TypeError due to the InstanceListener[TServiceInfo] instantiation issue.
    # This error originates from within the InstanceListener's own __init__ or
    # generic type handling when TServiceInfo is not a concrete type.
    with pytest.raises(TypeError) as excinfo:
        await host.start_discovery(mock_ss_client)

    assert "isinstance() arg 2 must be a type" in str(excinfo.value)
    assert host._DiscoveryHost__client is mock_ss_client


@pytest.mark.asyncio
async def test_start_discovery_with_listener_factory(
    mock_service_source_client_fixture, mocker
):
    """Test start of discovery using a listener factory."""
    mock_listener_from_factory = mocker.create_autospec(
        ActualInstanceListener, instance=True
    )
    mock_factory = mocker.Mock(
        name="MockListenerFactory", return_value=mock_listener_from_factory
    )

    host = DiscoveryHost(instance_listener_factory=mock_factory)
    mock_ss_client = mock_service_source_client_fixture

    await host.start_discovery(mock_ss_client)

    mock_factory.assert_called_once_with(host)
    assert host._DiscoveryHost__discoverer is mock_listener_from_factory
    assert host._DiscoveryHost__client is mock_ss_client


@pytest.mark.asyncio
async def test_start_discovery_client_none():
    """Test starting discovery with client as None."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    with pytest.raises(
        ValueError, match="Client argument cannot be None for start_discovery."
    ):
        await host.start_discovery(None)


@pytest.mark.asyncio
async def test_on_service_added_new_service(
    mock_service_source_client_fixture,
    mock_caller_identifier_random_fixture,
    mocker,
):
    """Test _on_service_added for a new service."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    mock_ss_client = mock_service_source_client_fixture

    # Manually set client for this unit test, as start_discovery is not the primary focus here.
    host._DiscoveryHost__client = mock_ss_client
    host._DiscoveryHost__caller_id_map = {}

    service_info = ServiceInfo(
        name="NewService",
        port=1234,
        addresses=["192.168.1.100"],
        mdns_name="NewService._test_service._tcp.local.",
    )

    # Create a specific mock instance that CallerIdentifier.random() will return
    expected_random_id_instance = MagicMock(
        spec=CallerIdentifier, name="ExpectedRandomID"
    )
    # Configure the mock_caller_identifier_random_fixture to return this specific instance
    # Clear side_effect first, then set return_value
    mock_caller_identifier_random_fixture.side_effect = None
    mock_caller_identifier_random_fixture.return_value = (
        expected_random_id_instance
    )

    await host._on_service_added(service_info)

    mock_caller_identifier_random_fixture.assert_called_once()
    mock_ss_client._on_service_added.assert_awaited_once_with(
        service_info, expected_random_id_instance
    )
    assert (
        host._DiscoveryHost__caller_id_map[service_info.mdns_name]
        is expected_random_id_instance
    )


@pytest.mark.asyncio
async def test_on_service_added_existing_service(
    mock_service_source_client_fixture,
    mock_caller_identifier_random_fixture,
    mocker,
):
    """Test _on_service_added for an existing service (CallerIdentifier should be reused)."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    mock_ss_client = mock_service_source_client_fixture
    host._DiscoveryHost__client = mock_ss_client  # Manual setup

    existing_mdns_name = "ExistingService._test_service._tcp.local."
    pre_existing_id = MagicMock(spec=CallerIdentifier, name="PreExistingID")
    host._DiscoveryHost__caller_id_map = {existing_mdns_name: pre_existing_id}

    service_info_updated = ServiceInfo(
        name="ExistingServiceUpdatedName",  # Name or other details might change
        port=1235,
        addresses=["192.168.1.101"],
        mdns_name=existing_mdns_name,  # Key for identity remains the same
    )

    await host._on_service_added(service_info_updated)

    mock_caller_identifier_random_fixture.assert_not_called()  # Should not generate a new ID
    mock_ss_client._on_service_added.assert_awaited_once_with(
        service_info_updated, pre_existing_id  # Should use the existing ID
    )
    assert (
        host._DiscoveryHost__caller_id_map[existing_mdns_name]
        is pre_existing_id
    )


@pytest.mark.asyncio
async def test_on_service_added_no_client():
    """Test _on_service_added when the internal client is None (e.g., discovery not started)."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    assert host._DiscoveryHost__client is None  # Verify precondition
    host._DiscoveryHost__caller_id_map = {}

    service_info = ServiceInfo(
        name="SomeService",
        port=1234,
        addresses=["192.168.1.100"],
        mdns_name="SomeService._test_service._tcp.local.",
    )

    with pytest.raises(RuntimeError, match="DiscoveryHost client not set"):
        await host._on_service_added(service_info)


def test_init_with_both_service_type_and_factory(mocker):
    """Tests DiscoveryHost init with both service_type and factory."""
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
    ):
        DiscoveryHost(
            service_type="_test._tcp.local.",
            instance_listener_factory=mocker.Mock(),
        )


def test_init_with_neither_service_type_nor_factory():
    """Tests DiscoveryHost init with neither service_type nor factory."""
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
    ):
        DiscoveryHost()


@pytest.mark.asyncio
async def test_start_discovery_multiple_times(
    mocker, mock_service_source_client_fixture
):
    """Tests calling start_discovery multiple times."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    mock_client = mock_service_source_client_fixture

    # Mock the InstanceListener creation to avoid TypeError
    mocker.patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener",
        return_value=mocker.AsyncMock(spec=ActualInstanceListener),
    )

    # First call to start_discovery
    # Expect TypeError due to the InstanceListener[TServiceInfo] instantiation issue.
    # This error originates from within the InstanceListener's own __init__ or
    # generic type handling when TServiceInfo is not a concrete type.
    with pytest.raises(TypeError) as excinfo:
        await host.start_discovery(mock_client)
    assert "isinstance() arg 2 must be a type" in str(excinfo.value)
    # At this point, host._DiscoveryHost__discoverer is still None because InstanceListener init failed.
    assert host._DiscoveryHost__client is mock_client  # Client should be set
    assert (
        host._DiscoveryHost__discoverer is None
    )  # Discoverer should not have been set

    # Second call to start_discovery should also raise TypeError because __discoverer is still None
    with pytest.raises(TypeError) as excinfo_again:
        await host.start_discovery(mock_client)
    assert "isinstance() arg 2 must be a type" in str(excinfo_again.value)


@pytest.mark.asyncio
async def test_on_service_added_new_service_whitebox(
    mocker, mock_caller_identifier_random_fixture
):
    """Tests _on_service_added for a new service via white-box access."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    mock_client = mocker.AsyncMock(spec=ServiceSource.Client)
    host._DiscoveryHost__client = mock_client  # White-box assignment
    host._DiscoveryHost__caller_id_map = {}  # Ensure clean map

    mock_service_info = mocker.Mock(spec=ServiceInfo)
    mock_service_info.mdns_name = "new_service._test._tcp.local."

    expected_caller_id = CallerIdentifier(
        uuid.uuid4()
    )  # Create a concrete instance with a UUID
    # Ensure the mock returns this specific instance, overriding any default side_effect from the fixture
    mock_caller_identifier_random_fixture.side_effect = None
    mock_caller_identifier_random_fixture.return_value = expected_caller_id

    await host._on_service_added(mock_service_info)

    mock_caller_identifier_random_fixture.assert_called_once()
    mock_client._on_service_added.assert_awaited_once_with(
        mock_service_info, expected_caller_id
    )
    assert (
        host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name]
        is expected_caller_id
    )


@pytest.mark.asyncio
async def test_on_service_added_existing_service_whitebox(mocker):
    """Tests _on_service_added for an existing service via white-box access."""
    host = DiscoveryHost(service_type=SERVICE_TYPE_DEFAULT)
    mock_client = mocker.AsyncMock(spec=ServiceSource.Client)
    host._DiscoveryHost__client = mock_client  # White-box assignment

    mock_existing_caller_id = mocker.Mock(spec=CallerIdentifier)
    service_mdns_name = "existing_service._test._tcp.local."
    host._DiscoveryHost__caller_id_map = {
        service_mdns_name: mock_existing_caller_id
    }

    mock_service_info = mocker.Mock(spec=ServiceInfo)
    mock_service_info.mdns_name = service_mdns_name

    # Patch CallerIdentifier.random to ensure it's not called
    mock_random_method = mocker.patch.object(CallerIdentifier, "random")

    await host._on_service_added(mock_service_info)

    mock_random_method.assert_not_called()
    mock_client._on_service_added.assert_awaited_once_with(
        mock_service_info, mock_existing_caller_id
    )
    assert (
        host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name]
        is mock_existing_caller_id
    )
