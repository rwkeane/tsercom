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
import typing  # For Any, Callable etc.
from unittest.mock import patch, Mock  # Added for clarity if used directly
from tsercom.discovery.mdns.instance_listener import MdnsListenerFactory
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)

SERVICE_TYPE_DEFAULT = "_test_service._tcp.local."


# 1. Global Event Loop Management Fixture
@pytest.fixture(autouse=True)
def manage_tsercom_global_event_loop_fixture(
    request: pytest.FixtureRequest,
) -> typing.Iterator[None]:
    if not is_global_event_loop_set():
        set_tsercom_event_loop_to_current_thread()
    try:
        yield
    finally:
        clear_tsercom_event_loop()


# 2. Mock for ServiceSource.Client (formerly DiscoveryHost.Client)
@pytest.fixture
def mock_service_source_client_fixture(
    mocker: Mock,
) -> AsyncMock:
    client: AsyncMock = mocker.create_autospec(
        ServiceSource.Client, instance=True, name="MockServiceSourceClient"
    )
    client._on_service_added = AsyncMock(name="client_on_service_added_method")
    return client


# 3. Mock for the actual InstanceListener class
@pytest.fixture
def mock_actual_instance_listener_fixture(
    mocker: Mock,
) -> typing.Iterator[typing.Tuple[MagicMock, MagicMock]]:
    mock_listener_instance: MagicMock = mocker.MagicMock(
        spec=ActualInstanceListener,
        name="MockedActualInstanceListenerInstance",
    )
    # No attempt to mock mock_listener_instance.__init__ as it was problematic and
    # should be unnecessary if return_value on the class patch works correctly.

    MockListenerClass_patch: MagicMock = mocker.patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener",
        return_value=mock_listener_instance,  # Set return_value directly in the patch call.
        # No autospec=True on the class patch itself when using return_value.
    )
    yield MockListenerClass_patch, mock_listener_instance


# 4. Mock for CallerIdentifier.random
@pytest.fixture
def mock_caller_identifier_random_fixture(
    mocker: Mock,
) -> typing.Iterator[MagicMock]:
    mock_random: MagicMock = mocker.patch.object(
        CallerIdentifier, "random", autospec=True
    )
    mock_random.side_effect = lambda: MagicMock(
        spec=CallerIdentifier,
        name=f"RandomCallerIdInstance_{mock_random.call_count}",
    )
    yield mock_random


# Removed test_start_discovery_successfully as per plan


@pytest.mark.asyncio
async def test_start_discovery_with_listener_factory(
    mock_service_source_client_fixture: AsyncMock, mocker: Mock
) -> None:
    """Test start of discovery using a listener factory."""
    mock_listener_from_factory: MagicMock = mocker.create_autospec(
        ActualInstanceListener, instance=True
    )
    # Define the expected factory type for casting
    ExpectedInstanceListenerFactory = typing.Callable[
        [ActualInstanceListener.Client], ActualInstanceListener[ServiceInfo]
    ]
    mock_factory: MagicMock = mocker.Mock(
        name="MockListenerFactory", return_value=mock_listener_from_factory
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        instance_listener_factory=typing.cast(
            ExpectedInstanceListenerFactory, mock_factory
        )
    )
    mock_ss_client: AsyncMock = mock_service_source_client_fixture

    await host.start_discovery(mock_ss_client)

    mock_factory.assert_called_once_with(host)
    assert host._DiscoveryHost__discoverer is mock_listener_from_factory  # type: ignore[attr-defined]
    assert host._DiscoveryHost__client is mock_ss_client  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_start_discovery_client_none() -> None:
    """Test starting discovery with client as None."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    with pytest.raises(
        ValueError, match="Client argument cannot be None for start_discovery."
    ):
        await host.start_discovery(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_on_service_added_new_service(
    mock_service_source_client_fixture: AsyncMock,
    mock_caller_identifier_random_fixture: MagicMock,
    mocker: Mock,
) -> None:
    """Test _on_service_added for a new service."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_ss_client: AsyncMock = mock_service_source_client_fixture

    # Manually set client for this unit test, as start_discovery is not the primary focus here.
    host._DiscoveryHost__client = mock_ss_client  # type: ignore[attr-defined]
    host._DiscoveryHost__caller_id_map = {}  # type: ignore[attr-defined]

    service_info: ServiceInfo = ServiceInfo(
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
    assert host._DiscoveryHost__caller_id_map[service_info.mdns_name] is expected_random_id_instance  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_service_added_existing_service(
    mock_service_source_client_fixture: AsyncMock,
    mock_caller_identifier_random_fixture: MagicMock,
    mocker: Mock,
) -> None:
    """Test _on_service_added for an existing service (CallerIdentifier should be reused)."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_ss_client: AsyncMock = mock_service_source_client_fixture
    host._DiscoveryHost__client = mock_ss_client  # type: ignore[attr-defined] # Manual setup

    existing_mdns_name: str = "ExistingService._test_service._tcp.local."
    pre_existing_id: MagicMock = MagicMock(
        spec=CallerIdentifier, name="PreExistingID"
    )
    host._DiscoveryHost__caller_id_map = {existing_mdns_name: pre_existing_id}  # type: ignore[attr-defined]

    service_info_updated: ServiceInfo = ServiceInfo(
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
    assert host._DiscoveryHost__caller_id_map[existing_mdns_name] is pre_existing_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_service_added_no_client() -> None:
    """Test _on_service_added when the internal client is None (e.g., discovery not started)."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    assert host._DiscoveryHost__client is None  # type: ignore[attr-defined] # Verify precondition
    host._DiscoveryHost__caller_id_map = {}  # type: ignore[attr-defined]

    service_info: ServiceInfo = ServiceInfo(
        name="SomeService",
        port=1234,
        addresses=["192.168.1.100"],
        mdns_name="SomeService._test_service._tcp.local.",
    )

    with pytest.raises(RuntimeError, match="DiscoveryHost client not set"):
        await host._on_service_added(service_info)


def test_init_with_both_service_type_and_factory(mocker: Mock) -> None:
    """Tests DiscoveryHost init with both service_type and factory."""
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
    ):
        # Define the expected factory type for casting
        ExpectedInstanceListenerFactory = typing.Callable[
            [ActualInstanceListener.Client],
            ActualInstanceListener[ServiceInfo],
        ]
        DiscoveryHost[ServiceInfo](  # type: ignore[call-overload]
            service_type="_test._tcp.local.",
            instance_listener_factory=typing.cast(
                ExpectedInstanceListenerFactory, mocker.Mock()
            ),
        )


def test_init_with_neither_service_type_nor_factory() -> None:
    """Tests DiscoveryHost init with neither service_type nor factory."""
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
    ):
        DiscoveryHost[ServiceInfo]()  # type: ignore[call-overload]


@pytest.mark.asyncio
async def test_start_discovery_multiple_times(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    """Tests calling start_discovery multiple times."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_client: AsyncMock = mock_service_source_client_fixture

    # Mock the InstanceListener creation to avoid TypeError
    # This test needs to be rewritten based on new behavior.
    # For now, let's mock InstanceListener to return a functional mock
    # that doesn't raise TypeError, so we can test the "already started" logic.

    mock_listener_instance: AsyncMock = mocker.AsyncMock(
        spec=ActualInstanceListener
    )
    mock_listener_instance.start = AsyncMock()  # ensure it has a start method

    with patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener",
        return_value=mock_listener_instance,
    ) as MockedInstanceListenerClass:
        # First call: successful
        await host.start_discovery(mock_client)
        assert host._DiscoveryHost__discoverer is mock_listener_instance  # type: ignore[attr-defined]
        MockedInstanceListenerClass.assert_called_once()
        # Ensure the internal factory called InstanceListener with the correct args
        # The factory DiscoveryHost creates calls InstanceListener(self, service_type, mdns_listener_factory=None)
        # 'self' here is the host instance, which is the client for InstanceListener
        # print(MockedInstanceListenerClass.call_args_list)
        assert MockedInstanceListenerClass.call_args[0][0] is host  # client
        assert (
            MockedInstanceListenerClass.call_args[0][1] == SERVICE_TYPE_DEFAULT
        )  # service_type
        # mdns_listener_factory is not passed by default if service_type is used
        assert (
            MockedInstanceListenerClass.call_args[1].get(
                "mdns_listener_factory"
            )
            is None
        )

        # Second call: should raise RuntimeError because discovery is already started
        with pytest.raises(
            RuntimeError, match="Discovery has already been started."
        ):
            await host.start_discovery(mock_client)

        # Ensure InstanceListener was not called again for the second attempt
        MockedInstanceListenerClass.assert_called_once()


@pytest.mark.asyncio
async def test_on_service_added_new_service_whitebox(
    mocker: Mock, mock_caller_identifier_random_fixture: MagicMock
) -> None:
    """Tests _on_service_added for a new service via white-box access."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_client: AsyncMock = mocker.AsyncMock(spec=ServiceSource.Client)
    host._DiscoveryHost__client = mock_client  # type: ignore[attr-defined] # White-box assignment
    host._DiscoveryHost__caller_id_map = {}  # type: ignore[attr-defined] # Ensure clean map

    mock_service_info: MagicMock = mocker.Mock(spec=ServiceInfo)
    mock_service_info.mdns_name = "new_service._test._tcp.local."

    expected_caller_id: CallerIdentifier = CallerIdentifier(
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
    assert host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name] is expected_caller_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_service_added_existing_service_whitebox(
    mocker: Mock,
) -> None:
    """Tests _on_service_added for an existing service via white-box access."""
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_client: AsyncMock = mocker.AsyncMock(spec=ServiceSource.Client)
    host._DiscoveryHost__client = mock_client  # type: ignore[attr-defined] # White-box assignment

    mock_existing_caller_id: MagicMock = mocker.Mock(spec=CallerIdentifier)
    service_mdns_name: str = "existing_service._test._tcp.local."
    host._DiscoveryHost__caller_id_map = {  # type: ignore[attr-defined]
        service_mdns_name: mock_existing_caller_id
    }

    mock_service_info: MagicMock = mocker.Mock(spec=ServiceInfo)
    mock_service_info.mdns_name = service_mdns_name

    # Patch CallerIdentifier.random to ensure it's not called
    mock_random_method: MagicMock = mocker.patch.object(
        CallerIdentifier, "random"
    )

    await host._on_service_added(mock_service_info)

    mock_random_method.assert_not_called()
    mock_client._on_service_added.assert_awaited_once_with(
        mock_service_info, mock_existing_caller_id
    )
    assert host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name] is mock_existing_caller_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mdns_listener_factory_invoked_via_instance_listener_on_start(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    """
    Tests that DiscoveryHost, when given a service_type and mdns_listener_factory,
    correctly passes the factory to the internally created InstanceListener,
    and that this factory is invoked, and its product (a listener) has its start() method called.
    """
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture

    # This is the mock listener that our mdns_listener_factory will return
    mock_listener_product: MagicMock = MagicMock(
        spec=ActualInstanceListener
    )  # Using ActualInstanceListener for spec
    mock_listener_product.start = MagicMock()

    # This is our custom mdns_listener_factory
    # TODO: Fix typing for mdns_listener_factory to satisfy MyPy call-overload
    mock_mdns_factory: MagicMock = MagicMock(
        return_value=mock_listener_product
    )

    # This is the mock for the InstanceListener instance that DiscoveryHost's internal factory will create.
    # The key is to ensure that *this* InstanceListener, when created, uses our mock_mdns_factory.
    # mock_instance_listener_created_by_host_factory = MagicMock(
    #     spec=ActualInstanceListener
    # ) # This variable was unused.
    # We don't need this mock_instance_listener_created_by_host_factory to *do* anything complex,
    # as the core logic (calling mdns_listener_factory and listener.start()) is what we are testing,
    # which happens *inside* the *actual* InstanceListener's __init__ if not mocked too deeply.

    # We need to patch the InstanceListener class that DiscoveryHost uses.
    # When DiscoveryHost creates its internal `instance_factory`, this factory then calls
    # `InstanceListener(client, service_type, mdns_listener_factory=our_factory)`.
    # The default __init__ of the *actual* InstanceListener should then call `our_factory`.

    # To achieve this, we will let the actual InstanceListener be created,
    # but we need to ensure *its* __init__ method correctly calls the mdns_listener_factory
    # and then calls start() on the listener product.
    # The most straightforward way to verify this without overly complex mocking of InstanceListener itself
    # is to rely on the fact that if `mdns_listener_factory` is passed to InstanceListener's constructor,
    # it *will* be called, and its product's `start` method will be called.
    # The `DiscoveryHost` modification ensures that `__discoverer` is the result of this.

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(  # type: ignore[call-overload]
        service_type="test.service",
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_mdns_factory
        ),
    )

    # We need to mock the InstanceListener class that DiscoveryHost's factory instantiates.
    # The instance created by this mock will become host._DiscoveryHost__discoverer
    # The crucial part is that the *constructor* of the *actual* InstanceListener
    # is what invokes our mdns_listener_factory.
    # So, we don't want to mock away the InstanceListener constructor's logic entirely.
    # Instead, we assert the outcome: mock_mdns_factory is called, start is called on its product.
    # And host.__discoverer is set (or not, in case of error).

    # The previous change to DiscoveryHost wrapped the factory call in a try-except.
    # The instance_listener_factory in DiscoveryHost for the service_type case is:
    #   def instance_factory(client) -> InstanceListener[ServiceInfoT]:
    #       return InstanceListener[ServiceInfoT](
    #           client, service_type, mdns_listener_factory=mdns_listener_factory
    #       )
    # This `InstanceListener` is what we need to interact with.

    # Spy on the actual InstanceListener's __init__ method to verify it's called correctly
    # and to control its created instance for assertion.
    # However, the logic we want to test (factory call, listener.start()) is *inside* __init__.

    # Let's use a side effect on the InstanceListener class itself.
    # When DiscoveryHost calls `InstanceListener(self, "test.service", mock_mdns_factory)`,
    # this side_effect will run.

    original_instance_listener_init: typing.Callable[..., None] = (
        ActualInstanceListener.__init__
    )
    # instance_listener_init_spy = MagicMock( # This variable was unused.
    #     wraps=original_instance_listener_init
    # )

    # This will hold the actual instance created by the *real* InstanceListener init
    created_instance_holder: typing.Dict[
        str, ActualInstanceListener[ServiceInfo]
    ] = {}

    def init_side_effect(
        actual_self: ActualInstanceListener[ServiceInfo],
        client: ServiceSource.Client,
        service_type: str,
        mdns_listener_factory: typing.Optional[MdnsListenerFactory] = None,
    ) -> None:
        # Call the original __init__
        original_instance_listener_init(
            actual_self,
            client,
            service_type,
            mdns_listener_factory=mdns_listener_factory,
        )
        # Store the created instance so we can check if host.__discoverer is it.
        created_instance_holder["instance"] = actual_self
        # The original __init__ should have called our mock_mdns_factory and its product's start method.

    with patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener.__init__",
        side_effect=init_side_effect,
        autospec=True,
    ) as mock_il_init:
        await host.start_discovery(mock_service_source_client)

        mock_il_init.assert_called_once()
        # Check args passed to InstanceListener.__init__ by DiscoveryHost's internal factory
        # args[0] is 'self' of InstanceListener, args[1] is client (DiscoveryHost instance), args[2] is service_type
        # kwargs['mdns_listener_factory'] is our factory
        call_args = mock_il_init.call_args
        assert (
            call_args[0][1] is host
        )  # client passed to InstanceListener is the DiscoveryHost instance
        assert call_args[0][2] == "test.service"  # service_type
        assert call_args[1]["mdns_listener_factory"] is mock_mdns_factory

        # Now check if our mdns_factory was called by the original InstanceListener.__init__
        mock_mdns_factory.assert_called_once()
        # The InstanceListener passes itself (actual_self from init_side_effect) as the MdnsListener.Client
        # and the service_type.
        assert mock_mdns_factory.call_args[0][
            0
        ] is created_instance_holder.get(
            "instance"
        )  # client is the InstanceListener instance
        assert (
            mock_mdns_factory.call_args[0][1] == "test.service"
        )  # service_type

        # Check that the product of the factory (mock_listener_product) had its start method called
        mock_listener_product.start.assert_called_once()

        # Check that host.__discoverer is the instance of InstanceListener created
        assert (
            host._DiscoveryHost__discoverer
            is created_instance_holder.get(  # type: ignore[attr-defined]
                "instance"
            )
        )
        assert (
            host._DiscoveryHost__discoverer is not None  # type: ignore[attr-defined]
        )


@pytest.mark.asyncio
async def test_discovery_host_handles_mdns_factory_exception_gracefully(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    """
    Tests that DiscoveryHost handles exceptions raised by the mdns_listener_factory gracefully.
    The __discoverer should be None, and an error should be logged (mocked).
    """
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture

    # This factory will raise an exception when called
    # TODO: Fix typing for mdns_listener_factory to satisfy MyPy call-overload
    mock_failing_mdns_factory: MagicMock = MagicMock(
        side_effect=RuntimeError("Factory boom!")
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(  # type: ignore[call-overload]
        service_type="test.service.fail.factory",
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_failing_mdns_factory
        ),
    )

    # The actual InstanceListener.__init__ will call our failing factory.
    # The try-except in DiscoveryHost.__start_discovery_impl should catch this.

    # We need to ensure InstanceListener's __init__ is actually called with our factory.
    # We can patch logging.error to check it was called.
    with patch("logging.error") as mock_log_error:
        # We don't need to mock InstanceListener here, as the error occurs when
        # DiscoveryHost's internal instance_factory calls our mock_failing_mdns_factory
        # *through* the InstanceListener's constructor.
        # The exception occurs inside `self.__instance_listener_factory(self)`
        # which is `InstanceListener(...)` effectively.
        # That InstanceListener's init calls mdns_listener_factory.
        # That call to mdns_listener_factory is what raises.
        # This exception is then caught by DiscoveryHost's try-except.

        await host.start_discovery(mock_service_source_client)

        mock_failing_mdns_factory.assert_called_once()
        # The client passed to the factory by InstanceListener is the InstanceListener itself.
        # We can't easily get a handle to that specific InstanceListener instance if its init fails
        # partway through due to the factory error.
        # However, we know the factory *was* called.
        assert (
            mock_failing_mdns_factory.call_args[0][1]
            == "test.service.fail.factory"
        )  # service_type

        assert host._DiscoveryHost__discoverer is None  # type: ignore[attr-defined] # Crucial check
        mock_log_error.assert_called_once()
        assert (
            "Failed to initialize discovery listener: Factory boom!"
            in mock_log_error.call_args[0][0]
        )


@pytest.mark.asyncio
async def test_discovery_host_handles_listener_start_exception_gracefully(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    """
    Tests that DiscoveryHost handles exceptions from listener.start() gracefully.
    The __discoverer should be None, and an error logged.
    """
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture

    # Factory returns a listener whose start() method fails
    mock_listener_product_failing_start: MagicMock = MagicMock(
        spec=ActualInstanceListener
    )
    mock_listener_product_failing_start.start = MagicMock(
        side_effect=RuntimeError("Listener start boom!")
    )

    # TODO: Fix typing for mdns_listener_factory to satisfy MyPy call-overload
    mock_mdns_factory_good_product_bad_start: MagicMock = MagicMock(
        return_value=mock_listener_product_failing_start
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(  # type: ignore[call-overload]
        service_type="test.service.fail.start",
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_mdns_factory_good_product_bad_start
        ),
    )

    # Similar to the factory exception, the error from listener.start() (called within
    # InstanceListener's __init__) should be caught by DiscoveryHost's try-except.
    with patch("logging.error") as mock_log_error:
        await host.start_discovery(mock_service_source_client)

        mock_mdns_factory_good_product_bad_start.assert_called_once()
        assert (
            mock_mdns_factory_good_product_bad_start.call_args[0][1]
            == "test.service.fail.start"
        )

        mock_listener_product_failing_start.start.assert_called_once()  # Ensure start was attempted

        assert host._DiscoveryHost__discoverer is None  # type: ignore[attr-defined] # Crucial check
        mock_log_error.assert_called_once()
        assert (
            "Failed to initialize discovery listener: Listener start boom!"
            in mock_log_error.call_args[0][0]
        )
