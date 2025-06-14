import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock

from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.discovery.service_info import ServiceInfo
from tsercom.discovery.service_source import ServiceSource
from tsercom.discovery.mdns.instance_listener import (
    InstanceListener as ActualInstanceListener,
    MdnsListenerFactory,
)
from tsercom.discovery.mdns.mdns_listener import MdnsListener  # Added import
import typing
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)

SERVICE_TYPE_DEFAULT = "_test_service._tcp.local."


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


@pytest.fixture
def mock_service_source_client_fixture(
    mocker: Mock,
) -> AsyncMock:
    client: AsyncMock = mocker.create_autospec(
        ServiceSource.Client, instance=True, name="MockServiceSourceClient"
    )
    client._on_service_added = AsyncMock(name="client_on_service_added_method")
    return client


@pytest.fixture
def mock_actual_instance_listener_fixture(
    mocker: Mock,
) -> typing.Iterator[typing.Tuple[MagicMock, MagicMock]]:
    mock_listener_instance: MagicMock = mocker.MagicMock(
        spec=ActualInstanceListener,
        name="MockedActualInstanceListenerInstance",
    )
    mock_listener_instance.start = AsyncMock(
        name="MockedActualInstanceListenerInstance.start"
    )
    mock_listener_instance.async_stop = AsyncMock(
        name="MockedActualInstanceListenerInstance.async_stop"
    )
    MockListenerClass_patch: MagicMock = mocker.patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener",
        return_value=mock_listener_instance,
    )
    yield MockListenerClass_patch, mock_listener_instance


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


@pytest.mark.asyncio
async def test_start_discovery_with_listener_factory(
    mock_service_source_client_fixture: AsyncMock, mocker: Mock
) -> None:
    mock_listener_from_factory: MagicMock = mocker.create_autospec(
        ActualInstanceListener, instance=True
    )
    ExpectedInstanceListenerFactory = typing.Callable[
        [ActualInstanceListener.Client], ActualInstanceListener[ServiceInfo]
    ]
    mock_factory: Mock = mocker.Mock(
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
    assert host._DiscoveryHost__discoverer is mock_listener_from_factory
    assert host._DiscoveryHost__client is mock_ss_client


@pytest.mark.asyncio
async def test_start_discovery_client_none() -> None:
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
) -> None:
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_ss_client: AsyncMock = mock_service_source_client_fixture
    host._DiscoveryHost__client = mock_ss_client  # type: ignore[attr-defined]
    host._DiscoveryHost__caller_id_map = {}  # type: ignore[attr-defined]
    service_info: ServiceInfo = ServiceInfo(
        name="NewService",
        port=1234,
        addresses=["192.168.1.100"],
        mdns_name="NewService._test_service._tcp.local.",
    )
    expected_random_id_instance = MagicMock(
        spec=CallerIdentifier, name="ExpectedRandomID"
    )
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
    )  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_service_added_existing_service(
    mock_service_source_client_fixture: AsyncMock,
    mock_caller_identifier_random_fixture: MagicMock,
) -> None:
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_ss_client: AsyncMock = mock_service_source_client_fixture
    host._DiscoveryHost__client = mock_ss_client  # type: ignore[attr-defined]
    existing_mdns_name: str = "ExistingService._test_service._tcp.local."
    pre_existing_id: MagicMock = MagicMock(
        spec=CallerIdentifier, name="PreExistingID"
    )
    host._DiscoveryHost__caller_id_map = {existing_mdns_name: pre_existing_id}  # type: ignore[attr-defined]
    service_info_updated: ServiceInfo = ServiceInfo(
        name="ExistingServiceUpdatedName",
        port=1235,
        addresses=["192.168.1.101"],
        mdns_name=existing_mdns_name,
    )
    await host._on_service_added(service_info_updated)
    mock_caller_identifier_random_fixture.assert_not_called()
    mock_ss_client._on_service_added.assert_awaited_once_with(
        service_info_updated, pre_existing_id
    )
    assert (
        host._DiscoveryHost__caller_id_map[existing_mdns_name]
        is pre_existing_id
    )  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_service_added_no_client() -> None:
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    assert host._DiscoveryHost__client is None  # type: ignore[attr-defined]
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
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type', 'instance_listener_factory', or 'mdns_listener_factory' must be provided.",
    ):
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
    with pytest.raises(
        ValueError,
        match="Exactly one of 'service_type', 'instance_listener_factory', or 'mdns_listener_factory' must be provided.",
    ):
        DiscoveryHost[ServiceInfo]()  # type: ignore[call-overload]


@pytest.mark.asyncio
async def test_start_discovery_multiple_times(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_client_arg: AsyncMock = (
        mock_service_source_client_fixture  # Renamed arg
    )

    mock_listener_instance: AsyncMock = mocker.AsyncMock(
        spec=ActualInstanceListener
    )

    # Patch the factory directly on the host instance
    host._DiscoveryHost__instance_listener_factory = (
        mocker.Mock(  # Use standard Mock
            return_value=mock_listener_instance,
            name="MockedInstanceListenerFactoryOnHostInstance",
        )
    )

    await host.start_discovery(mock_client_arg)  # Use renamed arg

    host._DiscoveryHost__instance_listener_factory.assert_called_once_with(
        host
    )

    assert host._DiscoveryHost__discoverer is mock_listener_instance  # type: ignore[attr-defined]

    with pytest.raises(
        RuntimeError, match="Discovery has already been started."
    ):
        await host.start_discovery(mock_client_arg)  # Use renamed arg

    host._DiscoveryHost__instance_listener_factory.assert_called_once()


@pytest.mark.asyncio
async def test_on_service_added_new_service_whitebox(
    mocker: Mock, mock_caller_identifier_random_fixture: MagicMock
) -> None:
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_client_internal: AsyncMock = mocker.AsyncMock(
        spec=ServiceSource.Client
    )
    host._DiscoveryHost__client = mock_client_internal  # type: ignore[attr-defined]
    host._DiscoveryHost__caller_id_map = {}  # type: ignore[attr-defined]

    mock_service_info: MagicMock = mocker.Mock(spec=ServiceInfo)
    mock_service_info.mdns_name = "new_service._test._tcp.local."

    expected_caller_id: CallerIdentifier = CallerIdentifier(uuid.uuid4())
    mock_caller_identifier_random_fixture.side_effect = None
    mock_caller_identifier_random_fixture.return_value = expected_caller_id

    await host._on_service_added(mock_service_info)

    mock_caller_identifier_random_fixture.assert_called_once()
    mock_client_internal._on_service_added.assert_awaited_once_with(
        mock_service_info, expected_caller_id
    )
    assert (
        host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name]
        is expected_caller_id
    )  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_on_service_added_existing_service_whitebox(
    mocker: Mock,
) -> None:
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=SERVICE_TYPE_DEFAULT
    )
    mock_client_internal: AsyncMock = mocker.AsyncMock(
        spec=ServiceSource.Client
    )
    host._DiscoveryHost__client = mock_client_internal  # type: ignore[attr-defined]

    mock_existing_caller_id: MagicMock = mocker.Mock(spec=CallerIdentifier)
    service_mdns_name: str = "existing_service._test._tcp.local."
    host._DiscoveryHost__caller_id_map = {  # type: ignore[attr-defined]
        service_mdns_name: mock_existing_caller_id
    }

    mock_service_info: MagicMock = mocker.Mock(spec=ServiceInfo)
    mock_service_info.mdns_name = service_mdns_name

    mock_random_method: MagicMock = mocker.patch.object(
        CallerIdentifier, "random"
    )

    await host._on_service_added(mock_service_info)

    mock_random_method.assert_not_called()
    mock_client_internal._on_service_added.assert_awaited_once_with(
        mock_service_info, mock_existing_caller_id
    )
    assert (
        host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name]
        is mock_existing_caller_id
    )  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mdns_listener_factory_invoked_via_instance_listener_on_start(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture

    # This mock will be the product of the mdns_listener_factory
    # This represents the underlying listener (e.g., RecordListener)
    mock_underlying_listener = MagicMock(spec=MdnsListener)
    mock_underlying_listener.start = MagicMock(
        name="mock_underlying_listener.start"
    )

    # This factory will be passed to DiscoveryHost, and then to InstanceListener
    mock_mdns_factory: Mock = mocker.Mock(
        return_value=mock_underlying_listener
    )

    # We will patch ActualInstanceListener.start to verify DiscoveryHost calls it.
    # The real ActualInstanceListener will be created by DiscoveryHost's internal factory.
    # Removing patch of ActualInstanceListener.start to let the real method run.
    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=None,
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_mdns_factory
        ),
    )
    await host.start_discovery(mock_service_source_client)

    # 1. Verify the mdns_listener_factory was called by the real InstanceListener's __init__
    mock_mdns_factory.assert_called_once()

    # Check arguments passed to the mdns_listener_factory
    assert mock_mdns_factory.call_args is not None
    # The first argument to the factory should be the InstanceListener instance itself
    assert isinstance(
        mock_mdns_factory.call_args.args[0], ActualInstanceListener
    )
    # The second argument should be the service type string
    assert (
        mock_mdns_factory.call_args.args[1] == "_internal_default._tcp.local."
    )

    # 2. Verify that the underlying listener's start method was called by the real InstanceListener.start()
    mock_underlying_listener.start.assert_called_once()

    assert host._DiscoveryHost__discoverer is not None
    assert isinstance(host._DiscoveryHost__discoverer, ActualInstanceListener)


@pytest.mark.asyncio
async def test_discovery_host_handles_mdns_factory_exception_gracefully(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture
    mock_failing_mdns_factory: Mock = mocker.Mock(
        side_effect=RuntimeError("Factory boom!")
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=None,  # Changed: Use mdns_listener_factory as the sole configuration
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_failing_mdns_factory
        ),
    )

    with patch("logging.error") as mock_log_error:
        await host.start_discovery(mock_service_source_client)
        mock_failing_mdns_factory.assert_called_once()
        assert (
            mock_failing_mdns_factory.call_args.args[1]
            == "_internal_default._tcp.local."
        )
        assert host._DiscoveryHost__discoverer is None  # type: ignore[attr-defined]
        mock_log_error.assert_called_once()
        assert (
            "Failed to initialize or start discovery listener: Factory boom!"
            in mock_log_error.call_args.args[0]
        )


@pytest.mark.asyncio
async def test_discovery_host_handles_listener_start_exception_gracefully(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture

    # This mock represents the InstanceListener that DiscoveryHost would create
    mock_instance_listener = AsyncMock(spec=ActualInstanceListener)
    mock_instance_listener.start = AsyncMock(
        side_effect=RuntimeError("Listener start boom!")
    )

    # The factory used by DiscoveryHost's internal instance_factory
    # This factory is for the *underlying* MdnsListener (e.g. RecordListener)
    # This test is for when InstanceListener.start() itself fails.
    # So, the mdns_listener_factory is not the one failing.
    # We need DiscoveryHost's instance_listener_factory to produce our mock_instance_listener.

    def specific_instance_listener_factory(
        client_arg,
    ):  # Matches ExpectedInstanceListenerFactory
        # client_arg will be the DiscoveryHost instance
        assert client_arg is host  # Ensure factory is called with host
        return mock_instance_listener

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        instance_listener_factory=specific_instance_listener_factory
    )

    with patch("logging.error") as mock_log_error:
        await host.start_discovery(mock_service_source_client)

        # Check that the factory was called
        # (This check is implicit in specific_instance_listener_factory)

        # Check that the created discoverer's start method was called
        assert (
            mock_instance_listener.start.called
        )  # Changed from assert_called_once()

        # Since start() fails, __discoverer should be set to None
        assert host._DiscoveryHost__discoverer is None
        mock_log_error.assert_called_once()
        assert (
            "Failed to initialize or start discovery listener: Listener start boom!"
            in mock_log_error.call_args.args[0]
        )


# Note: MdnsListenerFactory and DiscoveryHost should already be imported in the target file.
# If not, ruff/black might help, or they need to be added manually if this causes issues.


def test_discovery_host_init_with_only_mdns_factory(mocker):
    """
    Tests that DiscoveryHost can be initialized with only an mdns_listener_factory.
    This scenario (service_type=None, instance_listener_factory=None, mdns_listener_factory=mock)
    was causing a ValueError due to the boolean check.
    """
    # mocker fixture is used to create mock objects
    from tsercom.discovery.mdns.instance_listener import (
        MdnsListenerFactory,
    )  # Explicit import for clarity
    from tsercom.discovery.discovery_host import (
        DiscoveryHost,
    )  # Explicit import for clarity

    mock_mdns_listener_factory = mocker.MagicMock(spec=MdnsListenerFactory)

    try:
        # Explicitly pass None for the other two main factory/type args
        host = DiscoveryHost(
            service_type=None,
            instance_listener_factory=None,
            mdns_listener_factory=mock_mdns_listener_factory,
        )
        # If we reach here, the ValueError was not raised, which is the primary point.
        assert host is not None

        # Further checks for internal consistency:
        # The `_DiscoveryHost__instance_listener_factory` should be the *generated*
        # internal factory, not a directly passed `instance_listener_factory`.
        assert callable(host._DiscoveryHost__instance_listener_factory)
        # This internal factory should be the one named 'instance_factory' inside __init__
        assert (
            host._DiscoveryHost__instance_listener_factory.__name__
            == "instance_factory"
        )

    except ValueError as e:
        pytest.fail(
            "DiscoveryHost raised an unexpected ValueError when initialized "
            f"only with mdns_listener_factory (service_type=None, instance_listener_factory=None): {e}"
        )
