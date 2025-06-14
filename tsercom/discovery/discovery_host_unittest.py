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
) -> typing.Iterator[typing.Tuple[MagicMock, AsyncMock]]:
    mock_listener_instance: AsyncMock = mocker.create_autospec(
        ActualInstanceListener,
        instance=True,
        name="MockedActualInstanceListenerInstance",
    )
    mock_listener_instance.start = AsyncMock(
        name="mock_listener_instance_start"
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
    mock_listener_from_factory: AsyncMock = mocker.create_autospec(
        ActualInstanceListener, instance=True
    )
    mock_listener_from_factory.start = AsyncMock(
        name="mock_listener_from_factory_start"
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
    mock_listener_from_factory.start.assert_awaited_once()
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
    mock_client_arg: AsyncMock = mock_service_source_client_fixture

    mock_listener_instance: AsyncMock = mocker.create_autospec(
        ActualInstanceListener, instance=True
    )
    mock_listener_instance.start = AsyncMock(
        name="mock_listener_instance_start_method"
    )

    mock_factory_on_host: Mock = mocker.Mock(
        return_value=mock_listener_instance,
        name="MockedInstanceListenerFactoryOnHostInstance",
    )
    host._DiscoveryHost__instance_listener_factory = mock_factory_on_host

    await host.start_discovery(mock_client_arg)  # Use renamed arg

    mock_factory_on_host.assert_called_once_with(host)
    mock_listener_instance.start.assert_awaited_once()

    assert host._DiscoveryHost__discoverer is mock_listener_instance

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
    mock_listener_product: AsyncMock = mocker.create_autospec(
        ActualInstanceListener,
        instance=True,
    )
    mock_listener_product.start = AsyncMock()
    mock_mdns_factory: Mock = mocker.Mock(return_value=mock_listener_product)

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=None,
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_mdns_factory
        ),
    )

    original_instance_listener_init: typing.Callable[..., None] = (
        ActualInstanceListener.__init__
    )
    created_instance_holder: typing.Dict[
        str, ActualInstanceListener[ServiceInfo]
    ] = {}

    def init_side_effect(
        actual_self: ActualInstanceListener[ServiceInfo],
        client: ActualInstanceListener.Client,
        service_type: str,
        mdns_listener_factory: typing.Optional[MdnsListenerFactory] = None,
    ) -> None:
        original_instance_listener_init(
            actual_self,
            client,
            service_type,
            mdns_listener_factory=mdns_listener_factory,
        )
        created_instance_holder["instance"] = actual_self

    with patch(
        "tsercom.discovery.mdns.instance_listener.InstanceListener.__init__",
        side_effect=init_side_effect,
        autospec=True,
    ) as mock_il_init:
        await host.start_discovery(mock_service_source_client)

        mock_il_init.assert_called_once()
        call_args = mock_il_init.call_args
        assert call_args.args[1] is host
        assert call_args.args[2] == "_internal_default._tcp.local."
        assert call_args.kwargs["mdns_listener_factory"] is mock_mdns_factory

        mock_mdns_factory.assert_called_once()
        assert mock_mdns_factory.call_args.args[
            0
        ] is created_instance_holder.get("instance")
        assert (
            mock_mdns_factory.call_args.args[1]
            == "_internal_default._tcp.local."
        )

        # This assertion is now on the InstanceListener's start, which calls the RecordListener's start
        # The mock_listener_product is the RecordListener in this context of mdns_listener_factory
        # The created_instance_holder["instance"] is the InstanceListener
        # So we need to check that the InstanceListener's start was awaited,
        # which internally awaits the RecordListener's start.
        assert (
            created_instance_holder["instance"]._InstanceListener__listener
            is mock_listener_product
        )
        mock_listener_product.start.assert_awaited_once()  # Check RecordListener's start

        assert host._DiscoveryHost__discoverer is created_instance_holder.get(
            "instance"
        )
        assert host._DiscoveryHost__discoverer is not None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_discovery_host_handles_mdns_factory_exception_gracefully(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture
    mock_failing_mdns_factory: Mock = mocker.Mock(
        side_effect=RuntimeError("Factory boom!")
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=None,
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
    # This mock_listener_product_failing_start represents the product of the mdns_listener_factory,
    # which is a RecordListener (or compatible). Its start method is what fails.
    mock_listener_product_failing_start: AsyncMock = mocker.create_autospec(
        ActualInstanceListener,
        instance=True,  # Use a compatible spec, RecordListener would be better if defined
    )
    mock_listener_product_failing_start.start = AsyncMock(  # start is async
        side_effect=RuntimeError("Listener start boom!")
    )
    mock_mdns_factory_arg: Mock = mocker.Mock(
        return_value=mock_listener_product_failing_start
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type=None,
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_mdns_factory_arg
        ),
    )

    with patch("logging.error") as mock_log_error:
        await host.start_discovery(mock_service_source_client)
        mock_mdns_factory_arg.assert_called_once()
        assert (
            mock_mdns_factory_arg.call_args.args[1]
            == "_internal_default._tcp.local."
        )
        # The InstanceListener's start method will call the (failing) start method of the
        # RecordListener produced by mock_mdns_factory_arg.
        # So, mock_listener_product_failing_start.start should have been awaited.
        mock_listener_product_failing_start.start.assert_awaited_once()
        assert host._DiscoveryHost__discoverer is None
        mock_log_error.assert_called_once()
        # The error message now comes from the DiscoveryHost's try-except block that calls
        # discoverer.start()
        assert (
            "Failed to initialize or start discovery listener: Listener start boom!"
            in mock_log_error.call_args.args[0]
        )


def test_discovery_host_init_with_only_mdns_factory(mocker):
    """
    Tests that DiscoveryHost can be initialized with only an mdns_listener_factory.
    This scenario (service_type=None, instance_listener_factory=None, mdns_listener_factory=mock)
    was causing a ValueError due to the boolean check.
    """
    from tsercom.discovery.mdns.instance_listener import (
        MdnsListenerFactory,
    )
    from tsercom.discovery.discovery_host import (
        DiscoveryHost,
    )

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
