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
) -> typing.Iterator[typing.Tuple[MagicMock, MagicMock]]:
    mock_listener_instance: MagicMock = mocker.MagicMock(
        spec=ActualInstanceListener,
        name="MockedActualInstanceListenerInstance",
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
    assert host._DiscoveryHost__caller_id_map[service_info.mdns_name] is expected_random_id_instance  # type: ignore[attr-defined]


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
    assert host._DiscoveryHost__caller_id_map[existing_mdns_name] is pre_existing_id  # type: ignore[attr-defined]


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
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
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
        match="Exactly one of 'service_type' or 'instance_listener_factory' must be provided.",
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
    assert host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name] is expected_caller_id  # type: ignore[attr-defined]


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
    assert host._DiscoveryHost__caller_id_map[mock_service_info.mdns_name] is mock_existing_caller_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mdns_listener_factory_invoked_via_instance_listener_on_start(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture
    mock_listener_product: MagicMock = mocker.MagicMock(
        spec=ActualInstanceListener
    )
    mock_listener_product.start = MagicMock()
    mock_mdns_factory: Mock = mocker.Mock(return_value=mock_listener_product)

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type="test.service",
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
        assert call_args.args[2] == "test.service"
        assert call_args.kwargs["mdns_listener_factory"] is mock_mdns_factory

        mock_mdns_factory.assert_called_once()
        assert mock_mdns_factory.call_args.args[
            0
        ] is created_instance_holder.get("instance")
        assert mock_mdns_factory.call_args.args[1] == "test.service"

        mock_listener_product.start.assert_called_once()
        assert host._DiscoveryHost__discoverer is created_instance_holder.get("instance")  # type: ignore[attr-defined]
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
        service_type="test.service.fail.factory",
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_failing_mdns_factory
        ),
    )

    with patch("logging.error") as mock_log_error:
        await host.start_discovery(mock_service_source_client)
        mock_failing_mdns_factory.assert_called_once()
        assert (
            mock_failing_mdns_factory.call_args.args[1]
            == "test.service.fail.factory"
        )
        assert host._DiscoveryHost__discoverer is None  # type: ignore[attr-defined]
        mock_log_error.assert_called_once()
        assert (
            "Failed to initialize discovery listener: Factory boom!"
            in mock_log_error.call_args.args[0]
        )


@pytest.mark.asyncio
async def test_discovery_host_handles_listener_start_exception_gracefully(
    mocker: Mock, mock_service_source_client_fixture: AsyncMock
) -> None:
    mock_service_source_client: AsyncMock = mock_service_source_client_fixture
    mock_listener_product_failing_start: MagicMock = mocker.MagicMock(
        spec=ActualInstanceListener
    )
    mock_listener_product_failing_start.start = MagicMock(
        side_effect=RuntimeError("Listener start boom!")
    )
    mock_mdns_factory_arg: Mock = mocker.Mock(
        return_value=mock_listener_product_failing_start
    )

    host: DiscoveryHost[ServiceInfo] = DiscoveryHost(
        service_type="test.service.fail.start",
        mdns_listener_factory=typing.cast(
            MdnsListenerFactory, mock_mdns_factory_arg
        ),
    )

    with patch("logging.error") as mock_log_error:
        await host.start_discovery(mock_service_source_client)
        mock_mdns_factory_arg.assert_called_once()
        assert (
            mock_mdns_factory_arg.call_args.args[1]
            == "test.service.fail.start"
        )
        mock_listener_product_failing_start.start.assert_called_once()
        assert host._DiscoveryHost__discoverer is None  # type: ignore[attr-defined]
        mock_log_error.assert_called_once()
        assert (
            "Failed to initialize discovery listener: Listener start boom!"
            in mock_log_error.call_args.args[0]
        )
