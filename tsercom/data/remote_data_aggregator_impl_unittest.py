import pytest
import datetime
import uuid  # Added
from _thread import LockType  # For isinstance check with threading.Lock
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, cast, Dict, List  # Added for type hints
from unittest.mock import MagicMock  # Added
from pytest_mock import MockerFixture  # Added for type hints

# Import actual classes from tsercom
from tsercom.caller_id.caller_identifier import CallerIdentifier  # Changed from dummy
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_aggregator import (
    RemoteDataAggregator,
)  # For client spec
from tsercom.data.exposed_data import ExposedData
from tsercom.data.data_timeout_tracker import (
    DataTimeoutTracker,
)  # For spec and patching
from tsercom.data.remote_data_organizer import (
    RemoteDataOrganizer,
)  # For spec and patching


# --- Concrete Dummy ExposedData for type checks ---
class DummyConcreteExposedData(ExposedData):
    def __init__(
        self, caller_id: CallerIdentifier, timestamp: datetime.datetime
    ):  # Changed DummyCallerIdentifier
        super().__init__(caller_id, timestamp)


# --- Fixtures ---


@pytest.fixture
def mock_thread_pool(mocker: MockerFixture) -> MagicMock:  # Changed types
    return cast(MagicMock, mocker.MagicMock(spec=ThreadPoolExecutor))


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:  # Changed types
    return cast(MagicMock, mocker.MagicMock(spec=RemoteDataAggregator.Client))


@pytest.fixture
def mock_data_timeout_tracker_class(
    mocker: MockerFixture,
) -> MagicMock:  # Changed types
    mock_cls = mocker.patch(
        "tsercom.data.remote_data_aggregator_impl.DataTimeoutTracker"
    )
    mock_instance = mocker.MagicMock(spec=DataTimeoutTracker)
    mock_instance.start = mocker.MagicMock()
    mock_instance.register = mocker.MagicMock()
    mock_cls.return_value = mock_instance
    return cast(MagicMock, mock_cls)


@pytest.fixture
def explicit_mock_tracker(mocker: MockerFixture) -> MagicMock:  # Changed types
    tracker_instance = mocker.MagicMock(spec=DataTimeoutTracker)
    tracker_instance.start = mocker.MagicMock()
    tracker_instance.register = mocker.MagicMock()
    return cast(MagicMock, tracker_instance)


@pytest.fixture
def mock_remote_data_organizer_class(
    mocker: MockerFixture,
) -> MagicMock:  # Changed types
    mock_cls = mocker.patch(
        "tsercom.data.remote_data_aggregator_impl.RemoteDataOrganizer"
    )
    mock_cls.return_value = mocker.MagicMock(spec=RemoteDataOrganizer)  # Default return
    return mock_cls


@pytest.fixture
def caller_id_1() -> CallerIdentifier:  # Changed type
    return CallerIdentifier(uuid.UUID("00000000-0000-0000-0000-000000000001"))


@pytest.fixture
def caller_id_2() -> CallerIdentifier:  # Changed type
    return CallerIdentifier(uuid.UUID("00000000-0000-0000-0000-000000000002"))


@pytest.fixture
def exposed_data_factory(
    caller_id_1: CallerIdentifier,
) -> Callable[..., DummyConcreteExposedData]:  # Added types
    def _factory(
        caller_id: CallerIdentifier = caller_id_1,
        timestamp: Optional[datetime.datetime] = None,
    ) -> DummyConcreteExposedData:  # Added types
        ts: datetime.datetime = timestamp or datetime.datetime.now()
        return DummyConcreteExposedData(caller_id, ts)

    return _factory


class InterpolableExposedData(ExposedData):
    def __init__(
        self,
        caller_id: CallerIdentifier,
        timestamp: datetime.datetime,
        value: float,
    ):
        super().__init__(caller_id, timestamp)
        self.value: float = value

    @property
    def frame_timestamp(self) -> datetime.datetime:
        return self.timestamp

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, InterpolableExposedData):
            return NotImplemented
        return self.frame_timestamp < other.frame_timestamp

    def __repr__(self) -> str:
        return (
            f"InterpolableExposedData(caller_id={self.caller_id}, "
            f"timestamp={self.frame_timestamp}, value={self.value})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterpolableExposedData):
            return False
        return (
            self.caller_id == other.caller_id
            and self.frame_timestamp == other.frame_timestamp
            and self.value == other.value
        )


# --- Test Cases ---


# 1. Initialization Tests
def test_init_with_thread_pool_only(mock_thread_pool: MagicMock) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    assert aggregator._RemoteDataAggregatorImpl__thread_pool is mock_thread_pool  # type: ignore[attr-defined]
    assert aggregator._RemoteDataAggregatorImpl__client is None  # type: ignore[attr-defined]
    assert aggregator._RemoteDataAggregatorImpl__tracker is None  # type: ignore[attr-defined]
    assert not aggregator._RemoteDataAggregatorImpl__organizers  # type: ignore[attr-defined]
    assert isinstance(aggregator._RemoteDataAggregatorImpl__lock, LockType)  # type: ignore[attr-defined]


def test_init_with_thread_pool_and_client(
    mock_thread_pool: MagicMock, mock_client: MagicMock
) -> None:
    aggregator: RemoteDataAggregatorImpl[
        DummyConcreteExposedData
    ] = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        cast(ThreadPoolExecutor, mock_thread_pool),
        client=cast(RemoteDataAggregator.Client, mock_client),
    )
    assert aggregator._RemoteDataAggregatorImpl__client is mock_client  # type: ignore[attr-defined]


def test_init_with_explicit_tracker(
    mock_thread_pool: MagicMock,
    mock_client: MagicMock,
    explicit_mock_tracker: MagicMock,
) -> None:
    aggregator: RemoteDataAggregatorImpl[
        DummyConcreteExposedData
    ] = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        cast(ThreadPoolExecutor, mock_thread_pool),
        client=cast(RemoteDataAggregator.Client, mock_client),
        tracker=cast(DataTimeoutTracker, explicit_mock_tracker),
    )
    assert aggregator._RemoteDataAggregatorImpl__tracker is explicit_mock_tracker  # type: ignore[attr-defined]
    explicit_mock_tracker.start.assert_not_called()


def test_init_with_timeout_creates_and_starts_tracker(
    mock_thread_pool: MagicMock,
    mock_client: MagicMock,
    mock_data_timeout_tracker_class: MagicMock,
) -> None:
    timeout_seconds: int = 60
    aggregator: RemoteDataAggregatorImpl[
        DummyConcreteExposedData
    ] = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        cast(ThreadPoolExecutor, mock_thread_pool),
        client=cast(RemoteDataAggregator.Client, mock_client),
        timeout=timeout_seconds,
    )
    mock_data_timeout_tracker_class.assert_called_once_with(
        timeout_seconds
    )
    mock_instance_created_by_class: MagicMock = (
        mock_data_timeout_tracker_class.return_value
    )
    mock_instance_created_by_class.start.assert_called_once()
    assert aggregator._RemoteDataAggregatorImpl__tracker is mock_instance_created_by_class  # type: ignore[attr-defined]


def test_init_asserts_not_timeout_and_tracker(
    mock_thread_pool: MagicMock, explicit_mock_tracker: MagicMock
) -> None:
    with pytest.raises(AssertionError):
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool), timeout=60, tracker=cast(DataTimeoutTracker, explicit_mock_tracker)  # type: ignore[call-overload]
        )


# 2. _on_data_ready() Tests
def test_on_data_ready_new_organizer_no_client_no_tracker(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    new_data: DummyConcreteExposedData = exposed_data_factory(caller_id=caller_id_1)

    mock_organizer_instance: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_instance.caller_id = caller_id_1
    mock_remote_data_organizer_class.return_value = mock_organizer_instance

    aggregator._on_data_ready(new_data)

    mock_remote_data_organizer_class.assert_called_once_with(
        cast(ThreadPoolExecutor, mock_thread_pool), new_data.caller_id, aggregator
    )
    mock_organizer_instance.start.assert_called_once()
    mock_organizer_instance._on_data_ready.assert_called_once_with(new_data)
    assert (
        aggregator._RemoteDataAggregatorImpl__organizers[caller_id_1]  # type: ignore[attr-defined]
        is mock_organizer_instance
    )


def test_on_data_ready_new_organizer_with_client_and_tracker(
    mock_thread_pool: MagicMock,
    mock_client: MagicMock,
    explicit_mock_tracker: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[
        DummyConcreteExposedData
    ] = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        cast(ThreadPoolExecutor, mock_thread_pool),
        client=cast(RemoteDataAggregator.Client, mock_client),
        tracker=cast(DataTimeoutTracker, explicit_mock_tracker),
    )
    new_data: DummyConcreteExposedData = exposed_data_factory(caller_id=caller_id_1)

    mock_organizer_instance: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_instance.caller_id = caller_id_1
    mock_remote_data_organizer_class.return_value = mock_organizer_instance

    aggregator._on_data_ready(new_data)

    mock_remote_data_organizer_class.assert_called_once()
    mock_organizer_instance.start.assert_called_once()
    explicit_mock_tracker.register.assert_called_once_with(mock_organizer_instance)
    mock_client._on_new_endpoint_began_transmitting.assert_called_once_with(
        aggregator, caller_id_1
    )
    mock_organizer_instance._on_data_ready.assert_called_once_with(new_data)


def test_on_data_ready_existing_organizer(
    mock_thread_pool: MagicMock,
    mock_client: MagicMock,
    explicit_mock_tracker: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[
        DummyConcreteExposedData
    ] = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        cast(ThreadPoolExecutor, mock_thread_pool),
        client=cast(RemoteDataAggregator.Client, mock_client),
        tracker=cast(DataTimeoutTracker, explicit_mock_tracker),
    )

    first_data_ts: datetime.datetime = datetime.datetime(2023, 1, 1, 12, 0, 0)
    first_data: DummyConcreteExposedData = exposed_data_factory(
        caller_id=caller_id_1, timestamp=first_data_ts
    )

    mock_organizer_instance: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_instance.caller_id = caller_id_1
    organizer_on_data_ready_method_mock: MagicMock = mocker.MagicMock(
        name="_on_data_ready_explicit_method_mock"
    )
    mock_organizer_instance._on_data_ready = organizer_on_data_ready_method_mock

    mock_remote_data_organizer_class.return_value = mock_organizer_instance

    aggregator._on_data_ready(first_data)
    assert organizer_on_data_ready_method_mock.call_count == 1

    mock_remote_data_organizer_class.reset_mock()
    mock_organizer_instance.start.reset_mock()
    explicit_mock_tracker.register.reset_mock()
    mock_client._on_new_endpoint_began_transmitting.reset_mock()

    second_data_ts: datetime.datetime = datetime.datetime(2023, 1, 1, 12, 0, 1)
    second_data: DummyConcreteExposedData = exposed_data_factory(
        caller_id=caller_id_1, timestamp=second_data_ts
    )
    aggregator._on_data_ready(second_data)

    mock_remote_data_organizer_class.assert_not_called()
    mock_organizer_instance.start.assert_not_called()
    explicit_mock_tracker.register.assert_not_called()
    mock_client._on_new_endpoint_began_transmitting.assert_not_called()

    retrieved_organizer: MagicMock = aggregator._RemoteDataAggregatorImpl__organizers[caller_id_1]  # type: ignore[attr-defined]
    assert retrieved_organizer is mock_organizer_instance
    assert retrieved_organizer._on_data_ready is organizer_on_data_ready_method_mock

    assert organizer_on_data_ready_method_mock.call_count == 2
    organizer_on_data_ready_method_mock.assert_called_with(second_data)


# 3. stop() Tests
def test_stop_with_caller_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    caller_id_2: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )

    mock_organizer_1: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_1.caller_id = caller_id_1
    mock_organizer_2: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_2.caller_id = caller_id_2
    mock_remote_data_organizer_class.side_effect = [
        mock_organizer_1,
        mock_organizer_2,
    ]
    aggregator._on_data_ready(exposed_data_factory(caller_id=caller_id_1))
    aggregator._on_data_ready(exposed_data_factory(caller_id=caller_id_2))

    aggregator.stop(caller_id_1)
    mock_organizer_1.stop.assert_called_once()
    mock_organizer_2.stop.assert_not_called()
    assert caller_id_1 in aggregator._RemoteDataAggregatorImpl__organizers  # type: ignore[attr-defined]

    with pytest.raises(
        KeyError,
        match="Caller ID .* not found in active organizers during stop.",
    ):
        aggregator.stop(CallerIdentifier(uuid.uuid4()))


def test_stop_all(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    caller_id_2: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer_1: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_1.caller_id = caller_id_1
    mock_organizer_2: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_2.caller_id = caller_id_2
    mock_remote_data_organizer_class.side_effect = [
        mock_organizer_1,
        mock_organizer_2,
    ]
    aggregator._on_data_ready(exposed_data_factory(caller_id=caller_id_1))
    aggregator._on_data_ready(exposed_data_factory(caller_id=caller_id_2))

    aggregator.stop()
    mock_organizer_1.stop.assert_called_once()
    mock_organizer_2.stop.assert_called_once()


# Helper for data retrieval tests
def _setup_aggregator_with_organizers_for_retrieval(
    aggregator: RemoteDataAggregatorImpl[Any],
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., ExposedData],
    organizers_map: Dict[CallerIdentifier, MagicMock],
) -> None:
    mock_organizer_instances: List[MagicMock] = []
    sorted_caller_ids = sorted(organizers_map.keys(), key=lambda cid: str(cid))

    for cid in sorted_caller_ids:
        mock_organizer_instances.append(organizers_map[cid])

    mock_remote_data_organizer_class.side_effect = mock_organizer_instances

    for cid in sorted_caller_ids:
        org_instance = organizers_map[cid]
        org_instance.caller_id = cid
        data: ExposedData = exposed_data_factory(caller_id=cid)
        aggregator._on_data_ready(data)


# has_new_data
def test_has_new_data_with_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    mock_organizer.has_new_data.return_value = True
    assert aggregator.has_new_data(caller_id_1) is True
    mock_organizer.has_new_data.assert_called_once()


def test_has_new_data_no_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    caller_id_2: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_o1: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_o2: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {
        caller_id_1: mock_o1,
        caller_id_2: mock_o2,
    }
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    mock_o1.has_new_data.return_value = True
    mock_o2.has_new_data.return_value = False
    expected: Dict[CallerIdentifier, bool] = {caller_id_1: True, caller_id_2: False}
    assert aggregator.has_new_data() == expected
    mock_o1.has_new_data.assert_called_once()
    mock_o2.has_new_data.assert_called_once()


# get_new_data
def test_get_new_data_with_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    expected_list: List[DummyConcreteExposedData] = [
        exposed_data_factory(caller_id=caller_id_1)
    ]
    mock_organizer.get_new_data.return_value = expected_list
    assert aggregator.get_new_data(caller_id_1) == expected_list
    mock_organizer.get_new_data.assert_called_once()


def test_get_new_data_no_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    caller_id_2: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_o1: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_o2: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {
        caller_id_1: mock_o1,
        caller_id_2: mock_o2,
    }
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    list1: List[DummyConcreteExposedData] = [
        exposed_data_factory(caller_id=caller_id_1)
    ]
    list2: List[DummyConcreteExposedData] = []
    mock_o1.get_new_data.return_value = list1
    mock_o2.get_new_data.return_value = list2
    expected: Dict[CallerIdentifier, List[DummyConcreteExposedData]] = {
        caller_id_1: list1,
        caller_id_2: list2,
    }
    assert aggregator.get_new_data() == expected


# get_most_recent_data
def test_get_most_recent_data_with_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    data: DummyConcreteExposedData = exposed_data_factory(caller_id=caller_id_1)
    mock_organizer.get_most_recent_data.return_value = data
    assert aggregator.get_most_recent_data(caller_id_1) is data


def test_get_most_recent_data_no_id(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    caller_id_2: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_o1: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_o2: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {
        caller_id_1: mock_o1,
        caller_id_2: mock_o2,
    }
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    data1: DummyConcreteExposedData = exposed_data_factory(caller_id=caller_id_1)
    mock_o1.get_most_recent_data.return_value = data1
    mock_o2.get_most_recent_data.return_value = None
    expected: Dict[CallerIdentifier, Optional[DummyConcreteExposedData]] = {
        caller_id_1: data1,
        caller_id_2: None,
    }
    assert aggregator.get_most_recent_data() == expected


# get_data_for_timestamp
def test_get_data_for_timestamp(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    exposed_data_factory: Callable[..., DummyConcreteExposedData],
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map: Dict[CallerIdentifier, MagicMock] = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        cast(Callable[..., ExposedData], exposed_data_factory),
        organizers_map,
    )

    timestamp: datetime.datetime = datetime.datetime.now()
    data: DummyConcreteExposedData = exposed_data_factory(caller_id=caller_id_1)
    mock_organizer.get_data_for_timestamp.return_value = data

    assert aggregator.get_data_for_timestamp(timestamp, caller_id_1) is data
    mock_organizer.get_data_for_timestamp.assert_called_once_with(timestamp)


# Test data retrieval for non-existent ID
def test_data_retrieval_non_existent_id(mock_thread_pool: MagicMock) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    non_existent_id = CallerIdentifier(uuid.uuid4())
    timestamp: datetime.datetime = datetime.datetime.now()

    assert aggregator.has_new_data(non_existent_id) is False

    with pytest.raises(KeyError, match="Caller ID .* not found for get_new_data."):
        aggregator.get_new_data(non_existent_id)
    with pytest.raises(
        KeyError, match="Caller ID .* not found for get_most_recent_data."
    ):
        aggregator.get_most_recent_data(non_existent_id)
    with pytest.raises(
        KeyError, match="Caller ID .* not found for get_data_for_timestamp."
    ):
        aggregator.get_data_for_timestamp(timestamp, non_existent_id)


# 5. _on_data_available() Test
def test_on_data_available_with_client(
    mock_thread_pool: MagicMock,
    mock_client: MagicMock,
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[
        DummyConcreteExposedData
    ] = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        cast(ThreadPoolExecutor, mock_thread_pool),
        client=cast(RemoteDataAggregator.Client, mock_client),
    )

    mock_calling_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_calling_organizer.caller_id = caller_id_1

    aggregator._on_data_available(
        cast(RemoteDataOrganizer[DummyConcreteExposedData], mock_calling_organizer)
    )

    mock_client._on_data_available.assert_called_once_with(aggregator, caller_id_1)


def test_on_data_available_no_client(
    mock_thread_pool: MagicMock, caller_id_1: CallerIdentifier, mocker: MockerFixture
) -> None:
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool), client=None
        )
    )
    mock_calling_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_calling_organizer.caller_id = caller_id_1

    try:
        aggregator._on_data_available(
            cast(RemoteDataOrganizer[DummyConcreteExposedData], mock_calling_organizer)
        )
    except Exception as e:  # pragma: no cover
        pytest.fail(f"_on_data_available raised an exception with no client: {e}")


# Test _on_data_ready with invalid data type
def test_on_data_ready_invalid_data_type(mock_thread_pool: MagicMock) -> None:
    """
    Tests that _on_data_ready() raises a TypeError if new_data is not ExposedData.
    """
    aggregator: RemoteDataAggregatorImpl[DummyConcreteExposedData] = (
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )

    class NotExposedData:  # Simple class not inheriting from ExposedData
        pass

    invalid_data_object = NotExposedData()

    with pytest.raises(
        TypeError,
        match=r"Expected new_data to be an instance of ExposedData, but got .*\.",
    ):
        aggregator._on_data_ready(invalid_data_object)  # type: ignore[arg-type]


# 6. get_interpolated_at() Tests
def test_get_interpolated_at_with_id_returns_organizer_result(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[InterpolableExposedData] = (
        RemoteDataAggregatorImpl[InterpolableExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)

    mock_remote_data_organizer_class.return_value = mock_organizer

    def interpolable_data_factory(
        caller_id: CallerIdentifier, timestamp: datetime.datetime, value: float
    ) -> InterpolableExposedData:
        return InterpolableExposedData(caller_id, timestamp, value)

    initial_data_time = datetime.datetime(2023, 10, 26, 12, 0, 0)
    aggregator._on_data_ready(
        interpolable_data_factory(caller_id_1, initial_data_time, 100.0)
    )
    actual_organizer_in_aggregator: MagicMock = (
        aggregator._RemoteDataAggregatorImpl__organizers[caller_id_1]  # type: ignore[attr-defined]
    )

    timestamp_to_query = datetime.datetime(2023, 10, 26, 12, 0, 5)
    expected_interpolated_data: InterpolableExposedData = interpolable_data_factory(
        caller_id_1, timestamp_to_query, 105.0
    )

    actual_organizer_in_aggregator.get_interpolated_at = mocker.MagicMock(
        return_value=expected_interpolated_data
    )

    result = aggregator.get_interpolated_at(timestamp_to_query, caller_id_1)

    actual_organizer_in_aggregator.get_interpolated_at.assert_called_once_with(
        timestamp_to_query
    )
    assert result == expected_interpolated_data


def test_get_interpolated_at_with_id_returns_none_from_organizer(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    caller_id_1: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[InterpolableExposedData] = (
        RemoteDataAggregatorImpl[InterpolableExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    mock_organizer: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_remote_data_organizer_class.return_value = mock_organizer

    def interpolable_data_factory(
        caller_id: CallerIdentifier, timestamp: datetime.datetime, value: float
    ) -> InterpolableExposedData:
        return InterpolableExposedData(caller_id, timestamp, value)

    initial_data_time = datetime.datetime(2023, 10, 26, 12, 0, 0)
    aggregator._on_data_ready(
        interpolable_data_factory(caller_id_1, initial_data_time, 100.0)
    )
    actual_organizer_in_aggregator: MagicMock = (
        aggregator._RemoteDataAggregatorImpl__organizers[caller_id_1]  # type: ignore[attr-defined]
    )

    timestamp_to_query = datetime.datetime(2023, 10, 26, 12, 0, 5)
    actual_organizer_in_aggregator.get_interpolated_at = mocker.MagicMock(
        return_value=None
    )

    result = aggregator.get_interpolated_at(timestamp_to_query, caller_id_1)

    actual_organizer_in_aggregator.get_interpolated_at.assert_called_once_with(
        timestamp_to_query
    )
    assert result is None


def test_get_interpolated_at_with_id_non_existent_id_raises_keyerror(
    mock_thread_pool: MagicMock,
) -> None:
    aggregator: RemoteDataAggregatorImpl[InterpolableExposedData] = (
        RemoteDataAggregatorImpl[InterpolableExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )
    non_existent_id = CallerIdentifier(uuid.uuid4())
    timestamp_to_query = datetime.datetime(2023, 10, 26, 12, 0, 0)

    with pytest.raises(
        KeyError, match="Caller ID .* not found for get_interpolated_at."
    ):
        aggregator.get_interpolated_at(timestamp_to_query, non_existent_id)


def test_get_interpolated_at_no_id_returns_dict_of_organizer_results(
    mock_thread_pool: MagicMock,
    mock_remote_data_organizer_class: MagicMock,
    caller_id_1: CallerIdentifier,
    caller_id_2: CallerIdentifier,
    mocker: MockerFixture,
) -> None:
    aggregator: RemoteDataAggregatorImpl[InterpolableExposedData] = (
        RemoteDataAggregatorImpl[InterpolableExposedData](
            cast(ThreadPoolExecutor, mock_thread_pool)
        )
    )

    mock_organizer_1: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_1.caller_id = caller_id_1
    mock_organizer_2: MagicMock = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_2.caller_id = caller_id_2

    def interpolable_data_factory(
        caller_id: CallerIdentifier, timestamp: datetime.datetime, value: float
    ) -> InterpolableExposedData:
        return InterpolableExposedData(caller_id, timestamp, value)

    ts_initial = datetime.datetime(2023, 10, 26, 12, 0, 0)

    # Setup for caller_id_1
    mock_remote_data_organizer_class.return_value = mock_organizer_1
    aggregator._on_data_ready(interpolable_data_factory(caller_id_1, ts_initial, 100.0))
    actual_org1: MagicMock = aggregator._RemoteDataAggregatorImpl__organizers[caller_id_1]  # type: ignore[attr-defined]
    assert actual_org1 is mock_organizer_1  # Verify setup

    # Setup for caller_id_2
    mock_remote_data_organizer_class.return_value = mock_organizer_2
    aggregator._on_data_ready(interpolable_data_factory(caller_id_2, ts_initial, 200.0))
    actual_org2: MagicMock = aggregator._RemoteDataAggregatorImpl__organizers[caller_id_2]  # type: ignore[attr-defined]
    assert actual_org2 is mock_organizer_2  # Verify setup

    timestamp_to_query = datetime.datetime(2023, 10, 26, 12, 0, 5)
    data1_interpolated: InterpolableExposedData = interpolable_data_factory(
        caller_id_1, timestamp_to_query, 105.0
    )

    actual_org1.get_interpolated_at = mocker.MagicMock(return_value=data1_interpolated)
    actual_org2.get_interpolated_at = mocker.MagicMock(return_value=None)

    result = aggregator.get_interpolated_at(timestamp_to_query)

    actual_org1.get_interpolated_at.assert_called_once_with(timestamp=timestamp_to_query)
    actual_org2.get_interpolated_at.assert_called_once_with(timestamp=timestamp_to_query)

    # Caller_id_2 results in None, so it should be omitted from the dict
    expected_dict: dict[CallerIdentifier, InterpolableExposedData] = {
        caller_id_1: data1_interpolated,
    }
    assert result == expected_dict
# Removed superfluous end of file marker
