import pytest
import datetime
import threading
from _thread import LockType  # For isinstance check with threading.Lock
from concurrent.futures import ThreadPoolExecutor


# Assuming CallerIdentifier.py is missing, define a functional dummy for tests.
class DummyCallerIdentifier:
    def __init__(self, id_str: str):
        self.id_str = id_str

    def __hash__(self):
        return hash(self.id_str)

    def __eq__(self, other):
        return (
            isinstance(other, DummyCallerIdentifier)
            and self.id_str == other.id_str
        )

    def __repr__(self):
        return f"DummyCallerIdentifier('{self.id_str}')"

    @property
    def name(self) -> str:
        return self.id_str


# Import actual classes from tsercom
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
        self, caller_id: DummyCallerIdentifier, timestamp: datetime.datetime
    ):
        super().__init__(caller_id, timestamp)


# --- Fixtures ---


@pytest.fixture
def mock_thread_pool(mocker):
    return mocker.MagicMock(spec=ThreadPoolExecutor)


@pytest.fixture
def mock_client(mocker):
    return mocker.MagicMock(spec=RemoteDataAggregator.Client)


@pytest.fixture
def mock_data_timeout_tracker_class(mocker):
    mock_cls = mocker.patch(
        "tsercom.data.remote_data_aggregator_impl.DataTimeoutTracker"
    )
    mock_instance = mocker.MagicMock(spec=DataTimeoutTracker)
    mock_instance.start = mocker.MagicMock()
    mock_instance.register = mocker.MagicMock()
    mock_cls.return_value = mock_instance
    return mock_cls


@pytest.fixture
def explicit_mock_tracker(mocker):
    tracker_instance = mocker.MagicMock(spec=DataTimeoutTracker)
    tracker_instance.start = mocker.MagicMock()
    tracker_instance.register = mocker.MagicMock()
    return tracker_instance


@pytest.fixture
def mock_remote_data_organizer_class(mocker):
    mock_cls = mocker.patch(
        "tsercom.data.remote_data_aggregator_impl.RemoteDataOrganizer"
    )
    mock_cls.return_value = mocker.MagicMock(
        spec=RemoteDataOrganizer
    )  # Default return
    return mock_cls


@pytest.fixture
def caller_id_1():
    return DummyCallerIdentifier("caller_1")


@pytest.fixture
def caller_id_2():
    return DummyCallerIdentifier("caller_2")


@pytest.fixture
def exposed_data_factory(caller_id_1):
    def _factory(caller_id=caller_id_1, timestamp=None):
        ts = timestamp or datetime.datetime.now()
        return DummyConcreteExposedData(caller_id, ts)

    return _factory


# --- Test Cases ---


# 1. Initialization Tests
def test_init_with_thread_pool_only(mock_thread_pool):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    assert (
        aggregator._RemoteDataAggregatorImpl__thread_pool is mock_thread_pool
    )
    assert aggregator._RemoteDataAggregatorImpl__client is None
    assert aggregator._RemoteDataAggregatorImpl__tracker is None
    assert not aggregator._RemoteDataAggregatorImpl__organizers
    assert isinstance(aggregator._RemoteDataAggregatorImpl__lock, LockType)


def test_init_with_thread_pool_and_client(mock_thread_pool, mock_client):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=mock_client
    )
    assert aggregator._RemoteDataAggregatorImpl__client is mock_client


def test_init_with_explicit_tracker(
    mock_thread_pool, mock_client, explicit_mock_tracker
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=mock_client, tracker=explicit_mock_tracker
    )
    assert (
        aggregator._RemoteDataAggregatorImpl__tracker is explicit_mock_tracker
    )
    explicit_mock_tracker.start.assert_not_called()


def test_init_with_timeout_creates_and_starts_tracker(
    mock_thread_pool, mock_client, mock_data_timeout_tracker_class
):
    timeout_seconds = 60
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=mock_client, timeout=timeout_seconds
    )
    mock_data_timeout_tracker_class.assert_called_once_with(timeout_seconds)
    mock_instance = mock_data_timeout_tracker_class.return_value
    mock_instance.start.assert_called_once()
    assert aggregator._RemoteDataAggregatorImpl__tracker is mock_instance


def test_init_asserts_not_timeout_and_tracker(
    mock_thread_pool, explicit_mock_tracker
):
    with pytest.raises(AssertionError):
        RemoteDataAggregatorImpl[DummyConcreteExposedData](
            mock_thread_pool, timeout=60, tracker=explicit_mock_tracker
        )


# 2. _on_data_ready() Tests
def test_on_data_ready_new_organizer_no_client_no_tracker(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    new_data = exposed_data_factory(caller_id=caller_id_1)

    mock_organizer_instance = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_instance.caller_id = caller_id_1
    mock_remote_data_organizer_class.return_value = mock_organizer_instance

    aggregator._on_data_ready(new_data)

    mock_remote_data_organizer_class.assert_called_once_with(
        mock_thread_pool, new_data.caller_id, aggregator
    )
    mock_organizer_instance.start.assert_called_once()
    mock_organizer_instance._on_data_ready.assert_called_once_with(new_data)
    assert (
        aggregator._RemoteDataAggregatorImpl__organizers[caller_id_1]
        is mock_organizer_instance
    )


def test_on_data_ready_new_organizer_with_client_and_tracker(
    mock_thread_pool,
    mock_client,
    explicit_mock_tracker,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=mock_client, tracker=explicit_mock_tracker
    )
    new_data = exposed_data_factory(caller_id=caller_id_1)

    mock_organizer_instance = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_instance.caller_id = caller_id_1
    mock_remote_data_organizer_class.return_value = mock_organizer_instance

    aggregator._on_data_ready(new_data)

    mock_remote_data_organizer_class.assert_called_once()
    mock_organizer_instance.start.assert_called_once()
    explicit_mock_tracker.register.assert_called_once_with(
        mock_organizer_instance
    )
    mock_client._on_new_endpoint_began_transmitting.assert_called_once_with(
        aggregator, caller_id_1
    )
    mock_organizer_instance._on_data_ready.assert_called_once_with(new_data)


def test_on_data_ready_existing_organizer(
    mock_thread_pool,
    mock_client,
    explicit_mock_tracker,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=mock_client, tracker=explicit_mock_tracker
    )

    first_data_ts = datetime.datetime(2023, 1, 1, 12, 0, 0)
    first_data = exposed_data_factory(
        caller_id=caller_id_1, timestamp=first_data_ts
    )

    mock_organizer_instance = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_instance.caller_id = caller_id_1
    # Explicitly create and assign the mock for the _on_data_ready method
    organizer_on_data_ready_method_mock = mocker.MagicMock(
        name="_on_data_ready_explicit_method_mock"
    )
    mock_organizer_instance._on_data_ready = (
        organizer_on_data_ready_method_mock
    )

    mock_remote_data_organizer_class.return_value = mock_organizer_instance

    aggregator._on_data_ready(first_data)
    assert organizer_on_data_ready_method_mock.call_count == 1

    mock_remote_data_organizer_class.reset_mock()
    mock_organizer_instance.start.reset_mock()
    explicit_mock_tracker.register.reset_mock()
    mock_client._on_new_endpoint_began_transmitting.reset_mock()

    second_data_ts = datetime.datetime(2023, 1, 1, 12, 0, 1)
    second_data = exposed_data_factory(
        caller_id=caller_id_1, timestamp=second_data_ts
    )
    aggregator._on_data_ready(second_data)

    mock_remote_data_organizer_class.assert_not_called()
    mock_organizer_instance.start.assert_not_called()
    explicit_mock_tracker.register.assert_not_called()
    mock_client._on_new_endpoint_began_transmitting.assert_not_called()

    retrieved_organizer = aggregator._RemoteDataAggregatorImpl__organizers[
        caller_id_1
    ]
    assert retrieved_organizer is mock_organizer_instance
    assert (
        retrieved_organizer._on_data_ready
        is organizer_on_data_ready_method_mock
    )

    assert organizer_on_data_ready_method_mock.call_count == 2
    organizer_on_data_ready_method_mock.assert_called_with(second_data)


# 3. stop() Tests
def test_stop_with_caller_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    caller_id_2,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )

    mock_organizer_1 = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_1.caller_id = caller_id_1
    mock_organizer_2 = mocker.MagicMock(spec=RemoteDataOrganizer)
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
    assert caller_id_1 in aggregator._RemoteDataAggregatorImpl__organizers

    # Expect KeyError when stopping a non-existent ID
    with pytest.raises(
        KeyError,
        match="Caller ID .* not found in active organizers during stop.",
    ):
        aggregator.stop(DummyCallerIdentifier("non_existent_id"))


def test_stop_all(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    caller_id_2,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_organizer_1 = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_organizer_1.caller_id = caller_id_1
    mock_organizer_2 = mocker.MagicMock(spec=RemoteDataOrganizer)
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
    aggregator,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    organizers_map,
):
    mock_organizer_instances = []
    sorted_caller_ids = sorted(
        organizers_map.keys(), key=lambda cid: cid.id_str
    )

    for cid in sorted_caller_ids:
        mock_organizer_instances.append(organizers_map[cid])

    mock_remote_data_organizer_class.side_effect = mock_organizer_instances

    for cid in sorted_caller_ids:
        org_instance = organizers_map[cid]
        org_instance.caller_id = cid
        data = exposed_data_factory(caller_id=cid)
        aggregator._on_data_ready(data)


# has_new_data
def test_has_new_data_with_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_organizer = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    mock_organizer.has_new_data.return_value = True
    assert aggregator.has_new_data(caller_id_1) is True
    mock_organizer.has_new_data.assert_called_once()


def test_has_new_data_no_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    caller_id_2,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_o1 = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_o2 = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_o1, caller_id_2: mock_o2}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    mock_o1.has_new_data.return_value = True
    mock_o2.has_new_data.return_value = False
    expected = {caller_id_1: True, caller_id_2: False}
    assert aggregator.has_new_data() == expected
    mock_o1.has_new_data.assert_called_once()
    mock_o2.has_new_data.assert_called_once()


# get_new_data
def test_get_new_data_with_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_organizer = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    expected_list = [exposed_data_factory(caller_id=caller_id_1)]
    mock_organizer.get_new_data.return_value = expected_list
    assert aggregator.get_new_data(caller_id_1) == expected_list
    mock_organizer.get_new_data.assert_called_once()


def test_get_new_data_no_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    caller_id_2,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_o1 = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_o2 = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_o1, caller_id_2: mock_o2}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    list1 = [exposed_data_factory(caller_id=caller_id_1)]
    list2 = []
    mock_o1.get_new_data.return_value = list1
    mock_o2.get_new_data.return_value = list2
    expected = {caller_id_1: list1, caller_id_2: list2}
    assert aggregator.get_new_data() == expected


# get_most_recent_data
def test_get_most_recent_data_with_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_organizer = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    data = exposed_data_factory(caller_id=caller_id_1)
    mock_organizer.get_most_recent_data.return_value = data
    assert aggregator.get_most_recent_data(caller_id_1) is data


def test_get_most_recent_data_no_id(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    caller_id_2,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_o1 = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_o2 = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_o1, caller_id_2: mock_o2}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    data1 = exposed_data_factory(caller_id=caller_id_1)
    mock_o1.get_most_recent_data.return_value = data1
    mock_o2.get_most_recent_data.return_value = None
    expected = {caller_id_1: data1, caller_id_2: None}
    assert aggregator.get_most_recent_data() == expected


# get_data_for_timestamp
def test_get_data_for_timestamp(
    mock_thread_pool,
    mock_remote_data_organizer_class,
    exposed_data_factory,
    caller_id_1,
    mocker,
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    mock_organizer = mocker.MagicMock(spec=RemoteDataOrganizer)
    organizers_map = {caller_id_1: mock_organizer}
    _setup_aggregator_with_organizers_for_retrieval(
        aggregator,
        mock_remote_data_organizer_class,
        exposed_data_factory,
        organizers_map,
    )

    timestamp = datetime.datetime.now()
    data = exposed_data_factory(caller_id=caller_id_1)
    mock_organizer.get_data_for_timestamp.return_value = data

    # Corrected argument order: timestamp first, then caller_id_1
    assert aggregator.get_data_for_timestamp(timestamp, caller_id_1) is data
    mock_organizer.get_data_for_timestamp.assert_called_once_with(timestamp)


# Test data retrieval for non-existent ID
def test_data_retrieval_non_existent_id(mock_thread_pool):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )
    non_existent_id = DummyCallerIdentifier("non_existent")
    timestamp = datetime.datetime.now()

    # has_new_data should return False for a non-existent ID, not raise KeyError
    assert aggregator.has_new_data(non_existent_id) is False

    # Other get* methods should still raise KeyError for a non-existent ID
    with pytest.raises(
        KeyError, match="Caller ID .* not found for get_new_data."
    ):
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
    mock_thread_pool, mock_client, caller_id_1, mocker
):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=mock_client
    )

    mock_calling_organizer = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_calling_organizer.caller_id = caller_id_1

    aggregator._on_data_available(mock_calling_organizer)

    mock_client._on_data_available.assert_called_once_with(
        aggregator, caller_id_1
    )


def test_on_data_available_no_client(mock_thread_pool, caller_id_1, mocker):
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool, client=None
    )
    mock_calling_organizer = mocker.MagicMock(spec=RemoteDataOrganizer)
    mock_calling_organizer.caller_id = caller_id_1

    try:
        aggregator._on_data_available(mock_calling_organizer)
    except Exception as e:  # pragma: no cover
        pytest.fail(
            f"_on_data_available raised an exception with no client: {e}"
        )


# Test _on_data_ready with invalid data type
def test_on_data_ready_invalid_data_type(mock_thread_pool):
    """
    Tests that _on_data_ready() raises a TypeError if new_data is not ExposedData.
    """
    aggregator = RemoteDataAggregatorImpl[DummyConcreteExposedData](
        mock_thread_pool
    )

    class NotExposedData:  # Simple class not inheriting from ExposedData
        pass

    invalid_data_object = NotExposedData()

    with pytest.raises(
        TypeError,
        match=r"Expected new_data to be an instance of ExposedData, but got .*\.",
    ):
        aggregator._on_data_ready(invalid_data_object)
