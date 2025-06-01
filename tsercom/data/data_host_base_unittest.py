import pytest
import datetime

try:
    from tsercom.caller_id.caller_identifier import CallerIdentifier
except ImportError:
    CallerIdentifier = None

from tsercom.data.data_host_base import DataHostBase
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.data.exposed_data import ExposedData


class DummyExposedData(ExposedData):
    """A dummy implementation of ExposedData for type hinting and instantiation."""

    def __init__(self, mocker, caller_id_mock, timestamp_mock):
        super().__init__(caller_id_mock, timestamp_mock)


@pytest.fixture
def mock_caller_id(mocker):
    """Provides a mock CallerIdentifier."""
    if CallerIdentifier:  # pragma: no cover
        return mocker.MagicMock(spec=CallerIdentifier)
    return mocker.MagicMock(name="GenericMockCallerId")


@pytest.fixture
def mock_datetime_now(mocker):
    """Provides a mock datetime.datetime.now()."""
    # This fixture isn't strictly for datetime.now() but a mock datetime object
    return mocker.MagicMock(spec=datetime.datetime, name="MockDateTimeObject")


@pytest.fixture
def mock_thread_watcher(mocker):
    """Fixture to mock ThreadWatcher."""
    mock = mocker.MagicMock(spec=ThreadWatcher)
    mock.create_tracked_thread_pool_executor.return_value = mocker.MagicMock(
        name="MockThreadPoolExecutor"
    )
    return mock


@pytest.fixture
def mock_remote_data_aggregator_fixture(mocker):
    """
    Mocks RemoteDataAggregatorImpl class as used in DataHostBase and handles its generic nature.
    Returns a dictionary with 'mock_class', 'mock_constructor_proxy', and 'mock_instance'.
    """
    mock_class = mocker.patch(
        "tsercom.data.data_host_base.RemoteDataAggregatorImpl"
    )

    mock_generic_alias_callable = mocker.MagicMock(
        name="RemoteDataAggregatorImpl[TDataType]_Callable"
    )
    mock_class.__getitem__.return_value = mock_generic_alias_callable

    mock_instance = mocker.MagicMock(
        spec=RemoteDataAggregatorImpl, name="RemoteDataAggregatorImpl_Instance"
    )
    mock_generic_alias_callable.return_value = mock_instance

    return {
        "mock_class": mock_class,
        "mock_constructor_proxy": mock_generic_alias_callable,
        "mock_instance": mock_instance,
    }


@pytest.fixture
def mock_data_timeout_tracker_fixture(mocker):
    """
    Mocks the DataTimeoutTracker class as used in DataHostBase and its start method.
    Returns the mocked class.
    """
    mock_tracker_class = mocker.patch(
        "tsercom.data.data_host_base.DataTimeoutTracker"
    )

    mock_tracker_instance = mocker.MagicMock(
        spec=DataTimeoutTracker, name="DataTimeoutTracker_Instance"
    )
    mock_tracker_instance.start = mocker.MagicMock(
        name="DataTimeoutTracker_Instance_start"
    )

    mock_tracker_class.return_value = mock_tracker_instance

    # To prevent real DataTimeoutTracker.start from running if any real instance were created,
    # though the class mock above should prevent real instances from DataHostBase.
    # This patches the .start method on the actual DataTimeoutTracker class.
    mocker.patch.object(
        DataTimeoutTracker,
        "start",
        new=mocker.MagicMock(
            name="Original_DataTimeoutTracker_Class_start_Method_Mocked"
        ),
        create=True,
    )

    return mock_tracker_class


def test_data_host_base_initialization_creates_remote_data_aggregator(
    mock_thread_watcher,
    mock_remote_data_aggregator_fixture,
    mock_data_timeout_tracker_fixture,
):
    mock_constructor_proxy = mock_remote_data_aggregator_fixture[
        "mock_constructor_proxy"
    ]
    mock_aggregator_instance = mock_remote_data_aggregator_fixture[
        "mock_instance"
    ]
    mock_tracker_instance = mock_data_timeout_tracker_fixture.return_value

    data_host = DataHostBase[DummyExposedData](
        watcher=mock_thread_watcher, timeout_seconds=10
    )

    expected_thread_pool = (
        mock_thread_watcher.create_tracked_thread_pool_executor.return_value
    )

    mock_constructor_proxy.assert_called_once_with(
        expected_thread_pool,
        None,
        tracker=mock_tracker_instance,
    )
    assert data_host._DataHostBase__aggregator is mock_aggregator_instance
    assert data_host._remote_data_aggregator() is mock_aggregator_instance


def test_data_host_base_initialization_with_timeout_creates_and_starts_timeout_tracker(
    mock_thread_watcher,
    mock_remote_data_aggregator_fixture,
    mock_data_timeout_tracker_fixture,
):
    mock_tracker_class = mock_data_timeout_tracker_fixture
    mock_tracker_instance = mock_tracker_class.return_value

    DataHostBase[DummyExposedData](
        watcher=mock_thread_watcher, timeout_seconds=10
    )

    mock_tracker_class.assert_called_once_with(10)
    mock_tracker_instance.start.assert_called_once()


def test_data_host_base_initialization_without_timeout_does_not_create_timeout_tracker(
    mock_thread_watcher,
    mock_remote_data_aggregator_fixture,
    mock_data_timeout_tracker_fixture,
):
    mock_constructor_proxy = mock_remote_data_aggregator_fixture[
        "mock_constructor_proxy"
    ]
    mock_tracker_class = mock_data_timeout_tracker_fixture

    DataHostBase[DummyExposedData](
        watcher=mock_thread_watcher, timeout_seconds=0
    )
    mock_tracker_class.assert_not_called()
    expected_thread_pool = (
        mock_thread_watcher.create_tracked_thread_pool_executor.return_value
    )
    mock_constructor_proxy.assert_called_once_with(expected_thread_pool, None)

    mock_constructor_proxy.reset_mock()

    DataHostBase[DummyExposedData](
        watcher=mock_thread_watcher, timeout_seconds=-1
    )
    mock_tracker_class.assert_not_called()
    mock_constructor_proxy.assert_called_once_with(expected_thread_pool, None)


def test_data_host_base_on_data_ready_calls_aggregator_on_data_ready(
    mock_thread_watcher,
    mock_remote_data_aggregator_fixture,
    mock_data_timeout_tracker_fixture,
    mock_caller_id,
    mock_datetime_now,
    mocker,
):
    mock_aggregator_instance = mock_remote_data_aggregator_fixture[
        "mock_instance"
    ]

    data_host = DataHostBase[DummyExposedData](
        watcher=mock_thread_watcher, timeout_seconds=0
    )

    test_data = DummyExposedData(
        mocker, caller_id_mock=mock_caller_id, timestamp_mock=mock_datetime_now
    )
    data_host._on_data_ready(test_data)

    mock_aggregator_instance._on_data_ready.assert_called_once_with(test_data)


def test_data_host_base_remote_data_aggregator_property_returns_instance(
    mock_thread_watcher,
    mock_remote_data_aggregator_fixture,
    mock_data_timeout_tracker_fixture,
):
    mock_aggregator_instance = mock_remote_data_aggregator_fixture[
        "mock_instance"
    ]

    data_host = DataHostBase[DummyExposedData](
        watcher=mock_thread_watcher, timeout_seconds=0
    )

    assert data_host._remote_data_aggregator() is mock_aggregator_instance
    assert data_host._DataHostBase__aggregator is mock_aggregator_instance
