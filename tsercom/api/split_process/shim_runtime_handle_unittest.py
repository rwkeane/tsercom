import pytest

import datetime

# Module to be tested & whose attributes will be patched
import tsercom.api.split_process.shim_runtime_handle as shim_module
from tsercom.api.split_process.shim_runtime_handle import ShimRuntimeHandle
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.data.event_instance import EventInstance
from tsercom.caller_id.caller_identifier import (
    CallerIdentifier,
)

# --- Fake Classes for Dependencies ---


class FakeThreadWatcher:
    def __init__(self):
        self.name = "FakeThreadWatcher"  # For debugging


class FakeMultiprocessQueueSink:
    def __init__(self, name="FakeQueueSink"):
        self.name = name
        self.put_blocking_called_with = None
        self.put_blocking_call_count = 0
        self.put_nowait_called_with = None
        self.put_nowait_call_count = 0
        self._put_nowait_return_value = True  # Default to success

    def put_blocking(
        self, data, timeout=None
    ):  # ShimRuntimeHandle uses default timeout
        self.put_blocking_called_with = data
        self.put_blocking_call_count += 1
        # Simulate success, real queue might raise Full

    def put_nowait(self, data):
        self.put_nowait_called_with = data
        self.put_nowait_call_count += 1
        return self._put_nowait_return_value  # Real queue returns bool

    def set_put_nowait_return_value(self, return_value: bool):
        self._put_nowait_return_value = return_value


class FakeMultiprocessQueueSource:
    def __init__(self, name="FakeQueueSource"):
        self.name = name
        # Not used by ShimRuntimeHandle directly, but DataReaderSource uses it


class FakeRemoteDataAggregatorImpl:
    def __init__(self):
        self.on_data_ready_called_with = None
        self.on_data_ready_call_count = 0

    def _on_data_ready(self, new_data):
        self.on_data_ready_called_with = new_data
        self.on_data_ready_call_count += 1


# --- Fake for DataReaderSource (to be patched) ---
g_fake_data_reader_source_instances = []  # Global to track instances


class FakeDataReaderSource:
    def __init__(self, watcher, queue, data_reader):
        self.watcher = watcher
        self.queue = queue
        self.data_reader = data_reader
        self.start_called = False
        self.start_call_count = 0
        self.stop_called = False
        self.stop_call_count = 0
        g_fake_data_reader_source_instances.append(self)

    def start(self):
        self.start_called = True
        self.start_call_count += 1

    def stop(self):
        self.stop_called = True
        self.stop_call_count += 1

    @classmethod
    def get_last_instance(cls):
        return (
            g_fake_data_reader_source_instances[-1]
            if g_fake_data_reader_source_instances
            else None
        )

    @classmethod
    def clear_instances(cls):
        global g_fake_data_reader_source_instances
        g_fake_data_reader_source_instances = []


# --- Pytest Fixtures ---


@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_event_q_sink():
    return FakeMultiprocessQueueSink(name="EventQSink")


@pytest.fixture
def fake_data_q_source():
    return FakeMultiprocessQueueSource(name="DataQSource")


@pytest.fixture
def fake_command_q_sink():
    return FakeMultiprocessQueueSink(name="CommandQSink")


@pytest.fixture
def fake_aggregator():
    return FakeRemoteDataAggregatorImpl()


@pytest.fixture(autouse=True)  # Ensure instance list is cleared for each test
def clear_drs_instances():
    FakeDataReaderSource.clear_instances()


@pytest.fixture
def patch_data_reader_source_in_shim_module(request):
    """Monkeypatches DataReaderSource in the shim_runtime_handle module's namespace."""
    original_class = getattr(shim_module, "DataReaderSource", None)
    setattr(shim_module, "DataReaderSource", FakeDataReaderSource)

    def cleanup():
        if original_class:
            setattr(shim_module, "DataReaderSource", original_class)
        elif hasattr(shim_module, "DataReaderSource"):  # If we added it
            delattr(shim_module, "DataReaderSource")

    request.addfinalizer(cleanup)


@pytest.fixture
def handle_block_true(
    fake_watcher,
    fake_event_q_sink,
    fake_data_q_source,
    fake_command_q_sink,
    fake_aggregator,
    patch_data_reader_source_in_shim_module,
):
    # patch_data_reader_source_in_shim_module ensures FakeDataReaderSource is used
    return ShimRuntimeHandle[
        str, str
    ](  # Assuming str for TDataType and TEventType for simplicity
        thread_watcher=fake_watcher,
        event_queue=fake_event_q_sink,
        data_queue=fake_data_q_source,
        runtime_command_queue=fake_command_q_sink,
        data_aggregator=fake_aggregator,
        block=True,
    )


@pytest.fixture
def handle_block_false(
    fake_watcher,
    fake_event_q_sink,
    fake_data_q_source,
    fake_command_q_sink,
    fake_aggregator,
    patch_data_reader_source_in_shim_module,
):
    return ShimRuntimeHandle[str, str](
        thread_watcher=fake_watcher,
        event_queue=fake_event_q_sink,
        data_queue=fake_data_q_source,
        runtime_command_queue=fake_command_q_sink,
        data_aggregator=fake_aggregator,
        block=False,  # Explicitly false, also default
    )


@pytest.fixture
def test_event_data():
    return "sample_event"


@pytest.fixture
def test_exposed_data():
    return "sample_exposed_data"  # Simple string for testing


# --- Unit Tests ---


def test_init(
    fake_watcher,
    fake_event_q_sink,
    fake_data_q_source,
    fake_command_q_sink,
    fake_aggregator,
    patch_data_reader_source_in_shim_module,
):
    """Test ShimRuntimeHandle.__init__."""
    # Instantiate within the test to ensure patch is active via fixture argument
    handle = ShimRuntimeHandle(
        thread_watcher=fake_watcher,
        event_queue=fake_event_q_sink,
        data_queue=fake_data_q_source,
        runtime_command_queue=fake_command_q_sink,
        data_aggregator=fake_aggregator,
        block=True,  # Explicit block value for this test instance
    )

    # thread_watcher is not stored directly on ShimRuntimeHandle, its usage is verified via DataReaderSource init
    assert handle._ShimRuntimeHandle__event_queue is fake_event_q_sink
    assert (
        handle._ShimRuntimeHandle__runtime_command_queue is fake_command_q_sink
    )
    assert (
        handle._ShimRuntimeHandle__data_aggregator is fake_aggregator
    )  # Corrected attribute name (with typo)
    assert handle._ShimRuntimeHandle__block is True  # Corrected attribute name

    # Verify DataReaderSource instantiation
    drs_instance = FakeDataReaderSource.get_last_instance()
    assert drs_instance is not None
    assert drs_instance.watcher is fake_watcher
    assert drs_instance.queue is fake_data_q_source
    assert (
        drs_instance.data_reader is fake_aggregator
    )  # ShimRuntimeHandle passes its aggregator
    assert (
        handle._ShimRuntimeHandle__data_reader_source is drs_instance
    )  # Corrected attribute name


def test_start(
    handle_block_false, fake_command_q_sink
):  # Using handle_block_false, block doesn't matter for start
    """Test ShimRuntimeHandle.start()."""
    drs_instance = (
        FakeDataReaderSource.get_last_instance()
    )  # Get instance created by fixture
    assert (
        drs_instance is not None
    ), "FakeDataReaderSource not instantiated by handle fixture"

    handle_block_false.start()

    assert drs_instance.start_called
    assert drs_instance.start_call_count == 1
    assert fake_command_q_sink.put_blocking_call_count == 1
    assert fake_command_q_sink.put_blocking_called_with == RuntimeCommand.START


def test_on_event_block_true(
    handle_block_true, fake_event_q_sink, test_event_data
):
    """Test on_event() when block is True."""
    handle_block_true.on_event(test_event_data)

    assert fake_event_q_sink.put_blocking_call_count == 1
    # assert fake_event_q_sink.put_blocking_called_with == test_event_data # Old assertion
    called_event_instance = fake_event_q_sink.put_blocking_called_with
    assert isinstance(called_event_instance, EventInstance)
    assert called_event_instance.data == test_event_data
    assert (
        called_event_instance.caller_id is None
    )  # As on_event was called with default caller_id
    assert isinstance(
        called_event_instance.timestamp, datetime.datetime
    )  # Check type of timestamp
    assert fake_event_q_sink.put_nowait_call_count == 0


def test_on_event_block_false(
    handle_block_false, fake_event_q_sink, test_event_data
):
    """Test on_event() when block is False."""
    fake_event_q_sink.set_put_nowait_return_value(True)  # Simulate success
    handle_block_false.on_event(test_event_data)

    assert fake_event_q_sink.put_nowait_call_count == 1
    # assert fake_event_q_sink.put_nowait_called_with == test_event_data # Old assertion
    called_event_instance = fake_event_q_sink.put_nowait_called_with
    assert isinstance(called_event_instance, EventInstance)
    assert called_event_instance.data == test_event_data
    assert called_event_instance.caller_id is None
    assert isinstance(called_event_instance.timestamp, datetime.datetime)
    assert fake_event_q_sink.put_blocking_call_count == 0


def test_on_event_block_false_queue_full(
    handle_block_false, fake_event_q_sink, test_event_data
):
    """Test on_event() when block is False and queue is full (put_nowait returns False)."""
    fake_event_q_sink.set_put_nowait_return_value(False)  # Simulate queue full

    # ShimRuntimeHandle's on_event for block=False does not check the return of put_nowait,
    # so no exception is expected here from ShimRuntimeHandle itself.
    # The real MultiprocessQueueSink might log, but ShimRuntimeHandle doesn't act on the bool.
    handle_block_false.on_event(test_event_data)

    assert fake_event_q_sink.put_nowait_call_count == 1
    # assert fake_event_q_sink.put_nowait_called_with == test_event_data # Old assertion
    called_event_instance = fake_event_q_sink.put_nowait_called_with
    assert isinstance(called_event_instance, EventInstance)
    assert called_event_instance.data == test_event_data
    assert called_event_instance.caller_id is None
    assert isinstance(called_event_instance.timestamp, datetime.datetime)
    assert fake_event_q_sink.put_blocking_call_count == 0


def test_stop(
    handle_block_false, fake_command_q_sink
):  # Using handle_block_false, block doesn't matter for stop
    """Test ShimRuntimeHandle.stop()."""
    drs_instance = FakeDataReaderSource.get_last_instance()
    assert drs_instance is not None

    handle_block_false.stop()

    assert fake_command_q_sink.put_blocking_call_count == 1
    assert fake_command_q_sink.put_blocking_called_with == RuntimeCommand.STOP
    assert drs_instance.stop_called
    assert drs_instance.stop_call_count == 1


def test_on_data_ready(handle_block_false, fake_aggregator, test_exposed_data):
    """Test ShimRuntimeHandle._on_data_ready()."""
    # _on_data_ready is a method of RemoteDataAggregator.Client, which ShimRuntimeHandle implements.
    # However, the prompt implies it's a method on ShimRuntimeHandle itself.
    # Assuming it's: handle_block_false._data_source._on_data_ready(test_exposed_data)
    # No, ShimRuntimeHandle passes its aggregator to DataReaderSource,
    # DataReaderSource calls aggregator._on_data_ready.
    # The test should be: ShimRuntimeHandle IS A RemoteDataAggregator.Client
    # So, ShimRuntimeHandle._on_data_ready IS the method to test.

    handle_block_false._on_data_ready(
        test_exposed_data
    )  # Call the method on ShimRuntimeHandle which calls self.__data_aggregtor._on_data_ready

    assert fake_aggregator.on_data_ready_call_count == 1
    assert fake_aggregator.on_data_ready_called_with == test_exposed_data


def test_get_remote_data_aggregator(handle_block_false, fake_aggregator):
    """Test ShimRuntimeHandle._get_remote_data_aggregator()."""
    retrieved_aggregator = handle_block_false._get_remote_data_aggregator()
    assert retrieved_aggregator is fake_aggregator


def test_on_event_with_caller_id_and_timestamp(
    handle_block_false, fake_event_q_sink, test_event_data, mocker
):
    """
    Tests on_event() with specific caller_id and timestamp.
    """
    mock_caller_id = CallerIdentifier.random()
    # Ensure timezone-aware datetime for consistency, as EventInstance might create one
    mock_timestamp = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(seconds=30)

    handle_block_false.on_event(
        test_event_data, caller_id=mock_caller_id, timestamp=mock_timestamp
    )

    # Since block=False, put_nowait should be called
    # fake_event_q_sink.put_blocking.assert_not_called() # This would fail if put_blocking was called for some reason
    assert fake_event_q_sink.put_blocking_call_count == 0  # More direct check
    assert fake_event_q_sink.put_nowait_call_count == 1

    # If FakeMultiprocessQueueSink directly stores the arg in an attribute:
    captured_event_instance = fake_event_q_sink.put_nowait_called_with

    assert isinstance(
        captured_event_instance, EventInstance
    ), "Captured object must be an EventInstance"
    assert (
        captured_event_instance.data == test_event_data
    ), "EventInstance data does not match"
    assert (
        captured_event_instance.caller_id is mock_caller_id
    ), "EventInstance caller_id does not match"
    assert (
        captured_event_instance.timestamp == mock_timestamp
    ), "EventInstance timestamp does not match"
