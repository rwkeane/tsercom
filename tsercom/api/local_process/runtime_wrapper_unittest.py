import pytest
import datetime
import importlib

from tsercom.api.local_process.runtime_wrapper import RuntimeWrapper
from tsercom.data.event_instance import EventInstance
from tsercom.caller_id.caller_identifier import (
    CallerIdentifier,
)  # For creating test_caller_id


class FakeAsyncPoller:
    def __init__(self):
        self.on_available_called = False
        self.on_available_arg = None
        self.on_available_call_count = 0

    def on_available(self, event_instance):
        self.on_available_called = True
        self.on_available_arg = event_instance
        self.on_available_call_count += 1


class FakeRemoteDataAggregatorImpl:
    def __init__(self):
        self.on_data_ready_called = False
        self.on_data_ready_arg = None
        self.on_data_ready_call_count = 0

    def _on_data_ready(self, new_data):
        self.on_data_ready_called = True
        self.on_data_ready_arg = new_data
        self.on_data_ready_call_count += 1


class FakeRuntimeCommandBridge:
    def __init__(self):
        self.start_called = False
        self.start_call_count = 0
        self.stop_called = False
        self.stop_call_count = 0

    def start(self):
        self.start_called = True
        self.start_call_count += 1

    def stop(self):
        self.stop_called = True
        self.stop_call_count += 1


@pytest.fixture
def fake_poller():
    return FakeAsyncPoller()


@pytest.fixture
def fake_aggregator():
    return FakeRemoteDataAggregatorImpl()


@pytest.fixture
def fake_bridge():
    return FakeRuntimeCommandBridge()


@pytest.fixture
def wrapper(fake_poller, fake_aggregator, fake_bridge):
    return RuntimeWrapper(
        event_poller=fake_poller,
        data_aggregator=fake_aggregator,
        bridge=fake_bridge,
    )


@pytest.fixture
def patch_datetime_now(request):
    """Fixture to monkeypatch datetime.datetime.now."""
    mock_now = datetime.datetime(2024, 1, 1, 12, 0, 0)  # A fixed point in time

    # Patch datetime.datetime.now
    # We need to patch it where it's looked up. RuntimeWrapper imports datetime directly.
    """Fixture to monkeypatch datetime.datetime.now by replacing the datetime class in the target module."""
    mock_now_fixed_time = datetime.datetime(2024, 1, 1, 12, 0, 0)

    # Path to the module where 'datetime.datetime' is used
    module_path = "tsercom.api.local_process.runtime_wrapper"

    # Import the module
    target_module = importlib.import_module(module_path)

    # Store the original datetime class from the target module
    original_datetime_in_module = target_module.datetime

    # Create a fake datetime class with a patched now()
    class PatchedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz:  # pragma: no cover
                return mock_now_fixed_time.replace(tzinfo=tz)
            return mock_now_fixed_time

    # Apply the patch: replace 'datetime' in the target_module's namespace
    target_module.datetime = PatchedDatetime

    def restore():
        # Restore the original datetime class in the target module
        target_module.datetime = original_datetime_in_module

    request.addfinalizer(restore)

    return mock_now_fixed_time


# Test __init__
def test_init(wrapper, fake_poller, fake_aggregator, fake_bridge):
    assert wrapper._RuntimeWrapper__event_poller is fake_poller
    assert wrapper._RuntimeWrapper__aggregator is fake_aggregator
    assert wrapper._RuntimeWrapper__bridge is fake_bridge


# Test start()
def test_start(wrapper, fake_bridge):
    wrapper.start()
    assert fake_bridge.start_called
    assert fake_bridge.start_call_count == 1


# Test stop()
def test_stop(wrapper, fake_bridge):
    wrapper.stop()
    assert fake_bridge.stop_called
    assert fake_bridge.stop_call_count == 1


# Tests for on_event()
def test_on_event_only_event(wrapper, fake_poller, patch_datetime_now):
    test_event_data = "my_event_data"
    wrapper.on_event(test_event_data)

    assert fake_poller.on_available_called
    assert fake_poller.on_available_call_count == 1
    event_instance = fake_poller.on_available_arg
    assert isinstance(event_instance, EventInstance)
    assert event_instance.data == test_event_data
    assert event_instance.caller_id is None
    assert (
        event_instance.timestamp == patch_datetime_now
    )  # Exact match due to patching


def test_on_event_with_caller_id(wrapper, fake_poller, patch_datetime_now):
    test_event_data = "event_with_caller"
    test_caller_id = CallerIdentifier.random()  # Corrected initialization
    wrapper.on_event(test_event_data, test_caller_id)

    assert fake_poller.on_available_called
    event_instance = fake_poller.on_available_arg
    assert isinstance(event_instance, EventInstance)
    assert event_instance.data == test_event_data
    assert event_instance.caller_id is test_caller_id
    assert event_instance.timestamp == patch_datetime_now


def test_on_event_with_explicit_timestamp(wrapper, fake_poller):
    test_event_data = "event_explicit_time"
    fixed_timestamp = datetime.datetime(2023, 1, 1, 0, 0, 0)
    wrapper.on_event(test_event_data, timestamp=fixed_timestamp)

    assert fake_poller.on_available_called
    event_instance = fake_poller.on_available_arg
    assert isinstance(event_instance, EventInstance)
    assert event_instance.data == test_event_data
    assert event_instance.caller_id is None
    assert event_instance.timestamp == fixed_timestamp


def test_on_event_with_caller_id_and_timestamp(wrapper, fake_poller):
    test_event_data = "event_all_args"
    test_caller_id = CallerIdentifier.random()  # Corrected initialization
    fixed_timestamp = datetime.datetime(2022, 1, 1, 0, 0, 0)
    wrapper.on_event(
        test_event_data, test_caller_id, timestamp=fixed_timestamp
    )

    assert fake_poller.on_available_called
    event_instance = fake_poller.on_available_arg
    assert isinstance(event_instance, EventInstance)
    assert event_instance.data == test_event_data
    assert event_instance.caller_id is test_caller_id
    assert event_instance.timestamp == fixed_timestamp


# Test _on_data_ready()
def test_on_data_ready(wrapper, fake_aggregator):
    test_data_payload = "my_test_payload"  # Assuming ExposedData or similar can be faked with a string for test
    wrapper._on_data_ready(test_data_payload)

    assert fake_aggregator.on_data_ready_called
    assert fake_aggregator.on_data_ready_call_count == 1
    assert fake_aggregator.on_data_ready_arg == test_data_payload


# Test _get_remote_data_aggregator()
def test_get_remote_data_aggregator(wrapper, fake_aggregator):
    retrieved_aggregator = wrapper._get_remote_data_aggregator()
    assert retrieved_aggregator is fake_aggregator
