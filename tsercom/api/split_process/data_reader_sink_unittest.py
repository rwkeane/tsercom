import pytest

from tsercom.api.split_process.data_reader_sink import DataReaderSink

# from tsercom.data.exposed_data import ExposedData # Not strictly needed if we use generic data for tests
# from tsercom.utils.multiprocess_queue_sink import MultiprocessQueueSink # This is what we're faking


class FakeMultiprocessQueueSink:
    def __init__(self):
        self.put_nowait_called_with = None
        self.put_nowait_call_count = 0
        self._put_nowait_return_value = True  # Default to success

    def put_nowait(self, data):
        self.put_nowait_called_with = data
        self.put_nowait_call_count += 1
        return self._put_nowait_return_value

    def set_put_nowait_return_value(self, return_value: bool):
        self._put_nowait_return_value = return_value


@pytest.fixture
def fake_queue():
    return FakeMultiprocessQueueSink()


@pytest.fixture
def test_data():
    return "test_payload"


# Tests for _on_data_ready() with is_lossy = True (default)
def test_on_data_ready_lossy_queue_not_full(fake_queue, test_data):
    """
    Case 1: is_lossy=True, Queue Not Full
    - Configure FakeMultiprocessQueueSink.put_nowait to return True.
    - Instantiate DataReaderSink with the fake queue.
    - Call sink._on_data_ready(test_data).
    - Verify queue.put_nowait was called with test_data.
    - No exception should be raised.
    """
    fake_queue.set_put_nowait_return_value(True)
    sink = DataReaderSink[str](
        fake_queue
    )  # Explicitly typing with str for TDataType

    sink._on_data_ready(test_data)

    assert fake_queue.put_nowait_call_count == 1
    assert fake_queue.put_nowait_called_with == test_data
    # No exception expected


def test_on_data_ready_lossy_queue_full(fake_queue, test_data):
    """
    Case 2: is_lossy=True, Queue Full
    - Configure FakeMultiprocessQueueSink.put_nowait to return False.
    - Instantiate DataReaderSink with the fake queue.
    - Call sink._on_data_ready(test_data).
    - Verify queue.put_nowait was called with test_data.
    - No exception should be raised (data is lost).
    """
    fake_queue.set_put_nowait_return_value(False)
    sink = DataReaderSink[str](fake_queue)

    sink._on_data_ready(test_data)

    assert fake_queue.put_nowait_call_count == 1
    assert fake_queue.put_nowait_called_with == test_data
    # No exception expected, data is lost


# Tests for _on_data_ready() with is_lossy = False
def test_on_data_ready_not_lossy_queue_not_full(fake_queue, test_data):
    """
    Case 1: is_lossy=False, Queue Not Full
    - Configure FakeMultiprocessQueueSink.put_nowait to return True.
    - Instantiate DataReaderSink with is_lossy=False and the fake queue.
    - Call sink._on_data_ready(test_data).
    - Verify queue.put_nowait was called with test_data.
    - No exception should be raised.
    """
    fake_queue.set_put_nowait_return_value(True)
    sink = DataReaderSink[str](fake_queue, is_lossy=False)

    sink._on_data_ready(test_data)

    assert fake_queue.put_nowait_call_count == 1
    assert fake_queue.put_nowait_called_with == test_data
    # No exception expected


def test_on_data_ready_not_lossy_queue_full_raises_assertion_error(
    fake_queue, test_data
):
    """
    Case 2: is_lossy=False, Queue Full
    - Configure FakeMultiprocessQueueSink.put_nowait to return False.
    - Instantiate DataReaderSink with is_lossy=False and the fake queue.
    - Use pytest.raises(AssertionError) to verify that calling sink._on_data_ready(test_data) raises an AssertionError.
    - Verify queue.put_nowait was called with test_data.
    """
    fake_queue.set_put_nowait_return_value(False)
    sink = DataReaderSink[str](fake_queue, is_lossy=False)

    with pytest.raises(
        RuntimeError,
        match="Queue full; data would be lost on non-lossy sink.",
    ):
        sink._on_data_ready(test_data)

    assert fake_queue.put_nowait_call_count == 1
    assert fake_queue.put_nowait_called_with == test_data


# Test constructor default for is_lossy
def test_constructor_default_is_lossy(fake_queue):
    sink = DataReaderSink[str](fake_queue)
    assert (
        sink._DataReaderSink__is_lossy is True
    )  # Accessing private for verification


def test_constructor_is_lossy_false(fake_queue):
    sink = DataReaderSink[str](fake_queue, is_lossy=False)
    assert (
        sink._DataReaderSink__is_lossy is False
    )  # Accessing private for verification
