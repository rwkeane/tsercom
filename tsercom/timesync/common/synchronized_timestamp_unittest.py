"""Tests for SynchronizedTimestamp."""

import datetime

import pytest
from google.protobuf import timestamp_pb2

# Corrected import to match how ServerTimestamp is exposed by the package
from tsercom.timesync.common.proto import (
    ServerTimestamp,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


# A fixed datetime object for consistent tests
NOW_DATETIME = datetime.datetime.utcnow()  # Naive UTC datetime
# Per SynchronizedTimestamp docstring, it expects naive datetimes.
# Let's assume these naive datetimes are implicitly UTC.
FIXED_DATETIME = datetime.datetime(
    2023, 10, 26, 12, 0, 0
)  # Naive, assumed UTC


def test_init_valid():
    """Tests initialization with a valid datetime object."""
    st = SynchronizedTimestamp(FIXED_DATETIME)
    assert st.as_datetime() is FIXED_DATETIME


def test_init_none():
    """Tests that an TypeError is raised if None is passed to __init__."""
    with pytest.raises(TypeError, match="Timestamp cannot be None."):
        SynchronizedTimestamp(None)  # pytype: disable=wrong-arg-types


def test_init_invalid_type():
    """Tests that an TypeError is raised if an invalid type is passed to __init__."""
    with pytest.raises(
        TypeError, match="Timestamp must be a datetime.datetime object."
    ):
        SynchronizedTimestamp(12345)  # pytype: disable=wrong-arg-types


def test_as_datetime():
    """Tests that as_datetime() returns the original datetime object."""
    st = SynchronizedTimestamp(FIXED_DATETIME)
    assert st.as_datetime() is FIXED_DATETIME


def test_to_grpc_type(mocker):
    """Tests conversion to the gRPC ServerTimestamp type."""
    st = SynchronizedTimestamp(FIXED_DATETIME)
    server_ts = st.to_grpc_type()

    assert isinstance(server_ts, ServerTimestamp)
    assert isinstance(server_ts.timestamp, timestamp_pb2.Timestamp)

    # Verify that the conversion back from the gRPC Timestamp matches the original datetime.
    # server_ts.timestamp.ToDatetime() returns a naive datetime representing UTC.
    # FIXED_DATETIME is now also a naive datetime representing UTC.
    converted_datetime = server_ts.timestamp.ToDatetime()
    assert converted_datetime == FIXED_DATETIME


# --- Tests for try_parse ---


def test_try_parse_none():
    """Tests try_parse with None input."""
    assert SynchronizedTimestamp.try_parse(None) is None


def test_try_parse_grpc_timestamp(mocker):
    """Tests try_parse with a valid google.protobuf.timestamp_pb2.Timestamp."""
    mock_grpc_ts = mocker.MagicMock(spec=timestamp_pb2.Timestamp)
    expected_datetime = datetime.datetime(2023, 1, 1, 10, 0, 0)
    mock_grpc_ts.ToDatetime.return_value = expected_datetime

    result = SynchronizedTimestamp.try_parse(mock_grpc_ts)
    assert isinstance(result, SynchronizedTimestamp)
    assert result.as_datetime() == expected_datetime
    mock_grpc_ts.ToDatetime.assert_called_once()


def test_try_parse_server_timestamp():
    """Tests try_parse with a valid ServerTimestamp."""
    expected_datetime = datetime.datetime(2023, 1, 1, 11, 0, 0)

    # Create a real inner google.protobuf.timestamp_pb2.Timestamp
    real_inner_ts = timestamp_pb2.Timestamp()
    real_inner_ts.FromDatetime(expected_datetime)

    # Create the ServerTimestamp wrapper
    server_ts_wrapper = ServerTimestamp()
    server_ts_wrapper.timestamp.CopyFrom(
        real_inner_ts
    )  # Use CopyFrom for message assignment

    result = SynchronizedTimestamp.try_parse(server_ts_wrapper)
    assert isinstance(result, SynchronizedTimestamp)
    assert result.as_datetime() == expected_datetime


def test_try_parse_grpc_timestamp_todatetime_value_error(mocker, caplog):
    """
    Tests try_parse with a google.protobuf.timestamp_pb2.Timestamp whose
    ToDatetime() method raises ValueError.
    """
    mock_grpc_ts = mocker.MagicMock(spec=timestamp_pb2.Timestamp)
    mock_grpc_ts.ToDatetime.side_effect = ValueError("Test ToDatetime error")

    result = SynchronizedTimestamp.try_parse(mock_grpc_ts)
    assert result is None
    mock_grpc_ts.ToDatetime.assert_called_once()
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        "Failed to parse gRPC Timestamp to datetime: Test ToDatetime error"
        in caplog.text
    )


def test_try_parse_invalid_type_raises_assertion_error():
    """
    Tests that try_parse raises an TypeError if an unexpected type
    (that is not None, Timestamp, or ServerTimestamp) is passed.
    """
    with pytest.raises(
        TypeError,
        match="Input must be a gRPC Timestamp or resolve to one.",
    ):
        SynchronizedTimestamp.try_parse(
            12345
        )  # pytype: disable=wrong-arg-types


# --- Tests for comparison methods ---

# Create some fixed SynchronizedTimestamp instances for comparison tests
ST_PAST = SynchronizedTimestamp(FIXED_DATETIME - datetime.timedelta(days=1))
ST_FIXED = SynchronizedTimestamp(
    FIXED_DATETIME
)  # Same as FIXED_DATETIME used in earlier tests
ST_FUTURE = SynchronizedTimestamp(FIXED_DATETIME + datetime.timedelta(days=1))

# Corresponding datetime objects
DT_PAST = ST_PAST.as_datetime()
DT_FIXED = ST_FIXED.as_datetime()
DT_FUTURE = ST_FUTURE.as_datetime()


def test_comparison_st_with_st():
    """Tests comparison methods between SynchronizedTimestamp instances."""
    assert ST_FIXED > ST_PAST
    assert not (ST_FIXED > ST_FUTURE)
    assert ST_FIXED >= ST_PAST
    assert ST_FIXED >= ST_FIXED
    assert not (ST_FIXED >= ST_FUTURE)

    assert ST_FIXED < ST_FUTURE
    assert not (ST_FIXED < ST_PAST)
    assert ST_FIXED <= ST_FUTURE
    assert ST_FIXED <= ST_FIXED
    assert not (ST_FIXED <= ST_PAST)

    assert ST_FIXED == ST_FIXED
    assert ST_FIXED == SynchronizedTimestamp(
        FIXED_DATETIME
    )  # Different instance, same time
    assert not (ST_FIXED == ST_PAST)

    assert ST_FIXED != ST_PAST
    assert not (ST_FIXED != ST_FIXED)


def test_comparison_st_with_datetime():
    """Tests comparison methods between SynchronizedTimestamp and datetime instances."""
    assert ST_FIXED > DT_PAST
    assert not (ST_FIXED > DT_FUTURE)
    assert ST_FIXED >= DT_PAST
    assert ST_FIXED >= DT_FIXED
    assert not (ST_FIXED >= DT_FUTURE)

    assert ST_FIXED < DT_FUTURE
    assert not (ST_FIXED < DT_PAST)
    assert ST_FIXED <= DT_FUTURE
    assert ST_FIXED <= DT_FIXED
    assert not (ST_FIXED <= DT_PAST)

    assert ST_FIXED == DT_FIXED
    assert not (ST_FIXED == DT_PAST)

    assert ST_FIXED != DT_PAST
    assert not (ST_FIXED != DT_FIXED)


def test_comparison_datetime_with_st():
    """Tests comparison (equality/inequality) with datetime on the left."""
    # Note: >, <, >=, <= are not defined if datetime is on the left,
    # as datetime.__gt__ etc. won't know about SynchronizedTimestamp.
    # However, __eq__ and __ne__ should work due to Python's fallback mechanism.
    assert DT_FIXED == ST_FIXED
    assert not (DT_FIXED == ST_PAST)

    assert DT_FIXED != ST_PAST
    assert not (DT_FIXED != ST_FIXED)


def test_equality_with_other_types():
    """Tests equality comparison with unrelated types."""
    assert not (ST_FIXED == "some_string")
    assert ST_FIXED != "some_string"
    assert not (ST_FIXED == 12345)
    assert ST_FIXED != 12345
    assert not (ST_FIXED == None)  # pylint: disable=singleton-comparison
    assert ST_FIXED != None  # pylint: disable=singleton-comparison


def test_comparison_invalid_type_raises_assertion_error():
    """
    Tests that comparison with an unsupported type in >, >=, <, <=
    raises an TypeError.
    """
    # Use regex to match the type name at the end of the error message
    err_match = r"Compare error: SynchronizedTimestamp/datetime vs \w+"
    with pytest.raises(TypeError, match=err_match):
        _ = ST_FIXED > "invalid_type"  # type: ignore
    with pytest.raises(TypeError, match=err_match):
        _ = ST_FIXED >= "invalid_type"  # type: ignore
    with pytest.raises(TypeError, match=err_match):
        _ = ST_FIXED < "invalid_type"  # type: ignore
    with pytest.raises(TypeError, match=err_match):
        _ = ST_FIXED <= "invalid_type"  # type: ignore
