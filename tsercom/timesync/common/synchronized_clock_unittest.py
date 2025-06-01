"""Tests for the SynchronizedClock ABC."""

import datetime
import pytest
from unittest.mock import MagicMock  # For spy return value comparison

from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


# --- Test Implementations for ABC Contract ---


class GoodClock(SynchronizedClock):
    """A concrete implementation of SynchronizedClock with all methods."""

    def sync(self, timestamp: datetime.datetime) -> SynchronizedTimestamp:
        return SynchronizedTimestamp(timestamp)

    def desync(self, time: SynchronizedTimestamp) -> datetime.datetime:
        return time.as_datetime()


class BadClockNoSync(SynchronizedClock):
    """Implementation missing the 'sync' method."""

    def desync(self, time: SynchronizedTimestamp) -> datetime.datetime:
        return time.as_datetime()


class BadClockNoDesync(SynchronizedClock):
    """Implementation missing the 'desync' method."""

    def sync(self, timestamp: datetime.datetime) -> SynchronizedTimestamp:
        return SynchronizedTimestamp(timestamp)


class BadClockNoMethods(SynchronizedClock):
    """Implementation missing all abstract methods."""

    pass


def test_good_clock_instantiation():
    """Tests that a class correctly implementing all abstract methods can be instantiated."""
    try:
        clock = GoodClock()
        assert isinstance(clock, SynchronizedClock)
    except TypeError:
        pytest.fail("GoodClock instantiation raised TypeError unexpectedly.")


def test_bad_clock_no_sync_instantiation():
    """Tests that a class missing 'sync' cannot be instantiated."""
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BadClockNoSync with abstract method sync",
    ):
        BadClockNoSync()


def test_bad_clock_no_desync_instantiation():
    """Tests that a class missing 'desync' cannot be instantiated."""
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BadClockNoDesync with abstract method desync",
    ):
        BadClockNoDesync()


def test_bad_clock_no_methods_instantiation():
    """Tests that a class missing all abstract methods cannot be instantiated."""
    # The error message might list one or all missing methods depending on Python version/implementation details
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BadClockNoMethods with abstract methods desync, sync",
    ):
        BadClockNoMethods()


# --- Test for 'now' Property ---


def test_now_property_calls_sync(mocker):
    """Tests that the 'now' property correctly calls the 'sync' method."""
    clock = GoodClock()

    # Spy on the sync method to check its arguments and return value
    # We also need to control its return value for assertion
    expected_synced_time = SynchronizedTimestamp(
        datetime.datetime.now(tz=datetime.timezone.utc)
    )
    mocker.patch.object(clock, "sync", return_value=expected_synced_time)

    time_before = datetime.datetime.now()
    current_sync_time = clock.now
    time_after = datetime.datetime.now()

    # Assert that sync was called once
    clock.sync.assert_called_once()  # type: ignore

    # Check the argument passed to sync
    call_args = clock.sync.call_args[0]  # type: ignore
    assert len(call_args) == 1
    passed_datetime = call_args[0]
    assert isinstance(passed_datetime, datetime.datetime)

    # The datetime passed to sync should be very close to now()
    # It should be naive as datetime.datetime.now() is naive.
    assert passed_datetime >= time_before
    assert passed_datetime <= time_after
    assert passed_datetime.tzinfo is None

    # Verify that the result of 'now' is what 'sync' returned
    assert current_sync_time is expected_synced_time
