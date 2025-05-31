"""Tests for FakeSynchronizedClock."""

import datetime
import pytest

from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


@pytest.fixture
def fake_clock() -> FakeSynchronizedClock:
    """Fixture to create a FakeSynchronizedClock instance."""
    return FakeSynchronizedClock()


def test_desync(fake_clock: FakeSynchronizedClock):
    """
    Tests that desync() returns the original datetime.datetime object
    from a SynchronizedTimestamp.
    """
    original_dt = datetime.datetime(2023, 10, 26, 12, 0, 0)
    st = SynchronizedTimestamp(original_dt)
    desynced_dt = fake_clock.desync(st)
    assert desynced_dt is original_dt


def test_sync(fake_clock: FakeSynchronizedClock):
    """
    Tests that sync() wraps a datetime.datetime object in a
    SynchronizedTimestamp.
    """
    original_dt = datetime.datetime(2023, 10, 26, 13, 0, 0)
    synced_st = fake_clock.sync(original_dt)

    assert isinstance(synced_st, SynchronizedTimestamp)
    assert synced_st.as_datetime() is original_dt


def test_now_property(fake_clock: FakeSynchronizedClock):
    """
    Tests the now property to ensure it returns a SynchronizedTimestamp
    wrapping a recent, naive datetime.
    """
    # Record time just before calling now
    time_before = datetime.datetime.now()

    current_st = fake_clock.now
    assert isinstance(current_st, SynchronizedTimestamp)

    # Record time just after calling now
    time_after = datetime.datetime.now()

    dt_obj = current_st.as_datetime()

    # Check that the datetime from the clock is between the times recorded
    # just before and after the call. This allows for minor execution delays.
    assert dt_obj >= time_before
    assert dt_obj <= time_after

    # Check that the datetime object is naive (no timezone info)
    assert dt_obj.tzinfo is None
