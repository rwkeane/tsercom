"""Tests for ServerSynchronizedClock."""

import datetime
import pytest

from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.timesync.server.server_synchronized_clock import (
    ServerSynchronizedClock,
)


@pytest.fixture
def server_clock() -> ServerSynchronizedClock:
    """Fixture to create a ServerSynchronizedClock instance."""
    return ServerSynchronizedClock()


def test_desync(server_clock: ServerSynchronizedClock):
    """
    Tests that desync() returns the original datetime.datetime object
    from a SynchronizedTimestamp.
    """
    original_dt = datetime.datetime(2023, 10, 26, 12, 0, 0)
    st = SynchronizedTimestamp(original_dt)
    desynced_dt = server_clock.desync(st)
    assert desynced_dt is original_dt


def test_sync(server_clock: ServerSynchronizedClock):
    """
    Tests that sync() wraps a datetime.datetime object in a
    SynchronizedTimestamp.
    """
    original_dt = datetime.datetime(2023, 10, 26, 13, 0, 0)
    synced_st = server_clock.sync(original_dt)

    assert isinstance(synced_st, SynchronizedTimestamp)
    assert synced_st.as_datetime() is original_dt


def test_now_property(server_clock: ServerSynchronizedClock):
    """
    Tests the now property to ensure it returns a SynchronizedTimestamp
    wrapping a recent, naive datetime, consistent with how ServerSynchronizedClock
    should operate (similar to FakeSynchronizedClock as it doesn't apply offsets).
    """
    # Record time just before calling now
    time_before = datetime.datetime.now()

    current_st = server_clock.now
    assert isinstance(current_st, SynchronizedTimestamp)

    # Record time just after calling now
    time_after = datetime.datetime.now()

    dt_obj = current_st.as_datetime()

    # Check that the datetime from the clock is between the times recorded
    # just before and after the call. This allows for minor execution delays.
    # A tolerance of 100ms should be more than enough.
    assert dt_obj >= time_before - datetime.timedelta(microseconds=10000)
    assert dt_obj <= time_after + datetime.timedelta(microseconds=10000)

    # Check that the datetime object is naive (no timezone info)
    assert dt_obj.tzinfo is None
