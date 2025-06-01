"""Tests for ClientSynchronizedClock."""

import datetime
import pytest

from tsercom.timesync.client.client_synchronized_clock import (
    ClientSynchronizedClock,
)
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from unittest.mock import Mock


class GoodClientImpl(ClientSynchronizedClock.Client):
    """A concrete implementation of ClientSynchronizedClock.Client."""

    def get_offset_seconds(self) -> float:
        return 0.0

    def get_synchronized_clock(self) -> SynchronizedClock:
        return Mock(spec=SynchronizedClock)

    def start_async(self) -> None:
        pass

    def stop(self) -> None:
        pass


class BadClientImpl(ClientSynchronizedClock.Client):
    """Implementation missing get_offset_seconds and other new abstract methods."""

    pass


def test_good_client_impl_instantiation():
    """Tests that a class correctly implementing Client can be instantiated."""
    try:
        client = GoodClientImpl()
        assert isinstance(client, ClientSynchronizedClock.Client)
    except TypeError:
        pytest.fail(
            "GoodClientImpl instantiation raised TypeError unexpectedly."
        )


def test_bad_client_impl_instantiation():
    """Tests that a class missing get_offset_seconds cannot be instantiated."""
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BadClientImpl with abstract methods get_offset_seconds, get_synchronized_clock, start_async, stop",
    ):
        BadClientImpl()


@pytest.fixture
def mock_client(mocker) -> ClientSynchronizedClock.Client:
    """Fixture to create a MagicMock instance of ClientSynchronizedClock.Client."""
    return mocker.MagicMock(spec=ClientSynchronizedClock.Client)


@pytest.fixture
def clock(
    mock_client: ClientSynchronizedClock.Client,
) -> ClientSynchronizedClock:
    """Fixture to create a ClientSynchronizedClock with a mock client."""
    return ClientSynchronizedClock(mock_client)


def test_init_with_client(mock_client: ClientSynchronizedClock.Client):
    """Tests instantiation of ClientSynchronizedClock with a client."""
    try:
        csc = ClientSynchronizedClock(mock_client)
        # Access the name-mangled attribute to verify it was stored
        assert (
            csc._ClientSynchronizedClock__client is mock_client
        )  # pyright: ignore[reportPrivateUsage]
    except Exception as e:  # pragma: no cover
        pytest.fail(f"ClientSynchronizedClock instantiation failed: {e}")


BASE_DT = datetime.datetime(2023, 1, 1, 12, 0, 0)


def test_sync_positive_offset(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests sync with a positive offset."""
    mock_client.get_offset_seconds.return_value = 10.0
    synced_ts = clock.sync(BASE_DT)
    expected_dt = BASE_DT + datetime.timedelta(seconds=10.0)
    assert synced_ts.as_datetime() == expected_dt


def test_sync_negative_offset(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests sync with a negative offset."""
    mock_client.get_offset_seconds.return_value = -5.0
    synced_ts = clock.sync(BASE_DT)
    expected_dt = BASE_DT - datetime.timedelta(seconds=5.0)
    assert synced_ts.as_datetime() == expected_dt


def test_sync_zero_offset(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests sync with a zero offset."""
    mock_client.get_offset_seconds.return_value = 0.0
    synced_ts = clock.sync(BASE_DT)
    assert synced_ts.as_datetime() == BASE_DT


def test_desync_positive_offset(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests desync with a positive offset (client ahead of server)."""
    mock_client.get_offset_seconds.return_value = 10.0
    server_dt = BASE_DT + datetime.timedelta(seconds=10.0)
    synced_ts = SynchronizedTimestamp(server_dt)
    local_dt = clock.desync(synced_ts)
    assert local_dt == BASE_DT


def test_desync_negative_offset(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests desync with a negative offset (client behind server)."""
    mock_client.get_offset_seconds.return_value = -5.0
    server_dt = BASE_DT - datetime.timedelta(seconds=5.0)
    synced_ts = SynchronizedTimestamp(server_dt)
    local_dt = clock.desync(synced_ts)
    assert local_dt == BASE_DT


def test_desync_zero_offset(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests desync with a zero offset."""
    mock_client.get_offset_seconds.return_value = 0.0
    server_dt = BASE_DT
    synced_ts = SynchronizedTimestamp(server_dt)
    local_dt = clock.desync(synced_ts)
    assert local_dt == server_dt


def test_now_property(
    clock: ClientSynchronizedClock, mock_client: ClientSynchronizedClock.Client
):
    """Tests the 'now' property."""
    offset_seconds = 15.0
    mock_client.get_offset_seconds.return_value = offset_seconds

    time_before_sync = datetime.datetime.now() + datetime.timedelta(
        seconds=offset_seconds
    )
    client_now_ts = clock.now
    time_after_sync = datetime.datetime.now() + datetime.timedelta(
        seconds=offset_seconds
    )

    assert isinstance(client_now_ts, SynchronizedTimestamp)
    synced_dt = client_now_ts.as_datetime()

    # Allow a small tolerance for processing time
    # Note: datetime.now() is naive, and ClientSynchronizedClock produces naive UTC-equivalent
    # if the input to sync() is naive (which datetime.now() is).
    # The comparison should be between naive datetimes.
    assert synced_dt >= time_before_sync - datetime.timedelta(
        microseconds=10000
    )
    assert synced_dt <= time_after_sync + datetime.timedelta(
        microseconds=10000
    )
    assert synced_dt.tzinfo is None
