# pylint: disable=no-name-in-module
"""Defines SynchronizedTimestamp for time-offset-aware timestamps."""

import dataclasses
import datetime
import logging  # Preserved as it's used by try_parse
from typing import Optional, TypeAlias, Union

from google.protobuf.timestamp_pb2 import Timestamp

from tsercom.timesync.common.proto import ServerTimestamp

TimestampType: TypeAlias = Union["SynchronizedTimestamp", datetime.datetime]


@dataclasses.dataclass(
    eq=False, order=False, unsafe_hash=False
)  # Retaining custom eq/order
class SynchronizedTimestamp:
    """
    A wrapper around a `datetime.datetime` object to represent a timestamp
    within a synchronized time context.

    This class ensures timestamps are explicitly handled as part of a
    synchronized system (e.g., aligned with a server's clock). It provides
    methods for converting to/from other timestamp representations like
    gRPC `Timestamp` and `ServerTimestamp` protobuf messages.

    It also supports direct comparison with other `SynchronizedTimestamp`
    objects or naive `datetime.datetime` objects. When comparing, it unwraps
    the underlying `datetime.datetime` object.
    """

    timestamp: datetime.datetime

    def __post_init__(self) -> None:
        """Performs post-initialization validation."""
        if self.timestamp is None:
            raise TypeError("Timestamp cannot be None.")
        if not isinstance(self.timestamp, datetime.datetime):
            raise TypeError("Timestamp must be a datetime.datetime object.")

    @classmethod
    def try_parse(
        cls, other: Timestamp | ServerTimestamp
    ) -> Optional["SynchronizedTimestamp"]:
        """Tries to parse 'other' into a SynchronizedTimestamp."""
        if other is None:
            return None

        if isinstance(other, ServerTimestamp):
            other = other.timestamp

        # This assertion is a useful precondition check.
        if not isinstance(other, Timestamp):
            raise TypeError(
                "Input must be a gRPC Timestamp or resolve to one."
            )

        try:
            dt_object = other.ToDatetime()
            return cls(dt_object)
        except ValueError as e:
            # Logging here is important for debugging potential data issues.
            logging.warning(
                "Failed to parse gRPC Timestamp to datetime: %s", e
            )
            return None

    def as_datetime(self) -> datetime.datetime:
        """Returns the underlying datetime.datetime object."""
        return self.timestamp

    def to_grpc_type(self) -> ServerTimestamp:
        """Converts this timestamp to a gRPC ServerTimestamp protobuf message."""
        timestamp_pb = Timestamp()
        timestamp_pb.FromDatetime(self.timestamp)
        return ServerTimestamp(timestamp=timestamp_pb)

    def __gt__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            msg = (
                "Compare error: SynchronizedTimestamp/datetime vs "
                f"{type(other_dt).__name__}"
            )
            raise TypeError(msg)
        return self.as_datetime() > other_dt

    def __ge__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            msg = (
                "Compare error: SynchronizedTimestamp/datetime vs "
                f"{type(other_dt).__name__}"
            )
            raise TypeError(msg)
        return self.as_datetime() >= other_dt

    def __lt__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            msg = (
                "Compare error: SynchronizedTimestamp/datetime vs "
                f"{type(other_dt).__name__}"
            )
            raise TypeError(msg)
        return self.as_datetime() < other_dt

    def __le__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            msg = (
                "Compare error: SynchronizedTimestamp/datetime vs "
                f"{type(other_dt).__name__}"
            )
            raise TypeError(msg)
        return self.as_datetime() <= other_dt

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        elif not isinstance(other, datetime.datetime):
            return False
        else:
            other_dt = other
        return self.as_datetime() == other_dt

    def __ne__(self, other: object) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        elif not isinstance(other, datetime.datetime):
            return True
        else:
            other_dt = other
        return self.as_datetime() != other_dt
