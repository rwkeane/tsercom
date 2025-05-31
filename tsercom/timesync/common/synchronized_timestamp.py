import dataclasses
import datetime
from typing import TypeAlias, Union, Optional
import logging
from google.protobuf.timestamp_pb2 import Timestamp

from tsercom.timesync.common.proto import ServerTimestamp

TimestampType: TypeAlias = Union["SynchronizedTimestamp", datetime.datetime]


@dataclasses.dataclass(
    eq=False, order=False, unsafe_hash=False
)  # Keep custom eq and order methods
class SynchronizedTimestamp:
    """
    A wrapper around a `datetime.datetime` object to represent a timestamp
    within a synchronized time context.

    This class ensures that timestamps are explicitly handled as being part of a
    synchronized system (e.g., aligned with a server's clock). It provides
    methods for conversion to and from other timestamp representations like
    gRPC `Timestamp` and `ServerTimestamp` protobuf messages.

    It also supports direct comparison with other `SynchronizedTimestamp` objects
    or naive `datetime.datetime` objects. When comparing, it unwraps the
    underlying `datetime.datetime` object for the actual comparison.
    """

    timestamp: datetime.datetime  # Field for the dataclass

    # __init__ will be auto-generated: def __init__(self, timestamp: datetime.datetime)
    # The original __init__ had assertions, which can be moved to __post_init__.

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        if (
            self.timestamp is None
        ):  # Changed from self.__timestamp to self.timestamp
            raise AssertionError("Timestamp cannot be None.")
        if not isinstance(self.timestamp, datetime.datetime):
            raise AssertionError(
                "Timestamp must be a datetime.datetime object."
            )

    @classmethod
    def try_parse(
        cls, other: Timestamp | ServerTimestamp
    ) -> Optional["SynchronizedTimestamp"]:
        if other is None:
            return None

        if isinstance(other, ServerTimestamp):
            other = other.timestamp

        assert isinstance(
            other, Timestamp
        ), "Input must be a google.protobuf.timestamp_pb2.Timestamp or can be resolved to one."

        try:
            dt_object = other.ToDatetime()
            return cls(dt_object)
        except ValueError as e:
            logging.warning(f"Failed to parse gRPC Timestamp to datetime: {e}")
            return None

    def as_datetime(self) -> datetime.datetime:
        return (
            self.timestamp
        )  # Changed from self.__timestamp to self.timestamp

    def to_grpc_type(self) -> ServerTimestamp:
        timestamp_pb = Timestamp()
        timestamp_pb.FromDatetime(
            self.timestamp
        )  # Changed from self.__timestamp to self.timestamp
        return ServerTimestamp(timestamp=timestamp_pb)

    # Keep all custom comparison methods as their logic is specific
    # (comparing unwrapped datetime and handling mixed types).

    def __gt__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other

        if not isinstance(other_dt, datetime.datetime):
            # To maintain original behavior, if it's not a datetime, it might raise error or return False
            # Depending on Python version, datetime comparison with non-datetime might raise TypeError.
            # The original code had an assert, which would raise an error. Let's keep that.
            raise TypeError(
                f"Comparison is only supported with SynchronizedTimestamp or datetime.datetime, got {type(other_dt)}"
            )
        return self.as_datetime() > other_dt

    def __ge__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            raise TypeError(
                f"Comparison is only supported with SynchronizedTimestamp or datetime.datetime, got {type(other_dt)}"
            )
        return self.as_datetime() >= other_dt

    def __lt__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            raise TypeError(
                f"Comparison is only supported with SynchronizedTimestamp or datetime.datetime, got {type(other_dt)}"
            )
        return self.as_datetime() < other_dt

    def __le__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        if not isinstance(other_dt, datetime.datetime):
            raise TypeError(
                f"Comparison is only supported with SynchronizedTimestamp or datetime.datetime, got {type(other_dt)}"
            )
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
