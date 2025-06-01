import dataclasses
import datetime
from typing import TypeAlias, Union, Optional
import logging
from google.protobuf.timestamp_pb2 import Timestamp

from tsercom.timesync.common.proto import ServerTimestamp

TimestampType: TypeAlias = Union["SynchronizedTimestamp", datetime.datetime]


@dataclasses.dataclass(eq=False, order=False, unsafe_hash=False)
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

    timestamp: datetime.datetime

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        if self.timestamp is None:
            raise TypeError("Timestamp cannot be None.")
        if not isinstance(self.timestamp, datetime.datetime):
            raise TypeError("Timestamp must be a datetime.datetime object.")

    @classmethod
    def try_parse(
        cls, other: Timestamp | ServerTimestamp
    ) -> Optional["SynchronizedTimestamp"]:
        """
        Attempts to parse a protobuf Timestamp or ServerTimestamp into a SynchronizedTimestamp.

        Args:
            other: The protobuf Timestamp or ServerTimestamp object to parse.
                   Can also be None.

        Returns:
            A SynchronizedTimestamp instance if parsing is successful,
            or None if the input is None or if parsing fails (e.g., due to
            invalid timestamp data which causes ToDatetime() to raise ValueError).

        Raises:
            TypeError: If the input `other` is not None and not one of the expected
                       protobuf timestamp types.
        """
        if other is None:
            return None

        if isinstance(other, ServerTimestamp):
            other = other.timestamp

        if not isinstance(other, Timestamp):
            raise TypeError(
                "Input must be a google.protobuf.timestamp_pb2.Timestamp or can be resolved to one."
            )

        try:
            dt_object = other.ToDatetime()
            return cls(dt_object)
        except ValueError as e:
            logging.warning(f"Failed to parse gRPC Timestamp to datetime: {e}")
            return None

    def as_datetime(self) -> datetime.datetime:
        """
        Returns the underlying naive datetime.datetime object.

        Returns:
            datetime.datetime: The naive datetime object.
        """
        return self.timestamp

    def to_grpc_type(self) -> ServerTimestamp:
        """
        Converts this SynchronizedTimestamp to a `ServerTimestamp` protobuf message.

        Returns:
            ServerTimestamp: The protobuf message representation.
        """
        timestamp_pb = Timestamp()
        timestamp_pb.FromDatetime(self.timestamp)
        return ServerTimestamp(timestamp=timestamp_pb)

    def _extract_comparable_datetime(self, other: object) -> datetime.datetime:
        """
        Helper to extract a datetime object from 'other' for comparison.

        Args:
            other: The object to extract datetime from. Can be SynchronizedTimestamp
                   or datetime.datetime.

        Returns:
            datetime.datetime: The extracted datetime object.

        Raises:
            TypeError: If 'other' is not a SynchronizedTimestamp or datetime.datetime.
        """
        if isinstance(other, SynchronizedTimestamp):
            return other.timestamp
        if isinstance(other, datetime.datetime):
            return other
        # pylint: disable=line-too-long
        raise TypeError(
            f"Comparison is only supported with SynchronizedTimestamp or datetime.datetime, got {type(other)}"
        )
        # pylint: enable=line-too-long

    def __gt__(self, other: TimestampType) -> bool:
        """Checks if this timestamp is greater than the other."""
        other_dt = self._extract_comparable_datetime(other)
        return self.timestamp > other_dt

    def __ge__(self, other: TimestampType) -> bool:
        """Checks if this timestamp is greater than or equal to the other."""
        other_dt = self._extract_comparable_datetime(other)
        return self.timestamp >= other_dt

    def __lt__(self, other: TimestampType) -> bool:
        """Checks if this timestamp is less than the other."""
        other_dt = self._extract_comparable_datetime(other)
        return self.timestamp < other_dt

    def __le__(self, other: TimestampType) -> bool:
        """Checks if this timestamp is less than or equal to the other."""
        other_dt = self._extract_comparable_datetime(other)
        return self.timestamp <= other_dt

    def __eq__(self, other: object) -> bool:
        """Checks if this timestamp is equal to the other."""
        if isinstance(other, SynchronizedTimestamp):
            return self.timestamp == other.timestamp
        if isinstance(other, datetime.datetime):
            return self.timestamp == other
        return False

    def __ne__(self, other: object) -> bool:
        """Checks if this timestamp is not equal to the other."""
        if isinstance(other, SynchronizedTimestamp):
            return self.timestamp != other.timestamp
        if isinstance(other, datetime.datetime):
            return self.timestamp != other
        return True
