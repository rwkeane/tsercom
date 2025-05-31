import datetime
from typing import TypeAlias, Union, Optional
import logging  # Added for logging
from google.protobuf.timestamp_pb2 import Timestamp

from tsercom.timesync.common.proto import ServerTimestamp

TimestampType: TypeAlias = Union["SynchronizedTimestamp", datetime.datetime]


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

    def __init__(self, timestamp: datetime.datetime) -> None:
        """
        Initializes a SynchronizedTimestamp.

        Args:
            timestamp: A naive `datetime.datetime` object. It's asserted that
                       this is not None and is an instance of `datetime.datetime`.
                       This timestamp is assumed to be in the synchronized time
                       domain.
        """
        assert timestamp is not None, "Timestamp cannot be None."
        assert isinstance(
            timestamp, datetime.datetime
        ), "Timestamp must be a datetime.datetime object."
        self.__timestamp = timestamp

    @classmethod
    def try_parse(
        cls, other: Timestamp | ServerTimestamp
    ) -> Optional["SynchronizedTimestamp"]:
        """
        Attempts to parse a gRPC Timestamp or a ServerTimestamp into a
        SynchronizedTimestamp.

        Args:
            other: The timestamp object to parse. This can be either a
                   `google.protobuf.timestamp_pb2.Timestamp` or a
                   `tsercom.timesync.common.proto.ServerTimestamp`.

        Returns:
            A new SynchronizedTimestamp instance.

        Raises:
            AssertionError: If `other` is not of the expected protobuf types
                            after potential unwrapping.
        """
        if other is None:
            return None

        if isinstance(other, ServerTimestamp):
            other = other.timestamp

        # At this point, 'other' should be a google.protobuf.timestamp_pb2.Timestamp.
        # This assertion helps catch incorrect input types.
        assert isinstance(
            other, Timestamp
        ), "Input must be a google.protobuf.timestamp_pb2.Timestamp or can be resolved to one."

        try:
            # The ToDatetime() method handles the conversion from seconds/nanos.
            dt_object = other.ToDatetime()
            return cls(dt_object)
        except ValueError as e:
            logging.warning(f"Failed to parse gRPC Timestamp to datetime: {e}")
            return None

    def as_datetime(self) -> datetime.datetime:
        """
        Returns the underlying naive datetime.datetime object.

        Returns:
            The naive `datetime.datetime` object that this SynchronizedTimestamp wraps.
        """
        return self.__timestamp

    def to_grpc_type(self) -> ServerTimestamp:
        """
        Converts this SynchronizedTimestamp to a `ServerTimestamp` protobuf message.

        The underlying `datetime.datetime` object is first converted to a
        `google.protobuf.timestamp_pb2.Timestamp`, which is then wrapped in
        our custom `ServerTimestamp` message.

        Returns:
            A `ServerTimestamp` protobuf message representing this timestamp.
        """
        timestamp_pb = Timestamp()
        timestamp_pb.FromDatetime(self.__timestamp)
        return ServerTimestamp(timestamp=timestamp_pb)

    def __gt__(self, other: TimestampType) -> bool:
        """
        Compares if this timestamp is greater than another.

        Args:
            other: Another `SynchronizedTimestamp` or a naive `datetime.datetime`.

        Returns:
            True if this timestamp is strictly greater than the other, False otherwise.
        """
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other

        assert isinstance(
            other_dt, datetime.datetime
        ), "Comparison is only supported with SynchronizedTimestamp or datetime.datetime."
        return self.as_datetime() > other_dt

    def __ge__(self, other: TimestampType) -> bool:
        """
        Compares if this timestamp is greater than or equal to another.

        Args:
            other: Another `SynchronizedTimestamp` or a naive `datetime.datetime`.

        Returns:
            True if this timestamp is greater than or equal to the other, False otherwise.
        """
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        assert isinstance(
            other_dt, datetime.datetime
        ), "Comparison is only supported with SynchronizedTimestamp or datetime.datetime."
        return self.as_datetime() >= other_dt

    def __lt__(self, other: TimestampType) -> bool:
        """
        Compares if this timestamp is less than another.

        Args:
            other: Another `SynchronizedTimestamp` or a naive `datetime.datetime`.

        Returns:
            True if this timestamp is strictly less than the other, False otherwise.
        """
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        assert isinstance(
            other_dt, datetime.datetime
        ), "Comparison is only supported with SynchronizedTimestamp or datetime.datetime."
        return self.as_datetime() < other_dt

    def __le__(self, other: TimestampType) -> bool:
        """
        Compares if this timestamp is less than or equal to another.

        Args:
            other: Another `SynchronizedTimestamp` or a naive `datetime.datetime`.

        Returns:
            True if this timestamp is less than or equal to the other, False otherwise.
        """
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        else:
            other_dt = other
        assert isinstance(
            other_dt, datetime.datetime
        ), "Comparison is only supported with SynchronizedTimestamp or datetime.datetime."
        return self.as_datetime() <= other_dt

    def __eq__(self, other: object) -> bool:
        """
        Compares if this timestamp is equal to another.

        Args:
            other: Another object to compare with. Can be `SynchronizedTimestamp`,
                   `datetime.datetime`, or any other type (which will result in False).

        Returns:
            True if this timestamp is equal to the other, False otherwise.
        """
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        elif not isinstance(other, datetime.datetime):
            return False  # Not comparable with other types
        else:
            other_dt = other
        return self.as_datetime() == other_dt

    def __ne__(self, other: object) -> bool:
        """
        Compares if this timestamp is not equal to another.

        Args:
            other: Another object to compare with. Can be `SynchronizedTimestamp`,
                   `datetime.datetime`, or any other type (which will result in True
                   if not a datetime or SynchronizedTimestamp).

        Returns:
            True if this timestamp is not equal to the other, False otherwise.
        """
        if isinstance(other, SynchronizedTimestamp):
            other_dt = other.as_datetime()
        elif not isinstance(other, datetime.datetime):
            return True  # Not comparable with other types, so considered not equal
        else:
            other_dt = other
        return self.as_datetime() != other_dt
