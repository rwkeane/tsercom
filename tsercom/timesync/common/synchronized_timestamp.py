import datetime
from typing import TypeAlias, Union
from google.protobuf.timestamp_pb2 import Timestamp

from tsercom.timesync.common.proto import ServerTimestamp

TimestampType: TypeAlias = Union["SynchronizedTimestamp", datetime.datetime]
class SynchronizedTimestamp:

    def __init__(self, timestamp: datetime.datetime):
        assert timestamp is not None
        assert isinstance(timestamp, datetime.datetime)
        self.__timestamp = timestamp

    @classmethod
    def try_parse(
        self, other: Timestamp | ServerTimestamp
    ) -> "SynchronizedTimestamp":
        if isinstance(other, ServerTimestamp):
            other = other.timestamp

        assert isinstance(other, Timestamp)
        timestamp = other.ToDatetime()
        return SynchronizedTimestamp(timestamp)

    def as_datetime(self) -> datetime.datetime:
        return self.__timestamp

    def to_grpc_type(self) -> ServerTimestamp:
        timestamp_pb = Timestamp()
        timestamp_pb.FromDatetime(self.__timestamp)
        return ServerTimestamp(timestamp=timestamp_pb)

    def __gt__(self, other : TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other = other.as_datetime()

        assert isinstance(other, datetime.datetime)
        return self.as_datetime() > other

    def __ge__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other = other.as_datetime()

        assert isinstance(other, datetime.datetime)
        return self.as_datetime() >= other

    def __lt__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other = other.as_datetime()

        assert isinstance(other, datetime.datetime)
        return self.as_datetime() < other

    def __le__(self, other: TimestampType) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other = other.as_datetime()

        assert isinstance(other, datetime.datetime)
        return self.as_datetime() <= other

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other = other.as_datetime()

        elif not isinstance(other, datetime.datetime):
            return False
        return self.as_datetime() == other

    def __ne__(self, other: object) -> bool:
        if isinstance(other, SynchronizedTimestamp):
            other = other.as_datetime()

        elif not isinstance(other, datetime.datetime):
            return True
        return self.as_datetime() != other
