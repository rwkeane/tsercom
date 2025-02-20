import caller_id.caller_id_pb2 as _caller_id_pb2
import timesync.common.time_pb2 as _time_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TestConnectionCall(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TestConnectionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("timestamp", "size", "array")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    timestamp: _time_pb2.ServerTimestamp
    size: _containers.RepeatedScalarFieldContainer[int]
    array: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, timestamp: _Optional[_Union[_time_pb2.ServerTimestamp, _Mapping]] = ..., size: _Optional[_Iterable[int]] = ..., array: _Optional[_Iterable[float]] = ...) -> None: ...
