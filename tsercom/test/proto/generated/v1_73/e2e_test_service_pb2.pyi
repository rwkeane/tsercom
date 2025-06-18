from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EchoRequest(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class EchoResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class StreamDataRequest(_message.Message):
    __slots__ = ("data_id",)
    DATA_ID_FIELD_NUMBER: _ClassVar[int]
    data_id: int
    def __init__(self, data_id: _Optional[int] = ...) -> None: ...

class StreamDataResponse(_message.Message):
    __slots__ = ("data_chunk", "sequence_number")
    DATA_CHUNK_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    data_chunk: str
    sequence_number: int
    def __init__(self, data_chunk: _Optional[str] = ..., sequence_number: _Optional[int] = ...) -> None: ...
