"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tsercom.timesync.common.proto as time_pb2
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class TestConnectionCall(google.protobuf.message.Message):
    """
    python3 -m grpc_tools.protoc \\
    --proto_path=util/caller_id/proto \\
    --proto_path=util/rpc/proto \\
    --proto_path=timesync/common/proto \\
    --python_out=util/rpc \\
    --pyi_out=util/rpc \\
    util/rpc/proto/common.proto
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___TestConnectionCall = TestConnectionCall

@typing.final
class TestConnectionResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___TestConnectionResponse = TestConnectionResponse

@typing.final
class Tensor(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TIMESTAMP_FIELD_NUMBER: builtins.int
    SIZE_FIELD_NUMBER: builtins.int
    ARRAY_FIELD_NUMBER: builtins.int
    @property
    def timestamp(self) -> time_pb2.ServerTimestamp: ...
    @property
    def size(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def array(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]: ...
    def __init__(
        self,
        *,
        timestamp: time_pb2.ServerTimestamp | None = ...,
        size: collections.abc.Iterable[builtins.int] | None = ...,
        array: collections.abc.Iterable[builtins.float] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["timestamp", b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["array", b"array", "size", b"size", "timestamp", b"timestamp"]) -> None: ...

global___Tensor = Tensor
