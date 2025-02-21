from typing import Optional
import torch

from tsercom.rpc.proto import Tensor as GrpcTensor
from tsercom.timesync.common.synchronized_timestamp import SynchronizedTimestamp


class SerializableTensor:
    def __init__(
            self, tensor : torch.Tensor, timestamp : SynchronizedTimestamp):
        self.__tensor = tensor
        self.__timestamp = timestamp

    @property
    def tensor(self):
        return self.__tensor
    
    @property
    def timestamp(self):
        return self.__timestamp

    def to_grpc_type(self) -> GrpcTensor:
        size = list(self.__tensor.size())
        entries = self.__tensor.reshape(-1).tolist()
        return GrpcTensor(timestamp = self.__timestamp.to_grpc_type(),
                          size = size,
                          array = entries)
    
    @classmethod
    def try_parse(cls,
                  grpc_type : GrpcTensor) -> Optional['SerializableTensor']:
        timestamp = SynchronizedTimestamp.try_parse(grpc_type.timestamp)
        if timestamp is None:
            return None
        
        try:
            tensor = torch.Tensor(grpc_type.array)
            size = list(grpc_type.size)
            tensor.reshape(size)

            return SerializableTensor(tensor, timestamp)
        except Exception as e:
            print("Error deserializing Tensor:", e)
            return None