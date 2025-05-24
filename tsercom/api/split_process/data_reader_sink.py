from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)

TDataType = TypeVar("TDataType", bound=ExposedData)


class DataReaderSink(Generic[TDataType], RemoteDataReader[TDataType]):
    def __init__(
        self, queue: MultiprocessQueueSink[TDataType], is_lossy: bool = True
    ):
        self.__queue = queue
        self.__is_lossy = is_lossy

    def _on_data_ready(self, new_data: TDataType) -> None:
        success = self.__queue.put_nowait(new_data)
        if not success and not self.__is_lossy:
            raise RuntimeError("Queue is full and data would be lost on a non-lossy sink.")
