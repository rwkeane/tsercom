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
        print(f"DEBUG: [DataReaderSink.__init__] Initialized with queue: {queue}")

    def _on_data_ready(self, new_data: TDataType) -> None:
        # Attempt to get a meaningful value for logging
        data_value = "Unknown"
        caller_id_value = "UnknownCallerId"
        if hasattr(new_data, 'data') and hasattr(new_data, 'caller_id'): # AnnotatedInstance structure
            caller_id_value = str(new_data.caller_id)
            if hasattr(new_data.data, 'value'): # FakeData structure
                data_value = new_data.data.value
            else:
                data_value = str(new_data.data)
        elif hasattr(new_data, 'caller_id'): # If new_data itself has caller_id (e.g. ExposedData)
             caller_id_value = str(new_data.caller_id)
             if hasattr(new_data, 'value'):
                 data_value = new_data.value
             else:
                 data_value = str(new_data)
        else: # Fallback
            data_value = str(new_data)

        print(f"DEBUG: [DataReaderSink._on_data_ready] Received new data: {data_value} for caller_id: {caller_id_value}")
        
        success = self.__queue.put_nowait(new_data)
        print(f"DEBUG: [DataReaderSink._on_data_ready] Data put to queue success: {success} for caller_id: {caller_id_value}")
        assert success or self.__is_lossy, "Queue is full"
