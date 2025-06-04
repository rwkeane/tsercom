"""Defines DataReaderSink for sending data to a multiprocess queue."""

from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)

DataTypeT = TypeVar("DataTypeT", bound=ExposedData)


# pylint: disable=R0903 # Abstract interface/protocol class
class DataReaderSink(Generic[DataTypeT], RemoteDataReader[DataTypeT]):
    """Implements RemoteDataReader to send data to a MultiprocessQueueSink.

    This class acts as a sink in a data pipeline, taking data items via
    the `_on_data_ready` method and putting them onto a multiprocess queue.
    It can be configured to be lossy or non-lossy.
    """

    def __init__(
        self, queue: MultiprocessQueueSink[DataTypeT], is_lossy: bool = True
    ) -> None:
        """Initializes the DataReaderSink.

        Args:
            queue: The multiprocess queue sink to which data will be sent.
            is_lossy: If True (default), data is dropped if the queue is full.
                      If False, a RuntimeError is raised if the queue is full,
                      to prevent data loss.
        """
        self.__queue = queue
        self.__is_lossy = is_lossy

    def _on_data_ready(self, new_data: DataTypeT) -> None:
        """Handles new data by putting it onto the multiprocess queue.

        This method is called when new data is available to be processed.
        It attempts to put the data onto the configured queue. Behavior
        when the queue is full depends on the `is_lossy` setting.

        Args:
            new_data: The data item to be sent to the queue.

        Raises:
            RuntimeError: If `is_lossy` is False and the queue is full.
        """
        success = self.__queue.put_nowait(new_data)
        # If putting to queue failed (e.g., full) and this sink is not lossy, raise an error.
        if not success and not self.__is_lossy:
            raise RuntimeError(
                "Queue full; data would be lost on non-lossy sink."
            )
