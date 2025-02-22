from typing import Generic, Optional, TypeVar

from tsercom.data.data_host import DataHost
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.task_runner import TaskRunner


TDataType = TypeVar("TDataType", bound = ExposedData)
class DataHostBase(Generic[TDataType], DataHost[TDataType], RemoteDataReader):
    """
    This class defines the implementation of DataHost to be used for the impl
    versions of all ServerHost and ClientHost classes.
    """
    def __init__(self,
                 task_runner : TaskRunner,
                 aggregation_client : \
                        Optional[RemoteDataAggregator[TDataType].Client] = None,
                 timeout_seconds : int = 60,
                 *args,
                 **kwargs):
        self.__aggregator = RemoteDataAggregatorImpl(
                task_runner, aggregation_client, timeout_seconds)
    
        super().__init__(*args, **kwargs)

    def _on_data_ready(self, new_data : TDataType):
        self.__aggregator._on_data_ready(new_data)
    
    def _remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        return self.__aggregator