from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData


TDataType = TypeVar("TDataType", bound = ExposedData)
class RemoteDataReader(ABC, Generic[TDataType]):
    """
    This interface is to be implemented by classes that process remote data. The
    MAIN use-case of this class is to have a thin, light-weight stand in for a 
    RemoteDataOrganizer that can be exposed to lower layers.
    """
    @abstractmethod
    def _on_data_ready(self, new_data : TDataType):
        pass