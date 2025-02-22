from abc import ABC, abstractmethod
import datetime
from typing import Dict, Generic, List, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData


TDataType = TypeVar("TDataType", bound = ExposedData)
class RemoteDataAggregator(ABC, Generic[TDataType]):
    """
    This class is responsible for providing an aggregate view of all remote data
    access and reading. Additionally, this class obscures the internals of the
    service, so that only a CallerID and TDataType are exposed to the user.
    """
    class Client(ABC):
        """
        Optional callback API for receiving callback events based on changes in
        state of the remote end of the connection.
        """
        @abstractmethod
        def _on_data_available(self,
                               aggregator : 'RemoteDataAggregator[TDataType]',
                               caller_id : CallerIdentifier):
            """
            Called when new data is available for |caller_id|.
            """
            pass

        @abstractmethod
        def _on_new_endpoint_began_transmitting(
                self,
                aggregator : 'RemoteDataAggregator[TDataType]',
                caller_id : CallerIdentifier):
            """
            Called when a new endpoint associated with |caller_id| is found.
            """
            pass
        
    @overload
    def stop(self):
        """
        Stops all RemoteDataOrganizers currently tracked by this instance.
        """
        ...

    @overload
    def stop(self, id : CallerIdentifier):
        """
        Stops the RemoteDataOrganizer assocaited with |id|.
        """
        ...

    @abstractmethod
    def stop(self, id : Optional[CallerIdentifier] = None):
        pass

    @overload
    def has_new_data(self) -> Dict[CallerIdentifier, bool]:
        """
        Returns a dictionary specifying whether new data is available for each
        returned CallerId.
        """
        ...

    @overload
    def has_new_data(self, id : CallerIdentifier) -> bool:
        """
        Returns whether data is available for the caller |id|.
        """
        ...

    @abstractmethod
    def has_new_data(self, id : Optional[CallerIdentifier] = None):
        pass
        
    @overload
    def get_new_data(self) -> Dict[CallerIdentifier, List[TDataType]]:
        """
        Retrieves all new data for all callers.
        """
        ...
        
    @overload
    def get_new_data(self, id : CallerIdentifier) -> List[TDataType]:
        """
        Retrieves new data for caller |id|.
        """
        ...

    @abstractmethod
    def get_new_data(self, id : Optional[CallerIdentifier] = None):
        pass
        
    @overload
    def get_most_recent_data(self) -> Dict[CallerIdentifier, TDataType | None]:
        """
        Returns the most recently received data for all callers, if such data
        has not yet timed out.
        """
        ...
        
    @overload
    def get_most_recent_data(self, id : CallerIdentifier) -> TDataType | None:
        """
        Returns the most recently received data for caller |id|, if such data
        has not yet timed out.
        """
        ...

    @abstractmethod
    def get_most_recent_data(self, id : Optional[CallerIdentifier] = None):
        pass
        
    @overload
    def get_data_for_timestamp(self, timestamp : datetime.datetime) \
            -> Dict[CallerIdentifier, TDataType | None]:
        """
        Returns the most recent data received before |timestamp| for all
        callers, if such data has not yet timed out.
        """
        ...
        
    @overload
    def get_data_for_timestamp(
            self,
            id : CallerIdentifier, 
            timestamp : datetime.datetime) -> TDataType | None:
        """
        Returns the most recent data received before |timestamp| for caller
        |id|, if such data has not yet timed out.
        """
        ...

    @abstractmethod
    def get_data_for_timestamp(
            self,
            id : CallerIdentifier, 
            timestamp : Optional[datetime.datetime] = None):
        pass