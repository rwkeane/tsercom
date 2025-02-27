from concurrent.futures import ThreadPoolExecutor
import datetime
import threading
from typing import Dict, Generic, List, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.thread_watcher import ThreadWatcher


TDataType = TypeVar("TDataType", bound = ExposedData)
class RemoteDataAggregatorImpl(Generic[TDataType],
                               RemoteDataAggregator[TDataType],
                               RemoteDataOrganizer.Client,
                               RemoteDataReader[TDataType]):
    """
    Main implementation of the RemoteDataAggregator. This instance is separate
    from the interface to limit what is shown to a user of the class.
    """
    def __init__(self,
                 thread_pool : ThreadPoolExecutor,
                 client : Optional[RemoteDataAggregator.Client] = None,
                 timeout_seconds : int = 60):
        self.__thread_pool = thread_pool
        self.__client = client
        self.__timeout_seconds = timeout_seconds

        self.__organizers : Dict[CallerIdentifier, RemoteDataOrganizer] = {}
        self.__lock = threading.Lock()

    def stop(self, id : Optional[CallerIdentifier] = None):
        with self.__lock:
            if not id is None:
                assert id in self.__organizers
                return
            
            for key, val in self.__organizers.items():
                val.stop()

    def has_new_data(self, id : Optional[CallerIdentifier] = None):
        with self.__lock:
            if not id is None:
                assert id in self.__organizers
                return self.__organizers[id].has_new_data()
            
            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.has_new_data()
            return results

    def get_new_data(self, id : Optional[CallerIdentifier] = None):
        with self.__lock:
            if not id is None:
                assert id in self.__organizers
                return self.__organizers[id].get_new_data()
            
            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_new_data()
            return results
        
    def get_most_recent_data(self, id : Optional[CallerIdentifier] = None):
        with self.__lock:
            if not id is None:
                assert id in self.__organizers
                return self.__organizers[id].get_most_recent_data()
            
            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_most_recent_data()
            return results
    
    def get_data_for_timestamp(
            self,
            id : CallerIdentifier, 
            timestamp : Optional[datetime.datetime] = None):
        with self.__lock:
            if not id is None:
                assert id in self.__organizers
                return self.__organizers[id].get_data_for_timestamp(id,
                                                                    timestamp)
            
            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_data_for_timestamp(id, timestamp)
            return results
    
    def _on_data_available(self, data_organizer : 'RemoteDataOrganizer'):
        if not self.__client is None:
            self.__client._on_data_available(self, data_organizer.caller_id)

    def _on_data_ready(self, new_data : TDataType):
        assert issubclass(type(new_data), ExposedData), type(new_data)
        
        # Find or create the RemoteDataOrganizer.
        found = False
        data_organizer : RemoteDataOrganizer = None
        with self.__lock:
            found = new_data.caller_id in self.__organizers
            if not found:
                data_organizer = RemoteDataOrganizer(
                        self.__thread_pool,
                        new_data.caller_id,
                        self,
                        self.__timeout_seconds)
                data_organizer.start()
                self.__organizers[new_data.caller_id] = data_organizer
            else:
                data_organizer = self.__organizers[new_data.caller_id]

        # Call into it.
        assert not data_organizer is None
        data_organizer._on_data_ready(new_data)

        # Inform the client if needed
        if not found and not self.__client is None:
            self.__client._on_new_endpoint_began_transmitting(
                    self, data_organizer.caller_id)