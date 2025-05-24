from concurrent.futures import ThreadPoolExecutor
import datetime
import threading
from typing import Dict, Generic, List, Optional, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.data_timeout_tracker import DataTimeoutTracker
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_organizer import RemoteDataOrganizer
from tsercom.data.remote_data_reader import RemoteDataReader

TDataType = TypeVar("TDataType", bound=ExposedData)


class RemoteDataAggregatorImpl(
    Generic[TDataType],
    RemoteDataAggregator[TDataType],
    RemoteDataOrganizer.Client,
    RemoteDataReader[TDataType],
):
    """
    Main implementation of the RemoteDataAggregator. This instance is separate
    from the interface to limit what is shown to a user of the class.
    """

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
    ): ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        tracker: DataTimeoutTracker,
    ): ...

    @overload
    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        timeout: int,
    ): ...

    def __init__(
        self,
        thread_pool: ThreadPoolExecutor,
        client: Optional[RemoteDataAggregator.Client] = None,
        *,
        tracker: Optional[DataTimeoutTracker] = None,
        timeout: Optional[int] = None,
    ):
        assert not timeout or not tracker
        if tracker is None and timeout is not None:
            tracker = DataTimeoutTracker(timeout)
            tracker.start()

        self.__thread_pool = thread_pool
        self.__client = client
        self.__tracker = tracker

        # TODO: Can this be a CallerIdMap?
        self.__organizers: Dict[
            CallerIdentifier, RemoteDataOrganizer[TDataType]
        ] = {}
        self.__lock = threading.Lock()

    # Added add_data method as requested by the prompt
    def add_data(self, instance: TDataType):
        # Attempt to get a meaningful value for logging
        data_value = "Unknown"
        if hasattr(instance, 'data'): # Assuming instance might be an AnnotatedInstance
            if hasattr(instance.data, 'value'): # Assuming instance.data might be FakeData
                data_value = instance.data.value
            else:
                data_value = str(instance.data)
        else: # Fallback if instance is not an AnnotatedInstance or similar structure
            if hasattr(instance, 'value'): # instance might be FakeData itself
                data_value = instance.value
            else:
                data_value = str(instance)
        print(f"DEBUG: [RemoteDataAggregatorImpl.add_data] Instance: {data_value}")
        self._on_data_ready(instance)


    def stop(self, id: Optional[CallerIdentifier] = None) -> None:
        with self.__lock:
            if id is not None:
                assert id in self.__organizers
                self.__organizers[id].stop()
                return

            for key, val in self.__organizers.items():
                val.stop()

    def has_new_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, bool] | bool:
        with self.__lock:
            if id is not None:
                # If the id is not known, it means no data has been received for it.
                # So, it cannot have "new" data.
                if id not in self.__organizers:
                    print(f"DEBUG: [RemoteDataAggregatorImpl.has_new_data] ID {id} not in organizers. Returning False.")
                    return False
                return self.__organizers[id].has_new_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.has_new_data()
            # print(f"DEBUG: [RemoteDataAggregatorImpl.has_new_data] No ID provided. Returning dict: {results}")
            return results

    def get_new_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, List[TDataType]] | List[TDataType]:
        with self.__lock:
            if id is not None:
                # If the id is not known, return an empty list as no data can be fetched.
                if id not in self.__organizers:
                    print(f"DEBUG: [RemoteDataAggregatorImpl.get_new_data] ID {id} not in organizers. Returning empty list.")
                    return []
                return self.__organizers[id].get_new_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_new_data()
            # print(f"DEBUG: [RemoteDataAggregatorImpl.get_new_data] No ID provided. Returning dict of lists.")
            return results

    def get_most_recent_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        with self.__lock:
            if id is not None:
                if id not in self.__organizers:
                    return None # Or raise an error, depending on desired behavior for unknown ID
                return self.__organizers[id].get_most_recent_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_most_recent_data()
            return results

    def get_data_for_timestamp(  # type: ignore
        self,
        id: CallerIdentifier, # This method implies id must be specific
        timestamp: datetime.datetime,
    ) -> TDataType | None: # Simpler return type if id is mandatory
        with self.__lock:
            if id not in self.__organizers:
                return None # Or raise an error
            return self.__organizers[id].get_data_for_timestamp(timestamp)

    def _on_data_available(  # type: ignore
        self, data_organizer: "RemoteDataOrganizer[TDataType]"
    ) -> None:
        if self.__client is not None:
            # Log when data becomes available from an organizer
            print(f"DEBUG: [RemoteDataAggregatorImpl._on_data_available] Data available from organizer: {data_organizer.caller_id}")
            self.__client._on_data_available(self, data_organizer.caller_id)

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

        print(f"DEBUG: [RemoteDataAggregatorImpl._on_data_ready] new_data: {data_value} from caller_id: {caller_id_value}")
        assert issubclass(type(new_data), ExposedData), type(new_data)

        # Find or create the RemoteDataOrganizer.
        found = False
        data_organizer: RemoteDataOrganizer[TDataType] # Type hint for clarity
        with self.__lock:
            if new_data.caller_id not in self.__organizers: # Corrected check
                print(f"DEBUG: [RemoteDataAggregatorImpl._on_data_ready] Creating new RemoteDataOrganizer for caller_id: {new_data.caller_id}")
                data_organizer = RemoteDataOrganizer(
                    self.__thread_pool, new_data.caller_id, self
                )
                if self.__tracker is not None:
                    self.__tracker.register(data_organizer)
                data_organizer.start()
                self.__organizers[new_data.caller_id] = data_organizer
                found = False # Explicitly false as it was just created
            else:
                data_organizer = self.__organizers[new_data.caller_id]
                found = True # Explicitly true as it was found
                print(f"DEBUG: [RemoteDataAggregatorImpl._on_data_ready] Using existing RemoteDataOrganizer for caller_id: {new_data.caller_id}")


        # Call into it.
        assert data_organizer is not None # Should always be true due to logic above
        print(f"DEBUG: [RemoteDataAggregatorImpl._on_data_ready] Calling data_organizer._on_data_ready for caller_id: {new_data.caller_id}")
        data_organizer._on_data_ready(new_data) # This is RemoteDataOrganizer._on_data_ready

        # Inform the client if needed
        if not found and self.__client is not None: # Only if it's a newly created organizer
            print(f"DEBUG: [RemoteDataAggregatorImpl._on_data_ready] Informing client about new endpoint: {new_data.caller_id}")
            self.__client._on_new_endpoint_began_transmitting(
                self, data_organizer.caller_id # Pass self as RemoteDataAggregator, and organizer's caller_id
            )
