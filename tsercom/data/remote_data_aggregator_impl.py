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

        self.__organizers: Dict[
            CallerIdentifier, RemoteDataOrganizer[TDataType]
        ] = {}
        self.__lock = threading.Lock()

    # The add_data method was added for debugging and is not part of the original class structure.
    # It is being removed as per the cleanup task.
    # def add_data(self, instance: TDataType):
    #     data_value = "Unknown"
    #     if hasattr(instance, 'data'): 
    #         if hasattr(instance.data, 'value'): 
    #             data_value = instance.data.value
    #         else:
    #             data_value = str(instance.data)
    #     else: 
    #         if hasattr(instance, 'value'): 
    #             data_value = instance.value
    #         else:
    #             data_value = str(instance)
    #     # print(f"DEBUG: [RemoteDataAggregatorImpl.add_data] Instance: {data_value}")
    #     self._on_data_ready(instance)


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
                if id not in self.__organizers:
                    return False
                return self.__organizers[id].has_new_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.has_new_data()
            return results

    def get_new_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, List[TDataType]] | List[TDataType]:
        with self.__lock:
            if id is not None:
                if id not in self.__organizers:
                    return []
                return self.__organizers[id].get_new_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_new_data()
            return results

    def get_most_recent_data(  # type: ignore
        self, id: Optional[CallerIdentifier] = None
    ) -> Dict[CallerIdentifier, TDataType | None] | TDataType | None:
        with self.__lock:
            if id is not None:
                if id not in self.__organizers:
                    return None 
                return self.__organizers[id].get_most_recent_data()

            results = {}
            for key, val in self.__organizers.items():
                results[key] = val.get_most_recent_data()
            return results

    def get_data_for_timestamp(  # type: ignore
        self,
        id: CallerIdentifier, 
        timestamp: datetime.datetime,
    ) -> TDataType | None: 
        with self.__lock:
            if id not in self.__organizers:
                return None 
            return self.__organizers[id].get_data_for_timestamp(timestamp)

    def _on_data_available(  # type: ignore
        self, data_organizer: "RemoteDataOrganizer[TDataType]"
    ) -> None:
        if self.__client is not None:
            self.__client._on_data_available(self, data_organizer.caller_id)

    def _on_data_ready(self, new_data: TDataType) -> None:
        assert issubclass(type(new_data), ExposedData), type(new_data)

        data_organizer: RemoteDataOrganizer[TDataType] 
        with self.__lock:
            if new_data.caller_id not in self.__organizers: 
                data_organizer = RemoteDataOrganizer(
                    self.__thread_pool, new_data.caller_id, self
                )
                if self.__tracker is not None:
                    self.__tracker.register(data_organizer)
                data_organizer.start()
                self.__organizers[new_data.caller_id] = data_organizer
                # Inform client for new endpoint after lock is released if necessary,
                # or ensure client call is lock-safe if called here.
                # For now, keeping informational client call outside critical lock section if possible
                # by using a flag and calling after. If _on_new_endpoint_began_transmitting needs
                # some state protected by this lock, it must be called here or self.__lock re-entered.
                # Current structure has it after this block but it's not conditional on 'found' anymore.
                # Let's assume it's okay to call it after the lock or it handles its own locking.
                # Re-evaluating the 'found' flag logic:
                is_new_organizer = True
            else:
                data_organizer = self.__organizers[new_data.caller_id]
                is_new_organizer = False
        
        data_organizer._on_data_ready(new_data) 

        if is_new_organizer and self.__client is not None: 
            self.__client._on_new_endpoint_began_transmitting(
                self, data_organizer.caller_id 
            )
