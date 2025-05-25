"""Provides a thread-safe map for associating CallerIdentifiers with generic objects."""

import threading
from typing import Callable, Dict, Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier

TType = TypeVar("TType")


class CallerIdMap(Generic[TType]):
    """A thread-safe map that associates CallerIdentifier instances with generic objects.

    This class provides a dictionary-like structure where keys are CallerIdentifier
    objects and values are instances of a generic type TType. It ensures that
    access to the map is synchronized using a threading.Lock.
    """

    def __init__(self) -> None:
        """Initializes a new CallerIdMap instance."""
        self.__lock = threading.Lock()
        self.__map: Dict[CallerIdentifier, TType] = {}

    def find_instance(
        self, caller_id: CallerIdentifier, factory: Callable[[], TType]
    ) -> TType:
        """Finds an instance associated with a caller_id, creating it if necessary.

        If an instance for the given `caller_id` already exists in the map, it is
        returned. Otherwise, the provided `factory` function is called to create a
        new instance, which is then stored in the map and returned. This operation
        is thread-safe.
        """
        with self.__lock:
            if caller_id not in self.__map:
                self.__map[caller_id] = factory()
            return self.__map[caller_id]

    def for_all_items(self, function: Callable[[TType], None]) -> None:
        """Executes a function for all items currently in the map.

        This method iterates over a snapshot of the map's values, so modifications
        to the map during iteration will not affect the items processed. This
        operation is thread-safe.
        """
        # Create a snapshot of values to iterate over to avoid issues with map modification during iteration.
        items = []
        with self.__lock:
            items = list(self.__map.values())

        for val in items:
            function(val)

    def count(self) -> int:
        """Returns the number of items in the map.

        This operation is thread-safe.
        """
        with self.__lock:
            return len(self.__map)

    def __len__(self) -> int:
        """Returns the number of items in the map, making `len()` usable on instances.

        This operation is thread-safe.
        """
        return self.count()
