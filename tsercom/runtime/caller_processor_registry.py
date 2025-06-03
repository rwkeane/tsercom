import threading
from typing import Dict, Callable, Any, Optional

from tsercom.caller_id.caller_identifier import CallerIdentifier


class CallerProcessorRegistry:
    """
    Manages the creation and retrieval of processor instances for CallerIdentifiers.

    A factory provided at initialization is used to create new processor
    instances on demand for each unique CallerIdentifier.
    """

    def __init__(self, processor_factory: Callable[[CallerIdentifier], Any]):
        """
        Initializes the CallerProcessorRegistry.

        Args:
            processor_factory: A callable that takes a CallerIdentifier
                               and returns a new processor instance.
        """
        if processor_factory is None:
            raise ValueError("processor_factory cannot be None")
        self.__processor_factory = processor_factory
        self.__processors: Dict[CallerIdentifier, Any] = {}
        self.__lock = threading.Lock()

    def get_or_create_processor(self, caller_id: CallerIdentifier) -> Any:
        """
        Retrieves an existing processor for the given CallerIdentifier,
        or creates a new one using the factory if it doesn't exist.

        Args:
            caller_id: The CallerIdentifier for which to get/create the processor.

        Returns:
            The processor instance associated with the CallerIdentifier.
        """
        with self.__lock:
            if caller_id not in self.__processors:
                # Ensure the factory is only called while holding the lock
                # if the processor truly needs to be created.
                processor = self.__processor_factory(caller_id)
                self.__processors[caller_id] = processor
            return self.__processors[caller_id]

    def get_processor(self, caller_id: CallerIdentifier) -> Optional[Any]:
        """
        Retrieves an existing processor for the given CallerIdentifier, if it exists.

        Args:
            caller_id: The CallerIdentifier for which to get the processor.

        Returns:
            The processor instance or None if not found.
        """
        with self.__lock:
            return self.__processors.get(caller_id)

    def remove_processor(self, caller_id: CallerIdentifier) -> bool:
        """
        Removes the processor associated with the given CallerIdentifier.

        Args:
            caller_id: The CallerIdentifier whose processor is to be removed.

        Returns:
            True if a processor was found and removed, False otherwise.
        """
        with self.__lock:
            if caller_id in self.__processors:
                del self.__processors[caller_id]
                return True
            return False

    def get_all_processors(self) -> Dict[CallerIdentifier, Any]:
        """Returns a shallow copy of the current processors dictionary."""
        with self.__lock:
            return dict(self.__processors)
