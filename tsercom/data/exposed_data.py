"""Defines the ExposedData abstract base class.

This module contains the `ExposedData` class, which serves as a fundamental
interface for all data objects that are managed and transmitted within the
Tsercom system. It mandates that all data carry a `caller_id` and a `timestamp`.
"""

import datetime
from abc import ABC, abstractmethod

from tsercom.caller_id.caller_identifier import CallerIdentifier


class ExposedData(ABC):
    """Abstract base class for data exposed by a tsercom runtime.

    Subclasses must implement the `caller_id` and `timestamp` properties.
    These properties are essential for tracking the origin and timing of data
    flowing through the Tsercom system.
    """

    @property
    @abstractmethod
    def caller_id(self) -> CallerIdentifier | None:
        """Return the identifier of the instance that generated this data.

        Returns:
            Optional CallerIdentifier. Can be None if data is not specific
            to a single caller (e.g., a broadcast event).

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def timestamp(self) -> datetime.datetime:
        """Return the timestamp when this data was generated or recorded.

        Returns:
            A datetime object representing the data's timestamp.
            It is recommended this be a UTC timestamp.

        """
        raise NotImplementedError
