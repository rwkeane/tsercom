"""Tests for Stopable."""

import pytest

from tsercom.util.stopable import Stopable


class GoodStopable(Stopable):
    """A concrete implementation of Stopable that correctly implements stop."""

    async def stop(self) -> None:
        """Stops the Stopable."""
        pass


class BadStopable(Stopable):
    """A concrete implementation of Stopable that does not implement stop."""

    pass


def test_good_stopable_instantiation():
    """Tests that GoodStopable can be instantiated."""
    good_stopable = GoodStopable()
    assert isinstance(good_stopable, Stopable)


def test_bad_stopable_instantiation():
    """Tests that BadStopable cannot be instantiated due to missing stop method."""
    with pytest.raises(TypeError):
        BadStopable()
