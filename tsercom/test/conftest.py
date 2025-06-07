import asyncio
from typing import Generator  # Added for type hint

import pytest
from pytest import FixtureRequest  # Added for type hint

from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    is_global_event_loop_set,
    set_tsercom_event_loop,
)


@pytest.fixture(autouse=True, scope="function")
def manage_tsercom_loop(
    request: FixtureRequest,
) -> Generator[None, None, None]:
    # Only apply this fixture if the test is marked with asyncio
    if not request.node.get_closest_marker("asyncio"):
        yield
        return

    # Get the event loop provided by pytest-asyncio for the current test
    # This assumes pytest-asyncio has already set up an event loop for the test.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running (e.g. test is not marked with @pytest.mark.asyncio properly,
        # or it's a synchronous test that shouldn't use this part of the fixture)
        # then this fixture should not attempt to set the tsercom global loop.
        yield
        return

    initial_loop_was_set = is_global_event_loop_set()
    if not initial_loop_was_set:
        set_tsercom_event_loop(loop)

    yield

    if not initial_loop_was_set:  # Only clear if this fixture set it
        clear_tsercom_event_loop()
