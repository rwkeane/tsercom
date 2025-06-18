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
    # Get the event loop. If a test is marked with @pytest.mark.asyncio,
    # pytest-asyncio provides one. Otherwise, get_event_loop() might create one.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop is currently running. Create one for this function's scope.
        # This ensures that if sync tests call async code that needs a loop, one is available.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop_created_by_fixture = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_created_by_fixture = True

    initial_tsercom_loop_was_set = is_global_event_loop_set()
    if not initial_tsercom_loop_was_set:
        set_tsercom_event_loop(loop)

    yield

    if not initial_tsercom_loop_was_set:
        clear_tsercom_event_loop()

    if loop_created_by_fixture:
        loop.close()
