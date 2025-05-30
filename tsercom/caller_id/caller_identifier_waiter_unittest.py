import pytest
import asyncio

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.caller_id.caller_identifier_waiter import CallerIdentifierWaiter


@pytest.mark.asyncio
async def test_set_then_get_id():
    """
    Tests setting the CallerIdentifier first, then getting it.
    """
    waiter = CallerIdentifierWaiter()
    cid = CallerIdentifier.random()

    # Set the ID
    await waiter.set_caller_id(cid)

    # Get the ID
    retrieved_cid = await waiter.get_caller_id_async()

    assert (
        retrieved_cid is cid
    ), "Retrieved CID should be the same as the one set."


@pytest.mark.asyncio
async def test_get_then_set_id():
    """
    Tests getting the CallerIdentifier while another task sets it concurrently.
    """
    waiter = CallerIdentifierWaiter()
    cid_to_set = CallerIdentifier.random()

    async def setter_task():
        await asyncio.sleep(0.01)  # Ensure get_caller_id_async starts waiting
        await waiter.set_caller_id(cid_to_set)

    # Concurrently get the ID and run the task that sets it
    results = await asyncio.gather(waiter.get_caller_id_async(), setter_task())

    retrieved_cid = results[0]  # Result from get_caller_id_async()
    assert (
        retrieved_cid is cid_to_set
    ), "Retrieved CID should be the one set by the concurrent task."


@pytest.mark.asyncio
async def test_set_id_multiple_times_raises_assertion():
    """
    Tests that calling set_caller_id multiple times raises an AssertionError.
    """
    waiter = CallerIdentifierWaiter()
    cid1 = CallerIdentifier.random()
    cid2 = CallerIdentifier.random()

    # First call should succeed
    await waiter.set_caller_id(cid1)

    # Second call with a different ID should raise RuntimeError
    with pytest.raises(RuntimeError):
        await waiter.set_caller_id(cid2)

    # Also test that setting with the same ID raises RuntimeError
    with pytest.raises(RuntimeError):
        await waiter.set_caller_id(cid1)


@pytest.mark.asyncio
async def test_has_id():
    """
    Tests the has_id() method logic.
    The method `has_id()` in `CallerIdentifierWaiter` is actually `return self.__caller_id is None`.
    So, `has_id()` is True if `__caller_id` is None (i.e., ID has *not* been set yet).
    `has_id()` is False if `__caller_id` is not None (i.e., ID *has* been set).
    The test description seems to have the True/False conditions inverted relative to this typical interpretation.
    Let's assume the method name `has_id` means "does it currently possess/hold an ID?".
    No, the prompt says: `Assert await waiter.has_id() is True (since __caller_id is initially None)`.
    This means the method name `has_id` is interpreted as "is the ID slot available?" or "is it waiting for an ID?".
    If `__caller_id` is `None`, it means it doesn't *have* the ID yet, it's waiting.
    The prompt states: `has_id()` is `True` (since `__caller_id` is initially `None`). This means `has_id()` actually checks `self.__caller_id is None`.
    And then: `set_caller_id(cid)`. `Assert await waiter.has_id() is False (since __caller_id is now set)`.
    This confirms `has_id` should be `self.__caller_id is None`.
    """
    waiter = CallerIdentifierWaiter()

    # Initially, __caller_id is None, so has_id() (meaning "is ID present?") should be False.
    assert (
        waiter.has_id() is False
    ), "Initially, has_id should be False (ID not yet set)."

    cid = CallerIdentifier.random()
    await waiter.set_caller_id(cid)

    # After setting, __caller_id is not None, so has_id() should be True.
    assert (
        waiter.has_id() is True
    ), "After setting ID, has_id should be True (ID is now set)."
