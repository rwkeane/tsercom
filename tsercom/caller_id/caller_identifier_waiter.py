import asyncio

from tsercom.caller_id.caller_identifier import CallerIdentifier


class CallerIdentifierWaiter:
    def __init__(self):
        self.__caller_id = None
        
        self.__barrier = asyncio.Event()

    async def get_caller_id_async(self):
        await self.__barrier.wait()
        return self.__caller_id
    
    async def has_id(self):
        return self.__caller_id is None
    
    async def set_caller_id(self, caller_id : CallerIdentifier):
        assert self.__caller_id is None
        self.__caller_id = caller_id
        self.__barrier.set()
