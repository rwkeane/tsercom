from typing import Optional
import uuid

from tsercom.caller_id.proto import CallerId


class CallerIdentifier:
    """
    This class acts as a simple wrapper around a GUID, allowing for
    serialization both to and from the gRPC CallerId type as well.
    """
    def __init__(self, id : Optional[uuid.UUID] = None):
        """
        Creates a new instance, either using |id| or a random GUID if no |id| is
        provided.
        """
        if id is None:
            id = uuid.uuid4()

        assert issubclass(type(id), uuid.UUID)
        self.__id = id
        
    @classmethod
    def try_parse(cls, id : str | CallerId) -> Optional['CallerIdentifier']:
        """
        Tries to parse |id| into a GUID, returning an insance of this object on
        success and None on failure.
        """
        if issubclass(type(id), CallerId):
            id = id.id

        if not isinstance(id, str):
            return None
        
        try:
            id = uuid.UUID(id)
            return CallerIdentifier(id)
        except ValueError:
            return None

    def to_grpc_type(self) -> CallerId:
        """
        Returns a gRPC CallerId representation of this object.
        """
        return CallerId(id = str(self.__id))
    
    def __hash__(self):
        return self.__id.__hash__()
    
    def __eq__(self, other):
        if other is None:
            return False
        
        if not issubclass(type(other), CallerIdentifier):
            return False
        
        return self.__id == other.__id
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.__id.__str__()
    
    def __repr__(self):
        return self.__id.__repr__()
    
    def __format__(self, format_spec: str):
        return self.__id.__format__(format_spec)