from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass(frozen=True) # Typically, type descriptors should be immutable
class CustomDataType:
    """
    Represents the type of data, especially for serialization or when
    passing type information across process boundaries.

    It stores the module and class name of the data type.
    An optional 'properties' field can store further details like
    tensor shape/dtype if needed, though this might be better handled
    by specific metadata structures for complex types.
    """
    module: str
    class_name: str
    properties: Optional[Dict[str, Any]] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"{self.module}.{self.class_name}"

    def __repr__(self) -> str:
        return f"CustomDataType(module='{self.module}', class_name='{self.class_name}', properties={self.properties})"

    @classmethod
    def from_type(cls, data_type: type) -> "CustomDataType":
        """
        Creates a CustomDataType instance from a Python type object.
        """
        return cls(module=data_type.__module__, class_name=data_type.__name__)

    # Example usage from AggregatingMultiprocessQueue:
    # meta_data_type = CustomDataType(module="torch", class_name="Tensor")
    # type_of_metadata = CustomDataType(module="tsercom.common.custom_data_type", class_name="CustomDataType")

def get_custom_data_type(obj: Any) -> Optional[CustomDataType]:
    """
    Infers the CustomDataType from an object.

    Args:
        obj: The object whose type is to be inferred.

    Returns:
        A CustomDataType instance if the type can be determined and is not
        a built-in primitive that doesn't typically need explicit typing
        (e.g., int, str, bool, NoneType), otherwise None.
        Returns CustomDataType for known complex types like torch.Tensor
        or user-defined classes.
    """
    if obj is None or isinstance(obj, (int, float, str, bool, bytes, dict, list, tuple, set)):
        # For basic Python types, we might not need to send CustomDataType,
        # or we could define specific ones if needed (e.g., for differentiating list of ints vs list of floats).
        # For now, returning None for these, assuming they are handled by default serialization.
        return None

    obj_type = type(obj)

    # Special handling for common types if needed, e.g., torch.Tensor
    # For example, if obj is a torch.Tensor:
    # if 'torch' in str(obj_type) and 'Tensor' in str(obj_type.__name__):
    #     return CustomDataType(module=obj_type.__module__, class_name=obj_type.__name__)

    try:
        # For most other objects, try to get module and class name.
        return CustomDataType.from_type(obj_type)
    except Exception: # pylint: disable=broad-except
        # If type inference fails for any reason, return None.
        return None
