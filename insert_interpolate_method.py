import ast
import datetime
import textwrap # Added for dedent
from typing import Optional, TypeVar, Generic # Ensure these are in scope for type hints if needed by AST generation

# DataTypeT would be defined in the scope of the class as RemoteDataOrganizer(Generic[DataTypeT])
# For the script's standalone parsing of the method string, it might need a placeholder.
# However, ast.parse can handle undefined names in type hints if they are strings or forward refs.
# Let's assume DataTypeT is understood as a TypeVar in the context of the class.

method_to_insert_str = """
    def get_interpolated_at(self, timestamp: datetime.datetime) -> Optional[DataTypeT]:
        \"\"\"Gets a linearly interpolated data value for the given timestamp.

        Args:
            timestamp: The datetime timestamp to get the interpolated data for.

        Returns:
            The interpolated data of type DataTypeT if successful,
            or None if interpolation is not possible (e.g., timestamp is
            outside the range of existing data, or data is not available).
            If the timestamp matches an existing keyframe, its data is returned.
        \"\"\"
        # Ensure to use the correct mangled name for private attributes if accessed directly
        # The subtask report for step 2.1 mentioned self.__data was refactored to SortedList.
        # The mangled name is _RemoteDataOrganizer__data
        data_list = self._RemoteDataOrganizer__data

        if not data_list:  # Check if SortedList is empty
            return None

        # bisect_key_left uses the key function (item.timestamp) provided at SortedList construction
        idx = data_list.bisect_key_left(timestamp)

        # Case 1: Timestamp matches an existing keyframe
        if idx < len(data_list) and data_list[idx].timestamp == timestamp:
            return data_list[idx]

        # Case 2: Timestamp is before the first keyframe
        if idx == 0:
            return None

        # Case 3: Timestamp is after the last keyframe
        if idx == len(data_list):
            return None

        # Case 4: Timestamp is between two keyframes (data_list[idx-1] and data_list[idx])
        p1 = data_list[idx-1]
        p2 = data_list[idx]

        t1_dt = p1.timestamp
        t2_dt = p2.timestamp

        # Convert datetime to float seconds for calculation
        t1_float = t1_dt.timestamp()
        t2_float = t2_dt.timestamp()
        target_t_float = timestamp.timestamp()

        # Ensure t1 and t2 are different to avoid division by zero
        if t1_float == t2_float:
            # This case implies p1 and p2 are at the same timestamp.
            # If an exact match for `timestamp` existed, it would have been caught.
            # If target_t_float is also t1_float, then p1 (or p2) is effectively an exact match.
            # Given prior checks, this means target_t_float is strictly between distinct t1 & t2,
            # or this is an edge case of duplicate timestamps in data not representing distinct points for interpolation.
            return p1 # Or p2; if timestamps are same, values should ideally be same or interpolation is ill-defined.

        # Interpolate 'value' attribute if present, else assume data item itself is numerical.
        # This matches the requirement: "perform a linear interpolation on their data".
        # "Data" here is assumed to be DataTypeT or a numerical component of it.
        val1 = getattr(p1, "value", p1)
        val2 = getattr(p2, "value", p2)

        if not (isinstance(val1, (int, float)) and isinstance(val2, (int, float))):
            # Log this? print(f"Warning: Cannot interpolate non-numerical data types: {type(val1)}, {type(val2)}")
            return None # Cannot interpolate non-numerical types with this simple logic

        interpolated_numeric_value = val1 + (target_t_float - t1_float) * (val2 - val1) / (t2_float - t1_float)

        if val1 is p1: # DataTypeT itself was numerical (p1 was not an object with .value)
            # If original was int, and interpolation made it float, result is float.
            # This is standard Python behavior for division and float multiplication.
            # If strict original type (e.g. int) is required for return, explicit casting could be added:
            # if isinstance(p1, int): return int(interpolated_numeric_value)
            return interpolated_numeric_value # Type will be float or int depending on operations
        else:
            # DataTypeT was an object, and we interpolated its 'value' attribute.
            # We must return an instance of DataTypeT.
            try:
                # Prefer dataclasses.replace if available (or similar pattern)
                if hasattr(p1, "replace") and callable(getattr(p1, "replace")):
                    # Assumes 'value' and 'timestamp' are valid fields for replace.
                    return p1.replace(value=interpolated_numeric_value, timestamp=timestamp)
                # Fallback: try to call constructor type(p1)(value=..., timestamp=...)
                # This is speculative and depends on DataTypeT's constructor signature.
                elif callable(type(p1)): # Check if type(p1) is a constructor
                    # This assumes a constructor like: MyClass(value=new_value, timestamp=new_timestamp, ...)
                    # It might need other fields from p1 if they are required by constructor.
                    # A common pattern for data objects is to have a constructor that accepts all its fields.
                    # For simplicity, we only explicitly set 'value' and 'timestamp'.
                    # Other fields would need to be copied from p1 if constructor doesn't handle defaults.
                    # This is the most fragile part due to unknown DataTypeT.
                    # A more robust approach would require DataTypeT to implement a specific interface for this.
                    return type(p1)(value=interpolated_numeric_value, timestamp=timestamp)
                else:
                    # print(f"Cannot construct new {type(p1)} for interpolated data. No suitable 'replace' or constructor identified.")
                    return None
            except Exception as e:
                # print(f"Error constructing new {type(p1)} instance: {e}. Returning None.")
                return None
"""

file_path_to_modify = "tsercom/data/remote_data_organizer.py"

with open(file_path_to_modify, "r") as f_read:
    source_code = f_read.read()

tree = ast.parse(source_code)
class_def_node = None
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "RemoteDataOrganizer":
        class_def_node = node
        break

if class_def_node:
    insertion_idx = len(class_def_node.body) # Default: append to class

    # Try to find a logical insertion point
    # After public getters, before private methods
    public_getters = ["get_new_data", "get_most_recent_data", "get_data_for_timestamp"]
    last_getter_idx = -1
    first_private_method_idx = -1

    for i, member_node in enumerate(class_def_node.body):
        if isinstance(member_node, ast.FunctionDef):
            if member_node.name in public_getters:
                last_getter_idx = i
            if member_node.name.startswith("_") and first_private_method_idx == -1:
                first_private_method_idx = i

    if last_getter_idx != -1:
        insertion_idx = last_getter_idx + 1
    elif first_private_method_idx != -1:
        insertion_idx = first_private_method_idx

    # Dedent the method string before parsing
    dedented_method_str = textwrap.dedent(method_to_insert_str)
    method_ast_nodes = ast.parse(dedented_method_str).body
    for new_node_idx, new_node_to_insert in enumerate(method_ast_nodes):
        class_def_node.body.insert(insertion_idx + new_node_idx, new_node_to_insert)

    new_source_code = ""
    try:
        from ast import unparse as ast_unparse_native # Python 3.9+
        new_source_code = ast_unparse_native(tree)
    except ImportError:
        try:
            import astunparse # Try external library for older Python
            new_source_code = astunparse.unparse(tree)
        except ImportError:
            # This fallback is critical for environments without Python 3.9+ or astunparse
            print("CRITICAL_ERROR: ast.unparse or astunparse not found. Cannot reliably generate modified code.")
            # To prevent writing a potentially broken file, exit with error
            # Or, could attempt manual string reconstruction if absolutely necessary, but it's very risky.
            raise RuntimeError("AST unparsing tool (Python 3.9+ or astunparse) not found.")

    with open(file_path_to_modify, "w") as f_write:
        f_write.write(new_source_code)
    print(f"Method get_interpolated_at inserted into {file_path_to_modify}")

else:
    print(f"CRITICAL_ERROR: Class RemoteDataOrganizer not found in {file_path_to_modify}")
    raise FileNotFoundError(f"Class RemoteDataOrganizer not found in {file_path_to_modify}, cannot insert method.")
