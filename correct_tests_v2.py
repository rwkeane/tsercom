import re
import ast
import textwrap # Ensure textwrap is imported
from typing import List, Dict, Any, Optional, Union # Ensure Optional and Union are imported

# Ensure helper functions are at the module level (top level of this script)
def get_data_creation_info(test_function_code: str) -> Dict[str, Dict[str, Any]]:
    var_info: Dict[str, Dict[str, Any]] = {}
    # Adjusted regex to better handle various ways of calling create_data, including direct datetime objects
    # This regex is still a heuristic and might not cover all edge cases.
    pattern = re.compile(
        r"(\w+)\s*=\s*create_data\s*\("         # var_name = create_data(
        r"[^,]+,\s*"                            # first arg (caller_id),
        r"([^,]+?)\s*"                          # second arg (timestamp_input - captured)
        r"(?:,\s*value_id\s*=\s*([\w\.]+))?\s*\)"   # optional third arg (value_id - captured)
    )

    for match in pattern.finditer(test_function_code):
        var_name = match.group(1)
        ts_input_str = match.group(2).strip() # Ensure stripped
        value_id_str = match.group(3)

        ts_val: Any = ts_input_str
        try:
            ts_val = float(ts_input_str)
        except ValueError:
            pass

        value_id_val: Any = 0
        if value_id_str:
            try:
                value_id_val = int(value_id_str)
            except ValueError:
                value_id_val = value_id_str

        var_info[var_name] = {"ts_input": ts_val, "value_id": value_id_val, "ts_str": ts_input_str}
    return var_info

def sort_list_assertion_vars(var_names: List[str], var_info: Dict[str, Dict[str, Any]]) -> List[str]:
    def get_sort_key(var_name: str) -> Any:
        info = var_info.get(var_name)
        if info:
            if isinstance(info["ts_input"], (int, float)):
                return info["ts_input"]
            return info["ts_str"]
        return var_name

    known_vars = [v for v in var_names if v in var_info]
    unknown_vars = [v for v in var_names if v not in var_info]

    try:
        known_vars.sort(key=get_sort_key)
    except TypeError:
        known_vars.sort(key=lambda vn: str(get_sort_key(vn)))
    return known_vars + unknown_vars

def parse_vars_from_list_str(list_str: str) -> List[str]:
    try:
        cleaned_str = list_str.strip()
        if not cleaned_str.startswith("[") or not cleaned_str.endswith("]"):
            return []
        cleaned_str = cleaned_str[1:-1].strip()
        if not cleaned_str:
            return []
        return [var.strip() for var in cleaned_str.split(',')]
    except Exception as e:
        return []

new_dummy_class_def_str = """
class DummyExposedDataForOrganizerTests(ExposedData):
    __test__ = False  # Mark this class as not a test class for pytest
    value: Optional[Any] = None # Declare class attribute with default

    def __init__(self, caller_id: 'DummyCallerIdentifier', timestamp: datetime.datetime, value: Optional[Any] = None):
        # ExposedData's actual __init__ is (self, timestamp: datetime.datetime)
        super().__init__(timestamp)  # Pass only timestamp to ExposedData's __init__
        self.caller_id = caller_id   # Manage caller_id in this subclass
        self.value = value

    def __repr__(self) -> str: # Added return type
        value_repr = getattr(self, "value", "N/A")
        caller_id_repr = "UnknownCaller"
        if hasattr(self, 'caller_id'):
            if hasattr(self.caller_id, 'id_str'):
                caller_id_repr = self.caller_id.id_str
            elif hasattr(self.caller_id, 'name'):
                caller_id_repr = self.caller_id.name

        ts_repr = self.timestamp.isoformat() if hasattr(self, 'timestamp') else "UnknownTimestamp"
        return f"DummyExposedDataForOrganizerTests(caller_id='{caller_id_repr}', timestamp='{ts_repr}', value={value_repr})"

    def replace(self, *, value: Any, timestamp: datetime.datetime) -> "DummyExposedDataForOrganizerTests":
        if not hasattr(self, 'caller_id'):
            placeholder_caller_id = DummyCallerIdentifier("placeholder_during_replace")
            return DummyExposedDataForOrganizerTests(caller_id=placeholder_caller_id, timestamp=timestamp, value=value)
        return DummyExposedDataForOrganizerTests(caller_id=self.caller_id, timestamp=timestamp, value=value)
"""

# --- Main script logic ---
test_file_path = "tsercom/data/remote_data_organizer_unittest.py"
with open(test_file_path, "r") as f:
    original_content = f.read()

# 1. Modify DummyExposedDataForOrganizerTests
class_def_pattern = re.compile(
    r"(class DummyExposedDataForOrganizerTests\(ExposedData\):.*?)(?=\n\n\n# --- Fixtures ---|\Z)",
    re.DOTALL | re.MULTILINE # Ensure we capture up to the fixtures or end of file
)
replacement_class_code_dedented = textwrap.dedent(new_dummy_class_def_str).strip()

def class_replacer(match):
    end_marker = match.group(1) # This captures the content *after* the class block usually
                                # The regex needs to capture the class block then the end marker separately
                                # For now, this might replace more than intended if not careful.
                                # A safer regex for replacing the class only:
                                # r"(class DummyExposedDataForOrganizerTests\(ExposedData\):(\n(?:    |\n).*?)*?)(?=\nclass|\n@pytest\.fixture|\Z)"
                                # For this script, the provided regex is used as is.
    # The goal is to replace the class and keep what's after.
    # The pattern "(class DummyExposedDataForOrganizerTests\(ExposedData\):.*?)(?=\n\n\n# --- Fixtures ---|\Z)"
    # means group 1 IS the class block. So we just replace group 0.
    return replacement_class_code_dedented + "\n" # Add a newline after the class

temp_content, num_replacements = class_def_pattern.subn(class_replacer, original_content, 1)

if num_replacements == 0:
    # Fallback: try to find the class and insert if the end marker logic failed
    class_start_str = "class DummyExposedDataForOrganizerTests(ExposedData):"
    class_end_str = "# --- Fixtures ---" # Common marker
    start_idx = original_content.find(class_start_str)
    if start_idx != -1:
        end_idx = original_content.find(class_end_str, start_idx)
        if end_idx != -1:
            original_content = original_content[:start_idx] + replacement_class_code_dedented + "\n\n" + original_content[end_idx:]
            print("SUCCESS (fallback): DummyExposedDataForOrganizerTests class definition updated with .replace() method.")
        else:
            print("WARNING (fallback): DummyExposedDataForOrganizerTests class end marker not found.")
    else:
        print("WARNING: DummyExposedDataForOrganizerTests class definition not found.")
else:
    original_content = temp_content
    print("SUCCESS: DummyExposedDataForOrganizerTests class definition updated with .replace() method.")


# 2. Ensure all .appendleft() and .append() on __data are .add()
data_manipulation_pattern_appendleft = re.compile(r"((?:organizer(?:_no_client|_custom_type)?)\._RemoteDataOrganizer__data)\.appendleft\(")
data_manipulation_pattern_append = re.compile(r"((?:organizer(?:_no_client|_custom_type)?)\._RemoteDataOrganizer__data)\.append\(")
original_content = data_manipulation_pattern_appendleft.sub(r"\1.add(", original_content)
original_content = data_manipulation_pattern_append.sub(r"\1.add(", original_content)
print("INFO: Replaced .appendleft() and .append() with .add() where found.")

# 3. Attempt to reorder lists in assertions
lines = original_content.splitlines(True)
output_lines = []
current_test_function_code_lines: List[str] = []
in_test_function_block = False
var_info_cache: Optional[Dict[str, Dict[str, Any]]] = None

list_assertion_pattern = re.compile(r"(\s*assert\s+list\s*\(\s*(?:organizer(?:_no_client|_custom_type)?)\._RemoteDataOrganizer__data\s*\)\s*==\s*)(\[.*?\])(\s*#.*)?")

for line_idx, current_line in enumerate(lines):
    stripped_line = current_line.strip()
    is_test_def_line = stripped_line.startswith("def test_")

    # Determine if exiting a block
    # A line is an exit if it's not empty, not indented (or less indented than typical test body)
    # and it's not a comment. This is heuristic.
    likely_end_of_block = in_test_function_block and \
                           (not current_line.startswith("    ") and \
                            not current_line.startswith("  # type: ignore") and \
                            stripped_line != "" and not stripped_line.startswith("#"))

    if is_test_def_line or (likely_end_of_block and not is_test_def_line) :
        if current_test_function_code_lines: # Process accumulated block
            full_test_body_for_parsing = "".join(current_test_function_code_lines)
            var_info_cache = get_data_creation_info(full_test_body_for_parsing)

            for block_line in current_test_function_code_lines:
                assertion_match = list_assertion_pattern.search(block_line)
                if assertion_match and var_info_cache is not None:
                    assertion_start_text, list_content_str, existing_comment_str = assertion_match.groups()
                    existing_comment_str = existing_comment_str or ""

                    vars_in_assertion = parse_vars_from_list_str(list_content_str)

                    if vars_in_assertion:
                        all_vars_known = all(v_name in var_info_cache for v_name in vars_in_assertion)
                        sorted_vars = sort_list_assertion_vars(vars_in_assertion, var_info_cache)
                        new_list_content_str = "[" + ", ".join(sorted_vars) + "]"

                        if new_list_content_str != list_content_str and all_vars_known:
                            output_lines.append(f"{assertion_start_text}{new_list_content_str}{existing_comment_str}  # SortedList: Ascending order (auto-adjusted)\n")
                        elif new_list_content_str == list_content_str and all_vars_known and "# SortedList:" not in existing_comment_str and "# TODO:" not in existing_comment_str:
                             output_lines.append(f"{assertion_start_text}{list_content_str}{existing_comment_str}  # SortedList: Order confirmed\n")
                        elif ("# TODO: Verify order" not in existing_comment_str and "# SortedList:" not in existing_comment_str):
                            output_lines.append(f"{assertion_start_text}{list_content_str}{existing_comment_str}  # TODO: Verify order (vars not in create_data)\n")
                        else:
                            output_lines.append(block_line)
                    else:
                        output_lines.append(block_line)
                else:
                    output_lines.append(block_line)
            current_test_function_code_lines = []

        if is_test_def_line:
            in_test_function_block = True
            current_test_function_code_lines.append(current_line)
        else: # Was end of block, and current line is not a test def, so append it
            in_test_function_block = False
            output_lines.append(current_line)

    elif in_test_function_block:
        current_test_function_code_lines.append(current_line)
    else: # Line is outside any test function block
        output_lines.append(current_line)

# Process any remaining block if file ends within a test function
if current_test_function_code_lines: # Process any remaining lines
    full_test_body_for_parsing = "".join(current_test_function_code_lines)
    var_info_cache = get_data_creation_info(full_test_body_for_parsing)
    for block_line in current_test_function_code_lines:
        assertion_match = list_assertion_pattern.search(block_line)
        if assertion_match and var_info_cache is not None: # Duplicated logic, refactor if time
            assertion_start_text, list_content_str, existing_comment_str = assertion_match.groups()
            existing_comment_str = existing_comment_str or ""
            vars_in_assertion = parse_vars_from_list_str(list_content_str)
            if vars_in_assertion:
                all_vars_known = all(v_name in var_info_cache for v_name in vars_in_assertion)
                sorted_vars = sort_list_assertion_vars(vars_in_assertion, var_info_cache)
                new_list_content_str = "[" + ", ".join(sorted_vars) + "]"
                if new_list_content_str != list_content_str and all_vars_known:
                    output_lines.append(f"{assertion_start_text}{new_list_content_str}{existing_comment_str}  # SortedList: Ascending order (auto-adjusted)\n")
                elif new_list_content_str == list_content_str and all_vars_known and "# SortedList:" not in existing_comment_str and "# TODO:" not in existing_comment_str:
                     output_lines.append(f"{assertion_start_text}{list_content_str}{existing_comment_str}  # SortedList: Order confirmed\n")
                elif ("# TODO: Verify order" not in existing_comment_str and "# SortedList:" not in existing_comment_str):
                    output_lines.append(f"{assertion_start_text}{list_content_str}{existing_comment_str}  # TODO: Verify order (vars not in create_data)\n")
                else:
                    output_lines.append(block_line)
            else:
                output_lines.append(block_line)
        else:
             output_lines.append(block_line)


with open(test_file_path, "w") as f:
    f.writelines(output_lines)

print(f"Applied test logic corrections to {test_file_path}.")
