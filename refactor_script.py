import os
import datetime
import torch # Assuming torch is available as it was installed.
import re

# ---- SCRIPT TO MODIFY THE FILE ----

TARGET_FILE_PATH = "tsercom/tensor/demuxer/tensor_demuxer.py"

try:
    with open(TARGET_FILE_PATH, "r") as f:
        original_content = f.read()
except FileNotFoundError:
    print(f"Error: The target file {TARGET_FILE_PATH} was not found.")
    exit(1)

# --- Modification logic ---

class_def_index = original_content.find("class TensorDemuxer:")
if class_def_index == -1:
    print(f"Error: Could not find 'class TensorDemuxer:' in {TARGET_FILE_PATH}")
    exit(1)

target_method_name = "on_update_received"
method_def_pattern = f"async def {target_method_name}("
method_def_index = original_content.find(method_def_pattern)

if method_def_index == -1:
    print(f"Error: Could not find 'async def {target_method_name}(' method in {TARGET_FILE_PATH}.")
    exit(1) # Expecting async def for this specific method

# Determine indentation (simplified approach, assuming common styling)
indentation = ""
line_start_for_indent = original_content.rfind("\n", 0, method_def_index) + 1
potential_indent = original_content[line_start_for_indent : method_def_index]
if potential_indent.isspace():
    indentation = potential_indent
else:
    # Fallback to finding __init__ indent
    init_method_index = original_content.find("\n    def __init__(") # common pattern
    if init_method_index != -1:
        indentation = "    "
    else: # last resort
        indentation = "    "
    # If method_def_index is at the start of the file or class without prior newline:
    if method_def_index > 0 and original_content[method_def_index-1] != '\n':
        # This case is complex for simple indent finding; rely on fallback or known structure
        pass


new_hook_method_definition = f"""
{indentation}async def _on_keyframe_updated(self, timestamp: datetime.datetime, new_tensor_state: torch.Tensor) -> None:
{indentation}    \"\"\"Called when a keyframe's tensor value is finalized.

{indentation}    This hook is triggered for both newly arrived chunks and for every keyframe
{indentation}    that gets re-calculated during a cascade.

{indentation}    Args:
{indentation}        timestamp: The timestamp of the updated keyframe.
{indentation}        new_tensor_state: The new tensor state for the keyframe.
{indentation}    \"\"\"
{indentation}    if self.client: # Accessing client via property
{indentation}        await self.client.on_tensor_changed(new_tensor_state, timestamp)
{indentation}    else:
{indentation}        pass
"""

# Insert the hook method definition before the target method
insertion_point = method_def_index
insertion_point = original_content.rfind("\n", 0, insertion_point) + 1 if insertion_point > 0 else 0

# Check if hook already exists to prevent duplicate insertion from reruns
hook_already_exists = "async def _on_keyframe_updated(" in original_content
modified_content = original_content

if not hook_already_exists:
    modified_content = (
        original_content[:insertion_point]
        + new_hook_method_definition
        + "\n"
        + original_content[insertion_point:]
    )
    print("Added _on_keyframe_updated method definition.")
else:
    print("_on_keyframe_updated method already exists. Skipping addition.")


num_replacements = 0
try:
    # CORRECTED regex for the client call: uses self.__client
    client_call_pattern = re.compile(
        r"(await\s+)?self\.__client\.on_tensor_changed\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)"
    )

    replacement_function = lambda m: f"{m.group(1)if m.group(1) else ''}self._on_keyframe_updated(timestamp={m.group(3).strip()}, new_tensor_state={m.group(2).strip()})"

    # Attempt to replace within the specific method first for safety, then global if that fails.
    # For simplicity in this iteration, due to issues with robustly finding method body,
    # directly attempt a global replacement. This is riskier if the call pattern
    # appears elsewhere with different semantics, but the previous attempt at targeted replacement failed.
    # The subtask description implies this call is the one to change for keyframe updates.

    # Perform global replacement on the (potentially) hook-added content
    modified_content_after_replacement, num_replacements = client_call_pattern.subn(replacement_function, modified_content)

    if num_replacements > 0:
        modified_content = modified_content_after_replacement
        print(f"Globally replaced {num_replacements} calls to self.__client.on_tensor_changed.")
    else:
        print("Global replacement: No calls to self.__client.on_tensor_changed were found or replaced.")

except Exception as e:
    print(f"An error occurred during regex replacement for {target_method_name}: {e}")
    # modified_content remains as it was before this try block if error, or with hook added.

# Ensure necessary imports (already checked in original file, but good practice)
# This part can be removed if we are sure imports are fine.
# For now, keeping it.
if "import datetime" not in modified_content and "from datetime import datetime" not in modified_content :
    if not re.search(r"if TYPE_CHECKING:\s*(?:[^}]*\n)*?\s*(?:import datetime|from datetime import datetime)", modified_content):
        modified_content = "import datetime\n" + modified_content
        # print("Added 'import datetime'.") # Less verbose

if "import torch" not in modified_content:
    if not re.search(r"if TYPE_CHECKING:\s*(?:[^}]*\n)*?\s*import torch", modified_content):
        modified_content = "import torch\n" + modified_content
        # print("Added 'import torch'.") # Less verbose


with open(TARGET_FILE_PATH, "w") as f:
    f.write(modified_content)

print(f"Refactoring script for {TARGET_FILE_PATH} completed.")

# Final verification prints (simplified)
with open(TARGET_FILE_PATH, "r") as f:
    final_content_check = f.read()

if "_on_keyframe_updated" not in final_content_check:
    print("Error: _on_keyframe_updated method definition not found in the final file.")
else:
    print("_on_keyframe_updated method is present.")

if num_replacements > 0:
    if "self._on_keyframe_updated(" not in final_content_check:
        print("Warning: Calls to self._on_keyframe_updated were expected (due to num_replacements > 0) but not found.")
    else:
        print("Calls to self._on_keyframe_updated are present.")
    if client_call_pattern.search(final_content_check): # Check if any old calls remain
        print("Warning: Some original self.__client.on_tensor_changed calls might still be present despite replacements.")
else: # num_replacements == 0
    if client_call_pattern.search(final_content_check):
        print("Note: No replacements made, and original 'self.__client.on_tensor_changed' calls still exist.")
    else:
        # This case means: no replacements AND no original calls found by the pattern.
        # This could be fine if the file initially had no such calls, or problem if they were missed.
        print("Note: No replacements made, and no 'self.__client.on_tensor_changed' calls matching the pattern were found in the final file.")

print("Script finished.")
