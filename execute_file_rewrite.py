import re

original_content = """""" # This will be replaced by the actual file content from the previous tool run

# Remove the start and end markers from read_files
start_marker_str = "[start of tsercom/threading/multiprocess/delegating_multiprocess_queue_factory_unittest.py]"
end_marker_str = "[end of tsercom/threading/multiprocess/delegating_multiprocess_queue_factory_unittest.py]"

# Clean the start
if original_content.startswith(start_marker_str):
    original_content = original_content[len(start_marker_str):].lstrip()

# Clean any occurrences of the end marker (especially if duplicated)
lines = original_content.splitlines()
cleaned_lines = [line for line in lines if line.strip() != end_marker_str] # Use strip() for robustness
python_code_content = "\n".join(cleaned_lines)

new_worker_code = """
def _minimal_producer_worker(
    tmp_queue: Any, # Should be torch.multiprocessing.Queue
    tensor_to_send: TensorType,
    result_q: Any # Standard ctx.Queue for status
):
    try:
        print(f"[MinimalProducerMODIFIED] Original tensor: {tensor_to_send}")

        # Ensure contiguity and then share memory
        if not tensor_to_send.is_contiguous():
            print("[MinimalProducerMODIFIED] Tensor not contiguous. Making it contiguous.")
            tensor_to_send = tensor_to_send.contiguous()

        print("[MinimalProducerMODIFIED] Calling share_memory_() on tensor.")
        tensor_to_send.share_memory_() # Explicitly share memory
        print(f"[MinimalProducerMODIFIED] Putting tensor after share_memory_(): {tensor_to_send}")
        tmp_queue.put(tensor_to_send)
        print(f"[MinimalProducerMODIFIED] Tensor put successfully.")
        result_q.put("producer_success")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[MinimalProducerMODIFIED] EXCEPTION: {e}\\n{tb_str}") # Escaped newline for Python string
        result_q.put(f"producer_exception: {type(e).__name__}: {e}")
"""

# Replace the old _minimal_producer_worker with the new one.
start_marker_func = "def _minimal_producer_worker("
# This regex looks for the function definition until the next 'def ' at the same indentation or end of string.
pattern = re.compile(r"^(def _minimal_producer_worker\(.*?\):(?:\n(?:[ \t].*)*\n*))+", re.MULTILINE)

match = pattern.search(python_code_content)
if match:
    # Ensure new_worker_code ends with a newline to separate it from the following code
    updated_content = pattern.sub(new_worker_code.strip() + "\n\n", python_code_content, 1)
    print("SUCCESS: _minimal_producer_worker replaced.")
else:
    print("ERROR: Could not find _minimal_producer_worker to replace. Original content (or part of it) will be written if this is an error.")
    updated_content = "ERROR: Replacement failed in Python script. Original content might be lost or corrupted."


with open("tsercom/threading/multiprocess/delegating_multiprocess_queue_factory_unittest.py", "w") as f:
    f.write(updated_content)
