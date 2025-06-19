import os
import sys # For exit status

file_path = "tsercom/data/remote_data_organizer_unittest.py"
temp_file_path = file_path + ".tmp"

# Characters to remove or replace.
# U+0001 (SOH) is \x01 in bytes
# U+0002 (STX) is \x02 in bytes
# Bad pattern: b'.\x01.add(\x02'
# Should become: b'.add('

try:
    with open(file_path, "rb") as f_in, open(temp_file_path, "wb") as f_out:
        for line_bytes in f_in:
            # Fix the specific ".SOH.add(STX" pattern
            modified_line_bytes = line_bytes.replace(b'.\x01.add(\x02', b'.add(')

            # Remove any remaining isolated SOH or STX characters
            modified_line_bytes = modified_line_bytes.replace(b'\x01', b'')
            modified_line_bytes = modified_line_bytes.replace(b'\x02', b'')

            f_out.write(modified_line_bytes)

    os.replace(temp_file_path, file_path)
    print(f"Cleaned non-printable characters from {file_path}")
    sys.exit(0) # Explicitly exit with 0 on success

except Exception as e:
    print(f"Error during cleaning script: {e}")
    # Restore backup if script failed partway
    if os.path.exists(file_path + ".bak"):
        if os.path.exists(temp_file_path): # remove temp if it exists
            os.remove(temp_file_path)
        os.replace(file_path + ".bak", file_path)
        print(f"Restored {file_path} from backup.")
    sys.exit(1) # Indicate failure
