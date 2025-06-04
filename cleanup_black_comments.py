import os
import re

# General patterns for full lines. Order might matter if one is a subset of another.
# More specific ones should ideally come first if there's overlap.
patterns = [
    re.compile(
        r"^\s*# Black-formatted import.*$"
    ),  # Handles if there's trailing text
    re.compile(
        r"^\s*# Black-formatted.*$"
    ),  # General catch-all for full lines
]

# General pattern for inline comments
inline_patterns = [
    re.compile(
        r"\s*# Black-formatted.*$"
    ),  # Matches '# Black-formatted' and any trailing text
]


def process_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    original_content = "".join(lines)
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        processed_line = line
        # Process inline comments first
        for ipattern in inline_patterns:
            # Important: sub() replaces, so if multiple inline patterns could match, ensure behavior is what you want.
            # Here, one general pattern is likely enough.
            processed_line = ipattern.sub("", processed_line)

        is_full_line_comment = False
        # Use rstrip to remove potential trailing newline before matching full line patterns
        line_to_match_full = line.rstrip("\r\n")
        if (
            processed_line == line_to_match_full
            or processed_line == line_to_match_full + "\n"
        ):  # No inline comment was removed from this line
            for pattern in patterns:
                if pattern.match(line_to_match_full):
                    is_full_line_comment = True
                    break

        if is_full_line_comment:
            # Current line is a full-line comment to be removed.
            if new_lines and new_lines[-1].strip() == "":
                new_lines.pop()  # Remove the preceding blank line
            i += 1
            continue

        # If it wasn't a full-line comment, add the processed_line
        # (which might have had an inline comment removed)
        # Only add if it's not completely empty after inline removal,
        # unless it was originally just a newline character.
        if processed_line.strip() == "" and line.strip() != "":
            # Line became blank after inline comment removal.
            # We effectively treat it as if the line was removed for the purpose of preceding-blank-line-logic.
            if new_lines and new_lines[-1].strip() == "":
                new_lines.pop()
            i += 1
            continue
        elif (
            processed_line.strip() == "" and line.strip() == ""
        ):  # Original line was blank
            new_lines.append(
                line
            )  # Preserve originally blank lines (that weren't preceding a removed comment)
            i += 1
        elif processed_line.strip() != "":  # Line has content
            new_lines.append(processed_line)
            i += 1
        else:  # Line is now empty but was not originally empty and not stripped by above. e.g only had an inline comment.
            i += 1  # Skip adding it.

    final_content = "".join(new_lines)

    if original_content != final_content:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Modified {filepath}")
        except Exception as e:
            print(f"Error writing {filepath}: {e}")


print("Starting comment removal process (round 2)...")
# Restrict walk to tsercom directory to avoid build artifacts
for root, _, files in os.walk("tsercom"):
    for file in files:
        if file.endswith(".py"):
            process_file(os.path.join(root, file))

print("Comment removal process (round 2) complete.")
