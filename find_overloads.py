import ast
import json
import sys

def find_overloaded_functions(file_paths):
    overloaded_functions = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as source_file:
                content = source_file.read()
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}", file=sys.stderr)
            continue

        if file_path.endswith("__init__.py"):
            try:
                tree = ast.parse(content)
                has_function_defs = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                if not has_function_defs:
                    continue  # Skip __init__.py if it has no function definitions
            except SyntaxError as e:
                print(f"SyntaxError parsing {file_path}: {e}", file=sys.stderr)
                continue


        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"SyntaxError parsing {file_path}: {e}", file=sys.stderr)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                is_overload = False
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute): # typing.overload
                        if isinstance(decorator.value, ast.Name) and decorator.value.id == 'typing' and decorator.attr == 'overload':
                            is_overload = True
                            break
                    elif isinstance(decorator, ast.Name): # @overload (assuming from typing import overload)
                        if decorator.id == 'overload':
                             # This is a simplification. A real check would involve
                             # confirming 'overload' was imported from 'typing'.
                             # For this script, we'll assume 'overload' implies 'typing.overload'
                             # if 'typing' is imported in the file.
                            has_typing_import = any(
                                isinstance(imp_node, ast.Import) and any(alias.name == 'typing' for alias in imp_node.names) or
                                isinstance(imp_node, ast.ImportFrom) and imp_node.module == 'typing'
                                for imp_node in ast.walk(tree) if isinstance(imp_node, (ast.Import, ast.ImportFrom))
                            )
                            if has_typing_import:
                                is_overload = True
                                break
                if is_overload:
                    overloaded_functions.append({
                        "file_path": file_path,
                        "function_name": node.name
                    })
    return overloaded_functions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_overloads.py <file1.py> <file2.py> ...", file=sys.stderr)
        sys.exit(1)

    files_to_check = sys.argv[1:]
    results = find_overloaded_functions(files_to_check)
    print(json.dumps(results, indent=2))
