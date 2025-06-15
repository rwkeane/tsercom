import ast
import re
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import grpc

kGeneratedDir = "generated"


def generate_proto_file(
    package_dir: Path, proto_file_path: str, import_paths: Iterable[str]
) -> None:
    # Escaped docstring:
    """Generates Python files for a single .proto file.

    Args:
        package_dir: The root directory of the package.
        proto_file_path: The path to the .proto file, relative to package_dir.
        import_paths: An iterable of import paths relative to package_dir.
    """
    absolute_proto_path = package_dir / proto_file_path
    output_dir = make_versioned_output_dir(absolute_proto_path.parent)

    protoc_import_paths = [f"-I{package_dir / path}" for path in import_paths]
    protoc_import_paths.append(f"-I{absolute_proto_path.parent}")
    protoc_import_paths.append(f"-I{package_dir}")
    protoc_import_paths.append(f"-I{package_dir.parent}")

    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        *protoc_import_paths,
        f"--python_out={output_dir}",
        f"--mypy_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        str(absolute_proto_path),
    ]
    print(f"Running command: {' '.join(command)}")

    env = os.environ.copy()
    local_bin_path = os.path.expanduser("~/.local/bin")
    if local_bin_path not in env["PATH"]:
        env["PATH"] = f"{local_bin_path}:{env['PATH']}"

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"Error compiling {proto_file_path}:")
        print(result.stderr)
        raise RuntimeError(
            f"protoc compilation failed for {proto_file_path}:\n{result.stderr}"
        )
    else:
        print(f"Successfully compiled {proto_file_path}")

    name = absolute_proto_path.name.split(".")[0]
    generate_init(package_dir, proto_file_path, output_dir.parent)

    pb2_file = output_dir / f"{name}_pb2.py"
    pyi_file = output_dir / f"{name}_pb2.pyi"
    modify_generated_file(pb2_file)
    modify_generated_file(pyi_file)


def make_versioned_output_dir(base_dir: Path) -> Path:
    """Creates a versioned output directory (e.g., base_dir/generated/v1_62)."""
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
        version_string = f"v{major_minor_version.replace('.', '_')}"
    except Exception as e:
        raise RuntimeError(
            f"Could not determine grpcio-tools version. Is it installed? Error: {e}"
        ) from e

    print(f"Generating protos for grpcio-tools version: {version_string}")
    generated_dir = base_dir / kGeneratedDir
    os.makedirs(generated_dir, exist_ok=True)
    versioned_output_base = generated_dir / version_string
    os.makedirs(versioned_output_base, exist_ok=True)
    return versioned_output_base


def modify_generated_file(file_path: Path) -> None:
    """Performs string replacement in the generated file."""
    updates: Dict[str, str] = {
        "import caller_id_pb2\n": "import tsercom.caller_id.proto as caller_id_pb2\n",
        "import time_pb2\n": "import tsercom.timesync.common.proto as time_pb2\n",
        "import common_pb2\n": "import tsercom.rpc.proto as common_pb2\n",
        "import tensor_pb2\n": "import tsercom.tensor.proto as tensor_pb2\n",  # Updated path
        "import caller_id_pb2 ": "import tsercom.caller_id.proto ",
        "import time_pb2 ": "import tsercom.timesync.common.proto ",
        "import common_pb2 ": "import tsercom.rpc.proto ",
        "import tensor_pb2 ": "import tsercom.tensor.proto ",  # Updated path
    }
    if not file_path.exists():
        print(f"Warning: Generated file not found, cannot modify: {file_path}")
        return
    try:
        with open(file_path, "r+") as f:
            content = f.read()
            for original, replacement in updates.items():
                content = content.replace(original, replacement)
            f.seek(0)
            f.write(content)
            f.truncate()
    except Exception as e:
        print(f"Error modifying file {file_path}: {e}")
        raise


def _update_pyproject_version_range(versioned_dirs: list[str]) -> None:
    """Updates pyproject.toml with the detected gRPC version range."""
    if not versioned_dirs:
        print(
            "Warning: No versioned directories. Skipping pyproject.toml update."
        )
        return
    versions = []
    for v_str in versioned_dirs:
        match = re.match(r"v(\d+)_(\d+)", v_str)
        if match:
            versions.append((int(match.group(1)), int(match.group(2))))
        else:
            match_single = re.match(r"v(\d+)", v_str)
            if match_single:
                versions.append((int(match_single.group(1)), 0))
            else:
                print(f"Warning: Could not parse version string: {v_str}")
    if not versions:
        print(
            "Warning: No valid versions parsed. Skipping pyproject.toml update."
        )
        return
    min_v, max_v = min(versions), max(versions)
    min_v_str = f"{min_v[0]}.{min_v[1]}.0"
    next_max_v_str = f"{max_v[0]}.{max_v[1] + 1}.0"
    try:
        script_dir = Path(__file__).parent.resolve()
        pyproject_toml = script_dir.parent / "pyproject.toml"
        if not pyproject_toml.exists():
            pyproject_toml = Path("pyproject.toml")
        content = pyproject_toml.read_text()
        patterns = [
            r'(grpcio\s*=\s*")[^"]*(")',
            r'(grpcio-tools\s*=\s*")[^"]*(")',
            r'(grpcio-status\s*=\s*")[^"]*(")',
            r'(grpcio-health-checking\s*=\s*")[^"]*(")',
        ]
        repl = f"\1>={min_v_str}, <{next_max_v_str}\2"
        for p in patterns:
            content = re.sub(p, repl, content)
        pyproject_toml.write_text(content)
        print(
            f"Updated {pyproject_toml} gRPC: >={min_v_str}, <{next_max_v_str}"
        )
    except Exception as e:
        print(f"Error updating pyproject.toml: {e}")


def get_public_symbols_from_file(filepath: Path) -> list[str]:
    """Gets a list of public class and function names defined in a Python file."""
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except Exception:
        return []
    return [
        n.name
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and not n.name.startswith("_")
    ]


def generate_init(
    package_dir: Path, proto_file_rel_path: str, generated_subpackage_dir: Path
) -> None:
    """
    Generates __init__.py for a protobuf module.
    package_dir: Absolute path to 'tsercom'.
    proto_file_rel_path: Relative path of .proto file from 'tsercom' (e.g., "tensor/proto/tensor.proto").
    generated_subpackage_dir: Path to dir containing versioned subdirs (e.g., tsercom/tensor/proto/generated).
    """

    def _parse_ver(v_str: str) -> Tuple[int, int]:
        match = re.match(r"v(\d+)_(\d+)", v_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        match_single = re.match(r"v(\d+)", v_str)
        if match_single:
            return int(match_single.group(1)), 0
        return (0, 0)

    init_content = '\nimport grpc\nimport subprocess  # noqa: F401\nfrom typing import TYPE_CHECKING\n\nif not TYPE_CHECKING:\n    try:\n        version = grpc.__version__\n        major_minor_version = ".".join(version.split(".")[:2])\n    except Exception:\n        major_minor_version = "1.62"\n    version_string = f"v{major_minor_version.replace(\'.\', \'_\')}"\n    if False: pass'

    proto_name = Path(proto_file_rel_path).name.split(".")[0]
    versioned_data = []

    init_py_location_dir = generated_subpackage_dir.parent
    module_py_path_prefix = (
        init_py_location_dir.relative_to(package_dir)
        .as_posix()
        .replace("/", ".")
    )

    for item in generated_subpackage_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            msg_syms = get_public_symbols_from_file(
                item / f"{proto_name}_pb2.pyi"
            )
            grpc_syms = get_public_symbols_from_file(
                item / f"{proto_name}_pb2_grpc.pyi"
            )
            if not grpc_syms:
                grpc_syms = get_public_symbols_from_file(
                    item / f"{proto_name}_pb2_grpc.py"
                )
            if msg_syms or grpc_syms:
                versioned_data.append((item.name, msg_syms, grpc_syms))

    if not versioned_data:
        print(
            f"Warning: No symbols for {proto_name} in {generated_subpackage_dir}"
        )
    _update_pyproject_version_range([vd[0] for vd in versioned_data if vd[0]])

    if versioned_data:
        sorted_vers = sorted(
            versioned_data, key=lambda x: _parse_ver(x[0]), reverse=True
        )
        # This is the fix for the elif chain: ensure the first one is an 'if' or the 'if False: pass' handles it.
        # The init_content starts with "if False: pass". The first real condition should be an elif.
        # is_first_elif = True # Removed as it's unused
        for v_name, msgs, grpcs in sorted_vers:
            # Corrected logic for if/elif
            # condition_keyword removed
            # This logic is tricky; the original script had 'if False: pass' then a series of 'elif'.
            # Let's stick to the original script's direct 'elif' chain following 'if False: pass'
            init_content += f'\n    elif version_string == "{v_name}":'
            # is_first_elif = False # No longer needed with direct elif
            if msgs:
                init_content += f"\n        from tsercom.{module_py_path_prefix}.{kGeneratedDir}.{v_name}.{proto_name}_pb2 import {', '.join(msgs)}"
            if grpcs:
                init_content += f"\n        from tsercom.{module_py_path_prefix}.{kGeneratedDir}.{v_name}.{proto_name}_pb2_grpc import {', '.join(grpcs)}"

        avail_vs = str(
            sorted([vd[0] for vd in versioned_data], key=_parse_ver)
        )
        init_content += f"\n    else:\n        raise ImportError(f\"No code for gRPC {{version}} ('{{version_string}}') for '{proto_name}'. Avail: {avail_vs}\")"
    else:
        init_content += f"\n    else: raise ImportError(f'No versioned code for proto '{proto_name}' found.')"

    init_content += "\nelse: # TYPE_CHECKING\n"
    if versioned_data:
        latest_v, latest_m, latest_g = max(
            versioned_data, key=lambda x: _parse_ver(x[0])
        )
        type_imports = []
        if latest_m:
            type_imports.extend(
                [
                    f"    from tsercom.{module_py_path_prefix}.{kGeneratedDir}.{latest_v}.{proto_name}_pb2 import {s} as {s}"
                    for s in latest_m
                ]
            )
        if latest_g:
            type_imports.extend(
                [
                    f"    from tsercom.{module_py_path_prefix}.{kGeneratedDir}.{latest_v}.{proto_name}_pb2_grpc import {s} as {s}"
                    for s in latest_g
                ]
            )
        if type_imports:
            init_content += "\n".join(type_imports) + "\n"
        else:
            init_content += "    pass # No symbols for TYPE_CHECKING\n"
    else:
        init_content += "    pass # No versioned data for TYPE_CHECKING\n"

    init_py_file = generated_subpackage_dir.parent / "__init__.py"
    try:
        init_py_file.write_text(init_content)
        print(f"Generated __init__.py at {init_py_file}")
    except IOError as e:
        print(f"Error writing __init__.py: {e}")
        raise


def generate_protos(project_root: Path) -> None:
    """Generates Python files from .proto files for the entire project."""
    assert isinstance(project_root, Path)
    package_dir = project_root / "tsercom"

    generate_proto_file(
        package_dir,
        "timesync/common/proto/time.proto",
        ["timesync/common/proto"],
    )
    generate_proto_file(
        package_dir, "caller_id/proto/caller_id.proto", ["caller_id/proto"]
    )
    generate_proto_file(
        package_dir,
        "rpc/proto/common.proto",
        ["caller_id/proto", "rpc/proto", "timesync/common/proto"],
    )
    # Updated call for tensor.proto:
    generate_proto_file(
        package_dir,
        "tensor/proto/tensor.proto",
        ["timesync/common/proto", "tensor/proto"],
    )
    generate_proto_file(
        package_dir, "test/proto/e2e_test_service.proto", ["test/proto"]
    )


if __name__ == "__main__":
    project_root_path = Path(__file__).resolve().parent.parent
    generate_protos(project_root_path)
