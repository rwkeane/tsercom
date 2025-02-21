import subprocess
import os
import sys
import re
from pathlib import Path
from typing import Dict, Iterable

def generate_protos(project_root : Path):
    """Generates Python files from .proto files."""

    assert isinstance(project_root, Path)
    package_dir = Path.joinpath(project_root, "tsercom")

    def generate_proto_file(proto_file_path: str, import_paths: Iterable[str]):
        """Generates Python files for a single .proto file."""

        absolute_proto_path = Path.joinpath(package_dir, proto_file_path)
        output_dir = absolute_proto_path.parent.parent

        os.makedirs(output_dir, exist_ok=True)  # Ensure output dir exists

        command = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            *[f"-I{package_dir / path}" for path in import_paths],
            f"--python_out={output_dir}",
            f"--mypy_out={output_dir}",
            str(absolute_proto_path),
        ]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error compiling {proto_file_path}:")
            print(result.stderr)
            raise RuntimeError(
                    f"protoc compilation failed for {proto_file_path}:\n{result.stderr}")
        else:
            print(f"Successfully compiled {proto_file_path}")

        name = absolute_proto_path.name.split(".")[0]

        # --- Get generated file paths (relative to output_dir) ---
        pb2_file = Path.joinpath(output_dir, f"{name}_pb2.py")
        pb2_grpc_file = pb2_file = Path.joinpath(output_dir, f"{name}_pb2_grpc.py")
        pyi_file = pb2_file = Path.joinpath(output_dir, f"{name}_pb2.pyi")

        modify_generated_file(pb2_file)       #Modify
        if os.path.exists(pb2_grpc_file): # Check if grpc exists
            modify_generated_file(pb2_grpc_file)  # gRPC modify
        modify_generated_file(pyi_file)      #Modify


    def modify_generated_file(file_path: Path):
        """Performs string replacement in the generated file."""

        updates: Dict[str, str] = {
            "\nimport caller_id_pb2": "\nfrom caller_id import caller_id_pb2",
            "\nimport time_pb2": "\nfrom timesync.common import time_pb2",
            "\nimport common_pb2": "\nfrom rpc import common_pb2",
        }

        try:
            with open(file_path, 'r+') as f:
                content = f.read()
                for original, replacement in updates.items():
                    content = content.replace(original, replacement)
                f.seek(0)
                f.write(content)
                f.truncate()
        except FileNotFoundError:
            print(f"Warning: Generated file not found: {file_path}")
        except Exception as e:
            print(f"Error modifying file {file_path}: {e}")
            raise

    # --- Define your proto files and their import paths ---
    generate_proto_file(
        "timesync/common/proto/time.proto",
        ["timesync/common/proto"],
    )
    generate_proto_file(
        "caller_id/proto/caller_id.proto",
        ["caller_id/proto"],
    )
    generate_proto_file(
        "rpc/proto/common.proto",
        ["caller_id/proto",
         "rpc/proto",
         "timesync/common/proto"],
    )

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_protos(project_root)