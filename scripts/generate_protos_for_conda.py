import os
from pathlib import Path
from generate_protos import generate_protos

if __name__ == "__main__":
    project_root = Path(os.environ["SRC_DIR"]) # Use SRC_DIR
    generate_protos(project_root)