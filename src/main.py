import os
import sys

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.presentation.gui import run_app
except ModuleNotFoundError:
    from src.presentation.GUI import run_app


def main() -> None:
    run_app(mode="camera")


if __name__ == "__main__":
    main()
