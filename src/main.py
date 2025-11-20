import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.GUI import run_app


def main() -> None:
    run_app(mode="camera")


if __name__ == "__main__":
    main()
