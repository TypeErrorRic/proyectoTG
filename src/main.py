import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar import AlgoritmosSegmentacion
import cv2


def main() -> None:

    window_name = "Segmentacion"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        resultado = AlgoritmosSegmentacion()
        if resultado is not None:
            cv2.imshow(window_name, resultado)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

