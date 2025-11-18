import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar import AlgoritmosSegmentacion

def main() -> None:
    import cv2
    while True:
        resultado = AlgoritmosSegmentacion()
        cv2.imshow("Segmentación", resultado)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
