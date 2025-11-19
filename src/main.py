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

    # Inicializar variables para cálculo de FPS
    prev_time = cv2.getTickCount()
    fps = 0

    while True:
        resultado = AlgoritmosSegmentacion()
        # Calcular FPS
        curr_time = cv2.getTickCount()
        time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        if resultado is not None:
            # Dibujar FPS en la imagen
            resultado_fps = resultado.copy()
            cv2.putText(resultado_fps, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(window_name, resultado_fps)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

