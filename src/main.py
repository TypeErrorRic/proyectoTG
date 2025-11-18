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