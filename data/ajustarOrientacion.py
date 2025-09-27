import cv2
import os

# === CONFIGURA ESTAS RUTAS ===
LABEL_DIR = r"./labels_png"   # carpeta donde están los labels .png
DEPTH_DIR = r"./depths_png16mm"    # carpeta donde están los depth .png

def fix_reflection_inplace(input_dir):
    # Lista de archivos PNG ordenados
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

    for fname in files:
        in_path = os.path.join(input_dir, fname)

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] No se pudo leer {in_path}")
            continue

        # === Anti-reflejo ===
        img_fixed = cv2.flip(img, 1)   # 1 = horizontal | 0 = vertical | -1 = ambos

        # Sobrescribir el archivo original
        cv2.imwrite(in_path, img_fixed)
        print(f"[OK] Corregido y sobrescrito: {in_path}")

def rotate_and_replace(input_dir):
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    for fname in files:
        in_path = os.path.join(input_dir, fname)

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] No se pudo leer {in_path}")
            continue

        # Rotar 270° antihorario (equivale a 90° horario)
        img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Sobrescribir el archivo original
        cv2.imwrite(in_path, img_rot)
        print(f"[OK] Rotado y sobrescrito: {in_path}")
