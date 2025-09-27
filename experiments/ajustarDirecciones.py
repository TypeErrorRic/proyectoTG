# make_manifest_relative.py
# Reescribe nyuv2_manifest.csv con rutas RELATIVAS a BASE

import os
import pandas as pd

# === Ajusta la carpeta raíz ===
BASE = r"D:\proyectoTG\data"
CSV_IN  = os.path.join(BASE, "nyuv2_manifest.csv")
CSV_OUT = os.path.join(BASE, "nyuv2_manifest_rel.csv")  # puedes sobrescribir si quieres

# === Subcarpetas esperadas ===
SUBDIRS = {
    "rgb_path": "images",
    "label_path": "labels_png",
    "depth_path": "depths_png16mm"
}

def to_rel(path, kind):
    if pd.isna(path) or not str(path).strip():
        return ""
    fname = os.path.basename(str(path))
    subdir = SUBDIRS.get(kind, "")
    rel = os.path.join(subdir, fname) if subdir else fname
    abs_p = os.path.join(BASE, rel)
    if os.path.exists(abs_p):
        return rel  # ruta relativa correcta
    else:
        # si no existe, dejamos solo el nombre para depuración
        return rel + "  #NOT_FOUND"

def main():
    df = pd.read_csv(CSV_IN)

    if "rgb_path" in df.columns:
        df["rgb_path"] = df["rgb_path"].apply(lambda p: to_rel(p, "rgb_path"))
    if "label_path" in df.columns:
        df["label_path"] = df["label_path"].apply(lambda p: to_rel(p, "label_path"))
    if "depth_path" in df.columns:
        df["depth_path"] = df["depth_path"].apply(lambda p: to_rel(p, "depth_path"))

    # normaliza split a minúsculas
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()

    df.to_csv(CSV_OUT, index=False)
    print(f"[OK] CSV guardado en {CSV_OUT}")
    print(df.head())

if __name__ == "__main__":
    main()
