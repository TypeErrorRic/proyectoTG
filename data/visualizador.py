# main.py — Visor RGB-D + Labels con nombres
# Controles: n/d/→ sig. | p/a/← ant. | o overlay | l leyenda | q/ESC salir

import os, csv, cv2, numpy as np

# === Ruta base ===
BASE  = r"."
ASSOC = os.path.join(BASE, "associations.txt")
CSV_PATH = os.path.join(BASE, "meta", "labels_map.csv")  # creado en el paso A

# --- cargar associations ---
pairs = []
with open(ASSOC, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts_rgb, rgb_file, ts_d, depth_file = line.split()
        pairs.append((ts_rgb, rgb_file, depth_file))
assert pairs, "[ERROR] associations.txt está vacío."

# --- Nombres NYU40 en español (fallback/auto) ---
NYU40_ES = {
    1:"muro",2:"suelo",3:"armario",4:"bada",5:"silla",6:"sofá",7:"mesa",8:"pared",
    9:"escritorio",10:"ventana",11:"libro",12:"estantería",13:"cuadro",
    14:"mostrador",15:"cortina",16:"puerta",17:"lámpara",18:"almohada",
    19:"espejo",20:"alfombra",21:"toalla",22:"palo",23:"cómoda",
    24:"ropa",25:"techo",26:"nevera",27:"televisor",28:"papel",29:"libreta",
    30:"toallero",31:"mesilla",32:"persiana",33:"estante",34:"vidrio",
    35:"cuenco",36:"almacenaje",37:"banqueta",38:"cojín",
    39:"silla oficina",40:"mueble auxiliar"
}

# --- utilidades imagen ---
def read_uint16_png(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.dtype != np.uint16:
        im = im.astype(np.uint16)
    return im

def colorize_labels(lbl, num_colors=4096, seed=42):
    rng = np.random.default_rng(seed)
    lut = np.zeros((num_colors, 3), dtype=np.uint8)
    lut[1:] = rng.integers(64, 255, size=(num_colors-1, 3), dtype=np.uint8)
    lbl = np.clip(lbl.astype(np.int64), 0, num_colors - 1)
    return lut[lbl]

def id_to_bgr(cls_id):
    rgb = colorize_labels(np.array([[cls_id]], dtype=np.int32))[0, 0]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])

def depth_norm_8u(d_mm):
    d = d_mm.astype(np.float32)
    d[d == 0] = np.nan
    if np.any(np.isfinite(d)):
        vmin = np.nanpercentile(d, 2.0)
        vmax = np.nanpercentile(d, 98.0)
        vmax = max(vmax, vmin + 1e-6)
        return (np.clip((d - vmin) / (vmax - vmin), 0, 1) * 255.0).astype(np.uint8)
    return np.zeros_like(d_mm, dtype=np.uint8)

def load_triplet(i):
    ts, rgb_rel, depth_rel = pairs[i]
    rgb_path   = os.path.join(BASE, rgb_rel)
    depth_path = os.path.join(BASE, depth_rel)
    lab_path   = os.path.join(BASE, "labels", f"{ts}.png")

    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    d_mm = read_uint16_png(depth_path)
    lbl  = read_uint16_png(lab_path)
    return ts, rgb_bgr, d_mm, lbl

# --- cargar mapa de nombres (flexible) ---
def load_label_namer(csv_path):
    """Devuelve una función name_for(id) y una cadena 'mode' descriptiva."""
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = [h.strip().lower() for h in (reader.fieldnames or [])]
            rows = list(reader)

        def col(*names):
            for n in names:
                if n in headers:
                    return n
            return None

        # Caso 1: id,name
        id_col = col("id","class_id")
        name_col = col("name","label","class_name")
        if id_col and name_col:
            table = {}
            for r in rows:
                try:
                    table[int(r[id_col])] = r[name_col].strip()
                except: pass
            print(f"[INFO] labels_map.csv: modo 'id,name' ({len(table)} nombres)")
            return lambda cid: table.get(int(cid), None), "direct"

        # Caso 2: fine_id, nyu40_id -> usa NYU40_ES
        fine_col = col("fine_id","raw_id","original_id")
        nyu40_col = col("nyu40_id","coarse_id","mapped_id")
        if fine_col and nyu40_col:
            mapping = {}
            for r in rows:
                try:
                    mapping[int(r[fine_col])] = int(r[nyu40_col])
                except: pass
            print(f"[INFO] labels_map.csv: modo 'fine→NYU40' ({len(mapping)} mapeos)")
            return (lambda cid: NYU40_ES.get(mapping.get(int(cid), -1), None)), "fine2nyu40"

        # Caso 3: nyu40_id, name (por si te pasaron ya los 40)
        nyu40_col2 = col("nyu40_id","id","class_id")
        name_col2  = col("name","label")
        if nyu40_col2 and name_col2:
            nyu40_names = {}
            for r in rows:
                try:
                    nyu40_names[int(r[nyu40_col2])] = r[name_col2].strip()
                except: pass
            print(f"[INFO] labels_map.csv: modo 'nyu40_id,name' ({len(nyu40_names)} nombres)")
            return (lambda cid: nyu40_names.get(int(cid), None)), "nyu40direct"

        print("[WARN] labels_map.csv no tiene columnas reconocidas. Se intentará auto-NYU40.")
    else:
        print(f"[WARN] No existe {csv_path}. Se intentará auto-NYU40.")

    # Fallback: si los ids del primer frame ⊆ 1..40, usa NYU40_ES
    ts0, _, _, lbl0 = load_triplet(0)
    vals = np.unique(lbl0)
    vals = vals[vals!=0]
    if len(vals) and np.max(vals) <= 40:
        print("[INFO] Detectado rango 1..40. Usando NYU40_ES.")
        return lambda cid: NYU40_ES.get(int(cid), None), "auto_nyu40"
    else:
        print("[INFO] No se pudo mapear nombres. Se mostrará 'clase ###'.")
        return lambda cid: None, "none"

NAME_FOR, NAME_MODE = load_label_namer(CSV_PATH)

# --- leyenda (top-K) ---
def make_legend(lbl, top_k=12):
    h, w = lbl.shape[:2]
    vals, counts = np.unique(lbl, return_counts=True)
    mask = vals != 0
    vals, counts = vals[mask], counts[mask]
    if vals.size == 0:
        panel = np.full((120, 320, 3), 30, np.uint8)
        cv2.putText(panel, "Sin etiquetas", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
        return panel
    order = np.argsort(-counts)
    vals, counts = vals[order][:top_k], counts[order][:top_k]
    perc = counts.astype(np.float32) * 100.0 / (h * w)

    row_h, pad, W = 24, 10, 460
    H = pad*2 + row_h*len(vals) + 30
    panel = np.full((H, W, 3), 30, np.uint8)
    cv2.putText(panel, f"Legend - {NAME_MODE}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
    y = pad + 24
    for cid, p in zip(vals, perc):
        color = id_to_bgr(int(cid))
        cv2.rectangle(panel, (10, y-14), (30, y+6), color, -1)
        name = NAME_FOR(int(cid)) or f"clase {int(cid)}"
        text = f"{name}  ({p:.1f}%)"
        cv2.putText(panel, text, (40, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)
        y += row_h
    return panel

# --- estado del visor ---
idx = 0
overlay_on = True
alpha_int = 45   # 0..100
show_legend = True
mouse_xy = None  # (x,y)

# --- ventanas + trackbars ---
cv2.namedWindow("RGB / Overlay", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth (mm)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Legend", cv2.WINDOW_NORMAL)

def on_trackbar_idx(pos):
    global idx
    idx = int(np.clip(pos, 0, len(pairs)-1))
    draw()

def on_trackbar_alpha(pos):
    global alpha_int
    alpha_int = int(np.clip(pos, 0, 100))
    draw()

cv2.createTrackbar("idx",   "RGB / Overlay", idx, len(pairs)-1, on_trackbar_idx)
cv2.createTrackbar("alpha", "RGB / Overlay", alpha_int, 100,    on_trackbar_alpha)

CUR_W = CUR_H = None

def on_mouse(event, x, y, flags, param):
    global mouse_xy
    if event == cv2.EVENT_MOUSEMOVE:
        if CUR_W is not None and CUR_H is not None and 0 <= x < CUR_W and 0 <= y < CUR_H:
            mouse_xy = (x, y)
            draw()
        else:
            mouse_xy = None

cv2.setMouseCallback("RGB / Overlay", on_mouse)

def draw():
    global CUR_W, CUR_H
    ts, rgb_bgr, d_mm, lbl = load_triplet(idx)
    H, W = rgb_bgr.shape[:2]
    CUR_W, CUR_H = W, H

    cv2.resizeWindow("RGB / Overlay", W, H)
    cv2.resizeWindow("Depth (mm)", W, H)

    alpha = alpha_int / 100.0
    d_norm = depth_norm_8u(d_mm)

    if overlay_on:
        lab_rgb = colorize_labels(lbl)
        lab_bgr = cv2.cvtColor(lab_rgb, cv2.COLOR_RGB2BGR)
        vis_rgb = cv2.addWeighted(rgb_bgr, 1 - alpha, lab_bgr, alpha, 0)
        title = f"Overlay ON | idx={idx} ts={ts} | alpha={alpha:.2f} | {NAME_MODE}"
    else:
        vis_rgb = rgb_bgr
        title = f"Overlay OFF | idx={idx} ts={ts} | {NAME_MODE}"

    # HUD (nombre + profundidad)
    if mouse_xy is not None:
        mx, my = mouse_xy
        if 0 <= mx < W and 0 <= my < H:
            cid = int(lbl[my, mx])
            name = NAME_FOR(cid) or f"clase {cid}"
            dmm = int(d_mm[my, mx])
            info = f"{name} | {dmm} mm ({dmm/1000.0:.2f} m)" if dmm > 0 else f"{name} | sin profundidad"
            (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x0 = min(mx + 10, max(0, W - tw - 20))
            y0 = min(my + 10, max(th + 10, H - 10))
            cv2.rectangle(vis_rgb, (x0-6, y0-th-8), (x0+tw+6, y0+6), (0,0,0), -1)
            cv2.putText(vis_rgb, info, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            cv2.drawMarker(vis_rgb, (mx, my), (255,255,255), cv2.MARKER_CROSS, 10, 1)

    # leyenda
    if show_legend:
        legend = make_legend(lbl)
        cv2.imshow("Legend", legend)
    else:
        cv2.imshow("Legend", np.zeros((1,1,3), np.uint8))

    cv2.imshow("RGB / Overlay", vis_rgb)
    cv2.setWindowTitle("RGB / Overlay", title)
    cv2.imshow("Depth (mm)", d_norm)

    cv2.setTrackbarPos("idx", "RGB / Overlay", idx)
    cv2.setTrackbarPos("alpha", "RGB / Overlay", alpha_int)

print("[Controles] n/d/→ sig. | p/a/← ant. | o overlay | l leyenda | q/ESC salir")
draw()

while True:
    key = cv2.waitKey(0) & 0xFF
    if key in (ord('q'), 27): break
    elif key in (ord('n'), ord('d'), 83): idx = (idx + 1) % len(pairs); draw()
    elif key in (ord('p'), ord('a'), 81): idx = (idx - 1) % len(pairs); draw()
    elif key == ord('o'): overlay_on = not overlay_on; draw()
    elif key == ord('l'): show_legend = not show_legend; draw()

cv2.destroyAllWindows()
