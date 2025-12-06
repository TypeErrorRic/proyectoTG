import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab

# Default parameter fallback used when runtime parameters are unavailable.
DEFAULT_CONFIG_FALLBACK: Dict[str, str] = {
    "subsample_stride": "1",
    "dist_thresh": "0.03",
    "max_iters": "400",
    "min_inliers": "400",
    "max_angle_deg": "60.0",
    "score_subset": "4096",
    "time_budget_ms": "120",
    "early_stop_ratio": "0.92",
    "batch_size": "128",
    "low_height_pct": "25.0",
    "roi_bottom_fraction": "0.34",
    "roi_expand_step": "0.2",
    "max_agg_points": "150000",
    "refine_full_res": "1",
    "refine_max_points": "200000",
    "refine_dist_mult": "1.6",
    "second_pass_mask": "1",
}


def ensure_upload_dir(upload_dir: str) -> None:
    """
    Create the uploads folder if it does not exist.
    """
    os.makedirs(upload_dir, exist_ok=True)


def load_upload_images(upload_dir: str, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")) -> List[str]:
    """
    Return the list of capture file paths inside the uploads folder, newest first.
    """
    ensure_upload_dir(upload_dir)
    collected: List[Tuple[float, str]] = []
    for name in os.listdir(upload_dir):
        if not name.lower().endswith(extensions):
            continue
        path = os.path.join(upload_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        collected.append((mtime, path))
    collected.sort(key=lambda item: item[0], reverse=True)
    return [path for _mtime, path in collected]


def visualize_capture(app: Any) -> None:
    """
    Bridge to show the requested capture using the gallery helpers on the app.
    """
    if app is None:
        return
    try:
        requested_idx = app._get_requested_capture_index()
    except Exception as exc:
        print(f"[GUI] no se pudo leer indice de captura: {exc}")
        return
    try:
        app._show_gallery_panel()
        app._refresh_gallery_images()
        app._show_gallery_image(requested_idx)
    except Exception as exc:
        print(f"[GUI] no se pudo visualizar captura: {exc}")


def make_icon(kind: str) -> Image.Image:
    """
    Generate a placeholder icon when the expected asset is missing.
    """
    size = 64
    accent = "#f2f2f2"
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if kind == "gear":
        draw.ellipse((12, 12, 52, 52), outline=accent, width=4)
        draw.rectangle((30, 6, 34, 58), fill=accent)
        draw.rectangle((6, 30, 58, 34), fill=accent)
        draw.ellipse((22, 22, 42, 42), fill="#5a5a5a", outline=accent, width=3)
    else:
        draw.rounded_rectangle((10, 18, 54, 46), radius=8, outline=accent, width=4)
        draw.rectangle((38, 10, 52, 20), fill=accent)
        draw.ellipse((26, 22, 40, 36), outline=accent, width=3)
    return img


def load_sidebar_icons(base_path: str, assets: Dict[str, str]) -> Dict[str, Image.Image]:
    """
    Load sidebar icons; fall back to placeholder icons if a file is missing.
    """
    icons: Dict[str, Image.Image] = {}
    for key, filename in assets.items():
        path = os.path.join(base_path, filename)
        try:
            img = Image.open(path).convert("RGBA")
            icons[key] = img
        except Exception as exc:
            print(f"[GUI] no se pudo cargar icono {filename}: {exc}")
            fallback_kind = "gear" if key == "config" else "camera"
            icons[key] = make_icon(fallback_kind)
    return icons


def init_config_defaults(
    runtime_params_loader: Callable[[], Dict[str, Any]],
    fallback: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Load default/last-used parameters from the segmentation module.
    """
    defaults = fallback or DEFAULT_CONFIG_FALLBACK
    try:
        runtime_params = runtime_params_loader()
    except Exception as exc:
        print(f"[GUI] no se pudieron leer los parametros actuales: {exc}")
        runtime_params = {}

    config_defaults: Dict[str, str] = {}
    for key, default in defaults.items():
        value = runtime_params.get(key, default)
        if isinstance(value, bool):
            value = int(value)
        config_defaults[key] = str(value)
    return config_defaults


def param_summary_fields() -> List[Tuple[str, str]]:
    """
    Keys and labels to show in the execution summary panel.
    """
    return [
        ("subsample_stride", "Submuestreo"),
        ("dist_thresh", "Umbral distancia"),
        ("max_iters", "Iteraciones max"),
        ("min_inliers", "Min inliers"),
        ("max_angle_deg", "Angulo max"),
        ("score_subset", "Score subset"),
        ("time_budget_ms", "Tiempo ms"),
        ("early_stop_ratio", "Corte temprano"),
        ("batch_size", "Batch size"),
        ("low_height_pct", "Percentil bajo"),
        ("roi_bottom_fraction", "ROI inferior"),
        ("refine_full_res", "Refino full-res"),
        ("refine_dist_mult", "Tol. refino"),
    ]


def validate_numeric_entry(proposed: str) -> bool:
    """
    Allow empty string or values that parse as float.
    """
    if proposed == "":
        return True
    try:
        float(proposed)
        return True
    except Exception:
        return False


def parse_config_params(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert UI strings into typed parameters; returns None on error.
    """
    specs = {
        "subsample_stride": (int, 0.0),
        "dist_thresh": (float, 0.0),
        "max_iters": (int, 0.0),
        "min_inliers": (int, 0.0),
        "max_angle_deg": (float, 0.0),
        "score_subset": (int, 0.0),
        "time_budget_ms": (float, 0.0),
        "early_stop_ratio": (float, 0.0),
        "batch_size": (int, 0.0),
        "low_height_pct": (float, -1.0),
        "roi_bottom_fraction": (float, 0.0),
        "roi_expand_step": (float, -0.01),
        "max_agg_points": (int, -0.1),
        "refine_max_points": (int, -0.1),
        "refine_dist_mult": (float, 0.99),
        "second_pass_mask": (int, -1.0),
    }
    parsed: Dict[str, Any] = {}
    errors = []

    # Campo especial: permitir texto en refine_full_res
    if "refine_full_res" in values:
        raw_refine = str(values.get("refine_full_res", "")).strip().lower()
        if raw_refine in ("true", "t", "si", "yes", "on", "1"):
            parsed["refine_full_res"] = True
        elif raw_refine in ("false", "f", "no", "off", "0"):
            parsed["refine_full_res"] = False
        else:
            errors.append("refine_full_res")

    for key, (caster, min_value) in specs.items():
        if key not in values:
            continue
        raw = values.get(key, "")
        try:
            val = caster(str(raw).strip())
        except Exception:
            errors.append(key)
            continue

        if key == "early_stop_ratio":
            if not (0.0 < float(val) <= 1.0):
                errors.append(key)
                continue
        elif key == "low_height_pct":
            if not (0.0 <= float(val) <= 100.0):
                errors.append(key)
                continue
        elif key == "roi_bottom_fraction":
            if not (0.0 < float(val) <= 1.0):
                errors.append(key)
                continue
        elif float(val) <= min_value:
            errors.append(key)
            continue
        parsed[key] = val

    if errors:
        print(f"[GUI] Parametros invalidos: {', '.join(errors)}")
        return None

    # Normalize boolean fields and practical limits
    for key in ("refine_full_res", "second_pass_mask"):
        if key in parsed:
            parsed[key] = bool(int(parsed[key]))
    if "roi_expand_step" in parsed:
        parsed["roi_expand_step"] = max(0.0, float(parsed["roi_expand_step"]))
    if "roi_bottom_fraction" in parsed:
        parsed["roi_bottom_fraction"] = max(0.01, min(1.0, float(parsed["roi_bottom_fraction"])))
    if "low_height_pct" in parsed:
        parsed["low_height_pct"] = max(0.0, min(100.0, float(parsed["low_height_pct"])))
    if "refine_dist_mult" in parsed:
        parsed["refine_dist_mult"] = max(1.0, float(parsed["refine_dist_mult"]))
    for key in ("max_agg_points", "refine_max_points"):
        if key in parsed:
            parsed[key] = max(0, int(parsed[key]))
    return parsed


def capture_panel_screenshot(panel: Optional[tk.Widget], upload_dir: str) -> Optional[str]:
    """
    Take a screenshot of a Tk widget and store it in the uploads folder.
    """
    if panel is None:
        print("[GUI] panel de video no disponible para captura.")
        return None

    try:
        panel.update_idletasks()
        x, y = panel.winfo_rootx(), panel.winfo_rooty()
        w, h = panel.winfo_width(), panel.winfo_height()
        if w <= 1 or h <= 1:
            print("[GUI] dimensiones invalidas para captura.")
            return None

        ensure_upload_dir(upload_dir)
        bbox = (x, y, x + w, y + h)
        image = ImageGrab.grab(bbox=bbox)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(upload_dir, f"captura_{timestamp}.png")
        image.save(filepath)
        print(f"[GUI] captura guardada en {filepath}")
        return filepath
    except Exception as exc:
        print(f"[GUI] no se pudo guardar captura: {exc}")
        return None
