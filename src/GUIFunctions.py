import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab

# Optional import to fetch current configuration parameters at capture time.
try:
    from src.utilities.segment import obtener_parametros_ground
except ModuleNotFoundError:
    try:
        from utilities.segment import obtener_parametros_ground  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - fallback when segment is unavailable
        obtener_parametros_ground = None  # type: ignore

# Optional import to toggle mask overlays from the GUI.
try:
    from src.utilities import helpers as helpers_mod
except ModuleNotFoundError:
    try:
        from utilities import helpers as helpers_mod  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - fallback when helpers is unavailable
        helpers_mod = None  # type: ignore

# Default parameter fallback used when runtime parameters are unavailable.
DEFAULT_CONFIG_FALLBACK: Dict[str, str] = {
    "subsample_stride": "1",
    "dist_thresh": "0.03",
    "max_iters": "400",
    "min_inliers": "400",
    "max_angle_deg": "60.0",
    "max_up_dot": "0.35",
    "score_subset": "4096",
    "early_stop_ratio": "0.92",
    "batch_size": "128",
    "low_height_pct": "25.0",
    "roi_bottom_fraction": "0.34",
    "roi_expand_step": "0.2",
    "max_agg_points": "150000",
    "refine_full_res": "1",
    "refine_max_points": "200000",
    "refine_dist_mult": "1.6",
    "ground_mask_refine": "0",
    "wall_subsample_stride": "2",
    "wall_dist_thresh": "0.03",
    "wall_max_iters": "300",
    "wall_min_inliers": "400",
    "wall_max_angle_deg": "20.0",
    "wall_score_subset": "4096",
    "wall_early_stop_ratio": "0.90",
    "wall_batch_size": "1024",
    "wall_refine_dist_mult": "1.6",
    "wall_mask_refine": "0",
    "ground_perp_deg": "20.0",
    "wall_ortho_deg": "20.0",
    "wall_parallel_deg": "10.0",
    "wall_parallel_distance_m": "0.60",
    "door_hue_tol": "18",
    "door_hsv_enabled": "1",
    "door_min_s": "30",
    "door_min_v": "20",
    "door_glare_s_max": "35",
    "door_glare_v_min": "210",
    "door_glare_v_clip": "200",
    "door_ground_parallel_deg": "15.0",
    "door_plane_inlier_ratio": "0.40",
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
        print(f"[GUI] no se pudieron leer los parámetros actuales: {exc}")
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
        ("subsample_stride", "Submuestreo (stride px)"),
        ("dist_thresh", "Umbral distancia al plano (m)"),
        ("max_iters", "Iteraciones máx. (RANSAC)"),
        ("min_inliers", "Mín. inliers (pts)"),
        ("max_angle_deg", "Ángulo máx. (grados)"),
        ("max_up_dot", "Max up dot (0-1)"),
        ("score_subset", "Subconjunto para puntuar (pts)"),
        ("early_stop_ratio", "Ratio corte temprano (0-1)"),
        ("batch_size", "Tamaño de lote (modelos)"),
        ("low_height_pct", "Percentil bajo de altura (%)"),
        ("roi_bottom_fraction", "Fracción inferior ROI (0-1)"),
        ("refine_full_res", "Refinar full-res"),
        ("refine_dist_mult", "Tolerancia refino (dist_mult)"),
        ("ground_mask_refine", "Mejorar máscara suelo (0/1)"),
        ("wall_subsample_stride", "Submuestreo (stride px)"),
        ("wall_dist_thresh", "Umbral distancia al plano (m)"),
        ("wall_max_iters", "Iteraciones máx. (RANSAC)"),
        ("wall_min_inliers", "Mín. inliers (pts)"),
        ("wall_max_angle_deg", "Ángulo máx. (grados)"),
        ("wall_score_subset", "Subconjunto para puntuar (pts)"),
        ("wall_early_stop_ratio", "Ratio corte temprano (0-1)"),
        ("wall_batch_size", "Tamaño de lote (modelos)"),
        ("wall_refine_dist_mult", "Tolerancia refino (dist_mult)"),
        ("wall_mask_refine", "Mejorar máscara pared (0/1)"),
        ("ground_perp_deg", "Perp. suelo (grados)"),
        ("wall_ortho_deg", "Orto paredes (grados)"),
        ("wall_parallel_deg", "Paralelo paredes (grados)"),
        ("wall_parallel_distance_m", "Dist. paredes (m)"),
        ("door_hue_tol", "Rango color (0-179)"),
        ("door_hsv_enabled", "Filtro HSV puerta (0/1)"),
        ("door_min_s", "Color min (0-255)"),
        ("door_min_v", "Luz min (0-255)"),
        ("door_glare_s_max", "Reflejo color (0-255)"),
        ("door_glare_v_min", "Reflejo luz (0-255)"),
        ("door_glare_v_clip", "Bajar reflejo (0-255)"),
        ("door_ground_parallel_deg", "Inclinación máx. (grados)"),
        ("door_plane_inlier_ratio", "Min puntos en plano (0-1)"),
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
        "max_up_dot": (float, -0.01),
        "score_subset": (int, 0.0),
        "early_stop_ratio": (float, 0.0),
        "batch_size": (int, 0.0),
        "low_height_pct": (float, -1.0),
        "roi_bottom_fraction": (float, 0.0),
        "roi_expand_step": (float, -0.01),
        "max_agg_points": (int, -0.1),
        "refine_max_points": (int, -0.1),
        "refine_dist_mult": (float, 0.99),
        "wall_subsample_stride": (int, 0.0),
        "wall_dist_thresh": (float, 0.0),
        "wall_max_iters": (int, 0.0),
        "wall_min_inliers": (int, 0.0),
        "wall_max_angle_deg": (float, 0.0),
        "wall_score_subset": (int, 0.0),
        "wall_early_stop_ratio": (float, 0.0),
        "wall_batch_size": (int, 0.0),
        "wall_refine_dist_mult": (float, 0.99),
        "ground_perp_deg": (float, 0.0),
        "wall_ortho_deg": (float, 0.0),
        "wall_parallel_deg": (float, 0.0),
        "wall_parallel_distance_m": (float, 0.0),
        "door_hue_tol": (int, -0.1),
        "door_min_s": (int, -0.1),
        "door_min_v": (int, -0.1),
        "door_glare_s_max": (int, -0.1),
        "door_glare_v_min": (int, -0.1),
        "door_glare_v_clip": (int, -0.1),
        "door_ground_parallel_deg": (float, 0.0),
        "door_plane_inlier_ratio": (float, 0.0),
    }
    parsed: Dict[str, Any] = {}
    errors = []

    # Campo especial: permitir texto en campos booleanos.
    for key in ("refine_full_res", "wall_mask_refine", "ground_mask_refine", "door_hsv_enabled"):
        if key not in values:
            continue
        raw_value = str(values.get(key, "")).strip().lower()
        if raw_value in ("true", "t", "si", "yes", "on", "1"):
            parsed[key] = True
        elif raw_value in ("false", "f", "no", "off", "0"):
            parsed[key] = False
        else:
            errors.append(key)

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
        elif key == "wall_early_stop_ratio":
            if not (0.0 < float(val) <= 1.0):
                errors.append(key)
                continue
        elif key == "max_up_dot":
            if not (0.0 <= float(val) <= 1.0):
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
        elif key == "door_hue_tol":
            if not (0 <= int(val) <= 179):
                errors.append(key)
                continue
        elif key in (
            "door_min_s",
            "door_min_v",
            "door_glare_s_max",
            "door_glare_v_min",
            "door_glare_v_clip",
        ):
            if not (0 <= int(val) <= 255):
                errors.append(key)
                continue
        elif key == "door_ground_parallel_deg":
            if not (0.0 <= float(val) <= 90.0):
                errors.append(key)
                continue
        elif key == "door_plane_inlier_ratio":
            if not (0.0 < float(val) <= 1.0):
                errors.append(key)
                continue
        elif float(val) <= min_value:
            errors.append(key)
            continue
        parsed[key] = val

    if errors:
        print(f"[GUI] Parámetros inválidos: {', '.join(errors)}")
        return None

    # Normalize boolean fields and practical limits
    for key in ("refine_full_res", "wall_mask_refine", "ground_mask_refine", "door_hsv_enabled"):
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
    if "wall_refine_dist_mult" in parsed:
        parsed["wall_refine_dist_mult"] = max(1.0, float(parsed["wall_refine_dist_mult"]))
    for key in ("max_agg_points", "refine_max_points"):
        if key in parsed:
            parsed[key] = max(0, int(parsed[key]))
    if "door_hue_tol" in parsed:
        parsed["door_hue_tol"] = max(0, min(179, int(parsed["door_hue_tol"])))
    for key in (
        "door_min_s",
        "door_min_v",
        "door_glare_s_max",
        "door_glare_v_min",
        "door_glare_v_clip",
    ):
        if key in parsed:
            parsed[key] = max(0, min(255, int(parsed[key])))
    return parsed


def capture_panel_screenshot(panel: Optional[tk.Widget], upload_dir: str) -> Optional[Dict[str, Any]]:
    """
    Take a screenshot of a Tk widget, store it, and return the capture plus the active config parameters.
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

        config_params: Dict[str, Any] = {}
        if obtener_parametros_ground:
            try:
                config_params = obtener_parametros_ground()
            except Exception as exc:
                print(f"[GUI] no se pudieron leer parametros de configuracion: {exc}")

        return {
            "image": image,
            "config_params": config_params,
        }
    except Exception as exc:
        print(f"[GUI] no se pudo guardar captura: {exc}")
        return None


def toggle_indicator_label(
    label: Optional[tk.Label],
    on_color: str = "#ffffff",
    off_color: str = "#000000",
) -> None:
    """
    Alternate the label foreground between two colors.
    """
    if label is None:
        return
    current = str(label.cget("fg")).lower()
    if current == off_color.lower():
        label.configure(fg=on_color)
    else:
        label.configure(fg=off_color)


def set_indicator_label_state(
    label: Optional[tk.Label],
    enabled: Optional[bool],
    on_color: str = "#ffffff",
    off_color: str = "#000000",
) -> None:
    """
    Set the label color based on a boolean state.
    """
    if label is None or enabled is None:
        return
    label.configure(fg=on_color if enabled else off_color)


def toggle_mask_flag(name: str) -> Optional[bool]:
    """
    Toggle a mask visibility flag in helpers and return the new state.
    """
    if helpers_mod is None:
        print("[GUI] helpers no disponibles para alternar máscaras.")
        return None
    try:
        return helpers_mod.toggle_mask_visibility(name)
    except Exception as exc:
        print(f"[GUI] no se pudo alternar máscara {name}: {exc}")
        return None


def on_indicator_floor(app: Any, label: Optional[tk.Label] = None) -> None:
    """
    Handler for the "Suelo" indicator button.
    """
    state = toggle_mask_flag("ground")
    if state is None:
        toggle_indicator_label(label)
    else:
        set_indicator_label_state(label, state)
    print("[GUI] indicador Suelo presionado.")


def on_indicator_wall(app: Any, label: Optional[tk.Label] = None) -> None:
    """
    Handler for the "Muro" indicator button.
    """
    state = toggle_mask_flag("wall")
    if state is None:
        toggle_indicator_label(label)
    else:
        set_indicator_label_state(label, state)
    print("[GUI] indicador Muro presionado.")


def on_indicator_door(app: Any, label: Optional[tk.Label] = None) -> None:
    """
    Handler for the "Puerta" indicator button.
    """
    if app is None:
        return
    state = toggle_mask_flag("door")
    if state is None:
        toggle_indicator_label(label)
    else:
        set_indicator_label_state(label, state)
    print("[GUI] indicador Puerta presionado.")
