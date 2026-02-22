r"""
\brief Interface with a sidebar, video display, parameter controls, database
header, and logo panel.
\details Execution mode shows segmented frames with buttons to switch between
test and streaming; a background thread handles capture at ~20 fps.
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any, Callable, List

import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

try:
    # Preferred import when running as a package/module.
    from src.GUIFunctions import (
        capture_panel_screenshot,
        ensure_upload_dir,
        init_config_defaults,
        load_sidebar_icons,
        load_upload_images,
        on_indicator_door,
        on_indicator_floor,
        on_indicator_wall,
        param_summary_fields,
        parse_config_params,
        validate_numeric_entry,
        visualize_capture,
    )
except ModuleNotFoundError:
    # Fallback for direct script execution from the src directory.
    from GUIFunctions import (
        capture_panel_screenshot,
        ensure_upload_dir,
        init_config_defaults,
        load_sidebar_icons,
        load_upload_images,
        on_indicator_door,
        on_indicator_floor,
        on_indicator_wall,
        param_summary_fields,
        parse_config_params,
        validate_numeric_entry,
        visualize_capture,
    )

# @note Adjust sys.path when executed as a script.
if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar import (
    AlgoritmosSegmentacion,
    actualizar_parametros_ground,
    liberar_recursos,
    obtener_parametros_ground,
    obtener_metricas,
)
from src.models.doorDetection import is_model_loading

# @note Limit display size to reduce rescale cost (match camera feed 640x480).
DISPLAY_MAX_W = 640
DISPLAY_MAX_H = 480
# @note FPS limit for running AlgoritmosSegmentacion (20 fps max).
TARGET_FRAME_TIME = 1.0 / 20.0
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")


class SegmentacionApp:
    r"""
    \brief Main window wiring panels with the capture/segmentation logic.
    \details Connects the sidebar, video view, parameter panel, database
    controls, and logo panel with the project's capture/segmentation logic.
    """

    def __init__(self, root: tk.Tk, mode: str = "prueba") -> None:
        self.root = root
        # @note Mode can be "camera" or "prueba".
        self.mode = mode
        self.active_page = "configuracion"
        self.running = True

        self.prev_time = time.perf_counter()
        self.fps: float = 0.0
        self.last_frame: Optional = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_frame_ts = 0.0
        self._stream_requested = False
        self._loading_msg_variants: List[str] = [
            "Modelos cargando...",
            "Modelos cargando..",
            "Modelos cargando.",
        ]
        self._loading_msg_idx: int = 0
        self._loading_msg_last_ts: float = time.perf_counter()

        self.photo_ref: Optional[ImageTk.PhotoImage] = None
        self.logo_image: Optional[ImageTk.PhotoImage] = None
        self.sidebar_icons_raw: Dict[str, Image.Image] = {}
        self.sidebar_icons: Dict[str, ImageTk.PhotoImage] = {}
        self._indicator_images: List[ImageTk.PhotoImage] = []
        self.config_vars: Dict[str, tk.StringVar] = {}
        self.config_defaults: Dict[str, str] = {}
        self._config_apply_btn: Optional[tk.Button] = None
        self._apply_btn_default_text: str = "Aplicar"
        self._apply_btn_default_bg: str = "#00b86b"
        self._apply_btn_default_activebg: str = "#21d087"
        self._apply_status_after_id: Optional[str] = None
        self.params_summary_labels: Dict[str, tk.Label] = {}
        self.params_summary_name_labels: List[tk.Label] = []
        self.sample_controls: Dict[str, Any] = {}
        self.dataset_index: int = 0
        self.dataset_index_var: Optional[tk.IntVar] = None
        self._updating_dataset_controls: bool = False
        self._updating_capture_controls: bool = False
        self.capture_controls: Dict[str, Any] = {}
        self._capturas_default_bg: Optional[str] = None
        self._capturas_default_activebg: Optional[str] = None
        self.btn_capturar: Optional[tk.Button] = None
        self.gallery_paths: List[str] = []
        self.gallery_index: int = 0
        self.gallery_photo_ref: Optional[ImageTk.PhotoImage] = None
        self.gallery_frame: Optional[tk.Frame] = None
        self.gallery_display: Optional[tk.Label] = None
        self.gallery_counter: Optional[tk.Label] = None
        self.gallery_prev_btn: Optional[tk.Button] = None
        self.gallery_next_btn: Optional[tk.Button] = None
        self._gallery_active: bool = False

        self.config_defaults = init_config_defaults(runtime_params_loader=obtener_parametros_ground)
        ensure_upload_dir(UPLOAD_DIR)
        self._configure_window()
        self._build_grid()
        assets = {"config": "analitica.png", "exec": "camara.png"}
        base_path = os.path.join(os.path.dirname(__file__), "images")
        self.sidebar_icons_raw = load_sidebar_icons(base_path, assets)
        self._build_panels()
        self.root.after(30, self._update_sidebar_icons)
        self.root.bind("<Configure>", self._on_resize)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_mode("config")
        # Do not auto-iniciar transmisión; arranca solo al presionar el botón.
        self._set_mode(self.mode, update_header=False)
        self._heartbeat()

    def _configure_window(self) -> None:
        r"""
        \brief Sets up the base window properties (title, size, and style flags).
        """
        self.root.title("Segmentacion")
        self.root.geometry("1250x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#2f2f2f")
        try:
            self.root.attributes("-fullscreen", False)
        except Exception:
            pass

    def _update_sidebar_icons(self) -> None:
        """
        \brief Resizes icons so they match the actual size of the buttons.
        """
        buttons = {"config": getattr(self, "btn_config", None), "exec": getattr(self, "btn_exec", None)}
        if not all(btn and btn.winfo_width() > 1 and btn.winfo_height() > 1 for btn in buttons.values()):
            self.root.after(30, self._update_sidebar_icons)
            return

        for key, btn in buttons.items():
            raw = self.sidebar_icons_raw.get(key)
            if raw is None:
                continue
            width, height = btn.winfo_width(), btn.winfo_height()
            margin = 8
            target_w = max(width - margin, 1)
            target_h = max(height - margin, 1)
            scale = min(target_w / max(raw.width, 1), target_h / max(raw.height, 1))
            resized = raw.resize((max(int(raw.width * scale), 1), max(int(raw.height * scale), 1)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            self.sidebar_icons[key] = photo
            btn.configure(image=photo, text="")

    def _build_grid(self) -> None:
        r"""
        \brief Configures the window grid for layout.
        \details Columns: 40 | 640 | 250 | 240 -> 1170 (approx 1170 with
        borders). Rows: 6 x 100 -> 600; weights keep proportions on resize.
        """
        self.root.grid_columnconfigure(0, minsize=40, weight=0)
        self.root.grid_columnconfigure(1, minsize=640, weight=3)
        self.root.grid_columnconfigure(2, minsize=250, weight=1)
        self.root.grid_columnconfigure(3, minsize=240, weight=1)
        for r in range(6):
            self.root.grid_rowconfigure(r, minsize=100, weight=1)

    def _build_panels(self) -> None:
        """
        \brief Creates and arranges the main UI panels.
        """
        # @note Sidebar panel.
        frame_sidebar = tk.Frame(self.root, bg="#333333")
        frame_sidebar.grid(row=0, column=0, rowspan=6, sticky="nsew")
        frame_sidebar.grid_propagate(False)
        self._build_sidebar(frame_sidebar)

        # @note Video panel.
        frame_video = tk.Frame(self.root, bg="#999999")
        frame_video.grid(row=0, column=1, rowspan=6, sticky="nsew")
        self._build_display_area(frame_video)

        # @note Parameter panel uses the full column (includes mode buttons).
        frame_params = tk.Frame(self.root, bg="#999999")
        frame_params.grid(row=0, column=2, rowspan=6, sticky="nsew")
        self._build_exec_controls(frame_params)

        # @note Database panel (rows 0-3) -> column 3.
        frame_db = tk.Frame(self.root, bg="#999999")
        frame_db.grid(row=0, column=3, rowspan=2, sticky="nsew")
        self._build_db_panel(frame_db)

        # @note Database panel (rows 0-3) -> column 3.
        frame_db = tk.Frame(self.root, bg="#999999")
        frame_db.grid(row=2, column=3, rowspan=2, sticky="nsew")
        self._build_captura(frame_db)


        # @note Logo panel (rows 4-5, column 3).
        frame_logo = tk.Frame(self.root, bg="#999999")
        frame_logo.grid(row=4, column=3, rowspan=2, sticky="nsew")
        self._build_logo(frame_logo)

        # @note Full-screen configuration panel (except sidebar).
        frame_config = tk.Frame(self.root, bg="#cccccc")
        frame_config.grid(row=0, column=1, rowspan=6, columnspan=3, sticky="nsew")
        self._build_config_placeholder(frame_config)
        frame_config.grid_remove()

        # @note References to visible frames.
        self.frames = {
            "sidebar": frame_sidebar,
            "video": frame_video,
            "params": frame_params,
            "db": frame_db,
            "logo": frame_logo,
            "config": frame_config,
        }
        self._adjust_sidebar_rows()

    def _build_sidebar(self, container: tk.Frame) -> None:
        """
        \brief Builds the sidebar with configuration and execution buttons.
        """
        container.grid_propagate(False)
        for r in range(6):
            container.rowconfigure(r, weight=1, uniform="sidebar_rows")
        container.columnconfigure(0, weight=1)

        top_wrapper = tk.Frame(container, bg="#333333")
        top_wrapper.grid(row=0, column=0, rowspan=3, sticky="nsew")
        bottom_wrapper = tk.Frame(container, bg="#333333")
        bottom_wrapper.grid(row=3, column=0, rowspan=3, sticky="nsew")

        for wrapper in (top_wrapper, bottom_wrapper):
            wrapper.grid_propagate(False)
            wrapper.columnconfigure(0, weight=1)
            wrapper.rowconfigure(0, weight=1)

        config_icon = self.sidebar_icons.get("config")
        self.btn_config = tk.Button(
            top_wrapper,
            image=config_icon,
            text="" if config_icon else "Config",
            bg="#4a4a4a",
            fg="white",
            bd=0,
            width=1,
            height=1,
            compound="center",
            relief=tk.FLAT,
            overrelief=tk.FLAT,
            activebackground="#4a4a4a",
            activeforeground="white",
            highlightthickness=0,
            takefocus=0,
            command=lambda: self._show_mode("config"),
        )
        self.btn_config.pack(fill="both", expand=True)

        exec_icon = self.sidebar_icons.get("exec")
        self.btn_exec = tk.Button(
            bottom_wrapper,
            image=exec_icon,
            text="" if exec_icon else "Ejecucion",
            bg="#5c5c5c",
            fg="white",
            bd=0,
            width=1,
            height=1,
            compound="center",
            relief=tk.FLAT,
            overrelief=tk.FLAT,
            activebackground="#5c5c5c",
            activeforeground="white",
            highlightthickness=0,
            takefocus=0,
            command=lambda: self._show_mode("exec"),
        )
        self.btn_exec.pack(fill="both", expand=True)

    def _build_display_area(self, container: tk.Frame) -> None:
        """
        \brief Builds the video display area and legend.
        """
        frame_video_inner = tk.Frame(container, bg="#7f7f7f", width=DISPLAY_MAX_W, height=DISPLAY_MAX_H, bd=1, relief=tk.SOLID)
        frame_video_inner.pack(side="top", pady=8, padx=8)
        frame_video_inner.pack_propagate(False)

        self.display_area = tk.Label(
            frame_video_inner,
            text="Esperando imagen...",
            bg="#7f7f7f",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief=tk.FLAT,
            anchor="center",
            compound="top",
            padx=10,
            pady=10,
        )
        self.display_area.pack(fill="both", expand=True)

        # Gallery frame (hidden initially) used to browse saved captures.
        self.gallery_frame = tk.Frame(
            container,
            bg="#7f7f7f",
            width=DISPLAY_MAX_W,
            height=DISPLAY_MAX_H,
            bd=1,
            relief=tk.SOLID,
        )
        self.gallery_frame.pack_propagate(False)
        self.gallery_display = tk.Label(
            self.gallery_frame,
            text="Sin capturas para mostrar.",
            bg="#7f7f7f",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief=tk.FLAT,
            anchor="center",
            compound="top",
            padx=10,
            pady=10,
        )
        self.gallery_display.pack(fill="both", expand=True)

        gallery_nav = tk.Frame(self.gallery_frame, bg="#7f7f7f")
        gallery_nav.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        self.gallery_prev_btn = tk.Button(
            gallery_nav,
            text="Anterior",
            bg="#5c5c5c",
            fg="white",
            activebackground="#4a4a4a",
            activeforeground="white",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=lambda: self._show_gallery_image(self.gallery_index - 1),
        )
        self.gallery_prev_btn.pack(side="left", expand=True, fill="x", padx=(0, 8))

        self.gallery_counter = tk.Label(
            gallery_nav,
            text="0/0",
            bg="#7f7f7f",
            fg="white",
            font=("Segoe UI", 9, "bold"),
        )
        self.gallery_counter.pack(side="left", padx=4)

        self.gallery_next_btn = tk.Button(
            gallery_nav,
            text="Siguiente",
            bg="#5c5c5c",
            fg="white",
            activebackground="#4a4a4a",
            activeforeground="white",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=lambda: self._show_gallery_image(self.gallery_index + 1),
        )
        self.gallery_next_btn.pack(side="left", expand=True, fill="x", padx=(8, 0))
        for btn in (self.gallery_prev_btn, self.gallery_next_btn):
            btn.configure(state=tk.DISABLED)

        mode_panel = tk.Frame(
            container,
            bg="#7f7f7f",
            bd=1,
            relief=tk.SOLID,
            width=DISPLAY_MAX_W,
            height=80,
            highlightthickness=0,
        )
        mode_panel.pack(side="top", pady=8, padx=8)
        mode_panel.pack_propagate(False)
        self.mode_panel = mode_panel

        # Buttons stack on the right: two stacked and one spanning their combined height.
        buttons_panel = tk.Frame(mode_panel, bg="#7f7f7f")
        buttons_panel.pack(side="right", padx=(6, 8), pady=8, fill="y")
        buttons_panel.grid_rowconfigure(0, weight=1)
        buttons_panel.grid_rowconfigure(1, weight=1)
        buttons_panel.grid_columnconfigure(0, weight=1)
        buttons_panel.grid_columnconfigure(1, weight=1)

        self.btn_start_stream = tk.Button(
            buttons_panel,
            text="Iniciar Transmisión",
            bg="#00b86b",
            fg="white",
            activebackground="#5c5c5c",
            activeforeground="white",
            bd=0,
            padx=8,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            command=self._start_stream,
        )
        self.btn_start_stream.grid(row=0, column=0, sticky="nsew", padx=4, pady=(0, 4))

        self.btn_stop_stream = tk.Button(
            buttons_panel,
            text="Detener Transmisión",
            bg="#e53935",
            fg="white",
            activebackground="#5c5c5c",
            activeforeground="white",
            bd=0,
            padx=8,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            command=self._stop_stream,
        )
        self.btn_stop_stream.grid(row=1, column=0, sticky="nsew", padx=4, pady=(4, 0))

        self.btn_capturar = tk.Button(
            buttons_panel,
            text="Capturar",
            bg="#f2c200",
            fg="#1f1f1f",
            activebackground="#ffd54f",
            activeforeground="#1f1f1f",
            bd=0,
            padx=10,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            command=self._capture_screenshot,
        )
        self.btn_capturar.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=4, pady=0)
        self.frame_video_inner = frame_video_inner

        # Content to the left of the buttons.
        mode_content = tk.Frame(mode_panel, bg="#7f7f7f")
        mode_content.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=(4, 6))

        self.mode_label_text = tk.StringVar(
            value="Modo de ejecucion: Camara RGB-D" if self.mode == "camera" else "Modo de ejecucion: Dataset de pruebas"
        )
        mode_label = tk.Label(
            mode_content,
            textvariable=self.mode_label_text,
            bg="#7f7f7f",
            fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        mode_label.pack(side="top", pady=(0, 2), anchor="w")

        indicators = tk.Frame(mode_content, bg="#7f7f7f")
        indicators.pack(side="top", pady=(0, 6), anchor="w")
        for color, text, handler in (
            ("#00b86b", "Suelo", on_indicator_floor),
            ("#1e88e5", "Muro", on_indicator_wall),
            ("#e53935", "Puerta", on_indicator_door),
        ):
            item = tk.Frame(indicators, bg="#7f7f7f")
            item.pack(side="left", padx=8)
            circle_img = self._create_indicator_image(color)
            self._indicator_images.append(circle_img)
            dot_btn = tk.Button(
                item,
                image=circle_img,
                bg="#7f7f7f",
                activebackground="#7f7f7f",
                bd=0,
                relief=tk.FLAT,
                overrelief=tk.FLAT,
                highlightthickness=0,
                takefocus=0,
                cursor="hand2",
            )
            dot_btn.pack(side="left")
            lbl = tk.Label(item, text=text, bg="#7f7f7f", fg="white", font=("Segoe UI", 11, "bold"))
            lbl.pack(side="left", padx=6)
            dot_btn.configure(command=lambda h=handler, l=lbl: h(self, l))

    def _create_indicator_image(self, color: str, size: int = 32, padding: int = 4) -> ImageTk.PhotoImage:
        """
        Create a circular image for indicator buttons.
        """
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (padding, padding, size - padding - 1, size - padding - 1),
            fill=color,
            outline=color,
        )
        return ImageTk.PhotoImage(image)

    def _show_gallery_panel(self) -> None:
        """
        Replace the live video panel with the saved captures viewer.
        """
        self._gallery_active = True
        if self.frame_video_inner:
            try:
                self.frame_video_inner.pack_forget()
            except Exception:
                pass
        if self.gallery_frame:
            kwargs = {"side": "top", "pady": 8, "padx": 8}
            if getattr(self, "mode_panel", None) is not None:
                kwargs["before"] = self.mode_panel
            try:
                self.gallery_frame.pack(**kwargs)
            except Exception:
                self.gallery_frame.pack(side="top", pady=8, padx=8)
        self._refresh_gallery_images()

    def _show_video_panel(self) -> None:
        """
        Restore the live video panel and hide the captures viewer.
        """
        self._gallery_active = False
        if self.gallery_frame:
            try:
                self.gallery_frame.pack_forget()
            except Exception:
                pass
        if self.frame_video_inner:
            try:
                self.frame_video_inner.pack_forget()
                kwargs = {"side": "top", "pady": 8, "padx": 8}
                if getattr(self, "mode_panel", None) is not None:
                    kwargs["before"] = self.mode_panel
                self.frame_video_inner.pack(**kwargs)
            except Exception:
                self.frame_video_inner.pack(side="top", pady=8, padx=8)

    def _update_gallery_nav_state(self) -> None:
        """
        Keep navigation buttons in sync with gallery bounds.
        """
        total = len(self.gallery_paths)
        at_start = self.gallery_index <= 0
        at_end = self.gallery_index >= max(total - 1, 0)
        prev_state = tk.NORMAL if total > 1 and not at_start else tk.DISABLED
        next_state = tk.NORMAL if total > 1 and not at_end else tk.DISABLED
        if self.gallery_prev_btn:
            self.gallery_prev_btn.configure(state=prev_state)
        if self.gallery_next_btn:
            self.gallery_next_btn.configure(state=next_state)
        if self.gallery_counter:
            self.gallery_counter.configure(text=f"{total and (self.gallery_index + 1) or 0}/{total}")
        self._set_capture_index_var(self.gallery_index + 1)

    def _refresh_gallery_images(self) -> None:
        """
        Load the list of captures from the uploads folder.
        """
        try:
            self.gallery_paths = load_upload_images(UPLOAD_DIR)
        except Exception as exc:
            print(f"[GUI] no se pudieron listar capturas: {exc}")
            self.gallery_paths = []
        self.gallery_index = 0
        self._sync_capture_slider_limits()
        self._show_gallery_image(self.gallery_index)

    def _show_gallery_image(self, index: int) -> None:
        """
        Display the requested capture image in the gallery frame.
        """
        if not self.gallery_display:
            return
        if not self.gallery_paths:
            self.gallery_photo_ref = None
            self.gallery_display.configure(
                text="No hay capturas en uploads.",
                image="",
                bg="#7f7f7f",
            )
            self._update_gallery_nav_state()
            return

        clamped = max(0, min(index, len(self.gallery_paths) - 1))
        self.gallery_index = clamped
        path = self.gallery_paths[clamped]
        try:
            img = Image.open(path)
            w, h = img.size
            scale = min(DISPLAY_MAX_W / max(w, 1), DISPLAY_MAX_H / max(h, 1), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.gallery_photo_ref = ImageTk.PhotoImage(img)
            self.gallery_display.configure(
                image=self.gallery_photo_ref,
                text=os.path.basename(path),
                bg="#7f7f7f",
                fg="white",
                compound=tk.BOTTOM,
            )
        except Exception as exc:
            self.gallery_photo_ref = None
            self.gallery_display.configure(
                image="",
                text=f"No se pudo cargar:\n{os.path.basename(path)}\n{exc}",
                bg="#7f7f7f",
                fg="white",
                compound=tk.TOP,
            )
        self._update_gallery_nav_state()
        self._set_capture_index_var(self.gallery_index + 1)

    def _set_capture_index_var(self, value: int) -> None:
        """
        Update capture index entry/slider without triggering callbacks.
        """
        if not hasattr(self, "capture_index_var") or self.capture_index_var is None:
            return
        self._updating_capture_controls = True
        try:
            self.capture_index_var.set(max(1, int(value)))
        finally:
            self._updating_capture_controls = False

    def _sync_capture_slider_limits(self) -> None:
        """
        Adjust capture slider limits according to available images.
        """
        slider = self.capture_controls.get("slider") if hasattr(self, "capture_controls") else None
        total = max(len(self.gallery_paths), 1)
        if not slider or not hasattr(slider, "configure"):
            return
        try:
            slider.configure(to=total, from_=1)
        except Exception:
            pass

    def _get_requested_capture_index(self) -> int:
        """
        Read the desired capture index (1-based) from the control.
        """
        try:
            val = int(self.capture_index_var.get()) if hasattr(self, "capture_index_var") else 1
        except Exception:
            val = 1
        return max(0, val - 1)

    def _on_capture_entry_apply(self) -> None:
        """
        Handle Enter key on the capture entry.
        """
        self._on_visualizar_capture()

    def _on_capture_slider(self, value: str) -> None:
        """
        Sync slider movement with gallery display.
        """
        if self._updating_capture_controls:
            return
        try:
            idx = int(float(value)) - 1
        except Exception:
            return
        self._show_gallery_image(idx)

    def _on_visualizar_capture(self) -> None:
        """
        Visualize the capture corresponding to the entered index.
        """
        visualize_capture(self)
    def _build_exec_controls(self, container: tk.Frame) -> None:
        """
        \brief Builds execution-mode controls and parameter placeholder panel.
        """
        # @note Prevent children from resizing the parameters panel.
        container.pack_propagate(False)
        container_bg = container.cget("bg")
        row_holder = tk.Frame(container, bg=container_bg)
        row_holder.pack(fill="x", padx=4, pady=8)

        self.btn_mode_test = tk.Button(
            row_holder,
            text="Prueba",
            bg="#e53935",
            fg="white",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            command=self._on_mode_prueba_pressed,
        )
        self.btn_mode_test.pack(side="left", padx=4, expand=True, fill="x")

        self.btn_mode_cam = tk.Button(
            row_holder,
            text="Transmision",
            bg="#e53935",
            fg="white",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            command=self._on_mode_transmision_pressed,
        )
        self.btn_mode_cam.pack(side="left", padx=4, expand=True, fill="x")

        params_panel = tk.Frame(
            container,
            bg="#b3b3b3",
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        params_panel.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        params_panel.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        params_panel.columnconfigure(0, weight=1)
        params_panel.rowconfigure(1, weight=1)

        params_title = tk.Label(
            params_panel,
            text="Panel de Parámetros",
            bg="#b3b3b3",
            fg="#1f1f1f",
            font=("Segoe UI", 11, "bold"),
            anchor="center",
            padx=8,
        )
        params_title.grid(row=0, column=0, sticky="ew", pady=(6, 3))

        params_body = tk.Frame(
            params_panel,
            bg="#f2f2f2",
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        params_body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        params_body.columnconfigure(0, weight=1)
        params_body.rowconfigure(1, weight=1)

        summary_header = tk.Frame(params_body, bg=params_body.cget("bg"))
        summary_header.grid(row=0, column=0, sticky="ew", pady=(4, 2))

        title_lbl = tk.Label(
            summary_header,
            text="Parametros\nactuales",
            bg=params_body.cget("bg"),
            fg="#1f1f1f",
            font=("Segoe UI", 10, "bold italic"),
            anchor="w",
            justify=tk.LEFT,
            padx=4,
        )
        title_lbl.pack(side="left", fill="x", expand=True)

        self.params_summary_filter_var = tk.StringVar(value="Camino\ntransitable")
        filter_menu = tk.OptionMenu(
            summary_header,
            self.params_summary_filter_var,
            "Camino\ntransitable",
            "Muros",
            "Puerta",
            command=lambda _value: self._apply_params_summary_filter(),
        )
        filter_menu.configure(
            bg="#d9d9d9",
            fg="#1f1f1f",
            activebackground="#cfcfcf",
            activeforeground="#1f1f1f",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
            width=12,
            anchor="center",
            justify="center",
            padx=8,
            pady=2,
        )
        filter_menu.pack(side="right", padx=(6, 4))

        summary_container = tk.Frame(params_body, bg=params_body.cget("bg"))
        summary_container.grid(row=1, column=0, sticky="nsew")
        summary_container.columnconfigure(0, weight=1)
        summary_container.rowconfigure(0, weight=1)

        summary_canvas = tk.Canvas(
            summary_container,
            bg=params_body.cget("bg"),
            bd=0,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        summary_scroll = tk.Scrollbar(
            summary_container,
            orient="vertical",
            command=summary_canvas.yview,
        )
        summary_canvas.configure(yscrollcommand=summary_scroll.set)

        summary_canvas.grid(row=0, column=0, sticky="nsew")
        summary_scroll.grid(row=0, column=1, sticky="ns")

        summary_frame = tk.Frame(summary_canvas, bg=params_body.cget("bg"))
        summary_window = summary_canvas.create_window((0, 0), window=summary_frame, anchor="nw")
        summary_frame.bind(
            "<Configure>",
            lambda _event: summary_canvas.configure(scrollregion=summary_canvas.bbox("all")),
        )
        def _on_summary_canvas_configure(event: tk.Event) -> None:
            summary_canvas.itemconfigure(summary_window, width=event.width)
            value_bbox = summary_frame.grid_bbox(1, 0)
            value_width = value_bbox[2] if value_bbox else 70
            wrap = max(int(event.width - value_width - 16), 80)
            for lbl in self.params_summary_name_labels:
                lbl.configure(wraplength=wrap)

        summary_canvas.bind("<Configure>", _on_summary_canvas_configure)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.columnconfigure(1, weight=0, minsize=70)

        all_fields = param_summary_fields()
        self._params_summary_group_keys = {
            "Camino transitable": [],
            "Muros": [],
            "Puerta": [],
        }
        for key, _label in all_fields:
            if key.startswith("door_"):
                group = "Puerta"
            elif key.startswith("wall_") or key in ("max_up_dot", "ground_perp_deg"):
                group = "Muros"
            else:
                group = "Camino transitable"
            self._params_summary_group_keys[group].append(key)

        self.params_summary_labels = {}
        self.params_summary_name_labels = []
        self._params_summary_row_labels = {}
        for idx, (key, label_text) in enumerate(all_fields):
            name_lbl = tk.Label(
                summary_frame,
                text=f"{label_text}:",
                bg=summary_frame.cget("bg"),
                fg="#333333",
                font=("Segoe UI", 10, "bold"),
                anchor="e",
                justify="right",
                wraplength=180,
                padx=4,
            )
            name_lbl.grid(row=idx, column=0, sticky="ew", pady=1)
            self.params_summary_name_labels.append(name_lbl)

            val_lbl = tk.Label(
                summary_frame,
                text="",
                bg=summary_frame.cget("bg"),
                fg="#1f1f1f",
                font=("Segoe UI", 10),
                anchor="w",
                padx=4,
            )
            val_lbl.grid(row=idx, column=1, sticky="w", pady=1)
            self.params_summary_labels[key] = val_lbl
            self._params_summary_row_labels[key] = (name_lbl, val_lbl)

        self._refresh_params_summary()
        self._apply_params_summary_filter()

    def _refresh_params_summary(self) -> None:
        """
        Update the params summary label with the latest defaults.
        """
        if not getattr(self, "params_summary_labels", None):
            return
        try:
            runtime_params = obtener_parametros_ground()
        except Exception:
            runtime_params = {}
        for key, lbl in self.params_summary_labels.items():
            val = runtime_params.get(key, self.config_defaults.get(key, "?"))
            if isinstance(val, bool):
                val = int(val)
            lbl.configure(text=str(val))

    def _apply_params_summary_filter(self) -> None:
        """
        Show only the parameter group selected in the summary filter.
        """
        if not getattr(self, "_params_summary_row_labels", None):
            return
        group = getattr(self, "params_summary_filter_var", None)
        group_value = group.get() if group else "Camino transitable"
        normalized = str(group_value).replace("\n", " ").strip()
        keys_to_show = set(self._params_summary_group_keys.get(normalized, []))
        for key, (name_lbl, val_lbl) in self._params_summary_row_labels.items():
            if key in keys_to_show:
                name_lbl.grid()
                val_lbl.grid()
            else:
                name_lbl.grid_remove()
                val_lbl.grid_remove()

    def _build_db_panel(self, container: tk.Frame) -> None:
        """
        \brief Builds the header for the database panel.
        """
        container_bg = container.cget("bg")
        panel = tk.Frame(container, bg="#b3b3b3", width=240, height=190, highlightthickness=0, bd=0)
        panel.pack(anchor="n", padx=6, pady=6)
        panel.pack_propagate(False)

        header = tk.Label(
            panel,
            text="Panel de Muestras de Datos",
            bg="#b3b3b3",
            fg="black",
            font=("Segoe UI", 10, "bold"),
            anchor="center",
            wraplength=210,
            justify="center",
            pady=4,
        )
        header.pack(side="top", fill="x", padx=6, pady=(4, 2))

        body = tk.Frame(panel, bg=container_bg)
        body.pack(fill="both", expand=True, padx=10, pady=8)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=0)
        body.grid_rowconfigure(1, weight=0)

        numeric_validator = (self.root.register(validate_numeric_entry), "%P")

        self.dataset_index_var = tk.IntVar(value=self.dataset_index + 1)
        entry_numero = tk.Entry(
            body,
            width=12,
            font=("Segoe UI", 10),
            validate="key",
            validatecommand=numeric_validator,
            textvariable=self.dataset_index_var,
        )
        entry_numero.grid(row=0, column=0, sticky="ew", padx=(4, 8), pady=(6, 8))
        entry_numero.bind("<Return>", lambda _event: self._on_dataset_apply())
        btn_aplicar = tk.Button(
            body,
            text="Aplicar",
            bg="#f2c200",
            activebackground="#ffd54f",
            fg="#1f1f1f",
            bd=0,
            width=12,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=self._on_dataset_apply,
        )
        btn_aplicar.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=(6, 8))

        slider = tk.Scale(
            body,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            length=200,
            showvalue=False,
            bg=container_bg,
            highlightthickness=0,
            troughcolor="#d5d5d5",
            variable=self.dataset_index_var,
            command=self._on_dataset_slider,
        )
        slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 8))

        nav_row = tk.Frame(panel, bg=panel.cget("bg"))
        nav_row.pack(side="bottom", fill="x", padx=10, pady=(4, 8))
        btn_atras = tk.Button(
            nav_row,
            text="Atras",
            bg="#e53935",
            activebackground="#f1625f",
            fg="white",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=lambda: self._change_dataset_index(-1),
        )
        btn_atras.pack(side="left", expand=True, fill="x", padx=(0, 10))
        btn_siguiente = tk.Button(
            nav_row,
            text="Siguiente",
            bg="#00b86b",
            activebackground="#21d087",
            fg="white",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=lambda: self._change_dataset_index(1),
        )
        btn_siguiente.pack(side="left", expand=True, fill="x", padx=(10, 0))
        self.sample_controls = {
            "entry": entry_numero,
            "apply": btn_aplicar,
            "slider": slider,
            "back": btn_atras,
            "next": btn_siguiente,
        }
        self._update_sample_panel_state()

    def _build_captura(self, container: tk.Frame) -> None:
        """
        \brief Builds the header for the database panel.
        """
        container_bg = container.cget("bg")
        panel = tk.Frame(container, bg="#b3b3b3", width=240, height=190, highlightthickness=0, bd=0)
        panel.pack(anchor="n", padx=6, pady=6)
        panel.pack_propagate(False)

        header = tk.Label(
            panel,
            text="Panel de Captura",
            bg="#b3b3b3",
            fg="black",
            font=("Segoe UI", 12, "bold"),
            anchor="center",
            pady=4,
        )
        header.pack(side="top", fill="x", padx=6, pady=(4, 2))

        body = tk.Frame(panel, bg=container_bg)
        body.pack(fill="both", expand=True, padx=10, pady=8)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=0)
        body.grid_rowconfigure(1, weight=0)

        numeric_validator = (self.root.register(validate_numeric_entry), "%P")

        self.capture_index_var = tk.IntVar(value=1)
        entry_numero = tk.Entry(
            body,
            width=12,
            font=("Segoe UI", 10),
            validate="key",
            validatecommand=numeric_validator,
            textvariable=self.capture_index_var,
        )
        entry_numero.grid(row=0, column=0, sticky="ew", padx=(4, 8), pady=(6, 8))
        entry_numero.insert(0, "1")
        entry_numero.bind("<Return>", lambda _event: self._on_capture_entry_apply())
        btn_aplicar = tk.Button(
            body,
            text="Visualizar",
            bg="#f2c200",
            activebackground="#ffd54f",
            fg="#1f1f1f",
            bd=0,
            width=12,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=self._on_visualizar_capture,
        )
        btn_aplicar.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=(6, 8))

        slider = tk.Scale(
            body,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            length=200,
            showvalue=False,
            bg=container_bg,
            highlightthickness=0,
            troughcolor="#d5d5d5",
            variable=self.capture_index_var,
            command=self._on_capture_slider,
        )
        slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 8))

        nav_row = tk.Frame(panel, bg=panel.cget("bg"))
        nav_row.pack(side="bottom", fill="x", padx=10, pady=(4, 8))
        btn_atras = tk.Button(
            nav_row,
            text="Borrar",
            bg="#e53935",
            activebackground="#f1625f",
            fg="white",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        )
        btn_atras.pack(side="left", expand=True, fill="x", padx=(0, 10))
        btn_siguiente = tk.Button(
            nav_row,
            text="Capturas",
            bg="#00b86b",
            activebackground="#21d087",
            fg="white",
            bd=0,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=self._on_capturas_pressed,
        )
        btn_siguiente.pack(side="left", expand=True, fill="x", padx=(10, 0))
        entry_numero.configure(state=tk.DISABLED)
        btn_aplicar.configure(state=tk.DISABLED)
        slider.configure(state=tk.DISABLED)
        btn_atras.configure(state=tk.DISABLED)
        self.btn_capturas = btn_siguiente
        if self._capturas_default_bg is None:
            self._capturas_default_bg = btn_siguiente.cget("bg")
            self._capturas_default_activebg = btn_siguiente.cget("activebackground")
        self.capture_controls = {
            "entry": entry_numero,
            "visualizar": btn_aplicar,
            "slider": slider,
            "borrar": btn_atras,
        }

    def _build_logo(self, container: tk.Frame) -> None:
        """
        \brief Loads and shows the logos with Univalle + gato on top and PSI below.
        """
        container_bg = container.cget("bg")
        max_w = max(container.winfo_reqwidth(), container.winfo_width(), 200)
        max_h = max(container.winfo_reqheight(), container.winfo_height(), 160)
        top_w = max((max_w - 40) // 2, 110)  # slightly smaller top logos
        row_h = max(max_h // 2, 90)
        bottom_w = max(max_w - 40, 140)

        def _load_logo(filename: str, target_w: int, target_h: int) -> Optional[ImageTk.PhotoImage]:
            img_path = os.path.join(os.path.dirname(__file__), "images", filename)
            try:
                img = Image.open(img_path).convert("RGBA")
                ratio = min(target_w / max(img.width, 1), target_h / max(img.height, 1), 1.0)
                if ratio < 1.0:
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                return ImageTk.PhotoImage(img)
            except Exception as exc:
                print(f"[GUI] error cargando logo {filename}: {exc}")
                return None

        logos_frame = tk.Frame(container, bg=container_bg)
        logos_frame.pack(expand=True, pady=2)

        self.logo_images = []
        logos_frame.grid_columnconfigure(0, weight=1)
        logos_frame.grid_columnconfigure(1, weight=1)

        layout = [
            ("univalle.png", 0, 0, top_w, row_h, 1),
            ("gato.png", 0, 1, top_w, row_h, 1),
            ("PSI_LOGO.png", 1, 0, bottom_w, row_h, 2),
        ]

        for fname, row, col, tgt_w, tgt_h, span in layout:
            photo = _load_logo(fname, tgt_w, tgt_h)
            if photo:
                self.logo_images.append(photo)
                lbl = tk.Label(logos_frame, image=photo, bg=container_bg, bd=0)
                pady = (6, 6) if fname == "PSI_LOGO.png" else (8, 8)
                lbl.grid(row=row, column=col, columnspan=span, padx=12, pady=pady, sticky="n")
            else:
                fallback = tk.Label(
                    logos_frame,
                    text=f"No se pudo cargar\n{fname}",
                    bg=container_bg,
                    fg="white",
                    font=("Segoe UI", 9, "bold"),
                    justify="center",
                    width=14,
                )
                fallback.grid(row=row, column=col, columnspan=span, padx=12, pady=pady, sticky="n")

    def _build_config_placeholder(self, container: tk.Frame) -> None:
        """
        \brief Configuration panel for RANSAC parameters.
        """
        container.configure(bg="#d9d9d9")
        wrapper = tk.Frame(container, bg="#d9d9d9")
        wrapper.pack(expand=True, fill="both", padx=22, pady=22)

        title = tk.Label(
            wrapper,
            text="Panel de Configuración",
            bg="#d9d9d9",
            fg="#1f1f1f",
            font=("Segoe UI", 16, "bold"),
            anchor="w",
        )
        title.pack(fill="x", pady=(0, 4))

        subtitle = tk.Label(
            wrapper,
            text="Ajusta los parámetros usados por el Aplicativo de Segmentación.",
            bg="#d9d9d9",
            fg="#333333",
            font=("Segoe UI", 10),
            anchor="w",
            justify="left",
        )
        subtitle.pack(fill="x", pady=(0, 10))

        body = tk.Frame(wrapper, bg="#d9d9d9")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3, uniform="cfg_cols")
        body.grid_columnconfigure(1, weight=1, uniform="cfg_cols")
        body.grid_rowconfigure(0, weight=1)

        form = tk.Frame(body, bg="#f2f2f2", bd=1, relief=tk.SOLID)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        form.grid_rowconfigure(0, weight=1)
        form.grid_columnconfigure(0, weight=1)

        form_canvas = tk.Canvas(
            form,
            bg=form.cget("bg"),
            bd=0,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        form_scroll = tk.Scrollbar(form, orient="vertical", command=form_canvas.yview)
        form_canvas.configure(yscrollcommand=form_scroll.set)

        form_canvas.grid(row=0, column=0, sticky="nsew")
        form_scroll.grid(row=0, column=1, sticky="ns")

        form_fields = tk.Frame(form_canvas, bg=form.cget("bg"), padx=10, pady=10)
        form_window = form_canvas.create_window((0, 0), window=form_fields, anchor="nw")
        form_fields.bind(
            "<Configure>",
            lambda _event: form_canvas.configure(scrollregion=form_canvas.bbox("all")),
        )
        form_canvas.bind(
            "<Configure>",
            lambda event: form_canvas.itemconfigure(form_window, width=event.width),
        )
        sections = [
            (
                "Camino transitable",
                [
                    ("subsample_stride", "Submuestreo (stride px)", "2"),
                    ("dist_thresh", "Umbral de distancia al plano (m)", "0.03"),
                    ("max_iters", "Iteraciones max (RANSAC)", "900"),
                    ("min_inliers", "Min inliers (pts)", "400"),
                    ("max_angle_deg", "Ángulo máximo (grados)", "60.0"),
                    ("score_subset", "Subconjunto para puntuar (pts)", "4096"),
                    ("early_stop_ratio", "Ratio corte temprano (0-1)", "0.92"),
                    ("batch_size", "Tamano de lote (modelos)", "256"),
                    ("low_height_pct", "Percentil bajo de altura (%)", "25.0"),
                    ("roi_bottom_fraction", "Fraccion inferior ROI (0-1)", "0.34"),
                    ("refine_full_res", "Refinar full-res", "1"),
                    ("ground_mask_refine", "Mejorar mascara suelo", "0"),
                    ("refine_dist_mult", "Tolerancia refino (dist_mult)", "1.6"),
                ],
            ),
            (
                "Muros",
                [
                    ("wall_subsample_stride", "Submuestreo (stride px)", "2"),
                    ("wall_dist_thresh", "Umbral de distancia al plano (m)", "0.03"),
                    ("wall_max_iters", "Iteraciones max (RANSAC)", "300"),
                    ("wall_min_inliers", "Min inliers (pts)", "400"),
                    ("wall_max_angle_deg", "Ángulo máximo (grados)", "20.0"),
                    ("wall_score_subset", "Subconjunto para puntuar (pts)", "4096"),
                    ("wall_early_stop_ratio", "Ratio corte temprano (0-1)", "0.90"),
                    ("wall_batch_size", "Tamano de lote (modelos)", "1024"),
                    ("wall_refine_dist_mult", "Tolerancia refino (dist_mult)", "1.6"),
                    ("max_up_dot", "Max up dot (0-1)", "0.35"),
                    ("ground_perp_deg", "Perp. suelo (grados)", "20.0"),
                    ("wall_ortho_deg", "Orto paredes (grados)", "20.0"),
                    ("wall_parallel_deg", "Paralelo paredes (grados)", "10.0"),
                    ("wall_parallel_distance_m", "Dist. paredes (m)", "0.60"),
                    ("wall_mask_refine", "Mejorar mascara pared", "0"),
                ],
            ),
            (
                "Puerta",
                [
                    ("door_hue_tol", "Rango color (0-179)", "18"),
                    ("door_min_s", "Color min (0-255)", "30"),
                    ("door_min_v", "Luz min (0-255)", "20"),
                    ("door_glare_s_max", "Reflejo color (0-255)", "35"),
                    ("door_glare_v_min", "Reflejo luz (0-255)", "210"),
                    ("door_glare_v_clip", "Bajar reflejo (0-255)", "200"),
                    ("door_ground_parallel_deg", "Inclinacion max (grados)", "15.0"),
                    ("door_plane_inlier_ratio", "Plano min (0-1)", "0.40"),
                ],
            ),
        ]

        numeric_validator = (self.root.register(validate_numeric_entry), "%P")

        def _bool_from_var(value: str) -> bool:
            return str(value).strip().lower() in ("1", "true", "t", "si", "yes", "on")

        def _toggle_bool_var(var: tk.StringVar) -> None:
            new_state = not _bool_from_var(var.get())
            var.set("1" if new_state else "0")

        def _update_toggle_button(btn: tk.Button, var: tk.StringVar) -> None:
            enabled = _bool_from_var(var.get())
            if enabled:
                btn.configure(
                    text="Desactivar",
                    bg="#e53935",
                    fg="white",
                    activebackground="#f1625f",
                    activeforeground="white",
                )
            else:
                btn.configure(
                    text="Activar",
                    bg="#00b86b",
                    fg="white",
                    activebackground="#21d087",
                    activeforeground="white",
                )

        for title_text, fields in sections:
            section = tk.LabelFrame(
                form_fields,
                text=title_text,
                bg=form.cget("bg"),
                fg="#1f1f1f",
                font=("Segoe UI", 11, "bold"),
                bd=1,
                relief=tk.GROOVE,
                padx=8,
                pady=6,
            )
            section.pack(fill="x", pady=(0, 10))
            section.grid_columnconfigure(1, weight=1)
            section.grid_columnconfigure(3, weight=1)

            for idx, (key, label_text, default) in enumerate(fields):
                row = idx // 2
                col_offset = 2 * (idx % 2)
                lbl = tk.Label(
                    section,
                    text=label_text,
                    bg=section.cget("bg"),
                    fg="#1f1f1f",
                    font=("Segoe UI", 10, "bold"),
                    anchor="w",
                    justify="left",
                    wraplength=220,
                    pady=4,
                )
                lbl.grid(row=row, column=col_offset, sticky="w", padx=(2, 8))

                var = tk.StringVar(value=default)
                self.config_vars[key] = var
                self.config_defaults.setdefault(key, default)
                if key in ("ground_mask_refine", "wall_mask_refine", "refine_full_res"):
                    btn = tk.Button(
                        section,
                        text="Activar",
                        bd=0,
                        padx=8,
                        pady=4,
                        font=("Segoe UI", 9, "bold"),
                        command=lambda v=var: _toggle_bool_var(v),
                    )
                    btn.grid(row=row, column=col_offset + 1, sticky="ew", padx=(0, 2), pady=2)
                    _update_toggle_button(btn, var)
                    var.trace_add("write", lambda *_args, b=btn, v=var: _update_toggle_button(b, v))
                else:
                    entry = tk.Entry(
                        section,
                        textvariable=var,
                        font=("Segoe UI", 10),
                        validate="key",
                        validatecommand=numeric_validator,
                    )
                    entry.grid(row=row, column=col_offset + 1, sticky="ew", padx=(0, 2), pady=2)

        # Ensure the entries reflect the latest defaults pulled from the runtime.
        for key, value in self.config_defaults.items():
            if key in self.config_vars:
                self.config_vars[key].set(str(value))

        # Action buttons always visible under the scrollable field list.
        actions = tk.Frame(form, bg=form.cget("bg"))
        actions.grid(row=1, column=0, columnspan=2, sticky="se", pady=(6, 8), padx=(0, 8))
        btn_cancel = tk.Button(
            actions,
            text="Cancelar",
            bg="#e53935",
            fg="white",
            activebackground="#f1625f",
            activeforeground="white",
            bd=0,
            padx=12,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            command=self._on_config_cancel if hasattr(self, "_on_config_cancel") else None,
        )
        btn_cancel.pack(side="right", padx=(6, 0))
        btn_apply = tk.Button(
            actions,
            text="Aplicar",
            bg="#00b86b",
            fg="#d9d9d9",
            activebackground="#21d087",
            activeforeground="#d9d9d9",
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            command=self._on_config_apply if hasattr(self, "_on_config_apply") else None,
        )
        btn_apply.pack(side="right", padx=(6, 0))
        self._config_apply_btn = btn_apply
        self._apply_btn_default_text = btn_apply.cget("text")
        self._apply_btn_default_bg = btn_apply.cget("bg")
        self._apply_btn_default_activebg = btn_apply.cget("activebackground")

        logos_panel = tk.Frame(body, bg="#d9d9d9")
        logos_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=(4, 0))
        logos_panel.grid_columnconfigure(0, weight=1)
        logos_panel.grid_rowconfigure(0, weight=1)

        # Mini logos panel on the configuration section
        def _load_logo(filename: str, target_w: int, target_h: int) -> Optional[ImageTk.PhotoImage]:
            img_path = os.path.join(os.path.dirname(__file__), "images", filename)
            try:
                img = Image.open(img_path).convert("RGBA")
                ratio = min(target_w / max(img.width, 1), target_h / max(img.height, 1), 1.0)
                if ratio < 1.0:
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                return ImageTk.PhotoImage(img)
            except Exception as exc:
                print(f"[GUI] error cargando logo config {filename}: {exc}")
                return None

        self.config_logo_images: list = []
        logo_layout = [
            ("gato.png", 200, 130),
            ("univalle.png", 200, 130),
            ("PSI_LOGO.png", 180, 90),
        ]
        for r, (fname, tw, th) in enumerate(logo_layout):
            photo = _load_logo(fname, tw, th)
            if photo:
                self.config_logo_images.append(photo)
                lbl = tk.Label(logos_panel, image=photo, bg="#d9d9d9", bd=0)
                lbl.grid(row=r, column=0, padx=6, pady=3, sticky="n")
            else:
                lbl = tk.Label(
                    logos_panel,
                    text=f"{fname}",
                    bg="#d9d9d9",
                    fg="#4a4a4a",
                    font=("Segoe UI", 9, "bold"),
                    justify="center",
                )
                lbl.grid(row=r, column=0, padx=6, pady=3, sticky="n")

        hint = tk.Label(
            wrapper,
            text="Desarrollado por Ricardo Pabón Serna - PSI - Universidad del Valle",
            bg="#d9d9d9",
            fg="#4a4a4a",
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=640,
        )
        hint.pack(fill="x", pady=(10, 0))

    def _set_apply_status(
        self,
        text: str,
        duration_ms: int = 1200,
        bg: Optional[str] = None,
        active_bg: Optional[str] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Temporarily change the apply button label to reflect status and optionally run a callback when it resets.
        """
        btn = getattr(self, "_config_apply_btn", None)
        if btn is None:
            return

        if self._apply_status_after_id is not None:
            try:
                self.root.after_cancel(self._apply_status_after_id)
            except Exception:
                pass
            self._apply_status_after_id = None

        kwargs = {"text": text}
        if bg is not None:
            kwargs["bg"] = bg
        if active_bg is not None:
            kwargs["activebackground"] = active_bg
        btn.configure(**kwargs)

        def _reset() -> None:
            btn.configure(
                text=self._apply_btn_default_text,
                bg=self._apply_btn_default_bg,
                activebackground=self._apply_btn_default_activebg,
            )
            self._apply_status_after_id = None
            if on_reset:
                try:
                    on_reset()
                except Exception as exc:
                    print(f"[GUI] error en callback de aplicar: {exc}")

        self._apply_status_after_id = self.root.after(duration_ms, _reset)

    def _on_config_apply(self) -> None:
        """
        \brief Apply configuration values to the segmentation thread.
        """
        raw_values = {key: var.get() for key, var in self.config_vars.items()}
        parsed = parse_config_params(raw_values)
        if parsed is None:
            self._set_apply_status("No aplicado", bg="#e53935", active_bg="#f1625f")
            return

        updated = actualizar_parametros_ground(parsed)
        for key, val in updated.items():
            if key in self.config_vars:
                self.config_vars[key].set(str(val))
                self.config_defaults[key] = str(val)
        self._refresh_params_summary()
        # Restart worker so the new parameters take effect immediately when running.
        if self._worker and self._worker.is_alive():
            self._restart_worker()
        print("[GUI] Parametros de segmentacion actualizados.")
        self._set_apply_status(
            "Aplicado",
            bg="#5ee68a",
            active_bg="#80f0a8",
            on_reset=lambda: self._show_mode("exec"),
        )

    def _on_config_cancel(self) -> None:
        """
        \brief Restore last applied/default values in the UI and runtime.
        """
        for key, value in self.config_defaults.items():
            if key in self.config_vars:
                self.config_vars[key].set(str(value))

        parsed = parse_config_params(self.config_defaults)
        if parsed:
            actualizar_parametros_ground(parsed)
            self._restart_worker()
        self._refresh_params_summary()

    def _update_sidebar(self, active: str) -> None:
        """
        \brief Updates sidebar button styles based on the active page.
        """
        if active == "config":
            self.btn_config.configure(bg="#3b3b3b")
            self.btn_exec.configure(bg="#5c5c5c")
        else:
            self.btn_exec.configure(bg="#3b3b3b")
            self.btn_config.configure(bg="#4a4a4a")
        self._adjust_sidebar_rows()

    def _show_mode(self, mode: str) -> None:
        """
        \brief Toggles between configuration and execution panels.
        """
        self.active_page = "configuracion" if mode == "config" else "ejecucion"
        if mode == "config":
            for key in ("video", "params", "db", "logo"):
                self.frames[key].grid_remove()
            self.frames["config"].grid()
        else:
            self.frames["config"].grid_remove()
            for key in ("video", "params", "db", "logo"):
                self.frames[key].grid()
        self._update_sidebar(mode)

    def _on_mode_prueba_pressed(self) -> None:
        """
        Handle user tap on the Prueba mode button.
        """
        self._disable_capturas_button()
        self._set_mode("prueba")

    def _on_mode_transmision_pressed(self) -> None:
        """
        Handle user tap on the Transmision mode button.
        """
        self._disable_capturas_button()
        self._set_mode("camera")

    def _set_mode(self, mode: str, update_header: bool = True) -> None:
        """
        \brief Switches the mode and updates button styles.
        """
        self.mode = mode
        if mode == "camera":
            self.btn_mode_cam.configure(relief=tk.SUNKEN, bg="#00b86b", activebackground="#21d087")
            self.btn_mode_test.configure(relief=tk.RAISED, bg="#e53935", activebackground="#f1625f")
            self.mode_label_text.set("Modo de ejecucion: Camara RGB-D")
        else:
            self.btn_mode_test.configure(relief=tk.SUNKEN, bg="#00b86b", activebackground="#21d087")
            self.btn_mode_cam.configure(relief=tk.RAISED, bg="#e53935", activebackground="#f1625f")
            self.mode_label_text.set("Modo de ejecucion: Dataset de pruebas")

        self._update_stream_controls_state()
        self._update_sample_panel_state()

        if mode == "prueba":
            # Ejecuta una pasada de pruebas iniciando/reiniciando el hilo.
            self._restart_worker()
        elif mode == "camera":
            if self._stream_requested:
                # Mantener transmisión si el usuario ya la inició, sin reiniciar si ya corre.
                if not (self._worker and self._worker.is_alive()):
                    self._start_worker()
            else:
                # Si no se ha iniciado transmisión, detiene el hilo.
                self._stop_stream()

    def _update_stream_controls_state(self) -> None:
        """
        Enable stream buttons only in camera mode.
        """
        state = tk.NORMAL if self.mode == "camera" else tk.DISABLED
        for btn in (getattr(self, "btn_start_stream", None), getattr(self, "btn_stop_stream", None)):
            if btn:
                btn.configure(state=state)

    def _update_sample_panel_state(self) -> None:
        """
        Enable/disable sample data panel controls depending on mode.
        """
        controls = getattr(self, "sample_controls", None)
        if not controls:
            return
        state = tk.NORMAL if self.mode == "prueba" else tk.DISABLED
        for widget in controls.values():
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _set_dataset_index(self, index: int, update_controls: bool = True) -> None:
        """
        Store the dataset index (0-based) and keep entry/slider in sync.
        """
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0:
            idx = 0
        self.dataset_index = idx
        if not update_controls:
            return
        self._updating_dataset_controls = True
        try:
            if self.dataset_index_var is not None:
                self.dataset_index_var.set(idx + 1)
            slider = self.sample_controls.get("slider") if hasattr(self, "sample_controls") else None
            if slider:
                try:
                    slider.configure(to=max(int(slider.cget("to")), idx + 1))
                    slider.set(idx + 1)
                except Exception:
                    pass
        finally:
            self._updating_dataset_controls = False

    def _change_dataset_index(self, delta: int) -> None:
        """
        Increment/decrement dataset index and refresh UI.
        """
        self._set_dataset_index(self.dataset_index + delta)

    def _on_dataset_apply(self) -> None:
        """
        Apply the index from the entry to select a dataset frame.
        """
        if self.dataset_index_var is None:
            return
        try:
            requested = int(self.dataset_index_var.get()) - 1
        except Exception:
            return
        self._set_dataset_index(requested)

    def _on_dataset_slider(self, value: str) -> None:
        """
        Sync slider movement with dataset index (value comes as string).
        """
        if self._updating_dataset_controls:
            return
        try:
            slider_val = int(float(value))
        except Exception:
            return
        self._set_dataset_index(slider_val - 1, update_controls=False)

    def _on_capturas_pressed(self) -> None:
        """
        Enable capture panel controls and highlight the Capturas action.
        """
        if self._capturas_default_bg is None and hasattr(self, "btn_capturas"):
            self._capturas_default_bg = self.btn_capturas.cget("bg")
            self._capturas_default_activebg = self.btn_capturas.cget("activebackground")

        for widget in self.capture_controls.values():
            try:
                widget.configure(state=tk.NORMAL)
            except Exception:
                pass
        for btn in (getattr(self, "btn_start_stream", None), getattr(self, "btn_stop_stream", None)):
            if btn:
                try:
                    btn.configure(state=tk.DISABLED)
                except Exception:
                    pass
        for widget in self.sample_controls.values():
            try:
                widget.configure(state=tk.DISABLED)
            except Exception:
                pass

        if hasattr(self, "btn_capturas") and self.btn_capturas:
            default_bg = self._capturas_default_bg or self.btn_capturas.cget("bg")
            default_active = self._capturas_default_activebg or self.btn_capturas.cget("activebackground")
            self.btn_capturas.configure(state=tk.NORMAL, bg=default_bg, activebackground=default_active)
        if getattr(self, "btn_capturar", None):
            try:
                self.btn_capturar.configure(state=tk.DISABLED)
            except Exception:
                pass
        self._show_gallery_panel()

    def _disable_capturas_button(self) -> None:
        """
        Reset the Capturas button styling and disable only its panel controls.
        """
        btn = getattr(self, "btn_capturas", None)
        if btn:
            if self._capturas_default_bg is None:
                self._capturas_default_bg = btn.cget("bg")
                self._capturas_default_activebg = btn.cget("activebackground")
            btn.configure(
                state=tk.NORMAL,
                bg=self._capturas_default_bg,
                activebackground=self._capturas_default_activebg,
            )

        for widget in self.capture_controls.values():
            try:
                widget.configure(state=tk.DISABLED)
            except Exception:
                pass
        if getattr(self, "btn_capturar", None):
            try:
                self.btn_capturar.configure(state=tk.NORMAL)
            except Exception:
                pass
        self._show_video_panel()

    def _start_worker(self) -> None:
        """
        \brief Starts the capture worker thread if not already running.
        """
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _restart_worker(self) -> None:
        """
        \brief Restarts the capture worker thread to apply mode changes.
        """
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _stop_worker(self) -> None:
        """
        Stop the capture worker and clear the latest frame.
        """
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None
        with self._frame_lock:
            self.last_frame = None
            self._last_frame_ts = 0.0

    def _start_stream(self) -> None:
        """
        Switch to camera mode and ensure the worker is running.
        """
        self._stream_requested = True
        self._show_mode("exec")
        self._set_mode("camera", update_header=False)
        self._restart_worker()

    def _stop_stream(self) -> None:
        """
        Stop streaming and clear the display.
        """
        self._stream_requested = False
        self._stop_worker()
        self.mode_label_text.set("Transmision detenida")
        if hasattr(self, "display_area"):
            self.display_area.configure(text="Transmision detenida", image="", bg="#7f7f7f")

    def _capture_screenshot(self) -> None:
        """
        Take a screenshot of the video panel and store it in the uploads folder.
        """
        # Prefer saving the latest processed frame to avoid OS-level screen grab issues (Wayland/Jetson).
        with self._frame_lock:
            frame = None if self.last_frame is None else self.last_frame.copy()

        if frame is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                ensure_upload_dir(UPLOAD_DIR)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(UPLOAD_DIR, f"captura_{timestamp}.png")
                image.save(filepath)
                print(f"[GUI] captura guardada en {filepath}")
                if self._gallery_active:
                    self._refresh_gallery_images()
                return
            except Exception as exc:
                print(f"[GUI] no se pudo guardar captura desde frame: {exc}")

        # Fallback: widget screenshot if no frame is available.
        capture_panel_screenshot(getattr(self, "frame_video_inner", None), UPLOAD_DIR)

    def _worker_loop(self) -> None:
        r"""
        \brief Capture thread that runs AlgoritmosSegmentacion.
        \details Stores the most recent frame and throttles to the target frame
        rate.
        """
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                frame = AlgoritmosSegmentacion(mode=self.mode, dataset_index=self.dataset_index)
                if frame is not None:
                    with self._frame_lock:
                        self.last_frame = frame
                        self._last_frame_ts = time.perf_counter()
                elapsed = time.perf_counter() - loop_start
                sleep_for = max(0.0, TARGET_FRAME_TIME - elapsed)
                time.sleep(sleep_for)
            except Exception as exc:
                # @note Log and slow down to avoid tight error loops.
                print(f"[GUI] error en loop de captura: {exc}")
                time.sleep(0.05)

    def _heartbeat(self) -> None:
        """
        \brief Periodic callback to refresh the UI and schedule the next tick.
        """
        if not self.running:
            return

        if self.active_page == "ejecucion":
            self._update_image()
        self.root.after(10, self._heartbeat)

    def _loading_message(self) -> str:
        """
        Build animated loading text for the display area.
        """
        now = time.perf_counter()
        if now - self._loading_msg_last_ts >= 0.35:
            self._loading_msg_last_ts = now
            self._loading_msg_idx = (self._loading_msg_idx + 1) % len(self._loading_msg_variants)
        return self._loading_msg_variants[self._loading_msg_idx]

    def _update_image(self) -> None:
        """
        \brief Retrieves the segmented image and draws it on the label.
        """
        if self._gallery_active:
            return

        with self._frame_lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
            frame_ts = self._last_frame_ts

        segmentation_active = self.mode == "prueba" or (
            self.mode == "camera" and self._stream_requested
        )
        show_loading = segmentation_active and is_model_loading()

        if frame is None:
            if show_loading:
                self.display_area.configure(
                    text=self._loading_message(),
                    image="",
                    bg="#7f7f7f",
                    compound="center",
                )
                return
            self.display_area.configure(
                text="Sin datos de segmentación.",
                image="",
                bg="#7f7f7f",
                compound="center",
            )
            return

        if show_loading:
            self.display_area.configure(
                text=self._loading_message(),
                image="",
                bg="#7f7f7f",
                compound="center",
            )
            return

        now = time.perf_counter()
        delta = now - frame_ts if frame_ts else 0.0
        overlay_text = ""
        if self.mode == "prueba":
            metrics = obtener_metricas()
            ransac_ms = None if not metrics else metrics.get("last_ransac_ms")
            overlay_text = (
                f"RANSAC: {ransac_ms:.1f} ms" if ransac_ms is not None else "RANSAC: -- ms"
            )
        else:
            if delta > 0:
                self.fps = 1.0 / max(delta, 1e-3)
            self.prev_time = now
            overlay_text = f"FPS: {self.fps:.2f}"

        cv2.putText(
            frame,
            overlay_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        h, w = frame.shape[:2]
        scale = min(DISPLAY_MAX_W / max(w, 1), DISPLAY_MAX_H / max(h, 1), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.photo_ref = ImageTk.PhotoImage(image=image)

        self.display_area.configure(
            image=self.photo_ref,
            text=overlay_text,
            bg="#7f7f7f",
            compound=tk.BOTTOM,
        )

    def _adjust_sidebar_rows(self) -> None:
        """
        \brief Ensures sidebar rows keep a fixed split (3 rows per button).
        """
        sidebar = getattr(self, "frames", {}).get("sidebar") if hasattr(self, "frames") else None
        if sidebar is None:
            return
        try:
            if not sidebar.winfo_exists():
                return
            sidebar_height = max(sidebar.winfo_height(), 1)
        except tk.TclError:
            return
        row_size = max(sidebar_height // 6, 1)
        for r in range(6):
            sidebar.rowconfigure(r, minsize=row_size, weight=1, uniform="sidebar_rows")

    def _on_resize(self, _event=None) -> None:
        """
        \brief Keep sidebar buttons stable during window resizes.
        """
        if not self.running:
            return
        self._adjust_sidebar_rows()

    def _on_close(self) -> None:
        """
        \brief Handles window close: stops threads and releases resources.
        """
        self.running = False
        try:
            self.root.unbind("<Configure>")
        except Exception:
            pass
        try:
            liberar_recursos()
        finally:
            self._stop_event.set()
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=1.0)
            self.root.destroy()


def run_app(mode: str = "prueba") -> None:
    """
    \brief Entry point to launch the segmentation GUI.
    """
    root = tk.Tk()
    SegmentacionApp(root, mode=mode)
    root.mainloop()


if __name__ == "__main__":
    # Allow launching directly on any platform; default to dataset mode to avoid
    # camera/GPU dependencies when they are not available.
    run_app(mode="prueba")
