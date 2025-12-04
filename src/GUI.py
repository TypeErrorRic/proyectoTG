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
from typing import Optional, Dict

import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# @note Adjust sys.path when executed as a script.
if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar2 import AlgoritmosSegmentacion, liberar_recursos

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
        self.active_page = "ejecucion"
        self.running = True

        self.prev_time = time.perf_counter()
        self.fps: float = 0.0
        self.last_frame: Optional = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_frame_ts = 0.0

        self.photo_ref: Optional[ImageTk.PhotoImage] = None
        self.logo_image: Optional[ImageTk.PhotoImage] = None
        self.sidebar_icons_raw: Dict[str, Image.Image] = {}
        self.sidebar_icons: Dict[str, ImageTk.PhotoImage] = {}

        self._ensure_upload_dir()
        self._configure_window()
        self._build_grid()
        self.sidebar_icons_raw = self._load_sidebar_icons()
        self._build_panels()
        self.root.after(30, self._update_sidebar_icons)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_mode("exec")
        self._set_mode(self.mode, update_header=True)
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

    def _ensure_upload_dir(self) -> None:
        """
        \brief Creates the uploads folder so users can drop images there.
        """
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    def _make_icon(self, kind: str) -> Image.Image:
        """
        \brief Fallback icon generator when image assets are missing.
        \return PIL image used as a placeholder icon.
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

    def _load_sidebar_icons(self) -> Dict[str, Image.Image]:
        """
        \brief Loads the raw icons for the sidebar buttons.
        \return Mapping from icon key to loaded PIL image.
        """
        icons: Dict[str, Image.Image] = {}
        assets = {"config": "analitica.png", "exec": "camara.png"}
        base_path = os.path.join(os.path.dirname(__file__), "images")

        for key, filename in assets.items():
            path = os.path.join(base_path, filename)
            try:
                img = Image.open(path).convert("RGBA")
                icons[key] = img
            except Exception as exc:
                print(f"[GUI] no se pudo cargar icono {filename}: {exc}")
                fallback_kind = "gear" if key == "config" else "camera"
                icons[key] = self._make_icon(fallback_kind)

        return icons

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
            margin = 80
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
        borders). Rows: 6 x 100 -> 600.
        """
        self.root.grid_columnconfigure(0, minsize=40)
        self.root.grid_columnconfigure(1, minsize=670)
        self.root.grid_columnconfigure(2, minsize=270)
        self.root.grid_columnconfigure(3, minsize=270)
        for r in range(6):
            self.root.grid_rowconfigure(r, minsize=100)

    def _build_panels(self) -> None:
        """
        \brief Creates and arranges the main UI panels.
        """
        # @note Sidebar panel.
        frame_sidebar = tk.Frame(self.root, bg="#333333")
        frame_sidebar.grid(row=0, column=0, rowspan=6, sticky="nsew")
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
        frame_db.grid(row=0, column=3, rowspan=1, sticky="nsew")
        self._build_db_panel(frame_db)

        # @note Logo panel (rows 4-5, column 3).
        frame_logo = tk.Frame(self.root, bg="#999999")
        frame_logo.grid(row=2, column=3, rowspan=2, sticky="nsew")
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

    def _build_sidebar(self, container: tk.Frame) -> None:
        """
        \brief Builds the sidebar with configuration and execution buttons.
        """
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)
        config_icon = self.sidebar_icons.get("config")
        self.btn_config = tk.Button(
            container,
            image=config_icon,
            text="" if config_icon else "Config",
            bg="#4a4a4a",
            fg="white",
            bd=0,
            width=10,
            height=4,
            compound="center",
            command=lambda: self._show_mode("config"),
        )
        self.btn_config.grid(row=0, column=0, sticky="nsew")
        exec_icon = self.sidebar_icons.get("exec")
        self.btn_exec = tk.Button(
            container,
            image=exec_icon,
            text="" if exec_icon else "Ejecucion",
            bg="#5c5c5c",
            fg="white",
            bd=0,
            width=10,
            height=4,
            compound="center",
            command=lambda: self._show_mode("exec"),
        )
        self.btn_exec.grid(row=1, column=0, sticky="nsew")

    def _build_display_area(self, container: tk.Frame) -> None:
        """
        \brief Builds the video display area and legend.
        """
        frame_video_inner = tk.Frame(container, bg="#7f7f7f", width=DISPLAY_MAX_W, height=DISPLAY_MAX_H, bd=1, relief=tk.SOLID)
        frame_video_inner.pack(side="top", pady=8)
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

        mode_panel = tk.Frame(
            container,
            bg="#7f7f7f",
            bd=1,
            relief=tk.SOLID,
            width=DISPLAY_MAX_W,
            height=80,
            highlightthickness=0,
        )
        mode_panel.pack(side="top", pady=0)
        mode_panel.pack_propagate(False)

        self.mode_label_text = tk.StringVar(
            value="Modo de ejecucion: Camara RGB-D" if self.mode == "camera" else "Modo de ejecucion: Dataset de pruebas"
        )
        mode_label = tk.Label(
            mode_panel,
            textvariable=self.mode_label_text,
            bg="#7f7f7f",
            fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        mode_label.pack(side="top", pady=(4, 2))

        indicators = tk.Frame(mode_panel, bg="#7f7f7f")
        indicators.pack(side="top", pady=(0, 6))
        for color, text in (("#00b86b", "Suelo"), ("#1e88e5", "Muro"), ("#e53935", "Puerta")):
            item = tk.Frame(indicators, bg="#7f7f7f")
            item.pack(side="left", padx=8)
            dot = tk.Canvas(item, width=32, height=32, highlightthickness=0, bg="#7f7f7f", bd=0)
            dot.create_oval(4, 4, 28, 28, fill=color, outline=color)
            dot.pack(side="left")
            lbl = tk.Label(item, text=text, bg="#7f7f7f", fg="white", font=("Segoe UI", 11, "bold"))
            lbl.pack(side="left", padx=6)

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
            command=lambda: self._set_mode("prueba"),
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
            command=lambda: self._set_mode("camera"),
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
            text="Panel de parametros",
            bg="#b3b3b3",
            fg="#1f1f1f",
            font=("Segoe UI", 11, "bold"),
            anchor="center",
            padx=8,
        )
        params_title.grid(row=0, column=0, sticky="ew", pady=(6, 3))

        params_body = tk.Label(
            params_panel,
            text="Espacio reservado para los\ncontroles de configuracion.",
            bg="#f2f2f2",
            fg="#333333",
            font=("Segoe UI", 10),
            bd=0,
            relief=tk.FLAT,
            justify=tk.LEFT,
            anchor="nw",
            padx=8,
            pady=8,
        )
        params_body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

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

        entry_numero = tk.Entry(body, width=12, font=("Segoe UI", 10))
        entry_numero.grid(row=0, column=0, sticky="ew", padx=(4, 8), pady=(6, 8))
        entry_numero.insert(0, "Numero")
        btn_aplicar = tk.Button(
            body,
            text="Aplicar",
            bg="#00b86b",
            activebackground="#21d087",
            fg="white",
            bd=0,
            width=12,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        )
        btn_aplicar.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=(6, 8))

        slider = tk.Scale(
            body,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=200,
            showvalue=False,
            bg=container_bg,
            highlightthickness=0,
            troughcolor="#d5d5d5",
        )
        slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 8))

        nav_row = tk.Frame(panel, bg=panel.cget("bg"))
        nav_row.pack(side="bottom", fill="x", padx=4, pady=(2, 4))
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
        )
        btn_atras.pack(side="left", expand=True, fill="x", padx=(0, 6))
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
        )
        btn_siguiente.pack(side="left", expand=True, fill="x", padx=(6, 0))

    def _build_logo(self, container: tk.Frame) -> None:
        """
        \brief Loads and centers the logo in the panel without changing its size.
        """
        container_bg = container.cget("bg")
        max_w = max(container.winfo_reqwidth(), container.winfo_width(), 1)
        max_h = max(container.winfo_reqheight(), container.winfo_height(), 1)
        if max_w <= 1:
            max_w = 200
        if max_h <= 1:
            max_h = 200

        img_path = os.path.join(os.path.dirname(__file__), "images", "univalle.png")
        try:
            img = Image.open(img_path).convert("RGBA")
            ratio = min(max_w / max(img.width, 1), max_h / max(img.height, 1), 1.0)
            if ratio < 1.0:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(img)
            logo_label = tk.Label(container, image=self.logo_image, bg=container_bg, bd=0)
            logo_label.pack(expand=True)
        except Exception as exc:
            fallback = tk.Label(
                container,
                text="No se pudo cargar\n70_rojo.png",
                bg=container_bg,
                fg="white",
                font=("Segoe UI", 10, "bold"),
                justify="center",
            )
            fallback.pack(expand=True)
            print(f"[GUI] error cargando logo: {exc}")

    def _build_config_placeholder(self, container: tk.Frame) -> None:
        """
        \brief Placeholder panel for configuration controls.
        """
        container.configure(bg="#d9d9d9")
        lbl = tk.Label(
            container,
            text="Panel de configuracion",
            bg="#d9d9d9",
            fg="#1f1f1f",
            font=("Segoe UI", 16, "bold"),
            pady=20,
        )
        lbl.pack(expand=True)

    def _update_sidebar(self, active: str) -> None:
        """
        \brief Updates sidebar button styles based on the active page.
        """
        if active == "config":
            self.btn_config.configure(bg="#3b3b3b", relief=tk.SOLID)
            self.btn_exec.configure(bg="#5c5c5c", relief=tk.RIDGE)
        else:
            self.btn_exec.configure(bg="#3b3b3b", relief=tk.SOLID)
            self.btn_config.configure(bg="#4a4a4a", relief=tk.RIDGE)

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

        if update_header:
            # @note Restart the thread with the newly selected mode.
            self._restart_worker()

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

    def _worker_loop(self) -> None:
        r"""
        \brief Capture thread that runs AlgoritmosSegmentacion.
        \details Stores the most recent frame and throttles to the target frame
        rate.
        """
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                frame = AlgoritmosSegmentacion(mode=self.mode)
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

    def _update_image(self) -> None:
        """
        \brief Retrieves the segmented image and draws it on the label.
        """
        with self._frame_lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
            frame_ts = self._last_frame_ts

        if frame is None:
            self.display_area.configure(
                text="Sin datos de segmentacion.",
                image="",
                bg="#7f7f7f",
            )
            return

        now = time.perf_counter()
        delta = now - frame_ts if frame_ts else 0.0
        if delta > 0:
            self.fps = 1.0 / max(delta, 1e-3)
        self.prev_time = now

        cv2.putText(
            frame,
            f"FPS: {self.fps:.2f}",
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
            text=f"FPS: {self.fps:.2f}",
            bg="#7f7f7f",
            compound=tk.BOTTOM,
        )

    def _on_close(self) -> None:
        """
        \brief Handles window close: stops threads and releases resources.
        """
        self.running = False
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
