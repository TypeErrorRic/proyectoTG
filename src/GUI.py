"""
Basic interface for visualizing the segmentation.

Includes top buttons (Configuration and Execution) and,
on the execution tab, shows the image returned by AlgoritmosSegmentacion.
The capture runs on a separate thread and is limited to 20 fps.
"""

import os
import sys
import time
import threading
from typing import Optional

import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# Adjust sys.path when executed as a script
if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar import AlgoritmosSegmentacion, liberar_recursos

# Limit display size to reduce rescale cost
DISPLAY_MAX_W = 900
DISPLAY_MAX_H = 520
# FPS limit for running AlgoritmosSegmentacion
TARGET_FRAME_TIME = 1.0 / 20.0  # 20 fps max
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")

class SegmentacionApp:
    """
    Main window with two tabs. The execution tab continually displays
    the output from AlgoritmosSegmentacion.
    """

    def __init__(self, root: tk.Tk, mode: str = "prueba") -> None:
        self.root = root
        self.mode = mode  # "camera" or "prueba" (test)
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

        self._ensure_upload_dir()
        self._configure_window()
        self._build_layout()
        self._build_pages()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._heartbeat()

    def _configure_window(self) -> None:
        self.root.title("Segmentacion")
        self.root.geometry("1100x700")
        self.root.configure(bg="#e6e6e6")
        try:
            self.root.state("zoomed")
        except Exception:
            pass
        try:
            self.root.attributes("-zoomed", True)
        except Exception:
            # Fallback for environments that do not support zoomed attribute
            self.root.update_idletasks()
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}+0+0")

    def _ensure_upload_dir(self) -> None:
        """
        Creates the uploads folder so users can drop images there.
        """
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    def _build_icons(self) -> None:
        self.icon_config = self._make_icon(kind="gear")
        self.icon_exec = self._make_icon(kind="camera")

    def _make_icon(self, kind: str) -> ImageTk.PhotoImage:
        """
        Draws a simple icon using PIL so we do not depend on external assets.
        """
        size = 48
        accent = "#f2f2f2"
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if kind == "gear":
            draw.ellipse((10, 10, 38, 38), outline=accent, width=3)
            draw.rectangle((22, 4, 26, 44), fill=accent)
            draw.rectangle((4, 22, 44, 26), fill=accent)
            draw.ellipse((18, 18, 30, 30), fill="#5a5a5a", outline=accent, width=2)
        else:
            draw.rounded_rectangle((6, 12, 42, 36), radius=6, outline=accent, width=3)
            draw.rectangle((28, 6, 40, 14), fill=accent)
            draw.ellipse((18, 16, 30, 28), outline=accent, width=3)

        return ImageTk.PhotoImage(image=img)

    def _build_layout(self) -> None:
        self._build_icons()

        shell = tk.Frame(self.root, bg="#e6e6e6")
        shell.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(shell, bg="#565656", width=90)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=(4, 8), pady=4)
        self.sidebar.grid_propagate(False)
        self.sidebar.rowconfigure(0, weight=1, uniform="sidebar")
        self.sidebar.rowconfigure(1, weight=1, uniform="sidebar")
        self.sidebar.columnconfigure(0, weight=1)

        self.container = tk.Frame(shell, bg="#e6e6e6")
        self.container.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)

        # Pages (shown/hidden via pack in _show_page)
        self.page_config = tk.Frame(self.container, bg="#e6e6e6")
        self.page_exec = tk.Frame(self.container, bg="#e6e6e6")

        self.btn_config = tk.Button(
            self.sidebar,
            image=self.icon_config,
            bg="#5a5a5a",
            activebackground="#707070",
            fg="white",
            bd=2,
            relief=tk.RIDGE,
            highlightthickness=0,
            command=lambda: self._show_page("configuracion"),
        )
        self.btn_config.grid(row=0, column=0, sticky="nsew")

        self.btn_exec = tk.Button(
            self.sidebar,
            image=self.icon_exec,
            bg="#5a5a5a",
            activebackground="#707070",
            fg="white",
            bd=2,
            relief=tk.RIDGE,
            highlightthickness=0,
            command=lambda: self._show_page("ejecucion"),
        )
        self.btn_exec.grid(row=1, column=0, sticky="nsew")

    def _build_pages(self) -> None:
        config_card = tk.Frame(self.page_config, bg="#7f7f7f", bd=2, relief=tk.GROOVE)
        config_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        label_config = tk.Label(
            config_card,
            text="Configuracion",
            bg="#7f7f7f",
            fg="#1f1f1f",
            font=("Segoe UI", 16, "bold"),
            anchor="center",
            padx=20,
            pady=20,
        )
        label_config.pack(expand=True)

        # Execution content: split into left (image) and right (controls) panels
        self.page_exec.columnconfigure(0, weight=3)
        self.page_exec.columnconfigure(1, weight=2)

        left_panel = tk.Frame(self.page_exec, bg="#e6e6e6")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=6)

        video_card = tk.Frame(left_panel, bg="#bfbfbf", bd=2, relief=tk.GROOVE)
        video_card.pack(fill=tk.BOTH, expand=True)

        self.header_label = tk.Label(
            video_card,
            text="Exactitud de la Medicion (Modo prueba seleccionado)",
            bg="#bfbfbf",
            fg="#1f1f1f",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=14,
            pady=10,
        )
        self.header_label.pack(fill=tk.X)

        self.display_area = tk.Label(
            video_card,
            text="Esperando imagen...",
            bg="#7f7f7f",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            bd=0,
            relief=tk.FLAT,
            anchor="center",
            compound=tk.TOP,
            padx=10,
            pady=10,
        )
        self.display_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=12)

        footer = tk.Label(
            video_card,
            text="Camino transitable / Puertas / Muros",
            bg="#bfbfbf",
            fg="#222222",
            font=("Segoe UI", 10),
            anchor="w",
            padx=12,
            pady=6,
        )
        footer.pack(fill=tk.X)

        right_panel = tk.Frame(self.page_exec, bg="#e6e6e6")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=6)
        right_panel.columnconfigure(0, weight=1)
        right_panel.columnconfigure(1, weight=1)
        right_panel.rowconfigure(1, weight=1)

        mode_card = tk.Frame(right_panel, bg="#eaeaea", bd=1, relief=tk.SOLID)
        mode_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        mode_title = tk.Label(
            mode_card,
            text="Modos de Ejecucion",
            bg="#eaeaea",
            fg="#2d2d2d",
            font=("Segoe UI", 11, "bold"),
            pady=10,
        )
        mode_title.pack(fill=tk.X)

        btn_group = tk.Frame(mode_card, bg="#eaeaea")
        btn_group.pack(pady=6)

        self.btn_mode_test = tk.Button(
            btn_group,
            text="Prueba",
            bg="#e53935",
            activebackground="#f1625f",
            fg="white",
            bd=0,
            padx=16,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            command=lambda: self._set_mode("prueba"),
        )
        self.btn_mode_test.pack(side=tk.LEFT, padx=6, ipadx=4, ipady=2)

        self.btn_mode_cam = tk.Button(
            btn_group,
            text="Transmision",
            bg="#c62828",
            activebackground="#e34f4f",
            fg="white",
            bd=0,
            padx=16,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            command=lambda: self._set_mode("camera"),
        )
        self.btn_mode_cam.pack(side=tk.LEFT, padx=6, ipadx=4, ipady=2)

        params_card = tk.Frame(right_panel, bg="#eaeaea", bd=1, relief=tk.SOLID)
        params_card.grid(row=1, column=0, sticky="nsew", padx=(0, 4))

        params_title = tk.Label(
            params_card,
            text="Parametros Configuracion",
            bg="#eaeaea",
            fg="#2d2d2d",
            font=("Segoe UI", 11, "bold"),
            pady=10,
        )
        params_title.pack(fill=tk.X)

        params_box = tk.Label(
            params_card,
            text="Resumen de parametros de configuracion.",
            bg="white",
            fg="#555555",
            font=("Segoe UI", 10),
            bd=1,
            relief=tk.SOLID,
            padx=12,
            pady=12,
        )
        params_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        selector_card = tk.Frame(right_panel, bg="#eaeaea", bd=1, relief=tk.SOLID)
        selector_card.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        selector_card.columnconfigure(0, weight=1)

        selector_title = tk.Label(
            selector_card,
            text="Selector de Base de Datos",
            bg="#eaeaea",
            fg="#2d2d2d",
            font=("Segoe UI", 11, "bold"),
            pady=10,
        )
        selector_title.pack(fill=tk.X)

        selector_top = tk.Frame(selector_card, bg="#eaeaea")
        selector_top.pack(fill=tk.X, padx=12, pady=(4, 4))

        number_label = tk.Label(selector_top, text="Numero", bg="#eaeaea", fg="#2d2d2d", font=("Segoe UI", 10))
        number_label.pack(side=tk.LEFT)

        self.db_number_entry = tk.Entry(selector_top, width=8, font=("Segoe UI", 10))
        self.db_number_entry.pack(side=tk.LEFT, padx=(6, 10))

        btn_aplicar = tk.Button(
            selector_top,
            text="Aplicar",
            bg="#00b86b",
            activebackground="#21d087",
            fg="white",
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            command=lambda: None,
        )
        btn_aplicar.pack(side=tk.LEFT)

        selector_scale = tk.Scale(
            selector_card,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=180,
            showvalue=False,
            bg="#eaeaea",
            highlightthickness=0,
            troughcolor="#d5d5d5",
        )
        selector_scale.pack(padx=12, pady=8)

        nav_btns = tk.Frame(selector_card, bg="#eaeaea")
        nav_btns.pack(padx=12, pady=(6, 12))

        btn_prev = tk.Button(
            nav_btns,
            text="Regresar",
            bg="#e53935",
            activebackground="#f1625f",
            fg="white",
            bd=0,
            padx=12,
            pady=8,
            width=10,
            font=("Segoe UI", 9, "bold"),
            command=lambda: None,
        )
        btn_prev.pack(side=tk.LEFT, padx=4)

        btn_next = tk.Button(
            nav_btns,
            text="Siguiente",
            bg="#00b86b",
            activebackground="#21d087",
            fg="white",
            bd=0,
            padx=12,
            pady=8,
            width=10,
            font=("Segoe UI", 9, "bold"),
            command=lambda: None,
        )
        btn_next.pack(side=tk.LEFT, padx=4)

        self._show_page("ejecucion")
        # Sync initial mode state
        self._set_mode(self.mode)
        # Start capture thread
        self._start_worker()

    def _show_page(self, page: str) -> None:
        self.active_page = page

        if page == "configuracion":
            self.page_exec.pack_forget()
            self.page_config.pack(fill=tk.BOTH, expand=True)
            self._update_sidebar(active="configuracion")
        else:
            self.page_config.pack_forget()
            self.page_exec.pack(fill=tk.BOTH, expand=True)
            self._update_sidebar(active="ejecucion")

    def _update_sidebar(self, active: str) -> None:
        if active == "configuracion":
            self.btn_config.configure(bg="#3b3b3b", relief=tk.SOLID)
            self.btn_exec.configure(bg="#5a5a5a", relief=tk.RIDGE)
        else:
            self.btn_exec.configure(bg="#3b3b3b", relief=tk.SOLID)
            self.btn_config.configure(bg="#5a5a5a", relief=tk.RIDGE)

    def _set_mode(self, mode: str, update_header: bool = True) -> None:
        """
        Switches the mode and updates button styles.
        """
        self.mode = mode
        if mode == "camera":
            self.btn_mode_cam.configure(relief=tk.SUNKEN, bg="#b71c1c")
            self.btn_mode_test.configure(relief=tk.RAISED, bg="#e53935")
        else:
            self.btn_mode_test.configure(relief=tk.SUNKEN, bg="#b71c1c")
            self.btn_mode_cam.configure(relief=tk.RAISED, bg="#c62828")

        if update_header:
            modo_txt = "camera" if mode == "camera" else "prueba"
            self.header_label.configure(
                text=f"Exactitud de la Medicion (Modo {modo_txt} seleccionado)"
            )
        # Restart worker with new mode
        self._restart_worker()

    def _start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _restart_worker(self) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        """
        Capture thread: runs AlgoritmosSegmentacion and stores the latest frame.
        """
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                frame = AlgoritmosSegmentacion(mode=self.mode)
                if frame is not None:
                    with self._frame_lock:
                        self.last_frame = frame
                        self._last_frame_ts = time.perf_counter()
                # Pause to enforce 20 fps cap (TARGET_FRAME_TIME)
                elapsed = time.perf_counter() - loop_start
                sleep_for = max(0.0, TARGET_FRAME_TIME - elapsed)
                time.sleep(sleep_for)
            except Exception as exc:
                # Log and throttle to avoid tight error loops
                print(f"[GUI] error en loop de captura: {exc}")
                time.sleep(0.05)

    def _heartbeat(self) -> None:
        if not self.running:
            return

        if self.active_page == "ejecucion":
            self._update_image()
        # Short interval to avoid choking segmentation; Tkinter queues the refresh.
        self.root.after(10, self._heartbeat)

    def _update_image(self) -> None:
        """
        Fetches the segmented image and draws it on the label.
        """
        # Read latest frame produced by the thread
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
            # Production fps, not the UI refresh rate
            self.fps = 1.0 / max(delta, 1e-3)
        self.prev_time = now

        # Draw FPS on the same frame to avoid expensive copies.
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

        # Reduce resolution before converting with PIL to minimize overhead.
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
        self.running = False
        try:
            liberar_recursos()
        finally:
            self._stop_event.set()
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=1.0)
            self.root.destroy()


def run_app(mode: str = "prueba") -> None:
    root = tk.Tk()
    SegmentacionApp(root, mode=mode)
    root.mainloop()
