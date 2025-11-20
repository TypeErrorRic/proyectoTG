"""
Interfaz basica para visualizar la segmentacion.

Incluye botones superiores (Configuracion y Ejecucion) y,
en la pestana de ejecucion, muestra la imagen devuelta por AlgoritmosSegmentacion.
La captura corre en un hilo separado y se limita a 20 fps.
"""

import os
import sys
import time
import threading
from typing import Optional

import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Ajustar sys.path si se ejecuta como script
if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar import AlgoritmosSegmentacion, liberar_recursos

# Limitar el tamano al mostrar para reducir costo de reescalado
DISPLAY_MAX_W = 900
DISPLAY_MAX_H = 520
# Limite de FPS para ejecucion de AlgoritmosSegmentacion
TARGET_FRAME_TIME = 1.0 / 20.0  # 20 fps max


class SegmentacionApp:
    """
    Ventana principal con dos pestanas. La pestana de ejecucion muestra
    continuamente la salida de AlgoritmosSegmentacion.
    """

    def __init__(self, root: tk.Tk, mode: str = "prueba") -> None:
        self.root = root
        self.mode = mode  # "camera" o "prueba"
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

        self._configure_window()
        self._build_navbar()
        self._build_pages()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._heartbeat()

    def _configure_window(self) -> None:
        self.root.title("Segmentacion")
        self.root.geometry("1100x700")
        self.root.configure(bg="#e6e6e6")

    def _build_navbar(self) -> None:
        navbar = tk.Frame(self.root, bg="#e6e6e6")
        navbar.pack(fill=tk.X, padx=12, pady=(12, 6))

        nav_inner = tk.Frame(navbar, bg="#e6e6e6")
        nav_inner.pack(expand=True)

        self.btn_config = tk.Button(
            nav_inner,
            text="Configuracion",
            bg="#f2b24a",
            activebackground="#f4c065",
            fg="#2d2d2d",
            bd=0,
            padx=50,
            pady=12,
            font=("Segoe UI", 11, "bold"),
            command=lambda: self._show_page("configuracion"),
        )
        self.btn_config.pack(side=tk.LEFT, ipadx=4, ipady=2, padx=4)

        self.btn_exec = tk.Button(
            nav_inner,
            text="Ejecucion",
            bg="#00b86b",
            activebackground="#21d087",
            fg="white",
            bd=0,
            padx=50,
            pady=12,
            font=("Segoe UI", 11, "bold"),
            command=lambda: self._show_page("ejecucion"),
        )
        self.btn_exec.pack(side=tk.LEFT, ipadx=4, ipady=2, padx=4)

    def _build_pages(self) -> None:
        self.container = tk.Frame(self.root, bg="#e6e6e6")
        self.container.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.page_config = tk.Frame(self.container, bg="#f7f7f7", bd=2, relief=tk.GROOVE)
        self.page_exec = tk.Frame(self.container, bg="#f7f7f7", bd=2, relief=tk.GROOVE)

        # Contenido de Configuracion (placeholder)
        label_config = tk.Label(
            self.page_config,
            text="Pantalla de configuracion (pendiente de implementar).",
            bg="#f7f7f7",
            fg="#444444",
            font=("Segoe UI", 11),
            anchor="center",
            padx=20,
            pady=20,
        )
        label_config.pack(expand=True)

        # Contenido de Ejecucion: dividir en panel izquierdo (imagen) y derecho (controles)
        self.page_exec.columnconfigure(0, weight=3)
        self.page_exec.columnconfigure(1, weight=1)

        left_panel = tk.Frame(self.page_exec, bg="#f7f7f7")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=10)
        right_panel = tk.Frame(self.page_exec, bg="#f7f7f7")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=10)

        self.header_label = tk.Label(
            left_panel,
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
            left_panel,
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
            left_panel,
            text="Camino transitable / Puertas / Muros",
            bg="#f7f7f7",
            fg="#444444",
            font=("Segoe UI", 10),
            anchor="w",
            padx=12,
            pady=6,
        )
        footer.pack(fill=tk.X)

        # Panel lateral de control
        ctrl_card = tk.Frame(right_panel, bg="#eaeaea", bd=1, relief=tk.SOLID)
        ctrl_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        mode_title = tk.Label(
            ctrl_card,
            text="Modos de Ejecucion",
            bg="#eaeaea",
            fg="#2d2d2d",
            font=("Segoe UI", 11, "bold"),
            pady=10,
        )
        mode_title.pack(fill=tk.X)

        btn_group = tk.Frame(ctrl_card, bg="#eaeaea")
        btn_group.pack(pady=6)

        self.btn_mode_cam = tk.Button(
            btn_group,
            text="Transmision (camera)",
            bg="#d1b3ff",
            activebackground="#c59aff",
            fg="#2d2d2d",
            bd=0,
            padx=16,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            command=lambda: self._set_mode("camera"),
        )
        self.btn_mode_cam.pack(side=tk.LEFT, padx=6, ipadx=4, ipady=2)

        self.btn_mode_test = tk.Button(
            btn_group,
            text="Prueba (prueba)",
            bg="#f6c04b",
            activebackground="#f4d074",
            fg="#2d2d2d",
            bd=0,
            padx=16,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            command=lambda: self._set_mode("prueba"),
        )
        self.btn_mode_test.pack(side=tk.LEFT, padx=6, ipadx=4, ipady=2)

        params_title = tk.Label(
            ctrl_card,
            text="Parametros Configuracion",
            bg="#eaeaea",
            fg="#2d2d2d",
            font=("Segoe UI", 11, "bold"),
            pady=10,
        )
        params_title.pack(fill=tk.X, pady=(12, 2))

        params_box = tk.Label(
            ctrl_card,
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

        selector_box = tk.Label(
            ctrl_card,
            text="Selector de elementos de la Base de Datos",
            bg="white",
            fg="#555555",
            font=("Segoe UI", 10),
            bd=1,
            relief=tk.SOLID,
            padx=12,
            pady=12,
        )
        selector_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        self._show_page("ejecucion")
        # Sincroniza estado inicial de modo
        self._set_mode(self.mode)
        # Lanzar hilo de captura
        self._start_worker()

    def _show_page(self, page: str) -> None:
        self.active_page = page

        for widget in (self.page_config, self.page_exec):
            widget.pack_forget()

        if page == "configuracion":
            self.page_config.pack(fill=tk.BOTH, expand=True)
            self.btn_config.configure(relief=tk.SUNKEN)
            self.btn_exec.configure(relief=tk.RAISED)
        else:
            self.page_exec.pack(fill=tk.BOTH, expand=True)
            self.btn_exec.configure(relief=tk.SUNKEN)
            self.btn_config.configure(relief=tk.RAISED)

    def _set_mode(self, mode: str, update_header: bool = True) -> None:
        """
        Cambia el modo y ajusta estilos de botones.
        """
        self.mode = mode
        if mode == "camera":
            self.btn_mode_cam.configure(relief=tk.SUNKEN, bg="#c59aff")
            self.btn_mode_test.configure(relief=tk.RAISED, bg="#f6c04b")
        else:
            self.btn_mode_test.configure(relief=tk.SUNKEN, bg="#f4d074")
            self.btn_mode_cam.configure(relief=tk.RAISED, bg="#d1b3ff")

        if update_header:
            modo_txt = "camera" if mode == "camera" else "prueba"
            self.header_label.configure(
                text=f"Exactitud de la Medicion (Modo {modo_txt} seleccionado)"
            )
        # Reiniciar hilo con nuevo modo
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
        Hilo de captura: ejecuta AlgoritmosSegmentacion y guarda el ultimo frame.
        """
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                frame = AlgoritmosSegmentacion(mode=self.mode)
                if frame is not None:
                    with self._frame_lock:
                        self.last_frame = frame
                        self._last_frame_ts = time.perf_counter()
                # Pausar para marcar el tope de 20 fps (TARGET_FRAME_TIME)
                elapsed = time.perf_counter() - loop_start
                sleep_for = max(0.0, TARGET_FRAME_TIME - elapsed)
                time.sleep(sleep_for)
            except Exception:
                time.sleep(0.01)

    def _heartbeat(self) -> None:
        if not self.running:
            return

        if self.active_page == "ejecucion":
            self._update_image()
        # Intervalo corto para no estrangular la segmentacion; Tkinter cola el refresco.
        self.root.after(10, self._heartbeat)

    def _update_image(self) -> None:
        """
        Obtiene la imagen segmentada y la dibuja en la etiqueta.
        """
        # Leer ultimo frame producido por el hilo
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
            # fps de produccion, no del refresco de UI
            self.fps = 1.0 / max(delta, 1e-3)
        self.prev_time = now

        # Dibujamos FPS sobre el mismo frame para evitar copias costosas.
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

        # Reducimos resolucion antes de pasar por PIL para minimizar overhead.
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
