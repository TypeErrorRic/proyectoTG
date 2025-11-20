"""
Interfaz basica para visualizar la segmentacion.

Se muestran dos botones superiores (Configuracion y Ejecucion) y,
en la pestana de ejecucion, la imagen devuelta por AlgoritmosSegmentacion.
"""

import os
import sys
import time
from typing import Optional

import cv2
import tkinter as tk
from PIL import Image, ImageTk

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.utilities.segmentar import AlgoritmosSegmentacion, liberar_recursos


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

        self.btn_config = tk.Button(
            navbar,
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
        self.btn_config.pack(side=tk.LEFT, ipadx=4, ipady=2)

        self.btn_exec = tk.Button(
            navbar,
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
        self.btn_exec.pack(side=tk.LEFT, ipadx=4, ipady=2)

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

        # Contenido de Ejecucion
        header = tk.Label(
            self.page_exec,
            text="Exactitud de la Medicion (Modo prueba seleccionado)",
            bg="#bfbfbf",
            fg="#1f1f1f",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=14,
            pady=10,
        )
        header.pack(fill=tk.X)

        self.display_area = tk.Label(
            self.page_exec,
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
            self.page_exec,
            text="Camino transitable / Puertas / Muros",
            bg="#f7f7f7",
            fg="#444444",
            font=("Segoe UI", 10),
            anchor="w",
            padx=12,
            pady=6,
        )
        footer.pack(fill=tk.X)

        self._show_page("ejecucion")

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

    def _heartbeat(self) -> None:
        if not self.running:
            return

        if self.active_page == "ejecucion":
            self._update_image()
        self.root.after(40, self._heartbeat)

    def _update_image(self) -> None:
        """
        Obtiene la imagen segmentada y la dibuja en la etiqueta.
        """
        try:
            frame = AlgoritmosSegmentacion(mode=self.mode)
        except Exception as exc:
            self.display_area.configure(
                text=f"Error al obtener imagen: {exc}",
                image="",
                bg="#d9534f",
            )
            return

        if frame is None:
            self.display_area.configure(
                text="Sin datos de segmentacion.",
                image="",
                bg="#7f7f7f",
            )
            return

        now = time.perf_counter()
        delta = now - self.prev_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.prev_time = now

        frame_with_fps = frame.copy()
        cv2.putText(
            frame_with_fps,
            f"FPS: {self.fps:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        frame_rgb = cv2.cvtColor(frame_with_fps, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((980, 600), Image.Resampling.LANCZOS)
        self.photo_ref = ImageTk.PhotoImage(image=image)

        self.display_area.configure(
            image=self.photo_ref,
            text="",
            bg="#7f7f7f",
        )

    def _on_close(self) -> None:
        self.running = False
        try:
            liberar_recursos()
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SegmentacionApp(root, mode="prueba")
    root.mainloop()


if __name__ == "__main__":
    main()
