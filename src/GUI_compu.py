import tkinter as tk


class GUICompuApp:
    """
    Layout estático 1200x600 con paneles fijos:
    - Barra lateral
    - Video
    - Ejecución
    - Parámetros
    - Base de datos
    - Logo
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self._configure_window()
        self._build_grid()
        self._build_panels()

    def _configure_window(self) -> None:
        self.root.title("Interfaz estática 1200x600")
        self.root.geometry("1200x600")
        self.root.resizable(False, False)
        self.root.attributes("-fullscreen", False)

    def _build_grid(self) -> None:
        # Columnas: 50 | 670 | 240 | 240  -> 1200
        self.root.grid_columnconfigure(0, minsize=50)
        self.root.grid_columnconfigure(1, minsize=670)
        self.root.grid_columnconfigure(2, minsize=240)
        self.root.grid_columnconfigure(3, minsize=240)
        # Filas: 200 | 200 | 200 -> 600
        self.root.grid_rowconfigure(0, minsize=200)
        self.root.grid_rowconfigure(1, minsize=200)
        self.root.grid_rowconfigure(2, minsize=200)

    def _build_panels(self) -> None:
        # Barra lateral
        frame_sidebar = tk.Frame(self.root, bg="#333333")
        frame_sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self._build_sidebar(frame_sidebar)
        # Panel de video
        frame_video = tk.Frame(self.root, bg="#555555")
        frame_video.grid(row=0, column=1, rowspan=3, sticky="nsew")
        # Panel de ejecución (fila 0)
        frame_exec = tk.Frame(self.root, bg="#777777")
        frame_exec.grid(row=0, column=2, sticky="nsew")
        # Panel de parámetros (filas 1 y 2)
        frame_params = tk.Frame(self.root, bg="#999999")
        frame_params.grid(row=1, column=2, rowspan=2, sticky="nsew")
        # Panel de base de datos (filas 0 y 1)  -> OJO: column=3
        frame_db = tk.Frame(self.root, bg="#AD7979")
        frame_db.grid(row=0, column=3, rowspan=2, sticky="nsew")
        # Panel de logo (fila 2, col 3)
        frame_logo = tk.Frame(self.root, bg="#423636")
        frame_logo.grid(row=2, column=3, sticky="nsew")

        # Mantener referencias por si se necesitan más adelante
        self.frames = {
            "sidebar": frame_sidebar,
            "video": frame_video,
            "exec": frame_exec,
            "params": frame_params,
            "db": frame_db,
            "logo": frame_logo,
        }

    def _build_sidebar(self, container: tk.Frame) -> None:
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)
        btn_config = tk.Button(container, text="Config", bg="#4a4a4a", fg="white", bd=0)
        btn_config.grid(row=0, column=0, sticky="nsew")
        btn_exec = tk.Button(container, text="Ejecucion", bg="#5c5c5c", fg="white", bd=0)
        btn_exec.grid(row=1, column=0, sticky="nsew")

    def run(self) -> None:
        self.root.mainloop()


def run_app() -> None:
    GUICompuApp().run()


if __name__ == "__main__":
    run_app()
