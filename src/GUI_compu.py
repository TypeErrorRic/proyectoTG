import tkinter as tk


class GUICompuApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.mode = "exec"
        self._configure_window()
        self._build_grid()
        self._build_panels()
        self._show_mode(self.mode)

    def _configure_window(self) -> None:
        self.root.title("Interfaz estatica 1200x600")
        self.root.geometry("1200x600")
        self.root.resizable(False, False)
        self.root.attributes("-fullscreen", False)

    def _build_grid(self) -> None:
        # Columnas: 40 | 670 | 240 | 240 -> 1190 (aprox 1200 con bordes)
        self.root.grid_columnconfigure(0, minsize=40)
        self.root.grid_columnconfigure(1, minsize=670)
        self.root.grid_columnconfigure(2, minsize=280)
        self.root.grid_columnconfigure(3, minsize=200)
        # Filas: 6 x 100 -> 600
        for r in range(6):
            self.root.grid_rowconfigure(r, minsize=100)

    def _build_panels(self) -> None:
        # Barra lateral
        frame_sidebar = tk.Frame(self.root, bg="#333333")
        frame_sidebar.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self._build_sidebar(frame_sidebar)

        # Panel de video
        frame_video = tk.Frame(self.root, bg="#555555")
        frame_video.grid(row=0, column=1, rowspan=6, sticky="nsew")
        frame_video_inner = tk.Frame(frame_video, bg="#7f7f7f", width=640, height=480, bd=1, relief=tk.SOLID)
        frame_video_inner.pack(side="top", pady=15)
        frame_video_inner.pack_propagate(False)

        self.mode_label_text = tk.StringVar(value="Modo de ejecucion: Camara RGB-D")
        mode_label = tk.Label(
            frame_video,
            textvariable=self.mode_label_text,
            bg="#555555",
            fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        mode_label.pack(side="top", pady=4)

        indicators = tk.Frame(frame_video, bg="#555555")
        indicators.pack(side="top", pady=6)
        for color, text in (("#e53935", "Suelo"), ("#00b86b", "Muro"), ("#1e88e5", "Puerta")):
            item = tk.Frame(indicators, bg="#555555")
            item.pack(side="left", padx=8)
            dot = tk.Canvas(item, width=18, height=18, highlightthickness=0, bg="#555555", bd=0)
            dot.create_oval(2, 2, 16, 16, fill=color, outline=color)
            dot.pack(side="left")
            lbl = tk.Label(item, text=text, bg="#555555", fg="white", font=("Segoe UI", 9, "bold"))
            lbl.pack(side="left", padx=4)

        # Panel de parametros ocupa toda la columna (incluye botones de modo)
        frame_params = tk.Frame(self.root, bg="#999999")
        frame_params.grid(row=0, column=2, rowspan=6, sticky="nsew")
        self._build_exec_controls(frame_params)

        # Panel de base de datos (filas 0-3) -> columna 3
        frame_db = tk.Frame(self.root, bg="#AD7979")
        frame_db.grid(row=0, column=3, rowspan=4, sticky="nsew")

        # Panel de logo (filas 4-5, columna 3)
        frame_logo = tk.Frame(self.root, bg="#423636")
        frame_logo.grid(row=4, column=3, rowspan=2, sticky="nsew")

        # Panel de configuracion a pantalla completa (excepto sidebar)
        frame_config = tk.Frame(self.root, bg="#cccccc")
        frame_config.grid(row=0, column=1, rowspan=6, columnspan=3, sticky="nsew")
        frame_config.grid_remove()

        # Referencias
        self.frames = {
            "sidebar": frame_sidebar,
            "video": frame_video,
            "video_display": frame_video_inner,
            "params": frame_params,
            "db": frame_db,
            "logo": frame_logo,
            "config": frame_config,
        }

    def _build_sidebar(self, container: tk.Frame) -> None:
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        container.columnconfigure(0, weight=1)
        btn_config = tk.Button(
            container,
            text="Config",
            bg="#4a4a4a",
            fg="white",
            bd=0,
            command=lambda: self._show_mode("config"),
        )
        btn_config.grid(row=0, column=0, sticky="nsew")
        btn_exec = tk.Button(
            container,
            text="Ejecucion",
            bg="#5c5c5c",
            fg="white",
            bd=0,
            command=lambda: self._show_mode("exec"),
        )
        btn_exec.grid(row=1, column=0, sticky="nsew")

    def _build_exec_controls(self, container: tk.Frame) -> None:
        # Evita que los hijos modifiquen el tamano del panel de parametros
        container.pack_propagate(False)
        container_bg = container.cget("bg")
        row_holder = tk.Frame(container, bg=container_bg)
        row_holder.pack(fill="x", padx=8, pady=15)
        for label in ("Prueba", "Transmision"):
            btn = tk.Button(
                row_holder,
                text=label,
                bg="#e53935",
                fg="white",
                bd=0,
                font=("Segoe UI", 10, "bold"),
                padx=12,
                pady=10,
            )
            btn.pack(side="left", padx=6, expand=True, fill="x")

        params_panel = tk.Frame(
            container,
            bg="#b3b3b3",
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        params_panel.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        params_panel.columnconfigure(0, weight=1)
        params_panel.rowconfigure(1, weight=1)

        params_title = tk.Label(
            params_panel,
            text="Panel de parametros",
            bg="#b3b3b3",
            fg="#1f1f1f",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            padx=8,
        )
        params_title.grid(row=0, column=0, sticky="ew", pady=(8, 4))

        params_body = tk.Label(
            params_panel,
            text="Espacio reservado para los controles de configuracion.",
            bg="#f2f2f2",
            fg="#333333",
            font=("Segoe UI", 10),
            bd=0,
            relief=tk.FLAT,
            justify=tk.LEFT,
            anchor="nw",
            padx=10,
            pady=10,
        )
        params_body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _show_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "config":
            for key in ("video", "params", "db", "logo"):
                self.frames[key].grid_remove()
            self.frames["config"].grid()
        else:
            self.frames["config"].grid_remove()
            for key in ("video", "params", "db", "logo"):
                self.frames[key].grid()

    def run(self) -> None:
        self.root.mainloop()


def run_app() -> None:
    GUICompuApp().run()


if __name__ == "__main__":
    run_app()
