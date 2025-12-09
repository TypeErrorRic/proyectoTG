"""
Login window for the segmentation application.
Handles user authentication and displays login dialog.
"""

import os
from typing import Optional, Dict, Any

import tkinter as tk
from PIL import Image, ImageTk


class LoginDialog:
    """
    Ventana de diálogo para inicio de sesión.
    """
    
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.result: Optional[Dict[str, Any]] = None
        
        # Crear Toplevel como ventana modal
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Inicio de Sesión")
        self.dialog.geometry("500x450")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg="#2f2f2f")
        
        # Configurar como modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Frame principal
        self.main_container = tk.Frame(self.dialog, bg="#2f2f2f")
        self.main_container.pack(expand=True, fill="both")
        
        self._build_ui()
        
        # Forzar foco
        self.dialog.focus_force()
        if hasattr(self, 'username_entry'):
            self.username_entry.focus()
        
        # Manejo de cierre
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _build_ui(self) -> None:
        """Construir interfaz de login."""
        # Frame principal
        main_frame = tk.Frame(self.main_container, bg="#2f2f2f")
        main_frame.pack(expand=True, fill="both", padx=40, pady=30)
        self.login_frame = main_frame  # Guardar referencia
        
        # Frame de logos
        logo_frame = tk.Frame(main_frame, bg="#2f2f2f")
        logo_frame.pack(pady=(0, 15))
        
        # Cargar y mostrar logos
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Logo PSI
            psi_path = os.path.join(script_dir, "images", "PSI_LOGO.png")
            if os.path.exists(psi_path):
                psi_img = Image.open(psi_path)
                psi_img = psi_img.resize((150, 60), Image.LANCZOS)
                psi_photo = ImageTk.PhotoImage(psi_img)
                psi_label = tk.Label(logo_frame, image=psi_photo, bg="#2f2f2f")
                psi_label.image = psi_photo  # Mantener referencia
                psi_label.pack(side="left", padx=10)
            
            # Logo Univalle
            univalle_path = os.path.join(script_dir, "images", "univalle.png")
            if os.path.exists(univalle_path):
                univalle_img = Image.open(univalle_path)
                univalle_img = univalle_img.resize((60, 60), Image.LANCZOS)
                univalle_photo = ImageTk.PhotoImage(univalle_img)
                univalle_label = tk.Label(logo_frame, image=univalle_photo, bg="#2f2f2f")
                univalle_label.image = univalle_photo  # Mantener referencia
                univalle_label.pack(side="left", padx=10)
        except Exception as e:
            print(f"[Login] Error cargando logos: {e}")
        
        # Título
        title = tk.Label(
            main_frame,
            text="Sistema de Segmentación",
            bg="#2f2f2f",
            fg="white",
            font=("Segoe UI", 16, "bold")
        )
        title.pack(pady=(10, 10))
        
        subtitle = tk.Label(
            main_frame,
            text="Inicie sesión para continuar",
            bg="#2f2f2f",
            fg="#cccccc",
            font=("Segoe UI", 10)
        )
        subtitle.pack(pady=(0, 25))
        
        # Campo de usuario
        user_label = tk.Label(
            main_frame,
            text="Usuario:",
            bg="#2f2f2f",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            anchor="w"
        )
        user_label.pack(fill="x", pady=(0, 5))
        
        self.username_entry = tk.Entry(
            main_frame,
            font=("Segoe UI", 11),
            bg="#4a4a4a",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT,
            bd=5
        )
        self.username_entry.pack(fill="x", pady=(0, 15))
        self.username_entry.bind("<Return>", lambda e: self.password_entry.focus())
        
        # Campo de contraseña
        pass_label = tk.Label(
            main_frame,
            text="Contraseña:",
            bg="#2f2f2f",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            anchor="w"
        )
        pass_label.pack(fill="x", pady=(0, 5))
        
        self.password_entry = tk.Entry(
            main_frame,
            font=("Segoe UI", 11),
            bg="#4a4a4a",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT,
            bd=5,
            show="●"
        )
        self.password_entry.pack(fill="x", pady=(0, 10))
        self.password_entry.bind("<Return>", lambda e: self._on_login())
        
        # Mensaje de error
        self.error_label = tk.Label(
            main_frame,
            text="",
            bg="#2f2f2f",
            fg="#ff5252",
            font=("Segoe UI", 9),
            wraplength=400,
            justify="center",
            height=3
        )
        self.error_label.pack(pady=(5, 10))
        
        # Botones
        button_frame = tk.Frame(main_frame, bg="#2f2f2f")
        button_frame.pack(fill="x", pady=(10, 0))
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancelar",
            bg="#e53935",
            fg="white",
            activebackground="#f1625f",
            activeforeground="white",
            bd=0,
            padx=20,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            command=self._on_cancel
        )
        cancel_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        login_btn = tk.Button(
            button_frame,
            text="Iniciar Sesión",
            bg="#00b86b",
            fg="white",
            activebackground="#21d087",
            activeforeground="white",
            bd=0,
            padx=20,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            command=self._on_login
        )
        login_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))
    
    def _on_login(self) -> None:
        """Manejar intento de login."""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        if not username or not password:
            self.error_label.configure(
                text="Por favor ingrese usuario y contraseña"
            )
            self.error_label.update()
            return
        
        try:
            # Usar DAO directamente (patrón DAO)
            try:
                from src.dao.user_dao import UserDAO
            except ModuleNotFoundError:
                # Fallback para imports directos
                import sys
                import os
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
                from src.dao.user_dao import UserDAO
            
            user_obj = UserDAO.authenticate(username, password)
            
            if user_obj is not None:
                # Convertir a dict para compatibilidad con código existente
                self.result = user_obj.to_dict()
                self.dialog.destroy()
            else:
                # Mostrar mensaje de error detallado
                error_msg = "⚠ Usuario o contraseña incorrectos.\nVerifique sus credenciales."
                self.error_label.configure(text=error_msg)
                self.error_label.update()
                self.dialog.update()
                self.password_entry.delete(0, tk.END)
                self.password_entry.focus()
                # Efecto visual: hacer parpadear el campo de contraseña
                self.password_entry.configure(bg="#5a3a3a")
                self.dialog.after(200, lambda: self.password_entry.configure(bg="#4a4a4a"))
        
        except Exception as exc:
            error_msg = f"Error de conexión:\n{str(exc)[:60]}"
            self.error_label.configure(text=error_msg)
            self.error_label.update()
            self.dialog.update()
    
    def _on_cancel(self) -> None:
        """Cancelar login."""
        self.result = None
        self.dialog.destroy()
    
    def get_user(self) -> Optional[Dict[str, Any]]:
        """Obtener usuario autenticado."""
        # Esperar a que se cierre la ventana modal
        self.dialog.wait_window()
        return self.result
