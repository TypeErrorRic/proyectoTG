import os
import sys
import tkinter as tk

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.GUI import SegmentacionApp
from src.LoginWindow import LoginDialog
from src.api.dbConection import test_query


def main() -> None:
    try:
        result = test_query()
        print(f"[DB] Conexion exitosa: {result}")
    except Exception as exc:
        print(f"[DB] No se pudo conectar/consultar: {exc}")
    
    # Crear ventana principal
    root = tk.Tk()
    
    # Mostrar diálogo de login
    print("[Main] Abriendo diálogo de login...")
    login_dialog = LoginDialog(root)
    user = login_dialog.get_user()
    print(f"[Main] Resultado de login: {user}")
    
    if user is None:
        # Login cancelado
        print("[Main] Login cancelado. Cerrando aplicación.")
        root.destroy()
        return
    
    print(f"[Main] Usuario autenticado: {user.get('username')} ({user.get('role')})")
    
    # Crear y mostrar GUI principal
    app = SegmentacionApp(root, current_user=user, mode="camera")
    root.mainloop()


if __name__ == "__main__":
    main()
