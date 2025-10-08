#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización en vivo de la cámara Intel RealSense (solo color o color + profundidad)
Compatible con Jetson Nano, Ubuntu y Windows.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# ============================
# 1) Configuración del pipeline
# ============================
pipeline = rs.pipeline()
config = rs.config()

# Habilita streams (ajusta resolución según tu modelo D435/D415/L515)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ============================
# 2) Inicializa la cámara
# ============================
pipeline.start(config)

# Crea un "colorizer" para visualizar la profundidad como colores
colorizer = rs.colorizer()

print("Presiona ESC para salir...")

try:
    while True:
        # Espera frames sincronizados
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convierte a arrays NumPy
        color_image = np.asanyarray(color_frame.get_data())
        depth_color_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Combina las vistas lado a lado
        combined = np.hstack((color_image, depth_color_image))

        # Muestra
        cv2.imshow('RealSense RGB + Depth', combined)
        if cv2.waitKey(1) & 0xFF == 27:  # tecla ESC
            break

finally:
    # Limpieza
    pipeline.stop()
    cv2.destroyAllWindows()