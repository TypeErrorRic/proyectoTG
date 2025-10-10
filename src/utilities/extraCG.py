from viewCamera import init_camera, extract_pointcloud

pipeline = init_camera()

try:
    frames = pipeline.wait_for_frames()
    points_xyz, colors_bgr = extract_pointcloud(frames, with_colors=True)
    if points_xyz is not None:
        print("Tengo", len(points_xyz), "bolitas (puntos).")
        print("Primeras 3 bolitas:", points_xyz[:3])
        if colors_bgr is not None:
            print("Sus colores BGR:", colors_bgr[:3])
finally:
    pipeline.stop()
