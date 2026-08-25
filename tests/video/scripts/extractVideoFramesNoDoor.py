"""Process an HDF5 RGB-D capture without loading or running door segmentation.

This is the no-door variant of ``extractVideoFrames.py``. Ground and wall
segmentation remain enabled. The door detector is replaced with an empty mask
before frame processing, so its TensorRT model is never initialized.
"""

from __future__ import annotations

import numpy as np

import extractVideoFrames as base


_load_base_runtime = base.load_processing_runtime


def load_processing_runtime_without_door() -> None:
    """Load the regular runtime and disable the lazy door detector."""
    _load_base_runtime()
    runtime = base.segmentacion._obtener_impl()

    def detectar_sin_puerta(imagen_rgb, *args, **kwargs):
        del args, kwargs
        return np.zeros(imagen_rgb.shape[:2], dtype=np.uint8)

    runtime.puerta.detectar = detectar_sin_puerta
    runtime.puerta.model_loading = False
    print(
        "Door segmentation disabled: the TensorRT door model will not be loaded.",
        flush=True,
    )


# All calls made by the original processing pipeline now use the no-door loader.
base.load_processing_runtime = load_processing_runtime_without_door


if __name__ == "__main__":
    raise SystemExit(base.main())
