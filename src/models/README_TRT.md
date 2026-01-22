# ONNX to TensorRT Conversion Guide

## Overview
This guide explains how to convert the ONNX model to TensorRT engine format for optimized inference on Jetson Nano.

## Files
- `mobilenetv2_unet_jetson.onnx` - Original ONNX model
- `onnx_to_engine.py` - Conversion script (ONNX → TensorRT engine)
- `unet_trt.py` - TensorRT inference script
- `unet.py` - Original ONNX Runtime inference script

## System Requirements
- NVIDIA Jetson Nano
- CUDA 10.2.300
- TensorRT 8.0.1.6
- Python 3.8.10
- pycuda

## Installation

Make sure you have the required dependencies:

```bash
# Install pycuda (if not already installed)
pip install pycuda

# TensorRT should already be installed on Jetson Nano
# Verify with:
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

## Step 1: Convert ONNX to TensorRT Engine

Run the conversion script:

```bash
cd src/models
python3 onnx_to_engine.py
```

This will:
- Load `mobilenetv2_unet_jetson.onnx`
- Build a TensorRT engine with FP16 precision (for better performance)
- Save the engine as `mobilenetv2_unet_jetson.engine`

**Note:** The conversion process may take 5-15 minutes on Jetson Nano. Be patient!

## Step 2: Test TensorRT Inference

Once the engine is created, test it:

```bash
python3 unet_trt.py
```

This will run a test inference with random input data.

## Step 3: Use in Your Application

Import and use the TensorRT inference:

```python
from unet_trt import TRTInference
import numpy as np

# Initialize (do this once)
trt_model = TRTInference("mobilenetv2_unet_jetson.engine")

# Run inference
input_data = np.random.randn(1, 5, 256, 256).astype(np.float32)
output = trt_model.infer(input_data)  # Shape: (1, 2, 256, 256)
```

## Performance Benefits

TensorRT engine vs ONNX Runtime:
- **Faster inference**: 2-5x speedup expected
- **Lower memory usage**: Optimized for Jetson Nano
- **FP16 precision**: Better performance with minimal accuracy loss

## Troubleshooting

### Out of Memory Error
If you get OOM during conversion, reduce the workspace size in `onnx_to_engine.py`:
```python
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * (1 << 20))  # 256MB instead of 512MB
```

### Engine build fails
- Check ONNX model is valid: `python3 unet.py`
- Ensure sufficient free memory: Close other applications
- Try without FP16: Set `fp16_mode=False` in `onnx_to_engine.py`

## Comparison

| Method | File Size | Inference Speed | Precision |
|--------|-----------|-----------------|-----------|
| ONNX Runtime | 265 KB | Baseline | FP32 |
| TensorRT Engine | ~300-500 KB | 2-5x faster | FP16 |

## Notes

- The engine file is **platform-specific** (Jetson Nano only)
- Engine files cannot be transferred to other devices
- Rebuild the engine if you update TensorRT or CUDA
- Input shape: (1, 5, 256, 256) - batch=1, channels=5, height=256, width=256
- Output shape: (1, 2, 256, 256) - batch=1, channels=2, height=256, width=256
