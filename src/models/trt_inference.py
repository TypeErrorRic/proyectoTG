#!/usr/bin/env python3
"""
UNet inference using TensorRT engine
"""
import tensorrt as trt
from cuda import cuda
import numpy as np


def check_cuda_err(err):
    """Check CUDA error and raise exception if failed"""
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")
    elif isinstance(err, tuple):
        err, val = err
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")
        return val
    return err


class TRTInference:
    """TensorRT inference wrapper"""

    def __init__(self, engine_path):
        """
        Initialize TensorRT inference

        Args:
            engine_path: Path to TensorRT engine file
        """
        # Initialize CUDA
        check_cuda_err(cuda.cuInit(0))
        self.cuda_ctx = check_cuda_err(cuda.cuCtxGetCurrent())

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        print(f"Loading TensorRT engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f"Engine loaded successfully")
        return engine

    def _allocate_buffers(self):
        """Allocate host and device buffers"""
        inputs = []
        outputs = []
        bindings = []
        stream = check_cuda_err(cuda.cuStreamCreate(0))

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = np.empty(size, dtype=dtype)
            nbytes = host_mem.nbytes
            device_mem = check_cuda_err(cuda.cuMemAlloc(nbytes))

            # Append to the appropriate list
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        """
        Run inference

        Args:
            input_data: numpy array of shape (1, 5, 256, 256)

        Returns:
            numpy array of shape (1, 2, 256, 256)
        """
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # Transfer input data to device
        check_cuda_err(cuda.cuMemcpyHtoDAsync(
            self.inputs[0]['device'],
            self.inputs[0]['host'].ctypes.data,
            self.inputs[0]['host'].nbytes,
            self.stream
        ))

        # Set input/output bindings
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp['name'], self.bindings[i])
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], self.bindings[len(self.inputs) + i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream)

        # Transfer predictions back from device
        check_cuda_err(cuda.cuMemcpyDtoHAsync(
            self.outputs[0]['host'].ctypes.data,
            self.outputs[0]['device'],
            self.outputs[0]['host'].nbytes,
            self.stream
        ))

        # Synchronize the stream
        check_cuda_err(cuda.cuStreamSynchronize(self.stream))

        # Reshape output to expected shape
        output_shape = self.engine.get_tensor_shape(self.outputs[0]['name'])
        output = self.outputs[0]['host'].reshape(output_shape)

        return output

    def __del__(self):
        """Cleanup"""
        # Free device memory
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                if 'device' in inp:
                    cuda.cuMemFree(inp['device'])
        if hasattr(self, 'outputs'):
            for out in self.outputs:
                if 'device' in out:
                    cuda.cuMemFree(out['device'])

        # Destroy stream
        if hasattr(self, 'stream'):
            cuda.cuStreamDestroy(self.stream)

        # Delete TensorRT objects
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


if __name__ == "__main__":
    import os

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_file = os.path.join(script_dir, "mobilenetv2_unet_jetson.engine")

    # Check if engine file exists
    if not os.path.exists(engine_file):
        print(f"ERROR: Engine file not found: {engine_file}")
        print(f"Please run onnx_to_engine.py first to create the engine file")
        exit(1)

    # Create inference engine
    print("Initializing TensorRT inference...")
    trt_infer = TRTInference(engine_file)

    # Create dummy input (same as original unet.py)
    x = np.random.randn(1, 5, 256, 256).astype(np.float32)

    print("Running inference...")
    y = trt_infer.infer(x)

    print(f"Output shape: {y.shape}")  # (1, 2, 256, 256)
    print(f"Output min/max: {y.min():.4f} / {y.max():.4f}")
    print("✓ TensorRT inference successful!")
