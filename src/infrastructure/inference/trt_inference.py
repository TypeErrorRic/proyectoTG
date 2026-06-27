#!/usr/bin/env python3
"""
UNet inference using TensorRT engine
"""
import tensorrt as trt
from cuda import cuda
import numpy as np


def check_cuda_err(err):
    """Check CUDA error and raise exception if failed"""
    if isinstance(err, tuple):
        if len(err) == 2:
            # Function returns (CUresult, value)
            err_code, val = err
            if err_code != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA error: {err_code}")
            return val
        else:
            # Function returns (CUresult,) - single element tuple
            err_code = err[0]
            if err_code != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA error: {err_code}")
            return None
    else:
        # Function returns only CUresult
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")
        return None


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

        for i in range(self.engine.num_bindings):
            # Get binding properties
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            name = self.engine.get_binding_name(i)

            # Allocate host and device buffers
            host_mem = np.empty(size, dtype=dtype)
            nbytes = host_mem.nbytes
            device_mem = check_cuda_err(cuda.cuMemAlloc(nbytes))

            # Append to the appropriate list
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'index': i})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'index': i})

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        """
        Run inference

        Args:
            input_data: numpy array of shape (1, 3, 256, 256)

        Returns:
            numpy array of shape (1, 256, 256)
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

        # Run inference (bindings are already set up in self.bindings)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)

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
        output_shape = self.engine.get_binding_shape(self.outputs[0]['index'])
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


class InferenciaTensorRT:
    """Adaptador con nombres del diagrama para inferencia TensorRT."""

    def __init__(self, ruta_modelo=None):
        self.motor = None
        self.contexto = None
        self.buffers_entrada = []
        self.buffers_salida = []
        if ruta_modelo is not None:
            self.cargar_modelo(ruta_modelo)

    def cargar_modelo(self, ruta_modelo: str) -> None:
        self.motor = TRTInference(ruta_modelo)
        self.contexto = self.motor.context
        self.buffers_entrada = self.motor.inputs
        self.buffers_salida = self.motor.outputs

    def inferir(self, entrada):
        if self.motor is None:
            raise RuntimeError("Modelo TensorRT no cargado.")
        return self.motor.infer(entrada)

    def liberar_recursos(self) -> None:
        if self.motor is not None:
            del self.motor
        self.motor = None
        self.contexto = None
        self.buffers_entrada = []
        self.buffers_salida = []


if __name__ == "__main__":
    import os

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_file = os.path.join(script_dir, "doors", "bisenetv2.engine")

    # Check if engine file exists
    if not os.path.exists(engine_file):
        print(f"ERROR: Engine file not found: {engine_file}")
        print(f"Please run onnx_to_engine.py first to create the engine file")
        exit(1)

    # Create inference engine
    print("Initializing TensorRT inference...")
    trt_infer = TRTInference(engine_file)

    # Create dummy input
    x = np.random.randn(1, 3, 256, 256).astype(np.float32)

    print("Running inference...")
    y = trt_infer.infer(x)

    print(f"Output shape: {y.shape}")  # (1, 256, 256)
    print(f"Output min/max: {y.min():.4f} / {y.max():.4f}")
    print("âœ“ TensorRT inference successful!")
