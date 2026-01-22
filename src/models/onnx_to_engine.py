#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine for Jetson Nano
"""
import tensorrt as trt
import os

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16_mode=True, max_batch_size=1):
    """
    Convert ONNX model to TensorRT engine

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16_mode: Use FP16 precision (recommended for Jetson Nano)
        max_batch_size: Maximum batch size
    """
    print(f"Building TensorRT engine from {onnx_path}")
    print(f"FP16 mode: {fp16_mode}")

    # Get absolute paths and save current directory
    onnx_path = os.path.abspath(onnx_path)
    engine_path = os.path.abspath(engine_path)
    onnx_dir = os.path.dirname(onnx_path)
    onnx_filename = os.path.basename(onnx_path)
    original_dir = os.getcwd()

    # Change to ONNX directory so TensorRT can find external data files (.onnx.data)
    print(f"Changing to directory: {onnx_dir}")
    os.chdir(onnx_dir)

    try:
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model (using filename only since we're in the right directory)
        print("Parsing ONNX model...")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print(f"Successfully parsed ONNX model")
        print(f"Network inputs: {network.num_inputs}")
        print(f"Network outputs: {network.num_outputs}")

        # Print input/output info
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"  Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

        # Create builder config
        config = builder.create_builder_config()

        # Set memory pool limit (important for Jetson Nano with limited RAM)
        # Using 512MB for workspace (adjust if needed)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * (1 << 20))

        # Enable FP16 mode if supported and requested
        if fp16_mode and builder.platform_has_fast_fp16:
            print("Enabling FP16 precision mode")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("Using FP32 precision mode")

        # Build engine
        print("Building TensorRT engine... This may take several minutes.")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("ERROR: Failed to build engine")
            return None

        # Save engine to file (using absolute path)
        print(f"Saving engine to {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        print(f"Successfully created TensorRT engine!")
        print(f"Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")

        return engine_path

    finally:
        # Always return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_file = os.path.join(script_dir, "mobilenetv2_unet_jetson.onnx")
    engine_file = os.path.join(script_dir, "mobilenetv2_unet_jetson.engine")

    # Check if ONNX file exists
    if not os.path.exists(onnx_file):
        print(f"ERROR: ONNX file not found: {onnx_file}")
        exit(1)

    # Build engine
    result = build_engine(
        onnx_path=onnx_file,
        engine_path=engine_file,
        fp16_mode=True,  # Use FP16 for better performance on Jetson
        max_batch_size=1
    )

    if result:
        print(f"\n✓ Conversion successful!")
        print(f"Engine file: {engine_file}")
    else:
        print(f"\n✗ Conversion failed!")
        exit(1)
