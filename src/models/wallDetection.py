#!/usr/bin/env python3
"""
Wall and Door Detection using TensorRT UNet model
"""
import os
import numpy as np
import cv2
from typing import Tuple, Optional

# Import TensorRT inference wrapper
from .trt_inference import TRTInference


class WallDetector:
    """Wall and Door detection using TensorRT optimized UNet model"""

    def __init__(self, engine_path: Optional[str] = None):
        """
        Initialize the wall detection model

        Args:
            engine_path: Path to TensorRT engine file. If None, uses default path.
        """
        self.model = None
        self.input_size = (256, 256)
        self.engine_path = engine_path

    def init_model(self, engine_path: Optional[str] = None) -> bool:
        """
        Initialize and load the TensorRT engine model

        Args:
            engine_path: Path to TensorRT engine file. If None, uses default path.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if engine_path is None:
            if self.engine_path is None:
                # Default path: same directory as this script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                engine_path = os.path.join(script_dir, "mobilenetv2_unet_jetson.engine")
            else:
                engine_path = self.engine_path

        # Check if engine file exists
        if not os.path.exists(engine_path):
            print(f"ERROR: Engine file not found: {engine_path}")
            print("Please run: ./run20.sh build-engine")
            return False

        try:
            print(f"Loading TensorRT engine from: {engine_path}")
            self.model = TRTInference(engine_path)
            self.engine_path = engine_path
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return False

    def _preprocess_inputs(
        self,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        floor_mask: np.ndarray
    ) -> np.ndarray:
        """
        Preprocess inputs for the model

        Args:
            depth_image: Depth image (H, W) or (H, W, 1)
            rgb_image: RGB image (H, W, 3)
            floor_mask: Floor mask (H, W) or (H, W, 1)

        Returns:
            Preprocessed input tensor of shape (1, 5, 256, 256)
        """
        # Ensure correct shapes
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        if floor_mask.ndim == 3:
            floor_mask = floor_mask[:, :, 0]

        # Resize images to model input size
        depth_resized = cv2.resize(depth_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb_resized = cv2.resize(rgb_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        floor_resized = cv2.resize(floor_mask, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Normalize depth (0-1 range, assuming depth in mm)
        depth_normalized = depth_resized.astype(np.float32) / 10000.0  # Adjust scaling as needed
        depth_normalized = np.clip(depth_normalized, 0, 1)

        # Normalize RGB (0-1 range)
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0

        # Normalize floor mask (0-1 range)
        floor_normalized = floor_resized.astype(np.float32) / 255.0

        # Stack inputs: [depth, R, G, B, floor_mask] -> (5, H, W)
        input_tensor = np.stack([
            depth_normalized,
            rgb_normalized[:, :, 0],  # R channel
            rgb_normalized[:, :, 1],  # G channel
            rgb_normalized[:, :, 2],  # B channel
            floor_normalized
        ], axis=0)

        # Add batch dimension: (1, 5, H, W)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor.astype(np.float32)

    def _postprocess_outputs(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model outputs

        Args:
            output: Model output of shape (1, 2, 256, 256)
            original_size: Original image size (height, width)

        Returns:
            Tuple of (wall_mask, door_mask) resized to original size
        """
        # Remove batch dimension: (2, 256, 256)
        output = output[0]

        # Extract wall and door channels
        wall_channel = output[0]  # First channel: wall
        door_channel = output[1]  # Second channel: door

        # Apply sigmoid to get probabilities (if not already applied)
        wall_prob = 1 / (1 + np.exp(-wall_channel))  # Sigmoid
        door_prob = 1 / (1 + np.exp(-door_channel))  # Sigmoid

        # Threshold to get binary masks (0.5 threshold)
        wall_mask = (wall_prob > 0.5).astype(np.uint8) * 255
        door_mask = (door_prob > 0.5).astype(np.uint8) * 255

        # Resize back to original size
        wall_mask = cv2.resize(wall_mask, (original_size[1], original_size[0]),
                              interpolation=cv2.INTER_NEAREST)
        door_mask = cv2.resize(door_mask, (original_size[1], original_size[0]),
                              interpolation=cv2.INTER_NEAREST)

        return wall_mask, door_mask

    def wallDetection(
        self,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        floor_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect walls and doors from depth, RGB, and floor mask

        Args:
            depth_image: Depth image (H, W) or (H, W, 1), depth values in mm
            rgb_image: RGB color image (H, W, 3), values 0-255
            floor_mask: Floor mask (H, W) or (H, W, 1), binary mask 0-255

        Returns:
            Tuple of (wall_mask, door_mask):
                - wall_mask: Binary mask (H, W) with walls marked as 255
                - door_mask: Binary mask (H, W) with doors marked as 255

        Raises:
            RuntimeError: If model is not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        # Store original size
        original_size = depth_image.shape[:2]

        # Preprocess inputs
        input_tensor = self._preprocess_inputs(depth_image, rgb_image, floor_mask)

        # Run inference
        output = self.model.infer(input_tensor)

        # Postprocess outputs
        wall_mask, door_mask = self._postprocess_outputs(output, original_size)

        return wall_mask, door_mask


# Standalone functions for backward compatibility
_global_detector = None


def init_model(engine_path: Optional[str] = None) -> bool:
    """
    Initialize the TensorRT wall detection model (standalone function)

    Args:
        engine_path: Path to TensorRT engine file. If None, uses default path.

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _global_detector
    _global_detector = WallDetector()
    return _global_detector.init_model(engine_path)


def wallDetection(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    floor_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect walls and doors from depth, RGB, and floor mask (standalone function)

    Args:
        depth_image: Depth image (H, W) or (H, W, 1), depth values in mm
        rgb_image: RGB color image (H, W, 3), values 0-255
        floor_mask: Floor mask (H, W) or (H, W, 1), binary mask 0-255

    Returns:
        Tuple of (wall_mask, door_mask):
            - wall_mask: Binary mask (H, W) with walls marked as 255
            - door_mask: Binary mask (H, W) with doors marked as 255

    Raises:
        RuntimeError: If model is not initialized
    """
    global _global_detector
    if _global_detector is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")
    return _global_detector.wallDetection(depth_image, rgb_image, floor_mask)


if __name__ == "__main__":
    """Test the wall detection module"""
    print("Testing Wall Detection Module")
    print("=" * 50)

    # Initialize model
    print("\n1. Initializing model...")
    if not init_model():
        print("Failed to initialize model")
        exit(1)

    # Create dummy test data
    print("\n2. Creating test data...")
    height, width = 480, 640
    depth_image = np.random.randint(500, 5000, (height, width), dtype=np.uint16)
    rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    floor_mask = np.random.randint(0, 2, (height, width), dtype=np.uint8) * 255

    # Run detection
    print("\n3. Running wall detection...")
    wall_mask, door_mask = wallDetection(depth_image, rgb_image, floor_mask)

    # Print results
    print("\n4. Results:")
    print(f"   Wall mask shape: {wall_mask.shape}")
    print(f"   Door mask shape: {door_mask.shape}")
    print(f"   Wall pixels detected: {np.sum(wall_mask > 0)}")
    print(f"   Door pixels detected: {np.sum(door_mask > 0)}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
