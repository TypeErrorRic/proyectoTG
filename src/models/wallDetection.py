#!/usr/bin/env python3
"""
Wall and Door Detection using TensorRT UNet model
"""
import os
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

# Import TensorRT inference wrapper
from .trt_inference import TRTInference


# Centralized state for lazy initialization
_runtime: Dict[str, Any] = {
    "model": None,
    "engine_path": None,
    "input_size": (192, 160),  # (width, height) - must match TensorRT engine
}


class WallDetector:
    """Wall and Door detection using TensorRT optimized UNet model"""

    def __init__(self, engine_path: Optional[str] = None):
        """
        Initialize the wall detection model

        Args:
            engine_path: Path to TensorRT engine file. If None, uses default path.
        """
        self.model = None
        self.input_size = (192, 160)
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
                engine_path = os.path.join(script_dir, "walls", "mobilenetv2_unet_jetson_160x192.engine")
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
            Preprocessed input tensor of shape (1, 5, 160, 192)
        """
        # Ensure correct shapes
        depth_image2 = depth_image.copy()
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        if floor_mask.ndim == 3:
            floor_mask = floor_mask[:, :, 0]

        # Resize images to model input size
        depth_resized = cv2.resize(depth_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb_resized = cv2.resize(rgb_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        floor_resized = cv2.resize(floor_mask, self.input_size, interpolation=cv2.INTER_NEAREST)

        # Normalize depth only if from dataset (uint8 format)
        depth_normalized = depth_resized.astype(np.float32)

        # DEBUG: Print depth statistics
        print(f"[DEBUG] Depth - min: {depth_normalized.min():.2f}, max: {depth_normalized.max():.2f}, mean: {depth_normalized.mean():.2f}")

        # Detect format and normalize only dataset format
        max_val = depth_normalized.max()
        if 10 < max_val <= 255:  # Likely uint8 [0, 255] from dataset
            print(f"[DEBUG] Detected dataset format (uint8), normalizing by 255")
            depth_normalized = depth_normalized / 255.0
        else:
            # RealSense format: no normalization
            print(f"[DEBUG] Detected RealSense format, no normalization applied")

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

        # DEBUG: Show all 5 input channels
        cv2.imshow("Input 0 - Depth (normalized)", depth_image2)
        cv2.imshow("Input 1 - Red channel", rgb_normalized[:, :, 0])
        cv2.imshow("Input 2 - Green channel", rgb_normalized[:, :, 1])
        cv2.imshow("Input 3 - Blue channel", rgb_normalized[:, :, 2])
        cv2.imshow("Input 4 - Floor mask", floor_normalized)
        cv2.waitKey(1)  # Non-blocking wait to update windows

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
            output: Model output of shape (1, 2, 160, 192)
            original_size: Original image size (height, width)

        Returns:
            Tuple of (wall_mask, door_mask) resized to original size
        """
        # Remove batch dimension: (2, 160, 192)
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


def _lazy_init(engine_path: Optional[str] = None) -> bool:
    """
    Lazy initialization of the TensorRT model.

    Returns True if model is ready, False otherwise.
    """
    # If model already loaded, return True
    if _runtime["model"] is not None:
        return True

    # Determine engine path
    if engine_path is None:
        engine_path = _runtime.get("engine_path")

    if engine_path is None:
        # Default path: same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, "walls", "mobilenetv2_unet_jetson_160x192.engine")

    # Check if engine file exists
    if not os.path.exists(engine_path):
        print(f"[wallDetection] Engine file not found: {engine_path}")
        return False

    try:
        print(f"[wallDetection] Loading TensorRT engine from: {engine_path}")
        _runtime["model"] = TRTInference(engine_path)
        _runtime["engine_path"] = engine_path
        print("[wallDetection] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[wallDetection] Failed to load model: {e}")
        return False


def _preprocess_inputs(
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
        Preprocessed input tensor of shape (1, 5, 160, 192)
    """
    input_size = _runtime["input_size"]

    # Ensure correct shapes
    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]
    if floor_mask.ndim == 3:
        floor_mask = floor_mask[:, :, 0]

    # Resize images to model input size
    depth_resized = cv2.resize(depth_image, input_size, interpolation=cv2.INTER_LINEAR)
    rgb_resized = cv2.resize(rgb_image, input_size, interpolation=cv2.INTER_LINEAR)
    floor_resized = cv2.resize(floor_mask, input_size, interpolation=cv2.INTER_NEAREST)

    # Normalize depth only if from dataset (uint8 format)
    depth_normalized = depth_resized.astype(np.float32)

    # DEBUG: Print depth statistics
    print(f"[DEBUG] Depth - min: {depth_normalized.min():.2f}, max: {depth_normalized.max():.2f}, mean: {depth_normalized.mean():.2f}")

    # Detect format and normalize only dataset format
    max_val = depth_normalized.max()
    if 10 < max_val <= 255:  # Likely uint8 [0, 255] from dataset
        print(f"[DEBUG] Detected dataset format (uint8), normalizing by 255")
        depth_normalized = depth_normalized / 255.0
    else:
        # RealSense format: no normalization
        print(f"[DEBUG] Detected RealSense format, no normalization applied")

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

    # DEBUG: Show all 5 input channels
    cv2.imshow("Input 0 - Depth (normalized)", depth_resized)
    cv2.imshow("Input 1 - Red channel", rgb_normalized[:, :, 0])
    cv2.imshow("Input 2 - Green channel", rgb_normalized[:, :, 1])
    cv2.imshow("Input 3 - Blue channel", rgb_normalized[:, :, 2])
    cv2.imshow("Input 4 - Floor mask", floor_normalized)
    cv2.waitKey(1)  # Non-blocking wait to update windows

    # Add batch dimension: (1, 5, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor.astype(np.float32)


def _postprocess_outputs(
    output: np.ndarray,
    original_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess model outputs

    Args:
        output: Model output of shape (1, 2, 160, 192)
        original_size: Original image size (height, width)

    Returns:
        Tuple of (wall_mask, door_mask) resized to original size
    """
    # Remove batch dimension: (2, 160, 192)
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
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    floor_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect walls and doors from depth, RGB, and floor mask

    Args:
        depth_image: Depth image (H, W) or (H, W, 1), depth values in meters or mm
        rgb_image: RGB color image (H, W, 3), values 0-255
        floor_mask: Floor mask (H, W) or (H, W, 1), binary mask 0-255

    Returns:
        Tuple of (wall_mask, door_mask):
            - wall_mask: Binary mask (H, W) with walls marked as 255
            - door_mask: Binary mask (H, W) with doors marked as 255

    Raises:
        RuntimeError: If model is not initialized or cannot be initialized
    """
    # Lazy init: try to load model if not loaded
    if not _lazy_init():
        raise RuntimeError("Model not initialized. Cannot run wallDetection.")

    # Store original size
    original_size = depth_image.shape[:2]

    # Preprocess inputs
    input_tensor = _preprocess_inputs(depth_image, rgb_image, floor_mask)

    # Run inference
    output = _runtime["model"].infer(input_tensor)

    # Postprocess outputs
    wall_mask, door_mask = _postprocess_outputs(output, original_size)

    return wall_mask, door_mask
