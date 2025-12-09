"""
Database connection and CRUD operations for segmentation application.

Uses PyMySQL to interact with MariaDB/MySQL database.
Environment variables: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

NOTE: This module now acts as a compatibility wrapper around the DAO layer.
For new code, prefer using the DAO classes directly from src.dao
"""

import os
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime

import pymysql
from pymysql.cursors import DictCursor


def get_db_config() -> Dict[str, Any]:
    """Read database connection config from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "arley"),
        "password": os.getenv("DB_PASSWORD", "qwerty"),
        "database": os.getenv("DB_NAME", "segmentation_app"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
    }


def get_connection():
    """Return a ready-to-use PyMySQL connection."""
    cfg = get_db_config()
    return pymysql.connect(**cfg)


def test_query() -> Dict[str, Any]:
    """
    Execute a test query to validate the connection.
    
    Returns dict with database name and current timestamp.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DATABASE() AS db, NOW() AS ahora")
            return cur.fetchone() or {}


# ============================================================
# USER OPERATIONS (Compatibility wrappers for DAO layer)
# ============================================================

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user by username.
    
    Returns user dict or None if not found.
    
    DEPRECATED: Use UserDAO.get_by_username() instead.
    """
    from src.dao.user_dao import UserDAO
    user = UserDAO.get_by_username(username)
    return user.to_dict() if user else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch user by ID.
    
    Returns user dict or None if not found.
    
    DEPRECATED: Use UserDAO.get_by_id() instead.
    """
    from src.dao.user_dao import UserDAO
    user = UserDAO.get_by_id(user_id)
    return user.to_dict() if user else None


def update_last_login(user_id: int) -> None:
    """
    Update user's last_login timestamp to current time.
    
    DEPRECATED: Use UserDAO.update_last_login() instead.
    """
    from src.dao.user_dao import UserDAO
    UserDAO.update_last_login(user_id)


def create_user(username: str, email: str, password_hash: str, 
                full_name: Optional[str] = None, role: str = "operator") -> int:
    """
    Create a new user.
    
    Returns the new user's ID.
    
    DEPRECATED: Use UserDAO.create() instead (note: takes plain password, not hash).
    """
    from src.dao.user_dao import UserDAO
    # Note: password_hash here is actually already hashed, so we can't use UserDAO.create directly
    # Keep original implementation for exact compatibility
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO `users` (`username`, `email`, `password_hash`, `full_name`, `role`)
                   VALUES (%s, %s, %s, %s, %s)""",
                (username, email, password_hash, full_name, role)
            )
        conn.commit()
        return cur.lastrowid


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Autenticar usuario con nombre de usuario y contraseña.
    
    Args:
        username: Nombre de usuario
        password: Contraseña en texto plano
    
    Returns:
        Diccionario con datos del usuario si es válido, None si falla autenticación.
        
    DEPRECATED: Use UserDAO.authenticate() instead.
    """
    from src.dao.user_dao import UserDAO
    user = UserDAO.authenticate(username, password)
    return user.to_dict() if user else None


def hash_password(password: str) -> str:
    """
    Generar hash MD5 de una contraseña.
    
    Args:
        password: Contraseña en texto plano
    
    Returns:
        Hash MD5 como string hexadecimal
        
    DEPRECATED: Use UserDAO._hash_password() instead (private method).
    """
    return hashlib.md5(password.encode('utf-8')).hexdigest()


# ============================================================
# CONFIGURATION OPERATIONS (Compatibility wrappers for DAO layer)
# ============================================================

def get_user_configurations(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all configurations for a user, ordered by most recently updated.
    
    Returns list of configuration dicts.
    
    DEPRECATED: Use ConfigurationDAO.get_user_configurations() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    configs = ConfigurationDAO.get_user_configurations(user_id)
    return [config.to_dict() for config in configs]


def get_default_configuration(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get user's default configuration.
    
    Returns configuration dict or None if no default is set.
    
    DEPRECATED: Use ConfigurationDAO.get_default_configuration() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    config = ConfigurationDAO.get_default_configuration(user_id)
    return config.to_dict() if config else None


def get_configuration_by_id(config_id: int) -> Optional[Dict[str, Any]]:
    """
    Get configuration by ID.
    
    Returns configuration dict or None if not found.
    
    DEPRECATED: Use ConfigurationDAO.get_by_id() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    config = ConfigurationDAO.get_by_id(config_id)
    return config.to_dict() if config else None


def create_configuration(user_id: int, config_name: str, params: Dict[str, Any],
                        description: Optional[str] = None, 
                        is_default: bool = False) -> int:
    """
    Create a new configuration for a user.
    
    Args:
        user_id: User ID
        config_name: Configuration name
        params: Dict with RANSAC parameters
        description: Optional description
        is_default: Mark as default configuration
    
    Returns the new configuration's ID.
    
    DEPRECATED: Use ConfigurationDAO.create() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    return ConfigurationDAO.create(user_id, config_name, params, description, is_default)


def update_configuration(config_id: int, params: Dict[str, Any],
                        config_name: Optional[str] = None,
                        description: Optional[str] = None) -> None:
    """
    Update an existing configuration's parameters.
    
    Args:
        config_id: Configuration ID
        params: Dict with RANSAC parameters to update
        config_name: Optional new name
        description: Optional new description
        
    DEPRECATED: Use ConfigurationDAO.update() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    ConfigurationDAO.update(config_id, params, config_name, description)


def delete_configuration(config_id: int) -> None:
    """
    Delete a configuration by ID.
    
    DEPRECATED: Use ConfigurationDAO.delete() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    ConfigurationDAO.delete(config_id)


def set_default_configuration(config_id: int, user_id: int) -> None:
    """
    Mark a configuration as default for the user.
    
    Unsets any other default configurations for that user.
    
    DEPRECATED: Use ConfigurationDAO.set_as_default() instead.
    """
    from src.dao.configuration_dao import ConfigurationDAO
    ConfigurationDAO.set_as_default(config_id, user_id)


# ============================================================
# CAPTURE OPERATIONS (Compatibility wrappers for DAO layer)
# ============================================================

def create_capture(user_id: int, filename: str,
                  mode: str = "camera",
                  configuration_id: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  image_bytes: Optional[bytes] = None) -> int:
    """
    Save a capture to the database.
    
    Args:
        user_id: User ID
        filename: Filename (e.g., "captura_20251208_120000.png")
        mode: "camera" or "dataset"
        configuration_id: Optional config ID used for this capture
        metadata: Optional dict with:
            - file_size_bytes
            - image_width, image_height
            - dataset_index
            - ransac_time_ms, fps
            - num_ground_pixels, num_wall_pixels, num_door_pixels
            - tags, notes
        image_bytes: Optional binary contents of the image (stored as LONGBLOB)
    
    Returns the new capture's ID.
    
    DEPRECATED: Use CaptureDAO.create() instead.
    """
    from src.dao.capture_dao import CaptureDAO
    return CaptureDAO.create(user_id, filename, mode, configuration_id, metadata, image_bytes)


def get_user_captures(user_id: int, limit: int = 100, 
                     favorites_only: bool = False) -> List[Dict[str, Any]]:
    """
    Get recent captures for a user.
    
    Args:
        user_id: User ID
        limit: Maximum number of captures to return
        favorites_only: If True, only return favorited captures
    
    Returns list of capture dicts.
    
    DEPRECATED: Use CaptureDAO.get_user_captures() instead.
    """
    from src.dao.capture_dao import CaptureDAO
    captures = CaptureDAO.get_user_captures(user_id, limit, favorites_only)
    return [capture.to_dict() for capture in captures]


def get_capture_by_id(capture_id: int) -> Optional[Dict[str, Any]]:
    """
    Get capture by ID.
    
    DEPRECATED: Use CaptureDAO.get_by_id() instead.
    """
    from src.dao.capture_dao import CaptureDAO
    capture = CaptureDAO.get_by_id(capture_id)
    return capture.to_dict() if capture else None


def toggle_favorite_capture(capture_id: int) -> bool:
    """
    Toggle favorite status of a capture.
    
    Returns the new favorite status (True/False).
    
    DEPRECATED: Use CaptureDAO.toggle_favorite() instead.
    """
    from src.dao.capture_dao import CaptureDAO
    return CaptureDAO.toggle_favorite(capture_id)


def delete_capture(capture_id: int) -> None:
    """
    Delete a capture by ID.
    
    DEPRECATED: Use CaptureDAO.delete() instead.
    """
    from src.dao.capture_dao import CaptureDAO
    CaptureDAO.delete(capture_id)


def update_capture_notes(capture_id: int, notes: str, tags: Optional[str] = None) -> None:
    """
    Update notes and optionally tags for a capture.
    
    DEPRECATED: Use CaptureDAO.update_notes() instead.
    """
    from src.dao.capture_dao import CaptureDAO
    CaptureDAO.update_notes(capture_id, notes, tags)


# ============================================================
# STATISTICS & REPORTING (Compatibility wrappers for DAO layer)
# ============================================================

def get_user_stats(user_id: int) -> Dict[str, Any]:
    """
    Get statistics for a user.
    
    Returns dict with:
        - total_configurations
        - total_captures
        - favorite_captures
        - last_capture_date
        
    DEPRECATED: Use UserDAO.get_stats() instead.
    """
    from src.dao.user_dao import UserDAO
    return UserDAO.get_stats(user_id)
