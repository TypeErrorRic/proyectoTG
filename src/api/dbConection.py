"""
Database connection and CRUD operations for segmentation application.

Uses PyMySQL to interact with MariaDB/MySQL database.
Environment variables: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
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
# USER OPERATIONS
# ============================================================

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user by username.
    
    Returns user dict or None if not found.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM `users` WHERE `username` = %s LIMIT 1",
                (username,)
            )
            return cur.fetchone()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch user by ID.
    
    Returns user dict or None if not found.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM `users` WHERE `id` = %s LIMIT 1",
                (user_id,)
            )
            return cur.fetchone()


def update_last_login(user_id: int) -> None:
    """Update user's last_login timestamp to current time."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE `users` SET `last_login` = NOW() WHERE `id` = %s",
                (user_id,)
            )
        conn.commit()


def create_user(username: str, email: str, password_hash: str, 
                full_name: Optional[str] = None, role: str = "operator") -> int:
    """
    Create a new user.
    
    Returns the new user's ID.
    """
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
    """
    user = get_user_by_username(username)
    if user is None:
        return None
    
    # Verificar contraseña con MD5
    stored_hash = user.get("password_hash", "")
    password_hash = hash_password(password)
    
    if stored_hash == password_hash:
        # Actualizar último login
        update_last_login(user["id"])
        return user
    
    return None


def hash_password(password: str) -> str:
    """
    Generar hash MD5 de una contraseña.
    
    Args:
        password: Contraseña en texto plano
    
    Returns:
        Hash MD5 como string hexadecimal
    """
    return hashlib.md5(password.encode('utf-8')).hexdigest()


# ============================================================
# CONFIGURATION OPERATIONS
# ============================================================

def get_user_configurations(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all configurations for a user, ordered by most recently updated.
    
    Returns list of configuration dicts.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM `configurations` 
                   WHERE `user_id` = %s 
                   ORDER BY `updated_at` DESC""",
                (user_id,)
            )
            return cur.fetchall()


def get_default_configuration(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get user's default configuration.
    
    Returns configuration dict or None if no default is set.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM `configurations` 
                   WHERE `user_id` = %s AND `is_default` = TRUE 
                   LIMIT 1""",
                (user_id,)
            )
            return cur.fetchone()


def get_configuration_by_id(config_id: int) -> Optional[Dict[str, Any]]:
    """
    Get configuration by ID.
    
    Returns configuration dict or None if not found.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM `configurations` WHERE `id` = %s LIMIT 1",
                (config_id,)
            )
            return cur.fetchone()


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
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # If marking as default, unset other defaults for this user
            if is_default:
                cur.execute(
                    "UPDATE `configurations` SET `is_default` = FALSE WHERE `user_id` = %s",
                    (user_id,)
                )
            
            cur.execute(
                """INSERT INTO `configurations` (
                    `user_id`, `config_name`, `description`, `is_default`,
                    `subsample_stride`, `dist_thresh`, `max_iters`, `min_inliers`,
                    `max_angle_deg`, `score_subset`, `time_budget_ms`, `early_stop_ratio`,
                    `batch_size`, `low_height_pct`, `roi_bottom_fraction`, `roi_expand_step`,
                    `max_agg_points`, `refine_full_res`, `refine_max_points`, 
                    `refine_dist_mult`, `second_pass_mask`
                   ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                   )""",
                (
                    user_id, config_name, description, is_default,
                    params.get("subsample_stride", 1),
                    params.get("dist_thresh", 0.03),
                    params.get("max_iters", 400),
                    params.get("min_inliers", 400),
                    params.get("max_angle_deg", 60.0),
                    params.get("score_subset", 4096),
                    params.get("time_budget_ms", 120.0),
                    params.get("early_stop_ratio", 0.92),
                    params.get("batch_size", 128),
                    params.get("low_height_pct", 25.0),
                    params.get("roi_bottom_fraction", 0.34),
                    params.get("roi_expand_step", 0.2),
                    params.get("max_agg_points", 150000),
                    params.get("refine_full_res", True),
                    params.get("refine_max_points", 200000),
                    params.get("refine_dist_mult", 1.6),
                    params.get("second_pass_mask", True),
                )
            )
        conn.commit()
        return cur.lastrowid


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
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Build dynamic UPDATE query based on provided params
            updates = []
            values = []
            
            if config_name is not None:
                updates.append("`config_name` = %s")
                values.append(config_name)
            
            if description is not None:
                updates.append("`description` = %s")
                values.append(description)
            
            # RANSAC parameters
            param_fields = [
                "subsample_stride", "dist_thresh", "max_iters", "min_inliers",
                "max_angle_deg", "score_subset", "time_budget_ms", "early_stop_ratio",
                "batch_size", "low_height_pct", "roi_bottom_fraction", "roi_expand_step",
                "max_agg_points", "refine_full_res", "refine_max_points",
                "refine_dist_mult", "second_pass_mask"
            ]
            
            for field in param_fields:
                if field in params:
                    updates.append(f"`{field}` = %s")
                    values.append(params[field])
            
            if not updates:
                return  # Nothing to update
            
            values.append(config_id)
            query = f"UPDATE `configurations` SET {', '.join(updates)} WHERE `id` = %s"
            cur.execute(query, values)
        conn.commit()


def delete_configuration(config_id: int) -> None:
    """Delete a configuration by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM `configurations` WHERE `id` = %s", (config_id,))
        conn.commit()


def set_default_configuration(config_id: int, user_id: int) -> None:
    """
    Mark a configuration as default for the user.
    
    Unsets any other default configurations for that user.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Unset all defaults for this user
            cur.execute(
                "UPDATE `configurations` SET `is_default` = FALSE WHERE `user_id` = %s",
                (user_id,)
            )
            # Set this one as default
            cur.execute(
                "UPDATE `configurations` SET `is_default` = TRUE WHERE `id` = %s AND `user_id` = %s",
                (config_id, user_id)
            )
        conn.commit()


# ============================================================
# CAPTURE OPERATIONS
# ============================================================

def create_capture(user_id: int, filename: str, filepath: str,
                  mode: str = "camera",
                  configuration_id: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
    """
    Save a capture to the database.
    
    Args:
        user_id: User ID
        filename: Filename (e.g., "captura_20251208_120000.png")
        filepath: Full or relative path to file
        mode: "camera" or "dataset"
        configuration_id: Optional config ID used for this capture
        metadata: Optional dict with:
            - file_size_bytes
            - image_width, image_height
            - dataset_index
            - ransac_time_ms, fps
            - num_ground_pixels, num_wall_pixels, num_door_pixels
            - tags, notes
    
    Returns the new capture's ID.
    """
    metadata = metadata or {}
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO `captures` (
                    `user_id`, `configuration_id`, `filename`, `filepath`,
                    `file_size_bytes`, `image_width`, `image_height`,
                    `mode`, `dataset_index`,
                    `ransac_time_ms`, `fps`,
                    `num_ground_pixels`, `num_wall_pixels`, `num_door_pixels`,
                    `tags`, `notes`
                   ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                   )""",
                (
                    user_id, configuration_id, filename, filepath,
                    metadata.get("file_size_bytes"),
                    metadata.get("image_width"),
                    metadata.get("image_height"),
                    mode,
                    metadata.get("dataset_index"),
                    metadata.get("ransac_time_ms"),
                    metadata.get("fps"),
                    metadata.get("num_ground_pixels"),
                    metadata.get("num_wall_pixels"),
                    metadata.get("num_door_pixels"),
                    metadata.get("tags"),
                    metadata.get("notes"),
                )
            )
        conn.commit()
        return cur.lastrowid


def get_user_captures(user_id: int, limit: int = 100, 
                     favorites_only: bool = False) -> List[Dict[str, Any]]:
    """
    Get recent captures for a user.
    
    Args:
        user_id: User ID
        limit: Maximum number of captures to return
        favorites_only: If True, only return favorited captures
    
    Returns list of capture dicts.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """SELECT * FROM `captures` WHERE `user_id` = %s"""
            if favorites_only:
                query += " AND `is_favorite` = TRUE"
            query += " ORDER BY `captured_at` DESC LIMIT %s"
            
            cur.execute(query, (user_id, limit))
            return cur.fetchall()


def get_capture_by_id(capture_id: int) -> Optional[Dict[str, Any]]:
    """Get capture by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM `captures` WHERE `id` = %s LIMIT 1", (capture_id,))
            return cur.fetchone()


def toggle_favorite_capture(capture_id: int) -> bool:
    """
    Toggle favorite status of a capture.
    
    Returns the new favorite status (True/False).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE `captures` SET `is_favorite` = NOT `is_favorite` WHERE `id` = %s",
                (capture_id,)
            )
            cur.execute("SELECT `is_favorite` FROM `captures` WHERE `id` = %s", (capture_id,))
            result = cur.fetchone()
        conn.commit()
        return bool(result["is_favorite"]) if result else False


def delete_capture(capture_id: int) -> None:
    """Delete a capture by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM `captures` WHERE `id` = %s", (capture_id,))
        conn.commit()


def update_capture_notes(capture_id: int, notes: str, tags: Optional[str] = None) -> None:
    """Update notes and optionally tags for a capture."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if tags is not None:
                cur.execute(
                    "UPDATE `captures` SET `notes` = %s, `tags` = %s WHERE `id` = %s",
                    (notes, tags, capture_id)
                )
            else:
                cur.execute(
                    "UPDATE `captures` SET `notes` = %s WHERE `id` = %s",
                    (notes, capture_id)
                )
        conn.commit()


# ============================================================
# STATISTICS & REPORTING
# ============================================================

def get_user_stats(user_id: int) -> Dict[str, Any]:
    """
    Get statistics for a user.
    
    Returns dict with:
        - total_configurations
        - total_captures
        - favorite_captures
        - last_capture_date
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT 
                    COUNT(DISTINCT cfg.id) AS total_configurations,
                    COUNT(DISTINCT cap.id) AS total_captures,
                    COUNT(DISTINCT CASE WHEN cap.is_favorite = TRUE THEN cap.id END) AS favorite_captures,
                    MAX(cap.captured_at) AS last_capture_date
                   FROM `users` u
                   LEFT JOIN `configurations` cfg ON u.id = cfg.user_id
                   LEFT JOIN `captures` cap ON u.id = cap.user_id
                   WHERE u.id = %s
                   GROUP BY u.id""",
                (user_id,)
            )
            return cur.fetchone() or {
                "total_configurations": 0,
                "total_captures": 0,
                "favorite_captures": 0,
                "last_capture_date": None
            }
