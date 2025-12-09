"""
Capture Data Access Object.
Handles all database operations related to image captures.
"""

from typing import List, Optional, Dict, Any

from src.models import Capture
from src.api.dbConection import get_connection


class CaptureDAO:
    """Data Access Object for Capture entities."""
    
    @staticmethod
    def get_by_id(capture_id: int) -> Optional[Capture]:
        """
        Get capture by ID.
        
        Args:
            capture_id: Capture ID
            
        Returns:
            Capture object or None if not found
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM `captures` WHERE `id` = %s LIMIT 1", (capture_id,))
                row = cur.fetchone()
                return Capture.from_dict(row) if row else None
    
    @staticmethod
    def get_user_captures(user_id: int, limit: int = 100, 
                         favorites_only: bool = False) -> List[Capture]:
        """
        Get recent captures for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of captures to return
            favorites_only: If True, only return favorited captures
            
        Returns:
            List of Capture objects
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """SELECT * FROM `captures` WHERE `user_id` = %s"""
                if favorites_only:
                    query += " AND `is_favorite` = TRUE"
                query += " ORDER BY `captured_at` DESC LIMIT %s"
                
                cur.execute(query, (user_id, limit))
                rows = cur.fetchall()
                return [Capture.from_dict(row) for row in rows]
    
    @staticmethod
    def create(user_id: int, filename: str,
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
            metadata: Optional dict with image and segmentation metadata
            image_bytes: Optional binary contents of the image (stored as LONGBLOB)
            
        Returns:
            ID of newly created capture
        """
        metadata = metadata or {}
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO `captures` (
                        `user_id`, `configuration_id`, `filename`,
                        `file_size_bytes`, `image_width`, `image_height`,
                        `mode`, `dataset_index`,
                        `ransac_time_ms`, `fps`,
                        `num_ground_pixels`, `num_wall_pixels`, `num_door_pixels`,
                        `tags`, `notes`,
                        `image_data`
                       ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                       )""",
                    (
                        user_id, configuration_id, filename,
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
                        image_bytes,
                    )
                )
            conn.commit()
            return cur.lastrowid
    
    @staticmethod
    def delete(capture_id: int) -> None:
        """
        Delete a capture by ID.
        
        Args:
            capture_id: Capture ID
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM `captures` WHERE `id` = %s", (capture_id,))
            conn.commit()
    
    @staticmethod
    def toggle_favorite(capture_id: int) -> bool:
        """
        Toggle favorite status of a capture.
        
        Args:
            capture_id: Capture ID
            
        Returns:
            New favorite status (True/False)
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
    
    @staticmethod
    def update_notes(capture_id: int, notes: str, tags: Optional[str] = None) -> None:
        """
        Update notes and optionally tags for a capture.
        
        Args:
            capture_id: Capture ID
            notes: Notes text
            tags: Optional tags
        """
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
