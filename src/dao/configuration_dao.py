"""
Configuration Data Access Object.
Handles all database operations related to configurations.
"""

from typing import List, Optional, Dict, Any

from src.models import Configuration
from src.api.dbConection import get_connection


class ConfigurationDAO:
    """Data Access Object for Configuration entities."""
    
    @staticmethod
    def get_by_id(config_id: int) -> Optional[Configuration]:
        """
        Get configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            Configuration object or None if not found
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM `configurations` WHERE `id` = %s LIMIT 1",
                    (config_id,)
                )
                row = cur.fetchone()
                return Configuration.from_dict(row) if row else None
    
    @staticmethod
    def get_user_configurations(user_id: int) -> List[Configuration]:
        """
        Get all configurations for a user, ordered by most recently updated.
        
        Args:
            user_id: User ID
            
        Returns:
            List of Configuration objects
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM `configurations` 
                       WHERE `user_id` = %s 
                       ORDER BY `updated_at` DESC""",
                    (user_id,)
                )
                rows = cur.fetchall()
                return [Configuration.from_dict(row) for row in rows]
    
    @staticmethod
    def get_default_configuration(user_id: int) -> Optional[Configuration]:
        """
        Get user's default configuration.
        
        Args:
            user_id: User ID
            
        Returns:
            Configuration object or None if no default is set
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM `configurations` 
                       WHERE `user_id` = %s AND `is_default` = TRUE 
                       LIMIT 1""",
                    (user_id,)
                )
                row = cur.fetchone()
                return Configuration.from_dict(row) if row else None
    
    @staticmethod
    def create(user_id: int, config_name: str, params: Dict[str, Any],
               description: Optional[str] = None, is_default: bool = False) -> int:
        """
        Create a new configuration for a user.
        
        Args:
            user_id: User ID
            config_name: Configuration name
            params: Dict with RANSAC parameters
            description: Optional description
            is_default: Mark as default configuration
            
        Returns:
            ID of newly created configuration
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
    
    @staticmethod
    def update(config_id: int, params: Dict[str, Any],
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
    
    @staticmethod
    def delete(config_id: int) -> None:
        """
        Delete a configuration by ID.
        
        Args:
            config_id: Configuration ID
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM `configurations` WHERE `id` = %s", (config_id,))
            conn.commit()
    
    @staticmethod
    def set_as_default(config_id: int, user_id: int) -> None:
        """
        Mark a configuration as default for the user.
        Unsets any other default configurations for that user.
        
        Args:
            config_id: Configuration ID
            user_id: User ID
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
