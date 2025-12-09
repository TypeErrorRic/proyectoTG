"""
User Data Access Object.
Handles all database operations related to users.
"""

import hashlib
from typing import Optional, Dict, Any

from src.models import User
from src.api.dbConection import get_connection


class UserDAO:
    """Data Access Object for User entities."""
    
    @staticmethod
    def get_by_username(username: str) -> Optional[User]:
        """
        Fetch user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User object or None if not found
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM `users` WHERE `username` = %s LIMIT 1",
                    (username,)
                )
                row = cur.fetchone()
                return User.from_dict(row) if row else None
    
    @staticmethod
    def get_by_id(user_id: int) -> Optional[User]:
        """
        Fetch user by ID.
        
        Args:
            user_id: User ID to search for
            
        Returns:
            User object or None if not found
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM `users` WHERE `id` = %s LIMIT 1",
                    (user_id,)
                )
                row = cur.fetchone()
                return User.from_dict(row) if row else None
    
    @staticmethod
    def create(username: str, email: str, password: str,
               full_name: Optional[str] = None, role: str = "operator") -> int:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password (will be hashed)
            full_name: Optional full name
            role: User role (default: "operator")
            
        Returns:
            ID of newly created user
        """
        password_hash = UserDAO._hash_password(password)
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO `users` (`username`, `email`, `password_hash`, `full_name`, `role`)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (username, email, password_hash, full_name, role)
                )
            conn.commit()
            return cur.lastrowid
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = UserDAO.get_by_username(username)
        if user is None:
            return None
        
        password_hash = UserDAO._hash_password(password)
        
        if user.password_hash == password_hash:
            # Update last login
            UserDAO.update_last_login(user.id)
            # Refresh user data to get updated last_login
            return UserDAO.get_by_id(user.id)
        
        return None
    
    @staticmethod
    def update_last_login(user_id: int) -> None:
        """
        Update user's last_login timestamp to current time.
        
        Args:
            user_id: User ID
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE `users` SET `last_login` = NOW() WHERE `id` = %s",
                    (user_id,)
                )
            conn.commit()
    
    @staticmethod
    def get_stats(user_id: int) -> Dict[str, Any]:
        """
        Get statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict with total_configurations, total_captures, favorite_captures, last_capture_date
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
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """
        Generate MD5 hash of a password.
        
        Args:
            password: Plain text password
            
        Returns:
            MD5 hash as hexadecimal string
        """
        return hashlib.md5(password.encode('utf-8')).hexdigest()
