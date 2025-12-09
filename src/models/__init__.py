"""
Domain models/entities for the segmentation application.
These classes represent the business objects used throughout the application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """User entity representing a system user."""
    id: int
    username: str
    email: str
    password_hash: str
    role: str = "operator"
    full_name: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create User instance from database dict."""
        return cls(
            id=data['id'],
            username=data['username'],
            email=data['email'],
            password_hash=data['password_hash'],
            role=data.get('role', 'operator'),
            full_name=data.get('full_name'),
            created_at=data.get('created_at'),
            last_login=data.get('last_login')
        )
    
    def to_dict(self) -> dict:
        """Convert User to dict (for backwards compatibility)."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'role': self.role,
            'full_name': self.full_name,
            'created_at': self.created_at,
            'last_login': self.last_login
        }


@dataclass
class Configuration:
    """Configuration entity for RANSAC parameters."""
    id: int
    user_id: int
    config_name: str
    description: Optional[str] = None
    is_default: bool = False
    
    # RANSAC parameters
    subsample_stride: int = 1
    dist_thresh: float = 0.03
    max_iters: int = 400
    min_inliers: int = 400
    max_angle_deg: float = 60.0
    score_subset: int = 4096
    time_budget_ms: float = 120.0
    early_stop_ratio: float = 0.92
    batch_size: int = 128
    low_height_pct: float = 25.0
    roi_bottom_fraction: float = 0.34
    roi_expand_step: float = 0.2
    max_agg_points: int = 150000
    refine_full_res: bool = True
    refine_max_points: int = 200000
    refine_dist_mult: float = 1.6
    second_pass_mask: bool = True
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Configuration':
        """Create Configuration instance from database dict."""
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            config_name=data['config_name'],
            description=data.get('description'),
            is_default=bool(data.get('is_default', False)),
            subsample_stride=data.get('subsample_stride', 1),
            dist_thresh=data.get('dist_thresh', 0.03),
            max_iters=data.get('max_iters', 400),
            min_inliers=data.get('min_inliers', 400),
            max_angle_deg=data.get('max_angle_deg', 60.0),
            score_subset=data.get('score_subset', 4096),
            time_budget_ms=data.get('time_budget_ms', 120.0),
            early_stop_ratio=data.get('early_stop_ratio', 0.92),
            batch_size=data.get('batch_size', 128),
            low_height_pct=data.get('low_height_pct', 25.0),
            roi_bottom_fraction=data.get('roi_bottom_fraction', 0.34),
            roi_expand_step=data.get('roi_expand_step', 0.2),
            max_agg_points=data.get('max_agg_points', 150000),
            refine_full_res=bool(data.get('refine_full_res', True)),
            refine_max_points=data.get('refine_max_points', 200000),
            refine_dist_mult=data.get('refine_dist_mult', 1.6),
            second_pass_mask=bool(data.get('second_pass_mask', True)),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    def to_dict(self) -> dict:
        """Convert Configuration to dict (for backwards compatibility)."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'config_name': self.config_name,
            'description': self.description,
            'is_default': self.is_default,
            'subsample_stride': self.subsample_stride,
            'dist_thresh': self.dist_thresh,
            'max_iters': self.max_iters,
            'min_inliers': self.min_inliers,
            'max_angle_deg': self.max_angle_deg,
            'score_subset': self.score_subset,
            'time_budget_ms': self.time_budget_ms,
            'early_stop_ratio': self.early_stop_ratio,
            'batch_size': self.batch_size,
            'low_height_pct': self.low_height_pct,
            'roi_bottom_fraction': self.roi_bottom_fraction,
            'roi_expand_step': self.roi_expand_step,
            'max_agg_points': self.max_agg_points,
            'refine_full_res': self.refine_full_res,
            'refine_max_points': self.refine_max_points,
            'refine_dist_mult': self.refine_dist_mult,
            'second_pass_mask': self.second_pass_mask,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


@dataclass
class Capture:
    """Capture entity for saved image captures."""
    id: int
    user_id: int
    filename: str
    configuration_id: Optional[int] = None
    
    # Image metadata
    file_size_bytes: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_data: Optional[bytes] = None
    
    # Capture metadata
    mode: str = "camera"
    dataset_index: Optional[int] = None
    ransac_time_ms: Optional[float] = None
    fps: Optional[float] = None
    
    # Segmentation metrics
    num_ground_pixels: Optional[int] = None
    num_wall_pixels: Optional[int] = None
    num_door_pixels: Optional[int] = None
    
    # User annotations
    tags: Optional[str] = None
    notes: Optional[str] = None
    is_favorite: bool = False
    
    captured_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Capture':
        """Create Capture instance from database dict."""
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            filename=data['filename'],
            configuration_id=data.get('configuration_id'),
            file_size_bytes=data.get('file_size_bytes'),
            image_width=data.get('image_width'),
            image_height=data.get('image_height'),
            image_data=data.get('image_data'),
            mode=data.get('mode', 'camera'),
            dataset_index=data.get('dataset_index'),
            ransac_time_ms=data.get('ransac_time_ms'),
            fps=data.get('fps'),
            num_ground_pixels=data.get('num_ground_pixels'),
            num_wall_pixels=data.get('num_wall_pixels'),
            num_door_pixels=data.get('num_door_pixels'),
            tags=data.get('tags'),
            notes=data.get('notes'),
            is_favorite=bool(data.get('is_favorite', False)),
            captured_at=data.get('captured_at')
        )
    
    def to_dict(self) -> dict:
        """Convert Capture to dict (for backwards compatibility)."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'configuration_id': self.configuration_id,
            'file_size_bytes': self.file_size_bytes,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_data': self.image_data,
            'mode': self.mode,
            'dataset_index': self.dataset_index,
            'ransac_time_ms': self.ransac_time_ms,
            'fps': self.fps,
            'num_ground_pixels': self.num_ground_pixels,
            'num_wall_pixels': self.num_wall_pixels,
            'num_door_pixels': self.num_door_pixels,
            'tags': self.tags,
            'notes': self.notes,
            'is_favorite': self.is_favorite,
            'captured_at': self.captured_at
        }


__all__ = ['User', 'Configuration', 'Capture']
