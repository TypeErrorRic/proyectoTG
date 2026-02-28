"""
Backwards-compatible palette exports.

Use `src.color.themes` for named themes and `src.color.models` for the data
model. This module is kept to avoid breaking existing imports.
"""

from .models import GUIColors
from .themes import GUI_COLORS
