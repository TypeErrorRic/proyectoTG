"""
Backwards-compatible palette exports.

Use `src.presentation.theme.themes` for named themes and `src.presentation.theme.models` for the data
model. This module is kept to avoid breaking existing imports.
"""

from .models import GUIColors
from .themes import GUI_COLORS
