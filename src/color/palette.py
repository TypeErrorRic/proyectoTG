"""
Centralized color palette used by the Tkinter GUI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GUIColors:
    # Base surfaces
    ROOT_BG: str = "#2f2f2f"
    SIDEBAR_BG: str = "#333333"
    SIDEBAR_ACTIVE_BG: str = "#3b3b3b"
    PANEL_BG: str = "#999999"
    PANEL_NEUTRAL_BG: str = "#7f7f7f"
    CONFIG_BG: str = "#cccccc"
    CARD_BG: str = "#b3b3b3"
    LIGHT_BG: str = "#f2f2f2"
    FORM_BG: str = "#d9d9d9"

    # Neutral controls
    BTN_NEUTRAL_BG: str = "#4a4a4a"
    BTN_NEUTRAL_DARK_BG: str = "#5c5c5c"
    BTN_LIGHT_HOVER_BG: str = "#cfcfcf"
    SLIDER_TROUGH_BG: str = "#d5d5d5"

    # Semantic actions
    SUCCESS_BG: str = "#00b86b"
    SUCCESS_HOVER_BG: str = "#21d087"
    SUCCESS_SOFT_BG: str = "#5ee68a"
    SUCCESS_SOFT_HOVER_BG: str = "#80f0a8"
    DANGER_BG: str = "#e53935"
    DANGER_HOVER_BG: str = "#f1625f"
    WARNING_BG: str = "#f2c200"
    WARNING_HOVER_BG: str = "#ffd54f"
    INFO_BG: str = "#1e88e5"

    # Text
    TEXT_LIGHT: str = "white"
    TEXT_DARK: str = "#1f1f1f"
    TEXT_MEDIUM: str = "#333333"
    TEXT_MUTED: str = "#4a4a4a"
    BLACK: str = "black"


GUI_COLORS = GUIColors()
