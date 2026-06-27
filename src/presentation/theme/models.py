"""
Color data models used by the GUI theme system.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GUIColors:
    # Base surfaces
    ROOT_BG: str
    SIDEBAR_BG: str
    SIDEBAR_ACTIVE_BG: str
    PANEL_BG: str
    PANEL_NEUTRAL_BG: str
    CONFIG_BG: str
    CARD_BG: str
    LIGHT_BG: str
    FORM_BG: str

    # Neutral controls
    BTN_NEUTRAL_BG: str
    BTN_NEUTRAL_DARK_BG: str
    BTN_LIGHT_HOVER_BG: str
    SLIDER_TROUGH_BG: str

    # Semantic actions
    SUCCESS_BG: str
    SUCCESS_HOVER_BG: str
    SUCCESS_SOFT_BG: str
    SUCCESS_SOFT_HOVER_BG: str
    DANGER_BG: str
    DANGER_HOVER_BG: str
    WARNING_BG: str
    WARNING_HOVER_BG: str
    INFO_BG: str

    # Text
    TEXT_LIGHT: str
    TEXT_DARK: str
    TEXT_MEDIUM: str
    TEXT_MUTED: str
    BLACK: str

