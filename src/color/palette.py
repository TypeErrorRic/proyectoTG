"""
Centralized color palette used by the Tkinter GUI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GUIColors:
    # Base surfaces
    ROOT_BG: str = "#2a2f35"
    SIDEBAR_BG: str = "#333941"
    SIDEBAR_ACTIVE_BG: str = "#434a54"
    PANEL_BG: str = "#e2e0db"
    PANEL_NEUTRAL_BG: str = "#989ea5"
    CONFIG_BG: str = "#f3f1ed"
    CARD_BG: str = "#d2cec7"
    LIGHT_BG: str = "#faf8f5"
    FORM_BG: str = "#ece8e1"

    # Neutral controls
    BTN_NEUTRAL_BG: str = "#747b84"
    BTN_NEUTRAL_DARK_BG: str = "#666e78"
    BTN_LIGHT_HOVER_BG: str = "#e2ddd5"
    SLIDER_TROUGH_BG: str = "#cdc8bf"

    # Semantic actions
    SUCCESS_BG: str = "#1f9d64"
    SUCCESS_HOVER_BG: str = "#34b479"
    SUCCESS_SOFT_BG: str = "#6bc799"
    SUCCESS_SOFT_HOVER_BG: str = "#86d4ad"
    DANGER_BG: str = "#d62834"
    DANGER_HOVER_BG: str = "#e64551"
    WARNING_BG: str = "#b47f2f"
    WARNING_HOVER_BG: str = "#c99649"
    INFO_BG: str = "#2f79c8"

    # Text
    TEXT_LIGHT: str = "white"
    TEXT_DARK: str = "#25282d"
    TEXT_MEDIUM: str = "#4a5058"
    TEXT_MUTED: str = "#737982"
    BLACK: str = "black"


GUI_COLORS = GUIColors()
