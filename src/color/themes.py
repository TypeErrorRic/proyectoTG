"""
Named GUI color themes.
"""

from .models import GUIColors


INSTITUTIONAL_COLORS = GUIColors(
    # Base surfaces
    ROOT_BG="#2a2f35",
    SIDEBAR_BG="#333941",
    SIDEBAR_ACTIVE_BG="#434a54",
    PANEL_BG="#e2e0db",
    PANEL_NEUTRAL_BG="#989ea5",
    CONFIG_BG="#f3f1ed",
    CARD_BG="#d2cec7",
    LIGHT_BG="#faf8f5",
    FORM_BG="#ece8e1",
    # Neutral controls
    BTN_NEUTRAL_BG="#747b84",
    BTN_NEUTRAL_DARK_BG="#666e78",
    BTN_LIGHT_HOVER_BG="#e2ddd5",
    SLIDER_TROUGH_BG="#cdc8bf",
    # Semantic actions
    SUCCESS_BG="#1f9d64",
    SUCCESS_HOVER_BG="#34b479",
    SUCCESS_SOFT_BG="#6bc799",
    SUCCESS_SOFT_HOVER_BG="#86d4ad",
    DANGER_BG="#d62834",
    DANGER_HOVER_BG="#e64551",
    WARNING_BG="#b47f2f",
    WARNING_HOVER_BG="#c99649",
    INFO_BG="#2f79c8",
    # Text
    TEXT_LIGHT="white",
    TEXT_DARK="#25282d",
    TEXT_MEDIUM="#4a5058",
    TEXT_MUTED="#737982",
    BLACK="black",
)


# Default theme used by the application.
GUI_COLORS = INSTITUTIONAL_COLORS

