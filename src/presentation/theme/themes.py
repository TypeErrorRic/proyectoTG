"""
Named GUI color themes.
"""

from .models import GUIColors


INSTITUTIONAL_COLORS = GUIColors(
    # Base surfaces
    ROOT_BG="#0F2948",
    SIDEBAR_BG="#1A3A5E",
    SIDEBAR_ACTIVE_BG="#2D567F",
    PANEL_BG="#F3F6FA",
    PANEL_NEUTRAL_BG="#759EA3",
    CONFIG_BG="#FFFFFF",
    CARD_BG="#E2E8F0",
    LIGHT_BG="#FFFFFF",
    FORM_BG="#EEF3F8",
    # Neutral controls
    BTN_NEUTRAL_BG="#677381",
    BTN_NEUTRAL_DARK_BG="#586473",
    BTN_LIGHT_HOVER_BG="#DCE4ED",
    SLIDER_TROUGH_BG="#C5CED9",
    # Semantic actions
    SUCCESS_BG="#2D6E5F",
    SUCCESS_HOVER_BG="#255D50",
    SUCCESS_SOFT_BG="#3D8171",
    SUCCESS_SOFT_HOVER_BG="#336F61",
    DANGER_BG="#A44557",
    DANGER_HOVER_BG="#8F394A",
    WARNING_BG="#9E6B2F",
    WARNING_HOVER_BG="#8F5E24",
    INFO_BG="#3A679A",
    # Text
    TEXT_LIGHT="white",
    TEXT_DARK="#202327",
    TEXT_MEDIUM="#132030",
    TEXT_MUTED="#1F2D3A",
    BLACK="black",
)


# Default theme used by the application.
GUI_COLORS = INSTITUTIONAL_COLORS
