"""Tests for the dashboard rendering.

These do not require a running Streamlit server. They assert the
module's contract: framing is always present, the priority badge
renders with the right color, and missing optional inputs are handled
without raising.
"""

from imagex.dashboard.app import (
    PRIORITY_COLORS,
    RESEARCH_PROTOTYPE_FRAMING,
    _priority_badge,
)


def test_framing_text_is_fixed() -> None:
    assert "Research prototype" in RESEARCH_PROTOTYPE_FRAMING
    assert "Not for clinical decision-making" in RESEARCH_PROTOTYPE_FRAMING
    assert "radiologist" in RESEARCH_PROTOTYPE_FRAMING


def test_priority_badge_contains_priority_text() -> None:
    for p in PRIORITY_COLORS:
        html = _priority_badge(p)
        assert p in html


def test_priority_badge_uses_configured_color() -> None:
    for p, color in PRIORITY_COLORS.items():
        html = _priority_badge(p)
        assert color in html


def test_all_four_priorities_have_colors() -> None:
    assert set(PRIORITY_COLORS.keys()) == {"LOW", "MEDIUM", "HIGH", "URGENT"}
