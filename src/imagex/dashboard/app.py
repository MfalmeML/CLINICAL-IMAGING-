"""Case dashboard. Streamlit. Research prototype only.

Renders a PipelineResult. This module is deliberately render-only: it
does not run inference, does not load models, and does not touch DICOM.
It consumes the JSON output of the pipeline and formats it for a human.

Every rendered page carries the non-clinical framing. That framing is
not optional and is not removable by configuration.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


RESEARCH_PROTOTYPE_FRAMING = (
    "Research prototype. Not for clinical decision-making. "
    "Every output must be reviewed by a qualified radiologist."
)


PRIORITY_COLORS = {
    "URGENT": "#b00020",
    "HIGH": "#d97706",
    "MEDIUM": "#2563eb",
    "LOW": "#374151",
}


def _priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "#374151")
    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:6px;font-weight:600;font-size:14px;">{priority}</span>'
    )


def _render_framing() -> None:
    st.markdown(
        f'<div style="background:#fef3c7;border-left:4px solid #d97706;'
        f'padding:8px 12px;font-size:13px;color:#78350f;">'
        f'{RESEARCH_PROTOTYPE_FRAMING}</div>',
        unsafe_allow_html=True,
    )


def _render_priority(priority: str) -> None:
    st.markdown("### Priority")
    st.markdown(_priority_badge(priority), unsafe_allow_html=True)


def _render_perception(perception: dict[str, Any]) -> None:
    st.markdown("### Perception")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Abnormality score", f"{perception['abnormality_score']:.2f}")
    with col2:
        st.metric("Lesions detected", len(perception.get("lesion_ids", [])))


def _render_temporal(changes: list[dict[str, Any]] | None) -> None:
    st.markdown("### Longitudinal change")
    if changes is None:
        st.info("No prior study available. No change analysis performed.")
        return

    if not changes:
        st.write("No lesions to compare.")
        return

    for c in changes:
        cls = c["change_class"]
        label = c["lesion_id"]
        if cls == "growing":
            st.error(
                f"**{label}**: growing. "
                f"Diameter change: {c['diameter_change_mm']:+.1f} mm. "
                f"Volume change: {c['volume_change_percent']:+.1f}%."
            )
        elif cls == "shrinking":
            st.success(
                f"**{label}**: shrinking. "
                f"Diameter change: {c['diameter_change_mm']:+.1f} mm."
            )
        elif cls == "new":
            st.warning(f"**{label}**: new lesion since prior study.")
        elif cls == "resolved":
            st.info(f"**{label}**: resolved since prior study.")
        else:
            st.write(f"**{label}**: stable.")


def _render_safety(triage_input: dict[str, Any]) -> None:
    st.markdown("### Safety")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Uncertainty", f"{triage_input['uncertainty']:.3f}")
    with col2:
        st.metric("OOD flagged", "yes" if triage_input["is_ood"] else "no")
    with col3:
        rec = triage_input["safety_recommendation"]
        st.metric("Recommendation", rec)

    if triage_input["safety_recommendation"] == "human-review":
        st.warning(
            "Safety engine recommends human review. Priority has been "
            "escalated to URGENT regardless of other signals."
        )


def _render_discrepancy(discrepancy: dict[str, Any] | None) -> None:
    st.markdown("### Second reader")
    if discrepancy is None:
        st.info("No radiologist read available for comparison.")
        return

    rec = discrepancy["recommendation"]
    if rec == "agreement":
        st.success(f"AI and radiologist reads: {rec}.")
    else:
        st.warning(
            f"**{rec}**. "
            f"Kind: {discrepancy['kind']}. "
            f"AI-only findings: {discrepancy['ai_only_lesion_ids']}. "
            f"Radiologist-only findings: {discrepancy['radiologist_only_lesion_ids']}. "
            "This is a discrepancy flag, not a verdict on either read."
        )


def render_case(result: dict[str, Any]) -> None:
    """Render one pipeline result. Extracted so tests can call it directly."""
    _render_framing()
    st.markdown(f"## Case {result.get('case_id', 'unknown')}")
    _render_priority(result["priority"])
    _render_perception(result["perception"])
    _render_temporal(result.get("temporal_changes"))
    _render_safety(result["triage_input"])
    _render_discrepancy(result.get("discrepancy"))


def main() -> None:
    st.set_page_config(
        page_title="IMAGEX case review",
        layout="wide",
    )
    st.title("IMAGEX case review")

    uploaded = st.file_uploader(
        "Upload a pipeline result JSON", type=["json"]
    )
    if uploaded is None:
        st.info("Upload a pipeline result JSON to render a case.")
        _render_framing()
        return

    try:
        result = json.load(uploaded)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return

    render_case(result)


if __name__ == "__main__":
    main()
