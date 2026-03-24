from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from Scripts.NeuroSolver.baseline_io import load_json


OUTPUT_ROOT = Path("outputs") / "poc_hh_knp"

def _fmt_float(value: Any, digits: int = 6, sci_small: bool = True) -> str:
    try:
        x = float(value)
    except Exception:
        return "NA"

    if not np.isfinite(x):
        return "NA"

    ax = abs(x)
    if sci_small and ((0.0 < ax < 1.0e-3) or ax >= 1.0e3):
        return f"{x:.3e}"
    return f"{x:.{digits}f}"

def _fmt_bool(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"

def build_markdown_from_report(report: Dict[str, Any]) -> str:
    s = report.get("single_fiber_anchor", {})
    w = report.get("three_fiber_weak_regime", {})
    h = report.get("three_fiber_harsh_ecs", {})
    c = report.get("comparisons", {})
    p = report.get("pass_fail", {})
    claim = report.get("claim", "HH + KNP proof-of-concept summary.")

    lines = []
    lines.append("# NeuroSolver HH+KNP Proof-of-Concept Summary")
    lines.append("")
    lines.append("## Claim")
    lines.append("")
    lines.append(claim)
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This run package tests the minimal claim that the fast HH/cable layer can "
        "propagate stably while the shared KNP/ECS layer remains finite, responds to "
        "activity, and scales upward under a harsher extracellular regime."
    )
    lines.append("")
    lines.append("## Case 1 — Single-fiber anchor")
    lines.append("")
    lines.append(f"- Propagation success: **{_fmt_bool(s.get('propagation_success'))}**")
    lines.append(f"- Peak membrane voltage: **{_fmt_float(s.get('peak_voltage_max_mV'))} mV**")
    lines.append(f"- End-to-end velocity: **{_fmt_float(s.get('end_to_end_velocity_m_per_s'))} m/s**")
    lines.append(f"- KNP |phi_e| peak: **{_fmt_float(s.get('knp_phi_peak_abs_mV'))} mV**")
    lines.append(f"- Max |delta K|: **{_fmt_float(s.get('max_abs_delta_K_mM'))} mM**")
    lines.append("")
    lines.append("## Case 2 — Three-fiber weak regime")
    lines.append("")
    lines.append(f"- Recruited fraction: **{_fmt_float(w.get('recruited_fraction'))}**")
    lines.append(f"- CNAP-like amplitude: **{_fmt_float(w.get('cnap_like_amplitude_mV'))} mV**")
    lines.append(f"- KNP |phi_e| peak: **{_fmt_float(w.get('knp_phi_peak_abs_mV'))} mV**")
    lines.append(f"- KNP max source loading: **{_fmt_float(w.get('knp_max_abs_source_mM_per_ms'))} mM/ms**")
    lines.append("")
    lines.append("## Case 3 — Three-fiber harsh ECS regime")
    lines.append("")
    lines.append(f"- Recruited fraction: **{_fmt_float(h.get('recruited_fraction'))}**")
    lines.append(f"- CNAP-like amplitude: **{_fmt_float(h.get('cnap_like_amplitude_mV'))} mV**")
    lines.append(f"- KNP |phi_e| peak: **{_fmt_float(h.get('knp_phi_peak_abs_mV'))} mV**")
    lines.append(f"- KNP max source loading: **{_fmt_float(h.get('knp_max_abs_source_mM_per_ms'))} mM/ms**")
    lines.append("")
    lines.append("## Weak vs harsh comparison")
    lines.append("")
    lines.append(f"- Harsh / weak KNP |phi_e| ratio: **{_fmt_float(c.get('harsh_vs_weak_knp_phi_ratio'))}**")
    lines.append(f"- Harsh / weak KNP source ratio: **{_fmt_float(c.get('harsh_vs_weak_knp_source_ratio'))}**")
    lines.append(f"- Harsh − weak CNAP-like amplitude: **{_fmt_float(c.get('harsh_minus_weak_cnap_like_amplitude_mV'))} mV**")
    lines.append(f"- Harsh − weak recruited fraction: **{_fmt_float(c.get('harsh_minus_weak_recruited_fraction'))}**")
    lines.append("")
    lines.append("## Pass / fail checks")
    lines.append("")
    for key, value in p.items():
        pretty = key.replace("_", " ")
        lines.append(f"- {pretty}: **{_fmt_bool(value)}**")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    if all(bool(v) for v in p.values()):
        lines.append(
            "The minimal HH+KNP proof of concept passes: fast propagation is stable, "
            "the slow shared-domain response is finite and nonzero, and the harsher ECS "
            "case increases slow-domain signal relative to the weak regime."
        )
    else:
        lines.append(
            "The proof of concept is only partially satisfied. At least one required "
            "stability or scaling check failed, so the run set should be treated as a "
            "debugging baseline rather than a completed validation package."
        )
    lines.append("")
    lines.append("## Output files to inspect")
    lines.append("")
    lines.append("- `single_fiber_anchor/summary.json`")
    lines.append("- `three_fiber_weak_regime/summary.json`")
    lines.append("- `three_fiber_harsh_ecs/summary.json`")
    lines.append("- `poc_report.json`")
    lines.append("- Architecture figures and KNP heatmaps inside each case folder")
    lines.append("")
    return "\n".join(lines)

def write_markdown_summary(output_root: Path = OUTPUT_ROOT) -> Path:
    report = load_json(output_root / "poc_report.json")
    markdown = build_markdown_from_report(report)

    outpath = output_root / "poc_summary.md"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(markdown, encoding="utf-8")
    return outpath

if __name__ == "__main__":
    outpath = write_markdown_summary()
    print(f"Wrote markdown summary to: {outpath.resolve()}")