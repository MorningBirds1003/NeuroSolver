from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from Scripts.NeuroSolver.baseline_io import load_json, save_json


OUTPUT_ROOT = Path("outputs") / "poc_hh_knp"

def _get_nested(d: Dict[str, Any], *keys, default=np.nan):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")

def _abs_peak_from_min_max(min_val: Any, max_val: Any) -> float:
    a = _as_float(min_val)
    b = _as_float(max_val)
    if not np.isfinite(a) and not np.isfinite(b):
        return float("nan")
    return float(max(abs(a), abs(b)))

def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return load_json(path)

def _case_metrics(case: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if case is None:
        return {
            "recruited_fraction": float("nan"),
            "cnap_like_amplitude_mV": float("nan"),
            "cv_mean_m_per_s": float("nan"),
            "latency_mean_ms": float("nan"),
            "peak_v_m_max_mV": float("nan"),
            "knp_phi_peak_abs_mV": float("nan"),
            "knp_phi_final_min_mV": float("nan"),
            "knp_phi_final_mean_mV": float("nan"),
            "knp_phi_abs_time_integral_mV_ms": float("nan"),
            "knp_source_peak_abs_mM_per_ms": float("nan"),
            "knp_source_abs_time_integral_mM": float("nan"),
            "delta_K_final_max_mM": float("nan"),
            "delta_K_final_mean_mM": float("nan"),
            "delta_K_abs_time_integral_mM_ms": float("nan"),
            "feedback_phi_peak_abs_mV": float("nan"),
            "feedback_phi_mean_mV": float("nan"),
            "vc_peak_E_near_mV": float("nan"),
            "vc_peak_E_far_mV": float("nan"),
            "feedback_enabled": float("nan"),
        }

    summary = case.get("summary", {})
    bundle = case.get("bundle_metrics", {})
    meta = case.get("metadata", {})

    peak_v_m = _as_float(summary.get("peak_v_m_max_mV"))
    if not np.isfinite(peak_v_m):
        peak_by_fiber = summary.get("peak_vm_by_fiber_mV", {})
        if isinstance(peak_by_fiber, dict) and len(peak_by_fiber) > 0:
            try:
                peak_v_m = float(max(float(v) for v in peak_by_fiber.values()))
            except Exception:
                peak_v_m = float("nan")

    peak_vc = summary.get("peak_vc_by_electrode_mV", {})
    vc_near = _as_float(peak_vc.get("E_near")) if isinstance(peak_vc, dict) else float("nan")
    vc_far = _as_float(peak_vc.get("E_far")) if isinstance(peak_vc, dict) else float("nan")

    return {
        "recruited_fraction": _as_float(bundle.get("recruited_fraction")),
        "cnap_like_amplitude_mV": _as_float(bundle.get("cnap_like_amplitude_mV")),
        "cv_mean_m_per_s": _as_float(bundle.get("cv_mean_m_per_s")),
        "latency_mean_ms": _as_float(bundle.get("latency_mean_ms")),
        "peak_v_m_max_mV": peak_v_m,
        "knp_phi_peak_abs_mV": _abs_peak_from_min_max(
            summary.get("knp_phi_min_mV"),
            summary.get("knp_phi_max_mV"),
        ),
        "knp_source_peak_abs_mM_per_ms": _as_float(summary.get("knp_max_abs_source_mM_per_ms")),
        "feedback_phi_peak_abs_mV": _abs_peak_from_min_max(
            summary.get("feedback_phi_min_mV"),
            summary.get("feedback_phi_max_mV"),
        ),
        "feedback_phi_mean_mV": _as_float(summary.get("feedback_phi_mean_mV")),
        "vc_peak_E_near_mV": vc_near,
        "vc_peak_E_far_mV": vc_far,
        "feedback_enabled": float(bool(meta.get("feedback_enabled", summary.get("feedback_enabled", False)))),
        "knp_phi_final_min_mV": _as_float(summary.get("knp_phi_final_min_mV")),
        "knp_phi_final_mean_mV": _as_float(summary.get("knp_phi_final_mean_mV")),
        "knp_phi_abs_time_integral_mV_ms": _as_float(summary.get("knp_phi_abs_time_integral_mV_ms")),
        "knp_source_abs_time_integral_mM": _as_float(summary.get("knp_source_abs_time_integral_mM")),
        "delta_K_final_max_mM": _as_float(summary.get("delta_K_final_max_mM")),
        "delta_K_final_mean_mM": _as_float(summary.get("delta_K_final_mean_mM")),
        "delta_K_abs_time_integral_mM_ms": _as_float(summary.get("delta_K_abs_time_integral_mM_ms")),
    }

def _delta_metrics(on_case: Optional[Dict[str, Any]], off_case: Optional[Dict[str, Any]]) -> Dict[str, float]:
    on = _case_metrics(on_case)
    off = _case_metrics(off_case)
    out: Dict[str, float] = {}

    for key in sorted(on.keys()):
        if key == "feedback_enabled":
            continue
        on_val = on[key]
        off_val = off[key]
        out[f"{key}_delta_on_minus_off"] = on_val - off_val
        out[f"{key}_ratio_on_over_off"] = (
            on_val / off_val
            if np.isfinite(off_val) and abs(off_val) > 0.0
            else float("nan")
        )
    return out

def build_poc_report(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    single = load_json(output_root / "single_fiber_anchor" / "summary.json")

    weak_off = _safe_load(output_root / "three_fiber_weak_feedback_off" / "summary.json")
    weak_on = _safe_load(output_root / "three_fiber_weak_feedback_on" / "summary.json")

    harsh_off = _safe_load(output_root / "three_fiber_harsh_feedback_off" / "summary.json")
    harsh_on = _safe_load(output_root / "three_fiber_harsh_feedback_on" / "summary.json")

    train_off = _safe_load(output_root / "three_fiber_harsh_train_feedback_off" / "summary.json")
    train_on = _safe_load(output_root / "three_fiber_harsh_train_feedback_on" / "summary.json")

    gain5_on = _safe_load(output_root / "three_fiber_harsh_train_gain5_feedback_on" / "summary.json")
    gain10_on = _safe_load(output_root / "three_fiber_harsh_train_gain10_feedback_on" / "summary.json")

    physio_low = _safe_load(output_root / "three_fiber_physio_train_calib_low_source" / "summary.json")
    physio_base = _safe_load(output_root / "three_fiber_physio_train_calib_baseline" / "summary.json")
    physio_high = _safe_load(output_root / "three_fiber_physio_train_calib_high_source" / "summary.json")

    single_peak = _as_float(single.get("peak_voltage_max_mV"))
    single_cv = _as_float(single.get("end_to_end_velocity_m_per_s"))
    single_knp_phi = _as_float(_get_nested(single, "knp_summary", "phi_peak_abs_mV"))
    single_dk = _as_float(_get_nested(single, "knp_summary", "species_max_abs_delta_mM", "K"))
    single_prop = bool(single.get("propagation_success", False))

    weak_off_m = _case_metrics(weak_off)
    weak_on_m = _case_metrics(weak_on)
    harsh_off_m = _case_metrics(harsh_off)
    harsh_on_m = _case_metrics(harsh_on)
    train_off_m = _case_metrics(train_off)
    train_on_m = _case_metrics(train_on)
    gain5_m = _case_metrics(gain5_on)
    gain10_m = _case_metrics(gain10_on)

    physio_low_m = _case_metrics(physio_low)
    physio_base_m = _case_metrics(physio_base)
    physio_high_m = _case_metrics(physio_high)

    report = {
        "claim": (
            "Higher-sensitivity HH + shared-KNP validation: stable fast propagation, "
            "increasing slow ECS state under repeated/harsher loading, explicit "
            "feedback-on vs feedback-off delta measurement, and physiological "
            "short-train source calibration."
        ),
        "single_fiber_anchor": {
            "propagation_success": single_prop,
            "peak_voltage_max_mV": single_peak,
            "end_to_end_velocity_m_per_s": single_cv,
            "knp_phi_peak_abs_mV": single_knp_phi,
            "max_abs_delta_K_mM": single_dk,
        },
        "weak_feedback_off": weak_off_m,
        "weak_feedback_on": weak_on_m,
        "harsh_feedback_off": harsh_off_m,
        "harsh_feedback_on": harsh_on_m,
        "harsh_train_feedback_off": train_off_m,
        "harsh_train_feedback_on": train_on_m,
        "harsh_train_gain5_feedback_on": gain5_m,
        "harsh_train_gain10_feedback_on": gain10_m,
        "physio_train_calib_low_source": physio_low_m,
        "physio_train_calib_baseline": physio_base_m,
        "physio_train_calib_high_source": physio_high_m,
        "feedback_deltas": {
            "weak_on_minus_off": _delta_metrics(weak_on, weak_off),
            "harsh_on_minus_off": _delta_metrics(harsh_on, harsh_off),
            "harsh_train_on_minus_off": _delta_metrics(train_on, train_off),
            "gain5_minus_train_on": _delta_metrics(gain5_on, train_on),
            "gain10_minus_train_on": _delta_metrics(gain10_on, train_on),
        },
        "physio_calibration_deltas": {
            "baseline_minus_low": _delta_metrics(physio_base, physio_low),
            "high_minus_baseline": _delta_metrics(physio_high, physio_base),
            "high_minus_low": _delta_metrics(physio_high, physio_low),
        },
        "slow_domain_scaling": {
            "harsh_vs_weak_knp_phi_ratio_feedback_on": (
                harsh_on_m["knp_phi_peak_abs_mV"] / weak_on_m["knp_phi_peak_abs_mV"]
                if np.isfinite(weak_on_m["knp_phi_peak_abs_mV"]) and weak_on_m["knp_phi_peak_abs_mV"] > 0.0
                else float("nan")
            ),
            "train_vs_harsh_knp_phi_ratio_feedback_on": (
                train_on_m["knp_phi_peak_abs_mV"] / harsh_on_m["knp_phi_peak_abs_mV"]
                if np.isfinite(harsh_on_m["knp_phi_peak_abs_mV"]) and harsh_on_m["knp_phi_peak_abs_mV"] > 0.0
                else float("nan")
            ),
            "gain5_vs_train_feedback_phi_ratio": (
                gain5_m["feedback_phi_peak_abs_mV"] / train_on_m["feedback_phi_peak_abs_mV"]
                if np.isfinite(train_on_m["feedback_phi_peak_abs_mV"]) and train_on_m["feedback_phi_peak_abs_mV"] > 0.0
                else float("nan")
            ),
            "gain10_vs_train_feedback_phi_ratio": (
                gain10_m["feedback_phi_peak_abs_mV"] / gain5_m["feedback_phi_peak_abs_mV"]
                if np.isfinite(gain5_m["feedback_phi_peak_abs_mV"]) and gain5_m["feedback_phi_peak_abs_mV"] > 0.0
                else float("nan")
            ),
            "physio_baseline_vs_low_deltaK_ratio": (
                physio_base_m["delta_K_abs_time_integral_mM_ms"] / physio_low_m["delta_K_abs_time_integral_mM_ms"]
                if np.isfinite(physio_low_m["delta_K_abs_time_integral_mM_ms"]) and physio_low_m["delta_K_abs_time_integral_mM_ms"] > 0.0
                else float("nan")
            ),
            "physio_high_vs_baseline_deltaK_ratio": (
                physio_high_m["delta_K_abs_time_integral_mM_ms"] / physio_base_m["delta_K_abs_time_integral_mM_ms"]
                if np.isfinite(physio_base_m["delta_K_abs_time_integral_mM_ms"]) and physio_base_m["delta_K_abs_time_integral_mM_ms"] > 0.0
                else float("nan")
            ),
            "physio_high_vs_low_feedback_phi_ratio": (
                physio_high_m["feedback_phi_peak_abs_mV"] / physio_low_m["feedback_phi_peak_abs_mV"]
                if np.isfinite(physio_low_m["feedback_phi_peak_abs_mV"]) and physio_low_m["feedback_phi_peak_abs_mV"] > 0.0
                else float("nan")
            ),
        },
        "pass_fail": {
            "single_fiber_propagates": bool(single_prop),
            "single_fiber_cv_finite": bool(np.isfinite(single_cv)),
            "single_fiber_knp_response_nonzero": bool(np.isfinite(single_dk) and single_dk > 0.0),
            "weak_knp_finite": bool(np.isfinite(weak_on_m["knp_phi_peak_abs_mV"])),
            "harsh_knp_exceeds_weak": bool(
                np.isfinite(weak_on_m["knp_phi_peak_abs_mV"])
                and np.isfinite(harsh_on_m["knp_phi_peak_abs_mV"])
                and harsh_on_m["knp_phi_peak_abs_mV"] > weak_on_m["knp_phi_peak_abs_mV"]
            ),
            "train_knp_exceeds_harsh": bool(
                np.isfinite(harsh_on_m["knp_phi_peak_abs_mV"])
                and np.isfinite(train_on_m["knp_phi_peak_abs_mV"])
                and train_on_m["knp_phi_peak_abs_mV"] > harsh_on_m["knp_phi_peak_abs_mV"]
            ),
            "feedback_gain5_exceeds_train_feedback_phi": bool(
                np.isfinite(train_on_m["feedback_phi_peak_abs_mV"])
                and np.isfinite(gain5_m["feedback_phi_peak_abs_mV"])
                and gain5_m["feedback_phi_peak_abs_mV"] > train_on_m["feedback_phi_peak_abs_mV"]
            ),
            "feedback_gain10_exceeds_gain5_feedback_phi": bool(
                np.isfinite(gain5_m["feedback_phi_peak_abs_mV"])
                and np.isfinite(gain10_m["feedback_phi_peak_abs_mV"])
                and gain10_m["feedback_phi_peak_abs_mV"] > gain5_m["feedback_phi_peak_abs_mV"]
            ),
            "physio_low_loaded": physio_low is not None,
            "physio_baseline_loaded": physio_base is not None,
            "physio_high_loaded": physio_high is not None,
            "physio_high_source_exceeds_low_source": bool(
                np.isfinite(physio_low_m["knp_source_peak_abs_mM_per_ms"])
                and np.isfinite(physio_high_m["knp_source_peak_abs_mM_per_ms"])
                and physio_high_m["knp_source_peak_abs_mM_per_ms"] > physio_low_m["knp_source_peak_abs_mM_per_ms"]
            ),
            "physio_high_phi_exceeds_low_phi": bool(
                np.isfinite(physio_low_m["feedback_phi_peak_abs_mV"])
                and np.isfinite(physio_high_m["feedback_phi_peak_abs_mV"])
                and physio_high_m["feedback_phi_peak_abs_mV"] > physio_low_m["feedback_phi_peak_abs_mV"]
            ),
            "physio_runs_keep_fast_metrics_stable": bool(
                np.isfinite(physio_low_m["cv_mean_m_per_s"])
                and np.isfinite(physio_base_m["cv_mean_m_per_s"])
                and np.isfinite(physio_high_m["cv_mean_m_per_s"])
                and abs(physio_high_m["cv_mean_m_per_s"] - physio_low_m["cv_mean_m_per_s"]) < 0.05
                and abs(physio_high_m["latency_mean_ms"] - physio_low_m["latency_mean_ms"]) < 0.05
                and abs(physio_high_m["cnap_like_amplitude_mV"] - physio_low_m["cnap_like_amplitude_mV"]) < 5.0e-4
            ),
        },
    }

    save_json(output_root / "poc_report.json", report)
    return report

def print_poc_report(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("NeuroSolver HH+KNP higher-sensitivity report")
    print("=" * 72)

    print("Single fiber anchor")
    print("-" * 72)
    s = report.get("single_fiber_anchor", {})
    print(f"propagation_success         : {s.get('propagation_success')}")
    print(f"peak_voltage_max_mV        : {s.get('peak_voltage_max_mV')}")
    print(f"end_to_end_velocity_m_per_s: {s.get('end_to_end_velocity_m_per_s')}")
    print(f"knp_phi_peak_abs_mV        : {s.get('knp_phi_peak_abs_mV')}")
    print(f"max_abs_delta_K_mM         : {s.get('max_abs_delta_K_mM')}")

    for name in [
        "weak_feedback_off",
        "weak_feedback_on",
        "harsh_feedback_off",
        "harsh_feedback_on",
        "harsh_train_feedback_off",
        "harsh_train_feedback_on",
        "harsh_train_gain5_feedback_on",
        "harsh_train_gain10_feedback_on",
        "physio_train_calib_low_source",
        "physio_train_calib_baseline",
        "physio_train_calib_high_source",
    ]:
        block = report.get(name, {})
        print("-" * 72)
        print(name)
        print(f"recruited_fraction         : {block.get('recruited_fraction')}")
        print(f"cnap_like_amplitude_mV     : {block.get('cnap_like_amplitude_mV')}")
        print(f"cv_mean_m_per_s            : {block.get('cv_mean_m_per_s')}")
        print(f"latency_mean_ms            : {block.get('latency_mean_ms')}")
        print(f"knp_phi_peak_abs_mV        : {block.get('knp_phi_peak_abs_mV')}")
        print(f"knp_source_peak_abs_mM_per_ms: {block.get('knp_source_peak_abs_mM_per_ms')}")
        print(f"feedback_phi_peak_abs_mV   : {block.get('feedback_phi_peak_abs_mV')}")
        print(f"vc_peak_E_near_mV          : {block.get('vc_peak_E_near_mV')}")
        print(f"vc_peak_E_far_mV           : {block.get('vc_peak_E_far_mV')}")

    print("-" * 72)
    print("Physio calibration deltas")
    for name, block in report.get("physio_calibration_deltas", {}).items():
        print(name)
        print(f"  knp_source_peak_abs_mM_per_ms delta : {block.get('knp_source_peak_abs_mM_per_ms_delta_on_minus_off')}")
        print(f"  feedback_phi_peak_abs_mV delta      : {block.get('feedback_phi_peak_abs_mV_delta_on_minus_off')}")
        print(f"  vc_peak_E_near_mV delta             : {block.get('vc_peak_E_near_mV_delta_on_minus_off')}")
        print(f"  cv_mean_m_per_s delta               : {block.get('cv_mean_m_per_s_delta_on_minus_off')}")

    print("-" * 72)
    print("Scaling ratios")
    for key, value in report.get("slow_domain_scaling", {}).items():
        print(f"{key:44s}: {value}")

    print("-" * 72)
    print("Pass / fail")
    for key, value in report["pass_fail"].items():
        print(f"{key:44s}: {value}")

    print("=" * 72)

if __name__ == "__main__":
    report = build_poc_report()
    print_poc_report(report)