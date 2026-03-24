"""
postprocess_sweep.py

Post-processing utilities for NeuroSolver architecture sweeps.

Reads a sweep_summary.csv file and generates trend plots such as:
- VC amplitude vs fiber count
- VC amplitude vs spacing
- peak Vm spread vs diameter heterogeneity
- VC amplitude vs fascicle separation
- delta K vs fiber count
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## sanity check plots
def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def export_quick_physiology_plots(
    csv_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Generate 3 high-value sanity-check plots from a NeuroSolver sweep CSV:

    1. fiber diameter vs CV
    2. spacing vs CNAP-like amplitude
    3. fiber count vs max_abs_delta_K_mM
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _ensure_numeric(
        df,
        [
            "fiber_diameter_mean_um",
            "cv_mean_m_per_s",
            "spacing_um",
            "cnap_like_amplitude_mV",
            "fiber_count",
            "max_abs_delta_K_mM",
            "cuff_inner_radius_um",
            "cuff_thickness_um",
        ],
    )

    # ------------------------------------------------------------------
    # 1. Fiber diameter vs CV
    # ------------------------------------------------------------------
    d1 = df.dropna(subset=["fiber_diameter_mean_um", "cv_mean_m_per_s"]).copy()
    if len(d1) > 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.scatter(d1["fiber_diameter_mean_um"], d1["cv_mean_m_per_s"])
        ax.set_xlabel("Mean fiber diameter (um)")
        ax.set_ylabel("Mean conduction velocity (m/s)")
        ax.set_title("Fiber diameter vs conduction velocity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "diameter_vs_cv.png", dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Spacing vs CNAP-like amplitude
    # ------------------------------------------------------------------
    d2 = df.dropna(subset=["spacing_um", "cnap_like_amplitude_mV"]).copy()
    if len(d2) > 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.scatter(d2["spacing_um"], d2["cnap_like_amplitude_mV"])

        # Optional simple mean trend by spacing
        grouped = d2.groupby("spacing_um", dropna=True)["cnap_like_amplitude_mV"].mean().reset_index()
        if len(grouped) > 1:
            ax.plot(grouped["spacing_um"], grouped["cnap_like_amplitude_mV"])

        ax.set_xlabel("Fiber spacing (um)")
        ax.set_ylabel("CNAP-like amplitude (mV)")
        ax.set_title("Fiber spacing vs CNAP-like amplitude")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "spacing_vs_cnap.png", dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Fiber count vs max |ΔK|
    # ------------------------------------------------------------------
    d3 = df.dropna(subset=["fiber_count", "max_abs_delta_K_mM"]).copy()
    if len(d3) > 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.scatter(d3["fiber_count"], d3["max_abs_delta_K_mM"])

        grouped = d3.groupby("fiber_count", dropna=True)["max_abs_delta_K_mM"].mean().reset_index()
        if len(grouped) > 1:
            ax.plot(grouped["fiber_count"], grouped["max_abs_delta_K_mM"])

        ax.set_xlabel("Fiber count")
        ax.set_ylabel("Max |ΔK| (mM)")
        ax.set_title("Fiber count vs ECS potassium drift")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fiber_count_vs_deltaK.png", dpi=200)
        plt.close(fig)
# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_sweep_summary(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Sweep summary not found: {path}")
    return pd.read_csv(path)

def save_plot(fig, output_path: str | Path, close: bool = True) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if close:
        plt.close(fig)

# -----------------------------------------------------------------------------
# Small plotting utilities
# -----------------------------------------------------------------------------
def _plot_grouped_lines(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: Optional[str],
    title: str,
    xlabel: str,
    ylabel: str,
):
    fig, ax = plt.subplots()

    work = df.copy()
    work = work.dropna(subset=[x_col, y_col])

    if group_col is None or group_col not in work.columns:
        work = work.sort_values(x_col)
        ax.plot(work[x_col], work[y_col], marker="o")
    else:
        for group_value, sub in work.groupby(group_col, dropna=True):
            sub = sub.sort_values(x_col)
            ax.plot(sub[x_col], sub[y_col], marker="o", label=str(group_value))
        ax.legend(title=group_col)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def _plot_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: Optional[str],
    title: str,
    xlabel: str,
    ylabel: str,
):
    fig, ax = plt.subplots()

    work = df.copy()
    work = work.dropna(subset=[x_col, y_col])

    if color_col is None or color_col not in work.columns:
        ax.scatter(work[x_col], work[y_col])
    else:
        for group_value, sub in work.groupby(color_col, dropna=True):
            ax.scatter(sub[x_col], sub[y_col], label=str(group_value))
        ax.legend(title=color_col)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

# -----------------------------------------------------------------------------
# Metric-specific plot builders
# -----------------------------------------------------------------------------
def plot_vc_vs_fiber_count(df: pd.DataFrame):
    """
    Plot max VC amplitude vs fiber count.
    Group by generator so different architecture families stay distinguishable.
    """
    return _plot_grouped_lines(
        df,
        x_col="fiber_count",
        y_col="max_peak_vc_mV",
        group_col="generator",
        title="VC amplitude vs fiber count",
        xlabel="Fiber count",
        ylabel="Max peak VC (mV)",
    )

def plot_delta_k_vs_fiber_count(df: pd.DataFrame):
    """
    Plot max |delta K| vs fiber count.
    """
    return _plot_grouped_lines(
        df,
        x_col="fiber_count",
        y_col="max_abs_delta_K_mM",
        group_col="generator",
        title="Max |ΔK| vs fiber count",
        xlabel="Fiber count",
        ylabel="Max |ΔK| (mM)",
    )

def plot_vc_vs_spacing(df: pd.DataFrame):
    """
    Plot max VC amplitude vs spacing for layouts that actually define spacing_um.
    """
    return _plot_grouped_lines(
        df,
        x_col="spacing_um",
        y_col="max_peak_vc_mV",
        group_col="generator",
        title="VC amplitude vs spacing",
        xlabel="Spacing (um)",
        ylabel="Max peak VC (mV)",
    )

def plot_vm_spread_vs_diameter_heterogeneity(df: pd.DataFrame):
    """
    Plot peak-Vm spread against fiber diameter standard deviation.
    Vm spread is approximated here by std_peak_vm_mV.
    """
    return _plot_scatter(
        df,
        x_col="fiber_diameter_std_um",
        y_col="std_peak_vm_mV",
        color_col="generator",
        title="Peak Vm spread vs diameter heterogeneity",
        xlabel="Fiber diameter std (um)",
        ylabel="Peak Vm std across fibers (mV)",
    )

def plot_vc_vs_fascicle_separation(df: pd.DataFrame):
    """
    Plot max VC amplitude vs fascicle separation for two-fascicle cases.
    """
    return _plot_grouped_lines(
        df,
        x_col="fascicle_separation_um",
        y_col="max_peak_vc_mV",
        group_col="intra_fascicle_spacing_um",
        title="VC amplitude vs fascicle separation",
        xlabel="Fascicle separation (um)",
        ylabel="Max peak VC (mV)",
    )

def plot_knp_source_vs_fiber_count(df: pd.DataFrame):
    """
    Plot KNP source magnitude vs fiber count.
    """
    return _plot_grouped_lines(
        df,
        x_col="fiber_count",
        y_col="knp_max_abs_source_mM_per_ms",
        group_col="generator",
        title="KNP source magnitude vs fiber count",
        xlabel="Fiber count",
        ylabel="Max |source| (mM/ms)",
    )

def plot_cnap_like_amplitude_vs_fiber_count(df: pd.DataFrame):
    """
    Plot CNAP-like amplitude proxy vs fiber count.
    """
    return _plot_grouped_lines(
        df,
        x_col="fiber_count",
        y_col="cnap_like_amplitude_mV",
        group_col="generator",
        title="CNAP-like amplitude vs fiber count",
        xlabel="Fiber count",
        ylabel="CNAP-like amplitude (mV)",
    )

def plot_latency_spread_vs_spacing(df: pd.DataFrame):
    """
    Plot latency spread vs spacing for layouts that define spacing_um.
    """
    return _plot_grouped_lines(
        df,
        x_col="spacing_um",
        y_col="latency_spread_ms",
        group_col="generator",
        title="Latency spread vs spacing",
        xlabel="Spacing (um)",
        ylabel="Latency spread (ms)",
    )

def plot_cv_vs_diameter_heterogeneity(df: pd.DataFrame):
    """
    Plot mean conduction velocity vs fiber diameter heterogeneity.
    """
    return _plot_scatter(
        df,
        x_col="fiber_diameter_std_um",
        y_col="cv_mean_m_per_s",
        color_col="generator",
        title="Conduction velocity vs diameter heterogeneity",
        xlabel="Fiber diameter std (um)",
        ylabel="Mean conduction velocity (m/s)",
    )

def plot_vc_vs_cuff_inner_radius(df: pd.DataFrame):
    """
    Plot max VC amplitude vs cuff inner radius.
    """
    return _plot_grouped_lines(
        df,
        x_col="cuff_inner_radius_um",
        y_col="max_peak_vc_mV",
        group_col="cuff_thickness_um",
        title="VC amplitude vs cuff inner radius",
        xlabel="Cuff inner radius (um)",
        ylabel="Max peak VC (mV)",
    )

def plot_cnap_like_amplitude_vs_cuff_thickness(df: pd.DataFrame):
    """
    Plot CNAP-like amplitude proxy vs cuff thickness.
    """
    return _plot_grouped_lines(
        df,
        x_col="cuff_thickness_um",
        y_col="cnap_like_amplitude_mV",
        group_col="cuff_inner_radius_um",
        title="CNAP-like amplitude vs cuff thickness",
        xlabel="Cuff thickness (um)",
        ylabel="CNAP-like amplitude (mV)",
    )

def plot_latency_spread_vs_cuff_inner_radius(df: pd.DataFrame):
    """
    Plot latency spread vs cuff inner radius.
    """
    return _plot_grouped_lines(
        df,
        x_col="cuff_inner_radius_um",
        y_col="latency_spread_ms",
        group_col="cuff_thickness_um",
        title="Latency spread vs cuff inner radius",
        xlabel="Cuff inner radius (um)",
        ylabel="Latency spread (ms)",
    )
# -----------------------------------------------------------------------------
# Batch exporter
# -----------------------------------------------------------------------------
def export_standard_sweep_plots(
    csv_path: str | Path,
    output_dir: str | Path = "outputs/architecture_sweep_post",
) -> pd.DataFrame:
    """
    Read sweep_summary.csv and export the standard post-processing plots.
    """
    df = load_sweep_summary(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All rows
    fig, _ = plot_vc_vs_fiber_count(df)
    save_plot(fig, output_dir / "vc_vs_fiber_count.png")

    fig, _ = plot_delta_k_vs_fiber_count(df)
    save_plot(fig, output_dir / "deltaK_vs_fiber_count.png")

    fig, _ = plot_knp_source_vs_fiber_count(df)
    save_plot(fig, output_dir / "knp_source_vs_fiber_count.png")

    # New bundle-level plots if metrics columns are present
    if "cnap_like_amplitude_mV" in df.columns:
        cnap_df = df[df["cnap_like_amplitude_mV"].notna()].copy()
        if not cnap_df.empty:
            fig, _ = plot_cnap_like_amplitude_vs_fiber_count(cnap_df)
            save_plot(fig, output_dir / "cnap_like_amplitude_vs_fiber_count.png")

    if "latency_spread_ms" in df.columns and "spacing_um" in df.columns:
        latency_df = df[
            df["latency_spread_ms"].notna() & df["spacing_um"].notna()
        ].copy()
        if not latency_df.empty:
            fig, _ = plot_latency_spread_vs_spacing(latency_df)
            save_plot(fig, output_dir / "latency_spread_vs_spacing.png")

    if "cv_mean_m_per_s" in df.columns and "fiber_diameter_std_um" in df.columns:
        cv_df = df[
            df["cv_mean_m_per_s"].notna() & df["fiber_diameter_std_um"].notna()
        ].copy()
        if not cv_df.empty:
            fig, _ = plot_cv_vs_diameter_heterogeneity(cv_df)
            save_plot(fig, output_dir / "cv_vs_diameter_heterogeneity.png")

    # Rows with spacing
    spacing_df = (
        df[df["spacing_um"].notna()].copy()
        if "spacing_um" in df.columns else df.iloc[0:0]
    )
    if not spacing_df.empty:
        fig, _ = plot_vc_vs_spacing(spacing_df)
        save_plot(fig, output_dir / "vc_vs_spacing.png")

    # Rows with diameter heterogeneity info
    diam_df = (
        df[df["fiber_diameter_std_um"].notna()].copy()
        if "fiber_diameter_std_um" in df.columns else df.iloc[0:0]
    )
    if not diam_df.empty:
        fig, _ = plot_vm_spread_vs_diameter_heterogeneity(diam_df)
        save_plot(fig, output_dir / "vm_spread_vs_diameter_heterogeneity.png")

    # Rows with two-fascicle info
    fasc_df = (
        df[df["fascicle_separation_um"].notna()].copy()
        if "fascicle_separation_um" in df.columns else df.iloc[0:0]
    )
    if not fasc_df.empty:
        fig, _ = plot_vc_vs_fascicle_separation(fasc_df)
        save_plot(fig, output_dir / "vc_vs_fascicle_separation.png")

    # New cuff-specific plots
    cuff_cols_present = all(
        col in df.columns
        for col in ["include_cuff", "cuff_inner_radius_um", "cuff_thickness_um"]
    )

    if cuff_cols_present:
        cuff_df = df[
            df["include_cuff"].astype(bool)
            & df["cuff_inner_radius_um"].notna()
            & df["cuff_thickness_um"].notna()
        ].copy()

        if not cuff_df.empty:
            fig, _ = plot_vc_vs_cuff_inner_radius(cuff_df)
            save_plot(fig, output_dir / "vc_vs_cuff_inner_radius.png")

            if "cnap_like_amplitude_mV" in cuff_df.columns:
                cnap_cuff_df = cuff_df[cuff_df["cnap_like_amplitude_mV"].notna()].copy()
                if not cnap_cuff_df.empty:
                    fig, _ = plot_cnap_like_amplitude_vs_cuff_thickness(cnap_cuff_df)
                    save_plot(fig, output_dir / "cnap_like_amplitude_vs_cuff_thickness.png")

            if "latency_spread_ms" in cuff_df.columns:
                latency_cuff_df = cuff_df[cuff_df["latency_spread_ms"].notna()].copy()
                if not latency_cuff_df.empty:
                    fig, _ = plot_latency_spread_vs_cuff_inner_radius(latency_cuff_df)
                    save_plot(fig, output_dir / "latency_spread_vs_cuff_inner_radius.png")

    return df

def main() -> None:
    csv_path = "outputs/architecture_sweep_rich/sweep_summary.csv"
    output_dir = "outputs/architecture_sweep_rich/postprocess"
    df = export_standard_sweep_plots(csv_path, output_dir)
    print(f"Processed {len(df)} sweep rows.")
    print(f"Wrote plots to: {Path(output_dir).resolve()}")

if __name__ == "__main__":
    main()