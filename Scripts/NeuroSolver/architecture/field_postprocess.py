from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
"""
Creation of heatmaps - use as a placeholder for the creation of heatmaps across time based data in poc_hh_knp_cases.py
"""

def plot_phi_e_xt_heatmap(
    field_result: Dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Extracellular potential map",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar_label: str = "phi_e (mV)",
) -> None:
    t_ms = np.asarray(field_result["t_ms"], dtype=float)
    x_um = np.asarray(field_result["x_um"], dtype=float)
    phi = np.asarray(field_result["phi_e_mV"], dtype=float)

    if phi.ndim != 2:
        raise ValueError("field_result['phi_e_mV'] must have shape (T, N_x).")
    if phi.shape[0] != t_ms.size:
        raise ValueError("phi_e_mV time axis does not match t_ms.")
    if phi.shape[1] != x_um.size:
        raise ValueError("phi_e_mV spatial axis does not match x_um.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[field_map] plotting heatmap to {output_path} "
        f"with phi shape={phi.shape}, t={t_ms.size}, x={x_um.size}"
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    im = ax.imshow(
        phi.T,
        aspect="auto",
        origin="lower",
        extent=[float(t_ms[0]), float(t_ms[-1]), float(x_um[0]), float(x_um[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Position x (um)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def plot_delta_phi_xt_heatmap(
    field_result: Dict[str, np.ndarray],
    output_path: str | Path,
    title: str = "Extracellular potential change map",
) -> None:
    phi = np.asarray(field_result["phi_e_mV"], dtype=float)
    phi0 = phi[0:1, :]
    delta_phi = phi - phi0

    absmax = float(np.max(np.abs(delta_phi))) if delta_phi.size > 0 else 1.0

    delta_result = dict(field_result)
    delta_result["phi_e_mV"] = delta_phi
    print(f"[field_map] plotting delta heatmap to {output_path}")
    plot_phi_e_xt_heatmap(
        delta_result,
        output_path=output_path,
        title=title,
        vmin=-absmax,
        vmax=absmax,
        cmap="RdBu_r",
        colorbar_label="delta phi_e (mV)",
    )