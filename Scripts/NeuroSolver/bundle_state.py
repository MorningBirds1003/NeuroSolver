"""
bundle_state.py

Per-fiber runtime state containers for NeuroSolver Phase 1.

Purpose
-------
This module makes bundle ownership explicit. It separates:
- per-fiber fast membrane/cable state,
- shared ECS state,
- shared VC state,
- shared material state,
- bundle-level diagnostics metadata.

That separation is the minimum architectural step needed before a true
shared-medium multi-fiber scheduler can scale cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters
from Scripts.NeuroSolver.propagation.bundle_geometry import BundleGeometry
from Scripts.NeuroSolver.propagation.cable_solver import initialize_cable_state

@dataclass
class BundleRuntimeState:
    """
    Bundle-level runtime state container.

    Notes
    -----
    fiber_states
        Maps fiber_id -> single-fiber cable state dict.

    shared_*_state
        Placeholders for subsystems that are shared across all fibers in a
        bundle, such as a common extracellular medium or shared material map.
    """
    fiber_states: Dict[int, Dict[str, np.ndarray]]
    shared_ecs_state: Dict[str, object] = field(default_factory=dict)
    shared_vc_state: Dict[str, object] = field(default_factory=dict)
    shared_material_state: Dict[str, object] = field(default_factory=dict)
    diagnostics_state: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass
class BundleHistoryBuffer:
    """
    Minimal Phase 1 history buffer for explicit per-fiber voltage state.

    This intentionally stores only time and per-fiber membrane voltage history.
    It is a lightweight bridge between "just run the fibers" and fuller
    bundle-level diagnostics.
    """
    t_ms: list[float] = field(default_factory=list)
    fiber_voltage_history: Dict[int, list[np.ndarray]] = field(default_factory=dict)

    def append(self, t_ms: float, bundle_state: BundleRuntimeState) -> None:
        """
        Append one fast-time snapshot from the current bundle runtime state.
        """
        self.t_ms.append(float(t_ms))
        for fiber_id, fiber_state in bundle_state.fiber_states.items():
            self.fiber_voltage_history.setdefault(int(fiber_id), []).append(
                np.asarray(fiber_state["V_m_mV"], dtype=float).copy()
            )

    def finalize(self) -> Dict[str, object]:
        """
        Convert internal Python lists into NumPy arrays for downstream use.
        """
        return {
            "t_ms": np.asarray(self.t_ms, dtype=float),
            "fiber_voltage_history": {
                fiber_id: np.asarray(history, dtype=float)
                for fiber_id, history in self.fiber_voltage_history.items()
            },
        }

def initialize_bundle_state(
    bundle_geometry: BundleGeometry,
    params: SimulationParameters = DEFAULT_PARAMS,
    fiber_state_initializer: Optional[
        Callable[[Dict[str, object], SimulationParameters], Dict[str, np.ndarray]]
    ] = None,
) -> BundleRuntimeState:
    """
    Instantiate one fast cable state per explicit fiber geometry.

    Design note
    -----------
    This function does not merge fibers and does not attempt interaction logic.
    It only makes state ownership explicit so later schedulers can batch,
    superpose, and couple shared subsystems in a controlled way.
    """
    if fiber_state_initializer is None:
        fiber_state_initializer = initialize_cable_state

    fiber_states: Dict[int, Dict[str, np.ndarray]] = {}
    resting_vm_means: Dict[int, float] = {}

    for fiber_id in bundle_geometry.fiber_ids:
        geometry = bundle_geometry.get_fiber_geometry(fiber_id)
        state = fiber_state_initializer(geometry=geometry, params=params)
        fiber_states[int(fiber_id)] = state
        resting_vm_means[int(fiber_id)] = float(np.mean(np.asarray(state["V_m_mV"], dtype=float)))

    return BundleRuntimeState(
        fiber_states=fiber_states,
        metadata={
            "bundle_id": bundle_geometry.metadata.get("bundle_id", "bundle_0"),
            "layout_name": bundle_geometry.layout_name,
            "total_fiber_count": int(bundle_geometry.total_fiber_count),
            "total_compartment_count": int(bundle_geometry.total_compartment_count),
            "fiber_ids": [int(fid) for fid in bundle_geometry.fiber_ids],
            "resting_vm_mean_by_fiber_mV": resting_vm_means,
        },
    )
