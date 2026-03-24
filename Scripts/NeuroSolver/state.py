"""
state.py

Top-level runtime state containers for NeuroSolver.

Purpose
-------
These dataclasses keep the multirate scheduler organized without forcing each
subsystem to invent its own ad hoc history buffer. This file does not implement
physics; it only stores time histories and packaged outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

@dataclass
class CableHistoryBuffer:
    """
    History buffer for fast cable simulation output.

    Core histories are stored explicitly. Extra histories can be attached
    dynamically through the `extra` argument in `append`.
    """
    t_ms: List[float] = field(default_factory=list)
    V_m_mV: List[np.ndarray] = field(default_factory=list)
    I_ion_uA_per_cm2: List[np.ndarray] = field(default_factory=list)
    I_app_uA_per_cm2: List[np.ndarray] = field(default_factory=list)
    I_app_abs_uA: List[np.ndarray] = field(default_factory=list)

    def append(
        self,
        t_ms: float,
        V_m_mV: np.ndarray,
        I_ion_uA_per_cm2: np.ndarray,
        I_app_uA_per_cm2: np.ndarray,
        I_app_abs_uA: np.ndarray,
        extra: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Append one fast-step snapshot.

        `extra` is intentionally generic so the scheduler can store additional
        histories such as extracellular potential or effective membrane voltage
        without changing this class every time a new diagnostic is added.
        """
        self.t_ms.append(float(t_ms))
        self.V_m_mV.append(np.asarray(V_m_mV, dtype=float).copy())
        self.I_ion_uA_per_cm2.append(np.asarray(I_ion_uA_per_cm2, dtype=float).copy())
        self.I_app_uA_per_cm2.append(np.asarray(I_app_uA_per_cm2, dtype=float).copy())
        self.I_app_abs_uA.append(np.asarray(I_app_abs_uA, dtype=float).copy())

        if extra is not None:
            for key, value in extra.items():
                if not hasattr(self, key):
                    setattr(self, key, [])
                getattr(self, key).append(np.asarray(value, dtype=float).copy())

    def finalize(self, geometry: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert accumulated lists into final arrays.

        The returned payload is shaped for downstream diagnostics and plotting,
        and explicitly aligns the geometry x-axis with the compartment axis of
        the voltage matrix.
        """
        V = np.asarray(self.V_m_mV, dtype=float)
        if V.ndim != 2:
            raise ValueError("CableHistoryBuffer: V_m_mV history has invalid shape.")

        n_compartments = V.shape[1]

        # Align the geometry axis to the actual solver compartment count.
        geom_x = np.asarray(geometry["x_um"], dtype=float)
        x_axis = geom_x[:n_compartments].copy()

        # Region labels may be longer than the voltage axis if geometry keeps
        # additional bookkeeping beyond the currently solved compartment set.
        region_type = np.asarray(geometry.get("region_type", []), dtype=object)
        if region_type.size >= n_compartments:
            region_type = region_type[:n_compartments].copy()
        else:
            region_type = np.full(n_compartments, "unknown", dtype=object)

        result = {
            "t_ms": np.asarray(self.t_ms, dtype=float),
            "V_m_mV": V,
            "I_ion_uA_per_cm2": np.asarray(self.I_ion_uA_per_cm2, dtype=float),
            "I_app_uA_per_cm2": np.asarray(self.I_app_uA_per_cm2, dtype=float),
            "I_app_abs_uA": np.asarray(self.I_app_abs_uA, dtype=float),
            "x_um": x_axis,
            "region_type": region_type,
        }

        # Include any dynamically attached histories as long as they look like
        # recorded time series.
        for key, value in self.__dict__.items():
            if key in result:
                continue
            if key in {"t_ms", "V_m_mV", "I_ion_uA_per_cm2", "I_app_uA_per_cm2", "I_app_abs_uA"}:
                continue
            if isinstance(value, list) and len(value) > 0:
                result[key] = np.asarray(value, dtype=float)

        return result

@dataclass
class KNPHistoryBuffer:
    """
    History buffer for slow 1D KNP output.
    """
    t_ms: List[float] = field(default_factory=list)
    phi_e_mV: List[np.ndarray] = field(default_factory=list)
    species_mM: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def append(
        self,
        t_ms: float,
        phi_e_mV: np.ndarray,
        concentrations_mM: Dict[str, np.ndarray],
    ) -> None:
        """
        Append one slow-step KNP snapshot.
        """
        self.t_ms.append(float(t_ms))
        self.phi_e_mV.append(np.asarray(phi_e_mV, dtype=float).copy())

        if not self.species_mM:
            self.species_mM = {name: [] for name in concentrations_mM.keys()}

        for name, arr in concentrations_mM.items():
            self.species_mM[name].append(np.asarray(arr, dtype=float).copy())

    def finalize(self, x_um: np.ndarray) -> Dict[str, object]:
        """
        Convert KNP histories into a finalized result payload.
        """
        return {
            "t_ms": np.asarray(self.t_ms, dtype=float),
            "x_um": np.asarray(x_um, dtype=float).copy(),
            "phi_e_mV": np.asarray(self.phi_e_mV, dtype=float),
            "species_mM": {
                name: np.asarray(history, dtype=float)
                for name, history in self.species_mM.items()
            },
        }

@dataclass
class MultirateSimulationResult:
    """
    Unified output bundle from the multirate scheduler.
    """
    cable_result: Dict[str, np.ndarray]
    vc_result: Optional[Dict[str, np.ndarray]] = None
    knp_result: Optional[Dict[str, object]] = None
    metadata: Optional[Dict[str, object]] = None
