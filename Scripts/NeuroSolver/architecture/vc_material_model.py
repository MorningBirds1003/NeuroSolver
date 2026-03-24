"""
vc_material_model.py

Material-aware shielding model for NeuroSolver VC calculations.

This preserves the current point-source superposition structure, but replaces
the single homogeneous conductivity with a pairwise effective conductivity
computed across concentric cylindrical material layers.

Intended use:
- endoneurium
- perineurium
- epineurium
- cuff polymer
- saline

This is not a FEM solver. It is a lightweight bridge between the current
homogeneous VC approximation and a future full material/field model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

@dataclass(frozen=True)
class CylindricalLayer:
    name: str
    r_inner_um: float
    r_outer_um: float
    conductivity_S_per_m: float

@dataclass(frozen=True)
class RadialMaterialModel:
    """
    Concentric cylindrical material model centered at (center_y_um, center_z_um).
    """
    center_y_um: float
    center_z_um: float
    layers: Sequence[CylindricalLayer]

    def radial_distance_um(self, y_um: float, z_um: float) -> float:
        dy = float(y_um) - float(self.center_y_um)
        dz = float(z_um) - float(self.center_z_um)
        return float(np.sqrt(dy * dy + dz * dz))

    def conductivity_at_radius(self, r_um: float) -> float:
        for layer in self.layers:
            if layer.r_inner_um <= r_um <= layer.r_outer_um:
                return float(layer.conductivity_S_per_m)

        # Outside defined layers: use outermost layer conductivity
        if len(self.layers) == 0:
            raise ValueError("RadialMaterialModel.layers must not be empty.")
        return float(self.layers[-1].conductivity_S_per_m)

    def effective_conductivity_between_radii(
        self,
        r0_um: float,
        r1_um: float,
        min_sigma_S_per_m: float = 1.0e-6,
    ) -> float:
        """
        Compute harmonic-mean conductivity across the radial interval [r0, r1].

        This approximates layered radial transport resistance without a full field solve.
        """
        a_um = float(min(r0_um, r1_um))
        b_um = float(max(r0_um, r1_um))

        if np.isclose(a_um, b_um):
            return float(self.conductivity_at_radius(a_um))

        total_dr_um = b_um - a_um
        if total_dr_um <= 0.0:
            return float(np.nan)

        resistance_sum = 0.0

        for layer in self.layers:
            left = max(a_um, float(layer.r_inner_um))
            right = min(b_um, float(layer.r_outer_um))
            if right <= left:
                continue

            dr_um = right - left
            sigma = max(float(layer.conductivity_S_per_m), float(min_sigma_S_per_m))
            resistance_sum += dr_um / sigma

        # If interval lies partly outside layer bounds, use the outermost conductivity
        covered_dr_um = 0.0
        for layer in self.layers:
            left = max(a_um, float(layer.r_inner_um))
            right = min(b_um, float(layer.r_outer_um))
            if right > left:
                covered_dr_um += (right - left)

        uncovered_dr_um = max(0.0, total_dr_um - covered_dr_um)
        if uncovered_dr_um > 0.0:
            sigma_outer = max(float(self.layers[-1].conductivity_S_per_m), float(min_sigma_S_per_m))
            resistance_sum += uncovered_dr_um / sigma_outer

        if resistance_sum <= 0.0:
            return float(np.nan)

        sigma_eff = total_dr_um / resistance_sum
        return float(sigma_eff)

    def effective_conductivity_between_points(
        self,
        source_y_um: float,
        source_z_um: float,
        electrode_y_um: float,
        electrode_z_um: float,
    ) -> float:
        rs_um = self.radial_distance_um(source_y_um, source_z_um)
        re_um = self.radial_distance_um(electrode_y_um, electrode_z_um)
        return float(self.effective_conductivity_between_radii(rs_um, re_um))

def build_default_peripheral_nerve_material_model(
    *,
    fascicle_radius_um: float,
    outer_nerve_radius_um: float,
    cuff_inner_radius_um: Optional[float] = None,
    cuff_thickness_um: float = 0.0,
    center_y_um: float = 0.0,
    center_z_um: float = 0.0,
    sigma_endoneurium_S_per_m: float = 0.60,
    sigma_perineurium_S_per_m: float = 0.02,
    sigma_epineurium_S_per_m: float = 0.08,
    sigma_cuff_polymer_S_per_m: float = 1.0e-8,
    sigma_saline_S_per_m: float = 1.50,
    perineurium_thickness_um: float = 8.0,
) -> RadialMaterialModel:
    """
    Build a pragmatic concentric-layer nerve/cuff material model.

    Geometry ordering:
    1. endoneurium
    2. perineurium
    3. epineurium
    4. cuff polymer (optional)
    5. saline
    """
    layers: List[CylindricalLayer] = []

    r_endo_in = 0.0
    r_endo_out = max(0.0, float(fascicle_radius_um) - float(perineurium_thickness_um))
    r_peri_in = r_endo_out
    r_peri_out = float(fascicle_radius_um)
    r_epi_in = r_peri_out
    r_epi_out = float(outer_nerve_radius_um)

    layers.append(
        CylindricalLayer(
            name="endoneurium",
            r_inner_um=r_endo_in,
            r_outer_um=r_endo_out,
            conductivity_S_per_m=float(sigma_endoneurium_S_per_m),
        )
    )
    layers.append(
        CylindricalLayer(
            name="perineurium",
            r_inner_um=r_peri_in,
            r_outer_um=r_peri_out,
            conductivity_S_per_m=float(sigma_perineurium_S_per_m),
        )
    )
    layers.append(
        CylindricalLayer(
            name="epineurium",
            r_inner_um=r_epi_in,
            r_outer_um=r_epi_out,
            conductivity_S_per_m=float(sigma_epineurium_S_per_m),
        )
    )

    if cuff_inner_radius_um is not None and cuff_thickness_um > 0.0:
        cuff_inner = float(cuff_inner_radius_um)
        cuff_outer = float(cuff_inner_radius_um) + float(cuff_thickness_um)
        layers.append(
            CylindricalLayer(
                name="cuff_polymer",
                r_inner_um=cuff_inner,
                r_outer_um=cuff_outer,
                conductivity_S_per_m=float(sigma_cuff_polymer_S_per_m),
            )
        )
        saline_inner = cuff_outer
    else:
        saline_inner = float(outer_nerve_radius_um)

    layers.append(
        CylindricalLayer(
            name="saline",
            r_inner_um=float(saline_inner),
            r_outer_um=1.0e9,
            conductivity_S_per_m=float(sigma_saline_S_per_m),
        )
    )

    return RadialMaterialModel(
        center_y_um=float(center_y_um),
        center_z_um=float(center_z_um),
        layers=layers,
    )