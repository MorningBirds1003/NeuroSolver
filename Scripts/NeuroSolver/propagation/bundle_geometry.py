"""
bundle_geometry.py

Bundle/fascicle/fiber placement helpers.

Role
----
- place fibers in transverse space,
- build one single-fiber geometry per placement,
- attach bundle-aware metadata.

This module only assembles geometry and metadata. It does not advance any
membrane, cable, VC, or KNP dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

from Scripts.NeuroSolver.params import DEFAULT_PARAMS, SimulationParameters
from Scripts.NeuroSolver.propagation.myelin_geometry import build_node_internode_geometry


@dataclass(frozen=True)
class FiberPlacement:
    """
    Placement for one fiber in the transverse bundle cross-section.

    Optional diameter overrides allow heterogeneous fibers inside one bundle
    while leaving the shared global preset unchanged.
    """
    fiber_id: int
    center_y_um: float
    center_z_um: float
    fascicle_id: int = 0
    label: str = ""
    fiber_diameter_um: Optional[float] = None
    axon_diameter_um: Optional[float] = None


@dataclass
class BundleGeometry:
    """
    Container for the assembled bundle geometry.

    Stores:
    - one geometry dict per fiber,
    - placement metadata,
    - aggregate extents/counts for bundle-level bookkeeping.
    """
    fibers: Dict[int, Dict[str, np.ndarray | int | float | None]]
    fiber_ids: List[int]
    placements: List[FiberPlacement]
    layout_name: str
    bundle_radius_um: float
    total_fiber_count: int
    total_compartment_count: int
    y_extent_um: Tuple[float, float]
    z_extent_um: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def get_fiber_geometry(self, fiber_id: int) -> Dict[str, np.ndarray | int | float | None]:
        """Return the geometry dict for one fiber by id."""
        return self.fibers[int(fiber_id)]


def _generate_hex(n_fibers: int, spacing_um: float) -> List[Tuple[float, float]]:
    """
    Generate a small deterministic hex-like packing in the transverse plane.

    This is a simple starter layout, not a dense-packing optimizer.
    """
    coords: List[Tuple[float, float]] = [(0.0, 0.0)]
    if n_fibers <= 1:
        return coords[:n_fibers]

    # Use a hex-style y/z step so neighbors remain approximately equally spaced.
    step_y = float(spacing_um)
    step_z = float(spacing_um) * math.sqrt(3.0) / 2.0

    ring = 1
    while len(coords) < n_fibers:
        pts = [
            (+ring * step_y, 0.0),
            (-ring * step_y, 0.0),
            (+0.5 * ring * step_y, +ring * step_z),
            (+0.5 * ring * step_y, -ring * step_z),
            (-0.5 * ring * step_y, +ring * step_z),
            (-0.5 * ring * step_y, -ring * step_z),
        ]
        for yz in pts:
            if len(coords) >= n_fibers:
                break
            if yz not in coords:
                coords.append(yz)
        ring += 1

    return coords[:n_fibers]


def _build_default_placements(
    n_fibers: int,
    spacing_um: float,
    layout_name: str,
) -> List[FiberPlacement]:
    """Build default placements for a named layout."""
    layout = str(layout_name).strip().lower()

    if layout == "linear":
        # Spread fibers evenly along y with z held at zero.
        ys = np.linspace(
            -0.5 * float(spacing_um) * (n_fibers - 1),
            +0.5 * float(spacing_um) * (n_fibers - 1),
            n_fibers,
        )
        return [
            FiberPlacement(
                fiber_id=i,
                center_y_um=float(y),
                center_z_um=0.0,
                fascicle_id=0,
                label=f"fiber_{i}",
            )
            for i, y in enumerate(ys)
        ]

    # Default fallback: compact hex-like packing.
    coords = _generate_hex(n_fibers=n_fibers, spacing_um=spacing_um)
    return [
        FiberPlacement(
            fiber_id=i,
            center_y_um=float(y),
            center_z_um=float(z),
            fascicle_id=0,
            label=f"fiber_{i}",
        )
        for i, (y, z) in enumerate(coords)
    ]


def build_bundle_geometry(
    params: SimulationParameters = DEFAULT_PARAMS,
    n_fibers: int = 1,
    layout_name: str = "hex",
    placements: Optional[Sequence[FiberPlacement]] = None,
    bundle_id: str = "bundle_0",
) -> BundleGeometry:
    """
    Build a bundle by instantiating one single-fiber geometry per placement.

    Each fiber is built explicitly so downstream solvers can stimulate, sample,
    and couple fibers independently instead of scaling one reference template.
    """
    spacing_um = float(getattr(params.geometry, "fiber_spacing_um", 25.0))

    if placements is None:
        placements = _build_default_placements(
            n_fibers=n_fibers,
            spacing_um=spacing_um,
            layout_name=layout_name,
        )

    fibers: Dict[int, Dict[str, np.ndarray | int | float | None]] = {}
    global_start = 0

    for placement in placements:
        # Permit per-fiber diameter overrides without altering the shared preset.
        fiber_params = params
        if placement.fiber_diameter_um is not None or placement.axon_diameter_um is not None:
            geom = params.geometry
            fiber_params = replace(
                params,
                geometry=replace(
                    geom,
                    fiber_diameter_um=float(
                        geom.fiber_diameter_um
                        if placement.fiber_diameter_um is None
                        else placement.fiber_diameter_um
                    ),
                    axon_diameter_um=float(
                        geom.axon_diameter_um
                        if placement.axon_diameter_um is None
                        else placement.axon_diameter_um
                    ),
                ),
            )

        # Build the 1D cable geometry and attach bundle-aware metadata.
        g = build_node_internode_geometry(
            params=fiber_params,
            fiber_id=int(placement.fiber_id),
            fascicle_id=int(placement.fascicle_id),
            fiber_center_y_um=float(placement.center_y_um),
            fiber_center_z_um=float(placement.center_z_um),
            compartment_global_offset=int(global_start),
        )

        # Copy arrays defensively so later code can mutate per-fiber state safely.
        g = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in g.items()
        }

        n = int(g["n_compartments"])
        fibers[int(placement.fiber_id)] = g
        global_start += n

    centers = np.array(
        [[float(p.center_y_um), float(p.center_z_um)] for p in placements],
        dtype=float,
    )

    if len(placements) > 0:
        # Estimate transverse bundle size using center distance plus outer radius.
        radii = np.array(
            [0.5 * float(fibers[int(p.fiber_id)]["outer_diameter_um"][0]) for p in placements],
            dtype=float,
        )
        center_norm = np.sqrt(np.sum(centers ** 2, axis=1))
        bundle_radius = float(np.max(center_norm + radii))
        y_min = float(np.min(centers[:, 0] - radii))
        y_max = float(np.max(centers[:, 0] + radii))
        z_min = float(np.min(centers[:, 1] - radii))
        z_max = float(np.max(centers[:, 1] + radii))
    else:
        bundle_radius = 0.0
        y_min = y_max = 0.0
        z_min = z_max = 0.0

    return BundleGeometry(
        fibers=fibers,
        fiber_ids=[int(p.fiber_id) for p in placements],
        placements=list(placements),
        layout_name=str(layout_name),
        bundle_radius_um=bundle_radius,
        total_fiber_count=len(placements),
        total_compartment_count=int(global_start),
        y_extent_um=(y_min, y_max),
        z_extent_um=(z_min, z_max),
        metadata={
            "bundle_id": str(bundle_id),
            "fiber_spacing_um": float(spacing_um),
        },
    )
