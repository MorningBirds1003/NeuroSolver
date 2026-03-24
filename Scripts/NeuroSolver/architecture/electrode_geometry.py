"""
electrode_geometry.py

Translate declarative electrode specs into VC sample points compatible with the
existing homogeneous-medium VC solver.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Optional

from Scripts.NeuroSolver.ECS.vc_solver import ElectrodeSamplePoint
from Scripts.NeuroSolver.architecture.architecture_schema import (
    CuffSpec,
    ElectrodeSpec,
    NerveArchitectureSpec,
)

def build_electrode_points_from_spec(
    electrode_specs: Sequence[ElectrodeSpec],
    ring_min_points: int = 8,
) -> List[ElectrodeSamplePoint]:
    """
    Convert high-level electrode descriptions into explicit sample points.

    Parameters
    ----------
    electrode_specs:
        Sequence of ElectrodeSpec objects.
    ring_min_points:
        Minimum number of sample points used to approximate a ring contact.
    """
    points: List[ElectrodeSamplePoint] = []

    for idx, spec in enumerate(electrode_specs):
        label = spec.label or f"electrode_{idx}"

        if spec.kind == "point":
            points.append(
                ElectrodeSamplePoint(
                    name=label,
                    x_um=float(spec.x_um),
                    y_um=float(spec.y_um),
                    z_um=float(spec.z_um),
                )
            )
            continue

        if spec.kind == "ring":
            if spec.radius_um is None or float(spec.radius_um) <= 0.0:
                raise ValueError("Ring electrodes require a positive radius_um.")

            point_count = max(int(spec.contact_count), int(ring_min_points))
            radius_um = float(spec.radius_um)
            for k in range(point_count):
                theta = 2.0 * math.pi * float(k) / float(point_count)
                points.append(
                    ElectrodeSamplePoint(
                        name=f"{label}_{k}",
                        x_um=float(spec.x_um),
                        y_um=float(spec.y_um) + radius_um * math.cos(theta),
                        z_um=float(spec.z_um) + radius_um * math.sin(theta),
                    )
                )
            continue

        if spec.kind == "multicontact":
            count = max(1, int(spec.contact_count))
            spacing_um = float(spec.spacing_um if spec.spacing_um is not None else 100.0)
            x0 = float(spec.x_um)
            for k in range(count):
                points.append(
                    ElectrodeSamplePoint(
                        name=f"{label}_{k}",
                        x_um=x0 + float(k) * spacing_um,
                        y_um=float(spec.y_um),
                        z_um=float(spec.z_um),
                    )
                )
            continue

        raise ValueError(f"Unsupported electrode kind: {spec.kind!r}")

    return points

def build_ring_cuff_electrode_points(
    cuff: CuffSpec,
    *,
    x_um: float = 0.0,
    contact_count: int = 8,
    contact_prefix: Optional[str] = None,
) -> List[ElectrodeSamplePoint]:
    """
    Build a circumferential contact set from a CuffSpec.

    Notes
    -----
    This uses the cuff inner radius as the contact ring radius. It is intended
    as a first-pass geometry-to-contact bridge for the existing homogeneous VC
    model.
    """
    cuff.validate()

    n = max(1, int(contact_count))
    prefix = contact_prefix or cuff.label or cuff.cuff_id

    radius_um = float(cuff.inner_radius_um)
    y0 = float(cuff.center_y_um)
    z0 = float(cuff.center_z_um)

    points: List[ElectrodeSamplePoint] = []
    for k in range(n):
        theta = 2.0 * math.pi * float(k) / float(n)
        points.append(
            ElectrodeSamplePoint(
                name=f"{prefix}_ring_{k}",
                x_um=float(x_um),
                y_um=y0 + radius_um * math.cos(theta),
                z_um=z0 + radius_um * math.sin(theta),
            )
        )
    return points

def build_multicontact_cuff_electrode_points(
    cuff: CuffSpec,
    *,
    axial_positions_um: Sequence[float],
    contacts_per_ring: int = 4,
    contact_prefix: Optional[str] = None,
) -> List[ElectrodeSamplePoint]:
    """
    Build a multicontact cuff as repeated circumferential rings along x.

    Parameters
    ----------
    cuff:
        Cuff geometry metadata.
    axial_positions_um:
        Axial x-locations of the contact rings.
    contacts_per_ring:
        Number of contacts distributed around each ring.
    """
    cuff.validate()

    prefix = contact_prefix or cuff.label or cuff.cuff_id
    points: List[ElectrodeSamplePoint] = []

    for ring_idx, x_um in enumerate(axial_positions_um):
        ring_points = build_ring_cuff_electrode_points(
            cuff,
            x_um=float(x_um),
            contact_count=int(contacts_per_ring),
            contact_prefix=f"{prefix}_ax{ring_idx}",
        )
        points.extend(ring_points)

    return points

def build_electrode_points_from_cuffs(
    cuffs: Sequence[CuffSpec],
    *,
    cuff_mode: str = "ring",
    ring_contact_count: int = 8,
    multicontact_axial_positions_um: Sequence[float] = (5000.0, 10000.0),
    multicontacts_per_ring: int = 4,
) -> List[ElectrodeSamplePoint]:
    """
    Convert CuffSpec objects into explicit VC sample points.

    Parameters
    ----------
    cuff_mode:
        "ring" or "multicontact"
    """
    points: List[ElectrodeSamplePoint] = []

    for cuff in cuffs:
        if cuff_mode == "ring":
            points.extend(
                build_ring_cuff_electrode_points(
                    cuff,
                    x_um=0.0,
                    contact_count=int(ring_contact_count),
                )
            )
        elif cuff_mode == "multicontact":
            points.extend(
                build_multicontact_cuff_electrode_points(
                    cuff,
                    axial_positions_um=multicontact_axial_positions_um,
                    contacts_per_ring=int(multicontacts_per_ring),
                )
            )
        else:
            raise ValueError(f"Unsupported cuff_mode: {cuff_mode!r}")

    return points

def build_electrode_points_from_architecture(
    architecture_spec: NerveArchitectureSpec,
    *,
    ring_min_points: int = 8,
    include_explicit_electrodes: bool = True,
    include_cuffs: bool = True,
    cuff_mode: str = "ring",
    cuff_ring_contact_count: int = 8,
    cuff_multicontact_axial_positions_um: Sequence[float] = (5000.0, 10000.0),
    cuff_multicontacts_per_ring: int = 4,
) -> List[ElectrodeSamplePoint]:
    """
    Build the full VC sample-point list from a NerveArchitectureSpec.

    This merges:
    - explicit ElectrodeSpec definitions
    - cuff-derived contacts from CuffSpec definitions
    """
    architecture_spec.validate()

    points: List[ElectrodeSamplePoint] = []

    if include_explicit_electrodes and architecture_spec.electrodes:
        points.extend(
            build_electrode_points_from_spec(
                architecture_spec.electrodes,
                ring_min_points=ring_min_points,
            )
        )

    if include_cuffs and architecture_spec.cuffs:
        points.extend(
            build_electrode_points_from_cuffs(
                architecture_spec.cuffs,
                cuff_mode=cuff_mode,
                ring_contact_count=int(cuff_ring_contact_count),
                multicontact_axial_positions_um=cuff_multicontact_axial_positions_um,
                multicontacts_per_ring=int(cuff_multicontacts_per_ring),
            )
        )

    return deduplicate_electrode_points(points)

def deduplicate_electrode_points(
    points: Sequence[ElectrodeSamplePoint],
    *,
    name_only: bool = True,
) -> List[ElectrodeSamplePoint]:
    """
    Remove duplicate contact names.

    First-pass behavior:
    - if name_only=True, duplicates are removed by contact name
    """
    seen = set()
    unique: List[ElectrodeSamplePoint] = []

    for point in points:
        key = point.name if name_only else (point.name, point.x_um, point.y_um, point.z_um)
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)

    return unique