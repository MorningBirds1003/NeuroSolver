"""
materials.py

Material-region definitions and lightweight ownership utilities for
NeuroSolver architecture studies.

Purpose
-------
This module is the first step toward bundle/cuff/material realism without
rewriting the core solver. It provides:

- named material presets
- conductivity, extracellular volume fraction alpha, and tortuosity lambda
- region descriptors for fascicles / cuff / saline / nerve envelope
- simple lookup helpers for architecture-owned regions

This is intentionally lightweight:
it does not yet solve anisotropic VC or region-aware KNP directly.
Instead, it creates a clean place to store the information that later solver
layers will consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Core material definitions
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialProperties:
    """
    Bulk material properties relevant to extracellular transport and VC.

    Parameters
    ----------
    name:
        Unique material name.
    sigma_S_per_m:
        Effective conductivity [S/m].
    alpha:
        Extracellular volume fraction [-].
    lambda_tortuosity:
        Tortuosity [-].
    relative_permittivity:
        Placeholder for later electric-field realism.
    notes:
        Free-form notes.
    metadata:
        Arbitrary annotations.
    """
    name: str
    sigma_S_per_m: float
    alpha: float
    lambda_tortuosity: float
    relative_permittivity: Optional[float] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("MaterialProperties.name must be non-empty.")
        if float(self.sigma_S_per_m) <= 0.0:
            raise ValueError(f"{self.name}: sigma_S_per_m must be positive.")
        if not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError(f"{self.name}: alpha must lie in (0, 1].")
        if float(self.lambda_tortuosity) < 1.0:
            raise ValueError(f"{self.name}: lambda_tortuosity should be >= 1.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -----------------------------------------------------------------------------
# Region ownership
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CircularRegion:
    """
    Simple circular transverse region assignment.

    Parameters
    ----------
    region_id:
        Unique identifier.
    material_name:
        Name of the owning material.
    center_y_um, center_z_um:
        Region center in transverse plane.
    radius_um:
        Region radius.
    kind:
        Region class, e.g. fascicle, cuff, saline, envelope.
    label:
        Optional display name.
    priority:
        Higher priority wins when multiple regions overlap.
    metadata:
        Arbitrary annotations.
    """
    region_id: str
    material_name: str
    center_y_um: float
    center_z_um: float
    radius_um: float
    kind: str = "generic"
    label: str = ""
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.region_id:
            raise ValueError("CircularRegion.region_id must be non-empty.")
        if not self.material_name:
            raise ValueError("CircularRegion.material_name must be non-empty.")
        if float(self.radius_um) <= 0.0:
            raise ValueError(f"{self.region_id}: radius_um must be positive.")

    def contains_point_um(self, y_um: float, z_um: float) -> bool:
        dy = float(y_um) - float(self.center_y_um)
        dz = float(z_um) - float(self.center_z_um)
        return (dy * dy + dz * dz) <= float(self.radius_um) ** 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MaterialLayout:
    """
    Collection of material definitions plus transverse ownership regions.
    """
    materials: Dict[str, MaterialProperties] = field(default_factory=dict)
    circular_regions: List[CircularRegion] = field(default_factory=list)
    default_material_name: str = "saline"

    def validate(self) -> None:
        if self.default_material_name not in self.materials:
            raise ValueError(
                f"default_material_name={self.default_material_name!r} "
                "not found in materials."
            )

        for material in self.materials.values():
            material.validate()

        region_ids = []
        for region in self.circular_regions:
            region.validate()
            if region.material_name not in self.materials:
                raise ValueError(
                    f"Region {region.region_id!r} references unknown material "
                    f"{region.material_name!r}."
                )
            region_ids.append(region.region_id)

        if len(region_ids) != len(set(region_ids)):
            raise ValueError("CircularRegion.region_id values must be unique.")

    def add_material(self, material: MaterialProperties) -> None:
        material.validate()
        self.materials[material.name] = material

    def add_region(self, region: CircularRegion) -> None:
        region.validate()
        if region.material_name not in self.materials:
            raise ValueError(
                f"Cannot add region {region.region_id!r}: "
                f"material {region.material_name!r} is not registered."
            )
        self.circular_regions.append(region)

    def get_material(self, name: str) -> MaterialProperties:
        return self.materials[name]

    def lookup_material_name_at_point_um(self, y_um: float, z_um: float) -> str:
        """
        Resolve the owning material at a transverse point.

        If multiple regions contain the point, highest priority wins.
        If tied, the last-added matching region wins after sorting by priority.
        """
        matches = [
            region for region in self.circular_regions
            if region.contains_point_um(y_um=y_um, z_um=z_um)
        ]
        if not matches:
            return self.default_material_name

        matches = sorted(matches, key=lambda r: (int(r.priority), self.circular_regions.index(r)))
        return matches[-1].material_name

    def lookup_material_at_point_um(self, y_um: float, z_um: float) -> MaterialProperties:
        return self.materials[self.lookup_material_name_at_point_um(y_um=y_um, z_um=z_um)]

    def summarize_point_um(self, y_um: float, z_um: float) -> Dict[str, Any]:
        mat = self.lookup_material_at_point_um(y_um=y_um, z_um=z_um)
        return {
            "material_name": mat.name,
            "sigma_S_per_m": float(mat.sigma_S_per_m),
            "alpha": float(mat.alpha),
            "lambda_tortuosity": float(mat.lambda_tortuosity),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_material_name": self.default_material_name,
            "materials": {name: mat.to_dict() for name, mat in self.materials.items()},
            "circular_regions": [region.to_dict() for region in self.circular_regions],
        }

# -----------------------------------------------------------------------------
# Default material presets
# -----------------------------------------------------------------------------
def build_default_material_library() -> Dict[str, MaterialProperties]:
    """
    Conservative first-pass material library.

    Notes
    -----
    These are prototype values meant to differentiate regions and support
    comparative sweeps. They are not yet a final validated physiology library.
    """
    library = {
        "saline": MaterialProperties(
            name="saline",
            sigma_S_per_m=1.5,
            alpha=1.0,
            lambda_tortuosity=1.0,
            notes="External bathing medium.",
        ),
        "endoneurium": MaterialProperties(
            name="endoneurium",
            sigma_S_per_m=0.6,
            alpha=0.25,
            lambda_tortuosity=1.6,
            notes="Prototype endoneurial ECS region.",
        ),
        "perineurium": MaterialProperties(
            name="perineurium",
            sigma_S_per_m=0.08,
            alpha=0.15,
            lambda_tortuosity=1.8,
            notes="Low-conductivity fascicle sheath proxy.",
        ),
        "epineurium": MaterialProperties(
            name="epineurium",
            sigma_S_per_m=0.25,
            alpha=0.30,
            lambda_tortuosity=1.4,
            notes="Prototype outer nerve connective region.",
        ),
        "cuff_polymer": MaterialProperties(
            name="cuff_polymer",
            sigma_S_per_m=1.0e-6,
            alpha=0.05,
            lambda_tortuosity=2.0,
            notes="Insulating cuff body proxy.",
        ),
        "metal_contact": MaterialProperties(
            name="metal_contact",
            sigma_S_per_m=1.0e6,
            alpha=1.0,
            lambda_tortuosity=1.0,
            notes="Idealized contact region placeholder.",
        ),
    }

    for mat in library.values():
        mat.validate()
    return library

def build_default_material_layout() -> MaterialLayout:
    """
    Build a blank layout with default material presets loaded.
    """
    materials = build_default_material_library()
    layout = MaterialLayout(
        materials=dict(materials),
        circular_regions=[],
        default_material_name="saline",
    )
    layout.validate()
    return layout

# -----------------------------------------------------------------------------
# Layout builders from architecture-like data
# -----------------------------------------------------------------------------
def add_fascicle_regions_from_spec(
    layout: MaterialLayout,
    fascicles: Iterable[Any],
    fascicle_material_name: str = "endoneurium",
) -> None:
    """
    Add one circular region per FascicleSpec-like object.

    Expected attributes:
    - fascicle_id
    - center_y_um
    - center_z_um
    - radius_um
    - label (optional)
    """
    for fascicle in fascicles:
        layout.add_region(
            CircularRegion(
                region_id=f"fascicle_{int(fascicle.fascicle_id)}",
                material_name=fascicle_material_name,
                center_y_um=float(fascicle.center_y_um),
                center_z_um=float(fascicle.center_z_um),
                radius_um=float(fascicle.radius_um),
                kind="fascicle",
                label=getattr(fascicle, "label", "") or f"fascicle_{int(fascicle.fascicle_id)}",
                priority=10,
            )
        )

def add_outer_nerve_region(
    layout: MaterialLayout,
    *,
    radius_um: float,
    center_y_um: float = 0.0,
    center_z_um: float = 0.0,
    material_name: str = "epineurium",
    region_id: str = "outer_nerve",
    priority: int = 5,
) -> None:
    """
    Add a circular outer nerve envelope.
    """
    layout.add_region(
        CircularRegion(
            region_id=region_id,
            material_name=material_name,
            center_y_um=float(center_y_um),
            center_z_um=float(center_z_um),
            radius_um=float(radius_um),
            kind="outer_nerve",
            label=region_id,
            priority=int(priority),
        )
    )

def add_ring_cuff_region(
    layout: MaterialLayout,
    *,
    inner_radius_um: float,
    thickness_um: float,
    center_y_um: float = 0.0,
    center_z_um: float = 0.0,
    material_name: str = "cuff_polymer",
    region_id_prefix: str = "cuff",
) -> None:
    """
    Approximate a cuff body as an outer circular region plus an inner void.

    Current implementation:
    - adds only the outer cuff body disk
    - relies on higher-priority inner nerve/fascicle regions to own the center

    This is enough for region bookkeeping in the first pass.
    """
    outer_radius_um = float(inner_radius_um) + float(thickness_um)
    layout.add_region(
        CircularRegion(
            region_id=f"{region_id_prefix}_body",
            material_name=material_name,
            center_y_um=float(center_y_um),
            center_z_um=float(center_z_um),
            radius_um=outer_radius_um,
            kind="cuff",
            label=f"{region_id_prefix}_body",
            priority=1,
            metadata={
                "inner_radius_um": float(inner_radius_um),
                "thickness_um": float(thickness_um),
            },
        )
    )

def build_material_layout_from_architecture_spec(
    architecture_spec: Any,
    *,
    outer_nerve_radius_um: Optional[float] = None,
    cuff_inner_radius_um: Optional[float] = None,
    cuff_thickness_um: Optional[float] = None,
) -> MaterialLayout:
    """
    Build a first-pass material layout from an architecture spec.

    Expected architecture_spec fields:
    - fascicles
    - metadata (optional)

    This function does not mutate the architecture spec.
    """
    layout = build_default_material_layout()

    if outer_nerve_radius_um is not None:
        add_outer_nerve_region(
            layout,
            radius_um=float(outer_nerve_radius_um),
            material_name="epineurium",
        )

    add_fascicle_regions_from_spec(
        layout,
        architecture_spec.fascicles,
        fascicle_material_name="endoneurium",
    )

    if cuff_inner_radius_um is not None and cuff_thickness_um is not None:
        add_ring_cuff_region(
            layout,
            inner_radius_um=float(cuff_inner_radius_um),
            thickness_um=float(cuff_thickness_um),
            material_name="cuff_polymer",
        )

    layout.validate()
    return layout