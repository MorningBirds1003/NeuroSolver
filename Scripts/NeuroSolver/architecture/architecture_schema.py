"""
architecture_schema.py

User-facing architecture-spec dataclasses for NeuroSolver.

Purpose
-------
This module defines lightweight, solver-agnostic objects that describe a
custom nerve architecture before it is converted into bundle geometry and fed
into the existing propagation scheduler.

Design rule
-----------
These dataclasses should describe what the user wants to build, not how the
solver stores runtime state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

@dataclass(frozen=True)
class FiberSpec:
    """
    Specification for one explicit fiber in the transverse nerve cross-section.
    """
    fiber_id: int
    fascicle_id: int
    center_y_um: float
    center_z_um: float
    fiber_diameter_um: Optional[float] = None
    axon_diameter_um: Optional[float] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.fiber_id) < 0:
            raise ValueError("fiber_id must be >= 0.")
        if int(self.fascicle_id) < 0:
            raise ValueError("fascicle_id must be >= 0.")
        if self.fiber_diameter_um is not None and float(self.fiber_diameter_um) <= 0.0:
            raise ValueError("fiber_diameter_um must be positive when provided.")
        if self.axon_diameter_um is not None and float(self.axon_diameter_um) <= 0.0:
            raise ValueError("axon_diameter_um must be positive when provided.")
        if (
            self.fiber_diameter_um is not None
            and self.axon_diameter_um is not None
            and float(self.axon_diameter_um) > float(self.fiber_diameter_um)
        ):
            raise ValueError("axon_diameter_um cannot exceed fiber_diameter_um.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class FascicleSpec:
    """
    Optional fascicle boundary/placement metadata.

    Notes
    -----
    The current solver stack does not yet enforce fascicle boundary physics.
    This object is still useful now for organization, plotting, recruitment
    grouping, and later material-property extensions.
    """
    fascicle_id: int
    center_y_um: float
    center_z_um: float
    radius_um: float
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.fascicle_id) < 0:
            raise ValueError("fascicle_id must be >= 0.")
        if float(self.radius_um) <= 0.0:
            raise ValueError("radius_um must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ElectrodeSpec:
    """
    Declarative electrode/contact specification.

    Supported kinds
    ---------------
    - "point": one explicit sampling point
    - "ring": a circumferential set of sample points at one x-location
    - "multicontact": repeated contacts along the x-axis
    """
    kind: Literal["point", "ring", "multicontact"] = "point"
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = 0.0
    radius_um: Optional[float] = None
    contact_count: int = 1
    spacing_um: Optional[float] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in {"point", "ring", "multicontact"}:
            raise ValueError(f"Unsupported electrode kind: {self.kind!r}")

        if int(self.contact_count) <= 0:
            raise ValueError("contact_count must be >= 1.")

        if self.kind == "ring":
            if self.radius_um is None or float(self.radius_um) <= 0.0:
                raise ValueError("Ring electrodes require a positive radius_um.")

        if self.kind == "multicontact":
            if self.spacing_um is not None and float(self.spacing_um) <= 0.0:
                raise ValueError("spacing_um must be positive when provided.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class MaterialRegionSpec:
    """
    Declarative named region assignment for later material ownership.

    Current supported shape:
    - circular region in transverse plane
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
            raise ValueError("region_id must be non-empty.")
        if not self.material_name:
            raise ValueError("material_name must be non-empty.")
        if float(self.radius_um) <= 0.0:
            raise ValueError("radius_um must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CuffSpec:
    """
    First-pass cuff geometry metadata.

    This is initially descriptive and used for region/material ownership plus
    electrode grouping. Solver coupling can consume it later.
    """
    cuff_id: str
    center_y_um: float = 0.0
    center_z_um: float = 0.0
    inner_radius_um: float = 300.0
    thickness_um: float = 100.0
    material_name: str = "cuff_polymer"
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.cuff_id:
            raise ValueError("cuff_id must be non-empty.")
        if float(self.inner_radius_um) <= 0.0:
            raise ValueError("inner_radius_um must be positive.")
        if float(self.thickness_um) <= 0.0:
            raise ValueError("thickness_um must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
@dataclass
class NerveArchitectureSpec:
    """
    Top-level custom nerve architecture description.
    """
    bundle_id: str
    length_um: float
    fibers: List[FiberSpec] = field(default_factory=list)
    fascicles: List[FascicleSpec] = field(default_factory=list)
    electrodes: List[ElectrodeSpec] = field(default_factory=list)
    material_regions: List[MaterialRegionSpec] = field(default_factory=list)
    cuffs: List[CuffSpec] = field(default_factory=list)
    outer_nerve_radius_um: Optional[float] = None    
    layout_name: str = "custom"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Basic structural validation for early user-side errors.
        """
        if not self.bundle_id:
            raise ValueError("bundle_id must be a non-empty string.")
        if float(self.length_um) <= 0.0:
            raise ValueError("length_um must be positive.")
        if len(self.fibers) == 0:
            raise ValueError("NerveArchitectureSpec must contain at least one FiberSpec.")

        for fiber in self.fibers:
            fiber.validate()
        for fascicle in self.fascicles:
            fascicle.validate()
        for electrode in self.electrodes:
            electrode.validate()
        for region in self.material_regions:
            region.validate()
        for cuff in self.cuffs:
            cuff.validate()

        region_ids = [r.region_id for r in self.material_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("Material region IDs must be unique within NerveArchitectureSpec.")

        cuff_ids = [c.cuff_id for c in self.cuffs]
        if len(cuff_ids) != len(set(cuff_ids)):
            raise ValueError("Cuff IDs must be unique within NerveArchitectureSpec.")

        if self.outer_nerve_radius_um is not None and float(self.outer_nerve_radius_um) <= 0.0:
            raise ValueError("outer_nerve_radius_um must be positive when provided.")        

        fiber_ids = [int(f.fiber_id) for f in self.fibers]
        if len(fiber_ids) != len(set(fiber_ids)):
            raise ValueError("Fiber IDs must be unique within NerveArchitectureSpec.")

        fascicle_ids = [int(f.fascicle_id) for f in self.fascicles]
        if len(fascicle_ids) != len(set(fascicle_ids)):
            raise ValueError("Fascicle IDs must be unique within NerveArchitectureSpec.")

        declared_fascicles = set(fascicle_ids)
        used_fascicles = {int(f.fascicle_id) for f in self.fibers}
        undeclared = sorted(used_fascicles - declared_fascicles)
        if self.fascicles and undeclared:
            raise ValueError(
                "All fiber fascicle IDs must exist in fascicles when explicit "
                f"fascicle specs are supplied. Missing: {undeclared}"
            )

    @property
    def fiber_count(self) -> int:
        return len(self.fibers)

    @property
    def fascicle_count(self) -> int:
        return len(self.fascicles)

    @property
    def electrode_count(self) -> int:
        return len(self.electrodes)

    def fiber_ids(self) -> List[int]:
        return [int(f.fiber_id) for f in self.fibers]

    def fascicle_ids(self) -> List[int]:
        return [int(f.fascicle_id) for f in self.fascicles]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "length_um": float(self.length_um),
            "layout_name": self.layout_name,
            "fibers": [f.to_dict() for f in self.fibers],
            "fascicles": [f.to_dict() for f in self.fascicles],
            "electrodes": [e.to_dict() for e in self.electrodes],
                        "material_regions": [r.to_dict() for r in self.material_regions],
            "cuffs": [c.to_dict() for c in self.cuffs],
            "outer_nerve_radius_um": (
                None if self.outer_nerve_radius_um is None else float(self.outer_nerve_radius_um)
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NerveArchitectureSpec":
        return cls(
            bundle_id=str(payload["bundle_id"]),
            length_um=float(payload["length_um"]),
            layout_name=str(payload.get("layout_name", "custom")),
            fibers=[FiberSpec(**item) for item in payload.get("fibers", [])],
            fascicles=[FascicleSpec(**item) for item in payload.get("fascicles", [])],
            electrodes=[ElectrodeSpec(**item) for item in payload.get("electrodes", [])],
                        material_regions=[
                MaterialRegionSpec(**item) for item in payload.get("material_regions", [])
            ],
            cuffs=[CuffSpec(**item) for item in payload.get("cuffs", [])],
            outer_nerve_radius_um=payload.get("outer_nerve_radius_um", None),
            metadata=dict(payload.get("metadata", {})),
        )