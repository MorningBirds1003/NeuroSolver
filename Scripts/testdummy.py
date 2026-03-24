from Scripts.NeuroSolver.architecture.architecture_schema import (
    CuffSpec,
    FascicleSpec,
    FiberSpec,
    NerveArchitectureSpec,
)
from Scripts.NeuroSolver.architecture.electrode_geometry import (
    build_electrode_points_from_architecture,
)

spec = NerveArchitectureSpec(
    bundle_id="cuff_contact_test",
    length_um=20000.0,
    outer_nerve_radius_um=260.0,
    fascicles=[
        FascicleSpec(
            fascicle_id=0,
            center_y_um=0.0,
            center_z_um=0.0,
            radius_um=180.0,
            label="main_fascicle",
        )
    ],
    fibers=[
        FiberSpec(fiber_id=0, fascicle_id=0, center_y_um=0.0, center_z_um=0.0),
        FiberSpec(fiber_id=1, fascicle_id=0, center_y_um=80.0, center_z_um=0.0),
    ],
    cuffs=[
        CuffSpec(
            cuff_id="cuff_A",
            center_y_um=0.0,
            center_z_um=0.0,
            inner_radius_um=320.0,
            thickness_um=120.0,
            label="ring_cuff",
        )
    ],
)

points = build_electrode_points_from_architecture(
    spec,
    cuff_mode="ring",
    cuff_ring_contact_count=8,
)
print(len(points))
for p in points:
    print(p)