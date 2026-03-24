"""
architecture_presets.py

Small reusable architecture builders for validation and demos.
"""

from __future__ import annotations

from Scripts.NeuroSolver.architecture.architecture_schema import (
    ElectrodeSpec,
    FascicleSpec,
    FiberSpec,
    NerveArchitectureSpec,
)

def make_three_fiber_demo_architecture(length_um: float = 20000.0) -> NerveArchitectureSpec:
    """
    Simple 3-fiber architecture for quick validation.
    """
    spec = NerveArchitectureSpec(
        bundle_id="architecture_validation",
        length_um=float(length_um),
        outer_nerve_radius_um=200.0,
        layout_name="custom_demo",
        fascicles=[
            FascicleSpec(
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                radius_um=160.0,
                label="main_fascicle",
            )
        ],
        fibers=[
            FiberSpec(
                fiber_id=0,
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                label="center_fiber",
            ),
            FiberSpec(
                fiber_id=1,
                fascicle_id=0,
                center_y_um=80.0,
                center_z_um=0.0,
                label="right_fiber",
            ),
            FiberSpec(
                fiber_id=2,
                fascicle_id=0,
                center_y_um=-80.0,
                center_z_um=0.0,
                label="left_fiber",
            ),
        ],
        electrodes=[
            ElectrodeSpec(
                kind="point",
                x_um=5000.0,
                y_um=100.0,
                z_um=0.0,
                label="E_near",
            ),
            ElectrodeSpec(
                kind="point",
                x_um=10000.0,
                y_um=250.0,
                z_um=0.0,
                label="E_far",
            ),
        ],
        metadata={"source": "architecture_presets.make_three_fiber_demo_architecture"},
    )
    spec.validate()
    return spec

def make_linear_fiber_array(
    fiber_count: int = 5,
    spacing_um: float = 75.0,
    length_um: float = 20000.0,
    fascicle_radius_um: float = 250.0,
) -> NerveArchitectureSpec:
    """
    Build a simple linear cross-sectional arrangement of fibers.
    """
    if fiber_count <= 0:
        raise ValueError("fiber_count must be positive.")
    if spacing_um <= 0.0:
        raise ValueError("spacing_um must be positive.")

    half = 0.5 * (fiber_count - 1)

    fibers = []
    for i in range(fiber_count):
        y = (i - half) * spacing_um
        fibers.append(
            FiberSpec(
                fiber_id=i,
                fascicle_id=0,
                center_y_um=float(y),
                center_z_um=0.0,
                label=f"fiber_{i}",
            )
        )

    spec = NerveArchitectureSpec(
        bundle_id=f"linear_{fiber_count}_fiber_architecture",
        length_um=float(length_um),
        layout_name="linear_custom",
        fascicles=[
            FascicleSpec(
                fascicle_id=0,
                center_y_um=0.0,
                center_z_um=0.0,
                radius_um=float(fascicle_radius_um),
                label="linear_fascicle",
            )
        ],
        fibers=fibers,
        electrodes=[
            ElectrodeSpec(
                kind="point",
                x_um=5000.0,
                y_um=150.0,
                z_um=0.0,
                label="E0",
            )
        ],
        metadata={"source": "architecture_presets.make_linear_fiber_array"},
    )
    spec.validate()
    return spec