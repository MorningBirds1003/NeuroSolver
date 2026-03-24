"""
write_demo_architecture.py

Write a starter architecture JSON file to disk.
"""

from __future__ import annotations

from Scripts.NeuroSolver.architecture.architecture_io import save_architecture_spec
from Scripts.NeuroSolver.architecture.architecture_presets import (
    make_three_fiber_demo_architecture,
)

def main() -> None:
    spec = make_three_fiber_demo_architecture(length_um=20000.0)
    path = save_architecture_spec(spec, "outputs/architectures/demo_architecture.json")
    print(f"Wrote demo architecture to: {path}")

if __name__ == "__main__":
    main()