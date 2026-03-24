"""
stimuli.py

Unified stimulus-generation utilities for NeuroSolver.

Purpose
-------
Centralize waveform generation so the same pulse definitions can be reused by:
- single-node membrane tests,
- cable simulations,
- extracellular VC source tests,
- future ECS / KNP perturbation studies.

Conventions
-----------
- Time: ms
- Intracellular absolute current: uA
- Membrane current density: uA/cm^2
- Extracellular source current: uA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

@dataclass(frozen=True)
class PulseSpec:
    """
    Generic single-phase pulse description.
    """
    start_ms: float
    width_ms: float
    amplitude: float

@dataclass(frozen=True)
class BiphasicPulseSpec:
    """
    Biphasic pulse definition.

    This is useful for stimulation protocols where net injected charge should be
    reduced or where a cathodic/anodic pair is desired explicitly.
    """
    start_ms: float
    phase1_width_ms: float
    phase1_amplitude: float
    interphase_gap_ms: float
    phase2_width_ms: float
    phase2_amplitude: float

@dataclass(frozen=True)
class PulseTrainSpec:
    """
    Repeated rectangular pulse train.
    """
    start_ms: float
    width_ms: float
    amplitude: float
    pulse_count: int
    interval_ms: float

def rectangular_pulse(
    t_ms: float,
    start_ms: float,
    width_ms: float,
    amplitude: float,
) -> float:
    """
    Return a single rectangular pulse value.

    The pulse is active on the half-open interval:
        [start_ms, start_ms + width_ms)
    """
    t_ms = float(t_ms)
    start_ms = float(start_ms)
    width_ms = float(width_ms)
    amplitude = float(amplitude)

    if start_ms <= t_ms < start_ms + width_ms:
        return amplitude
    return 0.0

def biphasic_pulse(
    t_ms: float,
    spec: BiphasicPulseSpec,
) -> float:
    """
    Return a biphasic pulse value at time t.
    """
    t_ms = float(t_ms)

    p1_start = spec.start_ms
    p1_end = spec.start_ms + spec.phase1_width_ms
    p2_start = p1_end + spec.interphase_gap_ms
    p2_end = p2_start + spec.phase2_width_ms

    if p1_start <= t_ms < p1_end:
        return float(spec.phase1_amplitude)
    if p2_start <= t_ms < p2_end:
        return float(spec.phase2_amplitude)
    return 0.0

def pulse_train(
    t_ms: float,
    spec: PulseTrainSpec,
) -> float:
    """
    Return a repeated rectangular pulse-train value at time t.
    """
    if spec.pulse_count <= 0:
        return 0.0

    for k in range(spec.pulse_count):
        start_k = float(spec.start_ms) + k * float(spec.interval_ms)
        value = rectangular_pulse(
            t_ms=t_ms,
            start_ms=start_k,
            width_ms=spec.width_ms,
            amplitude=spec.amplitude,
        )
        if value != 0.0:
            return value
    return 0.0

def membrane_density_stimulus_uA_per_cm2(
    t_ms: float,
    start_ms: float,
    width_ms: float,
    amplitude_uA_per_cm2: float,
) -> float:
    """
    Membrane current-density stimulus [uA/cm^2].
    """
    return rectangular_pulse(
        t_ms=t_ms,
        start_ms=start_ms,
        width_ms=width_ms,
        amplitude=amplitude_uA_per_cm2,
    )

def intracellular_node_stimulus_uA(
    t_ms: float,
    start_ms: float,
    width_ms: float,
    amplitude_uA: float,
) -> float:
    """
    Intracellular absolute current stimulus [uA].
    """
    return rectangular_pulse(
        t_ms=t_ms,
        start_ms=start_ms,
        width_ms=width_ms,
        amplitude=amplitude_uA,
    )

def extracellular_point_source_uA(
    t_ms: float,
    start_ms: float,
    width_ms: float,
    amplitude_uA: float,
) -> float:
    """
    Extracellular point-source current [uA].
    """
    return rectangular_pulse(
        t_ms=t_ms,
        start_ms=start_ms,
        width_ms=width_ms,
        amplitude=amplitude_uA,
    )

def stimulus_vector_single_index(
    n_entries: int,
    active_index: int,
    amplitude: float,
) -> np.ndarray:
    """
    Build a vector with one active entry.

    This is the simplest sparse stimulation pattern and is useful for selecting
    one node, one compartment, or one fiber without introducing a heavier
    targeting abstraction.
    """
    vec = np.zeros(int(n_entries), dtype=float)
    if 0 <= int(active_index) < int(n_entries):
        vec[int(active_index)] = float(amplitude)
    return vec

def pulse_train_vector_single_index(
    t_ms: float,
    n_entries: int,
    active_index: int,
    spec: PulseTrainSpec,
) -> np.ndarray:
    """
    Build a single-index vector driven by a pulse train.
    """
    amp = pulse_train(t_ms, spec)
    return stimulus_vector_single_index(n_entries=n_entries, active_index=active_index, amplitude=amp)
