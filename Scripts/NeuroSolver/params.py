
"""
params.py

Central parameter registry for the NeuroSolver project.

This makes the default parameter bundle explicitly preset-driven and
starts from a physiological myelinated-fiber baseline rather than a pure debug
harness. The objective is not to claim a fully validated nerve template yet;
it is to ensure that the active model is initialized from a plausible
mammalian myelinated-fiber regime and that downstream modules can query the
preset identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Tuple
import math


# -----------------------------------------------------------------------------
# Fundamental physical constants
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalConstants:
    faraday_C_per_mol: float = 96485.33212
    gas_constant_J_per_molK: float = 8.314462618
    absolute_zero_C: float = 273.15
    epsilon_0_F_per_m: float = 8.8541878128e-12
    water_relative_permittivity_37C: float = 74.0
# -----------------------------------------------------------------------------
# Thermal control
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TemperatureSettings:
    celsius: float = 37.0
    reference_celsius_for_q10: float = 6.3
    q10_default: float = 3.0
    use_temperature_scaling: bool = True

    @property
    def kelvin(self) -> float:
        return self.celsius + 273.15
# -----------------------------------------------------------------------------
# Ionic chemistry
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class IonSpecies:
    name: str
    symbol: str
    valence: int
    diffusion_m2_per_s: float
    extracellular_mM: float
    intracellular_mM: float
    comment: str = "baseline value"

@dataclass(frozen=True)
class IonPool:
    sodium: IonSpecies = IonSpecies(
        name="Sodium",
        symbol="Na",
        valence=+1,
        diffusion_m2_per_s=1.33e-9,
        extracellular_mM=145.0,
        intracellular_mM=12.0,
        comment="Typical mammalian-neuron-style baseline for initialization.",
    )
    potassium: IonSpecies = IonSpecies(
        name="Potassium",
        symbol="K",
        valence=+1,
        diffusion_m2_per_s=1.96e-9,
        extracellular_mM=4.0,
        intracellular_mM=140.0,
        comment="Typical mammalian-neuron-style baseline for initialization.",
    )
    chloride: IonSpecies = IonSpecies(
        name="Chloride",
        symbol="Cl",
        valence=-1,
        diffusion_m2_per_s=2.03e-9,
        extracellular_mM=119.0,
        intracellular_mM=4.2,
        comment="Typical mature-neuron chloride baseline.",
    )
    calcium: IonSpecies = IonSpecies(
        name="Calcium",
        symbol="Ca",
        valence=+2,
        diffusion_m2_per_s=0.71e-9,
        extracellular_mM=1.2,
        intracellular_mM=1.0e-4,
        comment="Free extracellular and resting intracellular Ca2+ baseline.",
    )
# -----------------------------------------------------------------------------
# Preset identity
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPresetDefaults:
    active_preset_name: str = "mammalian_myelinated_medium_fiber"
    description: str = (
        "Preset-driven baseline for a medium-diameter mammalian myelinated axon "
        "with physiology-first defaults."
    )
    runtime_enable_vc: bool = True
    runtime_enable_knp: bool = True
    runtime_enable_ecs_feedback: bool = True
# -----------------------------------------------------------------------------
# Membrane and conductance defaults
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MembraneDefaults:
    resting_potential_mV: float = -70.0
    initial_voltage_mV: float = -70.0
    membrane_capacitance_uF_per_cm2: float = 1.0
    axial_resistivity_ohm_cm: float = 70.0
    extracellular_conductivity_S_per_m: float = 0.30
    intracellular_conductivity_S_per_m: float = 0.60
    nodal_specific_membrane_resistance_ohm_cm2: float = 1000.0
    internodal_specific_membrane_resistance_ohm_cm2: float = 1.0e5
    myelin_capacitance_uF_per_cm2: float = 0.01
    leak_reversal_mV: float = -70.0
    sodium_reversal_mV: float = 66.6
    potassium_reversal_mV: float = -95.0
    chloride_reversal_mV: float = -89.4
    calcium_reversal_mV: float = 125.5

@dataclass(frozen=True)
class ConductanceDefaults:
    gbar_na_mS_per_cm2: float = 320.0
    gbar_k_mS_per_cm2: float = 60.0
    gbar_l_mS_per_cm2: float = 0.05
    gbar_persistent_na_mS_per_cm2: float = 0.0
    gbar_k_slow_mS_per_cm2: float = 0.0
    gbar_ca_mS_per_cm2: float = 0.0
    gbar_hcn_mS_per_cm2: float = 0.0

@dataclass(frozen=True)
class RegionConductanceSet:
    gbar_na_mS_per_cm2: float = 0.0
    gbar_k_mS_per_cm2: float = 0.0
    gbar_l_mS_per_cm2: float = 0.0
    gbar_persistent_na_mS_per_cm2: float = 0.0
    gbar_k_slow_mS_per_cm2: float = 0.0
    gbar_ca_mS_per_cm2: float = 0.0
    gbar_hcn_mS_per_cm2: float = 0.0

@dataclass(frozen=True)
class RegionalConductanceDefaults:
    node: RegionConductanceSet = field(
        default_factory=lambda: RegionConductanceSet(
            gbar_na_mS_per_cm2=320.0,
            gbar_k_mS_per_cm2=60.0,
            gbar_l_mS_per_cm2=0.05,
        )
    )
    internode: RegionConductanceSet = field(
        default_factory=lambda: RegionConductanceSet(
            gbar_na_mS_per_cm2=0.0,
            gbar_k_mS_per_cm2=0.0,
            gbar_l_mS_per_cm2=0.0001,
        )
    )
    paranode: RegionConductanceSet = field(default_factory=RegionConductanceSet)
    juxtaparanode: RegionConductanceSet = field(default_factory=RegionConductanceSet)
# -----------------------------------------------------------------------------
# Gate kinetics and initial conditions
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GateKineticsSettings:
    voltage_shift_mV: float = 0.0
    minimum_time_constant_ms: float = 1.0e-6
    clamp_gate_range: bool = True
    use_rush_larsen: bool = True
    use_temperature_scaling: bool = True
    q10: float = 2.0
    reference_celsius: float = 24.0
    temperature_celsius: float = 37.0

@dataclass(frozen=True)
class InitialGates:
    m: float = 0.05
    h: float = 0.60
    n: float = 0.32
# -----------------------------------------------------------------------------
# Geometry defaults
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GeometryDefaults:
    node_length_um: float = 1.0
    internode_length_um: float = 800.0
    axon_diameter_um: float = 7.0
    fiber_diameter_um: float = 10.0
    use_diameter_based_internode_length: bool = True
    internode_length_per_fiber_diameter: float = 80.0
    min_internode_length_um: float = 300.0
    max_internode_length_um: float = 1200.0
    use_total_length_driven_topology: bool = True
    myelin_wrap_count: int = 80
    periaxonal_space_nm: float = 15.0
    segment_length_um: float = 10.0
    total_length_mm: float = 20.0
    fiber_spacing_um: float = 25.0
    bundle_radius_um: float = 150.0
    domain_padding_um: float = 200.0
    raster_dx_um: float = 5.0
    raster_dy_um: float = 5.0
    raster_dz_um: float = 10.0
# -----------------------------------------------------------------------------
# Extracellular / KNP placeholders
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ECSDefaults:
    tortuosity: float = 1.6
    volume_fraction: float = 0.20
    enable_knp: bool = True
    enable_vc: bool = True
    conductivity_S_per_m: float = 0.30
    diffusivity_scale_factor: float = 1.0 / (1.6 ** 2)
    domain_x_um: float = 20000.0
    domain_y_um: float = 1000.0
    domain_z_um: float = 1000.0
# -----------------------------------------------------------------------------
# Electrode and stimulation defaults
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ElectrodeDefaults:
    contact_radius_um: float = 50.0
    contact_spacing_um: float = 250.0
    electrode_conductivity_S_per_m: float = 1.0e6
    interface_impedance_ohm_cm2: float = 1000.0

@dataclass(frozen=True)
class StimulusDefaults:
    pulse_amplitude_uA: float = 0.003
    pulse_start_ms: float = 1.0
    pulse_width_ms: float = 0.8
    pulse_amplitude_uA_per_cm2: float = 1000.0
    pulse_count: int = 1
    pulse_interval_ms: float = 5.0
    phase2_amplitude_ratio: float = 1.0
    point_source_current_uA: float = 0.0

@dataclass(frozen=True)
class KNPDefaults:
    enabled: bool = True
    source_volume_fraction_scaling: bool = True
    clearance_enabled: bool = True
    dx_um: float = 10.0
    dt_ms: float = 0.1
    t_stop_ms: float = 50.0
    domain_length_um: float = 20000.0
    floor_concentration_mM: float = 1.0e-9
    clearance_tau_K_ms: float = 500.0
    clearance_tau_Na_ms: float = 1000.0
    clearance_tau_Cl_ms: float = 1000.0
    clearance_tau_Ca_ms: float = 1000.0

    perturb_species: str = "K"
    perturb_center_um: float = 10000.0
    perturb_width_um: float = 100.0
    perturb_amplitude_mM: float = 0.05
    coupling_mode: str = "physiologic"
    source_scale_physiologic_mM_per_ms_per_uA: float = 1.0e-5
    source_scale_visibility_mM_per_ms_per_uA: float = 1.0e-2
    
    feedback_apply_phi_offset: bool = True
    feedback_apply_reversal_updates: bool = True
    feedback_subtract_initial_phi_reference: bool = True
    feedback_phi_gain: float = 1.0
    feedback_phi_clamp_abs_mV: float = 100.0
    feedback_sampling_mode: str = "linear"
    feedback_coordinate_mode: str = "auto"
    feedback_reversal_reduction: str = "mean"
    feedback_reversal_clamp_abs_shift_mV: float = 50.0
# -----------------------------------------------------------------------------
# Solver / recording / validation defaults
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverDefaults:
    dt_fast_ms: float = 0.005
    dt_slow_ms: float = 0.100
    t_stop_ms: float = 10.0
    sample_every_n_fast_steps: int = 10
    abstol: float = 1.0e-9
    reltol: float = 1.0e-6
    max_steps: int = 5_000_000

@dataclass(frozen=True)
class RecordingDefaults:
    record_voltage: bool = True
    record_gates: bool = True
    record_currents: bool = True
    record_every_n_steps: int = 1
    output_directory: str = "outputs"
    csv_filename: str = "trace.csv"

@dataclass(frozen=True)
class ValidationDefaults:
    spike_detection_threshold_mV: float = 0.0
    expected_resting_range_mV: Tuple[float, float] = (-90.0, -50.0)
    expected_peak_min_mV: float = 10.0
    expected_peak_max_mV: float = 80.0
    expected_ap_width_range_ms: Tuple[float, float] = (0.2, 5.0)
    expected_velocity_range_m_per_s: Tuple[float, float] = (1.0, 120.0)
# -----------------------------------------------------------------------------
# Region / material schema for the future freehand modeler
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialRegionDefaults:
    name: str = "generic_extracellular_space"
    conductivity_S_per_m: float = 0.30
    tortuosity: float = 1.6
    volume_fraction: float = 0.20
    relative_permittivity: float = 74.0
    is_conductive: bool = True
    is_membrane: bool = False
    is_electrode: bool = False
# -----------------------------------------------------------------------------
# Electrophysiology / VC / topology
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ElectrophysiologyPolicy:
    use_nernst_reversal_from_ion_pools: bool = True
    use_temperature_scaled_kinetics: bool = True
    use_dynamic_ecs_feedback: bool = True
    use_dynamic_ics_feedback: bool = False

@dataclass(frozen=True)
class VolumeConductionDefaults:
    enabled: bool = True
    source_model: str = "point_source"
    sigma_x_S_per_m: float = 0.30
    sigma_y_S_per_m: float = 0.30
    sigma_z_S_per_m: float = 0.30
    singularity_epsilon_um: float = 1.0
    include_electrode_contacts: bool = False

@dataclass(frozen=True)
class TopologyDefaults:
    node_count: int = 21
    include_paranodes: bool = False
    include_juxtaparanodes: bool = False
    sealed_end_boundary: bool = True
    use_uniform_internodes: bool = True

@dataclass(frozen=True)
class RegionMembraneSet:
    membrane_capacitance_uF_per_cm2: float = 1.0
    specific_membrane_resistance_ohm_cm2: float = 1000.0

@dataclass(frozen=True)
class RegionalMembraneDefaults:
    node: RegionMembraneSet = field(
        default_factory=lambda: RegionMembraneSet(
            membrane_capacitance_uF_per_cm2=1.0,
            specific_membrane_resistance_ohm_cm2=2000.0,
        )
    )
    internode: RegionMembraneSet = field(
        default_factory=lambda: RegionMembraneSet(
            membrane_capacitance_uF_per_cm2=0.01,
            specific_membrane_resistance_ohm_cm2=1.0e5,
        )
    )
    paranode: RegionMembraneSet = field(default_factory=RegionMembraneSet)
    juxtaparanode: RegionMembraneSet = field(default_factory=RegionMembraneSet)
# -----------------------------------------------------------------------------
# Aggregate immutable parameter bundle
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationParameters:
    preset: ModelPresetDefaults = field(default_factory=ModelPresetDefaults)
    topology: TopologyDefaults = field(default_factory=TopologyDefaults)
    policy: ElectrophysiologyPolicy = field(default_factory=ElectrophysiologyPolicy)

    physical: PhysicalConstants = field(default_factory=PhysicalConstants)
    temperature: TemperatureSettings = field(default_factory=TemperatureSettings)
    ions: IonPool = field(default_factory=IonPool)

    membrane: MembraneDefaults = field(default_factory=MembraneDefaults)
    conductance: ConductanceDefaults = field(default_factory=ConductanceDefaults)
    regional_conductance: RegionalConductanceDefaults = field(default_factory=RegionalConductanceDefaults)
    regional_membrane: RegionalMembraneDefaults = field(default_factory=RegionalMembraneDefaults)

    kinetics: GateKineticsSettings = field(default_factory=GateKineticsSettings)
    initial_gates: InitialGates = field(default_factory=InitialGates)

    geometry: GeometryDefaults = field(default_factory=GeometryDefaults)
    ecs: ECSDefaults = field(default_factory=ECSDefaults)
    vc: VolumeConductionDefaults = field(default_factory=VolumeConductionDefaults)
    knp: KNPDefaults = field(default_factory=KNPDefaults)

    electrode: ElectrodeDefaults = field(default_factory=ElectrodeDefaults)
    stimulus: StimulusDefaults = field(default_factory=StimulusDefaults)

    solver: SolverDefaults = field(default_factory=SolverDefaults)
    recording: RecordingDefaults = field(default_factory=RecordingDefaults)
    validation: ValidationDefaults = field(default_factory=ValidationDefaults)

    material_region_default: MaterialRegionDefaults = field(default_factory=MaterialRegionDefaults)

def build_medium_myelinated_physiology_params(
    base: SimulationParameters | None = None,
) -> SimulationParameters:
    """Return a preset-aligned physiological baseline for a medium myelinated fiber."""
    params = SimulationParameters() if base is None else base

    preset = replace(
        params.preset,
        active_preset_name="mammalian_myelinated_medium_fiber",
        description=(
            "Approximate medium-diameter mammalian myelinated fiber baseline "
            "used as the default NeuroSolver preset."
        ),
        runtime_enable_vc=True,
        runtime_enable_knp=True,
        runtime_enable_ecs_feedback=True,
    )

    geometry = replace(
        params.geometry,
        axon_diameter_um=7.0,
        fiber_diameter_um=10.0,
        node_length_um=1.0,
        use_diameter_based_internode_length=True,
        internode_length_per_fiber_diameter=80.0,
        min_internode_length_um=300.0,
        max_internode_length_um=1200.0,
        use_total_length_driven_topology=True,
        total_length_mm=20.0,
        segment_length_um=10.0,
    )

    topology = replace(
        params.topology,
        node_count=21,
        use_uniform_internodes=True,
    )

    membrane = replace(
        params.membrane,
        axial_resistivity_ohm_cm=70.0,
        leak_reversal_mV=-70.0,
    )

    temperature = replace(
        params.temperature,
        celsius=37.0,
        use_temperature_scaling=True,
    )

    kinetics = replace(
        params.kinetics,
        use_temperature_scaling=True,
        temperature_celsius=37.0,
    )

    policy = replace(
        params.policy,
        use_nernst_reversal_from_ion_pools=True,
        use_temperature_scaled_kinetics=True,
        use_dynamic_ecs_feedback=True,
    )

    vc = replace(params.vc, enabled=True)
    knp = replace(
        params.knp,
        enabled=True,
        coupling_mode="physiologic",
        domain_length_um=geometry.total_length_mm * 1000.0,
        perturb_center_um=0.5 * geometry.total_length_mm * 1000.0,
        perturb_width_um=100.0,
        perturb_amplitude_mM=0.05,
        clearance_enabled=True,
        source_volume_fraction_scaling=True,
        clearance_tau_K_ms=500.0,
        clearance_tau_Na_ms=1000.0,
        clearance_tau_Cl_ms=1000.0,
        clearance_tau_Ca_ms=1000.0,
    )
    ecs = replace(params.ecs, enable_knp=True, enable_vc=True, domain_x_um=geometry.total_length_mm * 1000.0)
    stimulus = replace(
        params.stimulus,
        pulse_amplitude_uA=0.003,
        pulse_width_ms=0.8,
        pulse_amplitude_uA_per_cm2=1000.0,
    )

    return replace(
        params,
        preset=preset,
        topology=topology,
        policy=policy,
        temperature=temperature,
        membrane=membrane,
        kinetics=kinetics,
        geometry=geometry,
        ecs=ecs,
        vc=vc,
        knp=knp,
        stimulus=stimulus,
    )

DEFAULT_PARAMS = build_medium_myelinated_physiology_params()
# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def celsius_to_kelvin(celsius: float) -> float:
    return celsius + DEFAULT_PARAMS.physical.absolute_zero_C

def thermal_voltage_volts(
    temperature_celsius: float,
    gas_constant_J_per_molK: float = DEFAULT_PARAMS.physical.gas_constant_J_per_molK,
    faraday_C_per_mol: float = DEFAULT_PARAMS.physical.faraday_C_per_mol,
) -> float:
    T_kelvin = celsius_to_kelvin(temperature_celsius)
    return gas_constant_J_per_molK * T_kelvin / faraday_C_per_mol

def nernst_potential_mV(
    z: int,
    c_out_mM: float,
    c_in_mM: float,
    temperature_celsius: float = DEFAULT_PARAMS.temperature.celsius,
) -> float:
    if z == 0:
        raise ValueError("Ion valence z must be nonzero for Nernst potential.")
    if c_out_mM <= 0.0 or c_in_mM <= 0.0:
        raise ValueError("Concentrations must be positive for Nernst potential.")
    psi_V = thermal_voltage_volts(temperature_celsius)
    return 1000.0 * (psi_V / z) * math.log(c_out_mM / c_in_mM)

def q10_scale_factor(
    temperature_celsius: float,
    reference_celsius: float = DEFAULT_PARAMS.temperature.reference_celsius_for_q10,
    q10: float = DEFAULT_PARAMS.temperature.q10_default,
) -> float:
    return q10 ** ((temperature_celsius - reference_celsius) / 10.0)

def params_to_nested_dict(params: SimulationParameters = DEFAULT_PARAMS) -> Dict[str, Any]:
    return asdict(params)

def summarize_reversal_potentials_mV(params: SimulationParameters = DEFAULT_PARAMS) -> Dict[str, float]:
    return {
        "E_Na_mV": nernst_potential_mV(
            z=params.ions.sodium.valence,
            c_out_mM=params.ions.sodium.extracellular_mM,
            c_in_mM=params.ions.sodium.intracellular_mM,
            temperature_celsius=params.temperature.celsius,
        ),
        "E_K_mV": nernst_potential_mV(
            z=params.ions.potassium.valence,
            c_out_mM=params.ions.potassium.extracellular_mM,
            c_in_mM=params.ions.potassium.intracellular_mM,
            temperature_celsius=params.temperature.celsius,
        ),
        "E_Cl_mV": nernst_potential_mV(
            z=params.ions.chloride.valence,
            c_out_mM=params.ions.chloride.extracellular_mM,
            c_in_mM=params.ions.chloride.intracellular_mM,
            temperature_celsius=params.temperature.celsius,
        ),
        "E_Ca_mV": nernst_potential_mV(
            z=params.ions.calcium.valence,
            c_out_mM=params.ions.calcium.extracellular_mM,
            c_in_mM=params.ions.calcium.intracellular_mM,
            temperature_celsius=params.temperature.celsius,
        ),
    }
# -----------------------------------------------------------------------------
# Geometry-derived helper functions
# -----------------------------------------------------------------------------
def um_to_cm(length_um: float) -> float:
    return length_um * 1.0e-4

def um_to_m(length_um: float) -> float:
    return length_um * 1.0e-6

def cylinder_lateral_area_cm2(length_um: float, diameter_um: float) -> float:
    length_cm = um_to_cm(length_um)
    diameter_cm = um_to_cm(diameter_um)
    return math.pi * diameter_cm * length_cm

def cylinder_cross_section_area_cm2(diameter_um: float) -> float:
    radius_cm = 0.5 * um_to_cm(diameter_um)
    return math.pi * radius_cm * radius_cm

def axial_resistance_ohm(length_um: float, diameter_um: float, rho_i_ohm_cm: float) -> float:
    length_cm = um_to_cm(length_um)
    area_cm2 = cylinder_cross_section_area_cm2(diameter_um)
    if area_cm2 <= 0.0:
        raise ValueError("Cylinder cross-sectional area must be positive.")
    return rho_i_ohm_cm * length_cm / area_cm2

def membrane_capacitance_uF(length_um: float, diameter_um: float, Cm_uF_per_cm2: float) -> float:
    area_cm2 = cylinder_lateral_area_cm2(length_um, diameter_um)
    return Cm_uF_per_cm2 * area_cm2

def specific_resistance_to_conductance_mS_per_cm2(Rm_ohm_cm2: float) -> float:
    if Rm_ohm_cm2 <= 0.0:
        raise ValueError("Specific membrane resistance must be positive.")
    return 1000.0 / Rm_ohm_cm2