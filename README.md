# NeuroSolver Workflow and Code Architecture

## how to run
1. open the neurosolver file in python
2. download the libraries used for the math and plotting in the file --> run: pip install numpy matplotlib pandas
3. Run the file poc_hh_knp_cases.py in turnin\neuro_sim_env final turnin\Scripts\NeuroSolver\architecture
4. wait for about 30 minutes(depends on core count). Output file is already made and has a sample run done previously if you don't want to run the code. 

NeuroSolver runs as a multirate, architecture-aware pipeline. A preset-driven parameter bundle is first loaded from `params.py`, then an explicit fiber or bundle geometry is built through the geometry and architecture modules. `architecture_runner.py` converts that architecture into runnable bundle geometry, electrode locations, and runtime metadata, and then passes execution to `scheduler.py`. The scheduler advances each fiber on the fast timescale using the membrane/cable layer (`gating.py`, `ion_channels.py`, `node_model.py`, and `cable_solver.py`), while shared extracellular readout and slow extracellular transport are handled separately by `vc_solver.py` and `knp_solver.py`. The slow extracellular state is then sampled back into the cable layer through `ecs_feedback.py`, allowing conservative feedback without forcing the entire system onto a single timestep. Finally, diagnostics, metrics, and plots are generated through the reporting and postprocessing modules.

## Full workflow version

### 1. Parameter and preset definition

The simulation begins in `params.py`. This file is the central registry for:
- physical constants,
- temperature settings,
- ionic baselines,
- membrane properties,
- conductance defaults,
- geometry defaults,
- extracellular and KNP settings,
- stimulation settings,
- solver cadence,
- preset identity.

In the current branch, the active default state is preset-driven rather than debug-only. That means the runtime starts from a physiologic myelinated-fiber baseline instead of from an arbitrary harness configuration.

### 2. Geometry and architecture construction

Single-fiber longitudinal geometry is built in `myelin_geometry.py`, which resolves:
- node and internode spacing,
- total cable length,
- diameter-dependent internode rules,
- compartment areas and axial resistances,
- bundle-embedding metadata such as `fiber_id`, `fascicle_id`, and transverse position.

Bundle-level spatial placement is assembled in `bundle_geometry.py`. Note that it does not solve any physics; it places fibers in the transverse plane and instantiates one explicit cable geometry per fiber. This is the step that turns the model from “one fiber multiplied by N” into “N real fibers with explicit coordinates.”

Architecture-level execution is then handled by `architecture_runner.py`. This file:
1. accepts the high-level nerve architecture,
2. builds bundle geometry,
3. constructs electrode/contact sample points,
4. allocates bundle runtime state,
5. calls the multirate scheduler,
6. wraps results into an integrated architecture result object,
7. provides summary functions and bundle-level field-map utilities.

### 3. Runtime state ownership

State ownership is split across:
- `state.py` for single-fiber history buffers and unified scheduler outputs,
- `bundle_state.py` for explicit per-fiber bundle runtime state and shared-domain placeholders.

This separation is important because it keeps:
- per-fiber fast states,
- shared VC state,
- shared KNP/ECS state,
- shared material state,
- diagnostics metadata

as distinct objects instead of mixing them into one ad hoc runtime dictionary.

### 4. Fast propagation layer

The fast propagation layer is made of:
- `gating.py` for gate kinetics,
- `ion_channels.py` for current equations,
- `node_model.py` for single-compartment active-node stepping,
- `cable_solver.py` for semi-implicit 1D cable propagation.

The role of this layer is to evolve membrane voltage and gating variables on the fast timescale. The current design keeps the fast propagation core per fiber, which is consistent with the project’s main design commitment: action-potential propagation should remain in the membrane/cable layer, not in the extracellular mass-transport solver.

### 5. Shared extracellular coupling layer

`coupling.py` translates membrane activity into shared-source payloads. It is the exchange layer between:
- the cable solver,
- the virtual electrode forward model,
- the KNP extracellular transport solver.

This file constructs:
- VC-ready source traces,
- KNP-ready shared source terms,
- source metadata for bundle-level aggregation.

### 6. Fast extracellular readout layer

`vc_solver.py` is the fast extracellular forward model. It computes virtual-electrode signals using a homogeneous-medium point-source approximation and supports superposition from many cable compartments and many fibers. In the current branch, this layer is intended for comparative extracellular readout rather than full final-field realism.

### 7. Slow extracellular mass-transport layer

`knp_solver.py` implements the shared 1D extracellular transport state. It owns:
- the KNP domain,
- extracellular concentration histories,
- effective diffusion scaling,
- conductivity estimation,
- the diffusive-potential-like extracellular term,
- slow source deposition,
- clearance and concentration evolution.

This layer is deliberately slower than the cable layer and is used to track extracellular loading and slow concentration change without forcing the full simulation onto a uniformly small timestep.

### 8. Slow ECS feedback layer

`ecs_feedback.py` closes the loop from the slow extracellular state back to the fast membrane layer. It samples the KNP state onto cable compartments and constructs:
- cable-facing extracellular potential offsets,
- reversal-potential overrides,
- sampled extracellular species vectors,
- feedback metadata.

This allows the membrane layer to feel the extracellular state while preserving the multirate split.

### 9. Multirate execution core

`scheduler.py` is the execution controller. It owns:
- single-fiber multirate runs,
- bundle-aware multirate runs,
- fast stepping of each fiber,
- slow KNP updates,
- VC superposition,
- ECS feedback timing,
- result and metadata packaging.

At each fast step, the scheduler advances cable states. At slower intervals, it updates the shared KNP state and re-samples feedback. VC traces are built from the fast membrane outputs and superposed across fibers when bundle mode is used.

### 10. Diagnostics, regression, and output

Validation and postprocessing are handled by:
- `regression_cases.py` for anchor cases,
- `node_diagnostics.py` for nodal propagation timing and velocity,
- `cable_diagnostics.py` for cable-level threshold and peak summaries,
- plotting and reporting modules for figures and summaries.

These modules are not part of the physics solve itself; they are the interpretation and validation layer that determines whether a run is merely executing or behaving in a physiologically meaningful way.

## Minimal file map

- `params.py` — central preset and parameter registry  
- `myelin_geometry.py` — single-fiber longitudinal geometry  
- `bundle_geometry.py` — bundle/fascicle/fiber placement  
- `bundle_state.py` — bundle runtime state containers  
- `state.py` — general runtime history/result containers  
- `architecture_runner.py` — architecture-to-simulation execution wrapper  
- `scheduler.py` — multirate control and per-fiber/shared-domain orchestration  
- `gating.py` — gate kinetics  
- `ion_channels.py` — ionic current equations  
- `node_model.py` — single-node active membrane model  
- `cable_solver.py` — fast 1D cable propagation  
- `coupling.py` — cable-to-VC/KNP source coupling  
- `vc_solver.py` — fast extracellular forward model  
- `knp_solver.py` — slow shared extracellular transport model  
- `ecs_feedback.py` — slow ECS-to-cable feedback bridge  
- `regression_cases.py` — validation anchor cases  
- `node_diagnostics.py` — nodal propagation metrics  
- `cable_diagnostics.py` — cable-level postprocessing metrics  

## Short interpretation

In practical terms, the code runs in this order:

1. define parameters and preset  
2. build fiber or bundle geometry  
3. allocate per-fiber and shared runtime state  
4. advance fast cable dynamics  
5. convert membrane activity into shared VC/KNP source terms  
6. update VC and KNP on their appropriate cadence  
7. sample the slow ECS state back onto each cable  
8. compute metrics and export plots/summaries  
