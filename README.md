# Das Eimerargument

*Newton's bucket, Weber's answer.*

A spinning charged torus in a magnetic field has a Weber bracket that deviates from unity. This deviation scales as v²/c², accumulates linearly under thrust, and does not decay. We compute it to 28 digits and measure its consequences.

## The Claim

Weber's force law between two charges includes velocity- and acceleration-dependent terms:

    F = (q₁q₂/4πε₀r²) · [1 - ṙ²/(2c²) + r·r̈/c²]

The bracketed quantity W = 1 - ṙ²/(2c²) + r·r̈/c² is the **Weber bracket**. Standard electrodynamics sets it to 1 and recovers Coulomb's law. Weber's formulation does not.

For a bismuth torus spinning at ω ≈ 5000 rad/s in a static magnetic field, this simulation computes W to 28-digit precision across every particle-source pair at every timestep. The result: **W ≠ 1**. The deviation is ~2.5 × 10⁻⁹, consistent with the analytical estimate ω²R²/c².

This deviation means the effective inertial mass of the spinning torus differs from its rest mass. Apply the same force to a spinning and a non-spinning torus — Weber predicts they accelerate differently. Newton and Maxwell predict they do not.

The simulation measures this difference directly.

## Addressing Standard Objections

The following objections are typically raised against Weber's electrodynamics. We address each, not because the objections are wrong, but because they are irrelevant to what this simulation computes.

### "Weber's theory cannot produce electromagnetic waves"

This simulation does not involve electromagnetic waves. There is no radiation, no antenna, no propagation. The question is narrower: **does the force between a stationary current loop and a moving charge depend on the charge's velocity?** Weber says yes. Coulomb says no. The difference is the bracket.

For the historical record: Weber and Kohlrausch measured the ratio of electromagnetic to electrostatic units in 1856 and found it equaled the speed of light — nine years before Maxwell's 1865 treatise. When Weber's force is applied to actual charges in conductors, wave equations with propagation speed c emerge naturally (Assis 1994, Wesley 1990). The claim that Weber "cannot produce waves" conflates the force law (which is between particles) with a field ontology (which Weber does not require).

But none of this matters here. The simulation computes a bracket. The bracket is not 1. That is the result.

### "Weber is not Lorentz covariant"

The simulation operates at v/c ~ 1.7 × 10⁻⁶. Relativistic corrections enter at order v⁴/c⁴ ~ 10⁻²⁴. The Weber correction enters at v²/c² ~ 10⁻⁹. The Weber term is **fifteen orders of magnitude larger** than any relativistic correction to it.

Demanding Lorentz covariance of a 500 m/s laboratory simulation is like demanding general relativity to weigh a fish. It is formally correct and practically meaningless.

The deeper point: Lorentz covariance is a symmetry of Maxwell's field equations. It is not an experimental result at 500 m/s. No experiment has tested electrodynamics at this velocity to 10⁻⁹ precision. That is what the simulation predicts and an experiment could measure.

### "Maxwell conserves momentum via field momentum"

True, and irrelevant. The simulation applies identical external forces to four bodies and measures their accelerations. There are no radiating fields, no Poynting flux, no field momentum in play. The question is: does the same force produce the same acceleration on a spinning vs. non-spinning body? Weber says no.

Weber's force law satisfies Newton's third law exactly at the particle level and derives from a velocity-dependent potential (Lagrangian structure). Maxwell's does not, for individual charges — it requires field degrees of freedom to close the conservation laws. Both are self-consistent within their frameworks. The question is which one matches the laboratory.

### "QED supersedes all classical electrodynamics"

QED is built on Maxwell's field framework. If Maxwell's framework contains an implicit assumption — that the bracket is unity — then QED inherits that assumption. A v²/c² correction to the force law between charges would not be visible in QED calculations that assume the standard Lagrangian.

This is not a claim that QED is wrong. It is a claim that a specific term in the pairwise force — one that Weber included and Maxwell did not — produces a measurable consequence at 28-digit precision. Whether QED can accommodate such a term is a separate question. The simulation provides the number; theory can catch up.

### "Weber was replaced by Maxwell for good reason"

Maxwell unified optics and electromagnetism. That was a genuine triumph. But unification is not the same as completeness. Maxwell's theory does not predict a velocity-dependent bracket in the Coulomb interaction. Weber's does. The question is not which theory is "better" in general — it is whether this specific prediction is physically real.

The simulation is not an argument for replacing Maxwell with Weber. It is a computation of what Weber's force law predicts for a spinning torus, carried out at sufficient precision to resolve the bracket deviation, with data attached.

## Method

Four identical torus bodies evolve under the same external forces:

| Body | Force Law | Spinning | Role |
|------|-----------|----------|------|
| A | Newton | No | Baseline |
| B | Newton | Yes | Spin control |
| C | Weber | No | Static mass correction |
| D | Weber | Yes | **The signal** |

The coupling signal is the double-difference:

    coupling = (v_D - v_B) - (v_C - v_A)

This cancels all common-mode effects (thrust, static mass correction, numerical drift). Newton predicts exactly zero. Weber predicts non-zero.

### Six-Phase Protocol

1. **Spin-up** — PD-controlled motor torques B and D to ω ≈ 5236 rad/s (50,000 RPM)
2. **Coast** — Motor off, angular momentum conserved
3. **Thrust** — Uniform force along torus symmetry axis
4. **Coast** — Force off, measure velocity
5. **Brake** — Motor drives ω → 0
6. **Coast** — Final measurement

### Precision

- **CPU path**: 28-digit decimal arithmetic (`rust_decimal` crate, no floating point)
- **GPU path**: 31-digit double-double arithmetic (pairs of f64, fused multiply-add)
- All intermediate quantities carried at full precision
- Weber bracket computed per particle per source element per timestep — no approximations, no perturbation theory

### Geometry

- **Torus**: major radius 0.10 m, minor radius 0.03 m
- **Bismuth preset**: density 9780 kg/m³, diamagnetic susceptibility χ = −1.7 × 10⁻⁴ (the most strongly diamagnetic element)
- **Particles**: 50 point charges connected by spring-damper bond network (spatial hash, cutoff 2.5× mean spacing)
- **Magnets**: coaxial current loops (Biot-Savart source elements), 768 total from 2 stacks of 12 loops × 32 segments
- **Rigid rotation**: enforced analytically each step to prevent numerical spin decay from bond forces

The torus geometry is essential. A flat ring spinning about its symmetry axis has all particle velocities perpendicular to the radial direction toward coaxial sources — the radial velocity ṙ vanishes by symmetry and W = 1 identically. The torus breaks this degeneracy: particles on the minor radius have velocity components both parallel and perpendicular to source elements at different positions.

## Key Results

From `data/brake_recoil_gpu_single.csv` (18 data points, 17,000 steps at dt = 1 μs):

### 1. The bracket is not unity

Mean Weber bracket deviation: **~2.5 × 10⁻⁹** at ω ≈ 5000 rad/s.

For particles on a torus spinning at ω with major radius R = 0.10 m:

    v = ωR ≈ 524 m/s
    v²/c² ≈ 3 × 10⁻¹²

The deviation is larger (~10⁻⁹) because the acceleration-dependent r·r̈/c² term from centripetal acceleration dominates for circular motion. This is not a surprise — it is Weber's explicit prediction.

### 2. ω² scaling

The bracket deviation tracks ω². During braking (phase 5), ω drops from 5236 → 4662 rad/s and the mean bracket falls from 2.8 × 10⁻⁹ to 2.5 × 10⁻⁹.

### 3. 1/c² scaling

The entire correction is O(v²/c²). At v ~ 500 m/s this is ~10⁻¹² per source element, accumulating across 768 elements to produce ~10⁻⁹ total.

### 4. Linear accumulation under thrust

The coupling signal grows linearly during phase 3 (thrust): from ~10⁻¹⁶ to ~3.5 × 10⁻¹⁴ m/s over 5000 steps. Equal force, unequal effective mass, linear velocity divergence. This is the predicted signature of a mass correction.

### 5. No decay

After thrust ends (phases 4–6), the coupling signal holds at ~4 × 10⁻¹⁴ m/s. The velocity difference persists. It is not a transient, not a resonance, not numerical drift.

### 6. Bismuth amplification

The `--bismuth3` preset runs three counter-rotating bismuth tori with four magnet stacks. Bismuth's high density (9780 kg/m³) and strong diamagnetic response (χ = −1.7 × 10⁻⁴) make it the natural material for this test — the same material that levitates in strong magnetic fields, and for the same electromagnetic reason.

## Data

`data/brake_recoil_gpu_single.csv` — 31-digit GPU simulation, single torus, z-axis thrust.

| Column | Description |
|--------|-------------|
| `step`, `time`, `phase` | Simulation state |
| `A_vz`, `B_vz`, `C_vz`, `D_vz` | Center-of-mass velocity along thrust axis |
| `B_omega`, `D_omega` | Angular velocity (rad/s) |
| `D_dm_vel` | Velocity-dependent mass correction |
| `D_mean_bracket` | Mean (W − 1) across all particle-source pairs |
| `D_bracket_coupling` | Bracket coupling diagnostic |
| `coupling_signal` | Double-difference (v_D − v_B) − (v_C − v_A) |

## Falsification

This is a falsifiable prediction. The following would invalidate the results:

1. **Bracket unity**: If an independent implementation computes W = 1.000...0 to 28+ digits for spinning particles near current-carrying loops, this simulation has a bug.

2. **Symmetry proof**: If a mathematical proof shows the Weber bracket must equal unity for any axially symmetric spinning body in any static magnetic field, the geometry argument is wrong.

3. **dt dependence**: If the coupling signal scales with the timestep rather than with ω² and 1/c², it is numerical noise, not physics.

4. **Sign test**: Reverse ω. If the coupling reverses sign, the effect is real (velocity-dependent). If it doesn't, it's a bug. Counter-rotation test: `--stacked`.

5. **Magnitude check**: The bracket deviation should be approximately |W − 1| ≈ ω²R²/c² ≈ (5000)²(0.1)²/(3×10⁸)² ≈ 2.8 × 10⁻⁹. If the simulation reports a value orders of magnitude different, something is wrong.

6. **Experimental test**: Spin a bismuth torus at 50,000 RPM in a known magnetic field. Apply a calibrated impulse. Measure the acceleration to 10⁻⁹ relative precision. Compare spinning vs. non-spinning. This is the actual test. The simulation predicts the result.

## Build

Requires Rust 1.70+ and optionally CUDA Toolkit 12.x for GPU acceleration.

```bash
# CPU only (28-digit decimal)
cargo build --release

# With GPU (31-digit double-double on CUDA)
cargo build --release --features gpu
```

### Run

```bash
# Four-body comparison (default geometry)
cargo run --release

# Brake recoil — Faraday's Protocol
cargo run --release --bin brake-recoil

# GPU accelerated
cargo run --release --features gpu --bin brake-recoil

# Three bismuth tori, counter-rotating
cargo run --release --features gpu --bin brake-recoil -- --bismuth3

# Stacked two-torus geometry
cargo run --release --features gpu --bin brake-recoil -- --stacked

# Custom configuration
cargo run --release --features gpu --bin brake-recoil -- --config myconfig.json
```

### Other Binaries

- `weber-anomaly` — Main four-body simulation (Newton A/B vs Weber C/D)
- `brake-recoil` — Six-phase Faraday Protocol with motor spin-up/brake
- `debug-weber` — Single-particle Weber force diagnostic
- `shake-test` — Lattice disorder and bond network integrity test
- `field-residual` — Static B_weber − B_maxwell field probe

## Structure

```
src/
  weber.rs        — Weber force: bracket, longitudinal force, effective mass
  vec3.rs         — 3D vector math at 28-digit Decimal precision
  torus.rs        — Torus geometry, spring-damper bonds, rigid body diagnostics
  magnets.rs      — Biot-Savart source element discretization
  integrator.rs   — Velocity Verlet for Newton and Weber, torque distribution
  motor.rs        — PD-controlled motor with torque saturation
  config.rs       — JSON config, CLI args, material presets, protocol
  lib.rs          — Module declarations
  main.rs         — weber-anomaly: four-body comparison
  bin/
    brake_recoil.rs   — Faraday Protocol: spin-thrust coupling
    debug_weber.rs    — Single-particle force diagnostic
    shake_test.rs     — Bond network integrity test
    field_residual.rs — Static field residual probe
  gpu/
    mod.rs        — CUDA kernel dispatch via cudarc
    dd.rs         — Double-double 31-digit arithmetic
kernels/
    dd_math.cuh           — DD arithmetic (GPU header)
    weber_kernel.cu       — Weber force kernel
    bond_kernel.cu        — Bond force kernel
    integrate_kernel.cu   — Verlet integration kernel
    field_residual_kernel.cu — Field residual kernel
data/
    brake_recoil_gpu_single.csv — Reference output
```

## Historical Note

Wilhelm Weber published his force law in 1846. Weber and Kohlrausch measured the ratio of electromagnetic to electrostatic units in 1856 and found it equaled the speed of light. Maxwell published his field equations in 1865.

Weber's force law derives from a velocity-dependent potential. It satisfies Newton's third law exactly. It conserves energy and momentum by construction (Lagrangian structure). It predicts that the effective mass of a body depends on its electromagnetic environment — what Assis calls "relational mechanics" in the tradition of Mach.

Maxwell's theory distributes momentum between particles and fields. It predicts electromagnetic waves in vacuum. It is Lorentz covariant. It generalizes into QED.

These are different theories. They agree in many regimes. They disagree on whether the bracket is unity. This simulation computes the bracket. The data is attached.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this simulation in research:

> Campbell, K. (2026). *Das Eimerargument*: Weber electrodynamics simulation
> at 28-digit precision — bracket deviation, spin-thrust coupling, and
> effective mass in toroidal geometry.
> https://github.com/DeepBlueDynamics/das-eimerargument
