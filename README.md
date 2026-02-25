# Weber Electrodynamics Simulation

A spinning charged torus in a magnetic field has a Weber bracket that deviates from unity. This deviation scales as v²/c², accumulates linearly under thrust, and does not decay.

## The Claim

Weber's force law predicts that the effective inertial mass of a body depends on its electromagnetic environment and its velocity relative to nearby source charges. For a torus spinning at angular velocity ω in a static magnetic field, the Weber bracket

    W = 1 - ṙ²/(2c²) + r·r̈/c²

deviates from 1 by approximately ω²R²/c², where R is the torus major radius. This deviation is unmeasurable in standard Newtonian electrodynamics (where the bracket is always unity by construction). The simulation computes this deviation to 28-digit precision and measures its consequences.

## Method

Four identical torus bodies evolve under the same external forces:

| Body | Force Law | Spinning | Role |
|------|-----------|----------|------|
| A | Newton | No | Baseline |
| B | Newton | Yes | Spin control |
| C | Weber | No | Static mass correction |
| D | Weber | Yes | **The signal** |

The coupling signal is defined as:

    coupling = (v_D - v_B) - (v_C - v_A)

Newton predicts this is exactly zero. Weber predicts it is not.

### Six-Phase Protocol

1. **Spin-up** — PD-controlled motor torques bodies B and D to ω ≈ 5236 rad/s (50,000 RPM)
2. **Coast** — Motor off, angular momentum conserved
3. **Thrust** — Uniform force applied to all four bodies along torus symmetry axis
4. **Coast** — Force off, measure velocity
5. **Brake** — Motor drives ω → 0
6. **Coast** — Final measurement

### Precision

- **CPU path**: 28-digit decimal arithmetic (`rust_decimal` crate, no floating point)
- **GPU path**: 31-digit double-double arithmetic (pairs of f64, fused multiply-add)
- All intermediate quantities (positions, velocities, accelerations, forces) carried at full precision
- Weber bracket W computed per particle per source element per timestep — no approximations

### Geometry

- Torus: major radius 0.10 m, minor radius 0.03 m
- Particles: 50 point charges connected by spring-damper bond network
- Bond network: spatial hash, cutoff 2.5× mean spacing
- Magnets: coaxial current loops (Biot-Savart source elements), 768 total from 2 stacks of 12 loops × 32 segments
- Rigid rotation enforced analytically each step (`reimpose_rigid_rotation`) to prevent numerical spin decay from bond forces

## Key Results

From `data/brake_recoil_gpu_single.csv` (18 data points, 17,000 steps at dt = 1 μs):

### 1. The bracket is not unity

Mean Weber bracket deviation from 1: **~2.5 × 10⁻⁹** at ω ≈ 5000 rad/s.

This is the ṙ²/(2c²) term. For particles on a torus spinning at ω with major radius R = 0.10 m:

    v = ωR ≈ 524 m/s
    v²/c² ≈ 3 × 10⁻¹²

The bracket deviation is larger (~10⁻⁹) because it includes the acceleration-dependent r·r̈/c² term from centripetal acceleration, which dominates for circular motion.

### 2. ω² scaling

The bracket deviation scales with ω². As the motor brakes (phase 5), ω drops from 5236 → 4662 rad/s, and the mean bracket tracks this decline: from 2.8 × 10⁻⁹ down to 2.5 × 10⁻⁹.

### 3. 1/c² scaling

The entire correction is O(v²/c²). At laboratory velocities (v ~ 500 m/s), this is ~10⁻¹² per term, accumulating across hundreds of source elements to produce the ~10⁻⁹ bracket deviation.

### 4. Planar degeneracy breaking

A flat ring spinning about its symmetry axis has all particle velocities perpendicular to the radial direction toward coaxial sources. The radial velocity ṙ = v · R̂ vanishes by symmetry, giving W = 1 identically.

A **torus** breaks this degeneracy: particles on the minor radius have velocity components both parallel and perpendicular to source elements at different angular positions. This is why the torus geometry is essential.

### 5. Linear accumulation during thrust

The coupling signal grows linearly during phase 3 (thrust): from ~10⁻¹⁶ to ~3.5 × 10⁻¹⁴ m/s over 5000 thrust steps. Equal force, unequal effective mass, linear velocity divergence.

### 6. No decay

After thrust ends (phases 4-6), the coupling signal holds at ~4 × 10⁻¹⁴ m/s. The velocity difference, once established, persists — it is not a transient or numerical artifact.

## Data

`data/brake_recoil_gpu_single.csv` — 31-digit GPU simulation, single torus, z-axis thrust.

Columns:
- `step`, `time`, `phase` — simulation state
- `A_vz`, `B_vz`, `C_vz`, `D_vz` — center-of-mass velocity along thrust axis for each body
- `B_omega`, `D_omega` — angular velocity (rad/s)
- `D_dm_vel` — velocity-dependent mass correction for Weber body D
- `D_mean_bracket` — mean (W - 1) across all particle-source pairs
- `D_bracket_coupling` — bracket coupling diagnostic
- `coupling_signal` — the double-difference (v_D - v_B) - (v_C - v_A)

## Falsification

This simulation is falsifiable. The following would invalidate the results:

1. **Bracket unity**: If an independent implementation computes W = 1.000...0 to 28+ digits for spinning particles near current-carrying loops, the simulation has a bug.

2. **Symmetry argument**: If a mathematical proof shows the Weber bracket must equal unity for any axially symmetric spinning body in any static magnetic field, the geometry argument is wrong.

3. **Numerical artifact**: If the coupling signal scales with dt (timestep) rather than with ω² and 1/c², it is numerical noise, not physics.

4. **Sign error**: If the sign of the coupling reverses when ω reverses, the effect is real. If it doesn't, it's a bug. (Counter-rotation test available via `--stacked` flag.)

5. **Magnitude check**: The bracket deviation should be approximately:

       |W - 1| ≈ ω²R²/c² ≈ (5000)²(0.1)²/(3×10⁸)² ≈ 2.8 × 10⁻⁹

   If the simulation reports a value orders of magnitude different, something is wrong.

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
# Four-body comparison (main binary)
cargo run --release

# Brake recoil test (Faraday's Protocol)
cargo run --release --bin brake-recoil

# With GPU
cargo run --release --features gpu --bin brake-recoil

# Multi-torus stacked geometry
cargo run --release --features gpu --bin brake-recoil -- --stacked

# Three bismuth tori
cargo run --release --features gpu --bin brake-recoil -- --bismuth3

# Custom config
cargo run --release --features gpu --bin brake-recoil -- --config myconfig.json
```

### Other Binaries

- `weber-anomaly` — Main four-body simulation (Newton A/B vs Weber C/D)
- `brake-recoil` — Six-phase Faraday Protocol with motor spin-up/brake
- `debug-weber` — Single-particle Weber force diagnostic
- `shake-test` — Lattice disorder and bond network integrity test
- `field-residual` — Static B_weber - B_maxwell field probe

## Structure

```
src/
  weber.rs        — Weber force law: bracket, longitudinal force, effective mass
  vec3.rs         — 3D vector math at 28-digit Decimal precision
  torus.rs        — Torus geometry, spring-damper bonds, rigid body diagnostics
  magnets.rs      — Biot-Savart source element discretization
  integrator.rs   — Velocity Verlet for Newton and Weber, torque distribution
  motor.rs        — PD-controlled motor with max torque saturation
  config.rs       — JSON config, CLI args, material presets, six-phase protocol
  lib.rs          — Module declarations
  main.rs         — weber-anomaly binary: four-body comparison
  bin/
    brake_recoil.rs   — Faraday Protocol: spin-thrust coupling measurement
    debug_weber.rs    — Single-particle force diagnostic
    shake_test.rs     — Bond network integrity test
    field_residual.rs — Static field residual probe
  gpu/
    mod.rs        — CUDA kernel dispatch via cudarc, device memory management
    dd.rs         — Double-double 31-digit arithmetic (DD struct)
kernels/
    dd_math.cuh           — DD arithmetic for GPU (header)
    weber_kernel.cu       — Weber force computation kernel
    bond_kernel.cu        — Spring-damper bond force kernel
    integrate_kernel.cu   — Velocity Verlet integration kernel
    field_residual_kernel.cu — Static field residual kernel
data/
    brake_recoil_gpu_single.csv — Reference simulation output
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this simulation in research, please cite:

> Campbell, K. (2026). Weber electrodynamics simulation at 28-digit precision:
> bracket deviation, spin-thrust coupling, and effective mass in toroidal geometry.
> https://github.com/DeepBlueDynamics/das-eimerargument
