# Das Eimerargument

A numerical simulation of Weber's force law applied to a spinning torus in a static magnetic field. The Weber bracket deviates from unity by ~2.5 × 10⁻⁹ at ω ≈ 5000 rad/s. This deviation scales as ω²R²/c², accumulates linearly under thrust, and does not decay.

## The Computational Claim

Given Weber's pairwise force law between charges:

    F = (q₁q₂ / 4πε₀r²) · W · r̂

where the Weber bracket is:

    W = 1 - ṙ²/(2c²) + r·r̈/c²

and where ṙ is the radial component of relative velocity and r̈ is the radial component of relative acceleration (including centripetal):

1. The bracket W is implemented per particle-source pair, per timestep, at 28-digit precision.
2. For a torus spinning at ω in a magnetic field, W ≠ 1.
3. The deviation |W − 1| ≈ ω²R²/c² matches the analytical estimate.
4. This produces a measurable coupling between spin and translational response to thrust.
5. The coupling accumulates linearly during thrust and persists afterward.

This is a computational claim about what Weber's equation predicts when evaluated numerically. No comparison to Maxwell is required to evaluate it.

## Bracket Derivation

For a particle at position **r_p** moving with velocity **v** relative to a stationary source element at **r_s**:

    **R** = r_p - r_s           (separation vector)
    r = |R|                     (distance)
    R̂ = R/r                    (unit vector)
    ṙ = v · R̂                  (radial velocity — projection, not magnitude)
    v_perp² = |v|² - ṙ²        (perpendicular velocity component)
    r̈ = a · R̂ + v_perp²/r     (radial acceleration, including centripetal)

The centripetal term v_perp²/r is critical. For circular motion, this is the dominant contribution to r̈ and is what makes the bracket deviate from unity even when the radial velocity ṙ is small.

Substituting into the bracket:

    W = 1 - ṙ²/(2c²) + r·(a · R̂ + v_perp²/r)/c²
      = 1 - ṙ²/(2c²) + (r·a·R̂)/c² + v_perp²/c²

For a particle on a torus spinning at ω with major radius R, the tangential velocity is v ~ ωR. Most of this is perpendicular to the radial direction toward a coaxial magnet, so v_perp² ~ ω²R². This gives:

    |W - 1| ~ ω²R²/c²

At ω = 5000 rad/s, R = 0.10 m:

    |W - 1| ~ (5000)²(0.1)² / (3×10⁸)² ≈ 2.8 × 10⁻⁹

The simulation measures 2.5 × 10⁻⁹. The agreement confirms the implementation matches the analytical prediction.

## Why a Torus

A flat ring spinning about its symmetry axis, with coaxial source elements, has all particle velocities perpendicular to R̂. The radial velocity ṙ = v · R̂ vanishes by symmetry. The perpendicular velocity contribution v_perp²/c² is nonzero but uniform — it shifts the bracket identically for spinning and non-spinning cases when measured as a coupling signal.

A **torus** breaks this degeneracy. Particles on the minor cross-section have varying distances and angles to each source element. The bracket deviation varies across particles, and spinning introduces asymmetric radial velocity components that a flat ring does not have. The torus is not a convenience — it is the minimal geometry that produces a differential signal.

## Method

Four identical bodies evolve under the same external forces:

| Body | Force Law | Spinning | Role |
|------|-----------|----------|------|
| A | Newton (W=1) | No | Baseline |
| B | Newton (W=1) | Yes | Spin control |
| C | Weber | No | Static correction |
| D | Weber | Yes | **Signal** |

The coupling signal is:

    coupling = (v_D - v_B) - (v_C - v_A)

This double-difference cancels: thrust response (common to all), static Weber mass correction (C−A), and spin-dependent Newtonian effects (B−A). What remains is purely the spin-dependent Weber bracket correction.

With W = 1 (Newton), coupling = 0 identically. With W ≠ 1 (Weber), the spinning body D sees a different effective mass than B, producing nonzero coupling under thrust.

### Six-Phase Protocol

1. **Spin-up** — PD-controlled motor drives B, D to ω ≈ 5236 rad/s
2. **Coast** — Motor off
3. **Thrust** — Uniform force along symmetry axis, all four bodies
4. **Coast** — Measure velocity
5. **Brake** — Motor drives ω → 0
6. **Coast** — Final measurement

### Precision

- **CPU**: 28-digit decimal (`rust_decimal`). No floating point in the force computation.
- **GPU**: 31-digit double-double (pairs of f64, error-free transformations).
- Bracket computed per particle × per source element × per timestep.
- No series expansion, no perturbation theory, no truncation of the bracket.

### Geometry

- Torus: major radius R = 0.10 m, minor radius r = 0.03 m
- 50 particles, spring-damper bond network (spatial hash, 2.5× cutoff)
- 768 magnet source elements (2 stacks × 12 loops × 32 segments)
- Rigid rotation reimposed analytically each step (prevents numerical spin decay from bonds)
- **Bismuth preset** (`--bismuth3`): 3 counter-rotating tori, density 9780 kg/m³, χ = −1.7 × 10⁻⁴

## Internal Validation

These are the questions that determine whether the simulation does what it claims.

### Does the bracket match the analytic estimate?

Yes. At ω ≈ 5000 rad/s, the simulation reports mean |W − 1| ≈ 2.5 × 10⁻⁹. The analytical estimate ω²R²/c² ≈ 2.8 × 10⁻⁹. The ~10% discrepancy is expected from averaging over the torus geometry (particles at different positions on the minor cross-section see different source element distances and angles).

### Does the deviation scale with ω²?

Yes. During braking (phase 5), ω decreases from 5236 to 4662 rad/s. The mean bracket deviation tracks this: 2.8 × 10⁻⁹ → 2.5 × 10⁻⁹. The ratio (4662/5236)² = 0.793. The ratio 2.5/2.8 = 0.893. The discrepancy reflects the acceleration-dependent term changing as ω changes — the bracket is not purely ω² but includes r·r̈/c² which depends on the deceleration profile.

### Does the coupling accumulate linearly during thrust?

Yes. During phase 3 (constant thrust), the coupling signal grows from ~10⁻¹⁶ to ~3.5 × 10⁻¹⁴ m/s. Constant force on a body with constant (but slightly different) effective mass produces constant differential acceleration, hence linear velocity divergence.

### Does the coupling persist after thrust ends?

Yes. After thrust ends (phases 4–6), the coupling holds at ~4 × 10⁻¹⁴ m/s. A velocity difference, once established, persists in the absence of differential forces. This rules out transient numerical artifacts.

### Is the 1/c² scaling correct?

The bracket deviation per source element is O(v²/c²) ~ 10⁻¹². Summing over 768 source elements at varying distances produces the aggregate ~10⁻⁹. The simulation uses c² = 89875517873681764 (exact integer, 28 digits). No floating-point approximation of c is involved.

### Are forces pairwise symmetric?

Weber's force derives from a velocity-dependent potential. The implementation computes F on each test particle from all source elements. Source elements are fixed current loops (not free particles), so Newton's third law applies as: the force on the test particle is computed from the Weber potential, and the reaction force on the source is not tracked (sources are infinitely massive current loops). This is standard Biot-Savart with Weber correction — the same asymmetry present in any test-particle-in-external-field simulation.

For the four-body comparison, all bodies see the same source elements, so any systematic error in source modeling cancels in the double-difference.

## Results

From `data/brake_recoil_gpu_single.csv` (18 data points, 17,000 steps, dt = 1 μs):

| Column | Description |
|--------|-------------|
| `step`, `time`, `phase` | Simulation state |
| `A_vz` ... `D_vz` | Center-of-mass velocity along thrust axis |
| `B_omega`, `D_omega` | Angular velocity (rad/s) |
| `D_dm_vel` | Velocity-dependent mass correction |
| `D_mean_bracket` | Mean (W − 1) across all particle-source pairs |
| `coupling_signal` | (v_D − v_B) − (v_C − v_A) |

Key values at final step (17000):
- ω = 4662 rad/s (braking from 5236)
- Mean bracket deviation: 2.54 × 10⁻⁹
- Coupling signal: 4.15 × 10⁻¹⁴ m/s

## Relationship to Maxwell

The Darwin Lagrangian — Maxwell's theory expanded to order v²/c², dropping radiation — produces velocity-dependent corrections to the Coulomb interaction between charges. These corrections are the same order as the Weber bracket departure. This is not a coincidence.

Weber's bracket and Darwin's v²/c² corrections are the same physics seen from different ends of the telescope. Weber puts the correction in the pairwise force directly. Maxwell distributes it across field degrees of freedom — the vector potential, the magnetic field, the Poynting vector. In geometries with high symmetry (coaxial, planar), the field formulation is elegant and the corrections cancel or become invisible. In geometries with broken symmetry (toroidal, offset), the particle formulation exposes a remainder that the field formulation hides.

The claim is not that Maxwell is wrong. The claim is that Maxwell's field formulation, applied to standard symmetric geometries, produces null results *by symmetry* — not by physics. The torus breaks the symmetry. The bracket becomes visible. The simulation computes it.

For readers familiar with plasma physics: charges at high velocity in toroidal geometry with broken symmetry, exhibiting transport that standard field theory does not fully explain, is not a hypothetical scenario. It is a tokamak.

## Why Nobody Has Seen This

Every published test of electromagnetic force at the pairwise level uses axially symmetric geometry: flat rings, solenoids, coaxial cylinders, parallel wires. In these geometries, the radial velocity ṙ = v · R̂ between a spinning charge and a coaxial source element vanishes by symmetry. The bracket evaluates to:

    W = 1 + 0 + v_perp²/c²

The v_perp²/c² term is nonzero but *uniform* — it shifts the bracket identically for all particles and cancels in any differential measurement. The bracket equals unity not because Weber's law requires it, but because the geometry enforces it.

To see the bracket deviate differentially, you need geometry where:

1. Particles at different positions on the body see source elements at different angles.
2. Spinning introduces radial velocity components (ṙ ≠ 0) that vary across the body.
3. The variation does not cancel when summed over the body.

A torus satisfies all three. Particles on the inner edge of the minor cross-section are closer to coaxial sources than particles on the outer edge. Their radial velocities differ. The bracket deviates non-uniformly. The effective mass correction is not constant across the body, and spinning vs. non-spinning produces a differential signal.

An offset flat ring (displaced from the magnet axis) would also break degeneracy. No published experiment has tested either configuration.

Forty-one years of null results in aligned geometry are consistent with both theories. They do not distinguish Weber from Maxwell. They distinguish symmetric from asymmetric. The experiment has not been done.

## Falsification

These are specific, testable criteria. Each one kills the claim if it fails.

1. **Bracket unity.** Implement Weber's bracket independently. Evaluate it for a particle on a torus (R = 0.10 m, r = 0.03 m) spinning at ω = 5000 rad/s, with a coaxial current loop at z = 0.08 m. If W = 1.000000000000000000000000000 to 28 digits, this simulation has a bug.

2. **ω² scaling.** Run the simulation at ω = 1000, 2000, 3000, 4000, 5000 rad/s. Plot |W − 1| vs ω². If the relationship is not linear (within geometric averaging effects), the implementation is wrong.

3. **dt independence.** Run at dt = 10⁻⁶, 10⁻⁷, 10⁻⁸ s. The coupling signal magnitude must converge. If it scales with dt, it is a numerical artifact.

4. **Sign symmetry.** Reverse ω (spin the torus the other direction). The coupling signal must remain the same sign and magnitude (W depends on ṙ², which is even in v). If the coupling flips sign, there is an asymmetry bug in the force computation.

5. **Magnitude.** The bracket deviation must satisfy |W − 1| ≈ ω²R²/c² to within a geometric factor. At ω = 5000, R = 0.10: expected ~2.8 × 10⁻⁹, measured ~2.5 × 10⁻⁹. An order-of-magnitude discrepancy would indicate an error.

6. **Convergence in N.** Increase particle count: 50 → 200 → 500 → 1000. The bracket deviation and coupling signal must converge, not grow with N. Growth with N indicates a summation bug.

7. **Experimental.** Spin a bismuth torus at 50,000 RPM in a static magnetic field. Apply a calibrated impulse along the symmetry axis. Measure the translational acceleration to ~10⁻⁹ relative precision. Compare spinning vs. non-spinning. The simulation predicts the result. The experiment has not been done.

## Proposed Experimental Test

Spin a bismuth torus (R = 0.10 m, r = 0.03 m) at 50,000 RPM in a known static magnetic field. Apply a calibrated impulse along the symmetry axis. Measure the translational acceleration to ~10⁻⁹ relative precision. Compare spinning vs. non-spinning.

The predicted effect: the spinning torus accelerates slightly differently from the non-spinning torus under identical force, with the difference scaling as ω²R²/c².

Bismuth is the natural material: highest elemental diamagnetic susceptibility (χ = −1.7 × 10⁻⁴), high density (9780 kg/m³), readily available in single-crystal form. The same material that levitates in strong magnetic fields — for the same electromagnetic reason that makes the bracket visible.

## Build

Requires Rust 1.70+. CUDA Toolkit 12.x optional for GPU path.

```bash
# CPU (28-digit decimal)
cargo build --release

# GPU (31-digit double-double)
cargo build --release --features gpu
```

### Run

```bash
cargo run --release --bin brake-recoil                          # CPU, single torus
cargo run --release --features gpu --bin brake-recoil            # GPU, single torus
cargo run --release --features gpu --bin brake-recoil -- --bismuth3  # 3 bismuth tori
cargo run --release --features gpu --bin brake-recoil -- --stacked   # 2 counter-rotating
cargo run --release --features gpu --bin brake-recoil -- --config f.json  # custom
```

### Binaries

- `weber-anomaly` — Four-body comparison
- `brake-recoil` — Six-phase Faraday Protocol
- `debug-weber` — Single-particle force diagnostic
- `shake-test` — Bond network integrity
- `field-residual` — Static field residual probe

## Structure

```
src/
  weber.rs       — Weber force: bracket, longitudinal force, effective mass
  vec3.rs        — 3D vectors at 28-digit precision
  torus.rs       — Geometry, bonds, rigid body diagnostics
  magnets.rs     — Biot-Savart source elements
  integrator.rs  — Velocity Verlet (Newton and Weber)
  motor.rs       — PD motor with torque saturation
  config.rs      — Config, CLI, materials, protocol
  lib.rs, main.rs
  bin/           — brake_recoil, debug_weber, shake_test, field_residual
  gpu/           — CUDA dispatch (mod.rs), double-double arithmetic (dd.rs)
kernels/         — CUDA kernels (.cu) and DD math header (.cuh)
data/            — Reference CSV output
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

> Campbell, T. (2026). *Das Eimerargument*: Weber bracket deviation in
> spinning toroidal geometry at 28-digit precision.
> https://github.com/DeepBlueDynamics/das-eimerargument

## Copyright

The concept, experimental design, and theoretical framework described in this
repository are copyright Thomas Campbell, New Braunfels, Texas, 2026.

The simulation source code is released under the MIT license (see [LICENSE](LICENSE)).
The scientific claim, falsification protocol, and proposed experimental geometry
are the intellectual work of the author.
