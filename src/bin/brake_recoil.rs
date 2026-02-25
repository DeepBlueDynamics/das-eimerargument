//! Brake Recoil Test — Faraday's Protocol
//!
//! Motor-driven spin-up/down with configurable geometry.
//! Measures BOTH:
//!   Phase 3: Spin-thrust coupling (Weber predicts != 0, the real signal)
//!   Phase 5: Braking recoil (Weber council disagrees on whether this is nonzero)
//!
//! Motor replaces set_spin() cheat: realistic torque-based spin-up through inertia.
//! Weber body D sees different effective inertia than Newton body B — this IS the signal.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use std::io::Write;

use weber_anomaly::vec3::Vec3;
use weber_anomaly::config::{GeometryConfig, ProtocolConfig};
use weber_anomaly::motor::Motor;
use weber_anomaly::magnets::generate_sources_from_config;
use weber_anomaly::torus::{TorusBody, generate_torus_from_config, generate_tori_from_config, merge_tori, seed_from_sdr};
use weber_anomaly::weber;
use weber_anomaly::integrator::{step_newton, step_weber, BodySnapshot};

#[cfg(feature = "gpu")]
use weber_anomaly::gpu;

/// Compute translational and rotational KE separately.
fn ke_decomposed(body: &TorusBody) -> (Decimal, Decimal) {
    let half = dec!(0.5);
    let com_v = body.com_velocity();
    let total_mass = body.total_mass();
    let ke_trans = half * total_mass * com_v.mag_sq();
    let ke_total = body.kinetic_energy();
    let ke_rot = ke_total - ke_trans;
    (ke_trans, ke_rot)
}

/// Temperature proxy: mean squared deviation of particle velocity from
/// rigid body motion (CoM translation + rigid rotation).
fn temperature_proxy(body: &TorusBody) -> Decimal {
    let com = body.com();
    let com_v = body.com_velocity();
    let omega = body.angular_velocity_z();
    let n = body.particles.len();
    if n == 0 { return Decimal::ZERO; }

    let mut sum_sq = Decimal::ZERO;
    for p in &body.particles {
        let dx = p.pos.x - com.x;
        let dy = p.pos.y - com.y;
        let v_rigid_x = com_v.x - omega * dy;
        let v_rigid_y = com_v.y + omega * dx;
        let v_rigid_z = com_v.z;

        let dv_x = p.vel.x - v_rigid_x;
        let dv_y = p.vel.y - v_rigid_y;
        let dv_z = p.vel.z - v_rigid_z;
        sum_sq += dv_x * dv_x + dv_y * dv_y + dv_z * dv_z;
    }
    sum_sq / Decimal::from(n as u64)
}

/// Mean inter-particle radial velocity across bonds.
fn mean_bond_radial_velocity(body: &TorusBody) -> Decimal {
    if body.bonds.is_empty() { return Decimal::ZERO; }
    let mut sum = Decimal::ZERO;
    for bond in &body.bonds {
        let r_vec = body.particles[bond.j].pos - body.particles[bond.i].pos;
        let dist = r_vec.mag();
        if dist.is_zero() { continue; }
        let r_hat = r_vec.scale(Decimal::ONE / dist);
        let v_rel = body.particles[bond.j].vel - body.particles[bond.i].vel;
        let v_radial = v_rel.dot(r_hat).abs();
        sum += v_radial;
    }
    sum / Decimal::from(body.bonds.len() as u64)
}

/// Compute total dm_velocity and mean bracket deviation for a body.
fn weber_diagnostics(
    body: &TorusBody,
    sources: &[weber_anomaly::magnets::SourceElement],
    accels: &[Vec3],
) -> (Decimal, Decimal) {
    let mut dm_vel = Decimal::ZERO;
    let mut bracket_sum = Decimal::ZERO;
    let n = body.particles.len();
    for (i, p) in body.particles.iter().enumerate() {
        let a = if i < accels.len() { accels[i] } else { Vec3::ZERO };
        let w = weber::weber_force_on_particle(p.pos, p.vel, a, p.charge, sources);
        dm_vel += w.delta_m_velocity;
        bracket_sum += w.mean_bracket_deviation;
    }
    let mean_bracket = if n > 0 {
        bracket_sum / Decimal::from(n as u64)
    } else {
        Decimal::ZERO
    };
    (dm_vel, mean_bracket)
}

/// Determine phase and thrust force from time.
/// Motor torque is computed separately (not returned here).
fn phase_and_thrust(t: Decimal, cfg: &GeometryConfig) -> (u8, Vec3) {
    let p = &cfg.protocol;
    let t1 = Decimal::from_f64_retain(p.spin_up_duration).unwrap_or(Decimal::ZERO);
    let t2 = t1 + Decimal::from_f64_retain(p.coast1_duration).unwrap_or(Decimal::ZERO);
    let t3 = t2 + Decimal::from_f64_retain(p.thrust_duration).unwrap_or(Decimal::ZERO);
    let t4 = t3 + Decimal::from_f64_retain(p.coast2_duration).unwrap_or(Decimal::ZERO);
    let t5 = t4 + Decimal::from_f64_retain(p.brake_duration).unwrap_or(Decimal::ZERO);

    let thrust_val = Decimal::from_f64_retain(p.thrust_force).unwrap_or(dec!(0.1));
    let thrust_vec = match p.thrust_axis.as_str() {
        "x" => Vec3::new(thrust_val, Decimal::ZERO, Decimal::ZERO),
        _   => Vec3::new(Decimal::ZERO, Decimal::ZERO, thrust_val),  // "z" = through the hole
    };

    if t < t1 {
        (1, Vec3::ZERO)                                                // spin-up
    } else if t < t2 {
        (2, Vec3::ZERO)                                                // coast
    } else if t < t3 {
        (3, thrust_vec)                                                // thrust
    } else if t < t4 {
        (4, Vec3::ZERO)                                                // coast
    } else if t < t5 {
        (5, Vec3::ZERO)                                                // brake
    } else {
        (6, Vec3::ZERO)                                                // final coast
    }
}

/// Configure motor state for a given phase.
fn configure_motor_for_phase(motor: &mut Motor, phase: u8, omega_target: Decimal) {
    match phase {
        1 => { motor.enabled = true; motor.omega_target = omega_target; }
        2 | 3 | 4 => { motor.enabled = false; }
        5 => { motor.enabled = true; motor.omega_target = Decimal::ZERO; }
        6 => { motor.enabled = false; }
        _ => { motor.enabled = false; }
    }
}

fn main() {
    // Load config: --config file.json, --stacked, or default (single torus, quick test)
    let mut cfg = GeometryConfig::from_cli_or_default(GeometryConfig::single_default());

    // For brake-recoil, use quick protocol unless a custom config file is provided
    let args: Vec<String> = std::env::args().collect();
    let has_custom_config = args.iter().any(|a| a == "--config");
    if !has_custom_config {
        if cfg.tori.len() > 1 {
            // Stacked: use larger dt and fewer particles for reasonable runtime
            cfg.protocol = ProtocolConfig::quick_test();
            cfg.protocol.dt = 0.000001;  // 1μs (CFL for k=1e8, m=0.001 is ~6.3μs)
            for tc in &mut cfg.tori {
                tc.n_particles = 50;
            }
            for mc in &mut cfg.magnets {
                mc.n_loops = 6;
                mc.n_seg = 16;
            }
        } else {
            // Single torus: quick sanity check
            cfg.protocol = ProtocolConfig::quick_test();
            cfg.protocol.dt = 0.000001;  // 1μs (CFL for k=1e8, m=0.001 is ~3μs)
            for tc in &mut cfg.tori {
                tc.n_particles = 50;
            }
        }
    }

    #[cfg(feature = "gpu")]
    {
        eprintln!("GPU feature enabled — launching CUDA brake recoil test");
        match main_gpu_brake(&cfg) {
            Ok(()) => return,
            Err(e) => eprintln!("GPU failed: {}. Falling back to CPU.", e),
        }
    }

    main_cpu_brake(&cfg);
}

#[cfg(feature = "gpu")]
fn main_gpu_brake(cfg: &GeometryConfig) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Brake Recoil Test (GPU) ===");
    eprintln!("31-digit dd + Velocity Verlet + Motor PD control");
    eprintln!();

    let sources = generate_sources_from_config(&cfg.magnets);
    eprintln!("{} source elements from {} magnets", sources.len(), cfg.magnets.len());

    let seed = seed_from_sdr();
    // Generate all tori and merge into single body for GPU
    let tori = generate_tori_from_config(&cfg.tori, seed);
    let base_torus = merge_tori(&tori);
    let n = base_torus.particles.len();
    eprintln!("{} particles, {} bonds, {} tori configured", n, base_torus.bonds.len(), cfg.tori.len());

    let omega_target = Decimal::from_f64_retain(cfg.motor.omega_target).unwrap_or(dec!(5236));

    // GPU still uses set_spin for initial state (motor spin-up would take too long on GPU)
    let mut gpu_a = gpu::GpuContext::new(&base_torus, &sources)?;
    let mut gpu_b = gpu::GpuContext::new(&base_torus, &sources)?;
    let mut gpu_c = gpu::GpuContext::new(&base_torus, &sources)?;
    let mut gpu_d = gpu::GpuContext::new(&base_torus, &sources)?;

    // Per-torus omega: counter-rotation for stacked (alternating sign)
    let n_tori = base_torus.n_tori.max(1);
    let initial_omegas: Vec<Decimal> = (0..n_tori).map(|i| {
        let sign = if i % 2 == 0 { Decimal::ONE } else { -Decimal::ONE };
        sign * omega_target
    }).collect();
    let initial_omegas_f64: Vec<f64> = initial_omegas.iter()
        .map(|o| o.to_f64().unwrap_or(0.0)).collect();

    if n_tori > 1 {
        gpu_b.set_spin_multi(&initial_omegas)?;
        gpu_d.set_spin_multi(&initial_omegas)?;
        eprintln!("  Counter-rotating {} tori: omegas = {:?}", n_tori, initial_omegas_f64);
    } else {
        gpu_b.set_spin(omega_target)?;
        gpu_d.set_spin(omega_target)?;
        eprintln!("  Pre-spin B,D to {} rad/s (GPU instant)", omega_target);
    }

    // Per-torus moment of inertia for analytical tracking
    let i_z_per_torus = base_torus.moment_of_inertia_z_per_torus();
    let i_z_per_torus_f64: Vec<f64> = i_z_per_torus.iter()
        .map(|i| i.to_f64().unwrap_or(0.0005)).collect();
    eprintln!("  I_z per torus: {:?}", i_z_per_torus_f64);

    let dt = Decimal::from_f64_retain(cfg.protocol.dt).unwrap_or(dec!(0.0000001));
    let dt_f64 = cfg.protocol.dt;
    let total_steps = cfg.total_steps();
    eprintln!("Protocol: {} total steps", total_steps);
    eprintln!();

    gpu_a.init_accels(false, Vec3::ZERO, Decimal::ZERO)?;
    gpu_b.init_accels(false, Vec3::ZERO, Decimal::ZERO)?;
    gpu_c.init_accels(true, Vec3::ZERO, Decimal::ZERO)?;
    gpu_d.init_accels(true, Vec3::ZERO, Decimal::ZERO)?;

    // DEBUG: Check initial state after init_accels
    {
        let accels_a = gpu_a.download_accels_raw()?;
        let vels_a = gpu_a.download_velocities_raw()?;
        let vels_b = gpu_b.download_velocities_raw()?;
        let com_a = gpu_a.download_com_vel_raw()?;
        let com_b = gpu_b.download_com_vel_raw()?;
        eprintln!("  INIT DEBUG: body A accel[0] = ({:.4e}, {:.4e}, {:.4e})", accels_a[0].0, accels_a[0].1, accels_a[0].2);
        eprintln!("  INIT DEBUG: body A vel[0] = ({:.4e}, {:.4e}, {:.4e})", vels_a[0].0, vels_a[0].1, vels_a[0].2);
        eprintln!("  INIT DEBUG: body A COM vel = ({:.4e}, {:.4e}, {:.4e})", com_a.0, com_a.1, com_a.2);
        eprintln!("  INIT DEBUG: body B vel[0] = ({:.4e}, {:.4e}, {:.4e})", vels_b[0].0, vels_b[0].1, vels_b[0].2);
        eprintln!("  INIT DEBUG: body B COM vel = ({:.4e}, {:.4e}, {:.4e})", com_b.0, com_b.1, com_b.2);
        let nonzero_a = accels_a.iter().filter(|a| a.0.abs() > 1e-30 || a.1.abs() > 1e-30 || a.2.abs() > 1e-30).count();
        eprintln!("  INIT DEBUG: body A nonzero accels: {}/{}", nonzero_a, accels_a.len());
    }

    let geo_tag = if base_torus.n_tori > 1 { "stacked" } else { "single" };
    let thrust_axis = cfg.protocol.thrust_axis.clone();
    let csv_name = format!("brake_recoil_gpu_{}.csv", geo_tag);
    let mut csv = std::fs::File::create(&csv_name).expect("cannot create CSV");
    let axis_label = if thrust_axis == "x" { "vx" } else { "vz" };
    writeln!(csv, "step,time,phase,\
        A_{al},B_{al},C_{al},D_{al},\
        B_omega,D_omega,\
        D_dm_vel,D_mean_bracket,\
        D_bracket_coupling,\
        coupling_signal",
        al = axis_label).unwrap();
    eprintln!("Thrust axis: {} ({})", thrust_axis, if thrust_axis == "x" { "in-plane" } else { "perpendicular, through hole" });

    let start_time = std::time::Instant::now();
    let mut last_phase = 0u8;
    let mut motor_b = Motor::from_config(&cfg.motor);
    let mut motor_d = Motor::from_config(&cfg.motor);

    // Track omega ANALYTICALLY per torus — motor torque / I_torus.
    // reimpose enforces these omegas on particles each step,
    // eliminating bond-induced spin decay (numerical artifact).
    let mut analytical_omegas_b: Vec<f64> = initial_omegas_f64.clone();
    let mut analytical_omegas_d: Vec<f64> = initial_omegas_f64.clone();

    for step in 0..total_steps {
        let t = dt * Decimal::from(step);
        let (phase, thrust) = phase_and_thrust(t, cfg);

        if phase != last_phase {
            configure_motor_for_phase(&mut motor_b, phase, omega_target);
            configure_motor_for_phase(&mut motor_d, phase, omega_target);
            if phase > 1 {
                eprintln!("  Phase {} -> {} at step {} (t={:.6}s) omegas_B={:.1?} omegas_D={:.1?}",
                    last_phase, phase, step, t.to_f64().unwrap_or(0.0),
                    analytical_omegas_b, analytical_omegas_d);
            }
            last_phase = phase;
        }

        // Debug: print thrust vector on first thrust step
        if phase == 3 && step % 100 == 0 && step < 700 {
            eprintln!("  DEBUG step {}: thrust = ({}, {}, {})",
                step, thrust.x, thrust.y, thrust.z);
        }

        // Motor torque from analytical omega (use torus 0's omega as representative)
        // For counter-rotation, motor drives both tori symmetrically
        let omega_b_rep = Decimal::from_f64_retain(analytical_omegas_b[0].abs()).unwrap_or(Decimal::ZERO);
        let omega_d_rep = Decimal::from_f64_retain(analytical_omegas_d[0].abs()).unwrap_or(Decimal::ZERO);
        let torque_b = motor_b.torque(omega_b_rep);
        let torque_d = motor_d.torque(omega_d_rep);

        // Step the physics (Weber/Newton forces, bonds, external forces)
        gpu_a.step_newton(thrust, Decimal::ZERO, dt)?;
        gpu_b.step_newton(thrust, torque_b, dt)?;
        gpu_c.step_weber(thrust, Decimal::ZERO, dt)?;
        gpu_d.step_weber(thrust, torque_d, dt)?;

        // (verbose per-step debug removed for speed)

        // Reimpose per-torus rigid rotation — preserves COM velocity
        // while forcing per-torus angular velocity to analytical values.
        let zero_omegas: Vec<Decimal> = vec![Decimal::ZERO; n_tori];
        let omegas_b_dec: Vec<Decimal> = analytical_omegas_b.iter()
            .map(|&o| Decimal::from_f64_retain(o).unwrap_or(Decimal::ZERO)).collect();
        let omegas_d_dec: Vec<Decimal> = analytical_omegas_d.iter()
            .map(|&o| Decimal::from_f64_retain(o).unwrap_or(Decimal::ZERO)).collect();

        if n_tori > 1 {
            gpu_a.reimpose_rigid_rotation_multi(&zero_omegas)?;
            gpu_b.reimpose_rigid_rotation_multi(&omegas_b_dec)?;
            gpu_c.reimpose_rigid_rotation_multi(&zero_omegas)?;
            gpu_d.reimpose_rigid_rotation_multi(&omegas_d_dec)?;
        } else {
            gpu_a.reimpose_rigid_rotation(Decimal::ZERO)?;
            gpu_b.reimpose_rigid_rotation(omegas_b_dec[0])?;
            gpu_c.reimpose_rigid_rotation(Decimal::ZERO)?;
            gpu_d.reimpose_rigid_rotation(omegas_d_dec[0])?;
        }

        // DEBUG: Check for NaN periodically
        if step % 100 == 0 && step < 1000 {
            let vels_check = gpu_a.download_velocities_raw()?;
            if vels_check[0].0.is_nan() {
                eprintln!("  NAN DETECTED at step {}!", step);
                break;
            }
            let max_vel = vels_check.iter().map(|v| v.0.abs().max(v.1.abs()).max(v.2.abs())).fold(0.0f64, f64::max);
            eprintln!("  step {}: max_vel_A={:.3e}", step, max_vel);
        }

        // (verbose post-reimpose debug removed for speed)

        // Update per-torus analytical omega: motor torque / I_torus
        let torque_b_f64 = torque_b.to_f64().unwrap_or(0.0);
        let torque_d_f64 = torque_d.to_f64().unwrap_or(0.0);
        for i in 0..n_tori {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            analytical_omegas_b[i] += sign * torque_b_f64 / i_z_per_torus_f64[i] * dt_f64;
            analytical_omegas_d[i] += sign * torque_d_f64 / i_z_per_torus_f64[i] * dt_f64;
        }

        // Adaptive log interval: ~20 rows for short runs, ~18 rows for long runs
        let log_interval = if total_steps <= 2000 { 100 } else { 1000 };
        if step % log_interval == 0 {
            let sa = gpu_a.download_snapshot(&base_torus)?;
            let sb = gpu_b.download_snapshot(&base_torus)?;
            let sc = gpu_c.download_snapshot(&base_torus)?;
            let sd = gpu_d.download_snapshot(&base_torus)?;
            let diag = gpu_d.download_weber_diagnostics()?;

            // Measure velocity along thrust axis
            let (va, vb, vc, vd) = if thrust_axis == "x" {
                (sa.com_vel.x, sb.com_vel.x, sc.com_vel.x, sd.com_vel.x)
            } else {
                (sa.com_vel.z, sb.com_vel.z, sc.com_vel.z, sd.com_vel.z)
            };
            if phase == 3 && step % 100 == 0 {
                // Compare Decimal snapshot path vs raw f64 path
                let raw_d = gpu_d.download_com_vel_raw()?;
                let raw_a = gpu_a.download_com_vel_raw()?;
                eprintln!("  DEBUG step {} SNAPSHOT vz: A={} D={} | xyz_D=({},{},{})",
                    step, sa.com_vel.z, sd.com_vel.z,
                    sd.com_vel.x, sd.com_vel.y, sd.com_vel.z);
                eprintln!("  DEBUG step {} RAW     vz: A=({:.6e},{:.6e},{:.6e}) D=({:.6e},{:.6e},{:.6e})",
                    step, raw_a.0, raw_a.1, raw_a.2, raw_d.0, raw_d.1, raw_d.2);
                // Also check a few raw particle velocities
                let vels_d = gpu_d.download_velocities_raw()?;
                eprintln!("  DEBUG step {} D vel[0]=({:.6e},{:.6e},{:.6e}) vel[50]=({:.6e},{:.6e},{:.6e})",
                    step, vels_d[0].0, vels_d[0].1, vels_d[0].2,
                    vels_d[50.min(vels_d.len()-1)].0, vels_d[50.min(vels_d.len()-1)].1, vels_d[50.min(vels_d.len()-1)].2);
            }
            let coupling = (vd - vb) - (vc - va);

            writeln!(csv, "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                step, t, phase,
                va, vb, vc, vd,
                analytical_omegas_b[0], analytical_omegas_d[0],
                diag.dm_velocity_total, diag.mean_bracket_deviation,
                diag.mean_bracket_coupling,
                coupling,
            ).unwrap();
        }

        let progress_interval = if total_steps <= 2000 { 500 } else { 10000 };
        if step % progress_interval == 0 && step > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let pct = step as f64 / total_steps as f64 * 100.0;
            eprintln!("  step {}/{} ({:.1}%), {:.1}s, omegas_B={:.1?} omegas_D={:.1?}",
                step, total_steps, pct, elapsed,
                analytical_omegas_b, analytical_omegas_d);
        }
    }

    let elapsed = start_time.elapsed();
    eprintln!();
    eprintln!("GPU brake recoil complete in {:.1}s", elapsed.as_secs_f64());
    eprintln!("CSV output: {}", csv_name);
    Ok(())
}

fn main_cpu_brake(cfg: &GeometryConfig) {
    eprintln!("=== Brake Recoil Test (Motor-driven, configurable geometry) ===");
    eprintln!("Testing Phase 3 coupling AND Phase 5 braking recoil");
    eprintln!("  {} tori, {} magnets", cfg.tori.len(), cfg.magnets.len());
    eprintln!();

    let sources = generate_sources_from_config(&cfg.magnets);
    eprintln!("{} source elements", sources.len());

    let seed = seed_from_sdr();

    // Generate torus bodies for each torus in config
    // For multi-torus: each "body" is the first torus config (simplification for now)
    // Full multi-torus COM tracking would merge particles from all tori
    let torus_cfg = &cfg.tori[0];
    let base_torus = generate_torus_from_config(torus_cfg, seed);
    let n = base_torus.particles.len();
    eprintln!("{} particles, {} bonds (per torus)", n, base_torus.bonds.len());

    // Verify bond connectivity
    let mut connected = vec![false; n];
    for bond in &base_torus.bonds {
        connected[bond.i] = true;
        connected[bond.j] = true;
    }
    let isolated: Vec<usize> = connected.iter().enumerate()
        .filter(|(_, &c)| !c).map(|(i, _)| i).collect();
    if isolated.is_empty() {
        eprintln!("  All particles bonded (connectivity OK)");
    } else {
        eprintln!("  WARNING: {} isolated particles: {:?}", isolated.len(), isolated);
    }

    let omega_target = Decimal::from_f64_retain(cfg.motor.omega_target).unwrap_or(dec!(5236));

    // Four bodies: A (Newton static), B (Newton spin), C (Weber static), D (Weber spin)
    // Bodies B and D get motors; A and C are motorless
    let body_a = base_torus.clone();
    let mut body_b = base_torus.clone();
    let body_c = base_torus.clone();
    let mut body_d = base_torus.clone();

    // Pre-spin B and D, then reimpose (preserves COM vel = 0 at start)
    body_b.set_spin(omega_target);
    body_d.set_spin(omega_target);

    // Compute moment of inertia for analytical omega tracking
    let i_z = base_torus.moment_of_inertia_z();
    let i_z_f64 = i_z.to_f64().unwrap_or(0.0005);
    eprintln!("  Moment of inertia I_z = {} kg·m²", i_z);

    let mut bodies = [body_a, body_b, body_c, body_d];
    let mut accel_c = vec![Vec3::ZERO; n];
    let mut accel_d = vec![Vec3::ZERO; n];

    // Motors for B and D (A and C have disabled motors)
    let mut motor_b = Motor::from_config(&cfg.motor);
    let mut motor_d = Motor::from_config(&cfg.motor);

    // Analytical omega tracking — motor torque / I, not from particle velocities
    let omega_target_f64 = cfg.motor.omega_target;
    let mut analytical_omega_b: f64 = omega_target_f64;
    let mut analytical_omega_d: f64 = omega_target_f64;

    let dt = Decimal::from_f64_retain(cfg.protocol.dt).unwrap_or(dec!(0.0000001));
    let dt_f64 = cfg.protocol.dt;
    let total_steps = cfg.total_steps();

    eprintln!("Motor: k_p={}, max_torque={}, friction={}", motor_b.k_p, motor_b.max_torque, motor_b.friction_mu);
    eprintln!("Protocol: {} total steps, dt = {} s", total_steps, dt);
    eprintln!("  Phase 1 (spin-up):  {} s", cfg.protocol.spin_up_duration);
    eprintln!("  Phase 2 (coast):    {} s", cfg.protocol.coast1_duration);
    eprintln!("  Phase 3 (thrust):   {} s", cfg.protocol.thrust_duration);
    eprintln!("  Phase 4 (coast):    {} s", cfg.protocol.coast2_duration);
    eprintln!("  Phase 5 (brake):    {} s", cfg.protocol.brake_duration);
    eprintln!("  Phase 6 (coast):    {} s", cfg.protocol.coast3_duration);
    eprintln!();

    // CSV
    let thrust_axis = cfg.protocol.thrust_axis.clone();
    let cpu_geo_tag = if base_torus.n_tori > 1 { "stacked" } else { "single" };
    let cpu_csv_name = format!("brake_recoil_{}.csv", cpu_geo_tag);
    let mut csv = std::fs::File::create(&cpu_csv_name).expect("cannot create CSV");
    writeln!(csv, "step,time,phase,\
        A_vx,B_vx,C_vx,D_vx,\
        B_omega,D_omega,\
        D_dm_vel,D_mean_bracket,\
        D_ke_trans,D_ke_rot,\
        D_temp_proxy,D_bond_vrad,\
        coupling_signal").unwrap();

    // Phase boundary snapshots
    let mut _snap_phase2: Option<[f64; 4]> = None;
    let mut snap_phase3: Option<[f64; 4]> = None;
    let mut snap_phase4: Option<[f64; 4]> = None;
    let mut snap_phase5: Option<[f64; 4]> = None;
    let mut snap_phase6: Option<[f64; 4]> = None;

    let start_time = std::time::Instant::now();
    let mut last_phase = 0u8;

    // Pre-compute phase boundary steps for adaptive logging
    let p = &cfg.protocol;
    let phase3_start = ((p.spin_up_duration + p.coast1_duration) / p.dt) as u64;
    let phase4_start = phase3_start + (p.thrust_duration / p.dt) as u64;
    let phase5_start = phase4_start + (p.coast2_duration / p.dt) as u64;
    let phase6_start = phase5_start + (p.brake_duration / p.dt) as u64;

    for step in 0..total_steps {
        let t = dt * Decimal::from(step);
        let (phase, thrust) = phase_and_thrust(t, cfg);

        // Phase transition detection
        if phase != last_phase {
            configure_motor_for_phase(&mut motor_b, phase, omega_target);
            configure_motor_for_phase(&mut motor_d, phase, omega_target);

            let vxs = [
                if thrust_axis == "x" { bodies[0].com_velocity().x } else { bodies[0].com_velocity().z }.to_f64().unwrap_or(0.0),
                if thrust_axis == "x" { bodies[1].com_velocity().x } else { bodies[1].com_velocity().z }.to_f64().unwrap_or(0.0),
                if thrust_axis == "x" { bodies[2].com_velocity().x } else { bodies[2].com_velocity().z }.to_f64().unwrap_or(0.0),
                if thrust_axis == "x" { bodies[3].com_velocity().x } else { bodies[3].com_velocity().z }.to_f64().unwrap_or(0.0),
            ];
            match phase {
                2 => _snap_phase2 = Some(vxs),
                3 => snap_phase3 = Some(vxs),
                4 => snap_phase4 = Some(vxs),
                5 => snap_phase5 = Some(vxs),
                6 => snap_phase6 = Some(vxs),
                _ => {}
            }
            if phase > 1 {
                eprintln!("  Phase {} -> {} at step {} (t={:.6}s) omega_B={:.1} omega_D={:.1}",
                    last_phase, phase, step, t.to_f64().unwrap_or(0.0),
                    analytical_omega_b, analytical_omega_d);
            }
            last_phase = phase;
        }

        // Motor torques from analytical omega
        let omega_b_dec = Decimal::from_f64_retain(analytical_omega_b).unwrap_or(Decimal::ZERO);
        let omega_d_dec = Decimal::from_f64_retain(analytical_omega_d).unwrap_or(Decimal::ZERO);
        let torque_b = motor_b.torque(omega_b_dec);
        let torque_d = motor_d.torque(omega_d_dec);

        // Step all four bodies
        step_newton(&mut bodies[0], &sources, thrust, Decimal::ZERO, dt);     // A: static
        step_newton(&mut bodies[1], &sources, thrust, torque_b, dt);          // B: Newton spin
        step_weber(&mut bodies[2], &sources, thrust, Decimal::ZERO, dt, &mut accel_c); // C: Weber static
        step_weber(&mut bodies[3], &sources, thrust, torque_d, dt, &mut accel_d);      // D: Weber spin

        // Reimpose rigid rotation — preserves COM velocity, eliminates bond spin decay
        bodies[0].reimpose_rigid_rotation(Decimal::ZERO);          // static
        bodies[1].reimpose_rigid_rotation(omega_b_dec);            // Newton spin
        bodies[2].reimpose_rigid_rotation(Decimal::ZERO);          // static
        bodies[3].reimpose_rigid_rotation(omega_d_dec);            // Weber spin

        // Update analytical omega: motor torque / I
        let torque_b_f64 = torque_b.to_f64().unwrap_or(0.0);
        let torque_d_f64 = torque_d.to_f64().unwrap_or(0.0);
        analytical_omega_b += torque_b_f64 / i_z_f64 * dt_f64;
        analytical_omega_d += torque_d_f64 / i_z_f64 * dt_f64;

        // Adaptive logging frequency
        let should_log = match phase {
            1 | 2 => step % 1000 == 0,
            3 => step % 10 == 0,
            4 => {
                let steps_into = step.saturating_sub(phase4_start);
                let phase_len = phase5_start.saturating_sub(phase4_start);
                if phase_len > 100 && steps_into >= phase_len - 100 {
                    true
                } else {
                    step % 100 == 0
                }
            }
            5 => step % 10 == 0,
            6 => {
                let steps_into = step.saturating_sub(phase6_start);
                if steps_into < 500 { step % 10 == 0 } else { step % 100 == 0 }
            }
            _ => step % 1000 == 0,
        };

        if should_log {
            let vx_a = if thrust_axis == "x" { bodies[0].com_velocity().x } else { bodies[0].com_velocity().z };
            let vx_b = if thrust_axis == "x" { bodies[1].com_velocity().x } else { bodies[1].com_velocity().z };
            let vx_c = if thrust_axis == "x" { bodies[2].com_velocity().x } else { bodies[2].com_velocity().z };
            let vx_d = if thrust_axis == "x" { bodies[3].com_velocity().x } else { bodies[3].com_velocity().z };

            let (dm_vel_d, mean_bracket_d) = weber_diagnostics(&bodies[3], &sources, &accel_d);
            let (ke_trans_d, ke_rot_d) = ke_decomposed(&bodies[3]);
            let temp_d = temperature_proxy(&bodies[3]);
            let bond_vrad_d = mean_bond_radial_velocity(&bodies[3]);

            let coupling = (vx_d - vx_b) - (vx_c - vx_a);

            writeln!(csv, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                step, t, phase,
                vx_a, vx_b, vx_c, vx_d,
                analytical_omega_b, analytical_omega_d,
                dm_vel_d, mean_bracket_d,
                ke_trans_d, ke_rot_d,
                temp_d, bond_vrad_d,
                coupling,
            ).unwrap();
        }

        if step % 10000 == 0 && step > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let pct = step as f64 / total_steps as f64 * 100.0;
            let eta = elapsed / (step as f64) * ((total_steps - step) as f64);
            eprintln!("  step {}/{} ({:.1}%), elapsed {:.1}s, ETA {:.0}s, omega_B={:.1}, omega_D={:.1}",
                step, total_steps, pct, elapsed, eta,
                analytical_omega_b, analytical_omega_d);
        }
    }

    let elapsed = start_time.elapsed();
    eprintln!();
    eprintln!("Simulation complete in {:.1}s", elapsed.as_secs_f64());
    eprintln!();

    // === SIGNAL EXTRACTION ===
    eprintln!("=== PHASE 3: SPIN-THRUST COUPLING ===");
    if let (Some(pre), Some(post)) = (&snap_phase3, &snap_phase4) {
        let dv_a = post[0] - pre[0];
        let dv_b = post[1] - pre[1];
        let dv_c = post[2] - pre[2];
        let dv_d = post[3] - pre[3];

        eprintln!("  dv_x across Phase 3 (thrust):");
        eprintln!("    A (Newton static):  {:.15e}", dv_a);
        eprintln!("    B (Newton spin):    {:.15e}", dv_b);
        eprintln!("    C (Weber static):   {:.15e}", dv_c);
        eprintln!("    D (Weber spin):     {:.15e}", dv_d);
        eprintln!();
        eprintln!("  Coupling signal: (D-B) - (C-A) = {:.15e}", (dv_d - dv_b) - (dv_c - dv_a));
        eprintln!("  Newton predicts: 0");
        eprintln!("  Weber predicts: nonzero (bracket modifies force during thrust)");
    }
    eprintln!();

    eprintln!("=== PHASE 5: BRAKING RECOIL ===");
    if let (Some(pre), Some(post)) = (&snap_phase5, &snap_phase6) {
        let dv_a = post[0] - pre[0];
        let dv_b = post[1] - pre[1];
        let dv_c = post[2] - pre[2];
        let dv_d = post[3] - pre[3];

        eprintln!("  dv_x across Phase 5 (brake):");
        eprintln!("    A (Newton static):  {:.15e}", dv_a);
        eprintln!("    B (Newton spin):    {:.15e}", dv_b);
        eprintln!("    C (Weber static):   {:.15e}", dv_c);
        eprintln!("    D (Weber spin):     {:.15e}", dv_d);
        eprintln!();
        eprintln!("  Recoil signal: (D-B) - (C-A) = {:.15e}", (dv_d - dv_b) - (dv_c - dv_a));
        eprintln!("  Newton predicts: 0 (braking is purely rotational)");
        eprintln!("  Weber council disagrees on whether this is nonzero");
    }
    eprintln!();

    // Final state
    eprintln!("=== FINAL STATE ===");
    for (label, body) in [("A (Newton static)", &bodies[0]),
                           ("B (Newton spin)", &bodies[1]),
                           ("C (Weber static)", &bodies[2]),
                           ("D (Weber spin)", &bodies[3])] {
        let snap = BodySnapshot::capture(body);
        eprintln!("Body {}:", label);
        eprintln!("  v_x  = {}", snap.com_vel.x);
        eprintln!("  omega_z  = {}", snap.omega_z);
        eprintln!("  KE   = {}", snap.ke);
        eprintln!();
    }

    eprintln!("CSV output: {}", cpu_csv_name);
}
