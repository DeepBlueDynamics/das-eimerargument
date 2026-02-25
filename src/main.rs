use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use std::io::Write;

use weber_anomaly::vec3::Vec3;
use weber_anomaly::config::GeometryConfig;
use weber_anomaly::motor::Motor;
use weber_anomaly::magnets::generate_sources_from_config;
use weber_anomaly::torus::{self, generate_torus_from_config, generate_tori_from_config, merge_tori, seed_from_sdr};
use weber_anomaly::integrator::{step_newton, step_weber, BodySnapshot, total_weber_mass};

#[cfg(feature = "gpu")]
use weber_anomaly::gpu;

/// The four simulation bodies:
///   A: Newton, static     (baseline)
///   B: Newton, spinning   (spin control — should show no coupling)
///   C: Weber, static      (static mass correction only)
///   D: Weber, spinning    (the signal — spin-thrust coupling)
struct SimState {
    body_a: torus::TorusBody,
    body_b: torus::TorusBody,
    body_c: torus::TorusBody,
    body_d: torus::TorusBody,
    accel_c: Vec<Vec3>,
    accel_d: Vec<Vec3>,
}

/// Determine phase and thrust force from time. Motor handles torque separately.
fn phase_and_thrust(t: Decimal, cfg: &GeometryConfig) -> (u8, Vec3) {
    let p = &cfg.protocol;
    let t1 = Decimal::from_f64_retain(p.spin_up_duration).unwrap_or(Decimal::ZERO);
    let t2 = t1 + Decimal::from_f64_retain(p.coast1_duration).unwrap_or(Decimal::ZERO);
    let t3 = t2 + Decimal::from_f64_retain(p.thrust_duration).unwrap_or(Decimal::ZERO);
    let t4 = t3 + Decimal::from_f64_retain(p.coast2_duration).unwrap_or(Decimal::ZERO);
    let t5 = t4 + Decimal::from_f64_retain(p.brake_duration).unwrap_or(Decimal::ZERO);
    let thrust_val = Decimal::from_f64_retain(p.thrust_force).unwrap_or(dec!(0.1));

    if t < t1 {
        (1, Vec3::ZERO)
    } else if t < t2 {
        (2, Vec3::ZERO)
    } else if t < t3 {
        (3, Vec3::new(thrust_val, Decimal::ZERO, Decimal::ZERO))
    } else if t < t4 {
        (4, Vec3::ZERO)
    } else if t < t5 {
        (5, Vec3::ZERO)
    } else {
        (6, Vec3::ZERO)
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
    let cfg = GeometryConfig::from_cli_or_default(GeometryConfig::single_default());

    #[cfg(feature = "gpu")]
    {
        eprintln!("GPU feature enabled — launching CUDA accelerated simulation");
        match main_gpu(&cfg) {
            Ok(()) => return,
            Err(e) => {
                eprintln!("GPU initialization failed: {}. Falling back to CPU.", e);
            }
        }
    }

    main_cpu(&cfg);
}

#[cfg(feature = "gpu")]
fn main_gpu(cfg: &GeometryConfig) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Weber Inertial Anomaly Detector (GPU) ===");
    eprintln!("31-digit double-double CUDA + Velocity Verlet + Motor PD control");
    eprintln!();

    let sources = generate_sources_from_config(&cfg.magnets);
    eprintln!("  {} source elements from {} magnets", sources.len(), cfg.magnets.len());

    let seed = seed_from_sdr();
    let tori = generate_tori_from_config(&cfg.tori, seed);
    let base_torus = merge_tori(&tori);
    eprintln!("  {} particles, {} bonds, {} tori", base_torus.particles.len(), base_torus.bonds.len(), cfg.tori.len());

    let omega_target = Decimal::from_f64_retain(cfg.motor.omega_target).unwrap_or(dec!(5236));

    let mut gpu_a = gpu::GpuContext::new(&base_torus, &sources)?;
    let mut gpu_b = gpu::GpuContext::new(&base_torus, &sources)?;
    let mut gpu_c = gpu::GpuContext::new(&base_torus, &sources)?;
    let mut gpu_d = gpu::GpuContext::new(&base_torus, &sources)?;

    // GPU: pre-spin for now (motor spin-up on GPU needs per-step omega readback)
    gpu_b.set_spin(omega_target)?;
    gpu_d.set_spin(omega_target)?;
    eprintln!("  Pre-spin B,D to {} rad/s (GPU instant)", omega_target);

    let dt = Decimal::from_f64_retain(cfg.protocol.dt).unwrap_or(dec!(0.0000001));
    let total_steps = cfg.total_steps();
    eprintln!("Protocol: {} total steps, dt = {} s", total_steps, dt);
    eprintln!();

    gpu_a.init_accels(false, Vec3::ZERO, Decimal::ZERO)?;
    gpu_b.init_accels(false, Vec3::ZERO, Decimal::ZERO)?;
    gpu_c.init_accels(true, Vec3::ZERO, Decimal::ZERO)?;
    gpu_d.init_accels(true, Vec3::ZERO, Decimal::ZERO)?;

    let mut csv = std::fs::File::create("weber_simulation_gpu.csv").expect("cannot create CSV");
    writeln!(csv, "step,time,phase,\
        A_x,A_vx,A_omega,\
        B_x,B_vx,B_omega,\
        C_x,C_vx,C_omega,C_dm_vel,\
        D_x,D_vx,D_omega,D_dm_vel,D_bracket_coupling"
    ).unwrap();

    let log_interval = 1000u64;
    let mut last_phase = 0u8;
    let mut motor_b = Motor::from_config(&cfg.motor);
    let mut motor_d = Motor::from_config(&cfg.motor);
    const OMEGA_UPDATE_INTERVAL: u64 = 1000;
    let mut cached_omega_b = omega_target;
    let mut cached_omega_d = omega_target;

    eprintln!("Running GPU simulation...");
    let start_time = std::time::Instant::now();

    for step in 0..total_steps {
        let t = dt * Decimal::from(step);
        let (phase, thrust) = phase_and_thrust(t, cfg);

        if phase != last_phase {
            configure_motor_for_phase(&mut motor_b, phase, omega_target);
            configure_motor_for_phase(&mut motor_d, phase, omega_target);
            if phase > 1 {
                eprintln!("  Phase {} -> {} at t = {:.6} s (step {})", last_phase, phase,
                    t.to_f64().unwrap_or(0.0), step);
            }
            last_phase = phase;
        }

        // Update omega_z from GPU periodically (every 100 steps = 10μs)
        if step % OMEGA_UPDATE_INTERVAL == 0 {
            cached_omega_b = gpu_b.compute_omega_z()?;
            cached_omega_d = gpu_d.compute_omega_z()?;
        }
        let torque_b = motor_b.torque(cached_omega_b);
        let torque_d = motor_d.torque(cached_omega_d);

        gpu_a.step_newton(thrust, Decimal::ZERO, dt)?;
        gpu_b.step_newton(thrust, torque_b, dt)?;
        gpu_c.step_weber(thrust, Decimal::ZERO, dt)?;
        gpu_d.step_weber(thrust, torque_d, dt)?;

        if step % log_interval == 0 {
            let sa = gpu_a.download_snapshot(&base_torus)?;
            let sb = gpu_b.download_snapshot(&base_torus)?;
            let sc = gpu_c.download_snapshot(&base_torus)?;
            let sd = gpu_d.download_snapshot(&base_torus)?;
            let diag_d = gpu_d.download_weber_diagnostics()?;

            writeln!(csv, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                step, t, phase,
                sa.com.x, sa.com_vel.x, sa.omega_z,
                sb.com.x, sb.com_vel.x, sb.omega_z,
                sc.com.x, sc.com_vel.x, sc.omega_z, Decimal::ZERO,
                sd.com.x, sd.com_vel.x, sd.omega_z,
                diag_d.dm_velocity_total, diag_d.mean_bracket_coupling,
            ).unwrap();

            if step % (log_interval * 10) == 0 && step > 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let pct = step as f64 / total_steps as f64 * 100.0;
                let eta = elapsed / (step as f64) * ((total_steps - step) as f64);
                eprintln!("  step {}/{} ({:.1}%), elapsed {:.1}s, ETA {:.0}s",
                    step, total_steps, pct, elapsed, eta);
            }
        }
    }

    let elapsed = start_time.elapsed();
    eprintln!();
    eprintln!("GPU simulation complete in {:.1}s", elapsed.as_secs_f64());
    eprintln!("CSV output: weber_simulation_gpu.csv");
    Ok(())
}

fn main_cpu(cfg: &GeometryConfig) {
    eprintln!("=== Weber Inertial Anomaly Detector ===");
    eprintln!("28-digit decimal + Motor PD control");
    eprintln!("  {} tori, {} magnets", cfg.tori.len(), cfg.magnets.len());
    eprintln!();

    eprintln!("Generating magnet source elements...");
    let sources = generate_sources_from_config(&cfg.magnets);
    eprintln!("  {} source elements", sources.len());

    eprintln!("Generating torus bodies...");
    let seed = seed_from_sdr();
    let tori = generate_tori_from_config(&cfg.tori, seed);
    let base_torus = merge_tori(&tori);
    eprintln!("  {} particles, {} bonds, {} tori", base_torus.particles.len(), base_torus.bonds.len(), cfg.tori.len());

    let n = base_torus.particles.len();
    let omega_target = Decimal::from_f64_retain(cfg.motor.omega_target).unwrap_or(dec!(5236));

    // Four bodies start with zero velocity. Motor spins up B and D.
    let mut state = SimState {
        body_a: base_torus.clone(),
        body_b: base_torus.clone(),
        body_c: base_torus.clone(),
        body_d: base_torus.clone(),
        accel_c: vec![Vec3::ZERO; n],
        accel_d: vec![Vec3::ZERO; n],
    };

    let mut motor_b = Motor::from_config(&cfg.motor);
    let mut motor_d = Motor::from_config(&cfg.motor);

    let dt = Decimal::from_f64_retain(cfg.protocol.dt).unwrap_or(dec!(0.0000001));
    let total_steps = cfg.total_steps();

    eprintln!("Motor: k_p={}, max_torque={} N-m, friction={}", motor_b.k_p, motor_b.max_torque, motor_b.friction_mu);
    eprintln!("Protocol: {} total steps, dt = {} s", total_steps, dt);
    eprintln!("  Phase 1 (spin-up):  {} s", cfg.protocol.spin_up_duration);
    eprintln!("  Phase 2 (coast):    {} s", cfg.protocol.coast1_duration);
    eprintln!("  Phase 3 (thrust):   {} s", cfg.protocol.thrust_duration);
    eprintln!("  Phase 4 (coast):    {} s", cfg.protocol.coast2_duration);
    eprintln!("  Phase 5 (brake):    {} s", cfg.protocol.brake_duration);
    eprintln!("  Phase 6 (coast):    {} s", cfg.protocol.coast3_duration);
    eprintln!();

    let mut csv = std::fs::File::create("weber_simulation.csv").expect("cannot create CSV");
    writeln!(csv, "step,time,phase,\
        A_x,A_vx,A_omega,\
        B_x,B_vx,B_omega,\
        C_x,C_vx,C_omega,C_dm_static,C_dm_vel,\
        D_x,D_vx,D_omega,D_dm_static,D_dm_vel"
    ).unwrap();

    let log_interval = 1000u64;
    let mut last_phase = 0u8;

    let mut _snap_phase2_start: Option<[BodySnapshot; 4]> = None;
    let mut _snap_phase3_start: Option<[BodySnapshot; 4]> = None;
    let mut _snap_phase4_start: Option<[BodySnapshot; 4]> = None;
    let mut snap_phase5_start: Option<[BodySnapshot; 4]> = None;
    let mut snap_phase6_start: Option<[BodySnapshot; 4]> = None;

    eprintln!("Running simulation...");
    let start_time = std::time::Instant::now();

    for step in 0..total_steps {
        let t = dt * Decimal::from(step);
        let (phase, thrust) = phase_and_thrust(t, cfg);

        if phase != last_phase {
            configure_motor_for_phase(&mut motor_b, phase, omega_target);
            configure_motor_for_phase(&mut motor_d, phase, omega_target);

            let snaps = [
                BodySnapshot::capture(&state.body_a),
                BodySnapshot::capture(&state.body_b),
                BodySnapshot::capture(&state.body_c),
                BodySnapshot::capture(&state.body_d),
            ];
            match phase {
                2 => _snap_phase2_start = Some(snaps),
                3 => _snap_phase3_start = Some(snaps),
                4 => _snap_phase4_start = Some(snaps),
                5 => snap_phase5_start = Some(snaps),
                6 => snap_phase6_start = Some(snaps),
                _ => {}
            }
            if phase > 1 {
                eprintln!("  Phase {} -> {} at t = {:.6} s (step {}) omega_B={:.1} omega_D={:.1}",
                    last_phase, phase, t.to_f64().unwrap_or(0.0), step,
                    state.body_b.angular_velocity_z().to_f64().unwrap_or(0.0),
                    state.body_d.angular_velocity_z().to_f64().unwrap_or(0.0));
            }
            last_phase = phase;
        }

        // Compute motor torques
        let omega_b = state.body_b.angular_velocity_z();
        let omega_d = state.body_d.angular_velocity_z();
        let torque_b = motor_b.torque(omega_b);
        let torque_d = motor_d.torque(omega_d);

        step_newton(&mut state.body_a, &sources, thrust, Decimal::ZERO, dt);
        step_newton(&mut state.body_b, &sources, thrust, torque_b, dt);
        step_weber(&mut state.body_c, &sources, thrust, Decimal::ZERO, dt, &mut state.accel_c);
        step_weber(&mut state.body_d, &sources, thrust, torque_d, dt, &mut state.accel_d);

        if step % log_interval == 0 {
            let sa = BodySnapshot::capture(&state.body_a);
            let sb = BodySnapshot::capture(&state.body_b);
            let sc = BodySnapshot::capture(&state.body_c);
            let sd = BodySnapshot::capture(&state.body_d);

            let (_, dm_static_c, dm_vel_c) = total_weber_mass(&state.body_c, &sources, &state.accel_c);
            let (_, dm_static_d, dm_vel_d) = total_weber_mass(&state.body_d, &sources, &state.accel_d);

            writeln!(csv, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                step, t, phase,
                sa.com.x, sa.com_vel.x, sa.omega_z,
                sb.com.x, sb.com_vel.x, sb.omega_z,
                sc.com.x, sc.com_vel.x, sc.omega_z, dm_static_c, dm_vel_c,
                sd.com.x, sd.com_vel.x, sd.omega_z, dm_static_d, dm_vel_d,
            ).unwrap();

            if step % (log_interval * 10) == 0 && step > 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let pct = step as f64 / total_steps as f64 * 100.0;
                let eta = elapsed / (step as f64) * ((total_steps - step) as f64);
                eprintln!("  step {}/{} ({:.1}%), elapsed {:.1}s, ETA {:.0}s, omega_B={:.1}",
                    step, total_steps, pct, elapsed, eta,
                    state.body_b.angular_velocity_z().to_f64().unwrap_or(0.0));
            }
        }
    }

    let elapsed = start_time.elapsed();
    eprintln!();
    eprintln!("Simulation complete in {:.1}s", elapsed.as_secs_f64());
    eprintln!();

    // Final snapshots
    let snap_a = BodySnapshot::capture(&state.body_a);
    let snap_b = BodySnapshot::capture(&state.body_b);
    let snap_c = BodySnapshot::capture(&state.body_c);
    let snap_d = BodySnapshot::capture(&state.body_d);

    let (bare_d, dm_static_d, dm_vel_d) = total_weber_mass(&state.body_d, &sources, &state.accel_d);
    let (_bare_c, dm_static_c, dm_vel_c) = total_weber_mass(&state.body_c, &sources, &state.accel_c);

    eprintln!("=== RESULTS ===");
    eprintln!();

    let coupling = (snap_d.com.x - snap_b.com.x) - (snap_c.com.x - snap_a.com.x);
    eprintln!("Signal 1: Spin-Thrust Coupling");
    eprintln!("  dx_coupling = (x_D - x_B) - (x_C - x_A) = {}", coupling);
    eprintln!("  Newton predicts: 0");
    eprintln!("  Weber predicts: non-zero (spin modifies inertia)");
    eprintln!();

    if let (Some(ref pre), Some(ref post)) = (&snap_phase5_start, &snap_phase6_start) {
        let dv_d = post[3].com_vel.x - pre[3].com_vel.x;
        let dv_b = post[1].com_vel.x - pre[1].com_vel.x;
        eprintln!("Signal 2: Velocity Recoil across Brake");
        eprintln!("  dv_x (Weber spin, D): {}", dv_d);
        eprintln!("  dv_x (Newton spin, B): {}", dv_b);
        eprintln!("  Newton predicts: 0 (braking is rotational only)");
        eprintln!("  Weber predicts: positive (m_eff decreases -> v_x increases)");
        eprintln!();
    }

    eprintln!("Signal 3: Momentum Conservation");
    let m_eff_d = bare_d + dm_static_d + dm_vel_d;
    let p_weber = m_eff_d * snap_d.com_vel.x;
    let p_newton = snap_b.total_mass * snap_b.com_vel.x;
    eprintln!("  Weber body D: m_eff = {} (bare {} + dm_s {} + dm_v {})",
        m_eff_d, bare_d, dm_static_d, dm_vel_d);
    eprintln!("  p_x (Weber D) = m_eff * v_x = {}", p_weber);
    eprintln!("  p_x (Newton B) = m0 * v_x = {}", p_newton);
    eprintln!();

    eprintln!("=== FULL STATE ===");
    eprintln!();
    for (label, snap) in [("A (Newton static)", &snap_a), ("B (Newton spin)", &snap_b),
                           ("C (Weber static)", &snap_c), ("D (Weber spin)", &snap_d)] {
        eprintln!("Body {}:", label);
        eprintln!("  x_com    = {}", snap.com.x);
        eprintln!("  v_x      = {}", snap.com_vel.x);
        eprintln!("  omega_z  = {}", snap.omega_z);
        eprintln!("  KE       = {}", snap.ke);
        eprintln!("  p_x      = {}", snap.momentum.x);
        eprintln!("  L_z      = {}", snap.angular_momentum_z);
        eprintln!();
    }

    eprintln!("Weber mass corrections:");
    eprintln!("  Body C (static):  dm_static = {}, dm_vel = {}", dm_static_c, dm_vel_c);
    eprintln!("  Body D (spin):    dm_static = {}, dm_vel = {}", dm_static_d, dm_vel_d);
    eprintln!();
    eprintln!("THE BRACKET IS NOT UNITY: {}",
        if !dm_vel_d.is_zero() { "CONFIRMED" } else { "NOT DETECTED" });
    eprintln!();
    eprintln!("CSV output: weber_simulation.csv");
}
