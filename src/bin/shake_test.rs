//! Lattice disorder test — Faraday's experiment #4.
//!
//! Shakes the torus with increasing amplitude and measures:
//! - Bond strain energy (how stressed the lattice is)
//! - Weber mass correction (does disorder change effective mass?)
//! - Angular velocity stability (does shaking disrupt spin coherence?)
//! - Recovery after shaking stops (does the lattice self-heal?)
//!
//! Uses set_spin() for initial state (not motor) since we're testing
//! lattice response, not spin-up dynamics. Motor is orthogonal here.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::io::Write;

use weber_anomaly::vec3::Vec3;
use weber_anomaly::config::GeometryConfig;
use weber_anomaly::magnets::generate_sources_from_config;
use weber_anomaly::torus::{Rng64, generate_torus_from_config, seed_from_sdr};
use weber_anomaly::weber;
use weber_anomaly::integrator::{step_weber, BodySnapshot};

/// Compute total bond strain energy: sum 0.5 k (|r_ij| - r0)^2
fn bond_strain_energy(body: &weber_anomaly::torus::TorusBody) -> Decimal {
    let half = dec!(0.5);
    let mut energy = Decimal::ZERO;
    for bond in &body.bonds {
        let sep = body.particles[bond.j].pos - body.particles[bond.i].pos;
        let dist = sep.mag();
        let stretch = dist - bond.rest_length;
        energy += half * body.stiffness * stretch * stretch;
    }
    energy
}

/// Compute total Weber dm_velocity for the body.
fn weber_dm_velocity(body: &weber_anomaly::torus::TorusBody, sources: &[weber_anomaly::magnets::SourceElement], accels: &[Vec3]) -> Decimal {
    let mut dm_vel = Decimal::ZERO;
    for (i, p) in body.particles.iter().enumerate() {
        let a = if i < accels.len() { accels[i] } else { Vec3::ZERO };
        let w = weber::weber_force_on_particle(p.pos, p.vel, a, p.charge, sources);
        dm_vel += w.delta_m_velocity;
    }
    dm_vel
}

fn main() {
    eprintln!("=== Lattice Disorder Test (Shake) ===");
    eprintln!();

    let cfg = GeometryConfig::from_cli_or_default(GeometryConfig::single_default());
    let sources = generate_sources_from_config(&cfg.magnets);
    eprintln!("{} source elements", sources.len());

    // Reduced particle count for tractable test
    let mut torus_cfg = cfg.tori[0].clone();
    torus_cfg.n_particles = 20;
    let mut body = generate_torus_from_config(&torus_cfg, seed_from_sdr());
    let n = body.particles.len();
    eprintln!("{} particles, {} bonds", n, body.bonds.len());

    // Use set_spin for initial state (shake test measures lattice, not motor)
    let omega = dec!(5236);
    body.set_spin(omega);
    eprintln!("Initial spin: omega = {} rad/s (50K RPM)", omega);

    let mut rng = Rng64::new(12345);
    let mut accels = vec![Vec3::ZERO; n];
    let dt = dec!(0.0000001); // 0.1 us

    let mut csv = std::fs::File::create("shake_test.csv").expect("cannot create CSV");
    writeln!(csv, "step,phase,amplitude,rms_disp,omega_z,bond_strain,dm_velocity,com_x,com_vx,ke")
        .unwrap();

    let log_interval = 10;
    let steps_per_phase = 100;

    let amplitudes = [
        dec!(0.0),
        dec!(0.0001),
        dec!(0.0005),
        dec!(0.001),
        dec!(0.003),
        dec!(0.0),
    ];

    let phase_labels = [
        "baseline",
        "gentle (0.1mm)",
        "moderate (0.5mm)",
        "significant (1mm)",
        "severe (3mm)",
        "recovery",
    ];

    let start_time = std::time::Instant::now();
    let mut global_step = 0u64;

    for (phase_idx, &amplitude) in amplitudes.iter().enumerate() {
        eprintln!();
        eprintln!("--- Phase {}: {} (amplitude = {} m) ---",
            phase_idx, phase_labels[phase_idx], amplitude);

        let snap_before = BodySnapshot::capture(&body);
        eprintln!("  omega_z before: {}", snap_before.omega_z);

        for step in 0..steps_per_phase {
            let rms = if !amplitude.is_zero() {
                body.shake(amplitude, &mut rng)
            } else {
                Decimal::ZERO
            };

            step_weber(&mut body, &sources, Vec3::ZERO, Decimal::ZERO, dt, &mut accels);

            if step % log_interval == 0 {
                let snap = BodySnapshot::capture(&body);
                let strain = bond_strain_energy(&body);
                let dm_vel = weber_dm_velocity(&body, &sources, &accels);

                writeln!(csv, "{},{},{},{},{},{},{},{},{},{}",
                    global_step, phase_idx, amplitude,
                    rms, snap.omega_z, strain, dm_vel,
                    snap.com.x, snap.com_vel.x, snap.ke
                ).unwrap();
            }

            global_step += 1;
        }

        let snap_after = BodySnapshot::capture(&body);
        let strain = bond_strain_energy(&body);
        let dm_vel = weber_dm_velocity(&body, &sources, &accels);

        eprintln!("  omega_z after:  {}", snap_after.omega_z);
        eprintln!("  d_omega_z:     {}", snap_after.omega_z - snap_before.omega_z);
        eprintln!("  bond strain energy: {}", strain);
        eprintln!("  dm_velocity: {}", dm_vel);
        eprintln!("  KE: {}", snap_after.ke);

        let elapsed = start_time.elapsed().as_secs_f64();
        eprintln!("  elapsed: {:.1}s", elapsed);
    }

    let total_elapsed = start_time.elapsed();
    eprintln!();
    eprintln!("=== Shake test complete in {:.1}s ===", total_elapsed.as_secs_f64());
    eprintln!("CSV output: shake_test.csv");
    eprintln!();

    eprintln!("Key question: does lattice disorder affect Weber mass correction?");
    eprintln!("If dm_velocity changes with amplitude, the crystal's order matters.");
    eprintln!("If it recovers in the final phase, the effect is reversible.");
}
