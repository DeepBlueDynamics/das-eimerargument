//! Diagnostic: trace Weber force kernel values for a single particle.
//! Verifies that the deferred c^2 division and k_e->mu0/4pi conversion are correct.
//!
//! Uses set_spin() for initial state since we're debugging the force kernel,
//! not the motor. Motor is orthogonal here.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

use weber_anomaly::vec3::Vec3;
use weber_anomaly::config::GeometryConfig;
use weber_anomaly::magnets::generate_sources_from_config;
use weber_anomaly::torus::{generate_torus_from_config, seed_from_sdr};
use weber_anomaly::weber;

fn main() {
    eprintln!("=== Weber Force Kernel Diagnostic ===");
    eprintln!();

    let cfg = GeometryConfig::from_cli_or_default(GeometryConfig::single_default());
    let sources = generate_sources_from_config(&cfg.magnets);

    let mut torus_cfg = cfg.tori[0].clone();
    torus_cfg.n_particles = 5;
    let mut body = generate_torus_from_config(&torus_cfg, seed_from_sdr());

    // Use set_spin for diagnostic snapshot
    let omega = dec!(5236);
    body.set_spin(omega);

    let p = &body.particles[0];
    let v_mag = p.vel.mag();
    eprintln!("Particle 0:");
    eprintln!("  |vel| = {} m/s", v_mag);
    eprintln!("  v/c   = {:.6e}", v_mag.to_f64().unwrap() / 299792458.0);
    eprintln!();

    // Single source trace
    let s = &sources[0];
    let r_vec = p.pos - s.pos;
    let r = r_vec.mag();
    let inv_r = Decimal::ONE / r;
    let r_hat = r_vec.scale(inv_r);
    let r_dot = p.vel.dot(r_hat);
    let c_sq = weber::c_squared();
    let two_c_sq = dec!(2) * c_sq;
    let mu0_4pi = dec!(0.0000001);
    let dl_mag = s.dl.mag();

    eprintln!("Single source-particle pair:");
    eprintln!("  |r|      = {}", r);
    eprintln!("  r_dot    = {} m/s", r_dot);
    eprintln!("  r_dot^2/(2c^2) = {:.6e}", (r_dot * r_dot / two_c_sq).to_f64().unwrap());
    eprintln!();

    let dm_raw = mu0_4pi * p.charge * dl_mag * inv_r;
    eprintln!("dm_raw = (mu0/4pi)*q*|dl|/r = {:.6e}", dm_raw.to_f64().unwrap());
    eprintln!("  -> THIS is dm_static per source (no c^2 division needed)");
    eprintln!("  -> x{} sources ~ {:.6e} kg", sources.len(), dm_raw.to_f64().unwrap() * sources.len() as f64);
    eprintln!();

    let dm_vel_raw = dm_raw * r_dot * r_dot;
    eprintln!("dm_vel_raw = dm_raw * r_dot^2 = {:.6e}", dm_vel_raw.to_f64().unwrap());
    eprintln!("  -> /2c^2 = {:.6e} kg  (one c^2 division, deferred)",
        (dm_vel_raw / two_c_sq).to_f64().unwrap());
    eprintln!("  -> x{}/2c^2 ~ {:.6e} kg",
        sources.len(), dm_vel_raw.to_f64().unwrap() * sources.len() as f64 / two_c_sq.to_f64().unwrap());
    eprintln!();

    // Full kernel on particle 0
    let accel = Vec3::ZERO;
    let result = weber::weber_force_on_particle(p.pos, p.vel, accel, p.charge, &sources);

    eprintln!("=== Full kernel (particle 0, all {} sources) ===", sources.len());
    eprintln!("  dm_static          = {:.10e}", result.delta_m_static.to_f64().unwrap());
    eprintln!("  dm_velocity        = {:.10e}", result.delta_m_velocity.to_f64().unwrap());
    eprintln!("  mean (W-1)         = {:.10e}", result.mean_bracket_deviation.to_f64().unwrap());
    eprintln!("  |F_total|          = {:.10e}", result.f_total.mag().to_f64().unwrap());
    eprintln!("  |B_field|          = {:.10e}", result.b_field.mag().to_f64().unwrap());
    eprintln!();

    // All particles
    eprintln!("=== All {} particles ===", body.particles.len());
    let mut total_dm_s = Decimal::ZERO;
    let mut total_dm_v = Decimal::ZERO;
    let mut total_bracket = Decimal::ZERO;
    let bare = body.total_mass();

    for (i, p) in body.particles.iter().enumerate() {
        let w = weber::weber_force_on_particle(p.pos, p.vel, Vec3::ZERO, p.charge, &sources);
        total_dm_s += w.delta_m_static;
        total_dm_v += w.delta_m_velocity;
        total_bracket += w.mean_bracket_deviation;
        eprintln!("  particle {}: dm_s={:.4e}  dm_v={:.4e}  <W-1>={:.4e}",
            i, w.delta_m_static.to_f64().unwrap(),
            w.delta_m_velocity.to_f64().unwrap(),
            w.mean_bracket_deviation.to_f64().unwrap());
    }

    let n = body.particles.len() as f64;
    eprintln!();
    eprintln!("Totals:");
    eprintln!("  bare mass    = {} kg", bare);
    eprintln!("  sum dm_static  = {:.10e} kg", total_dm_s.to_f64().unwrap());
    eprintln!("  sum dm_vel     = {:.10e} kg", total_dm_v.to_f64().unwrap());
    eprintln!("  dm_s/m0      = {:.10e}", total_dm_s.to_f64().unwrap() / bare.to_f64().unwrap());
    eprintln!("  dm_v/m0      = {:.10e}", total_dm_v.to_f64().unwrap() / bare.to_f64().unwrap());
    eprintln!("  mean <W-1>   = {:.10e}", total_bracket.to_f64().unwrap() / n);
    eprintln!();
    eprintln!("THE BRACKET IS NOT UNITY: dm_velocity = {:.6e} kg",
        total_dm_v.to_f64().unwrap());
}
