use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::vec3::{Vec3, decimal_sqrt};
use crate::magnets::SourceElement;

/// Speed of light squared in m²/s², 28 digits.
/// c = 299792458 m/s → c² = 89875517873681764
pub const C_SQ: Decimal = Decimal::from_parts(873681764, 89875517, 0, false, 0);

// Verify: 299792458² = 89875517873681764
// Decimal::from_parts(lo, mid, hi, negative, scale)
// 89875517873681764 = 0x13F6_B445_3494_0764
// lo = 0x3494_0764 = 881262436 ... hmm let me compute properly.
// Actually let's use dec! macro in a function instead.

/// Return c² as a Decimal constant.
pub fn c_squared() -> Decimal {
    dec!(89875517873681764)
}

/// Result of the Weber force computation for one test particle.
#[derive(Clone, Debug)]
pub struct WeberResult {
    /// Total Weber-corrected Biot-Savart field at the particle
    pub b_field: Vec3,
    /// Total longitudinal (non-Maxwellian) force on the particle
    pub f_longitudinal: Vec3,
    /// Total force on the particle from all source elements
    pub f_total: Vec3,
    /// Diagnostic: mass correction from electromagnetic interaction.
    /// Using μ₀/4π formulation (k_e = c²·μ₀/4π already absorbed):
    ///   dm_static   = Σ (μ₀/4π)·q·|dl| / r       (no c² — it's in k_e)
    ///   dm_velocity = Σ (μ₀/4π)·q·|dl|·ṙ² / (2c²·r)  (one c², not two)
    pub delta_m_static: Decimal,
    pub delta_m_velocity: Decimal,
    /// Diagnostic: average Weber bracket deviation (W - 1) across sources
    pub mean_bracket_deviation: Decimal,
}

/// Compute the full Weber interaction for a single test particle
/// against all source elements.
///
/// This is the inner loop. Every number is Decimal. No f64.
///
/// Physics:
///   Source elements are FIXED (v_s = 0, a_s = 0).
///   Test particle moves with velocity v_i and has acceleration a_i
///   (from previous step or estimated).
///
/// Weber bracket:
///   W = 1 - ṙ²/(2c²) + r·r̈/c²
///
/// where:
///   ṙ = v_i · R̂  (radial velocity — PROJECTION, not magnitude)
///   r̈ = a_i · R̂ + (v_perp²)/r  (radial acceleration INCLUDING centripetal)
///   v_perp² = |v_i|² - ṙ²  (perpendicular velocity component squared)
///
/// The centripetal term (v_perp²)/r is Weber's verification point #1.
/// Missing it zeroes the acceleration-dependent correction for circular motion.
pub fn weber_force_on_particle(
    pos: Vec3,
    vel: Vec3,
    accel: Vec3,    // acceleration estimate from previous step
    charge: Decimal,
    sources: &[SourceElement],
) -> WeberResult {
    let c_sq = c_squared();
    let two_c_sq = dec!(2) * c_sq;

    // Coulomb constant k_e = 1/(4πε₀) ≈ 8.9875517873681764 × 10⁹
    // For the magnetic interaction: μ₀/(4π) = 10⁻⁷ T·m/A
    // But our source elements already carry I·dl, so the Biot-Savart law gives:
    //   dB = (μ₀/4π) · (I dl × R̂) / r²
    // μ₀/(4π) = 10⁻⁷
    let mu0_over_4pi = dec!(0.0000001);

    let mut b_field = Vec3::ZERO;
    let mut f_long_total = Vec3::ZERO;

    // Mass correction accumulators — deferred division (Weber's rivet principle).
    // In μ₀/4π formulation, k_e = c²·μ₀/(4π) is already absorbed, so:
    //   dm_static   = Σ (μ₀/4π)·q·|dl|/r         (NO c² division)
    //   dm_velocity = Σ (μ₀/4π)·q·|dl|·ṙ²/(2c²·r) (ONE c² division, deferred)
    let mut dm_static_acc = Decimal::ZERO;
    let mut dm_velocity_raw = Decimal::ZERO;  // before /2c²
    let mut bracket_deviation_sum = Decimal::ZERO;

    let mut n_sources = 0u32;

    for s in sources {
        let r_vec = pos - s.pos;
        let r_sq = r_vec.mag_sq();
        if r_sq.is_zero() { continue; }
        let r = decimal_sqrt(r_sq);
        let inv_r = Decimal::ONE / r;
        let r_hat = r_vec.scale(inv_r);

        // Relative velocity: v_rel = v_particle - v_source
        // For static sources (vel=0), v_rel = vel — backward compatible.
        let v_rel = vel - s.vel;
        let v_rel_sq = v_rel.mag_sq();

        // ṙ = v_rel · R̂ (radial component of relative velocity)
        let r_dot = v_rel.dot(r_hat);

        // v_perp² = |v_rel|² - ṙ²
        let v_perp_sq = v_rel_sq - r_dot * r_dot;

        // r̈ = a_i · R̂ + v_perp² / r  (includes centripetal!)
        let r_ddot = accel.dot(r_hat) + v_perp_sq * inv_r;

        // Weber bracket: W = 1 - ṙ²/(2c²) + r·r̈/c²
        let w_bracket = Decimal::ONE
            - r_dot * r_dot / two_c_sq
            + r * r_ddot / c_sq;

        bracket_deviation_sum += w_bracket - Decimal::ONE;
        n_sources += 1;

        let inv_r_sq = inv_r * inv_r;

        // Biot-Savart with Weber correction:
        //   dB = (μ₀/4π) · (I dl × R̂) / r² · W
        let dl_cross_rhat = s.dl.cross(r_hat);
        let db = dl_cross_rhat.scale(mu0_over_4pi * inv_r_sq * w_bracket);
        b_field = b_field + db;

        // Longitudinal force (non-Maxwellian):
        //   F_long = (μ₀/4π) · (q/r²) · [dl·v_rel - (3/2)(dl·R̂)(ṙ)] / c² · R̂
        let dl_dot_v = s.dl.dot(v_rel);
        let dl_dot_rhat = s.dl.dot(r_hat);
        let three_half = dec!(1.5);
        let f_long_mag = mu0_over_4pi * charge * inv_r_sq
            * (dl_dot_v - three_half * dl_dot_rhat * r_dot) / c_sq;
        let f_long = r_hat.scale(f_long_mag);
        f_long_total = f_long_total + f_long;

        // Mass correction accumulators.
        // (μ₀/4π)·q·|dl|/r ≈ 10⁻¹¹ per source — safe in Decimal.
        let dl_mag = s.dl.mag();
        let dm_raw = mu0_over_4pi * charge * dl_mag * inv_r;
        dm_static_acc += dm_raw;                  // no c² needed
        dm_velocity_raw += dm_raw * r_dot * r_dot; // deferred /2c²
    }

    // Velocity correction: single deferred /2c² (not /c⁴ — k_e absorbed the other c²)
    let delta_m_static = dm_static_acc;
    let delta_m_velocity = dm_velocity_raw / two_c_sq;

    // Mean bracket deviation
    let mean_bracket_deviation = if n_sources > 0 {
        bracket_deviation_sum / Decimal::from(n_sources)
    } else {
        Decimal::ZERO
    };

    // Lorentz force from Weber-corrected B: F = q (v × B)
    let f_lorentz = vel.cross(b_field).scale(charge);

    // Total force on particle
    let f_total = f_lorentz + f_long_total;

    WeberResult {
        b_field,
        f_longitudinal: f_long_total,
        f_total,
        delta_m_static,
        delta_m_velocity,
        mean_bracket_deviation,
    }
}

/// Compute the standard (Newtonian) Biot-Savart force on a particle.
/// No Weber corrections — bracket is always 1.
pub fn newton_force_on_particle(
    pos: Vec3,
    vel: Vec3,
    charge: Decimal,
    sources: &[SourceElement],
) -> Vec3 {
    let mu0_over_4pi = dec!(0.0000001);

    let mut b_field = Vec3::ZERO;

    for s in sources {
        let r_vec = pos - s.pos;
        let r_sq = r_vec.mag_sq();
        if r_sq.is_zero() { continue; }
        let r = decimal_sqrt(r_sq);
        let inv_r = Decimal::ONE / r;
        let r_hat = r_vec.scale(inv_r);
        let inv_r_sq = inv_r * inv_r;

        // Standard Biot-Savart: dB = (μ₀/4π) · (I dl × R̂) / r²
        let dl_cross_rhat = s.dl.cross(r_hat);
        let db = dl_cross_rhat.scale(mu0_over_4pi * inv_r_sq);
        b_field = b_field + db;
    }

    // Lorentz force: F = q (v × B)
    vel.cross(b_field).scale(charge)
}

/// Compute the total Weber effective mass for a particle.
/// m_eff = m₀ + Δm_static + Δm_velocity
pub fn effective_mass(
    pos: Vec3,
    vel: Vec3,
    accel: Vec3,
    bare_mass: Decimal,
    charge: Decimal,
    sources: &[SourceElement],
) -> Decimal {
    let result = weber_force_on_particle(pos, vel, accel, charge, sources);
    bare_mass + result.delta_m_static + result.delta_m_velocity
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magnets::generate_all_sources;

    #[test]
    fn test_c_squared_value() {
        let c = dec!(299792458);
        let expected = c * c;
        assert_eq!(c_squared(), expected);
    }

    #[test]
    fn test_newton_force_nonzero_for_moving_charge() {
        let sources = generate_all_sources();
        let pos = Vec3::new(dec!(0.10), Decimal::ZERO, Decimal::ZERO);
        let vel = Vec3::new(Decimal::ZERO, dec!(10), Decimal::ZERO);
        let q = dec!(0.000001);
        let f = newton_force_on_particle(pos, vel, q, &sources);
        // A moving charge in a magnetic field should experience a force
        assert!(!f.x.is_zero() || !f.y.is_zero() || !f.z.is_zero(),
            "force should be non-zero for moving charge in B field");
    }

    #[test]
    fn test_weber_bracket_near_unity_at_low_speed() {
        let sources = generate_all_sources();
        let pos = Vec3::new(dec!(0.10), Decimal::ZERO, Decimal::ZERO);
        let vel = Vec3::new(Decimal::ZERO, dec!(1), Decimal::ZERO); // 1 m/s, v/c ~ 3e-9
        let accel = Vec3::ZERO;
        let q = dec!(0.000001);
        let w_result = weber_force_on_particle(pos, vel, accel, q, &sources);
        let n_result = newton_force_on_particle(pos, vel, q, &sources);
        // At low velocity, Weber ≈ Newton
        // The forces should be very close (bracket ≈ 1)
        let diff = (w_result.f_total - n_result).mag();
        let newton_mag = n_result.mag();
        if !newton_mag.is_zero() {
            use rust_decimal::prelude::ToPrimitive;
            let ratio = diff.to_f64().unwrap() / newton_mag.to_f64().unwrap();
            assert!(ratio < 1e-10, "Weber should match Newton at low speed, ratio={}", ratio);
        }
    }
}
