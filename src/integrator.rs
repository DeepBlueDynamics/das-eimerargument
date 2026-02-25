use rust_decimal::Decimal;

use crate::vec3::Vec3;
use crate::torus::TorusBody;
use crate::magnets::SourceElement;
use crate::weber;

/// Symplectic Euler step for Newtonian mechanics.
///
/// Forces: bond forces + external thrust + standard Biot-Savart (no Weber).
/// Torque is applied as tangential force on each particle proportional to
/// its distance from the rotation axis.
pub fn step_newton(
    body: &mut TorusBody,
    sources: &[SourceElement],
    external_force: Vec3,    // thrust applied to CoM, distributed equally
    torque_z: Decimal,       // torque about z-axis
    dt: Decimal,
) {
    let n = body.particles.len();
    let inv_n = Decimal::ONE / Decimal::from(n as u64);

    // Bond forces
    let bond_f = body.bond_forces();

    // Compute per-particle forces
    let com = body.com();
    let mut accels = Vec::with_capacity(n);

    for i in 0..n {
        let p = &body.particles[i];

        // Biot-Savart force (standard, no Weber)
        let f_em = weber::newton_force_on_particle(p.pos, p.vel, p.charge, sources);

        // Distributed external force
        let f_ext = external_force.scale(inv_n);

        // Torque → tangential force: F_tang = τ / (N · d_perp) in tangential direction
        let rx = p.pos.x - com.x;
        let ry = p.pos.y - com.y;
        let d_perp_sq = rx * rx + ry * ry;
        let f_torque = if d_perp_sq.is_zero() {
            Vec3::ZERO
        } else {
            let d_perp = crate::vec3::decimal_sqrt(d_perp_sq);
            // Tangential direction: (-y, x, 0) / d_perp
            let tang = Vec3::new(-ry, rx, Decimal::ZERO).scale(Decimal::ONE / d_perp);
            // Force magnitude: τ / (N · d_perp)
            tang.scale(torque_z * inv_n / d_perp)
        };

        // Total force
        let f_total = bond_f[i] + f_em + f_ext + f_torque;

        // a = F / m
        let inv_m = Decimal::ONE / p.mass;
        accels.push(f_total.scale(inv_m));
    }

    // Symplectic Euler: update velocity, then position
    for i in 0..n {
        body.particles[i].vel = body.particles[i].vel + accels[i].scale(dt);
        body.particles[i].pos = body.particles[i].pos + body.particles[i].vel.scale(dt);
    }
}

/// Symplectic Euler step for Weber mechanics.
///
/// Forces: bond forces + external thrust + Weber-corrected Biot-Savart
///         + longitudinal force.
/// The acceleration-dependent term in the Weber bracket creates an
/// implicit equation (F depends on a, a depends on F). We solve this
/// by using the acceleration from the previous timestep as the estimate.
///
/// This is standard practice for velocity-dependent forces in symplectic
/// integrators and converges to O(dt²) accuracy.
pub fn step_weber(
    body: &mut TorusBody,
    sources: &[SourceElement],
    external_force: Vec3,
    torque_z: Decimal,
    dt: Decimal,
    prev_accels: &mut Vec<Vec3>,  // acceleration estimates from previous step
) {
    let n = body.particles.len();
    let inv_n = Decimal::ONE / Decimal::from(n as u64);

    // Ensure prev_accels has the right size
    if prev_accels.len() != n {
        *prev_accels = vec![Vec3::ZERO; n];
    }

    // Bond forces
    let bond_f = body.bond_forces();

    let com = body.com();
    let mut new_accels = Vec::with_capacity(n);

    for i in 0..n {
        let p = &body.particles[i];

        // Weber force (uses previous acceleration estimate for r̈)
        let w_result = weber::weber_force_on_particle(
            p.pos, p.vel, prev_accels[i], p.charge, sources,
        );

        // Distributed external force
        let f_ext = external_force.scale(inv_n);

        // Torque → tangential force
        let rx = p.pos.x - com.x;
        let ry = p.pos.y - com.y;
        let d_perp_sq = rx * rx + ry * ry;
        let f_torque = if d_perp_sq.is_zero() {
            Vec3::ZERO
        } else {
            let d_perp = crate::vec3::decimal_sqrt(d_perp_sq);
            let tang = Vec3::new(-ry, rx, Decimal::ZERO).scale(Decimal::ONE / d_perp);
            tang.scale(torque_z * inv_n / d_perp)
        };

        // Total force
        let f_total = bond_f[i] + w_result.f_total + f_ext + f_torque;

        // Use bare mass. Weber corrections are already in the force via W bracket.
        // Double-counting (W in force AND dm in mass) was wrong.
        let inv_m = Decimal::ONE / p.mass;
        new_accels.push(f_total.scale(inv_m));
    }

    // Symplectic Euler: update velocity, then position
    for i in 0..n {
        body.particles[i].vel = body.particles[i].vel + new_accels[i].scale(dt);
        body.particles[i].pos = body.particles[i].pos + body.particles[i].vel.scale(dt);
    }

    // Store accelerations for next step
    *prev_accels = new_accels;
}

/// Snapshot of a body's state for logging.
#[derive(Clone, Debug)]
pub struct BodySnapshot {
    pub com: Vec3,
    pub com_vel: Vec3,
    pub omega_z: Decimal,
    pub ke: Decimal,
    pub momentum: Vec3,
    pub angular_momentum_z: Decimal,
    pub total_mass: Decimal,
}

impl BodySnapshot {
    pub fn capture(body: &TorusBody) -> Self {
        BodySnapshot {
            com: body.com(),
            com_vel: body.com_velocity(),
            omega_z: body.angular_velocity_z(),
            ke: body.kinetic_energy(),
            momentum: body.momentum(),
            angular_momentum_z: body.angular_momentum_z(),
            total_mass: body.total_mass(),
        }
    }
}

/// Compute the Weber effective mass for the entire torus body.
/// Returns (total_bare_mass, total_delta_m_static, total_delta_m_velocity).
pub fn total_weber_mass(
    body: &TorusBody,
    sources: &[SourceElement],
    prev_accels: &[Vec3],
) -> (Decimal, Decimal, Decimal) {
    let mut dm_static = Decimal::ZERO;
    let mut dm_vel = Decimal::ZERO;
    let bare = body.total_mass();

    for (i, p) in body.particles.iter().enumerate() {
        let accel = if i < prev_accels.len() { prev_accels[i] } else { Vec3::ZERO };
        let w = weber::weber_force_on_particle(p.pos, p.vel, accel, p.charge, sources);
        dm_static += w.delta_m_static;
        dm_vel += w.delta_m_velocity;
    }

    (bare, dm_static, dm_vel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use crate::torus::generate_default_torus;
    use crate::magnets::generate_all_sources;

    #[test]
    fn test_newton_step_runs() {
        let mut body = generate_default_torus(42);
        let sources = generate_all_sources();
        let dt = dec!(0.0000001); // 0.1 μs
        let thrust = Vec3::ZERO;
        step_newton(&mut body, &sources, thrust, Decimal::ZERO, dt);
        // Should not panic
        let snap = BodySnapshot::capture(&body);
        assert!(!snap.total_mass.is_zero());
    }

    #[test]
    fn test_weber_step_runs() {
        let mut body = generate_default_torus(42);
        let sources = generate_all_sources();
        let dt = dec!(0.0000001);
        let thrust = Vec3::ZERO;
        let mut prev_accels = vec![Vec3::ZERO; 500];
        step_weber(&mut body, &sources, thrust, Decimal::ZERO, dt, &mut prev_accels);
        let snap = BodySnapshot::capture(&body);
        assert!(!snap.total_mass.is_zero());
    }
}
