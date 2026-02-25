use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::vec3::{Vec3, decimal_sin, decimal_cos};
use crate::config::{MagnetConfig, RotationSpec};

/// A single current element from the permanent magnets.
/// Stores position, dl vector (direction × current × segment length),
/// and velocity (zero for static magnets, nonzero for rotating magnets).
#[derive(Clone, Copy, Debug)]
pub struct SourceElement {
    pub pos: Vec3,
    pub dl: Vec3,
    /// Source velocity (zero for static magnets). For rotating magnets,
    /// v = ω × (pos - center). Used in Weber bracket as v_rel = v_particle - v_source.
    pub vel: Vec3,
}

/// Generate source elements for one magnet (stack of circular current loops).
///
/// Parameters from spec:
///   n_loops: 12 loops per magnet
///   n_seg: 32 segments per loop
///   r_mag: 0.05 m loop radius
///   z_centre: centre z position (+0.08 or -0.08)
///   dz_loop: 0.005 m spacing between loops
///   current: 1000 A effective current
pub fn generate_magnet(
    n_loops: usize,
    n_seg: usize,
    r_mag: Decimal,
    z_centre: Decimal,
    dz_loop: Decimal,
    current: Decimal,
) -> Vec<SourceElement> {
    let mut elements = Vec::with_capacity(n_loops * n_seg);
    let two_pi = dec!(6.2831853071795864769252867666);
    let seg_arc = two_pi / Decimal::from(n_seg as u64);
    let dl_mag = current * seg_arc * r_mag; // I * dθ * r = I * dl

    // Loops centred around z_centre, symmetric
    let half = Decimal::from(n_loops as u64 - 1) / dec!(2);

    for k in 0..n_loops {
        let z_k = z_centre + (Decimal::from(k as u64) - half) * dz_loop;

        for j in 0..n_seg {
            let theta = seg_arc * Decimal::from(j as u64);
            let cos_t = decimal_cos(theta);
            let sin_t = decimal_sin(theta);

            let pos = Vec3::new(r_mag * cos_t, r_mag * sin_t, z_k);

            // dl is tangent to the loop: I * (-sin θ, cos θ, 0) * arc_length
            let dl = Vec3::new(-sin_t * dl_mag, cos_t * dl_mag, Decimal::ZERO);

            elements.push(SourceElement { pos, dl, vel: Vec3::ZERO });
        }
    }
    elements
}

/// Generate the full set of source elements: top magnet + bottom magnet.
/// Returns 768 elements (2 × 12 × 32) per the spec.
pub fn generate_all_sources() -> Vec<SourceElement> {
    let n_loops = 12;
    let n_seg = 32;
    let r_mag = dec!(0.05);
    let dz_loop = dec!(0.005);
    let current = dec!(1000);
    let z_top = dec!(0.08);
    let z_bot = dec!(-0.08);

    let mut sources = generate_magnet(n_loops, n_seg, r_mag, z_top, dz_loop, current);
    sources.extend(generate_magnet(n_loops, n_seg, r_mag, z_bot, dz_loop, current));
    sources
}

/// Generate source elements from a list of MagnetConfig entries.
pub fn generate_sources_from_config(configs: &[MagnetConfig]) -> Vec<SourceElement> {
    let mut sources = Vec::new();
    for cfg in configs {
        sources.extend(generate_magnet(
            cfg.n_loops,
            cfg.n_seg,
            Decimal::from_f64_retain(cfg.r_mag).unwrap_or(dec!(0.05)),
            Decimal::from_f64_retain(cfg.z_centre).unwrap_or(Decimal::ZERO),
            Decimal::from_f64_retain(cfg.dz_loop).unwrap_or(dec!(0.005)),
            Decimal::from_f64_retain(cfg.current).unwrap_or(dec!(1000)),
        ));
    }
    sources
}

/// Convenience: stacked 3-magnet configuration (top/mid/bottom).
pub fn generate_stacked_sources() -> Vec<SourceElement> {
    let configs = vec![
        MagnetConfig { z_centre: 0.16, ..MagnetConfig::default() },
        MagnetConfig { z_centre: 0.00, ..MagnetConfig::default() },
        MagnetConfig { z_centre: -0.16, ..MagnetConfig::default() },
    ];
    generate_sources_from_config(&configs)
}

/// Apply rotation velocities to source elements.
/// For each element: vel = ω × (pos - center), where ω = omega * axis.
pub fn apply_rotation(sources: &mut [SourceElement], rot: &RotationSpec) {
    let omega_vec = Vec3::new(
        Decimal::from_f64_retain(rot.omega * rot.axis[0]).unwrap_or(Decimal::ZERO),
        Decimal::from_f64_retain(rot.omega * rot.axis[1]).unwrap_or(Decimal::ZERO),
        Decimal::from_f64_retain(rot.omega * rot.axis[2]).unwrap_or(Decimal::ZERO),
    );
    let center = Vec3::new(
        Decimal::from_f64_retain(rot.center[0]).unwrap_or(Decimal::ZERO),
        Decimal::from_f64_retain(rot.center[1]).unwrap_or(Decimal::ZERO),
        Decimal::from_f64_retain(rot.center[2]).unwrap_or(Decimal::ZERO),
    );
    for s in sources.iter_mut() {
        let r = s.pos - center;
        s.vel = omega_vec.cross(r); // v = ω × r
    }
}

/// Generate sources from config, applying rotation if specified.
pub fn generate_sources_from_config_with_rotation(configs: &[MagnetConfig]) -> Vec<SourceElement> {
    let mut all_sources = Vec::new();
    for cfg in configs {
        let start = all_sources.len();
        all_sources.extend(generate_magnet(
            cfg.n_loops,
            cfg.n_seg,
            Decimal::from_f64_retain(cfg.r_mag).unwrap_or(dec!(0.05)),
            Decimal::from_f64_retain(cfg.z_centre).unwrap_or(Decimal::ZERO),
            Decimal::from_f64_retain(cfg.dz_loop).unwrap_or(dec!(0.005)),
            Decimal::from_f64_retain(cfg.current).unwrap_or(dec!(1000)),
        ));
        // Apply rotation if this magnet has one
        if let Some(ref rot) = cfg.rotation {
            apply_rotation(&mut all_sources[start..], rot);
        }
    }
    all_sources
}
