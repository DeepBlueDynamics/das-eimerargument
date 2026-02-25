use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use std::collections::HashMap;

use crate::vec3::{Vec3, decimal_sqrt};
use crate::config::{TorusConfig, GeometryConfig, BodyConfig, MaterialSpec};

/// Type alias: all shapes produce a ParticleBody (same runtime representation).
pub type ParticleBody = TorusBody;

/// Fetch true random seed from the SDR entropy pool (gnosis-radio on :9080).
/// Falls back to system time if the radio isn't running.
pub fn seed_from_sdr() -> u64 {
    match fetch_sdr_entropy() {
        Some(bytes) => {
            let mut seed = 0u64;
            for (i, &b) in bytes.iter().take(8).enumerate() {
                seed |= (b as u64) << (i * 8);
            }
            if seed == 0 { seed = 1; }
            eprintln!("  Seed from SDR entropy: {:#018x}", seed);
            seed
        }
        None => {
            let fallback = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEAD_BEEF_CAFE_1234);
            eprintln!("  SDR not available, seed from system time: {:#018x}", fallback);
            fallback
        }
    }
}

fn fetch_sdr_entropy() -> Option<Vec<u8>> {
    use std::io::Read;
    use std::net::TcpStream;
    use std::time::Duration;

    let mut stream = TcpStream::connect_timeout(
        &"127.0.0.1:9080".parse().ok()?,
        Duration::from_millis(500),
    ).ok()?;
    stream.set_read_timeout(Some(Duration::from_millis(1000))).ok()?;
    stream.set_write_timeout(Some(Duration::from_millis(500))).ok()?;

    let request = "GET /api/entropy?bytes=64&format=json HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    std::io::Write::write_all(&mut stream, request.as_bytes()).ok()?;

    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;

    // Find the JSON body after headers
    let body = response.split("\r\n\r\n").nth(1)?;
    // Extract entropy_hex value
    let hex_start = body.find("\"entropy_hex\":\"")?;
    let hex_value = &body[hex_start + 15..];
    let hex_end = hex_value.find('"')?;
    let hex = &hex_value[..hex_end];

    // Decode hex to bytes
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&hex[i..i+2], 16).ok())
        .collect();

    if bytes.len() >= 8 { Some(bytes) } else { None }
}

/// A particle in the torus body.
#[derive(Clone, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub mass: Decimal,
    pub charge: Decimal,
}

/// A bond between two particles maintaining structural integrity.
#[derive(Clone, Debug)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    pub rest_length: Decimal,
}

/// The torus test body: particles + bonds + collective state.
#[derive(Clone)]
pub struct TorusBody {
    pub particles: Vec<Particle>,
    pub bonds: Vec<Bond>,
    pub stiffness: Decimal,
    pub damping: Decimal,
    /// Which torus each particle belongs to (for counter-rotation).
    /// Length matches particles. All zeros for single-torus bodies.
    pub torus_ids: Vec<usize>,
    /// Number of distinct tori in this body.
    pub n_tori: usize,
}

/// Simple xorshift64 PRNG for particle placement and perturbation.
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self { state: if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl TorusBody {
    /// Generate a torus body with N particles distributed uniformly
    /// through the toroidal volume using rejection sampling.
    ///
    /// The volume element for a torus is:
    ///   dV = (R + ρ cos φ) · ρ · dρ · dφ · dθ
    ///
    /// Sampling: ρ = minor_r · √U (corrects for ρ dρ area element),
    /// accept with probability (R + ρ cos φ) / (R + minor_r) for the
    /// toroidal Jacobian.
    pub fn generate(
        n: usize,
        major_r: Decimal,
        minor_r: Decimal,
        mass: Decimal,
        charge: Decimal,
        stiffness: Decimal,
        damping: Decimal,
        seed: u64,
    ) -> Self {
        Self::generate_with_offset(n, major_r, minor_r, mass, charge, stiffness, damping, seed, Decimal::ZERO)
    }

    /// Generate a torus body with N particles at a given z_offset.
    pub fn generate_with_offset(
        n: usize,
        major_r: Decimal,
        minor_r: Decimal,
        mass: Decimal,
        charge: Decimal,
        stiffness: Decimal,
        damping: Decimal,
        seed: u64,
        z_offset: Decimal,
    ) -> Self {
        let mut rng = Rng64::new(seed);
        let mut particles = Vec::with_capacity(n);

        let big_r = major_r.to_f64().unwrap_or(0.10);
        let small_r = minor_r.to_f64().unwrap_or(0.03);
        let two_pi = 2.0 * std::f64::consts::PI;

        // Rejection sampling in toroidal coordinates
        while particles.len() < n {
            let theta = rng.next_f64() * two_pi;
            let phi = rng.next_f64() * two_pi;
            let u = rng.next_f64();
            let rho = small_r * u.sqrt(); // ρ = r·√U for area correction

            // Accept with probability (R + ρ cos φ) / (R + r)
            let accept_prob = (big_r + rho * phi.cos()) / (big_r + small_r);
            if rng.next_f64() >= accept_prob {
                continue;
            }

            // Toroidal → Cartesian
            let d = big_r + rho * phi.cos();
            let x = d * theta.cos();
            let y = d * theta.sin();
            let z = rho * phi.sin();

            let pos = Vec3::new(
                Decimal::from_f64_retain(x).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(y).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(z).unwrap_or(Decimal::ZERO) + z_offset,
            );

            particles.push(Particle {
                pos,
                vel: Vec3::ZERO,
                mass,
                charge,
            });
        }

        // Build bond network via spatial hash
        // V = 2π²Rr², spacing = (V/N)^(1/3), cutoff = 1.6 × spacing
        let vol = two_pi * std::f64::consts::PI * big_r * small_r * small_r;
        let spacing = (vol / n as f64).cbrt();
        let cutoff = 1.6 * spacing;
        let cell_size = cutoff;
        let cutoff_sq_dec = {
            let cd = Decimal::from_f64_retain(cutoff * cutoff).unwrap_or(Decimal::ZERO);
            cd
        };

        // Spatial hash grid
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (idx, p) in particles.iter().enumerate() {
            let cx = (p.pos.x.to_f64().unwrap_or(0.0) / cell_size).floor() as i32;
            let cy = (p.pos.y.to_f64().unwrap_or(0.0) / cell_size).floor() as i32;
            let cz = (p.pos.z.to_f64().unwrap_or(0.0) / cell_size).floor() as i32;
            grid.entry((cx, cy, cz)).or_default().push(idx);
        }

        let mut bonds = Vec::new();

        for idx in 0..particles.len() {
            let px = particles[idx].pos.x.to_f64().unwrap_or(0.0);
            let py = particles[idx].pos.y.to_f64().unwrap_or(0.0);
            let pz = particles[idx].pos.z.to_f64().unwrap_or(0.0);
            let cx = (px / cell_size).floor() as i32;
            let cy = (py / cell_size).floor() as i32;
            let cz = (pz / cell_size).floor() as i32;

            for dx in -1..=1_i32 {
                for dy in -1..=1_i32 {
                    for dz in -1..=1_i32 {
                        let key = (cx + dx, cy + dy, cz + dz);
                        if let Some(neighbors) = grid.get(&key) {
                            for &j in neighbors {
                                if j <= idx { continue; }
                                let sep = particles[j].pos - particles[idx].pos;
                                let dist_sq = sep.mag_sq();
                                if dist_sq < cutoff_sq_dec && !dist_sq.is_zero() {
                                    let rest_length = decimal_sqrt(dist_sq);
                                    bonds.push(Bond { i: idx, j, rest_length });
                                }
                            }
                        }
                    }
                }
            }
        }

        let n = particles.len();
        TorusBody {
            particles,
            bonds,
            stiffness,
            damping,
            torus_ids: vec![0; n],
            n_tori: 1,
        }
    }

    /// Centre of mass position.
    pub fn com(&self) -> Vec3 {
        let mut total_pos = Vec3::ZERO;
        let mut total_mass = Decimal::ZERO;
        for p in &self.particles {
            total_pos = total_pos + p.pos.scale(p.mass);
            total_mass += p.mass;
        }
        if total_mass.is_zero() {
            Vec3::ZERO
        } else {
            total_pos.scale(Decimal::ONE / total_mass)
        }
    }

    /// Centre of mass velocity.
    pub fn com_velocity(&self) -> Vec3 {
        let mut total_vel = Vec3::ZERO;
        let mut total_mass = Decimal::ZERO;
        for p in &self.particles {
            total_vel = total_vel + p.vel.scale(p.mass);
            total_mass += p.mass;
        }
        if total_mass.is_zero() {
            Vec3::ZERO
        } else {
            total_vel.scale(Decimal::ONE / total_mass)
        }
    }

    /// Angular velocity about z-axis through CoM.
    /// ω_z = Σ m_i (x'_i v'_yi - y'_i v'_xi) / Σ m_i (x'_i² + y'_i²)
    /// where primed coords are relative to CoM.
    pub fn angular_velocity_z(&self) -> Decimal {
        use rust_decimal::prelude::ToPrimitive;
        let com = self.com();
        let com_v = self.com_velocity();
        let mut numerator: f64 = 0.0;
        let mut denominator: f64 = 0.0;
        for p in &self.particles {
            let m = p.mass.to_f64().unwrap_or(0.0);
            let rx = (p.pos.x - com.x).to_f64().unwrap_or(0.0);
            let ry = (p.pos.y - com.y).to_f64().unwrap_or(0.0);
            let vx = (p.vel.x - com_v.x).to_f64().unwrap_or(0.0);
            let vy = (p.vel.y - com_v.y).to_f64().unwrap_or(0.0);
            numerator += m * (rx * vy - ry * vx);
            denominator += m * (rx * rx + ry * ry);
        }
        if denominator.abs() < 1e-30 {
            Decimal::ZERO
        } else {
            Decimal::from_f64_retain(numerator / denominator).unwrap_or(Decimal::ZERO)
        }
    }

    /// Total kinetic energy (computed in f64 to avoid rust_decimal overflow
    /// at high angular velocities like 5236 rad/s).
    pub fn kinetic_energy(&self) -> Decimal {
        use rust_decimal::prelude::ToPrimitive;
        let mut ke: f64 = 0.0;
        for p in &self.particles {
            let m = p.mass.to_f64().unwrap_or(0.0);
            let vx = p.vel.x.to_f64().unwrap_or(0.0);
            let vy = p.vel.y.to_f64().unwrap_or(0.0);
            let vz = p.vel.z.to_f64().unwrap_or(0.0);
            ke += 0.5 * m * (vx * vx + vy * vy + vz * vz);
        }
        Decimal::from_f64_retain(ke).unwrap_or(Decimal::ZERO)
    }

    /// Compute bond forces and return as a force array indexed by particle.
    /// Bond force: F_ij = [k(|r_ij| - r₀) + γ(v_rel · r̂_ij)] r̂_ij
    pub fn bond_forces(&self) -> Vec<Vec3> {
        let mut forces = vec![Vec3::ZERO; self.particles.len()];
        for bond in &self.bonds {
            let r_vec = self.particles[bond.j].pos - self.particles[bond.i].pos;
            let dist = r_vec.mag();
            if dist.is_zero() { continue; }
            let inv_dist = Decimal::ONE / dist;
            let r_hat = r_vec.scale(inv_dist);

            let stretch = dist - bond.rest_length;
            let v_rel = self.particles[bond.j].vel - self.particles[bond.i].vel;
            let v_radial = v_rel.dot(r_hat);

            let f_mag = self.stiffness * stretch + self.damping * v_radial;
            let f = r_hat.scale(f_mag);

            forces[bond.i] = forces[bond.i] + f;
            forces[bond.j] = forces[bond.j] - f;
        }
        forces
    }

    /// Set rigid spin about z-axis through CoM.
    /// v_i = ω × (r_i - com) = ω_z · (-Δy, Δx, 0)
    pub fn set_spin(&mut self, omega_z: Decimal) {
        let com = self.com();
        for p in &mut self.particles {
            let dx = p.pos.x - com.x;
            let dy = p.pos.y - com.y;
            p.vel.x = -omega_z * dy;
            p.vel.y = omega_z * dx;
            p.vel.z = Decimal::ZERO;
        }
    }

    /// Reimpose rigid-body rotation while preserving COM velocity.
    /// v_i = v_com + ω_z × (r_i - com)
    ///
    /// Eliminates bond-induced spin decay (numerical artifact) while
    /// keeping translational degrees of freedom intact.
    pub fn reimpose_rigid_rotation(&mut self, omega_z: Decimal) {
        let com = self.com();
        let com_vel = self.com_velocity();
        for p in &mut self.particles {
            let dx = p.pos.x - com.x;
            let dy = p.pos.y - com.y;
            p.vel.x = com_vel.x + (-omega_z * dy);
            p.vel.y = com_vel.y + (omega_z * dx);
            p.vel.z = com_vel.z;
        }
    }

    /// Moment of inertia about the z-axis through COM: I_z = Σ m_i (x'² + y'²)
    pub fn moment_of_inertia_z(&self) -> Decimal {
        let com = self.com();
        let mut i_z = Decimal::ZERO;
        for p in &self.particles {
            let dx = p.pos.x - com.x;
            let dy = p.pos.y - com.y;
            i_z += p.mass * (dx * dx + dy * dy);
        }
        i_z
    }

    /// Set per-torus rigid spin (for counter-rotation).
    /// omegas[torus_id] gives the angular velocity for each torus.
    pub fn set_spin_multi(&mut self, omegas: &[Decimal]) {
        let com = self.com();
        for (idx, p) in self.particles.iter_mut().enumerate() {
            let t_id = self.torus_ids[idx];
            let omega = if t_id < omegas.len() { omegas[t_id] } else { Decimal::ZERO };
            let dx = p.pos.x - com.x;
            let dy = p.pos.y - com.y;
            p.vel.x = -omega * dy;
            p.vel.y = omega * dx;
            p.vel.z = Decimal::ZERO;
        }
    }

    /// Per-torus reimpose: preserves COM velocity, applies per-torus omega.
    pub fn reimpose_rigid_rotation_multi(&mut self, omegas: &[Decimal]) {
        let com = self.com();
        let com_vel = self.com_velocity();
        for (idx, p) in self.particles.iter_mut().enumerate() {
            let t_id = self.torus_ids[idx];
            let omega = if t_id < omegas.len() { omegas[t_id] } else { Decimal::ZERO };
            let dx = p.pos.x - com.x;
            let dy = p.pos.y - com.y;
            p.vel.x = com_vel.x + (-omega * dy);
            p.vel.y = com_vel.y + (omega * dx);
            p.vel.z = com_vel.z;
        }
    }

    /// Per-torus moment of inertia about the z-axis.
    pub fn moment_of_inertia_z_per_torus(&self) -> Vec<Decimal> {
        let com = self.com();
        let mut i_z = vec![Decimal::ZERO; self.n_tori.max(1)];
        for (idx, p) in self.particles.iter().enumerate() {
            let t_id = self.torus_ids[idx];
            if t_id < i_z.len() {
                let dx = p.pos.x - com.x;
                let dy = p.pos.y - com.y;
                i_z[t_id] += p.mass * (dx * dx + dy * dy);
            }
        }
        i_z
    }

    /// Total mass of the body.
    pub fn total_mass(&self) -> Decimal {
        self.particles.iter().map(|p| p.mass).sum()
    }

    /// Total momentum.
    pub fn momentum(&self) -> Vec3 {
        let mut p_total = Vec3::ZERO;
        for p in &self.particles {
            p_total = p_total + p.vel.scale(p.mass);
        }
        p_total
    }

    /// Shake the lattice: perturb every particle position by a random
    /// displacement of magnitude up to `amplitude`. Models thermal disorder.
    /// Returns the RMS displacement for logging.
    pub fn shake(&mut self, amplitude: Decimal, rng: &mut Rng64) -> Decimal {
        let mut sum_sq = Decimal::ZERO;
        for p in &mut self.particles {
            // Box-Muller-ish: uniform random direction, uniform magnitude
            let theta = rng.next_f64() * 2.0 * std::f64::consts::PI;
            let phi = rng.next_f64() * std::f64::consts::PI;
            let r = rng.next_f64(); // [0,1)
            let mag = amplitude * Decimal::from_f64_retain(r).unwrap_or(Decimal::ZERO);

            let sin_phi = Decimal::from_f64_retain(phi.sin()).unwrap_or(Decimal::ZERO);
            let cos_phi = Decimal::from_f64_retain(phi.cos()).unwrap_or(Decimal::ZERO);
            let sin_theta = Decimal::from_f64_retain(theta.sin()).unwrap_or(Decimal::ZERO);
            let cos_theta = Decimal::from_f64_retain(theta.cos()).unwrap_or(Decimal::ZERO);

            let dx = mag * sin_phi * cos_theta;
            let dy = mag * sin_phi * sin_theta;
            let dz = mag * cos_phi;

            p.pos.x += dx;
            p.pos.y += dy;
            p.pos.z += dz;

            sum_sq += dx * dx + dy * dy + dz * dz;
        }
        let n = Decimal::from(self.particles.len() as u64);
        if n.is_zero() { Decimal::ZERO } else { decimal_sqrt(sum_sq / n) }
    }

    /// Total angular momentum about z-axis through CoM.
    pub fn angular_momentum_z(&self) -> Decimal {
        use rust_decimal::prelude::ToPrimitive;
        let com = self.com();
        let com_v = self.com_velocity();
        let mut lz: f64 = 0.0;
        for p in &self.particles {
            let m = p.mass.to_f64().unwrap_or(0.0);
            let rx = (p.pos.x - com.x).to_f64().unwrap_or(0.0);
            let ry = (p.pos.y - com.y).to_f64().unwrap_or(0.0);
            let vx = (p.vel.x - com_v.x).to_f64().unwrap_or(0.0);
            let vy = (p.vel.y - com_v.y).to_f64().unwrap_or(0.0);
            lz += m * (rx * vy - ry * vx);
        }
        Decimal::from_f64_retain(lz).unwrap_or(Decimal::ZERO)
    }
}

/// Generate a torus body with default spec parameters.
/// Seed should be the same for all four bodies so they start identical.
pub fn generate_default_torus(seed: u64) -> TorusBody {
    TorusBody::generate(
        500,           // N particles
        dec!(0.10),    // R major radius
        dec!(0.03),    // r minor radius
        dec!(0.001),   // m₀ = 0.001 kg
        dec!(0.000001), // q = 1 μC
        dec!(10000),   // k = 10⁴ N/m
        dec!(10),      // γ = 10 N·s/m
        seed,
    )
}

/// Generate a torus body from a TorusConfig.
/// If material is set, resolves mass from density × volume / n_particles.
pub fn generate_torus_from_config(cfg: &TorusConfig, seed: u64) -> TorusBody {
    let mass = GeometryConfig::resolve_particle_mass(cfg);
    TorusBody::generate_with_offset(
        cfg.n_particles,
        Decimal::from_f64_retain(cfg.major_r).unwrap_or(dec!(0.10)),
        Decimal::from_f64_retain(cfg.minor_r).unwrap_or(dec!(0.03)),
        Decimal::from_f64_retain(mass).unwrap_or(dec!(0.001)),
        Decimal::from_f64_retain(cfg.charge).unwrap_or(dec!(0.000001)),
        Decimal::from_f64_retain(cfg.stiffness).unwrap_or(dec!(10000)),
        Decimal::from_f64_retain(cfg.damping).unwrap_or(dec!(10)),
        seed,
        Decimal::from_f64_retain(cfg.z_offset).unwrap_or(Decimal::ZERO),
    )
}

/// Generate multiple tori from config, one per TorusConfig entry.
/// Returns them all with the same seed (identical initial shape, different z).
pub fn generate_tori_from_config(configs: &[TorusConfig], seed: u64) -> Vec<TorusBody> {
    configs.iter().map(|cfg| generate_torus_from_config(cfg, seed)).collect()
}

/// Merge multiple TorusBody instances into a single body.
/// Bond indices are offset so each torus's bonds reference the correct particles.
/// Uses stiffness/damping from the first body.
pub fn merge_tori(tori: &[TorusBody]) -> TorusBody {
    if tori.is_empty() {
        return TorusBody {
            particles: vec![],
            bonds: vec![],
            stiffness: dec!(10000),
            damping: dec!(10),
            torus_ids: vec![],
            n_tori: 0,
        };
    }
    if tori.len() == 1 {
        return tori[0].clone();
    }

    let mut particles = Vec::new();
    let mut bonds = Vec::new();
    let mut torus_ids = Vec::new();

    for (t_idx, body) in tori.iter().enumerate() {
        let offset = particles.len();
        particles.extend(body.particles.iter().cloned());
        torus_ids.extend(std::iter::repeat(t_idx).take(body.particles.len()));
        for bond in &body.bonds {
            bonds.push(Bond {
                i: bond.i + offset,
                j: bond.j + offset,
                rest_length: bond.rest_length,
            });
        }
    }

    TorusBody {
        particles,
        bonds,
        stiffness: tori[0].stiffness,
        damping: tori[0].damping,
        torus_ids,
        n_tori: tori.len(),
    }
}

/// Generate a disk body: uniform particles in a thin cylinder.
/// Uses sqrt(r) correction for area uniformity.
pub fn generate_disk(
    n: usize,
    radius: f64,
    thickness: f64,
    z_offset: f64,
    mass: Decimal,
    charge: Decimal,
    stiffness: Decimal,
    damping: Decimal,
    seed: u64,
) -> TorusBody {
    let mut rng = Rng64::new(seed);
    let mut particles = Vec::with_capacity(n);
    let two_pi = 2.0 * std::f64::consts::PI;

    while particles.len() < n {
        let theta = rng.next_f64() * two_pi;
        let r = radius * rng.next_f64().sqrt(); // sqrt for area correction
        let z = (rng.next_f64() - 0.5) * thickness + z_offset;

        let x = r * theta.cos();
        let y = r * theta.sin();

        particles.push(Particle {
            pos: Vec3::new(
                Decimal::from_f64_retain(x).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(y).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(z).unwrap_or(Decimal::ZERO),
            ),
            vel: Vec3::ZERO,
            mass,
            charge,
        });
    }

    build_bond_network(particles, stiffness, damping)
}

/// Generate a cylinder body: uniform particles in a cylinder.
pub fn generate_cylinder(
    n: usize,
    radius: f64,
    height: f64,
    z_offset: f64,
    mass: Decimal,
    charge: Decimal,
    stiffness: Decimal,
    damping: Decimal,
    seed: u64,
) -> TorusBody {
    let mut rng = Rng64::new(seed);
    let mut particles = Vec::with_capacity(n);
    let two_pi = 2.0 * std::f64::consts::PI;

    while particles.len() < n {
        let theta = rng.next_f64() * two_pi;
        let r = radius * rng.next_f64().sqrt();
        let z = (rng.next_f64() - 0.5) * height + z_offset;

        let x = r * theta.cos();
        let y = r * theta.sin();

        particles.push(Particle {
            pos: Vec3::new(
                Decimal::from_f64_retain(x).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(y).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(z).unwrap_or(Decimal::ZERO),
            ),
            vel: Vec3::ZERO,
            mass,
            charge,
        });
    }

    build_bond_network(particles, stiffness, damping)
}

/// Generate a sphere body: uniform particles in a sphere (rejection sampling).
pub fn generate_sphere(
    n: usize,
    radius: f64,
    center: [f64; 3],
    mass: Decimal,
    charge: Decimal,
    stiffness: Decimal,
    damping: Decimal,
    seed: u64,
) -> TorusBody {
    let mut rng = Rng64::new(seed);
    let mut particles = Vec::with_capacity(n);

    while particles.len() < n {
        let x = (rng.next_f64() * 2.0 - 1.0) * radius;
        let y = (rng.next_f64() * 2.0 - 1.0) * radius;
        let z = (rng.next_f64() * 2.0 - 1.0) * radius;
        if x * x + y * y + z * z > radius * radius { continue; }

        particles.push(Particle {
            pos: Vec3::new(
                Decimal::from_f64_retain(x + center[0]).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(y + center[1]).unwrap_or(Decimal::ZERO),
                Decimal::from_f64_retain(z + center[2]).unwrap_or(Decimal::ZERO),
            ),
            vel: Vec3::ZERO,
            mass,
            charge,
        });
    }

    build_bond_network(particles, stiffness, damping)
}

/// Build a bond network from a particle list using spatial hash grid.
/// Shared by all shape generators.
fn build_bond_network(
    particles: Vec<Particle>,
    stiffness: Decimal,
    damping: Decimal,
) -> TorusBody {
    let n = particles.len();
    if n == 0 {
        return TorusBody {
            particles, bonds: vec![], stiffness, damping,
            torus_ids: vec![], n_tori: 1,
        };
    }

    // Estimate spacing from bounding volume
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    let mut min_z = f64::MAX;
    let mut max_z = f64::MIN;
    for p in &particles {
        let px = p.pos.x.to_f64().unwrap_or(0.0);
        let py = p.pos.y.to_f64().unwrap_or(0.0);
        let pz = p.pos.z.to_f64().unwrap_or(0.0);
        min_x = min_x.min(px); max_x = max_x.max(px);
        min_y = min_y.min(py); max_y = max_y.max(py);
        min_z = min_z.min(pz); max_z = max_z.max(pz);
    }
    let vol = (max_x - min_x).max(0.001) * (max_y - min_y).max(0.001) * (max_z - min_z).max(0.001);
    let spacing = (vol / n as f64).cbrt();
    let cutoff = 1.6 * spacing;
    let cell_size = cutoff;
    let cutoff_sq_dec = Decimal::from_f64_retain(cutoff * cutoff).unwrap_or(Decimal::ZERO);

    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (idx, p) in particles.iter().enumerate() {
        let cx = (p.pos.x.to_f64().unwrap_or(0.0) / cell_size).floor() as i32;
        let cy = (p.pos.y.to_f64().unwrap_or(0.0) / cell_size).floor() as i32;
        let cz = (p.pos.z.to_f64().unwrap_or(0.0) / cell_size).floor() as i32;
        grid.entry((cx, cy, cz)).or_default().push(idx);
    }

    let mut bonds = Vec::new();
    for idx in 0..particles.len() {
        let px = particles[idx].pos.x.to_f64().unwrap_or(0.0);
        let py = particles[idx].pos.y.to_f64().unwrap_or(0.0);
        let pz = particles[idx].pos.z.to_f64().unwrap_or(0.0);
        let cx = (px / cell_size).floor() as i32;
        let cy = (py / cell_size).floor() as i32;
        let cz = (pz / cell_size).floor() as i32;

        for dx in -1..=1_i32 {
            for dy in -1..=1_i32 {
                for dz in -1..=1_i32 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(neighbors) = grid.get(&key) {
                        for &j in neighbors {
                            if j <= idx { continue; }
                            let sep = particles[j].pos - particles[idx].pos;
                            let dist_sq = sep.mag_sq();
                            if dist_sq < cutoff_sq_dec && !dist_sq.is_zero() {
                                let rest_length = decimal_sqrt(dist_sq);
                                bonds.push(Bond { i: idx, j, rest_length });
                            }
                        }
                    }
                }
            }
        }
    }

    TorusBody {
        particles, bonds, stiffness, damping,
        torus_ids: vec![0; n], n_tori: 1,
    }
}

/// Generate a body from a BodyConfig (generalized shape).
pub fn generate_body_from_config(cfg: &BodyConfig, seed: u64) -> TorusBody {
    match cfg {
        BodyConfig::Torus { z_offset, major_r, minor_r, n_particles, mass, charge,
                            stiffness, damping, material, .. } => {
            let resolved_mass = if let Some(ref mat_name) = material {
                if let Some(mat) = MaterialSpec::from_name(mat_name) {
                    let vol = 2.0 * std::f64::consts::PI * std::f64::consts::PI
                        * major_r * minor_r * minor_r;
                    mat.density * vol / *n_particles as f64
                } else { *mass }
            } else { *mass };
            TorusBody::generate_with_offset(
                *n_particles,
                Decimal::from_f64_retain(*major_r).unwrap_or(dec!(0.10)),
                Decimal::from_f64_retain(*minor_r).unwrap_or(dec!(0.03)),
                Decimal::from_f64_retain(resolved_mass).unwrap_or(dec!(0.001)),
                Decimal::from_f64_retain(*charge).unwrap_or(dec!(0.000001)),
                Decimal::from_f64_retain(*stiffness).unwrap_or(dec!(10000)),
                Decimal::from_f64_retain(*damping).unwrap_or(dec!(10)),
                seed,
                Decimal::from_f64_retain(*z_offset).unwrap_or(Decimal::ZERO),
            )
        }
        BodyConfig::Disk { z_offset, radius, thickness, n_particles, mass, charge,
                           stiffness, damping, material, .. } => {
            let resolved_mass = if let Some(ref mat_name) = material {
                if let Some(mat) = MaterialSpec::from_name(mat_name) {
                    let vol = std::f64::consts::PI * radius * radius * thickness;
                    mat.density * vol / *n_particles as f64
                } else { *mass }
            } else { *mass };
            generate_disk(*n_particles, *radius, *thickness, *z_offset,
                Decimal::from_f64_retain(resolved_mass).unwrap_or(dec!(0.001)),
                Decimal::from_f64_retain(*charge).unwrap_or(dec!(0.000001)),
                Decimal::from_f64_retain(*stiffness).unwrap_or(dec!(10000)),
                Decimal::from_f64_retain(*damping).unwrap_or(dec!(10)),
                seed)
        }
        BodyConfig::Cylinder { z_offset, radius, height, n_particles, mass, charge,
                               stiffness, damping, material, .. } => {
            let resolved_mass = if let Some(ref mat_name) = material {
                if let Some(mat) = MaterialSpec::from_name(mat_name) {
                    let vol = std::f64::consts::PI * radius * radius * height;
                    mat.density * vol / *n_particles as f64
                } else { *mass }
            } else { *mass };
            generate_cylinder(*n_particles, *radius, *height, *z_offset,
                Decimal::from_f64_retain(resolved_mass).unwrap_or(dec!(0.001)),
                Decimal::from_f64_retain(*charge).unwrap_or(dec!(0.000001)),
                Decimal::from_f64_retain(*stiffness).unwrap_or(dec!(10000)),
                Decimal::from_f64_retain(*damping).unwrap_or(dec!(10)),
                seed)
        }
        BodyConfig::Sphere { center, radius, n_particles, mass, charge,
                             stiffness, damping, material, .. } => {
            let resolved_mass = if let Some(ref mat_name) = material {
                if let Some(mat) = MaterialSpec::from_name(mat_name) {
                    let vol = (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius;
                    mat.density * vol / *n_particles as f64
                } else { *mass }
            } else { *mass };
            generate_sphere(*n_particles, *radius, *center,
                Decimal::from_f64_retain(resolved_mass).unwrap_or(dec!(0.001)),
                Decimal::from_f64_retain(*charge).unwrap_or(dec!(0.000001)),
                Decimal::from_f64_retain(*stiffness).unwrap_or(dec!(10000)),
                Decimal::from_f64_retain(*damping).unwrap_or(dec!(10)),
                seed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_generation() {
        let torus = generate_default_torus(42);
        assert_eq!(torus.particles.len(), 500);
        assert!(!torus.bonds.is_empty(), "should have bonds");
    }

    #[test]
    fn test_com_at_origin() {
        let torus = generate_default_torus(42);
        let com = torus.com();
        // CoM should be near origin for a symmetric torus
        let dist = com.mag();
        // Allow some statistical noise from 500 random particles
        use rust_decimal::prelude::ToPrimitive;
        assert!(dist.to_f64().unwrap() < 0.02, "CoM should be near origin, got {}", dist);
    }

    #[test]
    fn test_zero_initial_velocity() {
        let torus = generate_default_torus(42);
        let v = torus.com_velocity();
        assert!(v.x.is_zero());
        assert!(v.y.is_zero());
        assert!(v.z.is_zero());
    }

    #[test]
    fn test_spin_sets_angular_velocity() {
        let mut torus = generate_default_torus(42);
        let omega = dec!(100); // 100 rad/s
        torus.set_spin(omega);
        let measured = torus.angular_velocity_z();
        // Should be close to 100
        let diff = (measured - omega).abs();
        use rust_decimal::prelude::ToPrimitive;
        assert!(
            diff.to_f64().unwrap() < 1.0,
            "angular velocity should be ~100, got {}", measured
        );
    }

    #[test]
    fn test_particles_inside_torus() {
        let torus = generate_default_torus(42);
        use rust_decimal::prelude::ToPrimitive;
        for p in &torus.particles {
            let x = p.pos.x.to_f64().unwrap();
            let y = p.pos.y.to_f64().unwrap();
            let z = p.pos.z.to_f64().unwrap();
            // Distance from z-axis
            let d = (x * x + y * y).sqrt();
            // Distance from torus ring centre
            let rho = ((d - 0.10).powi(2) + z * z).sqrt();
            assert!(
                rho < 0.035, // minor_r = 0.03, small margin
                "particle at ({}, {}, {}) has rho={} > 0.035", x, y, z, rho
            );
        }
    }
}
