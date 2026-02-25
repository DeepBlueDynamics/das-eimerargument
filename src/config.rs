use serde::{Deserialize, Serialize};

// ==========================================================================
// Material Properties
// ==========================================================================

/// Material properties for simulation bodies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialSpec {
    pub name: String,
    pub density: f64,           // kg/m³
    pub susceptibility: f64,    // χ_m (dimensionless, negative = diamagnetic)
    pub conductivity: f64,      // σ (S/m)
}

impl MaterialSpec {
    /// Relative permeability: μ_r = 1 + χ_m
    pub fn relative_permeability(&self) -> f64 {
        1.0 + self.susceptibility
    }

    pub fn bismuth() -> Self {
        MaterialSpec { name: "bismuth".into(), density: 9780.0, susceptibility: -1.7e-4, conductivity: 7.7e5 }
    }

    pub fn copper() -> Self {
        MaterialSpec { name: "copper".into(), density: 8960.0, susceptibility: -9.6e-6, conductivity: 5.96e7 }
    }

    pub fn aluminum() -> Self {
        MaterialSpec { name: "aluminum".into(), density: 2700.0, susceptibility: 2.2e-5, conductivity: 3.77e7 }
    }

    pub fn iron() -> Self {
        MaterialSpec { name: "iron".into(), density: 7874.0, susceptibility: 200000.0, conductivity: 1.0e7 }
    }

    pub fn default_material() -> Self {
        MaterialSpec { name: "default".into(), density: 1000.0, susceptibility: 0.0, conductivity: 0.0 }
    }

    /// Look up a preset material by name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "bismuth" => Some(Self::bismuth()),
            "copper" => Some(Self::copper()),
            "aluminum" | "aluminium" => Some(Self::aluminum()),
            "iron" => Some(Self::iron()),
            "default" => Some(Self::default_material()),
            _ => None,
        }
    }
}

// ==========================================================================
// Rotation Specification (for magnets)
// ==========================================================================

/// Angular rotation specification for spinning magnets (Faraday's paradox).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotationSpec {
    pub omega: f64,             // angular velocity (rad/s)
    #[serde(default = "default_z_axis")]
    pub axis: [f64; 3],        // rotation axis (default [0,0,1])
    #[serde(default)]
    pub center: [f64; 3],      // rotation center (default [0,0,0])
}

fn default_z_axis() -> [f64; 3] { [0.0, 0.0, 1.0] }

// ==========================================================================
// Geometry Configuration
// ==========================================================================

/// Full geometry + physics configuration for the simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryConfig {
    /// Torus bodies (legacy + primary path).
    #[serde(default)]
    pub tori: Vec<TorusConfig>,
    /// Generalized bodies (new shapes — used by MCP tools).
    #[serde(default)]
    pub bodies: Vec<BodyConfig>,
    pub magnets: Vec<MagnetConfig>,
    pub motor: MotorConfig,
    pub protocol: ProtocolConfig,
}

/// Configuration for a single torus body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TorusConfig {
    pub z_offset: f64,
    pub major_r: f64,
    pub minor_r: f64,
    pub n_particles: usize,
    pub mass: f64,
    pub charge: f64,
    pub stiffness: f64,
    pub damping: f64,
    /// Material name (e.g. "bismuth", "copper"). If set, density overrides mass.
    #[serde(default)]
    pub material: Option<String>,
    /// Human-readable label.
    #[serde(default)]
    pub label: Option<String>,
}

/// Configuration for a single magnet (stack of current loops).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MagnetConfig {
    pub z_centre: f64,
    pub n_loops: usize,
    pub n_seg: usize,
    pub r_mag: f64,
    pub dz_loop: f64,
    pub current: f64,
    /// Optional rotation for spinning magnets (Faraday's paradox).
    #[serde(default)]
    pub rotation: Option<RotationSpec>,
    /// Human-readable label.
    #[serde(default)]
    pub label: Option<String>,
}

/// Motor PD controller configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotorConfig {
    pub omega_target: f64,
    pub k_p: f64,
    pub max_torque: f64,
    pub friction_mu: f64,
}

/// Six-phase protocol timing and thrust.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolConfig {
    pub spin_up_duration: f64,
    pub coast1_duration: f64,
    pub thrust_duration: f64,
    pub coast2_duration: f64,
    pub brake_duration: f64,
    pub coast3_duration: f64,
    pub thrust_force: f64,
    /// Thrust direction: "z" (perpendicular to torus, through hole) or "x" (in-plane).
    /// Default: "z" — the physically correct axis for measuring Weber mass correction.
    #[serde(default = "default_thrust_axis")]
    pub thrust_axis: String,
    pub dt: f64,
}

fn default_thrust_axis() -> String { "z".to_string() }

// ==========================================================================
// Generalized Body Configuration (new shapes)
// ==========================================================================

/// Generalized body configuration — supports multiple shapes beyond torus.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "shape")]
pub enum BodyConfig {
    #[serde(rename = "torus")]
    Torus {
        z_offset: f64,
        major_r: f64,
        minor_r: f64,
        n_particles: usize,
        mass: f64,
        charge: f64,
        stiffness: f64,
        damping: f64,
        #[serde(default)]
        material: Option<String>,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        motor: Option<MotorConfig>,
    },
    #[serde(rename = "disk")]
    Disk {
        z_offset: f64,
        radius: f64,
        thickness: f64,
        n_particles: usize,
        mass: f64,
        charge: f64,
        stiffness: f64,
        damping: f64,
        #[serde(default)]
        material: Option<String>,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        motor: Option<MotorConfig>,
    },
    #[serde(rename = "cylinder")]
    Cylinder {
        z_offset: f64,
        radius: f64,
        height: f64,
        n_particles: usize,
        mass: f64,
        charge: f64,
        stiffness: f64,
        damping: f64,
        #[serde(default)]
        material: Option<String>,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        motor: Option<MotorConfig>,
    },
    #[serde(rename = "sphere")]
    Sphere {
        center: [f64; 3],
        radius: f64,
        n_particles: usize,
        mass: f64,
        charge: f64,
        stiffness: f64,
        damping: f64,
        #[serde(default)]
        material: Option<String>,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        motor: Option<MotorConfig>,
    },
}

impl BodyConfig {
    pub fn material(&self) -> Option<&str> {
        match self {
            BodyConfig::Torus { material, .. } |
            BodyConfig::Disk { material, .. } |
            BodyConfig::Cylinder { material, .. } |
            BodyConfig::Sphere { material, .. } => material.as_deref(),
        }
    }

    pub fn label(&self) -> Option<&str> {
        match self {
            BodyConfig::Torus { label, .. } |
            BodyConfig::Disk { label, .. } |
            BodyConfig::Cylinder { label, .. } |
            BodyConfig::Sphere { label, .. } => label.as_deref(),
        }
    }

    pub fn motor(&self) -> Option<&MotorConfig> {
        match self {
            BodyConfig::Torus { motor, .. } |
            BodyConfig::Disk { motor, .. } |
            BodyConfig::Cylinder { motor, .. } |
            BodyConfig::Sphere { motor, .. } => motor.as_ref(),
        }
    }

    /// Convert a TorusConfig to a BodyConfig::Torus.
    pub fn from_torus_config(tc: &TorusConfig) -> Self {
        BodyConfig::Torus {
            z_offset: tc.z_offset,
            major_r: tc.major_r,
            minor_r: tc.minor_r,
            n_particles: tc.n_particles,
            mass: tc.mass,
            charge: tc.charge,
            stiffness: tc.stiffness,
            damping: tc.damping,
            material: tc.material.clone(),
            label: tc.label.clone(),
            motor: None,
        }
    }
}

// ==========================================================================
// Defaults
// ==========================================================================

impl Default for TorusConfig {
    fn default() -> Self {
        TorusConfig {
            z_offset: 0.0,
            major_r: 0.10,
            minor_r: 0.03,
            n_particles: 500,
            mass: 0.001,
            charge: 0.000001,
            stiffness: 1e8,    // Must resist centrifugal force at 5236 rad/s
            damping: 0.01,    // Near-zero: rigid solid doesn't damp internal motion
            material: None,
            label: None,
        }
    }
}

impl Default for MagnetConfig {
    fn default() -> Self {
        MagnetConfig {
            z_centre: 0.08,
            n_loops: 12,
            n_seg: 32,
            r_mag: 0.05,
            dz_loop: 0.005,
            current: 1000.0,
            rotation: None,
            label: None,
        }
    }
}

impl Default for MotorConfig {
    fn default() -> Self {
        MotorConfig {
            omega_target: 5236.0,
            k_p: 10.0,
            max_torque: 50.0,
            friction_mu: 0.001,
        }
    }
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        ProtocolConfig {
            spin_up_duration: 2.0,
            coast1_duration: 0.5,
            thrust_duration: 1.0,
            coast2_duration: 0.5,
            brake_duration: 2.0,
            coast3_duration: 0.5,
            thrust_force: 0.1,
            thrust_axis: "z".to_string(),
            dt: 0.0000001, // 0.1 us
        }
    }
}

// ==========================================================================
// GeometryConfig methods
// ==========================================================================

impl GeometryConfig {
    /// Single-torus default: backward-compatible with original hardcoded layout.
    /// 1 torus at z=0, 2 magnets at z=+/-0.08.
    pub fn single_default() -> Self {
        GeometryConfig {
            tori: vec![TorusConfig::default()],
            bodies: vec![],
            magnets: vec![
                MagnetConfig { z_centre: 0.08, ..MagnetConfig::default() },
                MagnetConfig { z_centre: -0.08, ..MagnetConfig::default() },
            ],
            motor: MotorConfig::default(),
            protocol: ProtocolConfig::default(),
        }
    }

    /// Stacked 2-torus default: 2 tori + 3 magnets vertically stacked.
    ///
    /// ```text
    ///     [Magnet TOP]      z = +0.16
    ///  === Torus TOP ===    z = +0.08
    ///     [Magnet MID]      z =  0.00
    ///  === Torus BOT ===    z = -0.08
    ///     [Magnet BOT]      z = -0.16
    /// ```
    pub fn stacked_default() -> Self {
        GeometryConfig {
            tori: vec![
                TorusConfig { z_offset: 0.08, ..TorusConfig::default() },
                TorusConfig { z_offset: -0.08, ..TorusConfig::default() },
            ],
            bodies: vec![],
            magnets: vec![
                MagnetConfig { z_centre: 0.16, ..MagnetConfig::default() },
                MagnetConfig { z_centre: 0.00, ..MagnetConfig::default() },
                MagnetConfig { z_centre: -0.16, ..MagnetConfig::default() },
            ],
            motor: MotorConfig::default(),
            protocol: ProtocolConfig::default(),
        }
    }

    /// Three bismuth tori — 3 counter-rotating diamagnetic tori with 4 magnets.
    ///
    /// ```text
    ///     [Magnet 1]       z = +0.24
    ///  === Torus TOP ===   z = +0.16  (bismuth, ω = +ω₀)
    ///     [Magnet 2]       z = +0.08
    ///  === Torus MID ===   z =  0.00  (bismuth, ω = −ω₀)
    ///     [Magnet 3]       z = -0.08
    ///  === Torus BOT ===   z = -0.16  (bismuth, ω = +ω₀)
    ///     [Magnet 4]       z = -0.24
    /// ```
    pub fn three_bismuth_tori() -> Self {
        let bismuth = MaterialSpec::bismuth();
        let major_r = 0.10;
        let minor_r = 0.03;
        let vol = 2.0 * std::f64::consts::PI * std::f64::consts::PI * major_r * minor_r * minor_r;
        let n_particles = 500usize;
        let mass_per_particle = bismuth.density * vol / n_particles as f64;

        GeometryConfig {
            tori: vec![
                TorusConfig {
                    z_offset: 0.16, mass: mass_per_particle,
                    material: Some("bismuth".into()), label: Some("top".into()),
                    ..TorusConfig::default()
                },
                TorusConfig {
                    z_offset: 0.00, mass: mass_per_particle,
                    material: Some("bismuth".into()), label: Some("mid".into()),
                    ..TorusConfig::default()
                },
                TorusConfig {
                    z_offset: -0.16, mass: mass_per_particle,
                    material: Some("bismuth".into()), label: Some("bot".into()),
                    ..TorusConfig::default()
                },
            ],
            bodies: vec![],
            magnets: vec![
                MagnetConfig { z_centre: 0.24, label: Some("mag_top".into()), ..MagnetConfig::default() },
                MagnetConfig { z_centre: 0.08, label: Some("mag_upper".into()), ..MagnetConfig::default() },
                MagnetConfig { z_centre: -0.08, label: Some("mag_lower".into()), ..MagnetConfig::default() },
                MagnetConfig { z_centre: -0.24, label: Some("mag_bot".into()), ..MagnetConfig::default() },
            ],
            motor: MotorConfig::default(),
            protocol: ProtocolConfig::default(),
        }
    }

    /// Load from a JSON file.
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let config: GeometryConfig = serde_json::from_str(&text)?;
        Ok(config)
    }

    /// Save to a JSON file.
    pub fn to_json_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Load from CLI --config arg, or return default.
    pub fn from_cli_or_default(default: Self) -> Self {
        let args: Vec<String> = std::env::args().collect();
        for i in 0..args.len() {
            if args[i] == "--config" {
                if let Some(path) = args.get(i + 1) {
                    match Self::from_json_file(path) {
                        Ok(cfg) => {
                            eprintln!("  Loaded config from {}", path);
                            return cfg;
                        }
                        Err(e) => {
                            eprintln!("  WARNING: Failed to load config {}: {}", path, e);
                        }
                    }
                }
            }
            if args[i] == "--stacked" {
                eprintln!("  Using stacked (2-torus, 3-magnet) geometry");
                return Self::stacked_default();
            }
            if args[i] == "--bismuth3" {
                eprintln!("  Using 3 bismuth tori geometry");
                return Self::three_bismuth_tori();
            }
        }
        default
    }

    /// Total simulation duration in seconds.
    pub fn total_duration(&self) -> f64 {
        let p = &self.protocol;
        p.spin_up_duration + p.coast1_duration + p.thrust_duration
            + p.coast2_duration + p.brake_duration + p.coast3_duration
    }

    /// Total number of timesteps.
    pub fn total_steps(&self) -> u64 {
        (self.total_duration() / self.protocol.dt) as u64
    }

    /// Resolve particle mass for a torus: if material is set, compute from density × volume / n.
    pub fn resolve_particle_mass(tc: &TorusConfig) -> f64 {
        if let Some(ref mat_name) = tc.material {
            if let Some(mat) = MaterialSpec::from_name(mat_name) {
                let vol = 2.0 * std::f64::consts::PI * std::f64::consts::PI
                    * tc.major_r * tc.minor_r * tc.minor_r;
                return mat.density * vol / tc.n_particles as f64;
            }
        }
        tc.mass
    }
}

impl ProtocolConfig {
    /// Short protocol for quick smoke tests (original 18ms durations).
    pub fn quick_test() -> Self {
        ProtocolConfig {
            spin_up_duration: 0.005,
            coast1_duration: 0.001,
            thrust_duration: 0.005,
            coast2_duration: 0.001,
            brake_duration: 0.005,
            coast3_duration: 0.001,
            thrust_force: 0.1,
            thrust_axis: "z".to_string(),
            dt: 0.0000001,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_default_roundtrip() {
        let cfg = GeometryConfig::single_default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: GeometryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg2.tori.len(), 1);
        assert_eq!(cfg2.magnets.len(), 2);
    }

    #[test]
    fn test_stacked_default() {
        let cfg = GeometryConfig::stacked_default();
        assert_eq!(cfg.tori.len(), 2);
        assert_eq!(cfg.magnets.len(), 3);
        assert!((cfg.tori[0].z_offset - 0.08).abs() < 1e-10);
        assert!((cfg.tori[1].z_offset - (-0.08)).abs() < 1e-10);
    }

    #[test]
    fn test_total_steps() {
        let cfg = GeometryConfig::single_default();
        // 2+0.5+1+0.5+2+0.5 = 6.5s at 0.1us = 65M steps
        assert_eq!(cfg.total_steps(), 65_000_000);
    }

    #[test]
    fn test_quick_test_protocol() {
        let p = ProtocolConfig::quick_test();
        let total = p.spin_up_duration + p.coast1_duration + p.thrust_duration
            + p.coast2_duration + p.brake_duration + p.coast3_duration;
        assert!((total - 0.018).abs() < 1e-10);
    }

    #[test]
    fn test_three_bismuth_tori() {
        let cfg = GeometryConfig::three_bismuth_tori();
        assert_eq!(cfg.tori.len(), 3);
        assert_eq!(cfg.magnets.len(), 4);
        assert!(cfg.tori[0].mass > 0.01, "bismuth mass should be >> default: {}", cfg.tori[0].mass);
        assert_eq!(cfg.tori[0].material.as_deref(), Some("bismuth"));
    }

    #[test]
    fn test_material_presets() {
        let bi = MaterialSpec::bismuth();
        assert!(bi.susceptibility < 0.0, "bismuth is diamagnetic");
        assert!((bi.relative_permeability() - (1.0 - 1.7e-4)).abs() < 1e-8);
        let fe = MaterialSpec::iron();
        assert!(fe.susceptibility > 1.0, "iron is ferromagnetic");
        assert!(MaterialSpec::from_name("bismuth").is_some());
        assert!(MaterialSpec::from_name("unknown").is_none());
    }

    #[test]
    fn test_body_config_roundtrip() {
        let body = BodyConfig::Torus {
            z_offset: 0.0, major_r: 0.10, minor_r: 0.03, n_particles: 100,
            mass: 0.001, charge: 0.000001, stiffness: 1e8, damping: 0.01,
            material: Some("bismuth".into()), label: Some("test".into()), motor: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        let body2: BodyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(body2.label(), Some("test"));
        assert_eq!(body2.material(), Some("bismuth"));
    }

    #[test]
    fn test_resolve_particle_mass() {
        let tc = TorusConfig { material: Some("bismuth".into()), ..TorusConfig::default() };
        let mass = GeometryConfig::resolve_particle_mass(&tc);
        // Bismuth torus: V = 2π²(0.10)(0.03²) ≈ 0.001775 m³
        // mass = 9780 × 0.001775 / 500 ≈ 0.0347
        assert!(mass > 0.03 && mass < 0.04, "bismuth mass per particle: {}", mass);
    }

    #[test]
    fn test_backward_compat_json() {
        // Old-format JSON (no material/label/bodies/rotation) should still parse
        let old_json = r#"{
            "tori": [{"z_offset": 0.0, "major_r": 0.10, "minor_r": 0.03, "n_particles": 500, "mass": 0.001, "charge": 0.000001, "stiffness": 100000000, "damping": 0.01}],
            "magnets": [{"z_centre": 0.08, "n_loops": 12, "n_seg": 32, "r_mag": 0.05, "dz_loop": 0.005, "current": 1000.0}],
            "motor": {"omega_target": 5236.0, "k_p": 10.0, "max_torque": 50.0, "friction_mu": 0.001},
            "protocol": {"spin_up_duration": 2.0, "coast1_duration": 0.5, "thrust_duration": 1.0, "coast2_duration": 0.5, "brake_duration": 2.0, "coast3_duration": 0.5, "thrust_force": 0.1, "dt": 0.0000001}
        }"#;
        let cfg: GeometryConfig = serde_json::from_str(old_json).unwrap();
        assert_eq!(cfg.tori.len(), 1);
        assert!(cfg.tori[0].material.is_none());
        assert!(cfg.bodies.is_empty());
    }
}
