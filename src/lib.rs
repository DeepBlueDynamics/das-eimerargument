pub mod vec3;
pub mod config;
pub mod motor;
pub mod magnets;
pub mod torus;
pub mod weber;
pub mod integrator;

#[cfg(feature = "gpu")]
pub mod gpu;
