//! Field Residual Simulation (Method 1)
//!
//! Static computation of B_weber - B_maxwell across a toroidal probe grid.
//! Uses c_eff amplification to bring the Weber correction into f64 range,
//! then verifies 1/c² scaling for extrapolation to physical c.
//!
//! No time-stepping. No bonds. No Decimal. Pure f64 geometry.
//! With --features gpu, runs probe computation on CUDA (one thread per probe).

use std::f64::consts::PI;
use std::io::Write;

use weber_anomaly::config::{GeometryConfig, MagnetConfig};

#[cfg(feature = "gpu")]
use weber_anomaly::gpu;

// ============================================================================
// Source elements (f64 version, higher resolution than N-body)
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct Source {
    pos: [f64; 3],
    dl: [f64; 3],
}

/// Generate source elements for one magnet.
/// 24 loops × 64 segments = 1,536 per magnet.
fn generate_magnet_f64(
    n_loops: usize,
    n_seg: usize,
    r_mag: f64,
    z_centre: f64,
    dz_loop: f64,
    current: f64,
) -> Vec<Source> {
    let mut elements = Vec::with_capacity(n_loops * n_seg);
    let seg_arc = 2.0 * PI / n_seg as f64;
    let dl_mag = current * seg_arc * r_mag;

    let half = (n_loops as f64 - 1.0) / 2.0;

    for k in 0..n_loops {
        let z_k = z_centre + (k as f64 - half) * dz_loop;

        for j in 0..n_seg {
            let theta = seg_arc * j as f64;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            elements.push(Source {
                pos: [r_mag * cos_t, r_mag * sin_t, z_k],
                dl: [-sin_t * dl_mag, cos_t * dl_mag, 0.0],
            });
        }
    }
    elements
}

/// Generate all sources: 2 magnets × 24 loops × 64 segments = 3,072 elements.
/// (Paper spec calls this 6,144 with different counting — we match the geometry.)
fn generate_all_sources_f64() -> Vec<Source> {
    let n_loops = 24;
    let n_seg = 64;
    let r_mag = 0.05;
    let dz_loop = 0.005;
    let current = 1000.0;

    let mut sources = generate_magnet_f64(n_loops, n_seg, r_mag, 0.08, dz_loop, current);
    sources.extend(generate_magnet_f64(n_loops, n_seg, r_mag, -0.08, dz_loop, current));
    sources
}

/// Generate sources from MagnetConfig list (uses higher-res params for field residual).
fn generate_sources_from_config_f64(configs: &[MagnetConfig]) -> Vec<Source> {
    let mut sources = Vec::new();
    for cfg in configs {
        // Field residual uses higher resolution: 24 loops, 64 segments
        sources.extend(generate_magnet_f64(
            24, 64, cfg.r_mag, cfg.z_centre, cfg.dz_loop, cfg.current,
        ));
    }
    sources
}

// ============================================================================
// Probe grid on the torus surface
// ============================================================================

#[derive(Clone, Debug)]
struct ProbePoint {
    pos: [f64; 3],
    theta: f64,       // toroidal angle
    phi: f64,         // poloidal angle
    layer: usize,     // radial layer index
    d_perp: f64,      // distance from rotation axis
    torus_id: usize,  // which torus this probe belongs to
}

/// Generate probe points on the torus: N_theta × N_phi × N_rho.
fn generate_probes(
    n_theta: usize,
    n_phi: usize,
    n_rho: usize,
    r_major: f64,
    r_minor: f64,
) -> Vec<ProbePoint> {
    generate_probes_with_offset(n_theta, n_phi, n_rho, r_major, r_minor, 0.0, 0)
}

/// Generate probe points with z_offset and torus_id tag.
fn generate_probes_with_offset(
    n_theta: usize,
    n_phi: usize,
    n_rho: usize,
    r_major: f64,
    r_minor: f64,
    z_offset: f64,
    torus_id: usize,
) -> Vec<ProbePoint> {
    let mut probes = Vec::with_capacity(n_theta * n_phi * n_rho);

    for k in 0..n_rho {
        let rho = r_minor * (k as f64 + 1.0) / n_rho as f64;

        for j in 0..n_phi {
            let phi = 2.0 * PI * j as f64 / n_phi as f64;

            for i in 0..n_theta {
                let theta = 2.0 * PI * i as f64 / n_theta as f64;

                let d_perp = r_major + rho * phi.cos();
                let x = d_perp * theta.cos();
                let y = d_perp * theta.sin();
                let z = rho * phi.sin() + z_offset;

                probes.push(ProbePoint {
                    pos: [x, y, z],
                    theta,
                    phi,
                    layer: k,
                    d_perp,
                    torus_id,
                });
            }
        }
    }
    probes
}

/// Generate probes for all tori in config.
fn generate_probes_from_config(
    n_theta: usize, n_phi: usize, n_rho: usize,
    cfg: &GeometryConfig,
) -> Vec<ProbePoint> {
    let mut all_probes = Vec::new();
    for (id, tc) in cfg.tori.iter().enumerate() {
        all_probes.extend(generate_probes_with_offset(
            n_theta, n_phi, n_rho, tc.major_r, tc.minor_r, tc.z_offset, id,
        ));
    }
    all_probes
}

// ============================================================================
// Field computation — direct δW accumulation (no catastrophic cancellation)
// ============================================================================

/// Result for one probe point.
#[derive(Clone, Debug, Default)]
struct ProbeResult {
    // Transverse field residual: Σ dB × δW
    delta_b: [f64; 3],
    // Standard Biot-Savart field (for reference)
    b_maxwell: [f64; 3],
    // Longitudinal force density (separate observable)
    f_longitudinal: [f64; 3],
    // Mean bracket deviation
    mean_delta_w: f64,
    // Local v²/c² for correlation analysis
    v_sq_over_c_sq: f64,
    // Local R(a·R̂)/c² averaged over sources
    mean_accel_term: f64,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mag(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

const MU0_4PI: f64 = 1.0e-7;

/// Compute field residual at a single probe point using direct δW accumulation.
///
/// CRITICAL: We compute δW = v²/c² − (3/2)(v·R̂)²/c² + R(a·R̂)/c²
/// directly — never forming W = 1 + δW, never subtracting fields.
fn compute_probe(
    probe: &ProbePoint,
    omega: f64,
    c_eff_sq: f64,
    sources: &[Source],
) -> ProbeResult {
    // Rigid rotation kinematics about z-axis
    let vel = [-omega * probe.pos[1], omega * probe.pos[0], 0.0];
    let accel = [-omega * omega * probe.pos[0], -omega * omega * probe.pos[1], 0.0];
    let v_sq = dot(vel, vel);

    let mut delta_b = [0.0f64; 3];
    let mut b_maxwell = [0.0f64; 3];
    let mut f_long_total = [0.0f64; 3];
    let mut delta_w_sum = 0.0f64;
    let mut accel_term_sum = 0.0f64;
    let mut n = 0u32;

    for s in sources {
        let r_vec = [
            probe.pos[0] - s.pos[0],
            probe.pos[1] - s.pos[1],
            probe.pos[2] - s.pos[2],
        ];
        let r_sq = dot(r_vec, r_vec);
        if r_sq < 1.0e-30 { continue; }
        let r = r_sq.sqrt();
        let inv_r = 1.0 / r;
        let r_hat = scale(r_vec, inv_r);
        let inv_r_sq = inv_r * inv_r;

        // Standard Biot-Savart: dB = (μ₀/4π) (dl × R̂) / R²
        let dl_cross_rhat = cross(s.dl, r_hat);
        let db_maxwell = scale(dl_cross_rhat, MU0_4PI * inv_r_sq);
        b_maxwell = add(b_maxwell, db_maxwell);

        // δW = v²/c² − (3/2)(v·R̂)²/c² + R(a·R̂)/c²
        // Computed DIRECTLY — no W = 1 + δW, no subtraction
        let v_dot_rhat = dot(vel, r_hat);
        let a_dot_rhat = dot(accel, r_hat);
        let accel_term = r * a_dot_rhat / c_eff_sq;

        let delta_w = v_sq / c_eff_sq
            - 1.5 * v_dot_rhat * v_dot_rhat / c_eff_sq
            + accel_term;

        // Accumulate: ΔB = Σ dB_maxwell × δW
        delta_b = add(delta_b, scale(db_maxwell, delta_w));

        delta_w_sum += delta_w;
        accel_term_sum += accel_term;
        n += 1;

        // Longitudinal force: F_long = (μ₀/4π)(q/R²)[dl·v − (3/2)(dl·R̂)(v·R̂)] R̂ / c²
        let dl_dot_v = dot(s.dl, vel);
        let dl_dot_rhat = dot(s.dl, r_hat);
        let f_long_mag = MU0_4PI * inv_r_sq
            * (dl_dot_v - 1.5 * dl_dot_rhat * v_dot_rhat) / c_eff_sq;
        f_long_total = add(f_long_total, scale(r_hat, f_long_mag));
    }

    let n_f = n as f64;
    ProbeResult {
        delta_b,
        b_maxwell,
        f_longitudinal: f_long_total,
        mean_delta_w: if n > 0 { delta_w_sum / n_f } else { 0.0 },
        v_sq_over_c_sq: v_sq / c_eff_sq,
        mean_accel_term: if n > 0 { accel_term_sum / n_f } else { 0.0 },
    }
}

// ============================================================================
// Statistics
// ============================================================================

fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x < 1e-300 || var_y < 1e-300 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Linear regression y = a + b*x, returns (a, b, r²)
fn linreg(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-300 { return (0.0, 0.0, 0.0); }
    let b = (n * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n;
    let r = pearson_r(xs, ys);
    (a, b, r * r)
}

// ============================================================================
// Main sweep driver
// ============================================================================

fn main() {
    #[cfg(feature = "gpu")]
    {
        eprintln!("GPU feature enabled — launching CUDA field residual");
        match main_gpu_field() {
            Ok(()) => return,
            Err(e) => eprintln!("GPU failed: {}. Falling back to CPU.", e),
        }
    }

    main_cpu_field();
}

#[cfg(feature = "gpu")]
fn main_gpu_field() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Weber Field Residual (GPU) ===");
    eprintln!("f64 CUDA kernels, one thread per probe");
    eprintln!();

    let sources_raw = generate_all_sources_f64();
    eprintln!("{} source elements", sources_raw.len());

    let n_theta = 64;
    let n_phi = 32;
    let n_rho = 4;
    let r_major = 0.10;
    let r_minor = 0.03;
    let probes = generate_probes(n_theta, n_phi, n_rho, r_major, r_minor);
    eprintln!("{} probe points", probes.len());

    // Convert to GPU format
    let gpu_probes: Vec<_> = probes.iter().map(|p| {
        (p.pos[0], p.pos[1], p.pos[2], p.theta, p.phi, p.layer, p.d_perp)
    }).collect();
    let gpu_sources: Vec<_> = sources_raw.iter().map(|s| {
        (s.pos[0], s.pos[1], s.pos[2], s.dl[0], s.dl[1], s.dl[2])
    }).collect();

    let gpu = gpu::FieldResidualGpu::new(&gpu_probes, &gpu_sources)?;

    let c_eff_values: Vec<f64> = vec![1.0e2, 3.0e2, 1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5];
    let omega_values: Vec<f64> = (0..=12).map(|i| i as f64 * 100.0).collect();
    let physical_c_sq: f64 = 299792458.0_f64 * 299792458.0_f64;

    let mut csv_scaling = std::fs::File::create("field_residual_scaling_gpu.csv")?;
    writeln!(csv_scaling, "c_eff,c_eff_sq,omega_sq_slope,omega_sq_r_squared,extrapolated_delta_b_at_physical_c")?;

    let start = std::time::Instant::now();

    for &c_eff in &c_eff_values {
        let c_eff_sq = c_eff * c_eff;
        let mut omega_sq_points = Vec::new();
        let mut delta_b_rms_points = Vec::new();

        for &omega in &omega_values {
            let results = gpu.compute(omega, c_eff_sq)?;

            let mut db_sq_sum = 0.0f64;
            for r in &results {
                let db_mag = (r.delta_b[0]*r.delta_b[0] + r.delta_b[1]*r.delta_b[1] + r.delta_b[2]*r.delta_b[2]).sqrt();
                db_sq_sum += db_mag * db_mag;
            }
            let delta_b_rms = (db_sq_sum / results.len() as f64).sqrt();

            if omega > 0.1 {
                omega_sq_points.push(omega * omega);
                delta_b_rms_points.push(delta_b_rms);
            }
        }

        let (_, slope, r_sq) = linreg(&omega_sq_points, &delta_b_rms_points);
        let scale_factor = c_eff_sq / physical_c_sq;
        let extrapolated = delta_b_rms_points.last().copied().unwrap_or(0.0) * scale_factor;

        writeln!(csv_scaling, "{},{},{:.10e},{:.8},{:.10e}",
            c_eff, c_eff_sq, slope, r_sq, extrapolated)?;

        eprintln!("  c_eff={:.0e}: slope={:.6e}, R^2={:.6}, extrap={:.4e} T  [{:.1}s]",
            c_eff, slope, r_sq, extrapolated, start.elapsed().as_secs_f64());
    }

    // 1/c^2 scaling verification
    eprintln!();
    eprintln!("--- 1/c^2 scaling verification ---");
    let omega_test = 1200.0;
    let mut log_c_eff_sq = Vec::new();
    let mut log_delta_b = Vec::new();

    for &c_eff in &c_eff_values {
        let c_eff_sq = c_eff * c_eff;
        let results = gpu.compute(omega_test, c_eff_sq)?;
        let mut db_sq_sum = 0.0f64;
        for r in &results {
            let db_mag = (r.delta_b[0]*r.delta_b[0] + r.delta_b[1]*r.delta_b[1] + r.delta_b[2]*r.delta_b[2]).sqrt();
            db_sq_sum += db_mag * db_mag;
        }
        let rms = (db_sq_sum / results.len() as f64).sqrt();
        log_c_eff_sq.push(c_eff_sq.ln());
        log_delta_b.push(rms.ln());
    }

    let (_, slope, r_sq) = linreg(&log_c_eff_sq, &log_delta_b);
    eprintln!("  log-log slope: {:.4} (expect -1.000)", slope);
    eprintln!("  R^2: {:.6}", r_sq);
    if (slope + 1.0).abs() < 0.01 {
        eprintln!("  PASS: 1/c^2 scaling confirmed");
    } else {
        eprintln!("  WARNING: slope deviates from -1.0");
    }

    let total = start.elapsed();
    eprintln!();
    eprintln!("GPU field residual complete in {:.1}s", total.as_secs_f64());
    eprintln!("CSV output: field_residual_scaling_gpu.csv");
    Ok(())
}

fn main_cpu_field() {
    eprintln!("=== Weber Field Residual Simulation (Method 1) ===");
    eprintln!("f64 arithmetic with c_eff amplification");
    eprintln!();

    let sources = generate_all_sources_f64();
    eprintln!("{} source elements", sources.len());

    let n_theta = 64;
    let n_phi = 32;
    let n_rho = 4;
    let r_major = 0.10;
    let r_minor = 0.03;
    let probes = generate_probes(n_theta, n_phi, n_rho, r_major, r_minor);
    eprintln!("{} probe points ({}×{}×{})", probes.len(), n_theta, n_phi, n_rho);
    eprintln!();

    // c_eff values to sweep
    let c_eff_values: Vec<f64> = vec![1.0e2, 3.0e2, 1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5];

    // omega values to sweep (rad/s)
    let omega_values: Vec<f64> = (0..=12).map(|i| i as f64 * 100.0).collect();

    let physical_c_sq: f64 = 299792458.0_f64 * 299792458.0_f64;

    // Open CSV files
    let mut csv_sweep = std::fs::File::create("field_residual_sweeps.csv")
        .expect("cannot create sweep CSV");
    writeln!(csv_sweep, "omega,c_eff,c_eff_sq,\
        delta_b_rms,delta_b_max,\
        b_maxwell_rms,\
        corr_v2c2,corr_accel,\
        mean_delta_w,\
        f_long_rms,\
        n_probes").unwrap();

    let mut csv_scaling = std::fs::File::create("field_residual_scaling.csv")
        .expect("cannot create scaling CSV");
    writeln!(csv_scaling, "c_eff,c_eff_sq,\
        omega_sq_slope,omega_sq_r_squared,\
        extrapolated_delta_b_at_physical_c").unwrap();

    let mut csv_spatial = std::fs::File::create("field_residual_spatial.csv")
        .expect("cannot create spatial CSV");
    writeln!(csv_spatial, "omega,c_eff,theta,phi,layer,torus_id,\
        delta_b_mag,b_maxwell_mag,\
        delta_w,v_sq_c_sq,accel_term,\
        f_long_mag,d_perp").unwrap();

    let start = std::time::Instant::now();

    // === Validation 1: null check at ω=0 ===
    eprintln!("--- Validation 1: null check (ω=0) ---");
    {
        let c_eff_sq = 1.0e8; // c_eff = 10⁴
        let mut max_delta = 0.0f64;
        for probe in &probes {
            let result = compute_probe(probe, 0.0, c_eff_sq, &sources);
            let db_mag = mag(result.delta_b);
            if db_mag > max_delta { max_delta = db_mag; }
        }
        eprintln!("  max |ΔB| at ω=0: {:.6e}", max_delta);
        if max_delta < 1.0e-14 {
            eprintln!("  PASS: null check (< 1e-14)");
        } else {
            eprintln!("  FAIL: null check — nonzero residual at ω=0!");
        }
        eprintln!();
    }

    // === Validation 2: ω-reversal symmetry ===
    eprintln!("--- Validation 2: ω-reversal symmetry ---");
    {
        let c_eff_sq = 1.0e8;
        let omega_test = 600.0;
        let mut max_asym = 0.0f64;
        for probe in &probes {
            let r_pos = compute_probe(probe, omega_test, c_eff_sq, &sources);
            let r_neg = compute_probe(probe, -omega_test, c_eff_sq, &sources);
            let diff = mag([
                r_pos.delta_b[0] - r_neg.delta_b[0],
                r_pos.delta_b[1] - r_neg.delta_b[1],
                r_pos.delta_b[2] - r_neg.delta_b[2],
            ]);
            let avg = (mag(r_pos.delta_b) + mag(r_neg.delta_b)) / 2.0;
            if avg > 1e-30 {
                let asym = diff / avg;
                if asym > max_asym { max_asym = asym; }
            }
        }
        eprintln!("  max asymmetry |ΔB(+ω) - ΔB(-ω)| / avg: {:.6e}", max_asym);
        if max_asym < 1.0e-10 {
            eprintln!("  PASS: even function of ω");
        } else {
            eprintln!("  WARNING: asymmetry detected");
        }
        eprintln!();
    }

    // === Main sweep: ω × c_eff ===
    eprintln!("--- Main sweep ---");

    // For each c_eff, collect (ω², ΔB_rms) for scaling fit
    for &c_eff in &c_eff_values {
        let c_eff_sq = c_eff * c_eff;
        let mut omega_sq_points = Vec::new();
        let mut delta_b_rms_points = Vec::new();

        for &omega in &omega_values {
            let mut delta_b_mags = Vec::with_capacity(probes.len());
            let mut b_maxwell_mags = Vec::with_capacity(probes.len());
            let mut v2c2_vals = Vec::with_capacity(probes.len());
            let mut accel_vals = Vec::with_capacity(probes.len());
            let mut f_long_sq_sum = 0.0f64;
            let mut delta_w_sum = 0.0f64;

            for (pi, probe) in probes.iter().enumerate() {
                let result = compute_probe(probe, omega, c_eff_sq, &sources);
                let db_mag = mag(result.delta_b);
                let bm_mag = mag(result.b_maxwell);
                let fl_mag = mag(result.f_longitudinal);

                delta_b_mags.push(db_mag);
                b_maxwell_mags.push(bm_mag);
                v2c2_vals.push(result.v_sq_over_c_sq);
                accel_vals.push(result.mean_accel_term.abs());
                f_long_sq_sum += fl_mag * fl_mag;
                delta_w_sum += result.mean_delta_w;

                // Spatial CSV (only at c_eff=10⁴ to keep file size sane)
                if (c_eff - 1.0e4).abs() < 1.0 && omega > 0.1 {
                    // Subsample: every 8th probe for spatial map
                    if pi % 8 == 0 {
                        writeln!(csv_spatial, "{},{},{:.6},{:.6},{},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.6}",
                            omega, c_eff, probe.theta, probe.phi, probe.layer, probe.torus_id,
                            db_mag, bm_mag,
                            result.mean_delta_w, result.v_sq_over_c_sq, result.mean_accel_term,
                            fl_mag, probe.d_perp
                        ).unwrap();
                    }
                }
            }

            let n = probes.len() as f64;
            let delta_b_rms = (delta_b_mags.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
            let delta_b_max = delta_b_mags.iter().cloned().fold(0.0f64, f64::max);
            let b_maxwell_rms = (b_maxwell_mags.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
            let f_long_rms = (f_long_sq_sum / n).sqrt();
            let mean_delta_w = delta_w_sum / n;

            // Correlations (only meaningful when ω > 0)
            let corr_v2c2 = if omega > 0.1 { pearson_r(&v2c2_vals, &delta_b_mags) } else { 0.0 };
            let corr_accel = if omega > 0.1 { pearson_r(&accel_vals, &delta_b_mags) } else { 0.0 };

            writeln!(csv_sweep, "{},{},{},{:.10e},{:.10e},{:.10e},{:.6},{:.6},{:.10e},{:.10e},{}",
                omega, c_eff, c_eff_sq,
                delta_b_rms, delta_b_max,
                b_maxwell_rms,
                corr_v2c2, corr_accel,
                mean_delta_w,
                f_long_rms,
                probes.len()
            ).unwrap();

            if omega > 0.1 {
                omega_sq_points.push(omega * omega);
                delta_b_rms_points.push(delta_b_rms);
            }
        }

        // ω² scaling fit for this c_eff
        let (_, slope, r_sq) = linreg(&omega_sq_points, &delta_b_rms_points);

        // Extrapolate to physical c: ΔB(phys) = ΔB(c_eff) × (c_eff/c)²
        let scale_factor = c_eff_sq / physical_c_sq;
        let extrapolated = if let Some(&last) = delta_b_rms_points.last() {
            last * scale_factor
        } else { 0.0 };

        writeln!(csv_scaling, "{},{},{:.10e},{:.8},{:.10e}",
            c_eff, c_eff_sq, slope, r_sq, extrapolated
        ).unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        eprintln!("  c_eff={:.0e}: ω² slope={:.6e}, R²={:.6}, extrap={:.4e} T  [{:.1}s]",
            c_eff, slope, r_sq, extrapolated, elapsed);
    }
    eprintln!();

    // === 1/c² scaling verification ===
    eprintln!("--- 1/c² scaling verification ---");
    {
        // Use ω=1200 results across c_eff values
        let omega_test = 1200.0;
        let mut log_c_eff_sq = Vec::new();
        let mut log_delta_b = Vec::new();

        for &c_eff in &c_eff_values {
            let c_eff_sq = c_eff * c_eff;
            let mut db_sq_sum = 0.0f64;
            for probe in &probes {
                let result = compute_probe(probe, omega_test, c_eff_sq, &sources);
                db_sq_sum += mag(result.delta_b).powi(2);
            }
            let rms = (db_sq_sum / probes.len() as f64).sqrt();
            log_c_eff_sq.push((c_eff_sq).ln());
            log_delta_b.push(rms.ln());
        }

        let (_, slope, r_sq) = linreg(&log_c_eff_sq, &log_delta_b);
        eprintln!("  log-log slope: {:.4} (expect -1.000 for ΔB ∝ 1/c²)", slope);
        eprintln!("  R²: {:.6}", r_sq);
        if (slope + 1.0).abs() < 0.01 {
            eprintln!("  PASS: 1/c² scaling confirmed");
        } else {
            eprintln!("  WARNING: slope deviates from -1.0");
        }
        eprintln!();
    }

    // === Summary ===
    let total = start.elapsed();
    eprintln!("=== COMPLETE in {:.1}s ===", total.as_secs_f64());
    eprintln!();
    eprintln!("Output files:");
    eprintln!("  field_residual_sweeps.csv   — per (ω, c_eff) aggregate stats");
    eprintln!("  field_residual_scaling.csv  — ω² fit + extrapolation per c_eff");
    eprintln!("  field_residual_spatial.csv  — spatial fingerprint (c_eff=10⁴)");
    eprintln!();

    // Final extrapolation summary
    eprintln!("=== PHYSICAL c EXTRAPOLATION ===");
    eprintln!("  (Read from field_residual_scaling.csv)");
    eprintln!("  Expected: ΔB ~ 10⁻¹⁴ T at 100K RPM with 0.1T magnets");
    eprintln!("  SQUID sensitivity: ~10⁻¹⁵ T/√Hz → detectable in ~100s integration");
}
