// field_residual_kernel.cu — Field residual probe computation in f64
//
// Port of compute_probe() from field_residual.rs
// Plain f64 (no dd needed — c_eff amplification handles precision).
// One thread per probe point, reads all sources.
//
// This kernel is the first GPU validation target:
//   8192 probes × 3072 sources → verify 1/c² scaling matches CPU.

// ============================================================================
// Types (f64, no dd)
// ============================================================================

struct F64Source {
    double pos[3];
    double dl[3];
};

struct F64ProbePoint {
    double pos[3];
    double theta;
    double phi;
    int    layer;
    double d_perp;
};

struct F64ProbeResult {
    double delta_b[3];
    double b_maxwell[3];
    double f_longitudinal[3];
    double mean_delta_w;
    double v_sq_over_c_sq;
    double mean_accel_term;
};

#define MU0_4PI 1.0e-7

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__ void f64_cross(
    const double* a, const double* b, double* out
) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ double f64_dot(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ __forceinline__ double f64_mag(const double* v) {
    return sqrt(f64_dot(v, v));
}

// ============================================================================
// Main probe kernel
// ============================================================================

extern "C"
__global__ void field_residual_kernel(
    const F64ProbePoint* __restrict__ probes,
    const F64Source*     __restrict__ sources,
    F64ProbeResult*      __restrict__ results,
    double omega,
    double c_eff_sq,
    int n_probes,
    int n_sources
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_probes) return;

    F64ProbePoint probe = probes[tid];

    // Rigid rotation kinematics about z-axis
    double vel[3] = {
        -omega * probe.pos[1],
         omega * probe.pos[0],
         0.0
    };
    double accel[3] = {
        -omega * omega * probe.pos[0],
        -omega * omega * probe.pos[1],
         0.0
    };
    double v_sq = f64_dot(vel, vel);

    double delta_b[3] = {0.0, 0.0, 0.0};
    double b_maxwell[3] = {0.0, 0.0, 0.0};
    double f_long_total[3] = {0.0, 0.0, 0.0};
    double delta_w_sum = 0.0;
    double accel_term_sum = 0.0;
    int n_valid = 0;

    // Process all sources (no tiling needed for f64)
    for (int s = 0; s < n_sources; s++) {
        double r_vec[3] = {
            probe.pos[0] - sources[s].pos[0],
            probe.pos[1] - sources[s].pos[1],
            probe.pos[2] - sources[s].pos[2]
        };
        double r_sq = f64_dot(r_vec, r_vec);
        if (r_sq < 1.0e-30) continue;

        double r = sqrt(r_sq);
        double inv_r = 1.0 / r;
        double r_hat[3] = {r_vec[0]*inv_r, r_vec[1]*inv_r, r_vec[2]*inv_r};
        double inv_r_sq = inv_r * inv_r;

        // Standard Biot-Savart: dB = (mu0/4pi) (dl x R_hat) / R^2
        double dl_cross_rhat[3];
        f64_cross(sources[s].dl, r_hat, dl_cross_rhat);
        double bm_coeff = MU0_4PI * inv_r_sq;
        double db_maxwell[3] = {
            dl_cross_rhat[0] * bm_coeff,
            dl_cross_rhat[1] * bm_coeff,
            dl_cross_rhat[2] * bm_coeff
        };
        b_maxwell[0] += db_maxwell[0];
        b_maxwell[1] += db_maxwell[1];
        b_maxwell[2] += db_maxwell[2];

        // delta_W = v^2/c^2 - (3/2)(v.R_hat)^2/c^2 + R(a.R_hat)/c^2
        double v_dot_rhat = f64_dot(vel, r_hat);
        double a_dot_rhat = f64_dot(accel, r_hat);
        double accel_term = r * a_dot_rhat / c_eff_sq;

        double delta_w = v_sq / c_eff_sq
            - 1.5 * v_dot_rhat * v_dot_rhat / c_eff_sq
            + accel_term;

        // Accumulate: delta_B = Sum(dB_maxwell * delta_w)
        delta_b[0] += db_maxwell[0] * delta_w;
        delta_b[1] += db_maxwell[1] * delta_w;
        delta_b[2] += db_maxwell[2] * delta_w;

        delta_w_sum += delta_w;
        accel_term_sum += accel_term;
        n_valid++;

        // Longitudinal force
        double dl_dot_v = f64_dot(sources[s].dl, vel);
        double dl_dot_rhat = f64_dot(sources[s].dl, r_hat);
        double f_long_mag = MU0_4PI * inv_r_sq
            * (dl_dot_v - 1.5 * dl_dot_rhat * v_dot_rhat) / c_eff_sq;
        f_long_total[0] += r_hat[0] * f_long_mag;
        f_long_total[1] += r_hat[1] * f_long_mag;
        f_long_total[2] += r_hat[2] * f_long_mag;
    }

    double n_f = (double)n_valid;

    F64ProbeResult result;
    result.delta_b[0] = delta_b[0];
    result.delta_b[1] = delta_b[1];
    result.delta_b[2] = delta_b[2];
    result.b_maxwell[0] = b_maxwell[0];
    result.b_maxwell[1] = b_maxwell[1];
    result.b_maxwell[2] = b_maxwell[2];
    result.f_longitudinal[0] = f_long_total[0];
    result.f_longitudinal[1] = f_long_total[1];
    result.f_longitudinal[2] = f_long_total[2];
    result.mean_delta_w = (n_valid > 0) ? delta_w_sum / n_f : 0.0;
    result.v_sq_over_c_sq = v_sq / c_eff_sq;
    result.mean_accel_term = (n_valid > 0) ? accel_term_sum / n_f : 0.0;

    results[tid] = result;
}
