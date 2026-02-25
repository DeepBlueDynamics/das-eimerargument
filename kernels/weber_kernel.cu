// weber_kernel.cu — Weber force computation in double-double precision
//
// Port of weber_force_on_particle() from weber.rs
// One thread per particle (N=500), sources tiled through shared memory.
// All arithmetic in dd (31 significant digits).
//
// Thread mapping: blockDim.x threads per block, gridDim.x blocks
// Each thread computes the total Weber force on one particle from ALL sources.

#include "dd_math.cuh"

// Source element: position + current element dl + velocity (in dd3)
// vel is zero for static magnets, nonzero for rotating magnets.
struct GpuSource {
    dd3 pos;
    dd3 dl;
    dd3 vel;
};

// Per-particle output from the Weber kernel
struct WeberOutput {
    dd3 f_total;       // Total force (Lorentz + longitudinal)
    dd3 b_field;       // Weber-corrected B field
    dd  dm_velocity;   // Velocity-dependent mass correction
    dd  bracket_dev;   // Mean bracket deviation (W - 1)
    dd  bracket_coupling;  // Weber bracket contribution to coupling
    dd  elastic_coupling;  // (reserved — computed by bond kernel)
};

// Particle state
struct GpuParticle {
    dd3 pos;
    dd3 vel;
    dd3 accel;     // Previous step acceleration estimate
    dd  mass;
    dd  charge;
};

// Tile size for shared memory source loading
// 32 sources × 64 bytes (padded dd3 for pos) + 48 bytes (dd3 for dl) = ~3.5 KB per tile
// Actually we store full GpuSource in shmem — 2 × dd3 = 96 bytes, padded to 128
#define TILE_SIZE 32

// Shared memory tile for sources
struct SourceTile {
    dd3_padded pos;  // 64 bytes
    dd3_padded dl;   // 64 bytes
    dd3_padded vel;  // 64 bytes
};                   // 192 bytes per source × 32 = 6 KB per tile

extern "C"
__global__ void weber_force_kernel(
    const GpuParticle* __restrict__ particles,
    const GpuSource*   __restrict__ sources,
    WeberOutput*       __restrict__ outputs,
    int n_particles,
    int n_sources
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    // Load particle data (each thread reads its own particle)
    GpuParticle p = particles[tid];

    dd c_sq = DD_C_SQ;
    dd two_c_sq = DD_TWO_C_SQ;
    dd mu0_4pi = DD_MU0_4PI;

    dd3 b_field = dd3_zero();
    dd3 f_long_total = dd3_zero();

    dd dm_static_acc = DD_ZERO;
    dd dm_velocity_raw = DD_ZERO;
    dd bracket_deviation_sum = DD_ZERO;
    dd bracket_coupling_sum = DD_ZERO;
    int n_valid = 0;

    dd v_sq = dd3_mag_sq(p.vel);

    // Shared memory tile
    __shared__ SourceTile tile[TILE_SIZE];

    // Process sources in tiles
    int n_tiles = (n_sources + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < n_tiles; t++) {
        // Cooperative loading: each thread loads one source into shared memory
        int src_idx = t * TILE_SIZE + threadIdx.x;
        if (threadIdx.x < TILE_SIZE && src_idx < n_sources) {
            tile[threadIdx.x].pos = dd3_to_padded(sources[src_idx].pos);
            tile[threadIdx.x].dl  = dd3_to_padded(sources[src_idx].dl);
            tile[threadIdx.x].vel = dd3_to_padded(sources[src_idx].vel);
        }
        __syncthreads();

        // Each thread processes all sources in this tile
        int tile_end = min(TILE_SIZE, n_sources - t * TILE_SIZE);
        for (int s = 0; s < tile_end; s++) {
            dd3 s_pos = dd3_from_padded(tile[s].pos);
            dd3 s_dl  = dd3_from_padded(tile[s].dl);
            dd3 s_vel = dd3_from_padded(tile[s].vel);

            // R = pos_particle - pos_source
            dd3 r_vec = dd3_sub(p.pos, s_pos);
            dd r_sq = dd3_mag_sq(r_vec);

            if (dd_is_zero(r_sq)) continue;

            dd r = dd_sqrt(r_sq);
            dd inv_r = dd_recip(r);
            dd3 r_hat = dd3_scale(r_vec, inv_r);

            // Relative velocity: v_rel = v_particle - v_source
            // For static magnets (vel=0), v_rel = p.vel — backward compatible.
            dd3 v_rel = dd3_sub(p.vel, s_vel);
            dd v_rel_sq = dd3_mag_sq(v_rel);

            // r_dot = v_rel . R_hat (radial component of relative velocity)
            dd r_dot = dd3_dot(v_rel, r_hat);

            // v_perp^2 = |v_rel|^2 - r_dot^2
            dd v_perp_sq = dd_sub(v_rel_sq, dd_mul(r_dot, r_dot));

            // r_ddot = a . R_hat + v_perp^2 / r (includes centripetal)
            dd r_ddot = dd_add(dd3_dot(p.accel, r_hat),
                               dd_mul(v_perp_sq, inv_r));

            // Weber bracket: W = 1 - r_dot^2/(2c^2) + r*r_ddot/c^2
            dd w_bracket = dd_add(
                dd_sub(DD_ONE, dd_div(dd_mul(r_dot, r_dot), two_c_sq)),
                dd_div(dd_mul(r, r_ddot), c_sq)
            );

            dd bracket_dev = dd_sub(w_bracket, DD_ONE);
            bracket_deviation_sum = dd_add(bracket_deviation_sum, bracket_dev);

            // Track bracket coupling (the W-1 terms that affect force during thrust)
            bracket_coupling_sum = dd_add(bracket_coupling_sum, bracket_dev);

            n_valid++;

            dd inv_r_sq = dd_mul(inv_r, inv_r);

            // Biot-Savart with Weber correction:
            //   dB = (mu0/4pi) * (dl x R_hat) / R^2 * W
            dd3 dl_cross_rhat = dd3_cross(s_dl, r_hat);
            dd coeff = dd_mul(dd_mul(mu0_4pi, inv_r_sq), w_bracket);
            dd3 db = dd3_scale(dl_cross_rhat, coeff);
            b_field = dd3_add(b_field, db);

            // Longitudinal force (non-Maxwellian):
            //   F_long = (mu0/4pi) * (q/R^2) * [dl.v_rel - (3/2)(dl.R_hat)(r_dot)] / c^2 * R_hat
            dd dl_dot_v = dd3_dot(s_dl, v_rel);
            dd dl_dot_rhat = dd3_dot(s_dl, r_hat);
            dd f_long_mag = dd_div(
                dd_mul(
                    dd_mul(mu0_4pi, dd_mul(p.charge, inv_r_sq)),
                    dd_sub(dl_dot_v, dd_mul(DD_THREE_HALF, dd_mul(dl_dot_rhat, r_dot)))
                ),
                c_sq
            );
            dd3 f_long = dd3_scale(r_hat, f_long_mag);
            f_long_total = dd3_add(f_long_total, f_long);

            // Mass correction accumulators
            dd dl_mag = dd3_mag(s_dl);
            dd dm_raw = dd_mul(dd_mul(mu0_4pi, dd_mul(p.charge, dl_mag)), inv_r);
            dm_static_acc = dd_add(dm_static_acc, dm_raw);
            dm_velocity_raw = dd_add(dm_velocity_raw,
                dd_mul(dm_raw, dd_mul(r_dot, r_dot)));
        }

        __syncthreads();
    }

    // Velocity correction: single deferred /2c^2
    dd dm_velocity = dd_div(dm_velocity_raw, two_c_sq);

    // Mean bracket deviation
    dd mean_bracket = DD_ZERO;
    dd mean_bracket_coupling = DD_ZERO;
    if (n_valid > 0) {
        dd n_dd = dd_from_double((double)n_valid);
        mean_bracket = dd_div(bracket_deviation_sum, n_dd);
        mean_bracket_coupling = dd_div(bracket_coupling_sum, n_dd);
    }

    // Lorentz force from Weber-corrected B: F = q (v x B)
    dd3 v_cross_b = dd3_cross(p.vel, b_field);
    dd3 f_lorentz = dd3_scale(v_cross_b, p.charge);

    // Total force
    dd3 f_total = dd3_add(f_lorentz, f_long_total);

    // Write output
    outputs[tid].f_total = f_total;
    outputs[tid].b_field = b_field;
    outputs[tid].dm_velocity = dm_velocity;
    outputs[tid].bracket_dev = mean_bracket;
    outputs[tid].bracket_coupling = mean_bracket_coupling;
    outputs[tid].elastic_coupling = DD_ZERO;  // Set by bond kernel
}

// Newton force kernel (no Weber corrections — bracket always 1)
extern "C"
__global__ void newton_force_kernel(
    const GpuParticle* __restrict__ particles,
    const GpuSource*   __restrict__ sources,
    dd3*               __restrict__ forces_out,
    int n_particles,
    int n_sources
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    GpuParticle p = particles[tid];
    dd mu0_4pi = DD_MU0_4PI;

    dd3 b_field = dd3_zero();

    __shared__ SourceTile tile[TILE_SIZE];

    int n_tiles = (n_sources + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < n_tiles; t++) {
        int src_idx = t * TILE_SIZE + threadIdx.x;
        if (threadIdx.x < TILE_SIZE && src_idx < n_sources) {
            tile[threadIdx.x].pos = dd3_to_padded(sources[src_idx].pos);
            tile[threadIdx.x].dl  = dd3_to_padded(sources[src_idx].dl);
            tile[threadIdx.x].vel = dd3_to_padded(sources[src_idx].vel);
        }
        __syncthreads();

        int tile_end = min(TILE_SIZE, n_sources - t * TILE_SIZE);
        for (int s = 0; s < tile_end; s++) {
            dd3 s_pos = dd3_from_padded(tile[s].pos);
            dd3 s_dl  = dd3_from_padded(tile[s].dl);

            dd3 r_vec = dd3_sub(p.pos, s_pos);
            dd r_sq = dd3_mag_sq(r_vec);
            if (dd_is_zero(r_sq)) continue;

            dd r = dd_sqrt(r_sq);
            dd inv_r = dd_recip(r);
            dd3 r_hat = dd3_scale(r_vec, inv_r);
            dd inv_r_sq = dd_mul(inv_r, inv_r);

            dd3 dl_cross_rhat = dd3_cross(s_dl, r_hat);
            dd coeff = dd_mul(mu0_4pi, inv_r_sq);
            dd3 db = dd3_scale(dl_cross_rhat, coeff);
            b_field = dd3_add(b_field, db);
        }

        __syncthreads();
    }

    // F = q (v x B)
    dd3 v_cross_b = dd3_cross(p.vel, b_field);
    dd3 f = dd3_scale(v_cross_b, p.charge);
    forces_out[tid] = f;
}
