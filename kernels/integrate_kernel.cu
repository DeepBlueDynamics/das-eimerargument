// integrate_kernel.cu — Velocity Verlet integration + COM reduction
//
// Velocity Verlet (second-order, time-reversible):
//   1. v(t + dt/2) = v(t) + a(t) * dt/2         [half_step_vel]
//   2. x(t + dt)   = x(t) + v(t + dt/2) * dt     [update_pos]
//   3. Compute forces at new positions → a(t+dt)   [done by force kernels]
//   4. v(t + dt)   = v(t + dt/2) + a(t+dt) * dt/2 [half_step_vel again]
//
// Also: COM computation via parallel reduction, torque application,
//       external force distribution, kinetic energy, angular momentum.

#include "dd_math.cuh"

// Particle state (full — matches GpuParticle in weber_kernel.cu)
struct IntParticle {
    dd3 pos;
    dd3 vel;
    dd3 accel;
    dd  mass;
    dd  charge;
};

// ============================================================================
// Velocity Verlet half-step: v += a * dt/2
// ============================================================================
extern "C"
__global__ void half_step_vel_kernel(
    IntParticle* __restrict__ particles,
    const dd3*   __restrict__ accels,
    dd half_dt,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 dv = dd3_scale(accels[tid], half_dt);
    particles[tid].vel = dd3_add(particles[tid].vel, dv);
}

// ============================================================================
// Position update: x += v * dt
// ============================================================================
extern "C"
__global__ void update_pos_kernel(
    IntParticle* __restrict__ particles,
    dd dt,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 dx = dd3_scale(particles[tid].vel, dt);
    particles[tid].pos = dd3_add(particles[tid].pos, dx);
}

// ============================================================================
// Compute acceleration from total force: a = F / m
// Also stores into particle.accel for next step's bracket estimate.
// ============================================================================
extern "C"
__global__ void compute_accel_kernel(
    IntParticle* __restrict__ particles,
    const dd3*   __restrict__ total_forces,
    dd3*         __restrict__ accels_out,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd inv_m = dd_recip(particles[tid].mass);
    dd3 a = dd3_scale(total_forces[tid], inv_m);
    accels_out[tid] = a;
    particles[tid].accel = a;  // Store for Weber bracket estimate
}

// ============================================================================
// Apply external force (distributed equally) + torque (tangential)
// Adds to total_forces in-place (call AFTER weber/newton + bond kernels)
// ============================================================================
extern "C"
__global__ void apply_external_kernel(
    const IntParticle* __restrict__ particles,
    dd3*               __restrict__ total_forces,
    const dd3*         __restrict__ com_ptr,      // Pre-computed COM (device ptr)
    dd3 thrust,       // External thrust (applied equally to all)
    dd  torque_z,     // Torque about z-axis
    dd  inv_n,        // 1/N
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 com = com_ptr[0];

    // Distributed thrust: F_ext = thrust / N
    dd3 f_ext = dd3_scale(thrust, inv_n);

    // Torque -> tangential force: F_tang = tau / (N * d_perp) * tang_hat
    dd rx = dd_sub(particles[tid].pos.x, com.x);
    dd ry = dd_sub(particles[tid].pos.y, com.y);
    dd d_perp_sq = dd_add(dd_mul(rx, rx), dd_mul(ry, ry));

    dd3 f_torque = dd3_zero();
    if (!dd_is_zero(d_perp_sq)) {
        dd d_perp = dd_sqrt(d_perp_sq);
        dd inv_d = dd_recip(d_perp);
        // Tangential direction: (-y, x, 0) / d_perp
        dd3 tang = dd3_new(dd_mul(dd_neg(ry), inv_d),
                           dd_mul(rx, inv_d),
                           DD_ZERO);
        // Force magnitude: tau / (N * d_perp)
        dd f_mag = dd_mul(dd_mul(torque_z, inv_n), inv_d);
        f_torque = dd3_scale(tang, f_mag);
    }

    // Add to total force
    dd3 f_add = dd3_add(f_ext, f_torque);
    total_forces[tid] = dd3_add(total_forces[tid], f_add);
}

// ============================================================================
// COM reduction: parallel sum of m*pos and m*vel using warp shuffle + block
// ============================================================================

// Warp-level reduction for dd (using shuffle)
__device__ dd warp_reduce_dd(dd val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        dd other;
        other.hi = __shfl_down_sync(0xffffffff, val.hi, offset);
        other.lo = __shfl_down_sync(0xffffffff, val.lo, offset);
        val = dd_add(val, other);
    }
    return val;
}

__device__ dd3 warp_reduce_dd3(dd3 val) {
    val.x = warp_reduce_dd(val.x);
    val.y = warp_reduce_dd(val.y);
    val.z = warp_reduce_dd(val.z);
    return val;
}

// Block-level reduction
// Uses shared memory for inter-warp communication
extern "C"
__global__ void com_reduce_kernel(
    const IntParticle* __restrict__ particles,
    dd3* __restrict__ partial_mp,    // Partial sums of m*pos per block
    dd3* __restrict__ partial_mv,    // Partial sums of m*vel per block
    dd*  __restrict__ partial_mass,  // Partial sums of mass per block
    int n_particles
) {
    // Each thread accumulates for one particle
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    dd3 mp = dd3_zero();
    dd3 mv = dd3_zero();
    dd  m  = DD_ZERO;

    if (tid < n_particles) {
        IntParticle p = particles[tid];
        mp = dd3_scale(p.pos, p.mass);
        mv = dd3_scale(p.vel, p.mass);
        m  = p.mass;
    }

    // Warp-level reduce
    mp = warp_reduce_dd3(mp);
    mv = warp_reduce_dd3(mv);
    m  = warp_reduce_dd(m);

    // Inter-warp reduction via shared memory
    __shared__ dd3 s_mp[32];  // Max 32 warps per block (1024 threads)
    __shared__ dd3 s_mv[32];
    __shared__ dd  s_m[32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) {
        s_mp[warp] = mp;
        s_mv[warp] = mv;
        s_m[warp] = m;
    }
    __syncthreads();

    // First warp reduces the partial sums
    int n_warps = (blockDim.x + 31) / 32;
    if (warp == 0) {
        mp = (lane < n_warps) ? s_mp[lane] : dd3_zero();
        mv = (lane < n_warps) ? s_mv[lane] : dd3_zero();
        m  = (lane < n_warps) ? s_m[lane]  : DD_ZERO;

        mp = warp_reduce_dd3(mp);
        mv = warp_reduce_dd3(mv);
        m  = warp_reduce_dd(m);

        if (lane == 0) {
            partial_mp[blockIdx.x] = mp;
            partial_mv[blockIdx.x] = mv;
            partial_mass[blockIdx.x] = m;
        }
    }
}

// Final reduction of partial block sums (single block)
extern "C"
__global__ void com_finalize_kernel(
    const dd3* __restrict__ partial_mp,
    const dd3* __restrict__ partial_mv,
    const dd*  __restrict__ partial_mass,
    dd3* __restrict__ com_out,
    dd3* __restrict__ com_vel_out,
    int n_blocks
) {
    int tid = threadIdx.x;

    dd3 mp = dd3_zero();
    dd3 mv = dd3_zero();
    dd  m  = DD_ZERO;

    // Each thread loads one block's partial sum
    if (tid < n_blocks) {
        mp = partial_mp[tid];
        mv = partial_mv[tid];
        m  = partial_mass[tid];
    }

    mp = warp_reduce_dd3(mp);
    mv = warp_reduce_dd3(mv);
    m  = warp_reduce_dd(m);

    if (tid == 0) {
        if (!dd_is_zero(m)) {
            dd inv_m = dd_recip(m);
            com_out[0] = dd3_scale(mp, inv_m);
            com_vel_out[0] = dd3_scale(mv, inv_m);
        } else {
            com_out[0] = dd3_zero();
            com_vel_out[0] = dd3_zero();
        }
    }
}

// ============================================================================
// Set rigid spin about z-axis through COM: v_i = omega * (-dy, dx, 0)
// ============================================================================
extern "C"
__global__ void set_spin_kernel(
    IntParticle* __restrict__ particles,
    const dd3*   __restrict__ com_ptr,
    dd omega_z,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 com = com_ptr[0];
    dd dx = dd_sub(particles[tid].pos.x, com.x);
    dd dy = dd_sub(particles[tid].pos.y, com.y);

    particles[tid].vel.x = dd_neg(dd_mul(omega_z, dy));
    particles[tid].vel.y = dd_mul(omega_z, dx);
    particles[tid].vel.z = DD_ZERO;
}

// ============================================================================
// Reimpose rigid rotation preserving COM velocity:
//   v_i = v_com + omega_z * (-dy, dx, 0)
// Eliminates bond-induced spin decay while keeping translational signal intact.
// ============================================================================
extern "C"
__global__ void reimpose_spin_kernel(
    IntParticle* __restrict__ particles,
    const dd3*   __restrict__ com_ptr,
    const dd3*   __restrict__ com_vel_ptr,
    dd omega_z,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 com = com_ptr[0];
    dd3 com_vel = com_vel_ptr[0];
    dd dx = dd_sub(particles[tid].pos.x, com.x);
    dd dy = dd_sub(particles[tid].pos.y, com.y);

    // v_i = v_com + omega × (r_i - com)
    particles[tid].vel.x = dd_add(com_vel.x, dd_neg(dd_mul(omega_z, dy)));
    particles[tid].vel.y = dd_add(com_vel.y, dd_mul(omega_z, dx));
    particles[tid].vel.z = com_vel.z;
}

// ============================================================================
// Per-torus reimpose: counter-rotation support.
//   v_i = v_com + omegas[torus_ids[i]] * (-dy, dx, 0)
// Each torus can spin at a different omega (sign = direction).
// ============================================================================
extern "C"
__global__ void reimpose_spin_multi_kernel(
    IntParticle* __restrict__ particles,
    const dd3*   __restrict__ com_ptr,
    const dd3*   __restrict__ com_vel_ptr,
    const int*   __restrict__ torus_ids,
    const dd*    __restrict__ omegas,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 com = com_ptr[0];
    dd3 com_vel = com_vel_ptr[0];
    dd omega_z = omegas[torus_ids[tid]];

    dd dx = dd_sub(particles[tid].pos.x, com.x);
    dd dy = dd_sub(particles[tid].pos.y, com.y);

    particles[tid].vel.x = dd_add(com_vel.x, dd_neg(dd_mul(omega_z, dy)));
    particles[tid].vel.y = dd_add(com_vel.y, dd_mul(omega_z, dx));
    particles[tid].vel.z = com_vel.z;
}

// ============================================================================
// Per-torus set_spin (no COM velocity preservation):
//   v_i = omegas[torus_ids[i]] * (-dy, dx, 0)
// ============================================================================
extern "C"
__global__ void set_spin_multi_kernel(
    IntParticle* __restrict__ particles,
    const dd3*   __restrict__ com_ptr,
    const int*   __restrict__ torus_ids,
    const dd*    __restrict__ omegas,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    dd3 com = com_ptr[0];
    dd omega_z = omegas[torus_ids[tid]];

    dd dx = dd_sub(particles[tid].pos.x, com.x);
    dd dy = dd_sub(particles[tid].pos.y, com.y);

    particles[tid].vel.x = dd_neg(dd_mul(omega_z, dy));
    particles[tid].vel.y = dd_mul(omega_z, dx);
    particles[tid].vel.z = DD_ZERO;
}

// ============================================================================
// Kinetic energy reduction
// ============================================================================
extern "C"
__global__ void kinetic_energy_kernel(
    const IntParticle* __restrict__ particles,
    dd* __restrict__ partial_ke,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    dd ke = DD_ZERO;
    if (tid < n_particles) {
        dd v_sq = dd3_mag_sq(particles[tid].vel);
        ke = dd_mul(dd_mul(DD_HALF, particles[tid].mass), v_sq);
    }

    ke = warp_reduce_dd(ke);

    __shared__ dd s_ke[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) s_ke[warp] = ke;
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        ke = (lane < n_warps) ? s_ke[lane] : DD_ZERO;
        ke = warp_reduce_dd(ke);
        if (lane == 0) partial_ke[blockIdx.x] = ke;
    }
}

// ============================================================================
// Combine Weber EM forces + bond forces into total_forces (no CPU roundtrip)
// ============================================================================
struct WeberOutput {
    dd3 f_total;
    dd3 b_field;
    dd  dm_velocity;
    dd  bracket_dev;
    dd  bracket_coupling;
    dd  elastic_coupling;
};

extern "C"
__global__ void combine_weber_bond_kernel(
    const WeberOutput* __restrict__ weber_out,
    const dd3*         __restrict__ bond_forces,
    dd3*               __restrict__ total_forces,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    total_forces[tid] = dd3_add(weber_out[tid].f_total, bond_forces[tid]);
}

extern "C"
__global__ void combine_newton_bond_kernel(
    const dd3* __restrict__ em_forces,
    const dd3* __restrict__ bond_forces,
    dd3*       __restrict__ total_forces,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    total_forces[tid] = dd3_add(em_forces[tid], bond_forces[tid]);
}

// In-place add: forces[i] += addend[i]
// Used for Newton combine: d_forces already has EM forces, add bond_forces in-place
extern "C"
__global__ void add_forces_inplace_kernel(
    dd3*       __restrict__ forces,
    const dd3* __restrict__ addend,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    forces[tid] = dd3_add(forces[tid], addend[tid]);
}

// Angular velocity about z: omega = L_z / I_z
extern "C"
__global__ void angular_velocity_kernel(
    const IntParticle* __restrict__ particles,
    const dd3*         __restrict__ com_ptr,
    const dd3*         __restrict__ com_vel_ptr,
    dd* __restrict__ partial_num,   // Σ m(x'vy' - y'vx')
    dd* __restrict__ partial_den,   // Σ m(x'^2 + y'^2)
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    dd num = DD_ZERO;
    dd den = DD_ZERO;

    if (tid < n_particles) {
        dd3 com = com_ptr[0];
        dd3 com_vel = com_vel_ptr[0];
        IntParticle p = particles[tid];
        dd rx = dd_sub(p.pos.x, com.x);
        dd ry = dd_sub(p.pos.y, com.y);
        dd vx = dd_sub(p.vel.x, com_vel.x);
        dd vy = dd_sub(p.vel.y, com_vel.y);

        num = dd_mul(p.mass, dd_sub(dd_mul(rx, vy), dd_mul(ry, vx)));
        den = dd_mul(p.mass, dd_add(dd_mul(rx, rx), dd_mul(ry, ry)));
    }

    num = warp_reduce_dd(num);
    den = warp_reduce_dd(den);

    __shared__ dd s_num[32];
    __shared__ dd s_den[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) { s_num[warp] = num; s_den[warp] = den; }
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) / 32;
        num = (lane < n_warps) ? s_num[lane] : DD_ZERO;
        den = (lane < n_warps) ? s_den[lane] : DD_ZERO;
        num = warp_reduce_dd(num);
        den = warp_reduce_dd(den);
        if (lane == 0) {
            partial_num[blockIdx.x] = num;
            partial_den[blockIdx.x] = den;
        }
    }
}
