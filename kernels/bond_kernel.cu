// bond_kernel.cu — Bond force computation in double-double precision
//
// Port of TorusBody::bond_forces() from torus.rs
// One thread per bond (~1500 bonds for 500 particles).
// Atomic dd_add to per-particle force accumulators (low contention at ~3 bonds/particle).

#include "dd_math.cuh"

// Bond definition
struct GpuBond {
    int i;           // Particle index i
    int j;           // Particle index j
    dd rest_length;  // Equilibrium distance
};

// Per-particle state — must match GpuParticle layout in weber_kernel.cu
// (pos, vel, accel, mass, charge = 176 bytes per particle)
struct BondParticle {
    dd3 pos;
    dd3 vel;
    dd3 accel;  // unused by bond kernel, but needed for correct stride
    dd  mass;   // unused by bond kernel, but needed for correct stride
    dd  charge; // unused by bond kernel, but needed for correct stride
};

// Bond force kernel: F_ij = [k(|r_ij| - r0) + gamma(v_rel . r_hat)] * r_hat
// Applies ±F to particles i and j via atomic accumulation.
extern "C"
__global__ void bond_force_kernel(
    const BondParticle* __restrict__ particles,
    const GpuBond*           __restrict__ bonds,
    dd3*                     __restrict__ forces,    // Per-particle force accumulator
    dd stiffness,
    dd damping,
    int n_bonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_bonds) return;

    GpuBond bond = bonds[tid];

    // Load particle states
    dd3 pos_i = particles[bond.i].pos;
    dd3 pos_j = particles[bond.j].pos;
    dd3 vel_i = particles[bond.i].vel;
    dd3 vel_j = particles[bond.j].vel;

    // r_vec = pos_j - pos_i
    dd3 r_vec = dd3_sub(pos_j, pos_i);
    dd dist = dd3_mag(r_vec);

    if (dd_is_zero(dist)) return;

    dd inv_dist = dd_recip(dist);
    dd3 r_hat = dd3_scale(r_vec, inv_dist);

    // Stretch = |r_ij| - rest_length
    dd stretch = dd_sub(dist, bond.rest_length);

    // Relative velocity
    dd3 v_rel = dd3_sub(vel_j, vel_i);

    // Radial component of relative velocity
    dd v_radial = dd3_dot(v_rel, r_hat);

    // Force magnitude: k * stretch + gamma * v_radial
    dd f_mag = dd_add(dd_mul(stiffness, stretch), dd_mul(damping, v_radial));

    // Force vector
    dd3 f = dd3_scale(r_hat, f_mag);

    // Apply to particles via atomics:
    //   forces[i] += f
    //   forces[j] -= f
    dd3_atomic_add(&forces[bond.i], f);
    dd3_atomic_add(&forces[bond.j], dd3_neg(f));
}

// Zero out forces array before bond computation
extern "C"
__global__ void zero_forces_kernel(
    dd3* __restrict__ forces,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;
    forces[tid] = dd3_zero();
}
