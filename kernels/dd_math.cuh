// dd_math.cuh — Double-double arithmetic for CUDA
// Two f64s (hi, lo) give ~31 significant decimal digits.
// RTX 3060: f64 throughput ~363 GFLOPS, dd ops cost 2-6 f64 ops each.
//
// References:
//   Hida, Li, Bailey: "Library for Double-Double and Quad-Double Arithmetic" (2001)
//   Dekker: "A floating-point technique for extending the available precision" (1971)

#ifndef DD_MATH_CUH
#define DD_MATH_CUH

#include <math.h>

// ============================================================================
// Core dd type
// ============================================================================

struct dd {
    double hi;
    double lo;
};

// ============================================================================
// dd3: Vector of 3 dd components (48 bytes natural, 64 bytes padded for shmem)
// ============================================================================

struct dd3 {
    dd x, y, z;
};

// Padded version for shared memory (avoids 2-way bank conflicts on 16-byte dd)
struct dd3_padded {
    dd x, y, z;
    double _pad0, _pad1;  // 16 bytes padding → 64 bytes total
};

// ============================================================================
// Constants
// ============================================================================

__device__ __constant__ dd DD_ZERO = {0.0, 0.0};
__device__ __constant__ dd DD_ONE  = {1.0, 0.0};
__device__ __constant__ dd DD_TWO  = {2.0, 0.0};
__device__ __constant__ dd DD_HALF = {0.5, 0.0};
__device__ __constant__ dd DD_THREE_HALF = {1.5, 0.0};

// c^2 = 89875517873681764 (exact in dd — fits in 17 digits)
// hi = 89875517873681764.0 is exact since it's < 2^53 ≈ 9.0e15... no, 8.99e16 > 2^53
// Need to split: hi = 89875517873681760.0, lo = 4.0
__device__ __constant__ dd DD_C_SQ = {8.9875517873681760e16, 4.0};

// 2*c^2 = 179751035747363528
__device__ __constant__ dd DD_TWO_C_SQ = {1.7975103574736352e17, 8.0};

// mu0/(4*pi) = 1e-7 (exact in double)
__device__ __constant__ dd DD_MU0_4PI = {1.0e-7, 0.0};

// ============================================================================
// Fundamental dd operations (Dekker/Knuth TwoSum, TwoProd)
// ============================================================================

// Quick two-sum: assumes |a| >= |b|
__device__ __forceinline__ dd dd_quick_two_sum(double a, double b) {
    double s = a + b;
    double e = b - (s - a);
    return {s, e};
}

// Two-sum: no magnitude assumption
__device__ __forceinline__ dd dd_two_sum(double a, double b) {
    double s = a + b;
    double v = s - a;
    double e = (a - (s - v)) + (b - v);
    return {s, e};
}

// Two-product using FMA (exact in IEEE 754)
__device__ __forceinline__ dd dd_two_prod(double a, double b) {
    double p = a * b;
    double e = fma(a, b, -p);
    return {p, e};
}

// ============================================================================
// dd arithmetic
// ============================================================================

// Addition: dd + dd
__device__ __forceinline__ dd dd_add(dd a, dd b) {
    dd s = dd_two_sum(a.hi, b.hi);
    double e = a.lo + b.lo;
    s.lo += e;
    // Renormalize
    dd r = dd_quick_two_sum(s.hi, s.lo);
    return r;
}

// Subtraction: dd - dd
__device__ __forceinline__ dd dd_sub(dd a, dd b) {
    dd neg_b = {-b.hi, -b.lo};
    return dd_add(a, neg_b);
}

// Negation
__device__ __forceinline__ dd dd_neg(dd a) {
    return {-a.hi, -a.lo};
}

// Multiplication: dd * dd
__device__ __forceinline__ dd dd_mul(dd a, dd b) {
    dd p = dd_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    // Renormalize
    dd r = dd_quick_two_sum(p.hi, p.lo);
    return r;
}

// Multiplication: dd * double
__device__ __forceinline__ dd dd_mul_d(dd a, double b) {
    dd p = dd_two_prod(a.hi, b);
    p.lo += a.lo * b;
    dd r = dd_quick_two_sum(p.hi, p.lo);
    return r;
}

// FMA: a*b + c (all dd)
__device__ __forceinline__ dd dd_fma(dd a, dd b, dd c) {
    dd p = dd_mul(a, b);
    return dd_add(p, c);
}

// Division: dd / dd (Newton-Raphson refinement)
__device__ __forceinline__ dd dd_div(dd a, dd b) {
    double q1 = a.hi / b.hi;
    // r = a - q1 * b
    dd r = dd_sub(a, dd_mul_d(b, q1));
    double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul_d(b, q2));
    double q3 = r.hi / b.hi;
    // Combine
    dd q = dd_quick_two_sum(q1, q2);
    q = dd_add(q, {q3, 0.0});
    return q;
}

// Square root: Newton-Raphson in dd
__device__ __forceinline__ dd dd_sqrt(dd a) {
    if (a.hi <= 0.0) return DD_ZERO;
    double x = rsqrt(a.hi);  // 1/sqrt(a.hi), fast hardware approx
    double ax = a.hi * x;    // sqrt(a.hi) approx

    // Refine: Newton's method for sqrt
    // x_{n+1} = (x_n + a/x_n) / 2
    dd xn = {ax, 0.0};
    // One iteration in dd
    dd ratio = dd_div(a, xn);
    xn = dd_mul(dd_add(xn, ratio), DD_HALF);
    // Second iteration for full dd precision
    ratio = dd_div(a, xn);
    xn = dd_mul(dd_add(xn, ratio), DD_HALF);
    return xn;
}

// Reciprocal: 1/a
__device__ __forceinline__ dd dd_recip(dd a) {
    return dd_div(DD_ONE, a);
}

// Absolute value
__device__ __forceinline__ dd dd_abs(dd a) {
    if (a.hi < 0.0) return dd_neg(a);
    return a;
}

// Comparison: a > b
__device__ __forceinline__ bool dd_gt(dd a, dd b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
}

// Comparison: a < b
__device__ __forceinline__ bool dd_lt(dd a, dd b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

// Comparison: a == 0
__device__ __forceinline__ bool dd_is_zero(dd a) {
    return a.hi == 0.0 && a.lo == 0.0;
}

// Convert double to dd
__device__ __forceinline__ dd dd_from_double(double x) {
    return {x, 0.0};
}

// Convert dd to double (lossy)
__device__ __forceinline__ double dd_to_double(dd a) {
    return a.hi + a.lo;
}

// ============================================================================
// dd3 vector operations
// ============================================================================

__device__ __forceinline__ dd3 dd3_zero() {
    return {DD_ZERO, DD_ZERO, DD_ZERO};
}

__device__ __forceinline__ dd3 dd3_new(dd x, dd y, dd z) {
    return {x, y, z};
}

__device__ __forceinline__ dd3 dd3_add(dd3 a, dd3 b) {
    return {dd_add(a.x, b.x), dd_add(a.y, b.y), dd_add(a.z, b.z)};
}

__device__ __forceinline__ dd3 dd3_sub(dd3 a, dd3 b) {
    return {dd_sub(a.x, b.x), dd_sub(a.y, b.y), dd_sub(a.z, b.z)};
}

__device__ __forceinline__ dd3 dd3_neg(dd3 a) {
    return {dd_neg(a.x), dd_neg(a.y), dd_neg(a.z)};
}

// Dot product: a . b
__device__ __forceinline__ dd dd3_dot(dd3 a, dd3 b) {
    dd p1 = dd_mul(a.x, b.x);
    dd p2 = dd_mul(a.y, b.y);
    dd p3 = dd_mul(a.z, b.z);
    return dd_add(dd_add(p1, p2), p3);
}

// Cross product: a x b
__device__ __forceinline__ dd3 dd3_cross(dd3 a, dd3 b) {
    return {
        dd_sub(dd_mul(a.y, b.z), dd_mul(a.z, b.y)),
        dd_sub(dd_mul(a.z, b.x), dd_mul(a.x, b.z)),
        dd_sub(dd_mul(a.x, b.y), dd_mul(a.y, b.x))
    };
}

// Scale: v * s (dd3 * dd)
__device__ __forceinline__ dd3 dd3_scale(dd3 v, dd s) {
    return {dd_mul(v.x, s), dd_mul(v.y, s), dd_mul(v.z, s)};
}

// Scale by double
__device__ __forceinline__ dd3 dd3_scale_d(dd3 v, double s) {
    return {dd_mul_d(v.x, s), dd_mul_d(v.y, s), dd_mul_d(v.z, s)};
}

// Magnitude squared: |v|^2
__device__ __forceinline__ dd dd3_mag_sq(dd3 v) {
    return dd3_dot(v, v);
}

// Magnitude: |v|
__device__ __forceinline__ dd dd3_mag(dd3 v) {
    return dd_sqrt(dd3_mag_sq(v));
}

// Normalize: v / |v| (returns zero vector if |v| == 0)
__device__ __forceinline__ dd3 dd3_normalize(dd3 v) {
    dd m = dd3_mag(v);
    if (dd_is_zero(m)) return dd3_zero();
    dd inv_m = dd_recip(m);
    return dd3_scale(v, inv_m);
}

// ============================================================================
// Shared memory helpers: convert between dd3 and dd3_padded
// ============================================================================

__device__ __forceinline__ dd3_padded dd3_to_padded(dd3 v) {
    dd3_padded p;
    p.x = v.x;
    p.y = v.y;
    p.z = v.z;
    p._pad0 = 0.0;
    p._pad1 = 0.0;
    return p;
}

__device__ __forceinline__ dd3 dd3_from_padded(dd3_padded p) {
    return {p.x, p.y, p.z};
}

// ============================================================================
// Atomic dd_add (CAS loop on hi, then lo — approximate but sufficient for
// low-contention accumulation like bond forces at ~3 bonds/particle)
// ============================================================================

__device__ __forceinline__ void dd_atomic_add(dd* addr, dd val) {
    // Atomically add to hi
    unsigned long long int* addr_hi = (unsigned long long int*)&(addr->hi);
    unsigned long long int old_hi = *addr_hi;
    unsigned long long int assumed;
    do {
        assumed = old_hi;
        double old_val = __longlong_as_double(assumed);
        double new_val = old_val + val.hi;
        old_hi = atomicCAS(addr_hi, assumed, __double_as_longlong(new_val));
    } while (assumed != old_hi);

    // Atomically add to lo (includes residual from hi addition)
    unsigned long long int* addr_lo = (unsigned long long int*)&(addr->lo);
    unsigned long long int old_lo = *addr_lo;
    do {
        assumed = old_lo;
        double old_val = __longlong_as_double(assumed);
        double new_val = old_val + val.lo;
        old_lo = atomicCAS(addr_lo, assumed, __double_as_longlong(new_val));
    } while (assumed != old_lo);
}

__device__ __forceinline__ void dd3_atomic_add(dd3* addr, dd3 val) {
    dd_atomic_add(&(addr->x), val.x);
    dd_atomic_add(&(addr->y), val.y);
    dd_atomic_add(&(addr->z), val.z);
}

#endif // DD_MATH_CUH
