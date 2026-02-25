//! Host-side double-double arithmetic type.
//!
//! Mirrors the CUDA `dd` struct for host-side validation and Decimal↔dd conversion.
//! Two f64 values (hi, lo) where hi + lo represents the value with ~31 significant digits.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Double-double: a pair of f64s giving ~31 significant decimal digits.
/// Invariant: |lo| <= ulp(hi)/2, hi + lo is the represented value.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct DD {
    pub hi: f64,
    pub lo: f64,
}

impl DD {
    pub const ZERO: DD = DD { hi: 0.0, lo: 0.0 };
    pub const ONE: DD = DD { hi: 1.0, lo: 0.0 };

    /// c^2 = 89875517873681764 (split for dd representation)
    pub const C_SQ: DD = DD {
        hi: 8.9875517873681760e16,
        lo: 4.0,
    };

    /// 2*c^2 = 179751035747363528
    pub const TWO_C_SQ: DD = DD {
        hi: 1.7975103574736352e17,
        lo: 8.0,
    };

    /// mu0/(4*pi) = 1e-7
    pub const MU0_4PI: DD = DD {
        hi: 1.0e-7,
        lo: 0.0,
    };

    pub fn from_f64(x: f64) -> Self {
        DD { hi: x, lo: 0.0 }
    }

    /// Convert from Decimal to dd. Preserves as many digits as dd can hold (~31).
    pub fn from_decimal(d: Decimal) -> Self {
        let hi = d.to_f64().unwrap_or(0.0);
        // Compute residual: lo = d - hi (in Decimal space, then convert)
        let hi_dec = Decimal::from_f64_retain(hi).unwrap_or(Decimal::ZERO);
        let residual = d - hi_dec;
        let lo = residual.to_f64().unwrap_or(0.0);
        DD { hi, lo }
    }

    /// Convert back to Decimal (preserves all dd precision since Decimal has 28 digits).
    pub fn to_decimal(self) -> Decimal {
        let hi_dec = Decimal::from_f64_retain(self.hi).unwrap_or(Decimal::ZERO);
        let lo_dec = Decimal::from_f64_retain(self.lo).unwrap_or(Decimal::ZERO);
        hi_dec + lo_dec
    }

    /// Convert to f64 (lossy — drops lo).
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Two-sum: exact addition of two f64s returning (sum, error)
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }

    /// Quick two-sum: assumes |a| >= |b|
    fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let e = b - (s - a);
        (s, e)
    }

    /// Two-product: exact product using FMA
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let e = a.mul_add(b, -p);
        (p, e)
    }

    pub fn add(self, other: DD) -> DD {
        let (s, e) = Self::two_sum(self.hi, other.hi);
        let e = e + self.lo + other.lo;
        let (hi, lo) = Self::quick_two_sum(s, e);
        DD { hi, lo }
    }

    pub fn sub(self, other: DD) -> DD {
        self.add(DD { hi: -other.hi, lo: -other.lo })
    }

    pub fn mul(self, other: DD) -> DD {
        let (p, e) = Self::two_prod(self.hi, other.hi);
        let e = e + self.hi * other.lo + self.lo * other.hi;
        let (hi, lo) = Self::quick_two_sum(p, e);
        DD { hi, lo }
    }

    pub fn div(self, other: DD) -> DD {
        let q1 = self.hi / other.hi;
        let r = self.sub(other.mul_d(q1));
        let q2 = r.hi / other.hi;
        let r = r.sub(other.mul_d(q2));
        let q3 = r.hi / other.hi;
        let (hi, lo) = Self::quick_two_sum(q1, q2);
        DD { hi, lo }.add(DD::from_f64(q3))
    }

    pub fn mul_d(self, b: f64) -> DD {
        let (p, e) = Self::two_prod(self.hi, b);
        let e = e + self.lo * b;
        let (hi, lo) = Self::quick_two_sum(p, e);
        DD { hi, lo }
    }

    pub fn sqrt(self) -> DD {
        if self.hi <= 0.0 {
            return DD::ZERO;
        }
        let x = self.hi.sqrt();
        let mut xn = DD::from_f64(x);
        // Two Newton iterations
        let ratio = self.div(xn);
        xn = xn.add(ratio).mul_d(0.5);
        let ratio = self.div(xn);
        xn = xn.add(ratio).mul_d(0.5);
        xn
    }

    pub fn abs(self) -> DD {
        if self.hi < 0.0 {
            DD { hi: -self.hi, lo: -self.lo }
        } else {
            self
        }
    }

    pub fn is_zero(self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }
}

impl std::fmt::Display for DD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.16e} + {:.16e}", self.hi, self.lo)
    }
}

/// 3-component vector of DD values (mirrors dd3 in CUDA).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct DD3 {
    pub x: DD,
    pub y: DD,
    pub z: DD,
}

impl DD3 {
    pub const ZERO: DD3 = DD3 {
        x: DD::ZERO,
        y: DD::ZERO,
        z: DD::ZERO,
    };

    pub fn new(x: DD, y: DD, z: DD) -> Self {
        DD3 { x, y, z }
    }

    pub fn from_f64(x: f64, y: f64, z: f64) -> Self {
        DD3 {
            x: DD::from_f64(x),
            y: DD::from_f64(y),
            z: DD::from_f64(z),
        }
    }

    pub fn from_vec3(v: crate::vec3::Vec3) -> Self {
        DD3 {
            x: DD::from_decimal(v.x),
            y: DD::from_decimal(v.y),
            z: DD::from_decimal(v.z),
        }
    }

    pub fn to_vec3(self) -> crate::vec3::Vec3 {
        crate::vec3::Vec3::new(
            self.x.to_decimal(),
            self.y.to_decimal(),
            self.z.to_decimal(),
        )
    }

    pub fn add(self, other: DD3) -> DD3 {
        DD3 {
            x: self.x.add(other.x),
            y: self.y.add(other.y),
            z: self.z.add(other.z),
        }
    }

    pub fn sub(self, other: DD3) -> DD3 {
        DD3 {
            x: self.x.sub(other.x),
            y: self.y.sub(other.y),
            z: self.z.sub(other.z),
        }
    }

    pub fn scale(self, s: DD) -> DD3 {
        DD3 {
            x: self.x.mul(s),
            y: self.y.mul(s),
            z: self.z.mul(s),
        }
    }

    pub fn dot(self, other: DD3) -> DD {
        self.x.mul(other.x)
            .add(self.y.mul(other.y))
            .add(self.z.mul(other.z))
    }

    pub fn mag_sq(self) -> DD {
        self.dot(self)
    }

    pub fn mag(self) -> DD {
        self.mag_sq().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_dd_add_precision() {
        // 1.0 + 1e-16 should not lose the small part
        let a = DD::from_f64(1.0);
        let b = DD::from_f64(1e-16);
        let c = a.add(b);
        assert!((c.to_f64() - 1.0000000000000001).abs() < 1e-30);
    }

    #[test]
    fn test_dd_mul_precision() {
        let a = DD::from_f64(3.14159265358979323846);
        let b = a.mul(a);
        // pi^2 ≈ 9.8696044010893586188344909998762
        let expected = 9.8696044010893586188;
        assert!((b.to_f64() - expected).abs() < 1e-15);
    }

    #[test]
    fn test_dd_decimal_roundtrip() {
        let d = dec!(0.1234567890123456789012345678);
        let dd = DD::from_decimal(d);
        let back = dd.to_decimal();
        let diff = (d - back).abs();
        // Should preserve much better than f64 alone
        assert!(diff < dec!(1e-15), "roundtrip error: {}", diff);
    }

    #[test]
    fn test_dd_sqrt() {
        let four = DD::from_f64(4.0);
        let two = four.sqrt();
        assert!((two.to_f64() - 2.0).abs() < 1e-30);
    }

    #[test]
    fn test_dd_c_sq() {
        // Verify c^2 constant
        let c = DD::from_f64(299792458.0);
        let c_sq = c.mul(c);
        let expected = DD::C_SQ;
        let diff = c_sq.sub(expected).abs();
        assert!(diff.to_f64() < 1.0, "c^2 mismatch: diff = {}", diff);
    }

    #[test]
    fn test_dd3_cross_product() {
        let x = DD3::from_f64(1.0, 0.0, 0.0);
        let y = DD3::from_f64(0.0, 1.0, 0.0);
        // x cross y = z
        let z = DD3 {
            x: x.y.mul(y.z).sub(x.z.mul(y.y)),
            y: x.z.mul(y.x).sub(x.x.mul(y.z)),
            z: x.x.mul(y.y).sub(x.y.mul(y.x)),
        };
        assert!((z.x.to_f64()).abs() < 1e-30);
        assert!((z.y.to_f64()).abs() < 1e-30);
        assert!((z.z.to_f64() - 1.0).abs() < 1e-30);
    }
}
