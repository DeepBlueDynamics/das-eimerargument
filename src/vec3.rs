use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::ops::{Add, Sub, Mul, Neg};

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: Decimal,
    pub y: Decimal,
    pub z: Decimal,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: Decimal::ZERO,
        y: Decimal::ZERO,
        z: Decimal::ZERO,
    };

    pub fn new(x: Decimal, y: Decimal, z: Decimal) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Vec3) -> Decimal {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn mag_sq(self) -> Decimal {
        self.dot(self)
    }

    /// Magnitude via sqrt. Returns zero for zero vector.
    pub fn mag(self) -> Decimal {
        let s = self.mag_sq();
        if s.is_zero() {
            Decimal::ZERO
        } else {
            decimal_sqrt(s)
        }
    }

    pub fn normalize(self) -> Vec3 {
        let m = self.mag();
        if m.is_zero() {
            Vec3::ZERO
        } else {
            self.scale(Decimal::ONE / m)
        }
    }

    pub fn scale(self, s: Decimal) -> Vec3 {
        Vec3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<Decimal> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: Decimal) -> Vec3 {
        self.scale(rhs)
    }
}

/// Newton-Raphson sqrt for Decimal. Converges in ~10 iterations to 28 digits.
pub fn decimal_sqrt(val: Decimal) -> Decimal {
    if val.is_zero() {
        return Decimal::ZERO;
    }
    if val < Decimal::ZERO {
        panic!("decimal_sqrt of negative value: {}", val);
    }
    // Initial guess from f64
    use rust_decimal::prelude::ToPrimitive;
    let approx = val.to_f64().unwrap_or(1.0).sqrt();
    let mut x = Decimal::from_f64_retain(approx).unwrap_or(Decimal::ONE);
    let two = dec!(2);
    // 20 iterations is overkill but guarantees convergence at 28 digits
    for _ in 0..20 {
        x = (x + val / x) / two;
    }
    x
}

/// Decimal sin via Taylor series. Input in radians.
pub fn decimal_sin(theta: Decimal) -> Decimal {
    // Reduce to [-π, π] range
    let pi = dec!(3.1415926535897932384626433833);
    let two_pi = pi * dec!(2);
    let mut t = theta % two_pi;
    if t > pi {
        t = t - two_pi;
    } else if t < -pi {
        t = t + two_pi;
    }
    // Taylor: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    let mut sum = Decimal::ZERO;
    let mut term = t;
    let t_sq = t * t;
    for n in 0..20 {
        sum += term;
        let denom = Decimal::from((2 * n + 2) * (2 * n + 3));
        term = -term * t_sq / denom;
    }
    sum
}

/// Decimal cos via Taylor series. Input in radians.
pub fn decimal_cos(theta: Decimal) -> Decimal {
    let pi = dec!(3.1415926535897932384626433833);
    let two_pi = pi * dec!(2);
    let mut t = theta % two_pi;
    if t > pi {
        t = t - two_pi;
    } else if t < -pi {
        t = t + two_pi;
    }
    // Taylor: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    let mut sum = Decimal::ZERO;
    let mut term = Decimal::ONE;
    let t_sq = t * t;
    for n in 0..20 {
        sum += term;
        let denom = Decimal::from((2 * n + 1) * (2 * n + 2));
        term = -term * t_sq / denom;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_precision() {
        let four = dec!(4);
        let result = decimal_sqrt(four);
        assert_eq!(result, dec!(2));
    }

    #[test]
    fn test_dot_product() {
        let a = Vec3::new(dec!(1), dec!(2), dec!(3));
        let b = Vec3::new(dec!(4), dec!(5), dec!(6));
        assert_eq!(a.dot(b), dec!(32));
    }

    #[test]
    fn test_cross_product() {
        let x = Vec3::new(dec!(1), dec!(0), dec!(0));
        let y = Vec3::new(dec!(0), dec!(1), dec!(0));
        let z = x.cross(y);
        assert_eq!(z.x, dec!(0));
        assert_eq!(z.y, dec!(0));
        assert_eq!(z.z, dec!(1));
    }

    #[test]
    fn test_sin_cos_zero() {
        let s = decimal_sin(Decimal::ZERO);
        let c = decimal_cos(Decimal::ZERO);
        assert_eq!(s, Decimal::ZERO);
        assert_eq!(c, Decimal::ONE);
    }
}
