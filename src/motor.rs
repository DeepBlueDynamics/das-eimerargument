use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::config::MotorConfig;

/// PD-controlled motor with torque saturation and viscous friction.
///
/// Replaces the old `set_spin()` cheat: the motor applies realistic torque
/// each timestep, so the torus must spin up through real inertia. Weber's
/// effective mass correction fights the motor differently than Newton — this
/// IS the observable.
#[derive(Clone, Debug)]
pub struct Motor {
    pub omega_target: Decimal,
    pub k_p: Decimal,
    pub max_torque: Decimal,
    pub friction_mu: Decimal,
    pub enabled: bool,
}

impl Motor {
    pub fn from_config(cfg: &MotorConfig) -> Self {
        Motor {
            omega_target: Decimal::from_f64_retain(cfg.omega_target).unwrap_or(dec!(5236)),
            k_p: Decimal::from_f64_retain(cfg.k_p).unwrap_or(dec!(10)),
            max_torque: Decimal::from_f64_retain(cfg.max_torque).unwrap_or(dec!(50)),
            friction_mu: Decimal::from_f64_retain(cfg.friction_mu).unwrap_or(dec!(0.001)),
            enabled: true,
        }
    }

    /// Compute net torque given current angular velocity.
    /// Returns motor drive torque + friction torque.
    pub fn torque(&self, omega_current: Decimal) -> Decimal {
        if !self.enabled {
            // Even when motor is off, friction still acts
            return -self.friction_mu * omega_current;
        }

        let error = self.omega_target - omega_current;
        let tau_motor = (self.k_p * error)
            .min(self.max_torque)
            .max(-self.max_torque);
        let tau_friction = -self.friction_mu * omega_current;
        tau_motor + tau_friction
    }

    /// Create a disabled motor (no drive, minimal friction).
    pub fn disabled() -> Self {
        Motor {
            omega_target: Decimal::ZERO,
            k_p: Decimal::ZERO,
            max_torque: Decimal::ZERO,
            friction_mu: dec!(0.001),
            enabled: false,
        }
    }
}

impl Default for Motor {
    fn default() -> Self {
        Motor {
            omega_target: dec!(5236),
            k_p: dec!(10),
            max_torque: dec!(50),
            friction_mu: dec!(0.001),
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_at_rest_saturates() {
        let motor = Motor::default();
        let tau = motor.torque(Decimal::ZERO);
        // At omega=0, error=5236, k_p*error=52360, clamped to max_torque=50
        assert_eq!(tau, dec!(50));
    }

    #[test]
    fn test_motor_at_target_only_friction() {
        let motor = Motor::default();
        let tau = motor.torque(dec!(5236));
        // error=0, so motor torque=0, only friction = -0.001 * 5236 = -5.236
        let expected = dec!(-5.236);
        assert_eq!(tau, expected);
    }

    #[test]
    fn test_motor_disabled() {
        let motor = Motor::disabled();
        let tau = motor.torque(dec!(1000));
        // Only friction: -0.001 * 1000 = -1.0
        assert_eq!(tau, dec!(-1));
    }

    #[test]
    fn test_motor_braking() {
        let mut motor = Motor::default();
        motor.omega_target = Decimal::ZERO;
        // At omega=5236, error=-5236, k_p*error=-52360, clamped to -50
        let tau = motor.torque(dec!(5236));
        // Motor: -50, friction: -0.001*5236 = -5.236, total = -55.236
        let expected = dec!(-50) + dec!(-5.236);
        assert_eq!(tau, expected);
    }

    #[test]
    fn test_motor_from_config() {
        let cfg = MotorConfig {
            omega_target: 1000.0,
            k_p: 5.0,
            max_torque: 25.0,
            friction_mu: 0.01,
        };
        let motor = Motor::from_config(&cfg);
        assert_eq!(motor.omega_target, Decimal::from_f64_retain(1000.0).unwrap());
        assert!(motor.enabled);
    }
}
