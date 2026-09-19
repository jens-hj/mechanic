//! SI joint force laws using the same compiled parameters as the GPU backend.

use mechanic_core::{CoordinateDrive, DriveMode, JointKind};

use crate::PhysicsError;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PassiveForce {
    spring: [f64; 4],
    bump: [f64; 4],
}

impl PassiveForce {
    pub fn from_kind(kind: JointKind) -> Self {
        if let JointKind::Suspension(spec) = kind {
            let [spring, bump] = spec.passive_rows();
            Self {
                spring: spring.map(f64::from),
                bump: bump.map(f64::from),
            }
        } else {
            Self::default()
        }
    }

    // Positive force extends the coordinate; negative speed is compression.
    pub fn force(self, position: f64, speed: f64) -> f64 {
        let elastic = self.spring[0] * (self.spring[1] - position).max(0.0);
        let damping = if speed < 0.0 {
            self.spring[2]
        } else {
            self.spring[3]
        };
        let rubber = if self.bump[0] > 0.0 {
            let crush = (self.bump[3] - position - self.bump[2]).clamp(0.0, 0.55 * self.bump[1]);
            self.bump[0] * crush * (1.0 + (12.8 / 3.0) * (crush / self.bump[1]).powi(2))
        } else {
            0.0
        };
        elastic + rubber - damping * speed
    }

    // Global monotone-force slope bound. This permits a fixed positive factor
    // throughout the implicit solve even across damping and bump-contact changes.
    pub fn implicit_diagonal(self, dt: f64) -> f64 {
        dt * self.spring[2].max(self.spring[3])
            + dt * dt * (self.spring[0] + self.bump[0] * (1.0 + 12.8 * 0.55 * 0.55))
    }
}

pub(crate) fn validate_drive(drive: CoordinateDrive) -> Result<(), PhysicsError> {
    let nonnegative = [
        drive.max_speed,
        drive.max_acceleration,
        drive.source_a_max_acceleration,
        drive.source_a_no_load_speed,
        drive.source_b_max_acceleration,
        drive.source_b_no_load_speed,
    ];
    if nonnegative.iter().any(|v| v.is_nan() || *v < 0.0)
        || !drive.target_speed.is_finite()
        || !drive.target_angle.is_finite()
        || drive.min_angle.is_nan()
        || drive.max_angle.is_nan()
        || drive.min_angle == f32::INFINITY
        || drive.max_angle == f32::NEG_INFINITY
        || drive.min_angle > drive.max_angle
    {
        return Err(PhysicsError::InvalidCommand);
    }
    Ok(())
}

pub(crate) fn drive_target(drive: CoordinateDrive, position: f64) -> f64 {
    let maximum = f64::from(drive.max_speed);
    match drive.mode {
        DriveMode::Passive => 0.0,
        DriveMode::Speed => f64::from(drive.target_speed).clamp(-maximum, maximum),
        DriveMode::Angle => {
            let target = f64::from(drive.target_angle)
                .clamp(f64::from(drive.min_angle), f64::from(drive.max_angle));
            let error = target - position;
            if error.abs() < 0.0005 {
                return 0.0;
            }
            let brake = 0.8 * (2.0 * f64::from(drive.max_acceleration) * error.abs()).sqrt();
            error.signum() * (6.0 * error.abs()).min(brake).min(maximum)
        }
    }
}

// Recover SI output torque/force from the compile-time normalization. The new
// runtime applies that budget through coupled H, never as an independent qdd cap.
pub(crate) fn drive_budget(
    drive: CoordinateDrive,
    axis_inertia: f64,
    speed: f64,
    desired: f64,
    dt: f64,
) -> f64 {
    let requested = desired - speed;
    let fade = |no_load: f32| {
        if no_load <= 0.0 {
            0.0
        } else if requested.abs() > 1e-6 && speed * requested > 0.0 {
            (1.0 - speed.abs() / f64::from(no_load)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    let source = |acceleration: f32, no_load: f32| {
        let fraction = fade(no_load);
        if fraction == 0.0 {
            0.0
        } else {
            f64::from(acceleration) * fraction
        }
    };
    dt * axis_inertia
        * (source(
            drive.source_a_max_acceleration,
            drive.source_a_no_load_speed,
        ) + source(
            drive.source_b_max_acceleration,
            drive.source_b_no_load_speed,
        ))
}
