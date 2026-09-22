//! Physical quantities shared across crates. Each has exactly one definition,
//! here; every solver, runtime, and tool imports it rather than repeating a
//! literal.

use std::f32::consts::TAU;

use bevy_math::DVec3;

/// Largest editable angular travel endpoint magnitude, in radians.
pub const MAX_PROGRAMMED_TRAVEL_RADIANS: f32 = std::f32::consts::PI;

/// Minimum separation of editable angular travel endpoints, in radians.
pub const MIN_PROGRAMMED_TRAVEL_RADIANS: f32 = std::f32::consts::PI / 36.0;

/// Minimum separation of editable linear travel endpoints, in metres.
pub const MIN_PROGRAMMED_TRAVEL_METERS: f32 = 0.0025;

/// Shortest editable controller dwell, in seconds.
pub const MIN_DWELL_SECONDS: f32 = 0.1;

/// Fixed physics frequency, in ticks per second.
pub const TICK_RATE_HZ: u32 = 60;

/// Fixed physics step, in seconds.
pub const TICK_SECONDS: f64 = 1.0 / TICK_RATE_HZ as f64;

/// [`TICK_SECONDS`] for single-precision consumers such as the GPU runtime.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding the step to f32 is the purpose of this constant"
)]
pub const TICK_SECONDS_F32: f32 = TICK_SECONDS as f32;

/// Standard gravitational acceleration magnitude, in m/s².
pub const STANDARD_GRAVITY_M_S2: f64 = 9.81;

/// [`STANDARD_GRAVITY_M_S2`] for single-precision consumers.
pub const STANDARD_GRAVITY_M_S2_F32: f32 = 9.81;

/// World-space gravitational acceleration: standard gravity along negative Y.
pub const GRAVITY: DVec3 = DVec3::new(0.0, -STANDARD_GRAVITY_M_S2, 0.0);

/// Density of authored machine parts (controllers, engines, transmissions,
/// servos, seats, inputs, and dimension links), in kg/m³. Construction blocks
/// take their density from their material instead.
pub const MACHINE_PART_DENSITY_KG_M3: f32 = 500.0;

/// Maximum acceptable derived bearing-anchor separation, in metres.
pub const ANCHOR_TOLERANCE_METERS: f32 = 0.000_01;

/// Maximum acceptable derived bearing-axis separation, in degrees.
pub const AXIS_TOLERANCE_DEGREES: f32 = 0.001;

// The single-precision step must be exactly what `1.0 / 60.0` evaluates to in
// f32, so rounding the f64 step introduces no second, slightly different clock.
const _: () = assert!(TICK_SECONDS_F32 == 1.0 / 60.0);

/// Converts revolutions per minute to radians per second.
pub const fn rpm_to_rad_s(rpm: f32) -> f32 {
    rpm * TAU / 60.0
}

/// Converts radians per second to revolutions per minute.
pub const fn rad_s_to_rpm(rad_s: f32) -> f32 {
    rad_s * 60.0 / TAU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpm_and_radians_per_second_round_trip() {
        assert!((rpm_to_rad_s(60.0) - TAU).abs() < 1.0e-6);
        assert!((rad_s_to_rpm(rpm_to_rad_s(1_234.5)) - 1_234.5).abs() < 1.0e-3);
    }
}
