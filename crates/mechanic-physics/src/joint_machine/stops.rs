//! Joint limits participate in timed motion and instantaneous coupled impacts.

use super::{CompiledCreation, CoordinateDrive, MachineState, bounds};

pub(super) const POSITION_TOLERANCE: f64 = 1e-10;

#[derive(Clone, Copy, Debug)]
pub(super) struct StopHit {
    pub coordinate: usize,
    pub fraction: f64,
}

pub(super) fn closing(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &MachineState,
    tolerance: f64,
) -> bool {
    creation
        .dynamics
        .coordinate_velocities
        .iter()
        .enumerate()
        .any(|(coordinate, &row)| {
            let [lower, upper] = bounds(creation, drives, coordinate);
            let q = state.coordinates[coordinate];
            (lower.is_finite()
                && q <= lower + POSITION_TOLERANCE
                && state.velocities[row] < -tolerance)
                || (upper.is_finite()
                    && q >= upper - POSITION_TOLERANCE
                    && state.velocities[row] > tolerance)
        })
}

// The state contains midpoint drift rates. A crossing requests reintegration;
// an endpoint already within the fixed position tolerance can activate there.
pub(super) fn first_crossing(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    state: &MachineState,
    dt: f64,
) -> Option<StopHit> {
    let mut earliest: Option<StopHit> = None;
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        let q = state.coordinates[coordinate];
        let displacement = state.velocities[row] * dt;
        let [lower, upper] = bounds(creation, drives, coordinate);
        let target = if displacement < 0.0 && q + displacement < lower - POSITION_TOLERANCE {
            lower
        } else if displacement > 0.0 && q + displacement > upper + POSITION_TOLERANCE {
            upper
        } else {
            continue;
        };
        let fraction = ((target - q) / displacement).clamp(0.0, 1.0);
        if earliest.is_none_or(|hit| fraction < hit.fraction) {
            earliest = Some(StopHit {
                coordinate,
                fraction,
            });
        }
    }
    earliest
}

// A released limit can turn back within the trial. Its zero-velocity estimate
// only proposes a shorter interval; the caller reintegrates and validates it.
pub(super) fn reversal(
    creation: &CompiledCreation,
    drives: &[CoordinateDrive],
    initial: &MachineState,
    final_velocity: &[f64],
    tolerance: f64,
) -> Option<super::events::VelocityReversal> {
    let mut earliest: Option<super::events::VelocityReversal> = None;
    for (coordinate, &row) in creation.dynamics.coordinate_velocities.iter().enumerate() {
        let q = initial.coordinates[coordinate];
        let [lower, upper] = bounds(creation, drives, coordinate);
        for (sign, active) in [
            (1.0, q <= lower + POSITION_TOLERANCE),
            (-1.0, q >= upper - POSITION_TOLERANCE),
        ] {
            let before = sign * initial.velocities[row];
            let after = sign * final_velocity[row];
            if active && before > tolerance && after < -tolerance {
                let fraction = before / (before - after);
                if earliest.as_ref().is_none_or(|old| fraction < old.fraction) {
                    let mut jacobian = vec![0.0; final_velocity.len()];
                    jacobian[row] = sign;
                    earliest = Some(super::events::VelocityReversal {
                        point: None,
                        fraction,
                        row: jacobian,
                    });
                }
            }
        }
    }
    earliest
}
