//! Ground probes matching the lower footprint used by terrain collision.

use super::{KinematicCapsuleConfig, TerrainDensity, WorldPosition, slope_within};
use crate::query::raycast_density;
use bevy_math::DVec3;

pub(super) fn support_height(
    terrain: &impl TerrainDensity,
    feet: WorldPosition,
    config: KinematicCapsuleConfig,
    drop: f64,
    rise: f64,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    for offset in lower_offsets(config.radius) {
        let origin = WorldPosition(feet.0 + offset + DVec3::Y * rise);
        // A probe starting inside a wall must not find a floor through it.
        if terrain.density(origin) > 0.0 {
            continue;
        }
        let Some(hit) = raycast_density(terrain, origin, DVec3::NEG_Y, rise + drop) else {
            continue;
        };
        if slope_within(hit.normal, config.maximum_slope) {
            let height = hit.position.0.y - offset.y + 1.0e-4;
            best = Some(best.map_or(height, |previous| previous.max(height)));
        }
    }
    best
}

fn lower_offsets(radius: f64) -> [DVec3; 9] {
    let diagonal = std::f64::consts::FRAC_1_SQRT_2;
    [
        DVec3::ZERO,
        DVec3::new(1.0, 1.0, 0.0) * radius,
        DVec3::new(-1.0, 1.0, 0.0) * radius,
        DVec3::new(0.0, 1.0, 1.0) * radius,
        DVec3::new(0.0, 1.0, -1.0) * radius,
        DVec3::new(diagonal, 1.0, diagonal) * radius,
        DVec3::new(diagonal, 1.0, -diagonal) * radius,
        DVec3::new(-diagonal, 1.0, diagonal) * radius,
        DVec3::new(-diagonal, 1.0, -diagonal) * radius,
    ]
}
