//! Block spans, position-tick snapping, and cardinal directions.

use super::raycast::raycast_placement_plane_point;
use super::{
    BLOCK_SIZE_METERS, BlockVolume, GRID_UNIT_METERS, IVec3, PlacementBounds, PlacementError,
    PlacementGrid, PlacementPlane, Result, Vec, Vec3,
};
use mechanic_core::{
    CuboidSpec, FaceKind, GridRotation, POSITION_TICK_METERS, POSITION_TICKS_PER_HALF_GRID_UNIT,
};

/// One plane of [`block_box_specs`], addressed by the endpoint the pointer is
/// aiming at rather than a span.
///
/// Drags themselves all work in spans now. This survives because a good many
/// tests are written against an endpoint, and routing it through the box keeps
/// the two from ever describing different geometry.
#[cfg(test)]
pub(crate) fn block_sheet_specs(
    start: CuboidSpec,
    endpoint_units: IVec3,
    plane: PlacementPlane,
) -> Result<Vec<CuboidSpec>, PlacementError> {
    let start_units = start.pose.translation_half_units();
    let block_units = i32::from(start.dimensions[0].units()) * 2;
    let mut span = IVec3::ZERO;
    for axis in plane.tangent_axes() {
        span[axis] = rounded_div(endpoint_units[axis] - start_units[axis], block_units);
    }
    block_box_specs(start, span)
}

/// Every block in the solid cuboid spanning `span` blocks from `start`.
///
/// `span` counts blocks *beyond* the start block along each axis and may be
/// negative, so a zero span is the single starting block. This is what a drag
/// produces once Rotate has moved it into a third axis.
pub(crate) fn block_box_specs(
    start: CuboidSpec,
    span: IVec3,
) -> Result<Vec<CuboidSpec>, PlacementError> {
    Ok(BlockVolume::new(start, span)?.specs().collect())
}

/// The world-space bounds of the box a drag spans, without building its blocks.
pub(crate) fn block_box_bounds(start: CuboidSpec, span: IVec3) -> (Vec3, Vec3) {
    let block = f32::from(start.dimensions[0].units()) * GRID_UNIT_METERS;
    let centre = start.pose.translation();
    let reach = span.as_vec3() * block;
    let low = centre + reach.min(Vec3::ZERO) - Vec3::splat(block * 0.5);
    let high = centre + reach.max(Vec3::ZERO) + Vec3::splat(block * 0.5);
    (low, high)
}

/// Extends a drag's span by the pointer's motion within the active plane.
///
/// Only the plane's own two axes move; the third keeps whatever it already had.
/// That is what lets Rotate move the drag into a new plane without discarding the
/// extent already dragged, turning a rectangle into a box.
pub(crate) fn block_span_from_rays(
    start: CuboidSpec,
    plane: PlacementPlane,
    anchor_span: IVec3,
    press_origin: Vec3,
    press_direction: Vec3,
    current_origin: Vec3,
    current_direction: Vec3,
) -> Option<IVec3> {
    let press = raycast_placement_plane_point(press_origin, press_direction, start, plane)?;
    let current = raycast_placement_plane_point(current_origin, current_direction, start, plane)?;
    let steps = ((current - press) / BLOCK_SIZE_METERS).round().as_ivec3();
    let mut span = anchor_span;
    for axis in plane.tangent_axes() {
        span[axis] = anchor_span[axis].saturating_add(steps[axis]);
    }
    Some(span)
}

pub(super) fn snap_world_to_position_ticks(position: Vec3) -> IVec3 {
    (position / POSITION_TICK_METERS).round().as_ivec3()
}

#[expect(clippy::cast_possible_truncation)]
pub(super) fn rounded_position_tick(meters: f32) -> i32 {
    debug_assert!(meters.is_finite());
    (meters / POSITION_TICK_METERS).round() as i32
}

#[expect(clippy::cast_possible_truncation)]
pub(super) fn rounded_position_tick_f64(meters: f64) -> i32 {
    debug_assert!(meters.is_finite());
    (meters / f64::from(POSITION_TICK_METERS)).round() as i32
}

pub(super) fn snap_global_center_ticks(
    raw_local_ticks: IVec3,
    world_dimensions: [u8; 3],
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> IVec3 {
    let origin_ticks = placement_origin_ticks(bounds);
    let mut global = raw_local_ticks + origin_ticks;
    let step = grid.step_ticks();
    for axis in 0..3 {
        let plane_phase = if axis == 1 {
            0
        } else {
            POSITION_TICKS_PER_HALF_GRID_UNIT.rem_euclid(step)
        };
        let center_phase = (plane_phase
            + i32::from(world_dimensions[axis]) * POSITION_TICKS_PER_HALF_GRID_UNIT)
            .rem_euclid(step);
        global[axis] = quantise_with_phase(global[axis], step, center_phase);
    }
    global - origin_ticks
}

pub(super) fn placement_origin_ticks(bounds: PlacementBounds) -> IVec3 {
    match bounds {
        PlacementBounds::Garage
        | PlacementBounds::GarageBuild
        | PlacementBounds::GarageBuildFrame { .. } => IVec3::ZERO,
        PlacementBounds::World { origin } => IVec3::new(
            rounded_position_tick_f64(origin.x),
            0,
            rounded_position_tick_f64(origin.y),
        ),
    }
}

pub(super) fn quantise_with_phase(value: i32, step: i32, phase: i32) -> i32 {
    let shifted = value - phase;
    let quotient = shifted.div_euclid(step);
    let remainder = shifted.rem_euclid(step);
    phase
        + if remainder.saturating_mul(2) > step
            || (remainder.saturating_mul(2) == step && (shifted >= 0 || phase != 0))
        {
            quotient.saturating_add(1) * step
        } else {
            quotient * step
        }
}

#[cfg(test)]
pub(super) fn rounded_div(value: i32, divisor: i32) -> i32 {
    let half = divisor / 2;
    if value >= 0 {
        value.saturating_add(half) / divisor
    } else {
        value.saturating_sub(half) / divisor
    }
}

pub(super) fn rotation_y_to_normal(normal: Vec3) -> GridRotation {
    let (axis, sign) = cardinal_axis(normal);
    match (axis, sign) {
        (0, 1) => GridRotation::new(0, 0, 3),
        (0, _) => GridRotation::new(0, 0, 1),
        (1, 1) => GridRotation::default(),
        (1, _) => GridRotation::new(2, 0, 0),
        (2, 1) => GridRotation::new(1, 0, 0),
        _ => GridRotation::new(3, 0, 0),
    }
}

pub(super) fn cardinal_axis(normal: Vec3) -> (usize, i32) {
    let absolute = normal.abs();
    let axis = if absolute.x > absolute.y && absolute.x > absolute.z {
        0
    } else if absolute.y > absolute.z {
        1
    } else {
        2
    };
    (axis, if normal[axis] >= 0.0 { 1 } else { -1 })
}

pub(super) fn cardinal_direction(direction: Vec3) -> Vec3 {
    let (axis, sign) = cardinal_axis(direction);
    let mut cardinal = Vec3::ZERO;
    cardinal[axis] = if sign > 0 { 1.0 } else { -1.0 };
    cardinal
}

pub(super) fn face_for_normal(normal: Vec3) -> FaceKind {
    let (axis, sign) = cardinal_axis(normal);
    match (axis, sign) {
        (0, 1) => FaceKind::PositiveX,
        (0, _) => FaceKind::NegativeX,
        (1, 1) => FaceKind::PositiveY,
        (1, _) => FaceKind::NegativeY,
        (2, 1) => FaceKind::PositiveZ,
        _ => FaceKind::NegativeZ,
    }
}

pub(super) fn snap_cardinal(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.round(), vector.y.round(), vector.z.round())
}
