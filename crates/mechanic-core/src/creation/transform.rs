//! Rigid re-framing of document rows: quarter turns about Y and reflections.

use super::doc::RegionDoc;
use bevy_math::{IVec3, Vec3};

pub(super) const fn rotate_y_i32(position: IVec3, yaw: u8) -> IVec3 {
    match yaw % 4 {
        0 => position,
        1 => IVec3::new(position.z, position.y, -position.x),
        2 => IVec3::new(-position.x, position.y, -position.z),
        _ => IVec3::new(-position.z, position.y, position.x),
    }
}

pub(super) fn rotate_y_vec3(position: Vec3, yaw: u8) -> Vec3 {
    match yaw % 4 {
        0 => position,
        1 => Vec3::new(position.z, position.y, -position.x),
        2 => Vec3::new(-position.x, position.y, -position.z),
        _ => Vec3::new(-position.z, position.y, position.x),
    }
}

pub(super) fn transform_region_doc(region: &mut RegionDoc, yaw: u8, translation: IVec3) {
    let origin = IVec3::from_array(region.origin_steps);
    let size = IVec3::from_array(region.size_cells);
    let maximum = origin + size * crate::POSITION_TICKS_PER_GRID_UNIT;
    let corners = [
        IVec3::new(origin.x, origin.y, origin.z),
        IVec3::new(maximum.x, origin.y, origin.z),
        IVec3::new(origin.x, maximum.y, origin.z),
        IVec3::new(origin.x, origin.y, maximum.z),
        IVec3::new(maximum.x, maximum.y, maximum.z),
    ]
    .map(|corner| {
        rotate_y_i32(corner, yaw) + translation * crate::POSITION_TICKS_PER_HALF_GRID_UNIT
    });
    let minimum = corners
        .iter()
        .copied()
        .reduce(IVec3::min)
        .expect("region has corners");
    let maximum = corners
        .iter()
        .copied()
        .reduce(IVec3::max)
        .expect("region has corners");
    let old_divisions = core::mem::take(&mut region.divisions);
    let old_vertices = core::mem::take(&mut region.vertices);
    let old_counts = old_divisions.each_ref().map(|axis| axis.len() + 2);
    region.origin_steps = minimum.to_array();
    region.size_cells = ((maximum - minimum) / crate::POSITION_TICKS_PER_GRID_UNIT).to_array();
    region.divisions = match yaw % 4 {
        0 => old_divisions,
        1 => [
            old_divisions[2].clone(),
            old_divisions[1].clone(),
            reflected_divisions(&old_divisions[0], size.x),
        ],
        2 => [
            reflected_divisions(&old_divisions[0], size.x),
            old_divisions[1].clone(),
            reflected_divisions(&old_divisions[2], size.z),
        ],
        _ => [
            reflected_divisions(&old_divisions[2], size.z),
            old_divisions[1].clone(),
            old_divisions[0].clone(),
        ],
    };
    region.vertices = old_vertices
        .into_iter()
        .map(|([i, j, k], [x, y, z])| match yaw % 4 {
            0 => ([i, j, k], [x, y, z]),
            1 => (
                [
                    k,
                    j,
                    u16::try_from(old_counts[0] - 1).unwrap_or(u16::MAX) - i,
                ],
                [z, y, -x],
            ),
            2 => (
                [
                    u16::try_from(old_counts[0] - 1).unwrap_or(u16::MAX) - i,
                    j,
                    u16::try_from(old_counts[2] - 1).unwrap_or(u16::MAX) - k,
                ],
                [-x, y, -z],
            ),
            _ => (
                [
                    u16::try_from(old_counts[2] - 1).unwrap_or(u16::MAX) - k,
                    j,
                    i,
                ],
                [-z, y, x],
            ),
        })
        .collect();
}

pub(super) fn reflected_divisions(divisions: &[i32], size: i32) -> Vec<i32> {
    divisions
        .iter()
        .rev()
        .map(|position| size - position)
        .collect()
}
