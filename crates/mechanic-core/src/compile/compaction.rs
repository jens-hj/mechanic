//! Merging grid-aligned cuboids of one material into the fewest boxes.

use super::model::{ColliderShape, LocalCollider};
use crate::{MaterialProperties, PartId};
use bevy_math::{Quat, Vec3};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(super) struct CompactCuboid {
    pub(super) source_part: PartId,
    pub(super) compound_index: u32,
    pub(super) center: Vec3,
    pub(super) half_extents: Vec3,
    pub(super) material: MaterialProperties,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CompactCuboidKey {
    pub(super) material: [u32; 6],
    pub(super) other_bounds: [i64; 4],
    pub(super) along_minimum: i64,
}

/// Greedily merges touching, axis-aligned boxes within one rigid body and one
/// exact material response. Authored mass properties were already integrated
/// from source geometry, so this changes only the collision decomposition.
pub(super) fn compact_grid_aligned_cuboids(colliders: &mut Vec<LocalCollider>, start: usize) {
    let original = colliders.drain(start..).collect::<Vec<_>>();
    let mut rows_by_source = BTreeMap::<PartId, usize>::new();
    for collider in &original {
        *rows_by_source.entry(collider.source_part).or_default() += 1;
    }
    let mut boxes = Vec::new();
    let mut preserved = Vec::new();
    for collider in original {
        // Multi-row source geometry includes cylinders, pipe bends, and shaped
        // regions. Keep each authored decomposition contiguous and untouched.
        if rows_by_source[&collider.source_part] != 1 {
            preserved.push(collider);
            continue;
        }
        let ColliderShape::Cuboid {
            local_rotation,
            half_extents,
        } = collider.shape
        else {
            preserved.push(collider);
            continue;
        };
        let Some(axis_aligned_half_extents) =
            axis_aligned_half_extents(local_rotation, half_extents)
        else {
            preserved.push(LocalCollider {
                shape: ColliderShape::Cuboid {
                    local_rotation,
                    half_extents,
                },
                ..collider
            });
            continue;
        };
        boxes.push(CompactCuboid {
            source_part: collider.source_part,
            compound_index: collider.compound_index,
            center: collider.local_center,
            half_extents: axis_aligned_half_extents,
            material: collider.material_properties,
        });
    }

    for axis in 0..3 {
        boxes.sort_by_key(|cuboid| compact_cuboid_key(cuboid, axis));
        let mut merged: Vec<CompactCuboid> = Vec::with_capacity(boxes.len());
        for cuboid in boxes.drain(..) {
            let can_merge = merged.last().is_some_and(|previous| {
                same_compaction_lane(previous, &cuboid, axis)
                    && (cuboid_minimum(&cuboid, axis) - cuboid_maximum(previous, axis)).abs()
                        <= 1.0e-6
            });
            if can_merge {
                let previous = merged.last_mut().expect("merge candidate exists");
                let minimum = cuboid_minimum(previous, axis);
                let maximum = cuboid_maximum(&cuboid, axis);
                previous.center[axis] = (minimum + maximum) * 0.5;
                previous.half_extents[axis] = (maximum - minimum) * 0.5;
                previous.source_part = previous.source_part.min(cuboid.source_part);
            } else {
                merged.push(cuboid);
            }
        }
        boxes = merged;
    }

    colliders.extend(boxes.into_iter().map(|cuboid| LocalCollider {
        source_part: cuboid.source_part,
        compound_index: cuboid.compound_index,
        local_center: cuboid.center,
        material_properties: cuboid.material,
        shape: ColliderShape::Cuboid {
            local_rotation: Quat::IDENTITY,
            half_extents: cuboid.half_extents,
        },
    }));
    colliders.extend(preserved);
}

pub(super) fn axis_aligned_half_extents(rotation: Quat, half_extents: Vec3) -> Option<Vec3> {
    let rotated_axes = [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z];
    let mut used_world_axes = 0_u8;
    for axis in rotated_axes {
        let absolute = axis.abs();
        let (world_axis, largest) = [absolute.x, absolute.y, absolute.z]
            .into_iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))?;
        if largest < 1.0 - 1.0e-5 || used_world_axes & (1 << world_axis) != 0 {
            return None;
        }
        used_world_axes |= 1 << world_axis;
    }
    Some(
        rotated_axes[0].abs() * half_extents.x
            + rotated_axes[1].abs() * half_extents.y
            + rotated_axes[2].abs() * half_extents.z,
    )
}

pub(super) fn material_key(material: MaterialProperties) -> [u32; 6] {
    [
        material.density_kg_m3.to_bits(),
        material.static_friction.to_bits(),
        material.dynamic_friction.to_bits(),
        material.restitution.to_bits(),
        material.rolling_resistance.to_bits(),
        material.youngs_modulus_pa.to_bits(),
    ]
}

#[expect(clippy::cast_possible_truncation)]
pub(super) fn quantized_coordinate(value: f32) -> i64 {
    f64::from(value).mul_add(1_000_000.0, 0.0).round() as i64
}

pub(super) fn compact_cuboid_key(cuboid: &CompactCuboid, axis: usize) -> CompactCuboidKey {
    let other_axes = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    CompactCuboidKey {
        material: material_key(cuboid.material),
        other_bounds: [
            quantized_coordinate(cuboid_minimum(cuboid, other_axes[0])),
            quantized_coordinate(cuboid_maximum(cuboid, other_axes[0])),
            quantized_coordinate(cuboid_minimum(cuboid, other_axes[1])),
            quantized_coordinate(cuboid_maximum(cuboid, other_axes[1])),
        ],
        along_minimum: quantized_coordinate(cuboid_minimum(cuboid, axis)),
    }
}

pub(super) fn same_compaction_lane(
    left: &CompactCuboid,
    right: &CompactCuboid,
    axis: usize,
) -> bool {
    left.compound_index == right.compound_index
        && material_key(left.material) == material_key(right.material)
        && (0..3).filter(|&other| other != axis).all(|other| {
            (cuboid_minimum(left, other) - cuboid_minimum(right, other)).abs() <= 1.0e-6
                && (cuboid_maximum(left, other) - cuboid_maximum(right, other)).abs() <= 1.0e-6
        })
}

pub(super) fn cuboid_minimum(cuboid: &CompactCuboid, axis: usize) -> f32 {
    cuboid.center[axis] - cuboid.half_extents[axis]
}

pub(super) fn cuboid_maximum(cuboid: &CompactCuboid, axis: usize) -> f32 {
    cuboid.center[axis] + cuboid.half_extents[axis]
}
