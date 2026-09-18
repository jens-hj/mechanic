//! Cursor rays against simulated bodies, placed bearings, and their discs.

use crate::builder::{BEARING_DEPTH, SurfaceHit, face_geometry_from_ref};
use crate::pose::{
    live_placed_bearing_pose, simulation_bearing_pose, simulation_placed_bearing_pose,
};
use crate::{AppSimulation, PlacedBearing, builder, linear_editor, suspension_render};
use bevy::prelude::{Quat, Vec3};
use mechanic_core::{BearingDimensions, CompiledCreation, ConstructionGraph, FaceOwner, PartId};
use mechanic_gpu::GpuTransform;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SimulationHit {
    pub(crate) part: PartId,
    pub(crate) body_index: u32,
    pub(crate) distance: f32,
    pub(crate) point: Vec3,
    pub(crate) normal: Vec3,
}

pub(crate) fn raycast_simulation(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    origin: Vec3,
    direction: Vec3,
) -> Option<SimulationHit> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    let mut seen_regions = HashSet::new();
    creation
        .part_to_compound
        .iter()
        .filter_map(|&(part, body_index)| {
            if let Some(region) = graph.region_of(part)
                && !seen_regions.insert((body_index, region))
            {
                return None;
            }
            let transform = transforms.get(body_index as usize)?;
            let position = Vec3::from_slice(&transform.position[..3]);
            let rotation = Quat::from_array(transform.rotation);
            let initial = &creation.compounds[body_index as usize];
            // Resolve the ray into authored build space, where the shared exact
            // picker composes local part frames, regions, and evaluated features.
            let build_from_world = initial.root_rotation * rotation.inverse();
            let build_origin = initial.root_translation + build_from_world * (origin - position);
            let hit = builder::raycast_part_in_construction(
                graph,
                part,
                build_origin,
                build_from_world * direction,
            )?;
            Some(SimulationHit {
                part,
                body_index,
                distance: hit.distance,
                point: origin + direction * hit.distance,
                // Curved walls are valid strike targets without a flat mounting
                // face. Orient their impact effect back along the incoming ray.
                normal: builder::try_face_geometry_from_ref(hit.face, Some(graph))
                    .map_or(-direction, |face| build_from_world.inverse() * face.normal),
            })
        })
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

pub(crate) fn hovered_part(hit: Option<SurfaceHit>) -> Option<PartId> {
    match hit?.face.owner {
        FaceOwner::Part(part) => Some(part),
        FaceOwner::Ground => None,
    }
}

pub(crate) fn raycast_placed_bearings(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    let rotational = raycast_placed_bearings_with_pose(bearings, origin, direction, |bearing| {
        Some((
            bearing.anchor,
            face_geometry_from_ref(bearing.source, Some(graph)).normal,
        ))
    });
    rotational
        .into_iter()
        .chain(linear_editor::raycast_scene(
            graph, None, bearings, origin, direction,
        ))
        .chain(suspension_render::raycast_scene(
            graph, None, bearings, origin, direction,
        ))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

pub(crate) fn raycast_live_placed_bearings(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    if !simulation.is_running() {
        return raycast_placed_bearings(graph, bearings, origin, direction);
    }
    let rotational = raycast_placed_bearings_with_pose(bearings, origin, direction, |bearing| {
        live_placed_bearing_pose(graph, simulation, bearing)
    });
    rotational
        .into_iter()
        .chain(linear_editor::raycast_scene(
            graph,
            Some(simulation),
            bearings,
            origin,
            direction,
        ))
        .chain(suspension_render::raycast_scene(
            graph,
            Some(simulation),
            bearings,
            origin,
            direction,
        ))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

pub(crate) fn raycast_placed_bearings_with_pose(
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
    mut pose: impl FnMut(PlacedBearing) -> Option<(Vec3, Vec3)>,
) -> Option<(usize, f32)> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    bearings
        .iter()
        .enumerate()
        .filter_map(|(index, &bearing)| {
            if bearing.kind.is_translational() {
                return None;
            }
            let (anchor, axis) = pose(bearing)?;
            let distance =
                raycast_bearing_annulus(origin, direction, anchor, axis, bearing.dimensions)?;
            Some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
}

/// Bearing pick used for drive wiring. Unlike [`raycast_placed_bearings`] this
/// accepts the whole disc, including the hole and whatever is threaded through
/// it, because a wire is aimed at a joint rather than at its ring.
pub(crate) fn raycast_placed_bearing_discs(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    let rotational =
        raycast_placed_bearing_discs_with_pose(bearings, origin, direction, |bearing| {
            Some((
                bearing.anchor,
                face_geometry_from_ref(bearing.source, Some(graph)).normal,
            ))
        });
    rotational
        .into_iter()
        .chain(linear_editor::raycast_scene(
            graph, None, bearings, origin, direction,
        ))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

pub(crate) fn raycast_live_placed_bearing_discs(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    if !simulation.is_running() {
        return raycast_placed_bearing_discs(graph, bearings, origin, direction);
    }
    let rotational =
        raycast_placed_bearing_discs_with_pose(bearings, origin, direction, |bearing| {
            live_placed_bearing_pose(graph, simulation, bearing)
        });
    rotational
        .into_iter()
        .chain(linear_editor::raycast_scene(
            graph,
            Some(simulation),
            bearings,
            origin,
            direction,
        ))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

pub(crate) fn raycast_placed_bearing_discs_with_pose(
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
    mut pose: impl FnMut(PlacedBearing) -> Option<(Vec3, Vec3)>,
) -> Option<(usize, f32)> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    bearings
        .iter()
        .enumerate()
        .filter_map(|(index, &bearing)| {
            if bearing.kind.is_translational() {
                return None;
            }
            let (anchor, axis) = pose(bearing)?;
            let axis = axis.normalize();
            let slope = direction.dot(axis);
            if slope.abs() < 1.0e-6 {
                return None;
            }
            let distance = (anchor - origin).dot(axis) / slope;
            if distance <= 0.0 {
                return None;
            }
            let radius = (origin + direction * distance - anchor).length();
            (radius <= bearing.dimensions.outer_diameter() * 0.5).then_some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
}

pub(crate) fn raycast_bearing_annulus(
    origin: Vec3,
    direction: Vec3,
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
) -> Option<f32> {
    let axis = axis.normalize();
    let direction = direction.normalize();
    let offset = origin - anchor;
    let axial_origin = offset.dot(axis);
    let axial_direction = direction.dot(axis);
    let radial_origin = offset - axis * axial_origin;
    let radial_direction = direction - axis * axial_direction;
    let half_depth = BEARING_DEPTH * 0.5;
    let outer_radius = dimensions.outer_diameter() * 0.5;
    let inner_radius = dimensions.inner_diameter() * 0.5;
    let mut nearest = f32::INFINITY;

    let radial_a = radial_direction.length_squared();
    if radial_a > f32::EPSILON {
        for radius in [outer_radius, inner_radius] {
            if radius <= 0.0 {
                continue;
            }
            let radial_b = 2.0 * radial_origin.dot(radial_direction);
            let radial_c = radial_origin.length_squared() - radius * radius;
            let discriminant = radial_b.mul_add(radial_b, -4.0 * radial_a * radial_c);
            if discriminant < 0.0 {
                continue;
            }
            let root = discriminant.sqrt();
            for distance in [
                (-radial_b - root) / (2.0 * radial_a),
                (-radial_b + root) / (2.0 * radial_a),
            ] {
                let depth = axial_origin + axial_direction * distance;
                if distance >= 0.0 && depth.abs() <= half_depth + 1.0e-6 {
                    nearest = nearest.min(distance);
                }
            }
        }
    }

    if axial_direction.abs() > f32::EPSILON {
        for depth in [-half_depth, half_depth] {
            let distance = (depth - axial_origin) / axial_direction;
            if distance < 0.0 {
                continue;
            }
            let radial = radial_origin + radial_direction * distance;
            let radius_squared = radial.length_squared();
            if radius_squared <= outer_radius * outer_radius + f32::EPSILON
                && radius_squared >= inner_radius * inner_radius
            {
                nearest = nearest.min(distance);
            }
        }
    }

    nearest.is_finite().then_some(nearest)
}

pub(crate) fn raycast_simulation_bearings(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(BearingDimensions, f32)> {
    let graph_bearings = graph.bearings().filter_map(|(_, bearing)| {
        if bearing.kind.is_translational() {
            return None;
        }
        let (anchor, axis) = simulation_bearing_pose(graph, creation, transforms, bearing)?;
        let distance =
            raycast_bearing_annulus(origin, direction, anchor, axis, bearing.dimensions)?;
        Some((bearing.dimensions, distance))
    });
    let placed = placed_bearings.iter().filter_map(|&bearing| {
        if bearing.kind.is_translational() {
            return None;
        }
        let (anchor, axis) = simulation_placed_bearing_pose(graph, creation, transforms, bearing)?;
        let distance =
            raycast_bearing_annulus(origin, direction, anchor, axis, bearing.dimensions)?;
        Some((bearing.dimensions, distance))
    });
    graph_bearings
        .chain(placed)
        .min_by(|left, right| left.1.total_cmp(&right.1))
}
