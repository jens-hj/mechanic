//! Staging parts, block batches, and block volumes into a candidate graph with their welds.

use super::bearings::BearingAttachment;
use super::bounds::{
    bounds_overlap_interior, cuboid_world_bounds, validate_block_batch_in_bounds,
    validate_block_volume_in_bounds, validate_part_in_bounds, volume_candidate_range,
};
use super::snap::PlacementSnapIndex;
use super::welds::{rigid_body_parts, touching_face_pair};
use super::{
    BlockVolume, BlockVolumePlacement, CONTACT_EPSILON, CylinderPlacementCandidate, IVec3,
    PlacementBounds, PlacementCandidate, PlacementError, PlacementSupport, Result, SurfaceHit,
    ToString, UVec3, Vec, Vec3,
};
use mechanic_core::{
    BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, ConstructionGraph, ControllerSpec,
    CuboidSpec, DimensionLinkId, DimensionLinkSpec, EngineKind, EngineSpec, FaceKind, FaceOwner,
    FaceRef, GridRotation, InputSpec, POSITION_TICKS_PER_GRID_UNIT, PartId, PartSpec,
    RigidLinkSpec, SeatSpec, ServoSpec, TransmissionSpec, WeldSpec,
};
use std::collections::HashSet;

#[cfg(test)]
pub(crate) fn stage_cuboid(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
) -> Result<ConstructionGraph, PlacementError> {
    stage_block_batch(graph, candidate, &[candidate.spec])
}

#[cfg(test)]
pub(crate) fn stage_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, None, None, PlacementBounds::Garage)
}

pub(crate) fn stage_controller_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Controller,
        bounds,
    )
}

/// Stages one inert engine, auto-welding it like an ordinary block.
#[cfg(test)]
pub(crate) fn stage_engine_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    source: FaceOwner,
    kind: EngineKind,
) -> Result<ConstructionGraph, PlacementError> {
    stage_engine_from_source_in_bounds(graph, start, source, kind, PlacementBounds::Garage)
}

#[cfg(test)]
pub(crate) fn stage_engine_from_source_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    source: FaceOwner,
    kind: EngineKind,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        Some(source),
        FixedPartSpawn::Engine(kind),
        bounds,
    )
}

pub(crate) fn stage_engine_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    kind: EngineKind,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Engine(kind),
        bounds,
    )
}

/// Builds the only valid transmission candidate for a hovered output face.
#[cfg(test)]
pub(crate) fn transmission_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<(PartId, PlacementCandidate), PlacementError> {
    transmission_candidate_from_hit_in_bounds(graph, hit, PlacementBounds::Garage)
}

pub(crate) fn transmission_candidate_from_hit_in_bounds(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    bounds: PlacementBounds,
) -> Result<(PartId, PlacementCandidate), PlacementError> {
    let FaceOwner::Part(parent) = hit.face.owner else {
        return Err(PlacementError::TransmissionOutputOnly);
    };
    if hit.face.face != FaceKind::PositiveZ {
        return Err(PlacementError::TransmissionOutputOnly);
    }
    let spec = graph
        .next_transmission_spec(parent)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    validate_part_in_bounds(graph, PartSpec::Transmission(spec), bounds)?;
    Ok((
        parent,
        PlacementCandidate {
            spec: spec.cuboid(),
            attached_face: FaceKind::NegativeZ,
            anchor: Some(hit.point),
            support: PlacementSupport::Surface(hit.face.owner),
        },
    ))
}

/// Stages one graph-owned transmission, including its required weld and parent relation.
pub(crate) fn stage_transmission(
    graph: &ConstructionGraph,
    parent: PartId,
    candidate: PlacementCandidate,
) -> Result<ConstructionGraph, PlacementError> {
    let mut staged = graph.begin_edit();
    staged
        .apply(BuildCommand::AttachTransmission {
            parent,
            spec: TransmissionSpec::new(candidate.spec.pose),
        })
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

pub(crate) fn stage_servo_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Servo,
        bounds,
    )
}

pub(crate) fn stage_seat_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Seat,
        bounds,
    )
}

pub(crate) fn stage_input_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::Input,
        bounds,
    )
}

pub(crate) fn stage_dimension_link_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    id: DimensionLinkId,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        start.support.auto_weld_source(),
        FixedPartSpawn::DimensionLink(id),
        bounds,
    )
}

#[cfg(test)]
pub(crate) fn stage_block_batch_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_block_batch_from_source_in_bounds(graph, start, specs, source, PlacementBounds::Garage)
}

#[cfg(test)]
pub(crate) fn stage_block_batch_from_source_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceOwner,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, None, Some(source), bounds)
}

#[cfg(test)]
pub(crate) fn stage_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        start,
        specs,
        None,
        start.support.auto_weld_source(),
        bounds,
    )
}

#[cfg(test)]
pub(crate) fn stage_bearing_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_block_batch_in_bounds(
        graph,
        start,
        specs,
        source,
        anchor,
        dimensions,
        rigid_targets,
        PlacementBounds::Garage,
    )
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn stage_bearing_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        start,
        specs,
        Some(BearingAttachment::rotational(
            graph,
            source,
            anchor,
            dimensions,
            rigid_targets,
        )),
        None,
        bounds,
    )
}

pub(crate) fn validate_cylinder_candidate_in_bounds(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    if candidate.support != PlacementSupport::Free && candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    validate_part_in_bounds(graph, PartSpec::Cylinder(candidate.spec), bounds)
}

#[cfg(test)]
pub(crate) fn stage_cylinder_from_source(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        None,
        Some(source),
        PlacementBounds::Garage,
    )
}

pub(crate) fn stage_bearing_cylinder_in_bounds(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        Some(BearingAttachment::rotational(
            graph,
            source,
            anchor,
            dimensions,
            rigid_targets,
        )),
        None,
        bounds,
    )
}

pub(super) fn stage_connected_cylinder(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_cylinder_candidate_in_bounds(graph, candidate, bounds)?;
    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.begin_edit();
    let BuildOutcome::Spawned(part) = staged
        .apply(BuildCommand::SpawnCylinder(candidate.spec))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!()
    };
    let mut connections = Vec::new();
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(part, candidate.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions)
            .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: part,
            })
        }));
    } else {
        if weld_scope.is_none()
            && bounds == PlacementBounds::Garage
            && let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
        {
            connections.push(BuildCommand::Weld(WeldSpec { first, second }));
        }
        let mut tested_owners = HashSet::new();
        for other in existing_parts {
            if weld_scope
                .as_ref()
                .is_some_and(|members| !members.contains(&other))
                || !tested_owners.insert(connection_geometry_owner(&staged, other))
            {
                continue;
            }
            if let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
        }
    }
    staged
        .apply_batch(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

pub(super) fn stage_connected_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        specs,
        bearing,
        auto_weld_source,
        FixedPartSpawn::Cuboid,
        bounds,
    )
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn stage_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    bearing: Option<(FaceRef, Vec3, BearingDimensions, &[PartId])>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
    publication_generation: u64,
) -> Result<BlockVolumePlacement, PlacementError> {
    stage_connected_block_volume_in_bounds(
        graph,
        index,
        start,
        volume,
        bearing.map(|(source, anchor, dimensions, targets)| {
            BearingAttachment::rotational(graph, source, anchor, dimensions, targets)
        }),
        auto_weld_source,
        bounds,
        publication_generation,
    )
}

/// Validates and commits a regular block volume without constructing any
/// all-pairs candidate sets.
#[expect(clippy::too_many_arguments)]
pub(super) fn stage_connected_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    bounds: PlacementBounds,
    publication_generation: u64,
) -> Result<BlockVolumePlacement, PlacementError> {
    validate_block_volume_in_bounds(graph, index, start, volume, bounds)?;

    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.begin_edit();
    staged.reserve_parts_and_welds(volume.count(), volume.count().saturating_mul(6));
    let new_parts = staged.spawn_cuboids(volume.specs());

    let mut connections = Vec::with_capacity(volume.count().saturating_mul(3));
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        let first = new_parts[0];
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(first, start.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions)
            .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: first,
            })
        }));
    }

    append_internal_volume_welds(volume, &new_parts, &mut connections);
    if bearing.is_none() {
        if weld_scope.is_none() && bounds == PlacementBounds::Garage {
            append_ground_volume_welds(volume, &new_parts, &mut connections);
        }
        append_existing_volume_welds(
            graph,
            &staged,
            index,
            volume,
            &new_parts,
            weld_scope.as_ref(),
            &mut connections,
        );
    }
    let weld_count = connections
        .iter()
        .filter(|command| matches!(command, BuildCommand::Weld(_)))
        .count();
    staged
        .apply_batch_discarding_outcomes(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(BlockVolumePlacement {
        graph: staged.finish(),
        new_parts,
        weld_count,
        bounds: volume.bounds(),
        publication_generation,
    })
}

pub(super) fn append_internal_volume_welds(
    volume: BlockVolume,
    parts: &[PartId],
    connections: &mut Vec<BuildCommand>,
) {
    let counts = volume.dimensions();
    for x in 0..counts.x {
        for y in 0..counts.y {
            for z in 0..counts.z {
                let cell = UVec3::new(x, y, z);
                let first = volume.part_at_physical(parts, cell);
                for (axis, face) in [
                    (0, FaceKind::PositiveX),
                    (1, FaceKind::PositiveY),
                    (2, FaceKind::PositiveZ),
                ] {
                    let mut neighbour = cell;
                    neighbour[axis] += 1;
                    if neighbour[axis] >= counts[axis] {
                        continue;
                    }
                    let second = volume.part_at_physical(parts, neighbour);
                    let opposite = match face {
                        FaceKind::PositiveX => FaceKind::NegativeX,
                        FaceKind::PositiveY => FaceKind::NegativeY,
                        FaceKind::PositiveZ => FaceKind::NegativeZ,
                        _ => unreachable!(),
                    };
                    connections.push(BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(first, face),
                        second: FaceRef::part(second, opposite),
                    }));
                }
            }
        }
    }
}

pub(super) fn append_ground_volume_welds(
    volume: BlockVolume,
    parts: &[PartId],
    connections: &mut Vec<BuildCommand>,
) {
    let (minimum, _) = volume.bounds();
    if minimum.y.abs() > CONTACT_EPSILON {
        return;
    }
    let counts = volume.dimensions();
    for x in 0..counts.x {
        for z in 0..counts.z {
            connections.push(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(
                    volume.part_at_physical(parts, UVec3::new(x, 0, z)),
                    FaceKind::NegativeY,
                ),
                second: FaceRef::ground(),
            }));
        }
    }
}

pub(super) fn append_existing_volume_welds(
    graph: &ConstructionGraph,
    staged: &ConstructionGraph,
    index: &PlacementSnapIndex,
    volume: BlockVolume,
    parts: &[PartId],
    weld_scope: Option<&HashSet<PartId>>,
    connections: &mut Vec<BuildCommand>,
) {
    let (minimum, maximum) = volume.bounds();
    let mut tested_owners = HashSet::new();
    let simple_graph = graph.regions().next().is_none() && graph.shape_features().next().is_none();
    for target in index.nearby(minimum, maximum, CONTACT_EPSILON * 2.0) {
        if weld_scope.is_some_and(|members| !members.contains(&target.part)) {
            continue;
        }
        if simple_graph
            && target.frame == mechanic_core::ConstructionFrame::IDENTITY
            && let PartSpec::Cuboid(existing) = target.spec
            && let Some(cell) = direct_adjacent_unit_cell(volume, existing)
        {
            let part = volume.part_at_physical(parts, cell);
            if let Some((first, second)) =
                direct_block_face_pair(graph, part, volume.spec_at_physical(cell), target.part)
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
            continue;
        }
        let owner = connection_geometry_owner(graph, target.part);
        let (low, high) = volume_candidate_range(volume, target.minimum, target.maximum, true);
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    let cell = UVec3::new(x, y, z);
                    let part = volume.part_at_physical(parts, cell);
                    let (candidate_minimum, candidate_maximum) =
                        cuboid_world_bounds(volume.spec_at_physical(cell));
                    if !bounds_share_face(
                        candidate_minimum,
                        candidate_maximum,
                        target.minimum,
                        target.maximum,
                    ) || !tested_owners.insert((part, owner))
                    {
                        continue;
                    }
                    let direct = direct_block_face_pair(
                        graph,
                        part,
                        volume.spec_at_physical(cell),
                        target.part,
                    );
                    if let Some((first, second)) = direct.or_else(|| {
                        touching_face_pair(
                            staged,
                            FaceOwner::Part(part),
                            FaceOwner::Part(target.part),
                        )
                    }) {
                        connections.push(BuildCommand::Weld(WeldSpec { first, second }));
                    }
                }
            }
        }
    }
}

pub(super) fn direct_adjacent_unit_cell(
    volume: BlockVolume,
    existing: CuboidSpec,
) -> Option<UVec3> {
    if existing.pose.rotation != GridRotation::default()
        || !existing
            .dimensions
            .iter()
            .all(|dimension| dimension.units() == volume.start().dimensions[0].units())
    {
        return None;
    }
    let block_ticks =
        i32::from(volume.start().dimensions[0].units()) * POSITION_TICKS_PER_GRID_UNIT;
    let first_center = volume.start().pose.translation_position_ticks()
        + volume.span.min(IVec3::ZERO) * block_ticks;
    let existing_center = existing.pose.translation_position_ticks();
    let counts = volume.dimensions();
    let mut cell = IVec3::ZERO;
    let mut adjacent_axes = 0;
    for axis in 0..3 {
        let delta = existing_center[axis] - first_center[axis];
        if delta == -block_ticks {
            cell[axis] = 0;
            adjacent_axes += 1;
        } else if delta == i32::try_from(counts[axis]).expect("volume count fits i32") * block_ticks
        {
            cell[axis] = i32::try_from(counts[axis]).expect("volume count fits i32") - 1;
            adjacent_axes += 1;
        } else if delta >= 0
            && delta % block_ticks == 0
            && delta / block_ticks < i32::try_from(counts[axis]).expect("volume count fits i32")
        {
            cell[axis] = delta / block_ticks;
        } else {
            return None;
        }
    }
    (adjacent_axes == 1).then_some(cell.as_uvec3())
}

pub(super) fn direct_block_face_pair(
    graph: &ConstructionGraph,
    new_part: PartId,
    new_spec: CuboidSpec,
    existing_part: PartId,
) -> Option<(FaceRef, FaceRef)> {
    if graph.region_of(existing_part).is_some()
        || graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(existing_part))
    {
        return None;
    }
    let PartSpec::Cuboid(existing) = graph.part(existing_part).copied()? else {
        return None;
    };
    if existing.pose.rotation != GridRotation::default() {
        return None;
    }
    let (new_minimum, new_maximum) = cuboid_world_bounds(new_spec);
    let (old_minimum, old_maximum) = cuboid_world_bounds(existing);
    for (axis, positive, negative) in [
        (0, FaceKind::PositiveX, FaceKind::NegativeX),
        (1, FaceKind::PositiveY, FaceKind::NegativeY),
        (2, FaceKind::PositiveZ, FaceKind::NegativeZ),
    ] {
        if (new_maximum[axis] - old_minimum[axis]).abs() <= CONTACT_EPSILON {
            return Some((
                FaceRef::part(new_part, positive),
                FaceRef::part(existing_part, negative),
            ));
        }
        if (old_maximum[axis] - new_minimum[axis]).abs() <= CONTACT_EPSILON {
            return Some((
                FaceRef::part(new_part, negative),
                FaceRef::part(existing_part, positive),
            ));
        }
    }
    None
}

pub(super) fn bounds_share_face(
    first_minimum: Vec3,
    first_maximum: Vec3,
    second_minimum: Vec3,
    second_maximum: Vec3,
) -> bool {
    (0..3).any(|normal| {
        let touching = (first_maximum[normal] - second_minimum[normal]).abs() <= CONTACT_EPSILON
            || (second_maximum[normal] - first_minimum[normal]).abs() <= CONTACT_EPSILON;
        if !touching {
            return false;
        }
        let tangents = [(normal + 1) % 3, (normal + 2) % 3];
        tangents.into_iter().all(|axis| {
            first_maximum[axis].min(second_maximum[axis])
                - first_minimum[axis].max(second_minimum[axis])
                > CONTACT_EPSILON
        })
    })
}

#[derive(Clone, Copy)]
pub(super) enum FixedPartSpawn {
    Cuboid,
    Controller,
    Engine(EngineKind),
    Servo,
    Seat,
    Input,
    DimensionLink(DimensionLinkId),
}

#[expect(
    clippy::too_many_lines,
    reason = "placement, bearing attachment, and welding share one transaction"
)]
pub(super) fn stage_connected_part_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<BearingAttachment<'_>>,
    auto_weld_source: Option<FaceOwner>,
    spawn: FixedPartSpawn,
    bounds: PlacementBounds,
) -> Result<ConstructionGraph, PlacementError> {
    validate_block_batch_in_bounds(graph, start, specs, bounds)?;
    for (index, spec) in specs.iter().enumerate() {
        for other in &specs[..index] {
            let (minimum, maximum) = cuboid_world_bounds(*spec);
            let (other_minimum, other_maximum) = cuboid_world_bounds(*other);
            if bounds_overlap_interior(minimum, maximum, other_minimum, other_maximum) {
                return Err(PlacementError::BlocksOverlap);
            }
        }
    }

    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.begin_edit();
    let outcomes = staged
        .apply_batch(specs.iter().copied().map(|spec| match spawn {
            FixedPartSpawn::Cuboid => BuildCommand::Spawn(spec),
            FixedPartSpawn::Controller => {
                BuildCommand::SpawnController(ControllerSpec::new(spec.pose))
            }
            FixedPartSpawn::Engine(kind) => {
                BuildCommand::SpawnEngine(EngineSpec::new(kind, spec.pose))
            }
            FixedPartSpawn::Servo => BuildCommand::SpawnServo(ServoSpec::new(spec.pose)),
            FixedPartSpawn::Seat => BuildCommand::SpawnSeat(SeatSpec::new(spec.pose)),
            FixedPartSpawn::Input => BuildCommand::SpawnInput(InputSpec::new(spec.pose)),
            FixedPartSpawn::DimensionLink(id) => {
                BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(id, spec.pose))
            }
        }))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let new_parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("spawn batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();

    let mut connections = Vec::new();
    if let Some(BearingAttachment {
        source,
        anchor,
        dimensions,
        kind,
        axis,
        rigid_targets,
    }) = bearing
    {
        let first = *new_parts
            .first()
            .expect("validated block batches are never empty");
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(first, start.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions)
            .with_kind(kind),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: first,
            })
        }));
    }
    for (index, &part) in new_parts.iter().enumerate() {
        if bearing.is_none()
            && weld_scope.is_none()
            && bounds == PlacementBounds::Garage
            && let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
        {
            connections.push(BuildCommand::Weld(WeldSpec { first, second }));
        }
        let mut tested_owners = HashSet::new();
        for &other in &existing_parts {
            if bearing.is_some()
                || weld_scope
                    .as_ref()
                    .is_some_and(|members| !members.contains(&other))
                || !tested_owners.insert(connection_geometry_owner(&staged, other))
            {
                continue;
            }
            if let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
        }
        for &other in &new_parts[..index] {
            if let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Part(other))
            {
                connections.push(BuildCommand::Weld(WeldSpec { first, second }));
            }
        }
    }
    staged
        .apply_batch(connections)
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged.finish())
}

pub(super) fn connection_geometry_owner(
    graph: &ConstructionGraph,
    part: PartId,
) -> mechanic_core::SolidOwner {
    let owner = graph.region_of(part).map_or(
        mechanic_core::SolidOwner::Part(part),
        mechanic_core::SolidOwner::Region,
    );
    if graph.owner_has_shape_features(owner) {
        owner
    } else {
        mechanic_core::SolidOwner::Part(part)
    }
}

pub(super) fn bearing_connected_weld_scope(
    graph: &ConstructionGraph,
    source: FaceOwner,
) -> Option<HashSet<PartId>> {
    let FaceOwner::Part(seed) = source else {
        return None;
    };
    let members = rigid_body_parts(graph, seed)
        .into_iter()
        .collect::<HashSet<_>>();
    graph
        .bearings()
        .any(|(_, bearing)| {
            [bearing.source.owner, bearing.target.owner]
                .into_iter()
                .any(|owner| matches!(owner, FaceOwner::Part(part) if members.contains(&part)))
        })
        .then_some(members)
}
