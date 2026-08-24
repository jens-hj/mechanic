use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt,
};

use bevy::prelude::*;
use mechanic_core::{
    BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec, EngineKind, EngineSpec, FaceKind,
    FaceOwner, FaceRef, GridRotation, PartId, PartSpec, PendingOperation, RigidLinkSpec, WeldSpec,
    snap_world_to_grid,
};

pub(crate) const GROUND_HALF_SIZE: f32 = 10.0;
const CONTACT_EPSILON: f32 = 1.0e-5;
const GRID_UNIT_METERS: f32 = 0.25;
pub(crate) const BEARING_DEPTH: f32 = 0.10;
pub(crate) const MAX_DRAG_BLOCKS: usize = 4_096;
pub(crate) const BLOCK_SIZE_METERS: f32 = GRID_UNIT_METERS;
const BLOCK_SIZE_UNITS: u8 = 1;
const HALF_GRID_UNIT_METERS: f32 = GRID_UNIT_METERS * 0.5;

const ALL_FACES: [FaceKind; 6] = [
    FaceKind::PositiveX,
    FaceKind::NegativeX,
    FaceKind::PositiveY,
    FaceKind::NegativeY,
    FaceKind::PositiveZ,
    FaceKind::NegativeZ,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacementPlane {
    Xy,
    Xz,
    Yz,
}

impl PlacementPlane {
    pub(crate) fn from_normal(normal: Vec3) -> Self {
        match cardinal_axis(normal).0 {
            0 => Self::Yz,
            1 => Self::Xz,
            _ => Self::Xy,
        }
    }

    pub(crate) const fn cycle(self) -> Self {
        match self {
            Self::Xz => Self::Xy,
            Self::Xy => Self::Yz,
            Self::Yz => Self::Xz,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Xy => "XY",
            Self::Xz => "XZ",
            Self::Yz => "YZ",
        }
    }

    pub(crate) const fn normal_axis(self) -> usize {
        match self {
            Self::Xy => 2,
            Self::Xz => 1,
            Self::Yz => 0,
        }
    }

    pub(crate) const fn tangent_axes(self) -> [usize; 2] {
        match self {
            Self::Xy => [0, 1],
            Self::Xz => [0, 2],
            Self::Yz => [1, 2],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceHit {
    pub(crate) distance: f32,
    pub(crate) point: Vec3,
    pub(crate) face: FaceRef,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OrientedCuboidHit {
    pub(crate) distance: f32,
    pub(crate) point: Vec3,
    pub(crate) local_normal: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacementCandidate {
    pub(crate) spec: CuboidSpec,
    pub(crate) attached_face: FaceKind,
    pub(crate) anchor: Option<Vec3>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CylinderPlacementCandidate {
    pub(crate) spec: CylinderSpec,
    pub(crate) attached_face: FaceKind,
    pub(crate) anchor: Option<Vec3>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlacementError {
    OutsidePlatform,
    NoFaceOverlap,
    OverlapsPart(PartId),
    BearingOnGround,
    BearingOutsideFace,
    SameObject,
    ObjectsDoNotTouch,
    CurvedSurface,
    EmptyBlockBatch,
    BlocksOverlap,
    DragPlaneUnavailable,
    TooManyBlocks { count: usize, maximum: usize },
    Graph(String),
}

impl fmt::Display for PlacementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutsidePlatform => formatter.write_str("part would extend beyond the platform"),
            Self::NoFaceOverlap => formatter.write_str("cube does not overlap the selected face"),
            Self::OverlapsPart(part) => write!(formatter, "part overlaps {part:?}"),
            Self::BearingOnGround => formatter.write_str("bearings cannot attach to the ground"),
            Self::BearingOutsideFace => {
                formatter.write_str("the bearing anchor lies outside this face")
            }
            Self::SameObject => formatter.write_str("select two different objects"),
            Self::ObjectsDoNotTouch => formatter.write_str("the selected objects do not touch"),
            Self::CurvedSurface => {
                formatter.write_str("curved cylinder walls are not connection faces")
            }
            Self::EmptyBlockBatch => formatter.write_str("block drag did not produce any blocks"),
            Self::BlocksOverlap => formatter.write_str("dragged blocks overlap one another"),
            Self::DragPlaneUnavailable => {
                formatter.write_str("camera ray does not reach the selected drag plane")
            }
            Self::TooManyBlocks { count, maximum } => {
                write!(
                    formatter,
                    "drag would place {count} blocks; maximum is {maximum}"
                )
            }
            Self::Graph(error) => formatter.write_str(error),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceGeometry {
    pub(crate) center: Vec3,
    pub(crate) normal: Vec3,
    tangent_u: Vec3,
    tangent_v: Vec3,
    profile: FaceProfile,
}

#[derive(Clone, Copy, Debug)]
enum FaceProfile {
    Rectangle {
        half_u: f32,
        half_v: f32,
    },
    Annulus {
        inner_radius: f32,
        outer_radius: f32,
    },
    AnnularSector {
        inner_radius: f32,
        outer_radius: f32,
        half_angle: f32,
    },
    Ground,
}

pub(crate) fn raycast_construction(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
) -> Option<SurfaceHit> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    let ground = raycast_ground(origin, direction);
    graph
        .parts()
        .filter_map(|(part, spec)| raycast_part(origin, direction, part, *spec))
        .chain(ground)
        .filter(|hit| hit.distance >= 0.0 && hit.distance.is_finite())
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .unwrap_or(Ordering::Equal)
        })
}

pub(crate) fn raycast_construction_for_annulus(
    graph: &ConstructionGraph,
    origin: Vec3,
    direction: Vec3,
    inner_diameter: f32,
    outer_diameter: f32,
) -> Option<SurfaceHit> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction.length_squared() < f32::EPSILON
        || !inner_diameter.is_finite()
        || !outer_diameter.is_finite()
        || inner_diameter < 0.0
        || outer_diameter <= inner_diameter
    {
        return None;
    }
    let direction = direction.normalize();
    let placement_profile = FaceProfile::Annulus {
        inner_radius: inner_diameter * 0.5,
        outer_radius: outer_diameter * 0.5,
    };
    raycast_construction(graph, origin, direction)
        .into_iter()
        .chain(graph.parts().filter_map(|(part, spec)| match spec {
            PartSpec::Cylinder(spec) => {
                raycast_cylinder_bore_obstruction(origin, direction, part, *spec, placement_profile)
            }
            PartSpec::Cuboid(_) | PartSpec::Controller(_) | PartSpec::Engine(_) => None,
        }))
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

fn raycast_cylinder_bore_obstruction(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CylinderSpec,
    placement_profile: FaceProfile,
) -> Option<SurfaceHit> {
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    if local_direction.y.abs() <= f32::EPSILON {
        return None;
    }
    let outer_radius = spec.dimensions.outer_diameter() * 0.5;
    let half_length = spec.dimensions.axial_length() * 0.5;
    [
        (half_length, FaceKind::PositiveY),
        (-half_length, FaceKind::NegativeY),
    ]
    .into_iter()
    .filter_map(|(y, face_kind)| {
        let distance = (y - local_origin.y) / local_direction.y;
        if distance < 0.0 {
            return None;
        }
        let local_point = local_origin + local_direction * distance;
        let radial_squared = local_point
            .x
            .mul_add(local_point.x, local_point.z * local_point.z);
        if radial_squared > (outer_radius + CONTACT_EPSILON).powi(2) {
            return None;
        }
        let support = cylinder_face_geometry(spec, face_kind)
            .expect("cylinder end faces always expose flat geometry");
        if point_in_profile(local_point.x, local_point.z, support.profile) {
            return None;
        }
        let placement = FaceGeometry {
            center: origin + direction * distance,
            normal: -support.normal,
            tangent_u: support.tangent_u,
            tangent_v: support.tangent_v,
            profile: placement_profile,
        };
        profiles_overlap(support, placement).then_some(SurfaceHit {
            distance,
            point: placement.center,
            face: FaceRef::part(part, face_kind),
        })
    })
    .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

pub(crate) fn candidate_from_hit(graph: &ConstructionGraph, hit: SurfaceHit) -> PlacementCandidate {
    cuboid_candidate_from_hit(graph, hit, [BLOCK_SIZE_UNITS; 3])
}

/// Places a fixed-size authored cuboid flush with the face under the pointer.
pub(crate) fn cuboid_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
) -> PlacementCandidate {
    oriented_cuboid_candidate_from_hit(graph, hit, dimensions, 0)
}

/// Places a fixed-size cuboid with a quarter-turn yaw flush with a face.
pub(crate) fn oriented_cuboid_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    quarter_turns_y: u8,
) -> PlacementCandidate {
    let quarter_turns_y = quarter_turns_y % 4;
    let rotation = GridRotation::new(0, quarter_turns_y, 0);
    let world_dimensions = if quarter_turns_y.is_multiple_of(2) {
        dimensions
    } else {
        [dimensions[2], dimensions[1], dimensions[0]]
    };
    let support = face_geometry_from_ref(hit.face, Some(graph));
    let support_center_half_units = snap_world_to_half_grid(support.center);
    let mut center_half_units =
        support_center_half_units + snap_world_to_grid(hit.point - support.center) * 2;
    let (axis, sign) = cardinal_axis(support.normal);
    for tangent_axis in 0..3 {
        if tangent_axis == axis {
            continue;
        }
        // Horizontal block cells are centred on quarter-metre grid lines,
        // while vertical cells begin half a grid unit above the platform. An
        // even span therefore needs a half-cell centre on X/Z and a whole-cell
        // centre on Y. Preserve that lattice when the target and support have
        // different dimension parity.
        let desired_parity = if tangent_axis == 1 {
            i32::from(world_dimensions[tangent_axis] % 2)
        } else {
            i32::from((world_dimensions[tangent_axis] + 1) % 2)
        };
        if center_half_units[tangent_axis].rem_euclid(2) != desired_parity {
            center_half_units[tangent_axis] += 1;
        }
    }
    center_half_units[axis] =
        support_center_half_units[axis] + sign * i32::from(world_dimensions[axis]);

    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_half_grid(center_half_units, rotation),
    )
    .expect("the fixed block size is a valid core dimension");
    let attached_face = face_for_normal(rotation.quaternion().inverse() * -support.normal);
    let candidate_face = face_geometry(spec, attached_face);
    let anchor = overlap_center(support, candidate_face);
    PlacementCandidate {
        spec,
        attached_face,
        anchor,
    }
}

pub(crate) fn cylinder_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let support =
        try_face_geometry_from_ref(hit.face, Some(graph)).ok_or(PlacementError::CurvedSurface)?;
    let support_center_half_units = snap_world_to_half_grid(support.center);
    let mut center_half_units =
        support_center_half_units + snap_world_to_grid(hit.point - support.center) * 2;
    let (axis, sign) = cardinal_axis(support.normal);
    center_half_units[axis] =
        support_center_half_units[axis] + sign * i32::from(dimensions.axial_length_units());
    let rotation = rotation_y_to_normal(support.normal);
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_half_grid(center_half_units, rotation),
    );
    let attached_face = FaceKind::NegativeY;
    let candidate_face = cylinder_face_geometry(spec, attached_face)
        .expect("negative-y is a cylinder connection face");
    Ok(CylinderPlacementCandidate {
        spec,
        attached_face,
        anchor: supporting_face_overlap(graph, support, candidate_face),
    })
}

fn supporting_face_overlap(
    graph: &ConstructionGraph,
    selected: FaceGeometry,
    candidate: FaceGeometry,
) -> Option<Vec3> {
    overlap_center(selected, candidate).or_else(|| {
        graph.parts().find_map(|(_, spec)| {
            ALL_FACES.into_iter().find_map(|face| {
                part_face_geometry(*spec, face)
                    .and_then(|support| overlap_center(support, candidate))
            })
        })
    })
}

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
    stage_connected_block_batch(graph, start, specs, None, None)
}

/// Stages one control block, auto-welding it like an ordinary block.
pub(crate) fn stage_controller_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        Some(source),
        FixedPartSpawn::Controller,
    )
}

/// Stages one inert engine, auto-welding it like an ordinary block.
pub(crate) fn stage_engine_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    source: FaceOwner,
    kind: EngineKind,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        &[start.spec],
        None,
        Some(source),
        FixedPartSpawn::Engine(kind),
    )
}

pub(crate) fn stage_block_batch_from_source(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, None, Some(source))
}

pub(crate) fn stage_bearing_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(
        graph,
        start,
        specs,
        Some((source, anchor, dimensions, rigid_targets)),
        None,
    )
}

pub(crate) fn validate_cylinder_candidate(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
) -> Result<(), PlacementError> {
    if candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    validate_part(graph, PartSpec::Cylinder(candidate.spec))
}

pub(crate) fn stage_cylinder_from_source(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    source: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(graph, candidate, None, Some(source))
}

pub(crate) fn stage_bearing_cylinder(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    rigid_targets: &[PartId],
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_cylinder(
        graph,
        candidate,
        Some((source, anchor, dimensions, rigid_targets)),
        None,
    )
}

fn stage_connected_cylinder(
    graph: &ConstructionGraph,
    candidate: CylinderPlacementCandidate,
    bearing: Option<(FaceRef, Vec3, BearingDimensions, &[PartId])>,
    auto_weld_source: Option<FaceOwner>,
) -> Result<ConstructionGraph, PlacementError> {
    validate_cylinder_candidate(graph, candidate)?;
    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let weld_scope =
        auto_weld_source.and_then(|source| bearing_connected_weld_scope(graph, source));
    let mut staged = graph.clone();
    let BuildOutcome::Spawned(part) = staged
        .apply(BuildCommand::SpawnCylinder(candidate.spec))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!()
    };
    let mut connections = Vec::new();
    if let Some((source, anchor, dimensions, rigid_targets)) = bearing {
        let axis = face_geometry_from_ref(source, Some(graph)).normal;
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(part, candidate.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions),
        ));
        connections.extend(rigid_targets.iter().copied().map(|target| {
            BuildCommand::RigidLink(RigidLinkSpec {
                first: target,
                second: part,
            })
        }));
    } else {
        if weld_scope.is_none()
            && let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
        {
            connections.push(BuildCommand::Weld(WeldSpec { first, second }));
        }
        for other in existing_parts {
            if weld_scope
                .as_ref()
                .is_some_and(|members| !members.contains(&other))
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
    Ok(staged)
}

fn stage_connected_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<(FaceRef, Vec3, BearingDimensions, &[PartId])>,
    auto_weld_source: Option<FaceOwner>,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_part_batch(
        graph,
        start,
        specs,
        bearing,
        auto_weld_source,
        FixedPartSpawn::Cuboid,
    )
}

#[derive(Clone, Copy)]
enum FixedPartSpawn {
    Cuboid,
    Controller,
    Engine(EngineKind),
}

#[allow(clippy::too_many_arguments)] // Placement, bearing attachment, and welding share one transaction.
fn stage_connected_part_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<(FaceRef, Vec3, BearingDimensions, &[PartId])>,
    auto_weld_source: Option<FaceOwner>,
    spawn: FixedPartSpawn,
) -> Result<ConstructionGraph, PlacementError> {
    validate_block_batch(graph, start, specs)?;
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
    let mut staged = graph.clone();
    let outcomes = staged
        .apply_batch(specs.iter().copied().map(|spec| match spawn {
            FixedPartSpawn::Cuboid => BuildCommand::Spawn(spec),
            FixedPartSpawn::Controller => {
                BuildCommand::SpawnController(ControllerSpec::new(spec.pose))
            }
            FixedPartSpawn::Engine(kind) => {
                BuildCommand::SpawnEngine(EngineSpec::new(kind, spec.pose))
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
    if let Some((source, anchor, dimensions, rigid_targets)) = bearing {
        let first = *new_parts
            .first()
            .expect("validated block batches are never empty");
        let axis = face_geometry_from_ref(source, Some(graph)).normal;
        connections.push(BuildCommand::AddBearing(
            BearingSpec::new(
                source,
                FaceRef::part(first, start.attached_face),
                anchor,
                axis,
            )
            .with_dimensions(dimensions),
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
            && let Some((first, second)) =
                touching_face_pair(&staged, FaceOwner::Part(part), FaceOwner::Ground)
        {
            connections.push(BuildCommand::Weld(WeldSpec { first, second }));
        }
        for &other in &existing_parts {
            if bearing.is_some()
                || weld_scope
                    .as_ref()
                    .is_some_and(|members| !members.contains(&other))
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
    Ok(staged)
}

fn bearing_connected_weld_scope(
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

pub(crate) fn validate_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
) -> Result<(), PlacementError> {
    validate_candidate(graph, start)?;
    if specs.is_empty() {
        return Err(PlacementError::EmptyBlockBatch);
    }
    for spec in specs {
        validate_spec(graph, *spec)?;
    }
    Ok(())
}

pub(crate) fn block_sheet_specs(
    start: CuboidSpec,
    endpoint_units: IVec3,
    plane: PlacementPlane,
) -> Result<Vec<CuboidSpec>, PlacementError> {
    let start_units = start.pose.translation_half_units();
    let dimension_units = start.dimensions[0].units();
    let block_units = i32::from(dimension_units) * 2;
    let axes = plane.tangent_axes();
    let steps = axes.map(|axis| rounded_div(endpoint_units[axis] - start_units[axis], block_units));
    let counts = steps.map(|step| step.unsigned_abs() as usize + 1);
    let count = counts[0].saturating_mul(counts[1]);
    if count > MAX_DRAG_BLOCKS {
        return Err(PlacementError::TooManyBlocks {
            count,
            maximum: MAX_DRAG_BLOCKS,
        });
    }

    let mut specs = Vec::with_capacity(count);
    for first in inclusive_steps(steps[0]) {
        for second in inclusive_steps(steps[1]) {
            let mut center = start_units;
            center[axes[0]] += first * block_units;
            center[axes[1]] += second * block_units;
            specs.push(
                CuboidSpec::new(
                    [dimension_units; 3],
                    BuildPose::from_half_grid(center, GridRotation::default()),
                )
                .expect("dragged blocks retain the selected valid size"),
            );
        }
    }
    Ok(specs)
}

pub(crate) fn raycast_placement_plane(
    origin: Vec3,
    direction: Vec3,
    start: CuboidSpec,
    plane: PlacementPlane,
) -> Option<IVec3> {
    let point = raycast_placement_plane_point(origin, direction, start, plane)?;
    let mut endpoint = snap_world_to_half_grid(point);
    endpoint[plane.normal_axis()] = start.pose.translation_half_units()[plane.normal_axis()];
    Some(endpoint)
}

/// Intersects a pointer ray with the plane through the dragged block's centre.
///
/// This deliberately leaves the point unsnapped. Block dragging subtracts two
/// such points before quantizing, so the press position rather than the snapped
/// block centre is the gesture's origin.
pub(crate) fn raycast_placement_plane_point(
    origin: Vec3,
    direction: Vec3,
    start: CuboidSpec,
    plane: PlacementPlane,
) -> Option<Vec3> {
    let axis = plane.normal_axis();
    let denominator = direction[axis];
    if !origin.is_finite() || !direction.is_finite() || denominator.abs() <= f32::EPSILON {
        return None;
    }
    let coordinate = start.pose.translation()[axis];
    let distance = (coordinate - origin[axis]) / denominator;
    if distance < 0.0 || !distance.is_finite() {
        return None;
    }
    Some(origin + direction * distance)
}

/// Converts motion between two pointer rays into whole block steps.
pub(crate) fn block_sheet_endpoint_from_rays(
    start: CuboidSpec,
    plane: PlacementPlane,
    press_origin: Vec3,
    press_direction: Vec3,
    current_origin: Vec3,
    current_direction: Vec3,
) -> Option<IVec3> {
    let press = raycast_placement_plane_point(press_origin, press_direction, start, plane)?;
    let current = raycast_placement_plane_point(current_origin, current_direction, start, plane)?;
    let mut endpoint = start.pose.translation_half_units();
    let block_half_units = i32::from(start.dimensions[0].units()) * 2;
    let steps = ((current - press) / BLOCK_SIZE_METERS).round().as_ivec3();
    for axis in plane.tangent_axes() {
        endpoint[axis] =
            endpoint[axis].saturating_add(steps[axis].saturating_mul(block_half_units));
    }
    Some(endpoint)
}

fn snap_world_to_half_grid(position: Vec3) -> IVec3 {
    (position / HALF_GRID_UNIT_METERS).round().as_ivec3()
}

fn inclusive_steps(end: i32) -> impl Iterator<Item = i32> {
    let direction = end.signum();
    (0..=end.unsigned_abs())
        .map(move |step| i32::try_from(step).expect("drag step count fits in i32") * direction)
}

fn rounded_div(value: i32, divisor: i32) -> i32 {
    let half = divisor / 2;
    if value >= 0 {
        value.saturating_add(half) / divisor
    } else {
        value.saturating_sub(half) / divisor
    }
}

pub(crate) fn begin_weld(
    graph: &mut ConstructionGraph,
    face: FaceRef,
) -> Result<(), PlacementError> {
    graph
        .apply(BuildCommand::BeginPending(PendingOperation::Weld(face)))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(())
}

pub(crate) fn stage_weld_objects(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Result<ConstructionGraph, PlacementError> {
    if first == second {
        return Err(PlacementError::SameObject);
    }
    let Some((first_face, second_face)) = touching_weld_face_pair(graph, first, second) else {
        return Err(PlacementError::ObjectsDoNotTouch);
    };
    let mut staged = graph.clone();
    staged
        .apply(BuildCommand::Weld(WeldSpec {
            first: first_face,
            second: second_face,
        }))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged)
}

fn touching_weld_face_pair(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Option<(FaceRef, FaceRef)> {
    match (first, second) {
        (FaceOwner::Ground, FaceOwner::Part(part)) => rigid_body_parts(graph, part)
            .into_iter()
            .find_map(|member| {
                touching_face_pair(graph, FaceOwner::Ground, FaceOwner::Part(member))
            }),
        (FaceOwner::Part(part), FaceOwner::Ground) => rigid_body_parts(graph, part)
            .into_iter()
            .find_map(|member| {
                touching_face_pair(graph, FaceOwner::Part(member), FaceOwner::Ground)
            }),
        _ => touching_face_pair(graph, first, second),
    }
}

pub(crate) fn rigid_body_parts(graph: &ConstructionGraph, seed: PartId) -> Vec<PartId> {
    if graph.part(seed).is_none() {
        return Vec::new();
    }

    let mut neighbours = HashMap::<PartId, Vec<PartId>>::new();
    for (_, weld) in graph.welds() {
        if let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        {
            neighbours.entry(first).or_default().push(second);
            neighbours.entry(second).or_default().push(first);
        }
    }
    for (_, link) in graph.rigid_links() {
        neighbours.entry(link.first).or_default().push(link.second);
        neighbours.entry(link.second).or_default().push(link.first);
    }

    let mut members = HashSet::from([seed]);
    let mut pending = vec![seed];
    while let Some(part) = pending.pop() {
        if let Some(connected) = neighbours.get(&part) {
            for &candidate in connected {
                if members.insert(candidate) {
                    pending.push(candidate);
                }
            }
        }
    }

    graph
        .parts()
        .filter_map(|(part, _)| members.contains(&part).then_some(part))
        .collect()
}

pub(crate) fn bearing_anchor_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<Vec3, PlacementError> {
    if matches!(hit.face.owner, FaceOwner::Ground) {
        return Err(PlacementError::BearingOnGround);
    }
    let face =
        try_face_geometry_from_ref(hit.face, Some(graph)).ok_or(PlacementError::CurvedSurface)?;
    let mut anchor =
        face.center + snap_world_to_grid(hit.point - face.center).as_vec3() * GRID_UNIT_METERS;
    let (normal_axis, _) = cardinal_axis(face.normal);
    anchor[normal_axis] = face.center[normal_axis];
    let offset = anchor - face.center;
    let u = offset.dot(face.tangent_u);
    let v = offset.dot(face.tangent_v);
    let inside_face_extent = match face.profile {
        FaceProfile::Annulus { outer_radius, .. }
        | FaceProfile::AnnularSector { outer_radius, .. } => {
            u.mul_add(u, v * v) <= (outer_radius + CONTACT_EPSILON).powi(2)
        }
        _ => point_in_profile(u, v, face.profile),
    };
    if !inside_face_extent {
        return Err(PlacementError::BearingOutsideFace);
    }
    Ok(anchor)
}

pub(crate) fn bearing_attachment_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
) -> PlacementCandidate {
    candidate_from_hit(
        graph,
        SurfaceHit {
            distance: 0.0,
            point: anchor,
            face: source,
        },
    )
}

pub(crate) fn bearing_support_face(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
) -> Option<FaceRef> {
    bearing_support_face_excluding(graph, selected_face, anchor, dimensions, &HashSet::new())
}

pub(crate) fn bearing_support_face_excluding(
    graph: &ConstructionGraph,
    selected_face: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    excluded_parts: &HashSet<PartId>,
) -> Option<FaceRef> {
    let selected = face_geometry_from_ref(selected_face, Some(graph));
    let mut fallback = None;
    for (part, spec) in graph.parts() {
        if excluded_parts.contains(&part) {
            continue;
        }
        for face_kind in ALL_FACES {
            let face_ref = FaceRef::part(part, face_kind);
            let Some(face) = part_face_geometry(*spec, face_kind) else {
                continue;
            };
            if !faces_share_plane_and_normal(selected, face)
                || !bearing_ring_overlaps_face(anchor, dimensions, face)
            {
                continue;
            }
            if bearing_ring_contains_face_center(anchor, dimensions, face) {
                return Some(face_ref);
            }
            fallback.get_or_insert(face_ref);
        }
    }
    fallback
}

pub(crate) fn bearing_overlaps_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    candidate: PlacementCandidate,
) -> bool {
    let source_face = face_geometry_from_ref(source, Some(graph));
    let target_face = face_geometry(candidate.spec, candidate.attached_face);
    if source_face.normal.dot(target_face.normal) > -1.0 + CONTACT_EPSILON
        || (source_face.center - target_face.center)
            .dot(source_face.normal)
            .abs()
            > CONTACT_EPSILON
    {
        return false;
    }
    bearing_ring_overlaps_face(anchor, dimensions, target_face)
}

pub(crate) fn bearing_overlaps_cylinder_candidate(
    graph: &ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
    candidate: CylinderPlacementCandidate,
) -> bool {
    let source_face = face_geometry_from_ref(source, Some(graph));
    let target_face = cylinder_face_geometry(candidate.spec, candidate.attached_face)
        .expect("cylinder attachment face is flat");
    source_face.normal.dot(target_face.normal) <= -1.0 + CONTACT_EPSILON
        && (source_face.center - target_face.center)
            .dot(source_face.normal)
            .abs()
            <= CONTACT_EPSILON
        && bearing_ring_overlaps_face(anchor, dimensions, target_face)
}

pub(crate) fn stage_bearing_attachment(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_block_batch(
        graph,
        candidate,
        &[candidate.spec],
        source,
        anchor,
        dimensions,
        &[],
    )
}

fn bearing_ring_overlaps_face(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: FaceGeometry,
) -> bool {
    profiles_overlap(
        FaceGeometry {
            center: anchor,
            normal: face.normal,
            tangent_u: face.tangent_u,
            tangent_v: face.tangent_v,
            profile: FaceProfile::Annulus {
                inner_radius: dimensions.inner_diameter() * 0.5,
                outer_radius: dimensions.outer_diameter() * 0.5,
            },
        },
        face,
    )
}

fn bearing_ring_contains_face_center(
    anchor: Vec3,
    dimensions: BearingDimensions,
    face: FaceGeometry,
) -> bool {
    let offset = anchor - face.center;
    if offset.dot(face.normal).abs() > CONTACT_EPSILON {
        return false;
    }
    let radial = offset - face.normal * offset.dot(face.normal);
    point_in_profile(
        radial.dot(face.tangent_u),
        radial.dot(face.tangent_v),
        FaceProfile::Annulus {
            inner_radius: dimensions.inner_diameter() * 0.5,
            outer_radius: dimensions.outer_diameter() * 0.5,
        },
    )
}

fn faces_share_plane_and_normal(first: FaceGeometry, second: FaceGeometry) -> bool {
    first.normal.dot(second.normal) > 1.0 - CONTACT_EPSILON
        && (first.center - second.center).dot(first.normal).abs() <= CONTACT_EPSILON
}

pub(crate) fn face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> FaceGeometry {
    try_face_geometry_from_ref(face, graph).expect("face reference must expose flat geometry")
}

pub(crate) fn try_face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> Option<FaceGeometry> {
    match face.owner {
        FaceOwner::Ground => Some(FaceGeometry {
            center: Vec3::ZERO,
            normal: Vec3::Y,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Z,
            profile: FaceProfile::Ground,
        }),
        FaceOwner::Part(part) => {
            let spec = graph
                .and_then(|graph| graph.part(part))
                .copied()
                .expect("live face references have a part");
            part_face_geometry(spec, face.face)
        }
    }
}

pub(crate) fn face_geometry(spec: CuboidSpec, face: FaceKind) -> FaceGeometry {
    let rotation = spec.pose.rotation.quaternion();
    let size = spec.size_meters();
    let (normal, tangent_u, tangent_v, normal_extent, half_u, half_v) = match face {
        FaceKind::PositiveX => (Vec3::X, Vec3::Y, Vec3::Z, size.x, size.y, size.z),
        FaceKind::NegativeX => (-Vec3::X, Vec3::Y, Vec3::Z, size.x, size.y, size.z),
        FaceKind::PositiveY => (Vec3::Y, Vec3::X, Vec3::Z, size.y, size.x, size.z),
        FaceKind::NegativeY => (-Vec3::Y, Vec3::X, Vec3::Z, size.y, size.x, size.z),
        FaceKind::PositiveZ => (Vec3::Z, Vec3::X, Vec3::Y, size.z, size.x, size.y),
        FaceKind::NegativeZ => (-Vec3::Z, Vec3::X, Vec3::Y, size.z, size.x, size.y),
    };
    let normal = snap_cardinal(rotation * normal);
    FaceGeometry {
        center: spec.pose.translation() + normal * normal_extent * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * tangent_u),
        tangent_v: snap_cardinal(rotation * tangent_v),
        profile: FaceProfile::Rectangle {
            half_u: half_u * 0.5,
            half_v: half_v * 0.5,
        },
    }
}

fn cylinder_face_geometry(spec: CylinderSpec, face: FaceKind) -> Option<FaceGeometry> {
    if !matches!(face, FaceKind::PositiveY | FaceKind::NegativeY) {
        return None;
    }
    let rotation = spec.pose.rotation.quaternion();
    let local_normal = if face == FaceKind::PositiveY {
        Vec3::Y
    } else {
        Vec3::NEG_Y
    };
    let normal = snap_cardinal(rotation * local_normal);
    let profile = if spec.dimensions.sweep_angle_degrees() == 360 {
        FaceProfile::Annulus {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
        }
    } else {
        FaceProfile::AnnularSector {
            inner_radius: spec.dimensions.inner_diameter() * 0.5,
            outer_radius: spec.dimensions.outer_diameter() * 0.5,
            half_angle: spec.dimensions.sweep_angle_radians() * 0.5,
        }
    };
    Some(FaceGeometry {
        center: spec.pose.translation() + normal * spec.dimensions.axial_length() * 0.5,
        normal,
        tangent_u: snap_cardinal(rotation * Vec3::X),
        tangent_v: snap_cardinal(rotation * Vec3::Z),
        profile,
    })
}

fn part_face_geometry(spec: PartSpec, face: FaceKind) -> Option<FaceGeometry> {
    match spec {
        PartSpec::Cuboid(spec) => Some(face_geometry(spec, face)),
        PartSpec::Controller(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Engine(spec) => Some(face_geometry(spec.cuboid(), face)),
        PartSpec::Cylinder(spec) => cylinder_face_geometry(spec, face),
    }
}

fn validate_candidate(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
) -> Result<(), PlacementError> {
    validate_spec(graph, candidate.spec)?;
    if candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    Ok(())
}

fn validate_spec(graph: &ConstructionGraph, spec: CuboidSpec) -> Result<(), PlacementError> {
    validate_part(graph, PartSpec::Cuboid(spec))
}

fn validate_part(graph: &ConstructionGraph, spec: PartSpec) -> Result<(), PlacementError> {
    let (minimum, maximum) = part_world_bounds(spec);
    if minimum.x < -GROUND_HALF_SIZE - CONTACT_EPSILON
        || maximum.x > GROUND_HALF_SIZE + CONTACT_EPSILON
        || minimum.z < -GROUND_HALF_SIZE - CONTACT_EPSILON
        || maximum.z > GROUND_HALF_SIZE + CONTACT_EPSILON
        || minimum.y < -CONTACT_EPSILON
    {
        return Err(PlacementError::OutsidePlatform);
    }
    for (part, existing) in graph.parts() {
        if parts_overlap(spec, *existing) {
            return Err(PlacementError::OverlapsPart(part));
        }
    }
    Ok(())
}

fn touching_face_pair(
    graph: &ConstructionGraph,
    first: FaceOwner,
    second: FaceOwner,
) -> Option<(FaceRef, FaceRef)> {
    owner_faces(graph, first)
        .into_iter()
        .find_map(|first_face| {
            owner_faces(graph, second)
                .into_iter()
                .find_map(|second_face| {
                    overlap_center(
                        face_geometry_from_ref(first_face, Some(graph)),
                        face_geometry_from_ref(second_face, Some(graph)),
                    )
                    .map(|_| (first_face, second_face))
                })
        })
}

fn owner_faces(graph: &ConstructionGraph, owner: FaceOwner) -> Vec<FaceRef> {
    match owner {
        FaceOwner::Ground => vec![FaceRef::ground()],
        FaceOwner::Part(part) => match graph.part(part).copied() {
            Some(PartSpec::Cuboid(_) | PartSpec::Controller(_) | PartSpec::Engine(_)) => ALL_FACES
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            Some(PartSpec::Cylinder(_)) => [FaceKind::PositiveY, FaceKind::NegativeY]
                .into_iter()
                .map(|face| FaceRef::part(part, face))
                .collect(),
            None => Vec::new(),
        },
    }
}

fn raycast_ground(origin: Vec3, direction: Vec3) -> Option<SurfaceHit> {
    // The platform is a build surface from above, not a wall that hides the
    // construction when the camera is underneath it.
    if direction.y >= -f32::EPSILON {
        return None;
    }
    let distance = -origin.y / direction.y;
    let point = origin + direction * distance;
    (distance >= 0.0 && point.x.abs() <= GROUND_HALF_SIZE && point.z.abs() <= GROUND_HALF_SIZE)
        .then_some(SurfaceHit {
            distance,
            point,
            face: FaceRef::ground(),
        })
}

fn raycast_cuboid(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CuboidSpec,
) -> Option<SurfaceHit> {
    let hit = raycast_oriented_cuboid(
        origin,
        direction,
        spec.pose.translation(),
        spec.pose.rotation.quaternion(),
        spec.size_meters() * 0.5,
    )?;
    Some(SurfaceHit {
        distance: hit.distance,
        point: hit.point,
        face: FaceRef::part(part, face_for_normal(hit.local_normal)),
    })
}

fn raycast_part(origin: Vec3, direction: Vec3, part: PartId, spec: PartSpec) -> Option<SurfaceHit> {
    match spec {
        PartSpec::Cuboid(spec) => raycast_cuboid(origin, direction, part, spec),
        PartSpec::Controller(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Engine(spec) => raycast_cuboid(origin, direction, part, spec.cuboid()),
        PartSpec::Cylinder(spec) => raycast_cylinder(origin, direction, part, spec),
    }
}

fn raycast_cylinder(
    origin: Vec3,
    direction: Vec3,
    part: PartId,
    spec: CylinderSpec,
) -> Option<SurfaceHit> {
    let direction = direction.normalize();
    let rotation = spec.pose.rotation.quaternion();
    let inverse = rotation.inverse();
    let local_origin = inverse * (origin - spec.pose.translation());
    let local_direction = inverse * direction;
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let half_length = spec.dimensions.axial_length() * 0.5;
    let mut candidates = Vec::with_capacity(6);

    if local_direction.y.abs() > f32::EPSILON {
        for (y, face) in [
            (half_length, FaceKind::PositiveY),
            (-half_length, FaceKind::NegativeY),
        ] {
            let distance = (y - local_origin.y) / local_direction.y;
            if distance >= 0.0 {
                let point = local_origin + local_direction * distance;
                let radial_squared = point.x.mul_add(point.x, point.z * point.z);
                if radial_squared <= outer * outer + CONTACT_EPSILON
                    && radial_squared >= inner * inner - CONTACT_EPSILON
                    && point_in_cylinder_sweep(point.x, point.z, spec.dimensions)
                {
                    candidates.push((distance, face));
                }
            }
        }
    }
    for radius in [outer, inner] {
        if radius <= 0.0 {
            continue;
        }
        let a = local_direction
            .x
            .mul_add(local_direction.x, local_direction.z * local_direction.z);
        if a <= f32::EPSILON {
            continue;
        }
        let b = 2.0
            * local_origin
                .x
                .mul_add(local_direction.x, local_origin.z * local_direction.z);
        let c = local_origin
            .x
            .mul_add(local_origin.x, local_origin.z * local_origin.z)
            - radius * radius;
        let discriminant = b.mul_add(b, -4.0 * a * c);
        if discriminant < 0.0 {
            continue;
        }
        let root = discriminant.sqrt();
        for distance in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
            if distance >= 0.0 {
                let y = local_origin.y + local_direction.y * distance;
                let point = local_origin + local_direction * distance;
                if y.abs() <= half_length + CONTACT_EPSILON
                    && point_in_cylinder_sweep(point.x, point.z, spec.dimensions)
                {
                    candidates.push((distance, FaceKind::PositiveX));
                }
            }
        }
    }
    if spec.dimensions.sweep_angle_degrees() < 360 {
        let half_sweep = spec.dimensions.sweep_angle_radians() * 0.5;
        for (angle, outward) in [(-half_sweep, -1.0_f32), (half_sweep, 1.0_f32)] {
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let angular = Vec3::new(-angle.sin(), 0.0, angle.cos()) * outward;
            let denominator = local_direction.dot(angular);
            if denominator.abs() <= f32::EPSILON {
                continue;
            }
            let distance = -local_origin.dot(angular) / denominator;
            if distance < 0.0 {
                continue;
            }
            let point = local_origin + local_direction * distance;
            let radius = point.dot(radial);
            if point.y.abs() <= half_length + CONTACT_EPSILON
                && radius >= inner - CONTACT_EPSILON
                && radius <= outer + CONTACT_EPSILON
            {
                candidates.push((distance, FaceKind::PositiveX));
            }
        }
    }
    let (distance, face) = candidates
        .into_iter()
        .min_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))?;
    Some(SurfaceHit {
        distance,
        point: origin + direction * distance,
        face: FaceRef::part(part, face),
    })
}

fn point_in_cylinder_sweep(x: f32, z: f32, dimensions: CylinderDimensions) -> bool {
    dimensions.sweep_angle_degrees() == 360
        || z.atan2(x).abs() <= dimensions.sweep_angle_radians() * 0.5 + CONTACT_EPSILON
}

pub(crate) fn raycast_oriented_cuboid(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: Quat,
    half_extents: Vec3,
) -> Option<OrientedCuboidHit> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction.length_squared() < f32::EPSILON
        || !center.is_finite()
        || !rotation.is_finite()
        || !half_extents.is_finite()
        || half_extents.cmple(Vec3::ZERO).any()
    {
        return None;
    }
    let direction = direction.normalize();
    let inverse_rotation = rotation.inverse();
    let local_origin = inverse_rotation * (origin - center);
    let local_direction = inverse_rotation * direction;
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut hit_axis = 0;
    let mut hit_sign = -1.0;

    for axis in 0..3 {
        if local_direction[axis].abs() <= f32::EPSILON {
            if local_origin[axis] < -half_extents[axis] || local_origin[axis] > half_extents[axis] {
                return None;
            }
            continue;
        }
        let inverse = local_direction[axis].recip();
        let first = (-half_extents[axis] - local_origin[axis]) * inverse;
        let second = (half_extents[axis] - local_origin[axis]) * inverse;
        let axis_near = first.min(second);
        let axis_far = first.max(second);
        if axis_near > near {
            near = axis_near;
            hit_axis = axis;
            hit_sign = if first < second { -1.0 } else { 1.0 };
        }
        far = far.min(axis_far);
        if near > far {
            return None;
        }
    }
    if far < 0.0 {
        return None;
    }
    let distance = near.max(0.0);
    let local_normal = Vec3::from_array(match hit_axis {
        0 => [hit_sign, 0.0, 0.0],
        1 => [0.0, hit_sign, 0.0],
        _ => [0.0, 0.0, hit_sign],
    });
    Some(OrientedCuboidHit {
        distance,
        point: origin + direction * distance,
        local_normal,
    })
}

fn overlap_center(first: FaceGeometry, second: FaceGeometry) -> Option<Vec3> {
    if first.normal.dot(second.normal) > -1.0 + CONTACT_EPSILON
        || (first.center - second.center).dot(first.normal).abs() > CONTACT_EPSILON
    {
        return None;
    }
    profiles_overlap(first, second).then_some((first.center + second.center) * 0.5)
}

fn point_in_profile(u: f32, v: f32, profile: FaceProfile) -> bool {
    match profile {
        FaceProfile::Rectangle { half_u, half_v } => {
            u.abs() <= half_u + CONTACT_EPSILON && v.abs() <= half_v + CONTACT_EPSILON
        }
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => {
            let squared = u.mul_add(u, v * v);
            squared >= (inner_radius - CONTACT_EPSILON).max(0.0).powi(2)
                && squared <= (outer_radius + CONTACT_EPSILON).powi(2)
        }
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => {
            let squared = u.mul_add(u, v * v);
            squared >= (inner_radius - CONTACT_EPSILON).max(0.0).powi(2)
                && squared <= (outer_radius + CONTACT_EPSILON).powi(2)
                && v.atan2(u).abs() <= half_angle + CONTACT_EPSILON
        }
        FaceProfile::Ground => true,
    }
}

fn profiles_overlap(first: FaceGeometry, second: FaceGeometry) -> bool {
    match (first.profile, second.profile) {
        (FaceProfile::Ground, _) | (_, FaceProfile::Ground) => true,
        (FaceProfile::Rectangle { half_u, half_v }, FaceProfile::Rectangle { .. }) => {
            positive_rect_overlap(first, second, first.tangent_u, half_u)
                && positive_rect_overlap(first, second, first.tangent_v, half_v)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Rectangle { .. }) => {
            annulus_rectangle_overlap(first, second)
        }
        (FaceProfile::Rectangle { .. }, FaceProfile::Annulus { .. }) => {
            annulus_rectangle_overlap(second, first)
        }
        (FaceProfile::Annulus { .. }, FaceProfile::Annulus { .. }) => {
            let FaceProfile::Annulus {
                inner_radius: inner_a,
                outer_radius: outer_a,
            } = first.profile
            else {
                unreachable!()
            };
            let FaceProfile::Annulus {
                inner_radius: inner_b,
                outer_radius: outer_b,
            } = second.profile
            else {
                unreachable!()
            };
            let offset = second.center - first.center;
            let distance =
                Vec2::new(offset.dot(first.tangent_u), offset.dot(first.tangent_v)).length();
            distance < outer_a + outer_b - CONTACT_EPSILON
                && distance + outer_a > inner_b + CONTACT_EPSILON
                && distance + outer_b > inner_a + CONTACT_EPSILON
        }
        (FaceProfile::AnnularSector { .. }, _) | (_, FaceProfile::AnnularSector { .. }) => {
            sector_profiles_overlap(first, second)
        }
    }
}

fn sector_profiles_overlap(first: FaceGeometry, second: FaceGeometry) -> bool {
    let first_cells = profile_cells(first, first.center, first.tangent_u, first.tangent_v);
    let second_cells = profile_cells(second, first.center, first.tangent_u, first.tangent_v);
    first_cells.iter().any(|first| {
        second_cells
            .iter()
            .any(|second| convex_polygons_overlap(first, second))
    })
}

fn profile_cells(face: FaceGeometry, origin: Vec3, plane_u: Vec3, plane_v: Vec3) -> Vec<Vec<Vec2>> {
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    match face.profile {
        FaceProfile::Rectangle { half_u, half_v } => vec![vec![
            project(face.center - face.tangent_u * half_u - face.tangent_v * half_v),
            project(face.center + face.tangent_u * half_u - face.tangent_v * half_v),
            project(face.center + face.tangent_u * half_u + face.tangent_v * half_v),
            project(face.center - face.tangent_u * half_u + face.tangent_v * half_v),
        ]],
        FaceProfile::Annulus {
            inner_radius,
            outer_radius,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            inner_radius,
            outer_radius,
            std::f32::consts::PI,
        ),
        FaceProfile::AnnularSector {
            inner_radius,
            outer_radius,
            half_angle,
        } => annular_profile_cells(
            face,
            origin,
            plane_u,
            plane_v,
            inner_radius,
            outer_radius,
            half_angle,
        ),
        FaceProfile::Ground => Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn annular_profile_cells(
    face: FaceGeometry,
    origin: Vec3,
    plane_u: Vec3,
    plane_v: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    half_angle: f32,
) -> Vec<Vec<Vec2>> {
    let sweep = half_angle * 2.0;
    let segment_count = (1_u16..=24)
        .find(|&count| (f32::from(count) * (std::f32::consts::PI / 12.0) - sweep).abs() < 1.0e-4)
        .expect("annular profiles use 15-degree increments");
    let project = |point: Vec3| {
        let offset = point - origin;
        Vec2::new(offset.dot(plane_u), offset.dot(plane_v))
    };
    (0..segment_count)
        .map(|segment| {
            let first_angle = -half_angle + sweep * f32::from(segment) / f32::from(segment_count);
            let second_angle =
                -half_angle + sweep * f32::from(segment + 1) / f32::from(segment_count);
            let radial = |angle: f32| face.tangent_u * angle.cos() + face.tangent_v * angle.sin();
            let outer_first = project(face.center + radial(first_angle) * outer_radius);
            let outer_second = project(face.center + radial(second_angle) * outer_radius);
            if inner_radius == 0.0 {
                vec![project(face.center), outer_first, outer_second]
            } else {
                vec![
                    project(face.center + radial(first_angle) * inner_radius),
                    outer_first,
                    outer_second,
                    project(face.center + radial(second_angle) * inner_radius),
                ]
            }
        })
        .collect()
}

fn convex_polygons_overlap(first: &[Vec2], second: &[Vec2]) -> bool {
    first
        .iter()
        .zip(first.iter().cycle().skip(1))
        .chain(second.iter().zip(second.iter().cycle().skip(1)))
        .all(|(start, end)| {
            let edge = *end - *start;
            let axis = Vec2::new(-edge.y, edge.x).normalize();
            let project = |polygon: &[Vec2]| {
                polygon.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY),
                    |(minimum, maximum), point| {
                        let value = point.dot(axis);
                        (minimum.min(value), maximum.max(value))
                    },
                )
            };
            let (first_minimum, first_maximum) = project(first);
            let (second_minimum, second_maximum) = project(second);
            first_maximum.min(second_maximum) - first_minimum.max(second_minimum) > CONTACT_EPSILON
        })
}

fn positive_rect_overlap(
    first: FaceGeometry,
    second: FaceGeometry,
    axis: Vec3,
    first_half: f32,
) -> bool {
    let FaceProfile::Rectangle { half_u, half_v } = second.profile else {
        unreachable!()
    };
    let second_half =
        second.tangent_u.dot(axis).abs() * half_u + second.tangent_v.dot(axis).abs() * half_v;
    first_half + second_half - (second.center - first.center).dot(axis).abs() > CONTACT_EPSILON
}

fn annulus_rectangle_overlap(annulus: FaceGeometry, rectangle: FaceGeometry) -> bool {
    let FaceProfile::Annulus {
        inner_radius,
        outer_radius,
    } = annulus.profile
    else {
        unreachable!()
    };
    let FaceProfile::Rectangle { half_u, half_v } = rectangle.profile else {
        unreachable!()
    };
    let offset = annulus.center - rectangle.center;
    let center_u = offset.dot(rectangle.tangent_u).abs();
    let center_v = offset.dot(rectangle.tangent_v).abs();
    let nearest_u = (center_u - half_u).max(0.0);
    let nearest_v = (center_v - half_v).max(0.0);
    let nearest_squared = nearest_u.mul_add(nearest_u, nearest_v * nearest_v);
    let farthest_u = center_u + half_u;
    let farthest_v = center_v + half_v;
    let farthest_squared = farthest_u.mul_add(farthest_u, farthest_v * farthest_v);
    nearest_squared < (outer_radius - CONTACT_EPSILON).max(0.0).powi(2)
        && farthest_squared > (inner_radius + CONTACT_EPSILON).powi(2)
}

fn rotation_y_to_normal(normal: Vec3) -> GridRotation {
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

fn cuboid_world_bounds(spec: CuboidSpec) -> (Vec3, Vec3) {
    let rotation = Mat3::from_quat(spec.pose.rotation.quaternion());
    let half = spec.size_meters() * 0.5;
    let world_half = Vec3::new(
        rotation.x_axis.x.abs() * half.x
            + rotation.y_axis.x.abs() * half.y
            + rotation.z_axis.x.abs() * half.z,
        rotation.x_axis.y.abs() * half.x
            + rotation.y_axis.y.abs() * half.y
            + rotation.z_axis.y.abs() * half.z,
        rotation.x_axis.z.abs() * half.x
            + rotation.y_axis.z.abs() * half.y
            + rotation.z_axis.z.abs() * half.z,
    );
    let center = spec.pose.translation();
    (center - world_half, center + world_half)
}

pub(crate) fn part_world_bounds(spec: PartSpec) -> (Vec3, Vec3) {
    match spec {
        PartSpec::Cuboid(spec) => cuboid_world_bounds(spec),
        PartSpec::Controller(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Engine(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Cylinder(spec) => {
            let rotation = Mat3::from_quat(spec.pose.rotation.quaternion());
            let (local_minimum, local_maximum) = cylinder_local_bounds(spec.dimensions);
            let mut world_minimum = Vec3::splat(f32::INFINITY);
            let mut world_maximum = Vec3::splat(f32::NEG_INFINITY);
            for x in [local_minimum.x, local_maximum.x] {
                for y in [local_minimum.y, local_maximum.y] {
                    for z in [local_minimum.z, local_maximum.z] {
                        let point = spec.pose.translation() + rotation * Vec3::new(x, y, z);
                        world_minimum = world_minimum.min(point);
                        world_maximum = world_maximum.max(point);
                    }
                }
            }
            (world_minimum, world_maximum)
        }
    }
}

fn cylinder_local_bounds(dimensions: CylinderDimensions) -> (Vec3, Vec3) {
    let outer = dimensions.outer_diameter() * 0.5;
    let inner = dimensions.inner_diameter() * 0.5;
    let half_length = dimensions.axial_length() * 0.5;
    if dimensions.sweep_angle_degrees() == 360 {
        return (
            Vec3::new(-outer, -half_length, -outer),
            Vec3::new(outer, half_length, outer),
        );
    }

    let half_sweep = dimensions.sweep_angle_radians() * 0.5;
    let mut minimum = Vec3::new(f32::INFINITY, -half_length, f32::INFINITY);
    let mut maximum = Vec3::new(f32::NEG_INFINITY, half_length, f32::NEG_INFINITY);
    for angle in [
        -half_sweep,
        half_sweep,
        -std::f32::consts::FRAC_PI_2,
        0.0,
        std::f32::consts::FRAC_PI_2,
    ] {
        if angle.abs() > half_sweep + CONTACT_EPSILON {
            continue;
        }
        for radius in [inner, outer] {
            let point = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
            minimum = minimum.min(point);
            maximum = maximum.max(point);
        }
    }
    (minimum, maximum)
}

#[derive(Clone, Copy)]
struct CollisionBox {
    center: Vec3,
    rotation: Quat,
    half: Vec3,
}

fn parts_overlap(first: PartSpec, second: PartSpec) -> bool {
    part_collision_boxes(first).into_iter().any(|first| {
        part_collision_boxes(second)
            .into_iter()
            .any(|second| boxes_overlap(first, second))
    })
}

fn part_collision_boxes(spec: PartSpec) -> Vec<CollisionBox> {
    match spec {
        PartSpec::Controller(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Engine(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Cuboid(spec) => vec![CollisionBox {
            center: spec.pose.translation(),
            rotation: spec.pose.rotation.quaternion(),
            half: spec.size_meters() * 0.5,
        }],
        PartSpec::Cylinder(spec) => {
            let outer = spec.dimensions.outer_diameter() * 0.5;
            let inner = spec.dimensions.inner_diameter() * 0.5;
            let radial_half = (outer - inner) * 0.5;
            let center_radius = (outer + inner) * 0.5;
            let sweep = spec.dimensions.sweep_angle_radians();
            let segment_angle = sweep / 16.0;
            let tangent_half = outer * (segment_angle * 0.5).tan();
            let start_angle = if spec.dimensions.sweep_angle_degrees() == 360 {
                -segment_angle * 0.5
            } else {
                -sweep * 0.5
            };
            let rotation = spec.pose.rotation.quaternion();
            (0_u16..16)
                .map(|segment| {
                    let angle = start_angle + segment_angle * (f32::from(segment) + 0.5);
                    let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
                    CollisionBox {
                        center: spec.pose.translation() + rotation * (radial * center_radius),
                        rotation: rotation * Quat::from_rotation_y(-angle),
                        half: Vec3::new(
                            radial_half,
                            spec.dimensions.axial_length() * 0.5,
                            tangent_half,
                        ),
                    }
                })
                .collect()
        }
    }
}

fn boxes_overlap(first: CollisionBox, second: CollisionBox) -> bool {
    let first_axes = [
        first.rotation * Vec3::X,
        first.rotation * Vec3::Y,
        first.rotation * Vec3::Z,
    ];
    let second_axes = [
        second.rotation * Vec3::X,
        second.rotation * Vec3::Y,
        second.rotation * Vec3::Z,
    ];
    let offset = second.center - first.center;
    let separates = |axis: Vec3| {
        if axis.length_squared() <= 1.0e-10 {
            return false;
        }
        let axis = axis.normalize();
        let radius = |axes: [Vec3; 3], half: Vec3| {
            axes[0].dot(axis).abs() * half.x
                + axes[1].dot(axis).abs() * half.y
                + axes[2].dot(axis).abs() * half.z
        };
        offset.dot(axis).abs()
            >= radius(first_axes, first.half) + radius(second_axes, second.half) - CONTACT_EPSILON
    };
    if first_axes.into_iter().chain(second_axes).any(separates) {
        return false;
    }
    for first_axis in first_axes {
        for second_axis in second_axes {
            if separates(first_axis.cross(second_axis)) {
                return false;
            }
        }
    }
    true
}

fn bounds_overlap_interior(
    first_minimum: Vec3,
    first_maximum: Vec3,
    second_minimum: Vec3,
    second_maximum: Vec3,
) -> bool {
    (first_minimum.x < second_maximum.x - CONTACT_EPSILON
        && first_maximum.x > second_minimum.x + CONTACT_EPSILON)
        && (first_minimum.y < second_maximum.y - CONTACT_EPSILON
            && first_maximum.y > second_minimum.y + CONTACT_EPSILON)
        && (first_minimum.z < second_maximum.z - CONTACT_EPSILON
            && first_maximum.z > second_minimum.z + CONTACT_EPSILON)
}

fn cardinal_axis(normal: Vec3) -> (usize, i32) {
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

fn face_for_normal(normal: Vec3) -> FaceKind {
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

fn snap_cardinal(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.round(), vector.y.round(), vector.z.round())
}

#[cfg(test)]
mod tests {
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        CuboidSpec, CylinderDimensions, CylinderSpec, EngineKind, FaceKind, FaceOwner, FaceRef,
        GridRotation, PartSpec, PendingOperation, RigidLinkSpec, WeldSpec,
    };

    use super::{
        BLOCK_SIZE_METERS, PlacementCandidate, PlacementError, PlacementPlane, SurfaceHit,
        bearing_anchor_from_hit, bearing_attachment_candidate, bearing_overlaps_candidate,
        bearing_ring_overlaps_face, bearing_support_face, begin_weld, block_sheet_specs,
        candidate_from_hit, cuboid_candidate_from_hit, cylinder_candidate_from_hit,
        face_geometry_from_ref, oriented_cuboid_candidate_from_hit, raycast_construction,
        raycast_construction_for_annulus, raycast_placement_plane, rigid_body_parts,
        stage_bearing_attachment, stage_bearing_block_batch, stage_block_batch,
        stage_block_batch_from_source, stage_cuboid, stage_cylinder_from_source,
        stage_engine_from_source, stage_weld_objects, validate_part,
    };

    fn spawn_cube(graph: &mut ConstructionGraph, units: IVec3, size: u8) -> mechanic_core::PartId {
        let spec =
            CuboidSpec::new([size; 3], BuildPose::new(units, GridRotation::default())).unwrap();
        let Ok(BuildOutcome::Spawned(part)) = graph.apply(BuildCommand::Spawn(spec)) else {
            panic!("cube must spawn");
        };
        part
    }

    fn spawn_cylinder(
        graph: &mut ConstructionGraph,
        dimensions: CylinderDimensions,
        pose: BuildPose,
    ) -> mechanic_core::PartId {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                dimensions, pose,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        part
    }

    #[test]
    fn rays_pass_through_cylinder_bores_but_hit_annular_material() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);
        let through_bore =
            raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(through_bore.face.owner, FaceOwner::Part(cube));
        let annulus = raycast_construction(&graph, Vec3::new(0.4, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(annulus.face.owner, FaceOwner::Part(cylinder));
    }

    #[test]
    fn cylinder_sector_raycast_hits_retained_caps_and_cut_walls_only() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let cylinder = spawn_cylinder(
            &mut graph,
            dimensions,
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 4);

        let retained = raycast_construction(&graph, Vec3::new(0.3, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(retained.face.owner, FaceOwner::Part(cylinder));
        assert_eq!(retained.face.face, FaceKind::PositiveY);

        let missing = raycast_construction(&graph, Vec3::new(-0.3, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        assert_eq!(missing.face.owner, FaceOwner::Part(cube));

        let cut_wall = raycast_construction(&graph, Vec3::new(0.3, 2.0, 2.0), Vec3::NEG_Z).unwrap();
        assert_eq!(cut_wall.face.owner, FaceOwner::Part(cylinder));
        assert_eq!(cut_wall.face.face, FaceKind::PositiveX);
    }

    #[test]
    fn annular_placement_stops_at_a_bore_only_when_material_cannot_pass() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let cube = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);
        let origin = Vec3::new(0.0, 5.0, 0.0);

        let fitting =
            raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 0.0, 0.4).unwrap();
        assert_eq!(fitting.face.owner, FaceOwner::Part(cube));

        let obstructed =
            raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 0.0, 0.6).unwrap();
        assert_eq!(obstructed.face.owner, FaceOwner::Part(cylinder));
        assert_eq!(obstructed.face.face, FaceKind::PositiveY);
        let candidate = cylinder_candidate_from_hit(
            &graph,
            obstructed,
            CylinderDimensions::new(0.6, 0.0, 0.25).unwrap(),
        )
        .unwrap();
        let staged = stage_cylinder_from_source(&graph, candidate, obstructed.face.owner).unwrap();
        assert_eq!(staged.weld_count(), 1);

        let surrounding_sleeve =
            raycast_construction_for_annulus(&graph, origin, Vec3::NEG_Y, 1.1, 1.2).unwrap();
        assert_eq!(surrounding_sleeve.face.owner, FaceOwner::Part(cube));
    }

    #[test]
    fn bearing_can_center_over_a_bore_when_its_ring_has_support() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 8, 0), GridRotation::default()),
        );
        let source = FaceRef::part(cylinder, FaceKind::PositiveY);
        let hit = SurfaceHit {
            distance: 2.5,
            point: Vec3::new(0.0, 2.5, 0.0),
            face: source,
        };

        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert_eq!(anchor, hit.point);
        assert_eq!(
            bearing_support_face(
                &graph,
                source,
                anchor,
                BearingDimensions::new(0.6, 0.2).unwrap(),
            ),
            Some(source)
        );
        assert!(
            bearing_support_face(
                &graph,
                source,
                anchor,
                BearingDimensions::new(0.4, 0.2).unwrap(),
            )
            .is_none()
        );
    }

    #[test]
    fn small_blocks_can_occupy_a_large_cylinder_bore() {
        let mut graph = ConstructionGraph::new();
        spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.6, 1.0).unwrap(),
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        );
        let cube = CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let candidate = PlacementCandidate {
            spec: cube,
            attached_face: FaceKind::NegativeY,
            anchor: Some(Vec3::ZERO),
        };
        assert!(stage_cuboid(&graph, candidate).is_ok());
    }

    #[test]
    fn blocks_can_occupy_the_missing_side_of_a_cylinder_sector() {
        let mut graph = ConstructionGraph::new();
        spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(1.0, 0.0, 1.0)
                .unwrap()
                .with_sweep_angle_degrees(90)
                .unwrap(),
            BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
        );
        let cube = CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(-1, 4, 0), GridRotation::default()),
        )
        .unwrap();
        let candidate = PlacementCandidate {
            spec: cube,
            attached_face: FaceKind::NegativeY,
            anchor: Some(Vec3::ZERO),
        };

        assert!(stage_cuboid(&graph, candidate).is_ok());
    }

    #[test]
    fn cylinders_place_along_all_six_flat_face_normals() {
        let cases = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        for outward in cases {
            let mut graph = ConstructionGraph::new();
            let support = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
            let hit = super::raycast_cuboid(
                Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                -outward,
                support,
                graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            )
            .unwrap();
            let candidate = cylinder_candidate_from_hit(
                &graph,
                hit,
                CylinderDimensions::new(0.25, 0.0, 0.5).unwrap(),
            )
            .unwrap();
            let axis = candidate.spec.pose.rotation.quaternion() * Vec3::Y;
            assert!(axis.abs_diff_eq(outward, 1.0e-6));
            assert!(stage_cylinder_from_source(&graph, candidate, hit.face.owner).is_ok());
        }
    }

    #[test]
    fn thin_annular_cylinder_places_on_a_coplanar_block_sheet() {
        let mut graph = ConstructionGraph::new();
        let mut center = None;
        for x in -2..=2 {
            for z in -2..=2 {
                let spec = CuboidSpec::new(
                    [1; 3],
                    BuildPose::from_half_grid(IVec3::new(x * 2, 1, z * 2), GridRotation::default()),
                )
                .unwrap();
                let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                if x == 0 && z == 0 {
                    center = Some(part);
                }
            }
        }
        let center = center.unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 0.25, 0.0),
            face: FaceRef::part(center, FaceKind::PositiveY),
        };
        let candidate = cylinder_candidate_from_hit(
            &graph,
            hit,
            CylinderDimensions::new(0.75, 0.70, 0.25).unwrap(),
        )
        .unwrap();

        assert!(candidate.anchor.is_some());
        assert!(stage_cylinder_from_source(&graph, candidate, hit.face.owner).is_ok());
    }

    #[test]
    fn bearing_anchor_rejects_a_curved_cylinder_wall() {
        let mut graph = ConstructionGraph::new();
        let cylinder = spawn_cylinder(
            &mut graph,
            CylinderDimensions::new(0.5, 0.25, 0.5).unwrap(),
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        );
        let curved_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.25, 0.5, 0.0),
            face: FaceRef::part(cylinder, FaceKind::PositiveX),
        };

        assert_eq!(
            bearing_anchor_from_hit(&graph, curved_hit),
            Err(PlacementError::CurvedSurface)
        );
    }

    #[test]
    fn raycast_selects_nearest_cuboid_face_before_ground() {
        let mut graph = ConstructionGraph::new();
        spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = raycast_construction(&graph, Vec3::new(0.0, 4.0, 0.0), Vec3::NEG_Y)
            .expect("cube is under ray");
        assert_eq!(hit.face.face, FaceKind::PositiveY);
        assert!(matches!(hit.face.owner, FaceOwner::Part(_)));
        assert!((hit.point.y - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn raycast_from_below_ignores_the_floor_and_reaches_the_underside() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 4, 0), 1);

        let hit = raycast_construction(&graph, Vec3::new(0.0, -1.0, 0.0), Vec3::Y)
            .expect("the ray reaches the elevated block");

        assert_eq!(hit.face.owner, FaceOwner::Part(part));
        assert_eq!(hit.face.face, FaceKind::NegativeY);
        assert!((hit.point.y - 0.875).abs() < 1.0e-6);
    }

    #[test]
    fn fixed_quarter_metre_blocks_place_flush_on_ground_and_faces() {
        let graph = ConstructionGraph::new();
        let ground_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, ground_hit);
        assert_eq!(
            candidate
                .spec
                .dimensions
                .map(mechanic_core::GridDimension::units),
            [1; 3]
        );
        assert!((candidate.spec.pose.translation().y - BLOCK_SIZE_METERS * 0.5).abs() < 1.0e-6);
        let graph = stage_cuboid(&graph, candidate).unwrap();

        let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
            .expect("placed block is under ray");
        let attached = candidate_from_hit(&graph, top);
        assert!((attached.spec.pose.translation().y - 0.375).abs() < 1.0e-6);
        assert!(stage_cuboid(&graph, attached).is_ok());
    }

    #[test]
    fn gas_engine_places_flush_with_its_authored_footprint_and_stays_semantic() {
        let graph = ConstructionGraph::new();
        let ground_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = cuboid_candidate_from_hit(&graph, ground_hit, EngineKind::Gas.grid_units());

        assert_eq!(
            candidate.spec.pose.translation(),
            Vec3::new(0.125, 0.25, 0.0)
        );
        assert_eq!(candidate.spec.size_meters(), Vec3::new(0.5, 0.5, 0.75));

        let graph = stage_engine_from_source(&graph, candidate, FaceOwner::Ground, EngineKind::Gas)
            .unwrap();
        let (_, part) = graph.parts().next().expect("the engine was staged");
        assert!(matches!(
            part,
            PartSpec::Engine(engine) if engine.kind == EngineKind::Gas
        ));
        assert_eq!(graph.welds().count(), 1);
    }

    #[test]
    fn electric_engine_spans_two_by_two_ground_cells_without_a_half_block_offset() {
        let graph = ConstructionGraph::new();
        let candidate = cuboid_candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
            EngineKind::Electric.grid_units(),
        );

        assert_eq!(
            candidate.spec.pose.translation_half_units(),
            IVec3::new(1, 2, 1)
        );
        assert_eq!(candidate.spec.size_meters(), Vec3::splat(0.5));
        assert_eq!(
            super::cuboid_world_bounds(candidate.spec),
            (Vec3::new(-0.125, 0.0, -0.125), Vec3::new(0.375, 0.5, 0.375))
        );
    }

    #[test]
    fn quarter_turn_rotates_an_authored_footprint_and_survives_staging() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate =
            oriented_cuboid_candidate_from_hit(&graph, hit, EngineKind::Gas.grid_units(), 1);

        assert_eq!(candidate.spec.pose.rotation.quarter_turns_xyz(), [0, 1, 0]);
        assert_eq!(
            candidate.spec.pose.translation_half_units(),
            IVec3::new(0, 2, 1)
        );
        assert_eq!(candidate.attached_face, FaceKind::NegativeY);
        let (minimum, maximum) = super::cuboid_world_bounds(candidate.spec);
        assert!((maximum.x - minimum.x - 0.75).abs() < 1.0e-6);
        assert!((maximum.z - minimum.z - 0.50).abs() < 1.0e-6);

        let staged =
            stage_engine_from_source(&graph, candidate, FaceOwner::Ground, EngineKind::Gas)
                .unwrap();
        let (_, PartSpec::Engine(engine)) = staged.parts().next().unwrap() else {
            panic!("the staged part must remain an engine")
        };
        assert_eq!(engine.pose.rotation.quarter_turns_xyz(), [0, 1, 0]);
    }

    #[test]
    fn rotated_authored_parts_attach_flush_from_every_world_face() {
        for outward in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            let mut graph = ConstructionGraph::new();
            let support = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
            let hit = super::raycast_cuboid(
                Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                -outward,
                support,
                graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            )
            .expect("ray reaches requested support face");
            let candidate =
                oriented_cuboid_candidate_from_hit(&graph, hit, EngineKind::Gas.grid_units(), 1);
            let attached = super::face_geometry(candidate.spec, candidate.attached_face);

            assert!(attached.normal.abs_diff_eq(-outward, 1.0e-6));
            assert!(
                stage_engine_from_source(
                    &graph,
                    candidate,
                    FaceOwner::Part(support),
                    EngineKind::Gas,
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn placement_works_from_all_six_cuboid_faces() {
        let cases = [
            (Vec3::X, FaceKind::PositiveX),
            (Vec3::NEG_X, FaceKind::NegativeX),
            (Vec3::Y, FaceKind::PositiveY),
            (Vec3::NEG_Y, FaceKind::NegativeY),
            (Vec3::Z, FaceKind::PositiveZ),
            (Vec3::NEG_Z, FaceKind::NegativeZ),
        ];
        for (outward, expected_face) in cases {
            let mut graph = ConstructionGraph::new();
            let part = spawn_cube(&mut graph, IVec3::new(0, 16, 0), 4);
            let hit = super::raycast_cuboid(
                Vec3::new(0.0, 4.0, 0.0) + outward * 5.0,
                -outward,
                part,
                graph.part(part).copied().unwrap().as_cuboid().unwrap(),
            )
            .expect("ray reaches requested face");
            assert_eq!(hit.face.face, expected_face);
            let candidate = candidate_from_hit(&graph, hit);
            assert!(stage_cuboid(&graph, candidate).is_ok());
        }
    }

    #[test]
    fn side_placement_preserves_the_supporting_quarter_block_lattice() {
        for face in [
            FaceKind::PositiveX,
            FaceKind::NegativeX,
            FaceKind::PositiveZ,
            FaceKind::NegativeZ,
        ] {
            let graph = ConstructionGraph::new();
            let support = candidate_from_hit(
                &graph,
                SurfaceHit {
                    distance: 1.0,
                    point: Vec3::ZERO,
                    face: FaceRef::ground(),
                },
            );
            let graph = stage_cuboid(&graph, support).unwrap();
            let part = graph.parts().next().unwrap().0;
            let source = FaceRef::part(part, face);
            let source_face = super::face_geometry_from_ref(source, Some(&graph));

            let candidate = candidate_from_hit(
                &graph,
                SurfaceHit {
                    distance: 1.0,
                    point: source_face.center,
                    face: source,
                },
            );

            assert_eq!(
                candidate.spec.pose.translation_half_units().y,
                support.spec.pose.translation_half_units().y
            );
            assert!(stage_cuboid(&graph, candidate).is_ok());

            let bearing_candidate =
                bearing_attachment_candidate(&graph, source, source_face.center);
            assert_eq!(
                bearing_candidate.spec.pose.translation_half_units().y,
                support.spec.pose.translation_half_units().y
            );
            let attached = stage_bearing_attachment(
                &graph,
                bearing_candidate,
                source,
                source_face.center,
                BearingDimensions::default(),
            )
            .unwrap();
            assert_eq!(attached.bearing_count(), 1);
            assert_eq!(attached.weld_count(), 1);
        }
    }

    #[test]
    fn placement_rejects_cubes_extending_beyond_platform() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(super::GROUND_HALF_SIZE, 0.0, 0.0),
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        assert!(matches!(
            stage_cuboid(&graph, candidate),
            Err(PlacementError::OutsidePlatform)
        ));
    }

    #[test]
    fn cylinder_slice_platform_bounds_ignore_the_omitted_sector() {
        let graph = ConstructionGraph::new();
        let pose = BuildPose::new(IVec3::new(-40, 2, 0), GridRotation::default());
        let slice = CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        assert!(validate_part(&graph, PartSpec::Cylinder(CylinderSpec::new(slice, pose))).is_ok());

        let full = CylinderDimensions::new(1.0, 0.0, 1.0).unwrap();
        assert!(matches!(
            validate_part(&graph, PartSpec::Cylinder(CylinderSpec::new(full, pose))),
            Err(PlacementError::OutsidePlatform)
        ));
    }

    #[test]
    fn single_block_automatically_welds_to_touching_block() {
        let mut graph = ConstructionGraph::new();
        spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
            .expect("support block is under ray");
        let candidate = candidate_from_hit(&graph, hit);

        let graph = stage_cuboid(&graph, candidate).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.weld_count(), 1);
        assert_eq!(graph.compile().unwrap().compounds.len(), 1);
    }

    #[test]
    fn single_block_placed_on_ground_is_automatically_welded() {
        let graph = ConstructionGraph::new();
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );

        let graph = stage_cuboid(&graph, candidate).unwrap();

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.weld_count(), 1);
        assert!(graph.compile().unwrap().compounds[0].is_static);
    }

    #[test]
    fn dragged_sheet_is_face_connected_and_welded() {
        let graph = ConstructionGraph::new();
        let start = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let endpoint = start.spec.pose.translation_half_units() + IVec3::new(4, 0, 2);
        let specs = block_sheet_specs(start.spec, endpoint, PlacementPlane::Xz).unwrap();

        let graph = stage_block_batch(&graph, start, &specs).unwrap();

        assert_eq!(graph.part_count(), 6);
        assert_eq!(graph.weld_count(), 13);
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert!(compiled.compounds[0].is_static);
    }

    #[test]
    fn invalid_drag_batch_preserves_graph() {
        let graph = ConstructionGraph::new();
        let start = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let endpoint = start.spec.pose.translation_half_units() + IVec3::new(96, 0, 0);
        let specs = block_sheet_specs(start.spec, endpoint, PlacementPlane::Xz).unwrap();

        assert!(matches!(
            stage_block_batch(&graph, start, &specs),
            Err(PlacementError::OutsidePlatform)
        ));
        assert_eq!(graph.part_count(), 0);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn drag_plane_projection_and_cycle_are_deterministic() {
        let graph = ConstructionGraph::new();
        let start = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let endpoint = raycast_placement_plane(
            Vec3::new(2.0, 5.0, 3.0),
            Vec3::NEG_Y,
            start.spec,
            PlacementPlane::Xz,
        )
        .unwrap();

        assert_eq!(endpoint, IVec3::new(16, 1, 24));
        assert_eq!(PlacementPlane::Xz.cycle(), PlacementPlane::Xy);
        assert_eq!(PlacementPlane::Xy.cycle(), PlacementPlane::Yz);
        assert_eq!(PlacementPlane::Yz.cycle(), PlacementPlane::Xz);
    }

    #[test]
    fn weld_selects_two_objects_without_spawning_a_part() {
        let mut graph = ConstructionGraph::new();
        let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let right = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
        begin_weld(&mut graph, FaceRef::part(left, FaceKind::PositiveY)).unwrap();
        assert!(matches!(graph.pending(), Some(PendingOperation::Weld(_))));

        let graph =
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(right)).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.weld_count(), 1);
        assert!(graph.pending().is_none());
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert_eq!(compiled.compounds[0].source_parts.len(), 2);
    }

    #[test]
    fn weld_to_ground_resolves_contact_across_the_selected_rigid_body() {
        let mut graph = ConstructionGraph::new();
        let parts = [IVec3::new(0, 1, 0), IVec3::new(0, 3, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let [bottom, top] = parts;
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(bottom, FaceKind::PositiveY),
                second: FaceRef::part(top, FaceKind::NegativeY),
            }))
            .unwrap();

        let grounded = stage_weld_objects(&graph, FaceOwner::Part(top), FaceOwner::Ground).unwrap();

        assert_eq!(grounded.weld_count(), 2);
        let compiled = grounded.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 1);
        assert!(compiled.compounds[0].is_static);
    }

    #[test]
    fn weld_rejects_same_or_separated_objects_without_mutation() {
        let mut graph = ConstructionGraph::new();
        let left = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let far = spawn_cube(&mut graph, IVec3::new(8, 2, 0), 4);
        assert!(matches!(
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(left)),
            Err(PlacementError::SameObject)
        ));
        assert!(matches!(
            stage_weld_objects(&graph, FaceOwner::Part(left), FaceOwner::Part(far)),
            Err(PlacementError::ObjectsDoNotTouch)
        ));
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn bearing_anchor_snaps_without_mutating_the_graph() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.18, 1.0, -0.18),
            face: FaceRef::part(base, FaceKind::PositiveY),
        };
        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert_eq!(anchor, Vec3::new(0.25, 1.0, -0.25));

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert!(graph.pending().is_none());
    }

    #[test]
    fn bearing_anchor_preserves_a_side_faces_half_grid_height() {
        let graph = ConstructionGraph::new();
        let block = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let graph = stage_cuboid(&graph, block).unwrap();
        let part = graph.parts().next().unwrap().0;
        let source = FaceRef::part(part, FaceKind::PositiveX);
        let face = face_geometry_from_ref(source, Some(&graph));
        let hit = SurfaceHit {
            distance: 1.0,
            point: face.center + Vec3::new(0.0, 0.02, 0.10),
            face: source,
        };

        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();

        assert!(anchor.abs_diff_eq(face.center, 1.0e-6));
    }

    #[test]
    fn bearing_second_click_attaches_a_cuboid_without_collider_geometry_for_connector() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.0, 1.0, 0.0),
            face: source,
        };
        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        let candidate = bearing_attachment_candidate(&graph, source, anchor);

        let dimensions = BearingDimensions::new(0.75, 0.25).unwrap();
        let graph =
            stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.bearings().next().unwrap().1.shared_anchor, anchor);
        assert_eq!(graph.bearings().next().unwrap().1.dimensions, dimensions);
        assert!(graph.pending().is_none());
    }

    #[test]
    fn oversized_bearing_attaches_to_any_block_face_overlapped_by_its_ring() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::Y;
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.36, 1.0, 0.0),
                face: source,
            },
        );

        assert!(
            !super::face_geometry(candidate.spec, candidate.attached_face)
                .center
                .abs_diff_eq(anchor, 1.0e-5)
        );
        assert!(bearing_overlaps_candidate(
            &graph, source, anchor, dimensions, candidate,
        ));
        let attached =
            stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

        assert_eq!(attached.bearing_count(), 1);
        assert_eq!(attached.weld_count(), 0);
        assert_eq!(attached.bearings().next().unwrap().1.dimensions, dimensions);
    }

    #[test]
    fn bearing_overhang_claims_a_block_placed_on_an_adjacent_support_face() {
        let mut graph = ConstructionGraph::new();
        let source_part = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let adjacent_part = spawn_cube(&mut graph, IVec3::new(4, 2, 0), 4);
        let source = FaceRef::part(source_part, FaceKind::PositiveY);
        let adjacent_face = FaceRef::part(adjacent_part, FaceKind::PositiveY);
        let anchor = Vec3::Y;
        let dimensions = BearingDimensions::new(2.40, 0.10).unwrap();
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::new(1.0, 1.0, 0.0),
                face: adjacent_face,
            },
        );

        assert!(bearing_overlaps_candidate(
            &graph, source, anchor, dimensions, candidate,
        ));
        let attached =
            stage_bearing_attachment(&graph, candidate, source, anchor, dimensions).unwrap();

        assert_eq!(attached.bearing_count(), 1);
        assert_eq!(attached.weld_count(), 0);
        assert_eq!(attached.compile().unwrap().compounds.len(), 3);
    }

    #[test]
    fn block_face_entirely_inside_bearing_hole_is_not_covered() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let candidate = bearing_attachment_candidate(&graph, source, Vec3::Y);

        assert!(!bearing_overlaps_candidate(
            &graph,
            source,
            Vec3::Y,
            BearingDimensions::new(1.0, 0.50).unwrap(),
            candidate,
        ));
    }

    #[test]
    fn large_hollow_bearing_uses_a_ring_block_instead_of_the_center_block() {
        let mut graph = ConstructionGraph::new();
        let mut center = None;
        for x in -1..=1 {
            for z in -1..=1 {
                let spec = CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_half_grid(IVec3::new(x * 2, 1, z * 2), GridRotation::default()),
                )
                .unwrap();
                let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                else {
                    unreachable!()
                };
                if x == 0 && z == 0 {
                    center = Some(part);
                }
            }
        }
        let center = center.unwrap();
        let selected = FaceRef::part(center, FaceKind::PositiveY);
        let dimensions = BearingDimensions::new(0.75, 0.40).unwrap();

        let support =
            bearing_support_face(&graph, selected, Vec3::new(0.0, 0.25, 0.0), dimensions).unwrap();

        assert_ne!(support.owner, FaceOwner::Part(center));
        assert!(bearing_ring_overlaps_face(
            Vec3::new(0.0, 0.25, 0.0),
            dimensions,
            face_geometry_from_ref(support, Some(&graph)),
        ));
    }

    #[test]
    fn placement_from_bearing_body_welds_only_to_the_clicked_rigid_group() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::ZERO, 1);
        let attached = spawn_cube(&mut graph, IVec3::new(0, 1, 0), 1);
        let sibling = spawn_cube(&mut graph, IVec3::new(-1, 1, 0), 1);
        let neighbour = spawn_cube(&mut graph, IVec3::new(1, 2, 0), 1);
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        for target in [attached, sibling] {
            graph
                .apply(BuildCommand::AddBearing(
                    BearingSpec::new(
                        FaceRef::part(base, FaceKind::PositiveY),
                        FaceRef::part(target, FaceKind::NegativeY),
                        Vec3::new(0.0, 0.125, 0.0),
                        Vec3::Y,
                    )
                    .with_dimensions(dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: attached,
                second: sibling,
            }))
            .unwrap();
        let source = FaceRef::part(attached, FaceKind::PositiveY);
        let source_face = face_geometry_from_ref(source, Some(&graph));
        let candidate = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 0.0,
                point: source_face.center,
                face: source,
            },
        );

        let staged =
            stage_block_batch_from_source(&graph, candidate, &[candidate.spec], source.owner)
                .unwrap();
        let placed = staged
            .parts()
            .find_map(|(part, _)| graph.part(part).is_none().then_some(part))
            .unwrap();
        let attached_group = rigid_body_parts(&staged, attached);

        assert!(attached_group.contains(&placed));
        assert!(attached_group.contains(&sibling));
        assert!(!attached_group.contains(&neighbour));
        assert!(!attached_group.contains(&base));
        assert_eq!(staged.bearing_count(), 2);
        assert_eq!(staged.compile().unwrap().bearings.len(), 1);
    }

    #[test]
    fn one_bearing_groups_multiple_direct_attachments_into_one_rotor() {
        let mut graph = ConstructionGraph::new();
        let support = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(support, FaceKind::PositiveY);
        let dimensions = BearingDimensions::new(0.80, 0.10).unwrap();
        let candidates = [0.0, 0.25].map(|x| {
            candidate_from_hit(
                &graph,
                SurfaceHit {
                    distance: 0.0,
                    point: Vec3::new(x, 1.0, 0.0),
                    face: source,
                },
            )
        });

        for candidate in candidates {
            let rigid_targets = graph
                .bearings()
                .filter_map(|(_, bearing)| match bearing.target.owner {
                    FaceOwner::Part(part) => Some(part),
                    FaceOwner::Ground => None,
                })
                .collect::<Vec<_>>();
            graph = stage_bearing_block_batch(
                &graph,
                candidate,
                &[candidate.spec],
                source,
                Vec3::Y,
                dimensions,
                &rigid_targets,
            )
            .unwrap();
        }

        let targets = graph
            .parts()
            .filter_map(|(part, _)| (part != support).then_some(part))
            .collect::<Vec<_>>();
        assert_eq!(targets.len(), 2);
        assert_eq!(graph.bearing_count(), 2);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(graph.rigid_link_count(), 1);
        assert_eq!(rigid_body_parts(&graph, targets[0]), targets);
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.compounds.len(), 2);
        assert_eq!(compiled.bearings.len(), 1);
    }

    #[test]
    fn bearing_drag_attaches_one_internally_welded_sheet() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::Y;
        let candidate = bearing_attachment_candidate(&graph, source, anchor);
        let mut endpoint = candidate.spec.pose.translation_half_units();
        endpoint.x += 2;
        let specs = block_sheet_specs(candidate.spec, endpoint, PlacementPlane::Xz).unwrap();

        let graph = stage_bearing_block_batch(
            &graph,
            candidate,
            &specs,
            source,
            anchor,
            BearingDimensions::default(),
            &[],
        )
        .unwrap();

        assert_eq!(graph.part_count(), 3);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 1);
        assert_eq!(graph.compile().unwrap().compounds.len(), 2);
    }

    #[test]
    fn bearing_centres_and_attaches_on_a_quarter_metre_block() {
        let graph = ConstructionGraph::new();
        let block = candidate_from_hit(
            &graph,
            SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            },
        );
        let graph = stage_cuboid(&graph, block).unwrap();
        let base = graph.parts().next().unwrap().0;
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.1, 0.25, -0.1),
            face: source,
        };

        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert!(anchor.abs_diff_eq(Vec3::new(0.0, 0.25, 0.0), 1.0e-6));
        let candidate = bearing_attachment_candidate(&graph, source, anchor);
        let graph = stage_bearing_attachment(
            &graph,
            candidate,
            source,
            anchor,
            BearingDimensions::default(),
        )
        .unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn bearing_rejects_ground_but_allows_visual_overhang_at_face_edges() {
        let mut graph = ConstructionGraph::new();
        let ground_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        assert!(matches!(
            bearing_anchor_from_hit(&graph, ground_hit),
            Err(PlacementError::BearingOnGround)
        ));

        let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 2);
        let edge_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.25, 0.5, 0.25),
            face: FaceRef::part(part, FaceKind::PositiveY),
        };
        let anchor = bearing_anchor_from_hit(&graph, edge_hit).unwrap();
        assert_eq!(anchor, Vec3::new(0.25, 0.75, 0.25));
        let candidate = bearing_attachment_candidate(&graph, edge_hit.face, anchor);
        let dimensions = BearingDimensions::new(8.0, 0.10).unwrap();
        let attached =
            stage_bearing_attachment(&graph, candidate, edge_hit.face, anchor, dimensions).unwrap();
        assert_eq!(attached.bearings().next().unwrap().1.dimensions, dimensions);
        assert_eq!(graph.part_count(), 1);
    }

    #[test]
    fn rejected_overlap_does_not_mutate_source_graph() {
        let graph = ConstructionGraph::new();
        let base_hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, base_hit);
        let graph = stage_cuboid(&graph, candidate).unwrap();
        assert!(matches!(
            stage_cuboid(&graph, candidate),
            Err(PlacementError::OverlapsPart(_))
        ));
        assert_eq!(graph.part_count(), 1);
    }

    #[test]
    fn remove_cascades_through_incident_bearing() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let source = FaceRef::part(base, FaceKind::PositiveY);
        let anchor = Vec3::new(0.0, 1.0, 0.0);
        let candidate = bearing_attachment_candidate(&graph, source, anchor);
        let graph = stage_bearing_attachment(
            &graph,
            candidate,
            source,
            anchor,
            BearingDimensions::default(),
        )
        .unwrap();
        let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y).unwrap();
        let upper = match top.face.owner {
            FaceOwner::Part(part) => part,
            FaceOwner::Ground => panic!("top ray must hit attached part"),
        };
        let mut graph = graph;
        graph.apply(BuildCommand::Remove(upper)).unwrap();
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
    }
}
