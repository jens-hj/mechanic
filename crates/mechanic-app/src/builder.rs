use std::{cmp::Ordering, fmt};

use bevy::prelude::*;
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind,
    FaceOwner, FaceRef, GridRotation, PartId, PendingOperation, WeldSpec, snap_world_to_grid,
};

pub(crate) const GROUND_HALF_SIZE: f32 = 10.0;
const CONTACT_EPSILON: f32 = 1.0e-5;
const GRID_UNIT_METERS: f32 = 0.25;
pub(crate) const BEARING_DIAMETER: f32 = 0.25;
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

    const fn normal_axis(self) -> usize {
        match self {
            Self::Xy => 2,
            Self::Xz => 1,
            Self::Yz => 0,
        }
    }

    const fn tangent_axes(self) -> [usize; 2] {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlacementError {
    OutsidePlatform,
    NoFaceOverlap,
    OverlapsPart(PartId),
    BearingOnGround,
    BearingOutsideFace,
    SameObject,
    ObjectsDoNotTouch,
    EmptyBlockBatch,
    BlocksOverlap,
    DragPlaneUnavailable,
    TooManyBlocks { count: usize, maximum: usize },
    Graph(String),
}

impl fmt::Display for PlacementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutsidePlatform => formatter.write_str("cube would extend beyond the platform"),
            Self::NoFaceOverlap => formatter.write_str("cube does not overlap the selected face"),
            Self::OverlapsPart(part) => write!(formatter, "cube overlaps {part:?}"),
            Self::BearingOnGround => formatter.write_str("bearings cannot attach to the ground"),
            Self::BearingOutsideFace => {
                formatter.write_str("the 0.25 m bearing does not fit on this face")
            }
            Self::SameObject => formatter.write_str("select two different objects"),
            Self::ObjectsDoNotTouch => formatter.write_str("the selected objects do not touch"),
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
    half_u: f32,
    half_v: f32,
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
        .filter_map(|(part, spec)| raycast_cuboid(origin, direction, part, *spec))
        .chain(ground)
        .filter(|hit| hit.distance >= 0.0 && hit.distance.is_finite())
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .unwrap_or(Ordering::Equal)
        })
}

pub(crate) fn candidate_from_hit(graph: &ConstructionGraph, hit: SurfaceHit) -> PlacementCandidate {
    let support = face_geometry_from_ref(hit.face, Some(graph));
    let support_center_half_units = snap_world_to_half_grid(support.center);
    let mut center_half_units =
        support_center_half_units + snap_world_to_grid(hit.point - support.center) * 2;
    let (axis, sign) = cardinal_axis(support.normal);
    center_half_units[axis] = support_center_half_units[axis] + sign;

    let spec = CuboidSpec::new(
        [BLOCK_SIZE_UNITS; 3],
        BuildPose::from_half_grid(center_half_units, GridRotation::default()),
    )
    .expect("the fixed block size is a valid core dimension");
    let attached_face = face_for_normal(-support.normal);
    let candidate_face = face_geometry(spec, attached_face);
    let anchor = overlap_center(support, candidate_face);
    PlacementCandidate {
        spec,
        attached_face,
        anchor,
    }
}

#[cfg(test)]
pub(crate) fn stage_cuboid(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
) -> Result<ConstructionGraph, PlacementError> {
    stage_block_batch(graph, candidate, &[candidate.spec])
}

pub(crate) fn stage_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, None)
}

pub(crate) fn stage_bearing_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    source: FaceRef,
    anchor: Vec3,
) -> Result<ConstructionGraph, PlacementError> {
    stage_connected_block_batch(graph, start, specs, Some((source, anchor)))
}

fn stage_connected_block_batch(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bearing: Option<(FaceRef, Vec3)>,
) -> Result<ConstructionGraph, PlacementError> {
    validate_block_batch(graph, start, specs)?;
    for (index, spec) in specs.iter().enumerate() {
        for other in &specs[..index] {
            let (minimum, maximum) = world_bounds(*spec);
            let (other_minimum, other_maximum) = world_bounds(*other);
            if bounds_overlap_interior(minimum, maximum, other_minimum, other_maximum) {
                return Err(PlacementError::BlocksOverlap);
            }
        }
    }

    let existing_parts = graph.parts().map(|(part, _)| part).collect::<Vec<_>>();
    let mut staged = graph.clone();
    let outcomes = staged
        .apply_batch(specs.iter().copied().map(BuildCommand::Spawn))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    let new_parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("spawn batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();

    let mut connections = Vec::new();
    if let Some((source, anchor)) = bearing {
        let first = *new_parts
            .first()
            .expect("validated block batches are never empty");
        let axis = face_geometry_from_ref(source, Some(graph)).normal;
        connections.push(BuildCommand::AddBearing(BearingSpec::new(
            source,
            FaceRef::part(first, start.attached_face),
            anchor,
            axis,
        )));
    }
    for (index, &part) in new_parts.iter().enumerate() {
        for &other in existing_parts.iter().chain(&new_parts[..index]) {
            if bearing.is_some_and(|(source, _)| source.owner == FaceOwner::Part(other)) {
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
    let mut endpoint = snap_world_to_half_grid(origin + direction * distance);
    endpoint[axis] = start.pose.translation_half_units()[axis];
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
    let Some((first_face, second_face)) = touching_face_pair(graph, first, second) else {
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

pub(crate) fn bearing_anchor_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<Vec3, PlacementError> {
    if matches!(hit.face.owner, FaceOwner::Ground) {
        return Err(PlacementError::BearingOnGround);
    }
    let face = face_geometry_from_ref(hit.face, Some(graph));
    let mut anchor = snap_world_to_grid(hit.point).as_vec3() * GRID_UNIT_METERS;
    let (normal_axis, _) = cardinal_axis(face.normal);
    anchor[normal_axis] = face.center[normal_axis];
    let radius = BEARING_DIAMETER * 0.5;
    if face.half_u <= radius + CONTACT_EPSILON {
        anchor += face.tangent_u * (face.center - anchor).dot(face.tangent_u);
    }
    if face.half_v <= radius + CONTACT_EPSILON {
        anchor += face.tangent_v * (face.center - anchor).dot(face.tangent_v);
    }
    let offset = anchor - face.center;
    if offset.dot(face.tangent_u).abs() > face.half_u - radius + CONTACT_EPSILON
        || offset.dot(face.tangent_v).abs() > face.half_v - radius + CONTACT_EPSILON
    {
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

pub(crate) fn stage_bearing_attachment(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    source: FaceRef,
    anchor: Vec3,
) -> Result<ConstructionGraph, PlacementError> {
    stage_bearing_block_batch(graph, candidate, &[candidate.spec], source, anchor)
}

pub(crate) fn face_geometry_from_ref(
    face: FaceRef,
    graph: Option<&ConstructionGraph>,
) -> FaceGeometry {
    match face.owner {
        FaceOwner::Ground => FaceGeometry {
            center: Vec3::ZERO,
            normal: Vec3::Y,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Z,
            half_u: f32::INFINITY,
            half_v: f32::INFINITY,
        },
        FaceOwner::Part(part) => {
            let spec = graph
                .and_then(|graph| graph.part(part))
                .copied()
                .expect("live face references have a part");
            face_geometry(spec, face.face)
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
        half_u: half_u * 0.5,
        half_v: half_v * 0.5,
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
    let (minimum, maximum) = world_bounds(spec);
    if minimum.x < -GROUND_HALF_SIZE - CONTACT_EPSILON
        || maximum.x > GROUND_HALF_SIZE + CONTACT_EPSILON
        || minimum.z < -GROUND_HALF_SIZE - CONTACT_EPSILON
        || maximum.z > GROUND_HALF_SIZE + CONTACT_EPSILON
        || minimum.y < -CONTACT_EPSILON
    {
        return Err(PlacementError::OutsidePlatform);
    }
    for (part, spec) in graph.parts() {
        let (other_minimum, other_maximum) = world_bounds(*spec);
        if bounds_overlap_interior(minimum, maximum, other_minimum, other_maximum) {
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
    owner_faces(first).into_iter().find_map(|first_face| {
        owner_faces(second).into_iter().find_map(|second_face| {
            overlap_center(
                face_geometry_from_ref(first_face, Some(graph)),
                face_geometry_from_ref(second_face, Some(graph)),
            )
            .map(|_| (first_face, second_face))
        })
    })
}

fn owner_faces(owner: FaceOwner) -> Vec<FaceRef> {
    match owner {
        FaceOwner::Ground => vec![FaceRef::ground()],
        FaceOwner::Part(part) => ALL_FACES
            .into_iter()
            .map(|face| FaceRef::part(part, face))
            .collect(),
    }
}

fn raycast_ground(origin: Vec3, direction: Vec3) -> Option<SurfaceHit> {
    if direction.y.abs() <= f32::EPSILON {
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
    let mut center = first.center;
    for axis in 0..3 {
        if first.normal[axis].abs() > 0.5 {
            continue;
        }
        let (first_min, first_max) = face_axis_interval(first, axis);
        let (second_min, second_max) = face_axis_interval(second, axis);
        let minimum = first_min.max(second_min);
        let maximum = first_max.min(second_max);
        if maximum - minimum <= CONTACT_EPSILON {
            return None;
        }
        center[axis] = (minimum + maximum) * 0.5;
    }
    Some(center)
}

fn face_axis_interval(face: FaceGeometry, axis: usize) -> (f32, f32) {
    let radius =
        face.tangent_u[axis].abs() * face.half_u + face.tangent_v[axis].abs() * face.half_v;
    (face.center[axis] - radius, face.center[axis] + radius)
}

fn world_bounds(spec: CuboidSpec) -> (Vec3, Vec3) {
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
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind, FaceOwner,
        FaceRef, GridRotation, PendingOperation,
    };

    use super::{
        BLOCK_SIZE_METERS, PlacementError, PlacementPlane, SurfaceHit, bearing_anchor_from_hit,
        bearing_attachment_candidate, begin_weld, block_sheet_specs, candidate_from_hit,
        raycast_construction, raycast_placement_plane, stage_bearing_attachment,
        stage_bearing_block_batch, stage_block_batch, stage_cuboid, stage_weld_objects,
    };

    fn spawn_cube(graph: &mut ConstructionGraph, units: IVec3, size: u8) -> mechanic_core::PartId {
        let spec =
            CuboidSpec::new([size; 3], BuildPose::new(units, GridRotation::default())).unwrap();
        let Ok(BuildOutcome::Spawned(part)) = graph.apply(BuildCommand::Spawn(spec)) else {
            panic!("cube must spawn");
        };
        part
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
                *graph.part(part).unwrap(),
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
            let attached =
                stage_bearing_attachment(&graph, bearing_candidate, source, source_face.center)
                    .unwrap();
            assert_eq!(attached.bearing_count(), 1);
            assert_eq!(attached.weld_count(), 0);
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
        assert_eq!(graph.weld_count(), 7);
        assert_eq!(graph.compile().unwrap().compounds.len(), 1);
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

        let graph = stage_bearing_attachment(&graph, candidate, source, anchor).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.bearings().next().unwrap().1.shared_anchor, anchor);
        assert!(graph.pending().is_none());
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

        let graph = stage_bearing_block_batch(&graph, candidate, &specs, source, anchor).unwrap();

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
        let graph = stage_bearing_attachment(&graph, candidate, source, anchor).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
    }

    #[test]
    fn bearing_rejects_ground_and_face_edges() {
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
        assert!(matches!(
            bearing_anchor_from_hit(&graph, edge_hit),
            Err(PlacementError::BearingOutsideFace)
        ));
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
        let graph = stage_bearing_attachment(&graph, candidate, source, anchor).unwrap();
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
