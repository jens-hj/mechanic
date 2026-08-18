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

const ALL_FACES: [FaceKind; 6] = [
    FaceKind::PositiveX,
    FaceKind::NegativeX,
    FaceKind::PositiveY,
    FaceKind::NegativeY,
    FaceKind::PositiveZ,
    FaceKind::NegativeZ,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum BuildTool {
    #[default]
    Cuboid,
    Weld,
    Bearing,
}

impl BuildTool {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Cuboid => "Cuboid",
            Self::Weld => "Weld",
            Self::Bearing => "Bearing",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SizePreset(usize);

impl Default for SizePreset {
    fn default() -> Self {
        Self(1)
    }
}

impl SizePreset {
    const UNITS: [u8; 3] = [2, 4, 8];

    pub(crate) const fn units(self) -> u8 {
        Self::UNITS[self.0]
    }

    pub(crate) fn meters(self) -> f32 {
        f32::from(self.units()) * GRID_UNIT_METERS
    }

    pub(crate) fn smaller(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }

    pub(crate) fn larger(&mut self) {
        self.0 = (self.0 + 1).min(Self::UNITS.len() - 1);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceHit {
    pub(crate) distance: f32,
    pub(crate) point: Vec3,
    pub(crate) face: FaceRef,
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

pub(crate) fn candidate_from_hit(
    graph: &ConstructionGraph,
    size: SizePreset,
    hit: SurfaceHit,
) -> PlacementCandidate {
    let support = face_geometry_from_ref(hit.face, Some(graph));
    let half_units = i32::from(size.units()) / 2;
    let mut center_units = snap_world_to_grid(hit.point);
    let plane_units = snap_world_to_grid(support.center);
    let (axis, sign) = cardinal_axis(support.normal);
    center_units[axis] = plane_units[axis] + sign * half_units;

    let spec = CuboidSpec::new(
        [size.units(); 3],
        BuildPose::new(center_units, GridRotation::default()),
    )
    .expect("size presets are valid core dimensions");
    let attached_face = face_for_normal(-support.normal);
    let candidate_face = face_geometry(spec, attached_face);
    let anchor = overlap_center(support, candidate_face);
    PlacementCandidate {
        spec,
        attached_face,
        anchor,
    }
}

pub(crate) fn stage_cuboid(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
) -> Result<ConstructionGraph, PlacementError> {
    validate_candidate(graph, candidate)?;
    let mut staged = graph.clone();
    let BuildOutcome::Spawned(_) = staged
        .apply(BuildCommand::Spawn(candidate.spec))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!("spawn commands always return a spawned part")
    };
    Ok(staged)
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
    let offset = anchor - face.center;
    let radius = BEARING_DIAMETER * 0.5;
    if offset.dot(face.tangent_u).abs() > face.half_u - radius + CONTACT_EPSILON
        || offset.dot(face.tangent_v).abs() > face.half_v - radius + CONTACT_EPSILON
    {
        return Err(PlacementError::BearingOutsideFace);
    }
    Ok(anchor)
}

pub(crate) fn begin_bearing(
    graph: &mut ConstructionGraph,
    source: FaceRef,
    anchor: Vec3,
) -> Result<(), PlacementError> {
    graph
        .apply(BuildCommand::BeginPending(PendingOperation::Bearing {
            source,
            anchor,
        }))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(())
}

pub(crate) fn bearing_attachment_candidate(
    graph: &ConstructionGraph,
    size: SizePreset,
    source: FaceRef,
    anchor: Vec3,
) -> PlacementCandidate {
    candidate_from_hit(
        graph,
        size,
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
    validate_candidate(graph, candidate)?;
    let mut staged = graph.clone();
    let BuildOutcome::Spawned(part) = staged
        .apply(BuildCommand::Spawn(candidate.spec))
        .map_err(|error| PlacementError::Graph(error.to_string()))?
    else {
        unreachable!("spawn commands always return a spawned part")
    };
    let axis = face_geometry_from_ref(source, Some(graph)).normal;
    staged
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            source,
            FaceRef::part(part, candidate.attached_face),
            anchor,
            axis,
        )))
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok(staged)
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
    let (minimum, maximum) = world_bounds(candidate.spec);
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
    if candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
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
    let inverse_rotation = spec.pose.rotation.quaternion().inverse();
    let local_origin = inverse_rotation * (origin - spec.pose.translation());
    let local_direction = inverse_rotation * direction;
    let half = spec.size_meters() * 0.5;
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut hit_axis = 0;
    let mut hit_sign = -1.0;

    for axis in 0..3 {
        if local_direction[axis].abs() <= f32::EPSILON {
            if local_origin[axis] < -half[axis] || local_origin[axis] > half[axis] {
                return None;
            }
            continue;
        }
        let inverse = local_direction[axis].recip();
        let first = (-half[axis] - local_origin[axis]) * inverse;
        let second = (half[axis] - local_origin[axis]) * inverse;
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
    Some(SurfaceHit {
        distance,
        point: origin + direction * distance,
        face: FaceRef::part(part, face_for_normal(local_normal)),
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
        PlacementError, SizePreset, SurfaceHit, bearing_anchor_from_hit,
        bearing_attachment_candidate, begin_bearing, begin_weld, candidate_from_hit,
        raycast_construction, stage_bearing_attachment, stage_cuboid, stage_weld_objects,
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
    fn preset_sizes_place_on_ground_and_cuboid_faces() {
        for mut size in [SizePreset(0), SizePreset(1), SizePreset(2)] {
            let graph = ConstructionGraph::new();
            let ground_hit = SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: mechanic_core::FaceRef::ground(),
            };
            let candidate = candidate_from_hit(&graph, size, ground_hit);
            assert!((candidate.spec.pose.translation().y - size.meters() * 0.5).abs() < 1.0e-6);
            let graph = stage_cuboid(&graph, candidate).unwrap();

            let top = raycast_construction(&graph, Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)
                .expect("placed cube is under ray");
            size.larger();
            size.smaller();
            let attached = candidate_from_hit(&graph, size, top);
            assert!(stage_cuboid(&graph, attached).is_ok());
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
                *graph.part(part).unwrap(),
            )
            .expect("ray reaches requested face");
            assert_eq!(hit.face.face, expected_face);
            let candidate = candidate_from_hit(&graph, SizePreset::default(), hit);
            assert!(stage_cuboid(&graph, candidate).is_ok());
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
        let candidate = candidate_from_hit(&graph, SizePreset(2), hit);
        assert!(matches!(
            stage_cuboid(&graph, candidate),
            Err(PlacementError::OutsidePlatform)
        ));
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
    fn bearing_first_click_places_only_a_snapped_connector() {
        let mut graph = ConstructionGraph::new();
        let base = spawn_cube(&mut graph, IVec3::new(0, 2, 0), 4);
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::new(0.18, 1.0, -0.18),
            face: FaceRef::part(base, FaceKind::PositiveY),
        };
        let anchor = bearing_anchor_from_hit(&graph, hit).unwrap();
        assert_eq!(anchor, Vec3::new(0.25, 1.0, -0.25));

        begin_bearing(&mut graph, hit.face, anchor).unwrap();

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert!(matches!(
            graph.pending(),
            Some(PendingOperation::Bearing { .. })
        ));
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
        begin_bearing(&mut graph, source, anchor).unwrap();
        let candidate = bearing_attachment_candidate(&graph, SizePreset::default(), source, anchor);

        let graph = stage_bearing_attachment(&graph, candidate, source, anchor).unwrap();

        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.bearings().next().unwrap().1.shared_anchor, anchor);
        assert!(graph.pending().is_none());
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
        let candidate = candidate_from_hit(&graph, SizePreset::default(), base_hit);
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
        begin_bearing(&mut graph, source, anchor).unwrap();
        let candidate = bearing_attachment_candidate(&graph, SizePreset::default(), source, anchor);
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
