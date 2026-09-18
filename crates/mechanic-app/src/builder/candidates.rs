//! Placement candidates derived from a surface hit or from free space.

use super::faces::{
    FaceGeometry, FaceProfile, cylinder_face_geometry, face_geometry, overlap_center,
    part_face_geometry, point_in_profile, try_face_geometries_from_ref,
};
use super::grid::{
    cardinal_axis, cardinal_direction, face_for_normal, rotation_y_to_normal,
    snap_global_center_ticks, snap_world_to_position_ticks,
};
use super::{
    ALL_FACES, BLOCK_SIZE_UNITS, CONTACT_EPSILON, CylinderPlacementCandidate, PlacementBounds,
    PlacementCandidate, PlacementError, PlacementGrid, PlacementSupport, Result, SurfaceHit, Vec,
    Vec3, vec,
};
use mechanic_core::{
    BuildPose, ConstructionGraph, CuboidSpec, CylinderDimensions, CylinderSpec, FaceKind,
    FaceOwner, GridRotation, POSITION_TICK_METERS, POSITION_TICKS_PER_HALF_GRID_UNIT, PartSpec,
};

pub(crate) fn candidate_from_hit(graph: &ConstructionGraph, hit: SurfaceHit) -> PlacementCandidate {
    candidate_from_hit_with_grid(
        graph,
        hit,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn candidate_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> PlacementCandidate {
    candidate_from_hit_with_grid_and_supports(graph, hit, grid, bounds).0
}

pub(crate) fn candidate_from_hit_with_grid_and_supports(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> (PlacementCandidate, Vec<FaceGeometry>) {
    let supports = support_geometries_from_hit(graph, hit);
    let candidate = oriented_cuboid_candidate_from_supports(
        graph,
        hit,
        [BLOCK_SIZE_UNITS; 3],
        GridRotation::default(),
        grid,
        bounds,
        &supports,
    );
    (candidate, supports)
}

/// Places a fixed-size authored cuboid flush with the face under the pointer.
#[cfg(test)]
pub(crate) fn cuboid_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
) -> PlacementCandidate {
    oriented_cuboid_candidate_from_hit_with_grid(
        graph,
        hit,
        dimensions,
        GridRotation::default(),
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

/// Places a fixed-size cuboid with a grid-aligned orientation flush with a face.
#[cfg(test)]
pub(crate) fn oriented_cuboid_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    rotation: GridRotation,
) -> PlacementCandidate {
    oriented_cuboid_candidate_from_hit_with_grid(
        graph,
        hit,
        dimensions,
        rotation,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn oriented_cuboid_candidate_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    rotation: GridRotation,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> PlacementCandidate {
    let supports = support_geometries_from_hit(graph, hit);
    oriented_cuboid_candidate_from_supports(
        graph, hit, dimensions, rotation, grid, bounds, &supports,
    )
}

pub(super) fn oriented_cuboid_candidate_from_supports(
    _graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: [u8; 3],
    rotation: GridRotation,
    grid: PlacementGrid,
    bounds: PlacementBounds,
    supports: &[FaceGeometry],
) -> PlacementCandidate {
    let world_dimensions = oriented_grid_dimensions(dimensions, rotation);
    let support = support_at_hit(supports, hit.point)
        .expect("cuboid placement requires a flat support surface");
    let support_center_ticks = snap_world_to_position_ticks(support.center);
    let mut center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(hit.point),
        world_dimensions,
        grid,
        bounds,
    );
    let (axis, sign) = cardinal_axis(support.normal);
    center_ticks[axis] = support_center_ticks[axis]
        + sign * i32::from(world_dimensions[axis]) * POSITION_TICKS_PER_HALF_GRID_UNIT;

    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation),
    )
    .expect("the fixed block size is a valid core dimension");
    let attached_face = face_for_normal(rotation.quaternion().inverse() * -support.normal);
    let candidate_face = face_geometry(spec, attached_face);
    let anchor = supports
        .iter()
        .find_map(|support| overlap_center(support, &candidate_face));
    PlacementCandidate {
        spec,
        attached_face,
        anchor,
        support: PlacementSupport::Surface(hit.face.owner),
    }
}

pub(super) fn oriented_grid_dimensions(dimensions: [u8; 3], rotation: GridRotation) -> [u8; 3] {
    let mut world_dimensions = [0; 3];
    for (local_axis, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let (world_axis, _) = cardinal_axis(rotation.quaternion() * direction);
        world_dimensions[world_axis] = dimensions[local_axis];
    }
    world_dimensions
}

/// Builds a cuboid candidate around a point in empty Garage space.
pub(crate) fn free_cuboid_candidate(
    point: Vec3,
    view_direction: Vec3,
    dimensions: [u8; 3],
    rotation: GridRotation,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> PlacementCandidate {
    let world_dimensions = oriented_grid_dimensions(dimensions, rotation);
    let center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(point),
        world_dimensions,
        grid,
        bounds,
    );
    let spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation),
    )
    .expect("authored placement dimensions are valid");
    let view_normal = cardinal_direction(-view_direction);
    PlacementCandidate {
        spec,
        attached_face: face_for_normal(rotation.quaternion().inverse() * -view_normal),
        anchor: None,
        support: PlacementSupport::Free,
    }
}

/// Builds a cylinder candidate around a point with its axis facing the view.
pub(crate) fn free_cylinder_candidate(
    point: Vec3,
    view_direction: Vec3,
    dimensions: CylinderDimensions,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> CylinderPlacementCandidate {
    let axis = cardinal_direction(-view_direction);
    let axial_axis = cardinal_axis(axis).0;
    let mut approximate_dimensions = [1; 3];
    approximate_dimensions[axial_axis] = dimensions.axial_length_units();
    let center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(point),
        approximate_dimensions,
        grid,
        bounds,
    );
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation_y_to_normal(axis)),
    );
    CylinderPlacementCandidate {
        spec,
        attached_face: FaceKind::NegativeY,
        anchor: None,
        support: PlacementSupport::Free,
    }
}

#[cfg(test)]
pub(crate) fn cylinder_candidate_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    cylinder_candidate_from_hit_with_grid(
        graph,
        hit,
        dimensions,
        PlacementGrid::Centimetres25,
        PlacementBounds::Garage,
    )
}

pub(crate) fn cylinder_candidate_from_hit_with_grid(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
    dimensions: CylinderDimensions,
    grid: PlacementGrid,
    bounds: PlacementBounds,
) -> Result<CylinderPlacementCandidate, PlacementError> {
    let supports = support_geometries_from_hit(graph, hit);
    let support = support_at_hit(&supports, hit.point).ok_or(PlacementError::CurvedSurface)?;
    if let FaceOwner::Part(part) = hit.face.owner
        && matches!(graph.part(part), Some(PartSpec::PipeJunction(_)))
        && (hit.point - support.center).dot(support.normal).abs() > CONTACT_EPSILON
    {
        // Only an arm's flat end continues a pipe; its walls take branches.
        return Err(PlacementError::CurvedSurface);
    }
    let support_center_ticks = snap_world_to_position_ticks(support.center);
    let axial_axis = cardinal_axis(support.normal).0;
    let mut approximate_dimensions = [1; 3];
    approximate_dimensions[axial_axis] = dimensions.axial_length_units();
    let mut center_ticks = snap_global_center_ticks(
        snap_world_to_position_ticks(hit.point),
        approximate_dimensions,
        grid,
        bounds,
    );
    let (axis, sign) = cardinal_axis(support.normal);
    center_ticks[axis] = support_center_ticks[axis]
        + sign * i32::from(dimensions.axial_length_units()) * POSITION_TICKS_PER_HALF_GRID_UNIT;
    if let FaceOwner::Part(part) = hit.face.owner
        && matches!(graph.part(part), Some(PartSpec::PipeJunction(_)))
    {
        // A junction face only takes a pipe on its channel axis.
        for lateral in (0..3).filter(|&lateral| lateral != axis) {
            center_ticks[lateral] = support_center_ticks[lateral];
        }
    }
    let rotation = rotation_y_to_normal(support.normal);
    let spec = CylinderSpec::new(
        dimensions,
        BuildPose::from_position_ticks(center_ticks, rotation),
    );
    let attached_face = FaceKind::NegativeY;
    let candidate_face = cylinder_face_geometry(spec, attached_face)
        .expect("negative-y is a cylinder connection face");
    Ok(CylinderPlacementCandidate {
        spec,
        attached_face,
        anchor: supports
            .iter()
            .find_map(|support| overlap_center(support, &candidate_face))
            .or_else(|| supporting_face_overlap(graph, support, &candidate_face)),
        support: PlacementSupport::Surface(hit.face.owner),
    })
}

pub(super) fn support_geometries_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Vec<FaceGeometry> {
    if matches!(hit.face.owner, FaceOwner::Ground) {
        let mut center = hit.point;
        center.y = (center.y / POSITION_TICK_METERS).floor() * POSITION_TICK_METERS;
        return vec![FaceGeometry {
            center,
            normal: Vec3::Y,
            tangent_u: Vec3::X,
            tangent_v: Vec3::Z,
            profile: FaceProfile::Ground,
        }];
    }
    try_face_geometries_from_ref(hit.face, Some(graph))
}

pub(super) fn support_at_hit(supports: &[FaceGeometry], point: Vec3) -> Option<&FaceGeometry> {
    supports
        .iter()
        .find(|support| {
            let offset = point - support.center;
            offset.dot(support.normal).abs() <= CONTACT_EPSILON
                && point_in_profile(
                    offset.dot(support.tangent_u),
                    offset.dot(support.tangent_v),
                    &support.profile,
                )
        })
        .or_else(|| supports.first())
}

pub(super) fn supporting_face_overlap(
    graph: &ConstructionGraph,
    selected: &FaceGeometry,
    candidate: &FaceGeometry,
) -> Option<Vec3> {
    overlap_center(selected, candidate).or_else(|| {
        graph.parts().find_map(|(_, spec)| {
            ALL_FACES.into_iter().find_map(|face| {
                part_face_geometry(*spec, face)
                    .and_then(|support| overlap_center(&support, candidate))
            })
        })
    })
}
