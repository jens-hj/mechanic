//! Build-volume validation, world bounds of parts, and part-against-part overlap.

use super::snap::PlacementSnapIndex;
use super::{
    BlockVolume, CONTACT_EPSILON, GRID_UNIT_METERS, GROUND_HALF_SIZE, IVec3, Mat3, PlacementBounds,
    PlacementCandidate, PlacementError, PlacementSupport, Quat, Result, ToOwned, UVec3, Vec, Vec3,
    vec,
};
use mechanic_core::{
    ConstructionGraph, CuboidSpec, CylinderDimensions, PartId, PartSpec, PipeBendSpec,
};
use mechanic_world::WORLD_HALF_EXTENT_METERS;

pub(crate) fn validate_block_batch_in_bounds(
    graph: &ConstructionGraph,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_candidate_in_bounds(graph, start, bounds)?;
    if specs.is_empty() {
        return Err(PlacementError::EmptyBlockBatch);
    }
    for spec in specs {
        validate_spec_in_bounds(graph, *spec, bounds)?;
    }
    Ok(())
}

/// Preview counterpart to [`validate_block_batch_in_bounds`] that narrows exact
/// overlap tests through the placement index. The committed operation still
/// performs the full validation when it is applied.
pub(crate) fn validate_indexed_block_batch_in_bounds(
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    specs: &[CuboidSpec],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    if start.support != PlacementSupport::Free && start.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    if specs.is_empty() {
        return Err(PlacementError::EmptyBlockBatch);
    }
    for spec in specs {
        let part = PartSpec::Cuboid(*spec);
        let (minimum, maximum) = part_world_bounds(part);
        validate_world_bounds(minimum, maximum, bounds)?;
        for target in index.nearby(minimum, maximum, 0.0) {
            if parts_overlap_with_frame(part, target.spec, target.frame) {
                return Err(PlacementError::OverlapsPart(target.part));
            }
        }
    }
    Ok(())
}

/// Exact volume validation using the same spatial index as smart snapping.
pub(crate) fn validate_block_volume_in_bounds(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    start: PlacementCandidate,
    volume: BlockVolume,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    if start.spec != volume.start() {
        return Err(PlacementError::Graph(
            "block volume does not begin at its placement candidate".to_owned(),
        ));
    }
    if start.support != PlacementSupport::Free && start.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    let (minimum, maximum) = volume.bounds();
    validate_world_bounds(minimum, maximum, bounds)?;

    for target in index.nearby(minimum, maximum, 0.0) {
        if !bounds_overlap_interior(minimum, maximum, target.minimum, target.maximum) {
            continue;
        }
        let Some(existing) = graph.part(target.part).copied() else {
            continue;
        };
        let (low, high) = volume_candidate_range(volume, target.minimum, target.maximum, false);
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    let candidate = volume.spec_at_physical(UVec3::new(x, y, z));
                    let (candidate_minimum, candidate_maximum) = cuboid_world_bounds(candidate);
                    if bounds_overlap_interior(
                        candidate_minimum,
                        candidate_maximum,
                        target.minimum,
                        target.maximum,
                    ) && parts_overlap_with_frame(
                        PartSpec::Cuboid(candidate),
                        existing,
                        target.frame,
                    ) {
                        return Err(PlacementError::OverlapsPart(target.part));
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn volume_candidate_range(
    volume: BlockVolume,
    target_minimum: Vec3,
    target_maximum: Vec3,
    include_touching: bool,
) -> (UVec3, UVec3) {
    let (minimum, _) = volume.bounds();
    let block_size = f32::from(volume.start().dimensions[0].units()) * GRID_UNIT_METERS;
    let padding = i32::from(include_touching);
    let counts = volume.dimensions().as_ivec3();
    let low = (((target_minimum - minimum) / block_size).floor().as_ivec3()
        - IVec3::splat(padding))
    .clamp(IVec3::ZERO, counts - IVec3::ONE);
    let high = (((target_maximum - minimum) / block_size).ceil().as_ivec3()
        + IVec3::splat(padding))
    .clamp(IVec3::ZERO, counts - IVec3::ONE);
    (low.as_uvec3(), high.as_uvec3())
}

pub(super) fn validate_candidate_in_bounds(
    graph: &ConstructionGraph,
    candidate: PlacementCandidate,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_spec_in_bounds(graph, candidate.spec, bounds)?;
    if candidate.support != PlacementSupport::Free && candidate.anchor.is_none() {
        return Err(PlacementError::NoFaceOverlap);
    }
    Ok(())
}

pub(super) fn validate_spec_in_bounds(
    graph: &ConstructionGraph,
    spec: CuboidSpec,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    validate_part_in_bounds(graph, PartSpec::Cuboid(spec), bounds)
}

#[cfg(test)]
pub(super) fn validate_part(
    graph: &ConstructionGraph,
    spec: PartSpec,
) -> Result<(), PlacementError> {
    validate_part_in_bounds(graph, spec, PlacementBounds::Garage)
}

pub(super) fn validate_part_in_bounds(
    graph: &ConstructionGraph,
    spec: PartSpec,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let (minimum, maximum) = part_world_bounds(spec);
    validate_world_bounds(minimum, maximum, bounds)?;
    for (part, existing) in graph.parts() {
        let frame = graph
            .part_frame(part)
            .expect("validated parts have construction frames");
        if parts_overlap_with_frame(spec, *existing, frame) {
            return Err(PlacementError::OverlapsPart(part));
        }
    }
    Ok(())
}

pub(crate) fn validate_world_bounds(
    minimum: Vec3,
    maximum: Vec3,
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let outside = match bounds {
        PlacementBounds::GarageBuildFrame { to_garage } => {
            for x in [minimum.x, maximum.x] {
                for y in [minimum.y, maximum.y] {
                    for z in [minimum.z, maximum.z] {
                        let point = to_garage.point(Vec3::new(x, y, z));
                        validate_world_bounds(point, point, PlacementBounds::GarageBuild)?;
                    }
                }
            }
            false
        }
        PlacementBounds::Garage => {
            minimum.x < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.x > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.z < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.z > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.y < -CONTACT_EPSILON
        }
        PlacementBounds::GarageBuild => {
            minimum.x < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.x > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.z < -GROUND_HALF_SIZE - CONTACT_EPSILON
                || maximum.z > GROUND_HALF_SIZE + CONTACT_EPSILON
                || minimum.y < crate::garage::BUILD_MIN_Y - CONTACT_EPSILON
                || maximum.y > crate::garage::BUILD_MAX_Y + CONTACT_EPSILON
        }
        PlacementBounds::World { origin } => {
            f64::from(minimum.x) + origin.x < -WORLD_HALF_EXTENT_METERS
                || f64::from(maximum.x) + origin.x > WORLD_HALF_EXTENT_METERS
                || f64::from(minimum.z) + origin.y < -WORLD_HALF_EXTENT_METERS
                || f64::from(maximum.z) + origin.y > WORLD_HALF_EXTENT_METERS
        }
    };
    if outside {
        return Err(PlacementError::OutsidePlatform);
    }
    Ok(())
}

pub(super) fn cuboid_world_bounds(spec: CuboidSpec) -> (Vec3, Vec3) {
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

/// Axis-aligned authored bounds after applying the part's construction frame.
pub(crate) fn composed_part_world_bounds(
    graph: &ConstructionGraph,
    part: PartId,
) -> Option<(Vec3, Vec3)> {
    let frame = graph.part_frame(part)?;
    let (minimum, maximum) = part_world_bounds(*graph.part(part)?);
    Some(transformed_bounds(
        frame.translation(),
        frame.rotation(),
        minimum,
        maximum,
    ))
}

pub(crate) fn part_world_bounds(spec: PartSpec) -> (Vec3, Vec3) {
    match spec {
        PartSpec::Cuboid(spec) => cuboid_world_bounds(spec),
        PartSpec::Controller(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Engine(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Transmission(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Servo(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Seat(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::Input(spec) => cuboid_world_bounds(spec.cuboid()),
        PartSpec::DimensionLink(spec) => cuboid_world_bounds(spec.cuboid()),
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
        PartSpec::PipeJunction(spec) => transformed_bounds(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            Vec3::splat(-spec.dimensions.half_side()),
            Vec3::splat(spec.dimensions.half_side()),
        ),
        PartSpec::PipeBend(spec) => {
            let outer = spec.dimensions.outer_diameter() * 0.5;
            let radius = spec.dimensions.radius();
            let local_minimum = Vec3::new(-radius, -outer, -outer);
            let local_maximum = Vec3::new(outer, radius, outer);
            transformed_bounds(
                spec.pose.translation(),
                spec.pose.rotation.quaternion(),
                local_minimum,
                local_maximum,
            )
        }
    }
}

pub(super) fn transformed_bounds(
    translation: Vec3,
    rotation: Quat,
    local_minimum: Vec3,
    local_maximum: Vec3,
) -> (Vec3, Vec3) {
    let mut world_minimum = Vec3::splat(f32::INFINITY);
    let mut world_maximum = Vec3::splat(f32::NEG_INFINITY);
    for x in [local_minimum.x, local_maximum.x] {
        for y in [local_minimum.y, local_maximum.y] {
            for z in [local_minimum.z, local_maximum.z] {
                let point = translation + rotation * Vec3::new(x, y, z);
                world_minimum = world_minimum.min(point);
                world_maximum = world_maximum.max(point);
            }
        }
    }
    (world_minimum, world_maximum)
}

pub(super) fn cylinder_local_bounds(dimensions: CylinderDimensions) -> (Vec3, Vec3) {
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
pub(super) struct CollisionBox {
    pub(super) center: Vec3,
    pub(super) rotation: Quat,
    pub(super) half: Vec3,
}

pub(crate) fn parts_overlap(first: PartSpec, second: PartSpec) -> bool {
    parts_overlap_with_frame(first, second, mechanic_core::ConstructionFrame::IDENTITY)
}

/// Placement candidates use the current tool-view grid; committed parts may
/// belong to another rigid frame in that same view.
pub(super) fn parts_overlap_with_frame(
    candidate: PartSpec,
    target: PartSpec,
    frame: mechanic_core::ConstructionFrame,
) -> bool {
    let candidate_boxes = part_collision_boxes(candidate);
    let mut target_boxes = part_collision_boxes(target);
    if frame != mechanic_core::ConstructionFrame::IDENTITY {
        for shape in &mut target_boxes {
            shape.center = frame.point(shape.center);
            shape.rotation = frame.rotation() * shape.rotation;
        }
    }
    let (first_minimum, first_maximum) = collision_boxes_bounds(&candidate_boxes);
    let (second_minimum, second_maximum) = collision_boxes_bounds(&target_boxes);
    if (first_minimum - second_maximum)
        .cmpgt(Vec3::splat(CONTACT_EPSILON))
        .any()
        || (second_minimum - first_maximum)
            .cmpgt(Vec3::splat(CONTACT_EPSILON))
            .any()
    {
        return false;
    }
    candidate_boxes.into_iter().any(|candidate| {
        target_boxes
            .iter()
            .copied()
            .any(|target| boxes_overlap(candidate, target))
    })
}

/// Bounds the actual collision boxes, including the conservative wall boxes
/// outside a round pipe's ideal radius. Authored bounds alone can miss those.
pub(super) fn collision_boxes_bounds(boxes: &[CollisionBox]) -> (Vec3, Vec3) {
    boxes.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(minimum, maximum), shape| {
            let extent = (shape.rotation * Vec3::X).abs() * shape.half.x
                + (shape.rotation * Vec3::Y).abs() * shape.half.y
                + (shape.rotation * Vec3::Z).abs() * shape.half.z;
            (
                minimum.min(shape.center - extent),
                maximum.max(shape.center + extent),
            )
        },
    )
}

pub(super) fn part_collision_boxes(spec: PartSpec) -> Vec<CollisionBox> {
    match spec {
        PartSpec::Controller(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Engine(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Transmission(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Servo(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Seat(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::Input(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
        PartSpec::DimensionLink(spec) => part_collision_boxes(PartSpec::Cuboid(spec.cuboid())),
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
        PartSpec::PipeBend(spec) => pipe_bend_collision_boxes(spec),
        PartSpec::PipeJunction(spec) => {
            let rotation = spec.pose.rotation.quaternion();
            mechanic_core::pipe_junction_wall_boxes(spec)
                .into_iter()
                .map(|wall| CollisionBox {
                    center: spec.pose.translation() + rotation * wall.center,
                    rotation: rotation * wall.rotation,
                    half: wall.half_extents,
                })
                .collect()
        }
    }
}

pub(super) fn pipe_bend_collision_boxes(spec: PipeBendSpec) -> Vec<CollisionBox> {
    let outer = spec.dimensions.outer_diameter() * 0.5;
    let inner = spec.dimensions.inner_diameter() * 0.5;
    let half_radial = (outer - inner) * 0.5;
    let cross_radius = (outer + inner) * 0.5;
    let bend_radius = spec.dimensions.radius();
    let bend_step = std::f32::consts::FRAC_PI_2 / 12.0;
    let cross_step = std::f32::consts::TAU / 16.0;
    let part_rotation = spec.pose.rotation.quaternion();
    let mut boxes = Vec::with_capacity(12 * 16);
    for bend_slice in 0_u16..12 {
        let theta = -std::f32::consts::FRAC_PI_2 + bend_step * (f32::from(bend_slice) + 0.5);
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let tangent = Vec3::new(-theta.sin(), theta.cos(), 0.0);
        for sector in 0_u16..16 {
            let phi = cross_step * (f32::from(sector) + 0.5);
            let normal = radial * phi.cos() + Vec3::Z * phi.sin();
            let cross_tangent = -radial * phi.sin() + Vec3::Z * phi.cos();
            let mut center = Vec3::new(-bend_radius, bend_radius, 0.0)
                + radial * (bend_radius + cross_radius * phi.cos())
                + Vec3::Z * (cross_radius * phi.sin());
            let mut half = Vec3::new(
                half_radial,
                (bend_radius + outer) * (bend_step * 0.5).tan(),
                outer * (cross_step * 0.5).tan(),
            );
            // Tight bends reach their end planes from inner slices too, so
            // every box stays behind both caps. Creased bends leave slivers at
            // the crease that no shortening pulls back; they hold no material.
            let mut protrudes = false;
            for (plane_center, outward) in [
                (Vec3::new(-bend_radius, 0.0, 0.0), Vec3::NEG_X),
                (Vec3::new(0.0, bend_radius, 0.0), Vec3::Y),
            ] {
                protrudes |= trim_pipe_bend_box_to_end_plane(
                    &mut center,
                    &mut half,
                    [normal, tangent, cross_tangent],
                    plane_center,
                    outward,
                );
            }
            if protrudes {
                continue;
            }
            boxes.push(CollisionBox {
                center: spec.pose.translation() + part_rotation * center,
                rotation: part_rotation
                    * Quat::from_mat3(&Mat3::from_cols(normal, tangent, cross_tangent)),
                half,
            });
        }
    }
    boxes
}

/// Keeps the conservative bend tessellation behind its two exact tangent caps.
/// The box is shortened along whichever of its axes faces the cap most
/// directly; the bend's material never crosses either cap plane. Returns
/// whether the box still crosses the cap afterwards.
pub(super) fn trim_pipe_bend_box_to_end_plane(
    center: &mut Vec3,
    half: &mut Vec3,
    axes: [Vec3; 3],
    plane_center: Vec3,
    outward: Vec3,
) -> bool {
    let projections = axes.map(|axis| axis.dot(outward));
    let reach = |center: Vec3, half: Vec3| {
        (center - plane_center).dot(outward)
            + half.x * projections[0].abs()
            + half.y * projections[1].abs()
            + half.z * projections[2].abs()
    };
    let protrusion = reach(*center, *half) + CONTACT_EPSILON;
    if protrusion <= 0.0 {
        return false;
    }
    let index = (0..3)
        .max_by(|&left, &right| projections[left].abs().total_cmp(&projections[right].abs()))
        .expect("a box has three axes");
    let trim = (protrusion / projections[index].abs()).min(half[index] * 2.0);
    *center -= axes[index] * projections[index].signum() * trim * 0.5;
    half[index] -= trim * 0.5;
    reach(*center, *half) > 0.0
}

pub(super) fn boxes_overlap(first: CollisionBox, second: CollisionBox) -> bool {
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

pub(super) fn bounds_overlap_interior(
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

/// World bounds of every part in the graph, or `None` when it has no parts.
/// Bearing and suspension geometry is not included.
pub(crate) fn graph_part_bounds(graph: &ConstructionGraph) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for (part, _) in graph.parts() {
        let (low, high) = composed_part_world_bounds(graph, part)?;
        minimum = minimum.min(low);
        maximum = maximum.max(high);
    }
    minimum.is_finite().then_some((minimum, maximum))
}
