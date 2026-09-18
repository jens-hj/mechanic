//! Material layers: finding the flat surface under the cursor and staging a layer over it.

use super::bounds::{part_world_bounds, parts_overlap_with_frame, validate_world_bounds};
use super::{
    CONTACT_EPSILON, PlacementBounds, PlacementError, PlacementGrid, Result, SurfaceHit, ToString,
    Vec, Vec2, Vec3, vec,
};
use mechanic_core::{
    BuildCommand, ConstructionGraph, CuboidSpec, FaceKind, FaceOwner, PartId, PartSpec,
};
use std::collections::HashSet;

/// Radial slack for a hit to count as lying on a cylinder wall. Walls are
/// drawn and picked as facets whose chords sit slightly inside the true radius.
pub(super) const LAYER_WALL_TOLERANCE_METERS: f32 = 0.01;

/// Slack for a hit to count as lying on a flat face or end cap.
pub(super) const LAYER_FACE_TOLERANCE_METERS: f32 = 2.0e-3;

/// Ray-to-pull alignment beyond which a drag's projected distance is
/// ill-conditioned, so each frame may move only one step.
pub(super) const LAYER_DRAG_STABILITY: f32 = 0.25;

/// One part taking a layer on one of its own faces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayerMember {
    pub(crate) part: PartId,
    pub(crate) spec: PartSpec,
    pub(crate) face: mechanic_core::LayerFace,
}

/// A surface a new material layer grows from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerTarget {
    pub(crate) part: PartId,
    pub(crate) spec: PartSpec,
    pub(crate) face: mechanic_core::LayerFace,
    /// Construction frame the part is authored in.
    pub(crate) frame: mechanic_core::ConstructionFrame,
    /// Picked surface point in world space.
    pub(crate) anchor: Vec3,
    /// World direction a thicker layer grows toward.
    pub(crate) normal: Vec3,
    /// Every part the layer covers, the picked part first. A block face
    /// brings the whole flat surface it continues.
    pub(crate) members: Vec<LayerMember>,
}

/// A block face as an axis-aligned rectangle in its construction frame.
#[derive(Clone, Copy)]
pub(super) struct FaceRect {
    pub(super) plane: f32,
    pub(super) minimum: [f32; 2],
    pub(super) maximum: [f32; 2],
}

/// Slack for block faces to count as one plane or as sharing an edge.
pub(super) const LAYER_PLANE_TOLERANCE_METERS: f32 = 1.0e-4;

impl FaceRect {
    /// The slab `reach` metres thick on this face, along world `axis`.
    pub(super) fn slab(self, axis: usize, reach: f32) -> (Vec3, Vec3) {
        let mut low = Vec3::ZERO;
        let mut high = Vec3::ZERO;
        low[axis] = self.plane.min(self.plane + reach);
        high[axis] = self.plane.max(self.plane + reach);
        for (index, tangent) in [(axis + 1) % 3, (axis + 2) % 3].into_iter().enumerate() {
            low[tangent] = self.minimum[index];
            high[tangent] = self.maximum[index];
        }
        (low, high)
    }

    /// Whether two coplanar faces share an edge, not merely a corner.
    pub(super) fn touches_edge(self, other: Self) -> bool {
        let overlap = [0, 1].map(|index| {
            self.maximum[index].min(other.maximum[index])
                - self.minimum[index].max(other.minimum[index])
        });
        overlap
            .iter()
            .all(|&length| length >= -LAYER_PLANE_TOLERANCE_METERS)
            && overlap
                .iter()
                .any(|&length| length > LAYER_PLANE_TOLERANCE_METERS)
    }
}

/// Index of the cardinal axis a unit direction points along.
pub(super) fn cardinal_axis_index(direction: Vec3) -> usize {
    if direction.x.abs() > 0.5 {
        0
    } else if direction.y.abs() > 0.5 {
        1
    } else {
        2
    }
}

/// A block taking a layer on whichever of its faces points along `normal`.
pub(super) fn block_layer_member(part: PartId, spec: CuboidSpec, normal: Vec3) -> LayerMember {
    let local = spec.pose.rotation.quaternion().inverse() * normal;
    let axis = cardinal_axis_index(local);
    LayerMember {
        part,
        spec: PartSpec::Cuboid(spec),
        face: mechanic_core::LayerFace::Face(face_kind_on_axis(axis, local[axis] > 0.0)),
    }
}

/// A part's rigid body: everything welded or rigidly linked to it.
pub(super) fn rigid_body(graph: &ConstructionGraph, part: PartId) -> HashSet<PartId> {
    let mut neighbours = std::collections::HashMap::<PartId, Vec<PartId>>::new();
    let links = graph
        .welds()
        .filter_map(|(_, weld)| match (weld.first.owner, weld.second.owner) {
            (FaceOwner::Part(first), FaceOwner::Part(second)) => Some((first, second)),
            _ => None,
        })
        .chain(
            graph
                .rigid_links()
                .map(|(_, link)| (link.first, link.second)),
        );
    for (first, second) in links {
        neighbours.entry(first).or_default().push(second);
        neighbours.entry(second).or_default().push(first);
    }
    let mut body = HashSet::from([part]);
    let mut pending = vec![part];
    while let Some(current) = pending.pop() {
        for &next in neighbours.get(&current).into_iter().flatten() {
            if body.insert(next) {
                pending.push(next);
            }
        }
    }
    body
}

/// Parts outside `excluded` that may reach the box from `minimum` to
/// `maximum` in `frame`, each with its transform into that frame. Parts in
/// other frames are always kept, since their bounds are not comparable.
pub(super) fn nearby_obstacles(
    graph: &ConstructionGraph,
    frame: mechanic_core::ConstructionFrame,
    excluded: &HashSet<PartId>,
    minimum: Vec3,
    maximum: Vec3,
) -> Vec<(PartSpec, mechanic_core::ConstructionFrame)> {
    let into_frame = frame.inverse();
    graph
        .parts()
        .filter(|(other, _)| !excluded.contains(other))
        .filter_map(|(other, spec)| {
            let other_frame = graph.part_frame(other)?;
            if other_frame == frame {
                let (low, high) = part_world_bounds(*spec);
                if (low - maximum).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                    || (minimum - high).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                {
                    return None;
                }
            }
            Some((*spec, into_frame.compose(other_frame)))
        })
        .collect()
}

/// Every block face continuing the picked block's flat face: coplanar, facing
/// the same way, joined edge to edge, on the same rigid body and construction
/// frame, and not covered by another part.
pub(super) fn flat_surface_members(
    graph: &ConstructionGraph,
    part: PartId,
    cuboid: CuboidSpec,
    frame: mechanic_core::ConstructionFrame,
    local_normal: Vec3,
) -> Vec<LayerMember> {
    let normal = cuboid.pose.rotation.quaternion() * local_normal;
    let axis = cardinal_axis_index(normal);
    let positive = normal[axis] > 0.0;
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    let rect = |spec: CuboidSpec| {
        let (minimum, maximum) = part_world_bounds(PartSpec::Cuboid(spec));
        FaceRect {
            plane: if positive {
                maximum[axis]
            } else {
                minimum[axis]
            },
            minimum: tangents.map(|tangent| minimum[tangent]),
            maximum: tangents.map(|tangent| maximum[tangent]),
        }
    };
    let member = |part, spec| block_layer_member(part, spec, normal);

    let body = rigid_body(graph, part);
    let start = rect(cuboid);
    let mut candidates = vec![(part, cuboid, start)];
    candidates.extend(graph.parts().filter_map(|(other, spec)| {
        let PartSpec::Cuboid(other_cuboid) = *spec else {
            return None;
        };
        let face = rect(other_cuboid);
        (other != part
            && body.contains(&other)
            && graph.region_of(other).is_none()
            && graph.part_frame(other) == Some(frame)
            && (face.plane - start.plane).abs() <= LAYER_PLANE_TOLERANCE_METERS)
            .then_some((other, other_cuboid, face))
    }));
    if candidates.len() == 1 {
        return vec![member(part, cuboid)];
    }

    // Parts that could sit on the surface: anything near its footprint.
    let reach = if positive {
        MIN_LAYER_COVER_METERS
    } else {
        -MIN_LAYER_COVER_METERS
    };
    let (footprint_minimum, footprint_maximum) = candidates.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(minimum, maximum), (_, _, face)| {
            let (low, high) = face.slab(axis, reach);
            (minimum.min(low), maximum.max(high))
        },
    );
    let candidate_parts = candidates
        .iter()
        .map(|(candidate, _, _)| *candidate)
        .collect::<HashSet<_>>();
    let obstacles = nearby_obstacles(
        graph,
        frame,
        &candidate_parts,
        footprint_minimum,
        footprint_maximum,
    );
    let covered = |candidate: LayerMember| {
        candidate
            .spec
            .with_layer(
                candidate.face,
                MIN_LAYER_COVER_METERS,
                mechanic_core::ConstructionMaterial::Steel,
                mechanic_core::MaterialAppearance::BAKED,
            )
            .is_ok_and(|layered| {
                obstacles.iter().any(|&(obstacle, relative)| {
                    parts_overlap_with_frame(layered, obstacle, relative)
                })
            })
    };

    let mut reached = vec![false; candidates.len()];
    reached[0] = true;
    let mut members = vec![member(part, cuboid)];
    let mut frontier = vec![0];
    while let Some(index) = frontier.pop() {
        let face = candidates[index].2;
        for (next, &(other, other_cuboid, other_face)) in candidates.iter().enumerate() {
            if reached[next] || !face.touches_edge(other_face) {
                continue;
            }
            reached[next] = true;
            let candidate = member(other, other_cuboid);
            if !covered(candidate) {
                members.push(candidate);
                frontier.push(next);
            }
        }
    }
    members
}

/// Thickness probing whether a flat surface block is covered.
pub(super) const MIN_LAYER_COVER_METERS: f32 = mechanic_core::MIN_LAYER_THICKNESS_METERS;

pub(super) const fn face_kind_on_axis(axis: usize, positive: bool) -> FaceKind {
    match (axis, positive) {
        (0, true) => FaceKind::PositiveX,
        (0, false) => FaceKind::NegativeX,
        (1, true) => FaceKind::PositiveY,
        (1, false) => FaceKind::NegativeY,
        (_, true) => FaceKind::PositiveZ,
        (_, false) => FaceKind::NegativeZ,
    }
}

/// Resolves a hit on a cuboid face, or on a full cylinder's outer wall, bore,
/// or end cap, into a layer target. Rounded and chamfered surfaces lie off
/// every such surface and take no layer.
pub(crate) fn layer_target_from_hit(
    graph: &ConstructionGraph,
    hit: SurfaceHit,
) -> Result<LayerTarget, PlacementError> {
    let FaceOwner::Part(part) = hit.face.owner else {
        return Err(PlacementError::NotLayerSurface);
    };
    if graph.region_of(part).is_some() {
        return Err(PlacementError::NotLayerSurface);
    }
    let spec = graph
        .part(part)
        .copied()
        .ok_or(PlacementError::NotLayerSurface)?;
    let frame = graph
        .part_frame(part)
        .ok_or(PlacementError::NotLayerSurface)?;
    let pose = spec.pose();
    let local = pose.rotation.quaternion().inverse()
        * (frame.inverse().point(hit.point) - pose.translation());
    let (face, local_normal) = match spec {
        PartSpec::Cuboid(cuboid) => {
            let half = cuboid.size_meters() * 0.5;
            let gap = |axis: usize| (half[axis] - local[axis].abs()).abs();
            let axis = (0..3)
                .min_by(|&first, &second| gap(first).total_cmp(&gap(second)))
                .expect("a cuboid has three axes");
            if gap(axis) > LAYER_FACE_TOLERANCE_METERS {
                return Err(PlacementError::NotLayerSurface);
            }
            let positive = local[axis] > 0.0;
            let mut normal = Vec3::ZERO;
            normal[axis] = if positive { 1.0 } else { -1.0 };
            (
                mechanic_core::LayerFace::Face(face_kind_on_axis(axis, positive)),
                normal,
            )
        }
        PartSpec::Cylinder(cylinder) if cylinder.dimensions.sweep_angle_degrees() == 360 => {
            let radius = Vec2::new(local.x, local.z).length();
            let outer = cylinder.dimensions.outer_diameter() * 0.5;
            let inner = cylinder.dimensions.inner_diameter() * 0.5;
            let half_length = cylinder.dimensions.axial_length() * 0.5;
            let wall_tolerance = LAYER_WALL_TOLERANCE_METERS + radius * 0.02;
            let radial = Vec3::new(local.x, 0.0, local.z).normalize_or_zero();
            let cap_gap = (local.y.abs() - half_length).abs();
            let wall_gap = (radius - outer).abs();
            let bore_gap = if inner > 0.0 {
                (radius - inner).abs()
            } else {
                f32::INFINITY
            };
            let within_length = local.y.abs() <= half_length + LAYER_FACE_TOLERANCE_METERS;
            if cap_gap <= LAYER_FACE_TOLERANCE_METERS
                && cap_gap <= wall_gap.min(bore_gap)
                && radius <= outer + wall_tolerance
                && radius + wall_tolerance >= inner
            {
                let positive = local.y > 0.0;
                (
                    mechanic_core::LayerFace::Face(face_kind_on_axis(1, positive)),
                    Vec3::Y * if positive { 1.0 } else { -1.0 },
                )
            } else if within_length && wall_gap <= wall_tolerance && wall_gap <= bore_gap {
                (mechanic_core::LayerFace::OuterWall, radial)
            } else if within_length && bore_gap <= wall_tolerance {
                (mechanic_core::LayerFace::Bore, -radial)
            } else {
                return Err(PlacementError::NotLayerSurface);
            }
        }
        _ => return Err(PlacementError::NotLayerSurface),
    };
    let members = match spec {
        PartSpec::Cuboid(cuboid) => flat_surface_members(graph, part, cuboid, frame, local_normal),
        _ => vec![LayerMember { part, spec, face }],
    };
    Ok(LayerTarget {
        part,
        spec,
        face,
        frame,
        anchor: hit.point,
        normal: frame
            .vector(pose.rotation.quaternion() * local_normal)
            .normalize_or_zero(),
        members,
    })
}

/// One press-and-drag choosing a new layer's thickness along the surface
/// normal, committed only on release.
///
/// Like a Shape amount drag, pointer motion is measured on a plane through
/// the anchor that contains the normal and faces the camera, and accumulates
/// relative to the press, so dragging out thickens and dragging back thins.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerDrag {
    pub(crate) target: LayerTarget,
    pub(crate) material: mechanic_core::ConstructionMaterial,
    pub(crate) appearance: mechanic_core::MaterialAppearance,
    /// Snapped thickness shown now and committed on release, in metres.
    pub(crate) thickness: f32,
    pub(super) drag_plane_normal: Vec3,
    pub(super) last_drag_value: f32,
    pub(super) raw_thickness: f32,
}

impl LayerDrag {
    pub(crate) fn begin(
        target: LayerTarget,
        material: mechanic_core::ConstructionMaterial,
        appearance: mechanic_core::MaterialAppearance,
        thickness: f32,
        ray_origin: Vec3,
        ray_direction: Vec3,
    ) -> Self {
        let pull = target.normal;
        let anchor = target.anchor;
        let drag_plane_normal = (ray_direction - pull * ray_direction.dot(pull))
            .try_normalize()
            .unwrap_or_else(|| pull.any_orthonormal_vector());
        let projected = crate::shape_tool::project_onto_plane(
            ray_origin,
            ray_direction,
            anchor,
            drag_plane_normal,
        )
        .unwrap_or(anchor);
        Self {
            target,
            material,
            appearance,
            thickness,
            drag_plane_normal,
            last_drag_value: (projected - anchor).dot(pull),
            raw_thickness: thickness,
        }
    }

    /// Follows the pointer ray and snaps to the active placement step, never
    /// thinner than one step.
    pub(crate) fn update(&mut self, grid: PlacementGrid, ray_origin: Vec3, ray_direction: Vec3) {
        let step = grid.step_meters();
        let pull = self.target.normal;
        if let Some(projected) = crate::shape_tool::project_onto_plane(
            ray_origin,
            ray_direction,
            self.target.anchor,
            self.drag_plane_normal,
        ) {
            let drag_value = (projected - self.target.anchor).dot(pull);
            let mut delta = drag_value - self.last_drag_value;
            self.last_drag_value = drag_value;
            let alignment = ray_direction.normalize_or_zero().dot(pull).abs();
            if 1.0 - alignment * alignment < LAYER_DRAG_STABILITY {
                delta = delta.clamp(-step, step);
            }
            self.raw_thickness = (self.raw_thickness + delta).max(0.0);
        }
        self.thickness = ((self.raw_thickness / step).round() * step).max(step);
    }

    /// Drops pointer distance past the thickest layer that fit, so holding
    /// beyond a neighbour does not retry the refused layer every frame.
    pub(crate) const fn discard_rejected_excess(&mut self, accepted: f32) {
        self.raw_thickness = accepted;
        self.thickness = accepted;
    }
}

/// Every member of the target with a new layer, the picked part first.
pub(crate) fn layered_parts(
    target: &LayerTarget,
    thickness: f32,
    material: mechanic_core::ConstructionMaterial,
    appearance: mechanic_core::MaterialAppearance,
) -> Result<Vec<(PartId, PartSpec)>, PlacementError> {
    target
        .members
        .iter()
        .map(|member| {
            member
                .spec
                .with_layer(member.face, thickness, material, appearance)
                .map(|spec| (member.part, spec))
                .map_err(|error| PlacementError::Graph(error.to_string()))
        })
        .collect()
}

/// Checks layered parts against bounds and every part outside the layer,
/// comparing in the target's construction frame.
pub(crate) fn validate_layered_parts(
    graph: &ConstructionGraph,
    target: &LayerTarget,
    layered: &[(PartId, PartSpec)],
    bounds: PlacementBounds,
) -> Result<(), PlacementError> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for &(_, spec) in layered {
        let (low, high) = part_world_bounds(spec);
        minimum = minimum.min(low);
        maximum = maximum.max(high);
    }
    if target.frame == mechanic_core::ConstructionFrame::IDENTITY {
        validate_world_bounds(minimum, maximum, bounds)?;
    }
    let into_layer = target.frame.inverse();
    for (part, existing) in graph.parts() {
        if layered.iter().any(|&(member, _)| member == part) {
            continue;
        }
        let existing_frame = graph
            .part_frame(part)
            .expect("validated parts have construction frames");
        if existing_frame == target.frame {
            // Most parts are nowhere near the layer; skip them cheaply.
            let (low, high) = part_world_bounds(*existing);
            if (low - maximum).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
                || (minimum - high).cmpgt(Vec3::splat(CONTACT_EPSILON)).any()
            {
                continue;
            }
        }
        let relative = into_layer.compose(existing_frame);
        if layered
            .iter()
            .any(|&(_, spec)| parts_overlap_with_frame(spec, *existing, relative))
        {
            return Err(PlacementError::OverlapsPart(part));
        }
    }
    Ok(())
}

/// Adds a layer to every target member in place in one edit, keeping their
/// connections.
pub(crate) fn stage_layer(
    graph: &ConstructionGraph,
    target: &LayerTarget,
    thickness: f32,
    material: mechanic_core::ConstructionMaterial,
    appearance: mechanic_core::MaterialAppearance,
    bounds: PlacementBounds,
) -> Result<(ConstructionGraph, Vec<(PartId, PartSpec)>), PlacementError> {
    let layered = layered_parts(target, thickness, material, appearance)?;
    validate_layered_parts(graph, target, &layered, bounds)?;
    let mut staged = graph.begin_edit();
    staged
        .apply_batch(
            layered
                .iter()
                .map(|&(part, spec)| BuildCommand::SetLayers { part, spec }),
        )
        .map_err(|error| PlacementError::Graph(error.to_string()))?;
    Ok((staged.finish(), layered))
}
