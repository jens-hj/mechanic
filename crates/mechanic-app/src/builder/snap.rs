//! The placement snap index and smart-snap guides.

use super::bounds::{
    composed_part_world_bounds, cuboid_world_bounds, part_world_bounds, parts_overlap_with_frame,
};
use super::candidates::{support_at_hit, support_geometries_from_hit, supporting_face_overlap};
use super::faces::{FaceGeometry, cylinder_face_geometry, face_geometry, overlap_center};
use super::grid::{cardinal_axis, rounded_position_tick};
use super::{
    CylinderPlacementCandidate, GRID_UNIT_METERS, IVec3, PlacementCandidate, PlacementGrid,
    PlacementPlane, Resource, SurfaceHit, Vec, Vec3,
};
use mechanic_core::{
    BuildPose, ConstructionGraph, CuboidSpec, CylinderSpec, GridDimension, POSITION_TICK_METERS,
    PartId, PartSpec,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub(super) const SNAP_BIN_METERS: f32 = 1.0;

pub(crate) const SMART_SNAP_CAPTURE_METERS: f32 = 0.025;

#[derive(Clone, Copy, Debug)]
pub(super) struct SnapTarget {
    pub(super) part: PartId,
    pub(super) spec: PartSpec,
    pub(super) frame: mechanic_core::ConstructionFrame,
    pub(super) minimum: Vec3,
    pub(super) maximum: Vec3,
}

/// Spatially binned committed solid-part bounds used by placement previews.
#[derive(Resource, Default)]
pub(crate) struct PlacementSnapIndex {
    pub(super) targets: Vec<SnapTarget>,
    pub(super) bins: HashMap<IVec3, Vec<usize>>,
}

impl PlacementSnapIndex {
    pub(crate) fn rebuild(&mut self, graph: &ConstructionGraph) {
        self.targets.clear();
        self.bins.clear();
        for (part, spec) in graph.parts() {
            let (minimum, maximum) = composed_part_world_bounds(graph, part)
                .expect("indexed parts have construction frames");
            let target_index = self.targets.len();
            self.targets.push(SnapTarget {
                part,
                spec: *spec,
                frame: graph
                    .part_frame(part)
                    .expect("indexed parts have construction frames"),
                minimum,
                maximum,
            });
            let low = (minimum / SNAP_BIN_METERS).floor().as_ivec3();
            let high = (maximum / SNAP_BIN_METERS).floor().as_ivec3();
            for x in low.x..=high.x {
                for y in low.y..=high.y {
                    for z in low.z..=high.z {
                        self.bins
                            .entry(IVec3::new(x, y, z))
                            .or_default()
                            .push(target_index);
                    }
                }
            }
        }
    }

    pub(super) fn nearby(&self, minimum: Vec3, maximum: Vec3, radius: f32) -> Vec<SnapTarget> {
        if self.targets.len() <= 16_384 {
            return self
                .targets
                .iter()
                .copied()
                .filter(|target| {
                    aabb_distance(minimum, maximum, target.minimum, target.maximum) <= radius
                })
                .collect();
        }
        let low = ((minimum - Vec3::splat(radius)) / SNAP_BIN_METERS)
            .floor()
            .as_ivec3();
        let high = ((maximum + Vec3::splat(radius)) / SNAP_BIN_METERS)
            .floor()
            .as_ivec3();
        let mut indices = HashSet::new();
        for x in low.x..=high.x {
            for y in low.y..=high.y {
                for z in low.z..=high.z {
                    if let Some(bin) = self.bins.get(&IVec3::new(x, y, z)) {
                        indices.extend(bin.iter().copied());
                    }
                }
            }
        }
        let mut targets = indices
            .into_iter()
            .filter_map(|index| self.targets.get(index).copied())
            .filter(|target| {
                aabb_distance(minimum, maximum, target.minimum, target.maximum) <= radius
            })
            .collect::<Vec<_>>();
        targets.sort_by_key(|target| target.part);
        targets
    }

    pub(crate) fn overlaps(&self, spec: PartSpec) -> bool {
        let (minimum, maximum) = part_world_bounds(spec);
        self.nearby(minimum, maximum, 0.0)
            .into_iter()
            .any(|target| parts_overlap_with_frame(spec, target.spec, target.frame))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum GuideKind {
    Center,
    Edge,
}

/// One active object-alignment line in local build coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SmartGuide {
    pub(crate) axis: usize,
    pub(crate) coordinate: f32,
    pub(crate) from: Vec3,
    pub(crate) to: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AxisGuide {
    pub(super) delta: f32,
    pub(super) coordinate: f32,
    pub(super) kind: GuideKind,
    pub(super) part: PartId,
    pub(super) target_center: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BlockEndpointGuide {
    pub(super) span: i32,
    pub(super) pointer_delta: f32,
    pub(super) guide: AxisGuide,
}

/// Lets nearby committed object centers and edges override the global grid.
pub(crate) fn smart_snap_cuboid_candidate(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    hit: SurfaceHit,
    gridded: PlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    valid: impl FnMut(PlacementCandidate) -> bool,
) -> (PlacementCandidate, Vec<SmartGuide>) {
    let supports = support_geometries_from_hit(graph, hit);
    smart_snap_cuboid_candidate_with_supports(index, hit, gridded, grid, range, &supports, valid)
}

pub(crate) fn smart_snap_cuboid_candidate_with_supports(
    index: &PlacementSnapIndex,
    hit: SurfaceHit,
    gridded: PlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    supports: &[FaceGeometry],
    mut valid: impl FnMut(PlacementCandidate) -> bool,
) -> (PlacementCandidate, Vec<SmartGuide>) {
    let Some(support) = support_at_hit(supports, hit.point) else {
        return (gridded, Vec::new());
    };
    let (normal_axis, _) = cardinal_axis(support.normal);
    let tangent_axes = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let (grid_minimum, grid_maximum) = cuboid_world_bounds(gridded.spec);
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(grid_minimum, grid_maximum, range + cell_margin);
    let choices = tangent_axes.map(|axis| {
        axis_guides(
            axis,
            grid_minimum,
            grid_maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for axis_guides in guide_combinations(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (choice_index, guide) in axis_guides.iter().enumerate() {
            if let Some(guide) = guide {
                let axis = tangent_axes[choice_index];
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CuboidSpec::new(
            gridded.spec.dimensions.map(GridDimension::units),
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .expect("an aligned candidate remains dimensionally valid")
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate_face = face_geometry(spec, gridded.attached_face);
        let candidate = PlacementCandidate {
            spec,
            attached_face: gridded.attached_face,
            anchor: supports
                .iter()
                .find_map(|support| overlap_center(support, &candidate_face)),
            support: gridded.support,
        };
        if valid(candidate) {
            let center = candidate.spec.pose.translation();
            let rendered =
                render_smart_guides(axis_guides, &choices, tangent_axes, normal_axis, center);
            return (candidate, rendered);
        }
    }
    (gridded, Vec::new())
}

/// Cylinder counterpart of [`smart_snap_cuboid_candidate`].
pub(crate) fn smart_snap_cylinder_candidate(
    graph: &ConstructionGraph,
    index: &PlacementSnapIndex,
    hit: SurfaceHit,
    gridded: CylinderPlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(CylinderPlacementCandidate) -> bool,
) -> (CylinderPlacementCandidate, Vec<SmartGuide>) {
    let supports = support_geometries_from_hit(graph, hit);
    let Some(support) = support_at_hit(&supports, hit.point) else {
        return (gridded, Vec::new());
    };
    let (normal_axis, _) = cardinal_axis(support.normal);
    let tangent_axes = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let (grid_minimum, grid_maximum) = part_world_bounds(PartSpec::Cylinder(gridded.spec));
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(grid_minimum, grid_maximum, range + cell_margin);
    let choices = tangent_axes.map(|axis| {
        axis_guides(
            axis,
            grid_minimum,
            grid_maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for axis_guides in guide_combinations(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (choice_index, guide) in axis_guides.iter().enumerate() {
            if let Some(guide) = guide {
                let axis = tangent_axes[choice_index];
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CylinderSpec::new(
            gridded.spec.dimensions,
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate_face = cylinder_face_geometry(spec, gridded.attached_face)
            .expect("a cylinder placement uses a flat axial face");
        let candidate = CylinderPlacementCandidate {
            spec,
            attached_face: gridded.attached_face,
            anchor: supports
                .iter()
                .find_map(|support| overlap_center(support, &candidate_face))
                .or_else(|| supporting_face_overlap(graph, support, &candidate_face)),
            support: gridded.support,
        };
        if valid(candidate) {
            let center = candidate.spec.pose.translation();
            let rendered =
                render_smart_guides(axis_guides, &choices, tangent_axes, normal_axis, center);
            return (candidate, rendered);
        }
    }
    (gridded, Vec::new())
}

/// Three-axis object snapping for candidates that are not constrained to a face.
pub(crate) fn smart_snap_free_cuboid_candidate(
    index: &PlacementSnapIndex,
    gridded: PlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(PlacementCandidate) -> bool,
) -> (PlacementCandidate, Vec<SmartGuide>) {
    let (minimum, maximum) = cuboid_world_bounds(gridded.spec);
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(minimum, maximum, range + cell_margin);
    let choices = [0, 1, 2].map(|axis| {
        axis_guides(
            axis,
            minimum,
            maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for selected in guide_combinations_3(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (axis, guide) in selected.iter().enumerate() {
            if let Some(guide) = guide {
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CuboidSpec::new(
            gridded.spec.dimensions.map(GridDimension::units),
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .expect("an aligned free candidate remains dimensionally valid")
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate = PlacementCandidate { spec, ..gridded };
        if valid(candidate) {
            return (
                candidate,
                render_free_smart_guides(selected, &choices, spec.pose.translation()),
            );
        }
    }
    (gridded, Vec::new())
}

/// Cylinder counterpart of [`smart_snap_free_cuboid_candidate`].
pub(crate) fn smart_snap_free_cylinder_candidate(
    index: &PlacementSnapIndex,
    gridded: CylinderPlacementCandidate,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(CylinderPlacementCandidate) -> bool,
) -> (CylinderPlacementCandidate, Vec<SmartGuide>) {
    let (minimum, maximum) = part_world_bounds(PartSpec::Cylinder(gridded.spec));
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(minimum, maximum, range + cell_margin);
    let choices = [0, 1, 2].map(|axis| {
        axis_guides(
            axis,
            minimum,
            maximum,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for selected in guide_combinations_3(&choices) {
        let mut ticks = gridded.spec.pose.translation_position_ticks();
        for (axis, guide) in selected.iter().enumerate() {
            if let Some(guide) = guide {
                ticks[axis] += rounded_position_tick(guide.delta);
            }
        }
        let spec = CylinderSpec::new(
            gridded.spec.dimensions,
            BuildPose::from_position_ticks(ticks, gridded.spec.pose.rotation),
        )
        .with_material(gridded.spec.material)
        .with_appearance(gridded.spec.appearance);
        let candidate = CylinderPlacementCandidate { spec, ..gridded };
        if valid(candidate) {
            return (
                candidate,
                render_free_smart_guides(selected, &choices, spec.pose.translation()),
            );
        }
    }
    (gridded, Vec::new())
}

/// Point-sized smart snapping for bearing anchors constrained to a support face.
pub(crate) fn smart_snap_anchor(
    index: &PlacementSnapIndex,
    gridded: Vec3,
    normal_axis: usize,
    grid: PlacementGrid,
    range: f32,
    mut valid: impl FnMut(Vec3) -> bool,
) -> (Vec3, Vec<SmartGuide>) {
    let tangent_axes = match normal_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let cell_margin = smart_snap_cell_margin(grid);
    let targets = index.nearby(gridded, gridded, range + cell_margin);
    let choices = tangent_axes.map(|axis| {
        axis_guides(
            axis,
            gridded,
            gridded,
            &targets,
            SMART_SNAP_CAPTURE_METERS + cell_margin,
        )
    });
    for axis_guides in guide_combinations(&choices) {
        let mut anchor = gridded;
        for (choice_index, guide) in axis_guides.iter().enumerate() {
            if let Some(guide) = guide {
                let axis = tangent_axes[choice_index];
                anchor[axis] += guide.delta;
            }
        }
        anchor = (anchor / POSITION_TICK_METERS).round() * POSITION_TICK_METERS;
        anchor[normal_axis] = gridded[normal_axis];
        if valid(anchor) {
            return (
                anchor,
                render_smart_guides(axis_guides, &choices, tangent_axes, normal_axis, anchor),
            );
        }
    }
    (gridded, Vec::new())
}

/// Lets the opposite block of a plane drag acquire nearby object alignments.
///
/// The starting block remains sovereign: a guide may only select another whole
/// block span on its 25 cm lattice. Guides are then rebuilt from that final span
/// so their visibility cannot change while the selected plane stays unchanged.
pub(crate) fn smart_snap_block_span(
    index: &PlacementSnapIndex,
    start: CuboidSpec,
    plane: PlacementPlane,
    gridded_span: IVec3,
    pointer: Vec3,
    range: f32,
    mut valid: impl FnMut(IVec3) -> bool,
) -> (IVec3, Vec<SmartGuide>) {
    let block_size = f32::from(start.dimensions[0].units()) * GRID_UNIT_METERS;
    let targets = index.nearby(pointer, pointer, range);
    let tangent_axes = plane.tangent_axes();
    let choices = tangent_axes
        .map(|axis| block_endpoint_axis_guides(axis, start, pointer, block_size, &targets));

    for selected in block_endpoint_guide_combinations(&choices) {
        let mut span = gridded_span;
        for (choice_index, guide) in selected.iter().enumerate() {
            if let Some(guide) = guide {
                span[tangent_axes[choice_index]] = guide.span;
            }
        }
        if valid(span) {
            return (
                span,
                block_endpoint_guides(index, start, span, plane, range),
            );
        }
    }

    if valid(gridded_span) {
        (
            gridded_span,
            block_endpoint_guides(index, start, gridded_span, plane, range),
        )
    } else {
        (gridded_span, Vec::new())
    }
}

pub(super) fn block_endpoint_axis_guides(
    axis: usize,
    start: CuboidSpec,
    pointer: Vec3,
    block_size: f32,
    targets: &[SnapTarget],
) -> Vec<BlockEndpointGuide> {
    let start_center = start.pose.translation()[axis];
    let mut choices = Vec::new();
    for target in targets {
        let target_center = (target.minimum + target.maximum) * 0.5;
        if let Some(span) = integral_block_span(start_center, target_center[axis], block_size) {
            push_block_endpoint_choice(
                &mut choices,
                span,
                pointer[axis],
                target_center[axis],
                GuideKind::Center,
                *target,
            );
        }
        for target_edge in [target.minimum[axis], target.maximum[axis]] {
            for direction in [-1.0, 1.0] {
                let endpoint_center = target_edge - direction * block_size * 0.5;
                let Some(span) = integral_block_span(start_center, endpoint_center, block_size)
                else {
                    continue;
                };
                if span != 0 && (span.is_positive() != direction.is_sign_positive()) {
                    continue;
                }
                push_block_endpoint_choice(
                    &mut choices,
                    span,
                    pointer[axis],
                    target_edge,
                    GuideKind::Edge,
                    *target,
                );
            }
        }
    }
    choices.sort_by(|left, right| {
        left.pointer_delta
            .abs()
            .total_cmp(&right.pointer_delta.abs())
            .then_with(|| left.guide.kind.cmp(&right.guide.kind))
            .then_with(|| left.guide.part.cmp(&right.guide.part))
            .then_with(|| left.span.cmp(&right.span))
    });
    let mut distinct = Vec::new();
    for choice in choices {
        if distinct
            .iter()
            .all(|existing: &BlockEndpointGuide| existing.span != choice.span)
        {
            distinct.push(choice);
        }
    }
    distinct
}

pub(super) fn push_block_endpoint_choice(
    choices: &mut Vec<BlockEndpointGuide>,
    span: i32,
    pointer_coordinate: f32,
    guide_coordinate: f32,
    kind: GuideKind,
    target: SnapTarget,
) {
    let pointer_delta = guide_coordinate - pointer_coordinate;
    if pointer_delta.abs() > SMART_SNAP_CAPTURE_METERS {
        return;
    }
    choices.push(BlockEndpointGuide {
        span,
        pointer_delta,
        guide: AxisGuide {
            delta: pointer_delta,
            coordinate: guide_coordinate,
            kind,
            part: target.part,
            target_center: (target.minimum + target.maximum) * 0.5,
        },
    });
}

#[expect(clippy::cast_possible_truncation)]
pub(super) fn integral_block_span(start: f32, endpoint: f32, block_size: f32) -> Option<i32> {
    let steps = ((endpoint - start) / block_size).round();
    let span = steps as i32;
    ((start + steps * block_size - endpoint).abs() <= POSITION_TICK_METERS * 0.5).then_some(span)
}

pub(super) fn block_endpoint_guide_combinations(
    choices: &[Vec<BlockEndpointGuide>; 2],
) -> Vec<[Option<BlockEndpointGuide>; 2]> {
    let mut combinations = Vec::new();
    for first in choices[0].iter().copied().map(Some).chain([None]) {
        for second in choices[1].iter().copied().map(Some).chain([None]) {
            if first.is_none() && second.is_none() {
                continue;
            }
            let guides = [first, second];
            let displacement = guides
                .iter()
                .flatten()
                .map(|guide| guide.pointer_delta * guide.pointer_delta)
                .sum::<f32>()
                .sqrt();
            let center_count = guides
                .iter()
                .flatten()
                .filter(|guide| guide.guide.kind == GuideKind::Center)
                .count();
            let identities = guides.map(|guide| {
                guide.map_or((u32::MAX, u32::MAX), |guide| {
                    (guide.guide.part.index(), guide.guide.part.generation())
                })
            });
            combinations.push((guides, displacement, center_count, identities));
        }
    }
    combinations.sort_by(|left, right| {
        let left_count = left.0.iter().flatten().count();
        let right_count = right.0.iter().flatten().count();
        right_count
            .cmp(&left_count)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    combinations
        .into_iter()
        .map(|(guides, _, _, _)| guides)
        .collect()
}

pub(super) fn block_endpoint_guides(
    index: &PlacementSnapIndex,
    start: CuboidSpec,
    span: IVec3,
    plane: PlacementPlane,
    range: f32,
) -> Vec<SmartGuide> {
    let block_size = f32::from(start.dimensions[0].units()) * GRID_UNIT_METERS;
    let center = start.pose.translation() + span.as_vec3() * block_size;
    let half = Vec3::splat(block_size * 0.5);
    let targets = index.nearby(center - half, center + half, range);
    let tangent_axes = plane.tangent_axes();
    let normal_axis = plane.normal_axis();
    let mut rendered = Vec::new();

    for (choice_index, axis) in tangent_axes.into_iter().enumerate() {
        let line_axis = tangent_axes[1 - choice_index];
        let endpoint_edges: &[f32] = match span[axis].cmp(&0) {
            Ordering::Less => &[center[axis] - block_size * 0.5],
            Ordering::Greater => &[center[axis] + block_size * 0.5],
            Ordering::Equal => &[
                center[axis] - block_size * 0.5,
                center[axis] + block_size * 0.5,
            ],
        };
        for target in &targets {
            let target_center = (target.minimum + target.maximum) * 0.5;
            let mut coordinates = Vec::new();
            if (target_center[axis] - center[axis]).abs() <= POSITION_TICK_METERS * 0.5 {
                coordinates.push(target_center[axis]);
            }
            for endpoint_edge in endpoint_edges {
                for target_edge in [target.minimum[axis], target.maximum[axis]] {
                    if (target_edge - endpoint_edge).abs() <= POSITION_TICK_METERS * 0.5 {
                        coordinates.push(target_edge);
                    }
                }
            }
            for coordinate in coordinates {
                let mut from = center;
                let mut to = center;
                from[axis] = coordinate;
                to[axis] = coordinate;
                to[line_axis] = target_center[line_axis];
                to[normal_axis] = from[normal_axis];
                let guide = SmartGuide {
                    axis,
                    coordinate,
                    from,
                    to,
                };
                if !rendered.contains(&guide) {
                    rendered.push(guide);
                }
            }
        }
    }
    rendered
}

pub(super) fn smart_snap_cell_margin(grid: PlacementGrid) -> f32 {
    grid.step_meters() * 0.5
}

pub(super) fn guide_combinations(choices: &[Vec<AxisGuide>; 2]) -> Vec<[Option<AxisGuide>; 2]> {
    let mut combinations = Vec::new();
    for first in choices[0].iter().copied().map(Some).chain([None]) {
        for second in choices[1].iter().copied().map(Some).chain([None]) {
            if first.is_none() && second.is_none() {
                continue;
            }
            let guides = [first, second];
            let displacement = guides
                .iter()
                .flatten()
                .map(|guide| guide.delta * guide.delta)
                .sum::<f32>()
                .sqrt();
            let center_count = guides
                .iter()
                .flatten()
                .filter(|guide| guide.kind == GuideKind::Center)
                .count();
            let identities = guides.map(|guide| {
                guide.map_or((u32::MAX, u32::MAX), |guide| {
                    (guide.part.index(), guide.part.generation())
                })
            });
            combinations.push((guides, displacement, center_count, identities));
        }
    }
    combinations.sort_by(|left, right| {
        let left_count = left.0.iter().flatten().count();
        let right_count = right.0.iter().flatten().count();
        right_count
            .cmp(&left_count)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    // A multi-block region can contribute the same guide once per block. Keep
    // the best-ranked guide for each resulting pose; rendering still receives
    // the original choices and can show every coincident alignment line.
    let mut positions = HashSet::new();
    combinations
        .into_iter()
        .filter(|(guides, _, _, _)| {
            positions.insert(
                guides.map(|guide| guide.map_or(0, |guide| rounded_position_tick(guide.delta))),
            )
        })
        .map(|(guides, _, _, _)| guides)
        .collect()
}

pub(super) fn guide_combinations_3(choices: &[Vec<AxisGuide>; 3]) -> Vec<[Option<AxisGuide>; 3]> {
    let mut combinations = Vec::new();
    for x in choices[0].iter().copied().map(Some).chain([None]) {
        for y in choices[1].iter().copied().map(Some).chain([None]) {
            for z in choices[2].iter().copied().map(Some).chain([None]) {
                if x.is_none() && y.is_none() && z.is_none() {
                    continue;
                }
                let guides = [x, y, z];
                let displacement = guides
                    .iter()
                    .flatten()
                    .map(|guide| guide.delta * guide.delta)
                    .sum::<f32>()
                    .sqrt();
                let center_count = guides
                    .iter()
                    .flatten()
                    .filter(|guide| guide.kind == GuideKind::Center)
                    .count();
                let identities = guides.map(|guide| {
                    guide.map_or((u32::MAX, u32::MAX), |guide| {
                        (guide.part.index(), guide.part.generation())
                    })
                });
                combinations.push((guides, displacement, center_count, identities));
            }
        }
    }
    combinations.sort_by(|left, right| {
        let left_count = left.0.iter().flatten().count();
        let right_count = right.0.iter().flatten().count();
        right_count
            .cmp(&left_count)
            .then_with(|| left.1.total_cmp(&right.1))
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    // Free placement has the same duplicate-guide multiplier on three axes.
    let mut positions = HashSet::new();
    combinations
        .into_iter()
        .filter(|(guides, _, _, _)| {
            positions.insert(
                guides.map(|guide| guide.map_or(0, |guide| rounded_position_tick(guide.delta))),
            )
        })
        .map(|(guides, _, _, _)| guides)
        .collect()
}

pub(super) fn render_free_smart_guides(
    selected: [Option<AxisGuide>; 3],
    choices: &[Vec<AxisGuide>; 3],
    center: Vec3,
) -> Vec<SmartGuide> {
    let mut rendered = Vec::new();
    for (axis, selected) in selected.iter().enumerate() {
        let Some(selected) = selected else {
            continue;
        };
        for guide in &choices[axis] {
            if guide.kind != selected.kind
                || (guide.delta - selected.delta).abs() > POSITION_TICK_METERS * 0.5
                || (guide.coordinate - selected.coordinate).abs() > POSITION_TICK_METERS * 0.5
            {
                continue;
            }
            let mut from = center;
            from[axis] = guide.coordinate;
            let mut to = guide.target_center;
            to[axis] = guide.coordinate;
            let delta = (to - from).abs();
            let varying_axes = [delta.x, delta.y, delta.z]
                .into_iter()
                .filter(|component| *component > POSITION_TICK_METERS * 0.5)
                .count();
            if varying_axes != 1 {
                continue;
            }
            rendered.push(SmartGuide {
                axis,
                coordinate: guide.coordinate,
                from,
                to,
            });
        }
    }
    rendered
}

pub(super) fn render_smart_guides(
    selected: [Option<AxisGuide>; 2],
    choices: &[Vec<AxisGuide>; 2],
    tangent_axes: [usize; 2],
    normal_axis: usize,
    center: Vec3,
) -> Vec<SmartGuide> {
    let mut rendered = Vec::new();
    for (choice_index, selected) in selected.iter().enumerate() {
        let Some(selected) = selected else {
            continue;
        };
        let axis = tangent_axes[choice_index];
        let line_axis = tangent_axes[1 - choice_index];
        for guide in &choices[choice_index] {
            if guide.kind != selected.kind
                || (guide.delta - selected.delta).abs() > POSITION_TICK_METERS * 0.5
                || (guide.coordinate - selected.coordinate).abs() > POSITION_TICK_METERS * 0.5
            {
                continue;
            }
            let mut from = center;
            let mut to = center;
            from[axis] = guide.coordinate;
            to[axis] = guide.coordinate;
            to[line_axis] = guide.target_center[line_axis];
            to[normal_axis] = from[normal_axis];
            rendered.push(SmartGuide {
                axis,
                coordinate: guide.coordinate,
                from,
                to,
            });
        }
    }
    rendered
}

pub(super) fn axis_guides(
    axis: usize,
    raw_minimum: Vec3,
    raw_maximum: Vec3,
    targets: &[SnapTarget],
    capture: f32,
) -> Vec<AxisGuide> {
    let raw_center = f32::midpoint(raw_minimum[axis], raw_maximum[axis]);
    let mut guides = Vec::new();
    for target in targets {
        let target_center = (target.minimum + target.maximum) * 0.5;
        let center_delta = target_center[axis] - raw_center;
        if center_delta.abs() <= capture {
            guides.push(AxisGuide {
                delta: center_delta,
                coordinate: target_center[axis],
                kind: GuideKind::Center,
                part: target.part,
                target_center,
            });
        }
        for (raw_edge, target_edge) in
            [raw_minimum[axis], raw_maximum[axis]]
                .into_iter()
                .flat_map(|raw_edge| {
                    [target.minimum[axis], target.maximum[axis]]
                        .into_iter()
                        .map(move |target_edge| (raw_edge, target_edge))
                })
        {
            let delta = target_edge - raw_edge;
            if delta.abs() <= capture {
                guides.push(AxisGuide {
                    delta,
                    coordinate: target_edge,
                    kind: GuideKind::Edge,
                    part: target.part,
                    target_center,
                });
            }
        }
    }
    guides.sort_by(|left, right| {
        left.delta
            .abs()
            .total_cmp(&right.delta.abs())
            .then_with(|| left.kind.cmp(&right.kind))
            .then_with(|| left.part.cmp(&right.part))
    });
    guides.dedup_by(|left, right| {
        (left.delta - right.delta).abs() <= POSITION_TICK_METERS * 0.5
            && left.kind == right.kind
            && left.part == right.part
    });
    guides
}

pub(super) fn aabb_distance(
    first_min: Vec3,
    first_max: Vec3,
    second_min: Vec3,
    second_max: Vec3,
) -> f32 {
    let separation = (first_min - second_max)
        .max(second_min - first_max)
        .max(Vec3::ZERO);
    separation.length()
}
