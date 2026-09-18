//! The Shape tool: choosing an editable area, then moving its cage.
//!
//! Nothing can be shaped until a region is selected — a solid cuboid of blocks
//! picked out with the same drag the Block tool uses. Once one is, only its
//! cage moves, and no vertex may leave the region's original bounding box, so a
//! corner can only ever be drawn inward.
//!
//! Everything here works in integer lattice steps. Movement is constrained to a
//! fraction of a block rather than running free, so two corners line up because
//! they landed on the same sub-grid rather than because they were matched by
//! eye.

use bevy::prelude::*;
use mechanic_core::{
    CageIndex, EdgeChainRef, EdgeTreatment, EvaluatedSolid, POSITION_TICK_METERS,
    POSITION_TICKS_PER_GRID_UNIT, ShapeFeatureId, ShapeRegion, SolidOwner,
};

/// Active Shape workflow selected from the hold-Tab context wheel.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum ShapeEditMode {
    /// Select or move Shape-region cage vertices.
    #[default]
    Vertex,
    /// Apply a symmetric equal-setback cut to logical edge chains.
    Chamfer,
    /// Apply a constant-radius polygonal round to logical edge chains.
    Fillet,
}

impl ShapeEditMode {
    pub(crate) const ALL: [Self; 3] = [Self::Vertex, Self::Chamfer, Self::Fillet];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Vertex => "Vertex",
            Self::Chamfer => "Chamfer",
            Self::Fillet => "Fillet",
        }
    }
}

/// How close the pointer ray must pass to a cage vertex to grab it, in metres.
const VERTEX_PICK_RADIUS: f32 = 0.05;

/// How near the pointer must come before a cage vertex fades in, in metres.
pub(crate) const VERTEX_REVEAL_RADIUS: f32 = 1.2;

/// How close the pointer must come to an edge to be offered a new vertex there.
const EDGE_PICK_RADIUS: f32 = 0.06;

/// A logical feature edge under the pointer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FeatureEdgeHit {
    /// Stable complete-chain reference.
    pub(crate) target: EdgeChainRef,
    /// Closest point on the hovered tessellated segment.
    pub(crate) point: Vec3,
    /// Segment tangent used to orient the fallback drag plane.
    pub(crate) tangent: Vec3,
    /// Inward cross-section bisector along which amount increases.
    pub(crate) bisector: Vec3,
    /// Ray distance, for choosing between overlapping edges.
    pub(crate) distance: f32,
}

/// One chamfer/fillet amount drag, committed only on release.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FeatureDrag {
    pub(crate) validated_preview: Option<ValidatedFeaturePreview>,
    /// Existing feature being adjusted, or `None` for a new feature.
    pub(crate) feature: Option<ShapeFeatureId>,
    /// Every selected logical chain receiving the shared amount.
    pub(crate) targets: Vec<EdgeChainRef>,
    /// Profile selected by the active Shape mode.
    pub(crate) treatment: EdgeTreatment,
    /// Amount at press time.
    pub(crate) start_amount_ticks: u32,
    /// Current quantized preview amount.
    pub(crate) amount_ticks: u32,
    anchor: Vec3,
    drag_plane_normal: Vec3,
    bisector: Vec3,
    last_drag_value: f32,
    raw_amount_ticks: f64,
}

impl FeatureDrag {
    pub(crate) fn begin(
        hit: FeatureEdgeHit,
        targets: Vec<EdgeChainRef>,
        treatment: EdgeTreatment,
        feature: Option<ShapeFeatureId>,
        start_amount_ticks: u32,
        ray_origin: Vec3,
        ray_direction: Vec3,
    ) -> Self {
        let drag_plane_normal = (ray_direction - hit.bisector * ray_direction.dot(hit.bisector))
            .try_normalize()
            .or_else(|| hit.tangent.cross(hit.bisector).try_normalize())
            .unwrap_or(hit.tangent);
        let projected = project_onto_plane(ray_origin, ray_direction, hit.point, drag_plane_normal)
            .unwrap_or(hit.point);
        Self {
            validated_preview: None,
            feature,
            targets,
            treatment,
            start_amount_ticks,
            amount_ticks: start_amount_ticks,
            anchor: hit.point,
            drag_plane_normal,
            bisector: hit.bisector,
            last_drag_value: (projected - hit.point).dot(hit.bisector),
            raw_amount_ticks: f64::from(start_amount_ticks),
        }
    }

    /// Quantized positive amount proposed by the current pointer ray.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn proposed_amount(
        &mut self,
        snap: ShapeSnap,
        ray_origin: Vec3,
        ray_direction: Vec3,
    ) -> u32 {
        let Some(projected) = project_onto_plane(
            ray_origin,
            ray_direction,
            self.anchor,
            self.drag_plane_normal,
        ) else {
            return self.amount_ticks;
        };
        let drag_value = (projected - self.anchor).dot(self.bisector);
        let delta = f64::from((drag_value - self.last_drag_value) / POSITION_TICK_METERS);
        self.last_drag_value = drag_value;

        // Keep ordinary coalesced pointer motion: fillet previews can take
        // long enough that one frame legitimately spans several snap steps.
        // Only rate-limit a ray that is nearly parallel to the pull direction,
        // where its projected distance is ill-conditioned.
        let alignment = f64::from(ray_direction.normalize_or_zero().dot(self.bisector).abs());
        let projection_stability = 1.0 - alignment * alignment;
        let delta = if projection_stability >= 0.25 {
            delta
        } else {
            let maximum_delta = f64::from(snap.steps);
            delta.clamp(-maximum_delta, maximum_delta)
        };
        self.raw_amount_ticks = (self.raw_amount_ticks + delta).max(0.0);
        let raw = self.raw_amount_ticks.round() as i64;
        let increment = i64::from(snap.steps);
        let quantized = ((raw + increment / 2) / increment) * increment;
        u32::try_from(quantized.max(0)).unwrap_or(u32::MAX)
    }

    /// Drops pointer distance that geometry validation could not accept.
    ///
    /// Without this, holding still beyond the valid limit retries the same
    /// expensive failed preview every frame.
    pub(crate) fn discard_rejected_excess(&mut self, accepted_amount_ticks: u32) {
        self.raw_amount_ticks = f64::from(accepted_amount_ticks);
    }
}

/// A successfully validated drag step, reusable while its input revision and
/// feature parameters match. Geometry in the graph itself is shared cheaply.
#[derive(Clone, Debug)]
pub(crate) struct ValidatedFeaturePreview {
    pub(crate) source: mechanic_core::ConstructionGraph,
    pub(crate) graph: mechanic_core::ConstructionGraph,
    pub(crate) key: super::FeaturePreviewKey,
}

impl PartialEq for ValidatedFeaturePreview {
    fn eq(&self, other: &Self) -> bool {
        self.source.shares_revision(&other.source)
            && self.graph.shares_revision(&other.graph)
            && self.key == other.key
    }
}

/// Finds the nearest selectable logical edge, never a tessellation seam.
pub(crate) fn hovered_feature_edge(
    solid: &EvaluatedSolid,
    owner: SolidOwner,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<FeatureEdgeHit> {
    hovered_feature_edge_matching(solid, owner, ray_origin, ray_direction, |_| true)
}

/// Hit-tests only one stored source chain, used by dashed virtual overlays.
pub(crate) fn hovered_source_edge(
    solid: &EvaluatedSolid,
    target: EdgeChainRef,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<FeatureEdgeHit> {
    hovered_feature_edge_matching(solid, target.owner, ray_origin, ray_direction, |key| {
        key == target.edge
    })
}

/// Returns the ray entry distance for the solid's edge-pick-inflated bounds.
/// This lets silhouette edges be acquired before an ordinary surface hit
/// exists, including a pointer ray that passes just outside a cylinder rim.
#[cfg(test)]
pub(crate) fn inflated_bounds_ray_distance(
    solid: &EvaluatedSolid,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<f32> {
    let first = solid.vertices.first()?.position;
    let (mut minimum, mut maximum) = (first, first);
    for vertex in &solid.vertices[1..] {
        minimum = minimum.min(vertex.position);
        maximum = maximum.max(vertex.position);
    }
    inflated_aabb_ray_distance(minimum, maximum, ray_origin, ray_direction)
}

/// Returns the ray entry distance for an axis-aligned box enlarged by the
/// edge-pick radius.
pub(crate) fn inflated_aabb_ray_distance(
    mut minimum: Vec3,
    mut maximum: Vec3,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<f32> {
    minimum -= Vec3::splat(EDGE_PICK_RADIUS);
    maximum += Vec3::splat(EDGE_PICK_RADIUS);

    let mut entry = 0.0_f32;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let origin = ray_origin[axis];
        let direction = ray_direction[axis];
        if direction.abs() <= 1.0e-6 {
            if origin < minimum[axis] || origin > maximum[axis] {
                return None;
            }
            continue;
        }
        let first = (minimum[axis] - origin) / direction;
        let second = (maximum[axis] - origin) / direction;
        entry = entry.max(first.min(second));
        exit = exit.min(first.max(second));
        if exit < entry {
            return None;
        }
    }
    (exit >= 0.0).then_some(entry)
}

fn hovered_feature_edge_matching(
    solid: &EvaluatedSolid,
    owner: SolidOwner,
    ray_origin: Vec3,
    ray_direction: Vec3,
    accepts: impl Fn(mechanic_core::TopologyKey) -> bool,
) -> Option<FeatureEdgeHit> {
    let mut best = None::<FeatureEdgeHit>;
    for logical in &solid.logical_edges {
        if !logical.convex || !accepts(logical.key) {
            continue;
        }
        for &half_edge_index in &logical.half_edges {
            let half_edge = solid.half_edges[half_edge_index as usize];
            let next = solid.half_edges[half_edge.next as usize];
            let a = solid.vertices[half_edge.origin as usize].position;
            let b = solid.vertices[next.origin as usize].position;
            let (point, along, distance) = closest_ray_segment(ray_origin, ray_direction, a, b);
            if distance > EDGE_PICK_RADIUS {
                continue;
            }
            let first_normal = solid.surfaces[half_edge.face as usize].normal;
            let twin = solid.half_edges[half_edge.twin as usize];
            let second_normal = solid.surfaces[twin.face as usize].normal;
            let tangent = (b - a).normalize_or_zero();
            let bisector = -(first_normal + second_normal).normalize_or_zero();
            if tangent == Vec3::ZERO || bisector == Vec3::ZERO {
                continue;
            }
            let candidate = FeatureEdgeHit {
                target: EdgeChainRef {
                    owner,
                    edge: logical.key,
                },
                point,
                tangent,
                bisector,
                distance,
            };
            if best.is_none_or(|current| {
                along < (current.point - ray_origin).dot(ray_direction) - 1.0e-4
                    || (along <= (current.point - ray_origin).dot(ray_direction) + 1.0e-4
                        && distance < current.distance)
            }) {
                best = Some(candidate);
            }
        }
    }
    best
}

fn closest_ray_segment(
    ray_origin: Vec3,
    ray_direction: Vec3,
    a: Vec3,
    b: Vec3,
) -> (Vec3, f32, f32) {
    let segment = b - a;
    let offset = ray_origin - a;
    let aa = ray_direction.length_squared();
    let bb = ray_direction.dot(segment);
    let cc = segment.length_squared();
    let dd = ray_direction.dot(offset);
    let ee = segment.dot(offset);
    let denominator = aa * cc - bb * bb;
    let mut ray_t = if denominator.abs() > f32::EPSILON {
        (bb * ee - cc * dd) / denominator
    } else {
        -dd / aa
    };
    ray_t = ray_t.max(0.0);
    let segment_t = ((bb * ray_t + ee) / cc).clamp(0.0, 1.0);
    ray_t = ((bb * segment_t - dd) / aa).max(0.0);
    let ray_point = ray_origin + ray_direction * ray_t;
    let segment_point = a + segment * segment_t;
    (segment_point, ray_t, ray_point.distance(segment_point))
}

/// How far one vertex move travels, in lattice steps.
///
/// Free movement at the lattice's own 2.5 mm resolution is too loose to line
/// anything up: two corners only meet if the user hits the same value twice by
/// eye. Constraining every move to a fraction of a cell makes matching corners
/// the default outcome instead of a careful act.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShapeSnap {
    /// Step count one move covers. Always divides a whole cell.
    pub(crate) steps: i32,
}

impl Default for ShapeSnap {
    fn default() -> Self {
        // A quarter cell: coarse enough that corners line up on their own, fine
        // enough to shape with.
        Self {
            steps: POSITION_TICKS_PER_GRID_UNIT / 4,
        }
    }
}

impl ShapeSnap {
    /// Increments offered, coarsest first.
    const CHOICES: [i32; 5] = [
        POSITION_TICKS_PER_GRID_UNIT,
        POSITION_TICKS_PER_GRID_UNIT / 2,
        POSITION_TICKS_PER_GRID_UNIT / 4,
        POSITION_TICKS_PER_GRID_UNIT / 5,
        POSITION_TICKS_PER_GRID_UNIT / 20,
    ];

    /// Five-centimetre increment used when entering Chamfer or Fillet mode.
    pub(crate) const fn feature_default() -> Self {
        Self {
            steps: POSITION_TICKS_PER_GRID_UNIT / 5,
        }
    }

    /// Moves to the next increment, wrapping back to the coarsest.
    pub(crate) fn cycle(&mut self) {
        let next = Self::CHOICES
            .iter()
            .position(|&steps| steps == self.steps)
            .map_or(0, |index| (index + 1) % Self::CHOICES.len());
        self.steps = Self::CHOICES[next];
    }

    pub(crate) fn label(self) -> String {
        match self.steps {
            steps if steps == POSITION_TICKS_PER_GRID_UNIT => "Snap: 1 block".to_owned(),
            steps if steps == POSITION_TICKS_PER_GRID_UNIT / 2 => "Snap: 1/2 block".to_owned(),
            steps if steps == POSITION_TICKS_PER_GRID_UNIT / 4 => "Snap: 1/4 block".to_owned(),
            steps if steps == POSITION_TICKS_PER_GRID_UNIT / 5 => "Snap: 5 cm".to_owned(),
            _ => format!(
                "Snap: fine ({:.1} mm)",
                f64::from(self.steps) * f64::from(POSITION_TICK_METERS) * 1000.0
            ),
        }
    }

    /// Rounds one offset onto this increment.
    fn quantise(self, value: i32) -> i32 {
        let half = self.steps / 2;
        let rounded = if value >= 0 {
            (value + half) / self.steps
        } else {
            (value - half) / self.steps
        };
        rounded * self.steps
    }

    /// The next increment line strictly beyond `value` in `direction`.
    ///
    /// Starting off-increment pulls onto the grid rather than carrying the
    /// stray amount along, so a nudged corner ends up somewhere its neighbour
    /// can be sent too.
    fn step_from(self, value: i32, direction: i32) -> i32 {
        let index = value.div_euclid(self.steps);
        let on_line = value.rem_euclid(self.steps) == 0;
        if direction > 0 {
            (index + 1) * self.steps
        } else if on_line {
            (index - 1) * self.steps
        } else {
            index * self.steps
        }
    }
}

/// Which of a region's own centre planes mirror an edit.
///
/// Mirroring pairs opposite cage columns, so it is exactly symmetric on a cage
/// whose planes are evenly spaced — which is every cage until it is subdivided
/// off-centre.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShapeMirror {
    /// Mirror across the region's centre plane on x.
    pub(crate) x: bool,
    /// Mirror across the region's centre plane on z.
    pub(crate) z: bool,
}

impl ShapeMirror {
    pub(crate) fn label(self) -> String {
        match (self.x, self.z) {
            (false, false) => "Mirror off".to_owned(),
            (true, false) => "Mirror X".to_owned(),
            (false, true) => "Mirror Z".to_owned(),
            (true, true) => "Mirror X+Z".to_owned(),
        }
    }
}

/// A cage-vertex drag in progress.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VertexDrag {
    /// Vertex being moved.
    pub(crate) index: CageIndex,
    /// Offset the vertex had when the drag began.
    pub(crate) start_offset: [i16; 3],
    /// Where the vertex sat in the world when the drag began.
    pub(crate) start_position: Vec3,
    /// The one world axis this segment of the drag may change.
    pub(crate) axis: usize,
    /// Offset at the most recent axis change.
    anchor_offset: [i16; 3],
    /// Vertex position at the most recent axis change.
    anchor_position: Vec3,
    /// Plane normal the pointer is projected onto.
    plane_normal: Vec3,
    /// Where the pointer first met that plane.
    grab_point: Vec3,
    /// Offset the drag currently proposes.
    pub(crate) offset: [i16; 3],
    /// Other selected vertices moving with it, and the offsets they started
    /// from. A whole roofline is one drag rather than twelve.
    pub(crate) group: Vec<(CageIndex, [i16; 3])>,
}

impl VertexDrag {
    /// World position proposed for the primary vertex.
    pub(crate) fn position(&self) -> Vec3 {
        let delta = Vec3::from_array(self.offset.map(f32::from))
            - Vec3::from_array(self.start_offset.map(f32::from));
        self.start_position + delta * POSITION_TICK_METERS
    }

    /// Changes the movement axis and starts measuring this segment at the
    /// pointer's current position, so cycling never makes the vertex jump.
    pub(crate) fn cycle_axis(&mut self, ray_origin: Vec3, ray_direction: Vec3) {
        self.axis = (self.axis + 1) % 3;
        self.anchor_offset = self.offset;
        self.anchor_position = self.position();
        self.plane_normal = -ray_direction;
        self.grab_point = project_onto_plane(
            ray_origin,
            ray_direction,
            self.anchor_position,
            self.plane_normal,
        )
        .unwrap_or(self.anchor_position);
    }

    pub(crate) const fn axis_label(&self) -> &'static str {
        match self.axis {
            0 => "X",
            1 => "Y",
            _ => "Z",
        }
    }
}

/// Where one cage vertex sits, in metres.
pub(crate) fn vertex_position(region: &ShapeRegion, index: CageIndex) -> Option<Vec3> {
    region
        .vertex_steps(index)
        .map(|steps| steps.as_vec3() * POSITION_TICK_METERS)
}

/// Cage vertices close enough to the pointer to be drawn, with how near each is.
pub(crate) fn revealed_vertices(
    region: &ShapeRegion,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Vec<(CageIndex, Vec3, f32)> {
    region
        .vertices()
        .filter_map(|index| {
            let position = vertex_position(region, index)?;
            let distance = ray_distance(position, ray_origin, ray_direction)?;
            (distance <= VERTEX_REVEAL_RADIUS).then_some((index, position, distance))
        })
        .collect()
}

/// The cage vertex the pointer is over, if any.
pub(crate) fn hovered_vertex(
    region: &ShapeRegion,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<CageIndex> {
    let mut best: Option<(CageIndex, f32, f32)> = None;
    for index in region.vertices() {
        let Some(position) = vertex_position(region, index) else {
            continue;
        };
        let Some(distance) = ray_distance(position, ray_origin, ray_direction) else {
            continue;
        };
        if distance > VERTEX_PICK_RADIUS {
            continue;
        }
        let along = (position - ray_origin).dot(ray_direction);
        // Prefer the nearest vertex to the camera, then the best-centred one.
        if best.is_none_or(|(_, best_along, best_distance)| {
            along < best_along - 1.0e-4 || (along < best_along + 1.0e-4 && distance < best_distance)
        }) {
            best = Some((index, along, distance));
        }
    }
    best.map(|(index, _, _)| index)
}

/// Perpendicular distance from a point to a forward ray.
fn ray_distance(point: Vec3, ray_origin: Vec3, ray_direction: Vec3) -> Option<f32> {
    let offset = point - ray_origin;
    let along = offset.dot(ray_direction);
    (along > 0.0).then(|| (offset - ray_direction * along).length())
}

/// A new cage plane the pointer is close enough to be offered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EdgeInsertion {
    /// Axis the new plane splits.
    pub(crate) axis: usize,
    /// Position along that axis, in cells from the region origin.
    pub(crate) position: i32,
    /// Where the new vertex would appear.
    pub(crate) at: Vec3,
}

/// The grid position along a cage edge nearest the pointer, when one is close.
///
/// Only whole grid positions that are not already cage planes are offered, so an
/// inserted vertex lands somewhere its neighbours can be sent too.
pub(crate) fn edge_insertion(
    region: &ShapeRegion,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<EdgeInsertion> {
    let counts = region.plane_counts();
    let last = [counts[0] - 1, counts[1] - 1, counts[2] - 1];
    let size = region.size_cells();
    let mut best: Option<(EdgeInsertion, f32)> = None;

    for axis in 0..3 {
        let others = [(axis + 1) % 3, (axis + 2) % 3];
        // The four edges running along this axis sit on the cage's corners in
        // the other two.
        for first in [0_usize, last[others[0]]] {
            for second in [0_usize, last[others[1]]] {
                for position in 1..size[axis] {
                    if plane_exists(region, axis, position) {
                        continue;
                    }
                    let Some(at) = edge_point(region, axis, position, others, first, second) else {
                        continue;
                    };
                    let Some(distance) = ray_distance(at, ray_origin, ray_direction) else {
                        continue;
                    };
                    if distance > EDGE_PICK_RADIUS {
                        continue;
                    }
                    if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                        best = Some((EdgeInsertion { axis, position, at }, distance));
                    }
                }
            }
        }
    }
    best.map(|(insertion, _)| insertion)
}

fn plane_exists(region: &ShapeRegion, axis: usize, position: i32) -> bool {
    let grid = region.grid();
    let planes = grid.planes(axis);
    let origin = planes[0];
    planes
        .iter()
        .any(|half_units| (half_units - origin) / 2 == position)
}

/// Interpolates along one cage edge to where a new vertex would appear.
fn edge_point(
    region: &ShapeRegion,
    axis: usize,
    position: i32,
    others: [usize; 2],
    first: usize,
    second: usize,
) -> Option<Vec3> {
    let planes = region.grid();
    let origin = planes.planes(axis)[0];
    let cells = planes
        .planes(axis)
        .iter()
        .map(|half_units| (half_units - origin) / 2)
        .collect::<Vec<_>>();
    let upper = cells.iter().position(|&cell| cell > position)?;
    let (low, high) = (cells[upper - 1], cells[upper]);
    #[allow(clippy::cast_precision_loss)] // Cell counts are small.
    let blend = (position - low) as f32 / (high - low) as f32;

    let mut low_index = [0_u16; 3];
    low_index[axis] = u16::try_from(upper - 1).ok()?;
    low_index[others[0]] = u16::try_from(first).ok()?;
    low_index[others[1]] = u16::try_from(second).ok()?;
    let mut high_index = low_index;
    high_index[axis] = u16::try_from(upper).ok()?;

    let from = vertex_position(region, low_index)?;
    let to = vertex_position(region, high_index)?;
    Some(from.lerp(to, blend))
}

/// Starts a drag on `index`, carrying `selected` along with it.
pub(crate) fn begin_group_drag(
    region: &ShapeRegion,
    index: CageIndex,
    selected: &[CageIndex],
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> VertexDrag {
    let start_position = vertex_position(region, index).unwrap_or_default();
    let start_offset = region.offset(index);
    // Begin with the axis that reads most clearly in the current view. The
    // pointer is still measured on a camera-facing plane, but only travel along
    // this one axis is accepted.
    let axis = most_visible_axis(ray_direction);
    let plane_normal = -ray_direction;
    let grab_point = project_onto_plane(ray_origin, ray_direction, start_position, plane_normal)
        .unwrap_or(start_position);
    let group = if selected.contains(&index) {
        selected
            .iter()
            .filter(|&&other| other != index)
            .map(|&other| (other, region.offset(other)))
            .collect()
    } else {
        Vec::new()
    };
    VertexDrag {
        index,
        start_offset,
        start_position,
        axis,
        anchor_offset: start_offset,
        anchor_position: start_position,
        plane_normal,
        grab_point,
        offset: start_offset,
        group,
    }
}

/// Chooses the world axis with the largest screen projection.
fn most_visible_axis(ray_direction: Vec3) -> usize {
    let alignment = ray_direction.abs();
    if alignment.x <= alignment.y && alignment.x <= alignment.z {
        0
    } else if alignment.y <= alignment.z {
        1
    } else {
        2
    }
}

/// Advances a drag to the current pointer ray, returning the proposed offset.
pub(crate) fn drag_offset(
    region: &ShapeRegion,
    drag: &VertexDrag,
    snap: ShapeSnap,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> [i16; 3] {
    let Some(point) = project_onto_plane(
        ray_origin,
        ray_direction,
        drag.anchor_position,
        drag.plane_normal,
    ) else {
        return drag.offset;
    };
    let travel = (point - drag.grab_point) / POSITION_TICK_METERS;
    let mut proposed = drag.anchor_offset.map(i32::from);
    proposed[drag.axis] =
        snap.quantise(i32::from(drag.anchor_offset[drag.axis]) + round_to_i32(travel[drag.axis]));
    clamp_into_region(region, drag.index, proposed)
}

#[allow(clippy::cast_possible_truncation)] // Travel is bounded by the region.
fn round_to_i32(value: f32) -> i32 {
    value.round().clamp(-4096.0, 4096.0) as i32
}

/// Holds an offset inside the region's original bounding box.
///
/// This is the whole clamp: a corner can only be drawn inward, which is what
/// stops one region from growing into its neighbours.
pub(crate) fn clamp_into_region(
    region: &ShapeRegion,
    index: CageIndex,
    offset: [i32; 3],
) -> [i16; 3] {
    let Some(base) = region.base_steps(index) else {
        return [0; 3];
    };
    let (minimum, maximum) = region.bounds_steps();
    let mut clamped = [0_i16; 3];
    for axis in 0..3 {
        let wanted = base[axis] + offset[axis];
        let held = wanted.clamp(minimum[axis], maximum[axis]);
        clamped[axis] = i16::try_from(held - base[axis]).unwrap_or(0);
    }
    clamped
}

pub(crate) fn project_onto_plane(
    ray_origin: Vec3,
    ray_direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denominator = ray_direction.dot(plane_normal);
    if denominator.abs() < 1.0e-6 {
        return None;
    }
    let distance = (plane_point - ray_origin).dot(plane_normal) / denominator;
    (distance > 0.0).then(|| ray_origin + ray_direction * distance)
}

/// Every edit a finished drag implies, companions and mirrors included.
pub(crate) fn drag_edits(
    region: &ShapeRegion,
    drag: &VertexDrag,
    mirror: ShapeMirror,
) -> Vec<(CageIndex, [i16; 3])> {
    let delta = [
        i32::from(drag.offset[0]) - i32::from(drag.start_offset[0]),
        i32::from(drag.offset[1]) - i32::from(drag.start_offset[1]),
        i32::from(drag.offset[2]) - i32::from(drag.start_offset[2]),
    ];
    let mut edits = mirrored_edits(region, drag.index, drag.offset, mirror);
    for &(index, start) in &drag.group {
        let moved = clamp_into_region(
            region,
            index,
            [
                i32::from(start[0]) + delta[0],
                i32::from(start[1]) + delta[1],
                i32::from(start[2]) + delta[2],
            ],
        );
        edits.extend(mirrored_edits(region, index, moved, mirror));
    }
    edits
}

/// Moves every selected vertex one increment along one world axis.
pub(crate) fn nudge_edits(
    region: &ShapeRegion,
    selected: &[CageIndex],
    axis: usize,
    direction: i32,
    snap: ShapeSnap,
    mirror: ShapeMirror,
) -> Vec<(CageIndex, [i16; 3])> {
    let mut edits = Vec::new();
    for &index in selected {
        let current = region.offset(index);
        let mut wanted = [
            i32::from(current[0]),
            i32::from(current[1]),
            i32::from(current[2]),
        ];
        wanted[axis] = snap.step_from(wanted[axis], direction);
        let moved = clamp_into_region(region, index, wanted);
        if moved == current {
            continue;
        }
        edits.extend(mirrored_edits(region, index, moved, mirror));
    }
    edits
}

/// Expands one vertex edit across the region's active mirror planes.
///
/// A vertex on an active centre plane keeps its offset along that normal at
/// zero, so it cannot leave the plane and break the symmetry.
pub(crate) fn mirrored_edits(
    region: &ShapeRegion,
    index: CageIndex,
    offset: [i16; 3],
    mirror: ShapeMirror,
) -> Vec<(CageIndex, [i16; 3])> {
    let counts = region.plane_counts();
    let last = [
        u16::try_from(counts[0] - 1).unwrap_or(0),
        u16::try_from(counts[1] - 1).unwrap_or(0),
        u16::try_from(counts[2] - 1).unwrap_or(0),
    ];
    let on_centre = |axis: usize| index[axis] * 2 == last[axis];

    let mut offset = offset;
    if mirror.x && on_centre(0) {
        offset[0] = 0;
    }
    if mirror.z && on_centre(2) {
        offset[2] = 0;
    }

    let mut edits = vec![(index, offset)];
    let mirror_x = |index: CageIndex| [last[0] - index[0], index[1], index[2]];
    let mirror_z = |index: CageIndex| [index[0], index[1], last[2] - index[2]];
    if mirror.x && !on_centre(0) {
        edits.push((mirror_x(index), [-offset[0], offset[1], offset[2]]));
    }
    if mirror.z && !on_centre(2) {
        edits.push((mirror_z(index), [offset[0], offset[1], -offset[2]]));
    }
    if mirror.x && mirror.z && !on_centre(0) && !on_centre(2) {
        edits.push((
            mirror_z(mirror_x(index)),
            [-offset[0], offset[1], -offset[2]],
        ));
    }
    edits
}

/// The world axis a screen direction points most nearly along, and its sign.
pub(crate) fn screen_axis(direction: Vec3) -> (usize, i32) {
    let magnitude = direction.abs();
    let axis = if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
        0
    } else if magnitude.y >= magnitude.z {
        1
    } else {
        2
    };
    (axis, if direction[axis] >= 0.0 { 1 } else { -1 })
}

/// Marker size for a cage vertex at this distance, so nearer ones read stronger.
pub(crate) fn vertex_marker_size(distance: f32) -> f32 {
    let fade = (1.0 - distance / VERTEX_REVEAL_RADIUS).clamp(0.0, 1.0);
    0.012 + 0.018 * fade
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests;
