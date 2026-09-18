//! Pipe runs: node placement, leg extension, bends, and drag validation.

use crate::builder::{
    PipeNode, PipeRunPiece, PlacementBounds, PlacementError, pipe_run_pieces,
    validate_pipe_run_in_bounds,
};
use crate::camera;
use crate::editor::dimensions::CYLINDER_DIAMETER_STEP;
use crate::editor::hover::{
    BearingOffsetDrag, BlockAttachment, DRAG_DEAD_ZONE_RADIANS, PointerSample,
    refresh_bearing_offset_drag,
};
use crate::editor::state::EditorState;
use bevy::prelude::{Vec2, Vec3, format};
use mechanic_core::{
    ConstructionGraph, ConstructionMaterial, CylinderDimensions, GRID_UNIT_METERS,
    MAX_CYLINDER_OUTER_DIAMETER, MIN_CYLINDER_DIAMETER_GAP, MIN_CYLINDER_OUTER_DIAMETER,
    MaterialAppearance, PartSpec, PipeBendDimensions,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PipeEditMode {
    #[default]
    Length,
    OuterDiameter,
    InnerDiameter,
}

impl PipeEditMode {
    pub(crate) const fn next(self) -> Self {
        match self {
            Self::Length => Self::OuterDiameter,
            Self::OuterDiameter => Self::InnerDiameter,
            Self::InnerDiameter => Self::Length,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Length => "Length",
            Self::OuterDiameter => "Outer diameter",
            Self::InnerDiameter => "Inner diameter",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PipeDrag {
    pub(crate) attachment: BlockAttachment,
    pub(crate) start: Vec3,
    /// Sharp bend corners, each at the middle of its leg's last channel cell.
    pub(crate) corners: Vec<Vec3>,
    pub(crate) endpoint: Vec3,
    pub(crate) directions: Vec<Vec3>,
    /// Bend or junction joining each pair of legs.
    pub(crate) nodes: Vec<PipeNode>,
    pub(crate) pending_span: u8,
    /// Junction to make where this run branches off a pipe, applied on release.
    pub(crate) branch: Option<crate::builder::PipeBranch>,
    pub(crate) dimensions: CylinderDimensions,
    pub(crate) material: ConstructionMaterial,
    pub(crate) appearance: MaterialAppearance,
    pub(crate) mode: PipeEditMode,
    /// Initial bearing attachments drag laterally across the bearing plane.
    /// Rotating the edit mode or adding a bend ends this one-shot placement mode.
    pub(crate) bearing_offset: Option<BearingOffsetDrag>,
    pub(crate) choosing_direction: bool,
    pub(crate) press: PointerSample,
    pub(crate) anchor_endpoint: Vec3,
    pub(crate) anchor_dimensions: CylinderDimensions,
    pub(crate) pieces: Vec<PipeRunPiece>,
    pub(crate) error: Option<PlacementError>,
}

pub(crate) struct PipeValidation {
    pub(crate) graph: ConstructionGraph,
    pub(crate) pieces: Vec<PipeRunPiece>,
    pub(crate) bounds: PlacementBounds,
    pub(crate) result: Result<(), PlacementError>,
}

impl PipeValidation {
    pub(crate) fn validate(
        cached: &mut Option<Self>,
        graph: &ConstructionGraph,
        pieces: &[PipeRunPiece],
        bounds: PlacementBounds,
    ) -> Result<(), PlacementError> {
        if let Some(previous) = cached
            && previous.graph.shares_revision(graph)
            && previous.pieces == pieces
            && previous.bounds == bounds
        {
            return previous.result.clone();
        }
        let result = validate_pipe_run_in_bounds(graph, pieces, bounds);
        *cached = Some(Self {
            graph: graph.clone(),
            pieces: pieces.to_vec(),
            bounds,
            result: result.clone(),
        });
        result
    }
}

pub(crate) fn begin_pipe_node(graph: &ConstructionGraph, state: &mut EditorState) -> String {
    let Some(drag) = state.pipe_drag.as_mut() else {
        return "Hold primary while dragging a pipe before adding a bend".to_owned();
    };
    if drag.choosing_direction {
        return "Aim toward one of the four perpendicular turn directions".to_owned();
    }
    if drag.dimensions.sweep_angle_degrees() != 360 {
        return "Partial-cylinder sectors support straight runs only".to_owned();
    }
    drag.bearing_offset = None;
    let pending = PipeNode::Bend {
        span: drag.pending_span,
    };
    extend_pipe_leg_for_node(drag, pending);
    drag.choosing_direction = true;
    drag.anchor_endpoint = drag.endpoint;
    if let Some((cursor, (ray_origin, ray_direction))) =
        state.pointer_position.zip(state.pointer_ray)
    {
        drag.press = PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        };
    }
    rebuild_pipe_drag(graph, state);
    "Endpoint frozen — aim toward a perpendicular arrow; wheel changes bend size".to_owned()
}

/// Lengthens the current leg to the blocks its fittings need, so a fitting can
/// go on straight away, even right at the start of the run.
pub(crate) fn extend_pipe_leg_for_node(drag: &mut PipeDrag, pending: PipeNode) {
    let outer_diameter = drag.dimensions.outer_diameter();
    let required = drag
        .nodes
        .last()
        .map_or(0, |node| node.footprint_blocks(outer_diameter))
        + pending.footprint_blocks(outer_diameter);
    if pipe_leg_blocks(drag) >= f32::from(required) - 1.0e-3 {
        return;
    }
    let leg_start = drag.corners.last().copied().unwrap_or(drag.start);
    let start_inset = if drag.corners.is_empty() {
        0.0
    } else {
        pipe_corner_inset(outer_diameter)
    };
    let direction = *drag
        .directions
        .last()
        .expect("a pipe run has one direction");
    drag.endpoint = leg_start + direction * (f32::from(required) * GRID_UNIT_METERS - start_inset);
    drag.anchor_endpoint = drag.endpoint;
}

/// Distance from a leg's block boundary back to the corner of its bend: half
/// the pipe's channel, so the corner sits at the middle of the last cell.
pub(crate) fn pipe_corner_inset(outer_diameter: f32) -> f32 {
    f32::from(PipeBendDimensions::channel_blocks(outer_diameter)) * GRID_UNIT_METERS * 0.5
}

/// Snaps the current leg's endpoint to whole blocks for a dragged length
/// measured from the leg start. Legs after a bend count blocks from the bend's
/// square edge, one corner inset behind the corner, and never end inside it.
pub(crate) fn pipe_leg_endpoint(drag: &PipeDrag, dragged_length: f32) -> Vec3 {
    let leg_start = drag.corners.last().copied().unwrap_or(drag.start);
    let direction = *drag
        .directions
        .last()
        .expect("a pipe run has one direction");
    let inset = if drag.corners.is_empty() {
        0.0
    } else {
        pipe_corner_inset(drag.dimensions.outer_diameter())
    };
    let minimum = drag.nodes.last().map_or(1.0, |node| {
        f32::from(node.footprint_blocks(drag.dimensions.outer_diameter()))
    });
    let blocks = ((dragged_length + inset) / GRID_UNIT_METERS)
        .round()
        .clamp(minimum, 32.0);
    leg_start + direction * (blocks * GRID_UNIT_METERS - inset)
}

/// Whole blocks from the current leg's start boundary to its endpoint.
pub(crate) fn pipe_leg_blocks(drag: &PipeDrag) -> f32 {
    let leg_start = drag.corners.last().copied().unwrap_or(drag.start);
    let start_inset = if drag.corners.is_empty() {
        0.0
    } else {
        pipe_corner_inset(drag.dimensions.outer_diameter())
    };
    ((drag.endpoint.distance(leg_start) + start_inset) / GRID_UNIT_METERS).round()
}

/// Rebuilds corners and endpoint for a new corner inset, keeping each leg's
/// block count but never letting a leg get shorter than its bends need.
pub(crate) fn rebase_pipe_path(
    start: Vec3,
    corners: &mut [Vec3],
    endpoint: &mut Vec3,
    directions: &[Vec3],
    spans: &[u8],
    old_inset: f32,
    new_inset: f32,
) {
    let mut old_leg_start = start;
    let mut new_leg_start = start;
    for (index, corner) in corners.iter_mut().enumerate() {
        let (old_start_inset, new_start_inset) = if index == 0 {
            (0.0, 0.0)
        } else {
            (old_inset, new_inset)
        };
        let required = index.checked_sub(1).map_or(0, |previous| spans[previous]) + spans[index];
        let blocks = ((corner.distance(old_leg_start) + old_start_inset + old_inset)
            / GRID_UNIT_METERS)
            .round()
            .max(f32::from(required));
        old_leg_start = *corner;
        *corner = new_leg_start
            + directions[index] * (blocks * GRID_UNIT_METERS - new_start_inset - new_inset);
        new_leg_start = *corner;
    }
    let last = corners.len();
    let (old_start_inset, new_start_inset) = if last == 0 {
        (0.0, 0.0)
    } else {
        (old_inset, new_inset)
    };
    let required = spans.last().copied().unwrap_or(1);
    let blocks = ((endpoint.distance(old_leg_start) + old_start_inset) / GRID_UNIT_METERS)
        .round()
        .max(f32::from(required));
    *endpoint = new_leg_start + directions[last] * (blocks * GRID_UNIT_METERS - new_start_inset);
}

pub(crate) fn refresh_pipe_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    if refresh_bearing_offset_drag(graph, state, ray_origin, ray_direction) {
        return;
    }
    let Some(snapshot) = state.pipe_drag.as_ref().map(|drag| {
        (
            drag.press,
            drag.mode,
            drag.choosing_direction,
            drag.anchor_endpoint,
            drag.anchor_dimensions,
            drag.corners.last().copied().unwrap_or(drag.start),
            *drag
                .directions
                .last()
                .expect("a pipe run has one direction"),
        )
    }) else {
        return;
    };
    let (press, mode, choosing, anchor_endpoint, anchor_dimensions, leg_start, direction) =
        snapshot;
    if choosing {
        if let Some(outgoing) = pipe_turn_direction(direction, press.ray_direction, ray_direction) {
            let sample = PointerSample {
                cursor,
                ray_origin,
                ray_direction,
            };
            lock_pipe_node(graph, state, direction, outgoing, sample);
        }
        return;
    }
    if !camera::ray_drag_started(press.ray_direction, ray_direction) {
        return;
    }
    let drag = state.pipe_drag.as_mut().expect("pipe drag remains active");
    match mode {
        PipeEditMode::Length => {
            let Some(press_parameter) =
                closest_axis_parameter(leg_start, direction, press.ray_origin, press.ray_direction)
            else {
                invalidate_pipe_drag(state, PlacementError::DragPlaneUnavailable);
                return;
            };
            let Some(current_parameter) =
                closest_axis_parameter(leg_start, direction, ray_origin, ray_direction)
            else {
                invalidate_pipe_drag(state, PlacementError::DragPlaneUnavailable);
                return;
            };
            drag.endpoint = pipe_leg_endpoint(
                drag,
                anchor_endpoint.distance(leg_start) + current_parameter - press_parameter,
            );
        }
        PipeEditMode::OuterDiameter | PipeEditMode::InnerDiameter => {
            let delta = pipe_pointer_delta(press.ray_direction, ray_direction);
            let old_inset = pipe_corner_inset(drag.dimensions.outer_diameter());
            drag.dimensions = pipe_drag_dimensions(mode, anchor_dimensions, delta);
            let outer_diameter = drag.dimensions.outer_diameter();
            drag.pending_span =
                constrained_pipe_bend_span(outer_diameter, i16::from(drag.pending_span));
            for PipeNode::Bend { span } in &mut drag.nodes {
                *span = constrained_pipe_bend_span(outer_diameter, i16::from(*span));
            }
            let footprints = drag
                .nodes
                .iter()
                .map(|node| node.footprint_blocks(outer_diameter))
                .collect::<Vec<_>>();
            rebase_pipe_path(
                drag.start,
                &mut drag.corners,
                &mut drag.endpoint,
                &drag.directions,
                &footprints,
                old_inset,
                pipe_corner_inset(outer_diameter),
            );
        }
    }
    rebuild_pipe_drag(graph, state);
}

/// Places the bend or junction being chosen and starts dragging the next leg.
pub(crate) fn lock_pipe_node(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    incoming: Vec3,
    outgoing: Vec3,
    sample: PointerSample,
) {
    let drag = state.pipe_drag.as_mut().expect("pipe drag remains active");
    let outer_diameter = drag.dimensions.outer_diameter();
    let inset = pipe_corner_inset(outer_diameter);
    let corner = drag.endpoint - incoming * inset;
    let node = PipeNode::Bend {
        span: drag.pending_span,
    };
    let blocks = (drag.dimensions.axial_length() / GRID_UNIT_METERS)
        .round()
        .max(f32::from(node.footprint_blocks(outer_diameter)));
    drag.corners.push(corner);
    drag.nodes.push(node);
    drag.directions.push(outgoing);
    drag.endpoint = corner + outgoing * (blocks * GRID_UNIT_METERS - inset);
    drag.anchor_endpoint = drag.endpoint;
    drag.anchor_dimensions = drag.dimensions;
    drag.press = sample;
    drag.choosing_direction = false;
    rebuild_pipe_drag(graph, state);
    let PipeNode::Bend { span } = node;
    state.feedback = Some(format!(
        "Turn locked; dragging next leg — bend {span} × {span} blocks"
    ));
}

pub(crate) fn pipe_drag_dimensions(
    mode: PipeEditMode,
    anchor: CylinderDimensions,
    pointer_delta: f32,
) -> CylinderDimensions {
    let steps = (pointer_delta / 0.005).round();
    let mut outer = anchor.outer_diameter();
    let mut inner = anchor.inner_diameter();
    match mode {
        PipeEditMode::OuterDiameter => {
            outer = (outer + steps * CYLINDER_DIAMETER_STEP)
                .clamp(MIN_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_OUTER_DIAMETER);
            inner = inner.min(outer - MIN_CYLINDER_DIAMETER_GAP);
        }
        PipeEditMode::InnerDiameter => {
            inner = (inner + steps * CYLINDER_DIAMETER_STEP)
                .clamp(0.0, outer - MIN_CYLINDER_DIAMETER_GAP);
        }
        PipeEditMode::Length => unreachable!("length dragging does not resize a pipe diameter"),
    }
    CylinderDimensions::new(outer, inner, anchor.axial_length())
        .expect("clamped pipe dimensions remain valid")
        .with_sweep_angle_degrees(anchor.sweep_angle_degrees())
        .expect("the existing sector sweep remains valid")
}

pub(crate) fn rebuild_pipe_drag(graph: &ConstructionGraph, state: &mut EditorState) {
    let (points, nodes, dimensions, material, appearance) = {
        let drag = state.pipe_drag.as_ref().expect("pipe drag remains active");
        let mut points = Vec::with_capacity(drag.corners.len() + 2);
        points.push(drag.start);
        points.extend(drag.corners.iter().copied());
        points.push(drag.endpoint);
        (
            points,
            drag.nodes.clone(),
            drag.dimensions,
            drag.material,
            drag.appearance,
        )
    };
    let result = pipe_run_pieces(&points, &nodes, dimensions, material).and_then(|mut pieces| {
        for piece in &mut pieces {
            piece.spec = ordinary_part_with_appearance(piece.spec, appearance);
        }
        PipeValidation::validate(
            &mut state.pipe_validation,
            graph,
            &pieces,
            state.placement_bounds,
        )?;
        Ok(pieces)
    });
    let drag = state.pipe_drag.as_mut().expect("pipe drag remains active");
    match result {
        Ok(pieces) => {
            drag.pieces = pieces;
            drag.error = None;
            state.preview_error = None;
        }
        Err(error) => {
            drag.error = Some(error.clone());
            state.preview_error = Some(error);
        }
    }
}

pub(crate) fn ordinary_part_with_appearance(
    spec: PartSpec,
    appearance: MaterialAppearance,
) -> PartSpec {
    match spec {
        PartSpec::Cuboid(spec) => PartSpec::Cuboid(spec.with_appearance(appearance)),
        PartSpec::Cylinder(spec) => PartSpec::Cylinder(spec.with_appearance(appearance)),
        PartSpec::PipeBend(spec) => PartSpec::PipeBend(spec.with_appearance(appearance)),
        authored => authored,
    }
}

pub(crate) fn invalidate_pipe_drag(state: &mut EditorState, error: PlacementError) {
    let drag = state
        .pipe_drag
        .as_mut()
        .expect("pipe drag was checked by caller");
    drag.error = Some(error.clone());
    state.preview_error = Some(error);
}

pub(crate) fn closest_axis_parameter(
    axis_origin: Vec3,
    axis_direction: Vec3,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<f32> {
    let ray_direction = ray_direction.normalize();
    let offset = axis_origin - ray_origin;
    let parallel = axis_direction.dot(ray_direction);
    let denominator = 1.0 - parallel * parallel;
    (denominator > 1.0e-5)
        .then(|| (-axis_direction.dot(offset) + parallel * ray_direction.dot(offset)) / denominator)
}

pub(crate) fn pipe_pointer_delta(anchor: Vec3, current: Vec3) -> f32 {
    let pitch = current.y.asin() - anchor.y.asin();
    let yaw = wrap_angle(current.x.atan2(current.z) - anchor.x.atan2(anchor.z));
    if yaw.abs() >= pitch.abs() {
        yaw
    } else {
        -pitch
    }
}

pub(crate) fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Picks the cardinal direction the pointer aims at since the anchor ray.
/// Bends choose among the four perpendiculars of the incoming leg.
pub(crate) fn pipe_turn_direction(
    incoming: Vec3,
    anchor_ray: Vec3,
    current_ray: Vec3,
) -> Option<Vec3> {
    let anchor_ray = anchor_ray.normalize();
    let current_ray = current_ray.normalize();
    let mut aim = current_ray - anchor_ray * current_ray.dot(anchor_ray);
    aim -= incoming * aim.dot(incoming);
    if aim.length() < DRAG_DEAD_ZONE_RADIANS {
        return None;
    }
    let selected = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ]
    .into_iter()
    .filter(|candidate| candidate.dot(incoming).abs() < 0.5)
    .max_by(|left, right| left.dot(aim).total_cmp(&right.dot(aim)))?;
    (selected.dot(aim) >= DRAG_DEAD_ZONE_RADIANS).then_some(selected)
}

pub(crate) fn adjust_pipe_bend_span(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    direction: i8,
) -> (u8, String) {
    let drag = state
        .pipe_drag
        .as_mut()
        .expect("bend size adjustment requires an active pipe drag");
    let outer_diameter = drag.dimensions.outer_diameter();
    let inner_diameter = drag.dimensions.inner_diameter();
    let minimum = PipeBendDimensions::minimum_span(outer_diameter);
    let latest_bend = if drag.choosing_direction {
        None
    } else {
        drag.nodes.last().map(|&PipeNode::Bend { span }| span)
    };
    let current = latest_bend.unwrap_or(drag.pending_span);
    let requested = i16::from(current) + i16::from(direction);
    let span = constrained_pipe_bend_span(outer_diameter, requested);
    if latest_bend.is_some()
        && let Some(PipeNode::Bend { span: latest }) = drag.nodes.last_mut()
    {
        *latest = span;
    }
    drag.pending_span = span;
    if drag.choosing_direction {
        extend_pipe_leg_for_node(drag, PipeNode::Bend { span });
    }
    rebuild_pipe_drag(graph, state);
    let message = if requested < i16::from(minimum) {
        format!("Bend clamped to minimum {minimum} × {minimum} blocks for this diameter")
    } else if requested > i16::from(mechanic_core::MAX_GRID_UNITS) {
        format!(
            "Bend clamped to maximum {0} × {0} blocks",
            mechanic_core::MAX_GRID_UNITS
        )
    } else {
        let radius = PipeBendDimensions::new(outer_diameter, inner_diameter, span)
            .map_or(0.0, PipeBendDimensions::radius);
        format!("Bend {span} × {span} blocks — radius {radius:.3} m")
    };
    (span, message)
}

pub(crate) fn constrained_pipe_bend_span(outer_diameter: f32, requested: i16) -> u8 {
    let minimum = PipeBendDimensions::minimum_span(outer_diameter);
    u8::try_from(requested.clamp(i16::from(minimum), i16::from(mechanic_core::MAX_GRID_UNITS)))
        .expect("clamped spans fit a byte")
}
