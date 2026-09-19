//! Committing placements, deletions, block edits, and chroma strokes to the graph.

use crate::builder::{
    BlockVolume, PipeRunAttachment, PipeRunPiece, PlacementPlane, PlacementSupport,
    bearing_anchor_from_hit_with_grid, bearing_support_face, bearing_support_face_excluding,
    face_geometry_from_ref, stage_bearing_attachment_in_bounds, stage_bearing_cylinder_in_bounds,
    stage_block_volume_in_bounds, stage_controller_in_bounds, stage_dimension_link_in_bounds,
    stage_engine_in_bounds, stage_input_in_bounds, stage_pipe_run_in_bounds, stage_seat_in_bounds,
    stage_servo_in_bounds, stage_transmission, transmission_candidate_from_hit_in_bounds,
    validate_block_batch_in_bounds, validate_cylinder_candidate_in_bounds,
};
use crate::camera::{MaterialWheelState, PlayerState};
use crate::chroma::ChromaBrush;
use crate::controls::GameAction;
use crate::editor::dimensions::{BearingToolSettings, CylinderToolSettings};
use crate::editor::history::{ChromaStroke, EditorHistory, EditorSnapshot};
use crate::editor::hover::{
    BearingOffsetDrag, BlockAttachment, BlockDrag, DeleteDrag, DeleteTarget, PointerSample,
    clear_hover, refresh_tool_preview_with_cylinder,
};
use crate::editor::pipe::{
    PipeDrag, PipeEditMode, adjust_pipe_bend_span, constrained_pipe_bend_span,
};
use crate::editor::raycast::hovered_part;
use crate::editor::shape_actions::handle_layer_actions;
use crate::editor::state::{EditorGraph, EditorState};
use crate::editor::wiring::{
    disconnect_connector_links, handle_connector_actions, reverse_drive_wires,
};
use crate::hotbar::{SelectedMaterial, SelectedTool, Tool};
use crate::simulation::state::AppSimulation;
use crate::{
    builder, hotbar, linear_editor, live_edit, piston_editor, suspension_controls,
    suspension_editor, ui, weld_tool, world,
};
use bevy::prelude::{ButtonInput, IVec3, Res, ResMut, Vec3, format, vec};
use mechanic_core::{
    AppearanceTarget, BearingDimensions, BearingId, BuildCommand, BuildOutcome, ConstructionGraph,
    ConstructionMaterial, EngineKind, FaceOwner, MaterialAppearance, PartId, PartSpec,
};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlacedBearing {
    pub(crate) kind: mechanic_core::JointKind,
    pub(crate) axis: Vec3,
    pub(crate) source: mechanic_core::FaceRef,
    pub(crate) anchor: Vec3,
    pub(crate) dimensions: BearingDimensions,
}

impl PlacedBearing {
    /// Centre and radius, at the build pose, of the round plate a suspension or
    /// piston offers to attachments.
    pub(crate) fn moving_plate(self) -> Option<(Vec3, f32)> {
        match self.kind {
            mechanic_core::JointKind::Suspension(spec) => Some((
                self.anchor + self.axis * spec.initial_length(),
                spec.plates().diameter / 2.0,
            )),
            mechanic_core::JointKind::Piston(piston) => Some((
                piston.head_center(self.anchor, self.axis, 0.0),
                piston.dimensions.head_radius(),
            )),
            mechanic_core::JointKind::Rotational | mechanic_core::JointKind::Linear(_) => None,
        }
    }
}

pub(crate) fn appearance_target(
    graph: &ConstructionGraph,
    state: &EditorState,
) -> Option<AppearanceTarget> {
    let hit = state.hovered?;
    let FaceOwner::Part(part) = hit.face.owner else {
        return None;
    };
    let spec = graph.part(part)?;
    spec.appearance()?;
    if spec.is_layered() {
        // A layered part paints the band under the pointer.
        let pose = spec.pose();
        let local = pose.rotation.quaternion().inverse()
            * (graph.part_frame(part)?.inverse().point(hit.point) - pose.translation());
        return Some(AppearanceTarget::PartBand {
            part,
            band: spec.band_at_local_point(local),
        });
    }
    Some(
        graph
            .region_of(part)
            .map_or(AppearanceTarget::Part(part), AppearanceTarget::Region),
    )
}

pub(crate) fn target_appearance(
    graph: &ConstructionGraph,
    target: AppearanceTarget,
) -> Option<MaterialAppearance> {
    match target {
        AppearanceTarget::Part(part) => graph.part(part)?.appearance(),
        AppearanceTarget::Region(region) => Some(graph.region(region)?.appearance()),
        AppearanceTarget::PartBand { part, band } => graph
            .part(part)?
            .band(band)
            .map(|(_, appearance)| appearance),
    }
}

pub(crate) fn handle_chroma_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    brush: MaterialAppearance,
) {
    if suspension_editor::paint(graph, state, history, brush, actions) {
        return;
    }
    let started_remove = actions.just_pressed(GameAction::Secondary);
    if actions.just_pressed(GameAction::Primary) || started_remove {
        state.chroma_stroke = Some(ChromaStroke {
            previous: EditorSnapshot::capture(graph, state),
            targets: HashSet::new(),
            remove: started_remove,
            changed: false,
        });
    }

    if (actions.pressed(GameAction::Primary) || actions.pressed(GameAction::Secondary))
        && let Some(target) = appearance_target(graph, state)
    {
        let stroke = state
            .chroma_stroke
            .as_mut()
            .expect("a held Chroma button begins a stroke");
        if stroke.targets.insert(target) {
            let wanted = if stroke.remove {
                MaterialAppearance::BAKED
            } else {
                brush
            };
            if target_appearance(graph, target) != Some(wanted) {
                match graph.apply(BuildCommand::SetAppearance {
                    target,
                    appearance: wanted,
                }) {
                    Ok(BuildOutcome::AppearanceUpdated) => {
                        stroke.changed = true;
                        state.construction_mesh_dirty = true;
                    }
                    Ok(_) => unreachable!("appearance edits report their outcome"),
                    Err(error) => state.feedback = Some(error.to_string()),
                }
            }
        }
    }

    if (actions.just_released(GameAction::Primary) || actions.just_released(GameAction::Secondary))
        && let Some(stroke) = state.chroma_stroke.take()
        && stroke.changed
    {
        history.commit(stroke.previous);
        state.feedback = Some(if stroke.remove {
            "Restored baked appearance".to_owned()
        } else {
            "Painted construction appearance".to_owned()
        });
    }
}

#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
// Tool-specific input flows remain readable together.
pub(crate) fn handle_build_actions(
    motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    mut cylinder_settings: ResMut<CylinderToolSettings>,
    selected_material: Option<Res<SelectedMaterial>>,
    chroma_brush: Res<ChromaBrush>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    mut world_runtime: Option<ResMut<world::WorldRuntime>>,
) {
    if suspension_controls::actions(
        &mut graph,
        &mut state,
        &mut history,
        &actions,
        motion.delta,
        selection.active_editor_tool(),
        overlay.blocks_pointer() || !player.world_input_active() || wheel.open,
    ) {
        return;
    }
    if selection.active_editor_tool() == Some(Tool::Weld) {
        let blocked = overlay.blocks_pointer() || !player.world_input_active() || wheel.open;
        weld_tool::sync_mode(&mut state, selection.weld_mode);
        match selection.weld_mode {
            hotbar::WeldMode::Join => {
                weld_tool::join_actions(&mut graph.0, &mut state, &mut history, &actions, blocked);
            }
            hotbar::WeldMode::Place => weld_tool::actions(
                &mut graph.0,
                &simulation,
                &mut state,
                &mut history,
                &actions,
                blocked,
            ),
        }
        return;
    }
    let mut view = live_edit::EditorView::new(&mut graph, &mut state);
    let (graph, state) = view.parts();
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        if actions.just_released(GameAction::Primary) && state.suspension.drag.take().is_some() {
            clear_hover(state);
            state.feedback = Some("Suspension drag cancelled over interface".to_owned());
        }
        if actions.just_released(GameAction::Primary) && state.piston.drag.take().is_some() {
            clear_hover(state);
            state.feedback = Some("Piston placement cancelled over interface".to_owned());
        }
        if actions.just_released(GameAction::Primary) && state.block_drag.take().is_some() {
            clear_hover(state);
            state.feedback = Some("Block drag cancelled over hotbar".to_owned());
        }
        if actions.just_released(GameAction::Primary) && state.pipe_drag.take().is_some() {
            clear_hover(state);
            state.feedback = Some("Pipe run cancelled over hotbar".to_owned());
        }
        if actions.just_released(GameAction::Primary) && state.layer_drag.take().is_some() {
            clear_hover(state);
            state.feedback = Some("Layer cancelled over hotbar".to_owned());
        }
        if actions.just_released(GameAction::Secondary) && state.cancel_delete_gesture() {
            state.feedback = Some("Delete drag cancelled over hotbar".to_owned());
        }
        return;
    }
    let empty_hand_link_delete = selection.tool.is_none()
        && ((actions.just_pressed(GameAction::Secondary)
            && hovered_part(state.hovered)
                .or_else(|| state.hovered_simulation.map(|hit| hit.part))
                .is_some_and(|part| graph.0.dimension_link_id(part).is_some()))
            || (actions.just_released(GameAction::Secondary)
                && matches!(state.delete_target, Some(DeleteTarget::Part(part))
                    if graph.0.dimension_link_id(part).is_some())));
    let Some(tool) = selection
        .active_editor_tool()
        .or_else(|| empty_hand_link_delete.then_some(Tool::DimensionLink))
    else {
        return;
    };
    if tool == Tool::Shape {
        return;
    }
    if tool == Tool::Chroma {
        handle_chroma_actions(
            &actions,
            &mut graph.0,
            state,
            &mut history,
            chroma_brush.appearance,
        );
        return;
    }
    if actions.just_pressed(GameAction::Secondary) && state.suspension.drag.take().is_some() {
        state.suspension.controls.consume_until_release = true;
        clear_hover(state);
        state.feedback = Some("Suspension drag cancelled".to_owned());
        return;
    }
    if actions.just_pressed(GameAction::Secondary) && state.block_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Block drag cancelled".to_owned());
        return;
    }
    if actions.just_pressed(GameAction::Secondary) && state.pipe_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Pipe run cancelled".to_owned());
        return;
    }
    if actions.just_pressed(GameAction::Secondary) && state.layer_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Layer cancelled".to_owned());
        return;
    }
    if tool == Tool::Connector
        && actions.just_pressed(GameAction::Secondary)
        && state.wire_drag.take().is_some()
    {
        state.feedback = Some("Drive wire cancelled".to_owned());
        return;
    }
    if actions.just_pressed(GameAction::Secondary)
        && let Some(socket) = state
            .hovered_bearing
            .and_then(|index| state.placed_bearings.get(index).copied())
        && let Some(feedback) = reverse_drive_wires(&mut graph.0, state, &mut history, socket)
    {
        state.feedback = Some(feedback);
        return;
    }
    if tool == Tool::Connector && actions.just_pressed(GameAction::Secondary) {
        state.feedback = Some(disconnect_connector_links(
            &mut graph.0,
            state,
            &mut history,
        ));
        return;
    }
    if actions.just_pressed(GameAction::Secondary) {
        if let Some(index) = state.hovered_bearing {
            state.delete_target = Some(DeleteTarget::PlacedBearing(index));
            state.feedback = Some("Release right mouse to delete bearing".to_owned());
        } else if let Some(hit) = state.hovered
            && let FaceOwner::Part(part) = hit.face.owner
            && let Some(spec) = graph.0.part(part).copied()
        {
            match spec {
                PartSpec::Cuboid(spec) => {
                    let plane = PlacementPlane::from_normal(
                        face_geometry_from_ref(hit.face, Some(&graph.0)).normal,
                    );
                    let Some((cursor, (ray_origin, ray_direction))) =
                        state.pointer_position.zip(state.pointer_ray)
                    else {
                        state.feedback = Some("Pointer position is unavailable".to_owned());
                        return;
                    };
                    state.delete_drag = Some(DeleteDrag {
                        start: spec,
                        press: PointerSample {
                            cursor,
                            ray_origin,
                            ray_direction,
                        },
                        plane,
                        anchor_span: IVec3::ZERO,
                        span: IVec3::ZERO,
                        last_span: None,
                        parts: vec![part],
                        error: None,
                    });
                    state.delete_preview_revision = state.delete_preview_revision.wrapping_add(1);
                    state.feedback = Some(format!(
                        "Dragging delete on {} plane — release to remove, Rotate changes plane",
                        plane.label()
                    ));
                }
                PartSpec::Cylinder(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete cylinder".to_owned());
                }
                PartSpec::PipeBend(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete pipe bend".to_owned());
                }
                PartSpec::PipeJunction(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete pipe junction".to_owned());
                }
                PartSpec::Controller(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete control block".to_owned());
                }
                PartSpec::Engine(engine) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some(format!(
                        "Release right mouse to delete {} engine",
                        match engine.kind {
                            EngineKind::Gas => "gas",
                            EngineKind::Electric => "electric",
                        }
                    ));
                }
                PartSpec::Transmission(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some(
                        "Release right mouse to delete transmission and downstream blocks"
                            .to_owned(),
                    );
                }
                PartSpec::Servo(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete servo".to_owned());
                }
                PartSpec::Seat(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete seat".to_owned());
                }
                PartSpec::Input(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete Input".to_owned());
                }
                PartSpec::DimensionLink(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback =
                        Some("Release right mouse to delete Dimension Link".to_owned());
                }
            }
        } else if let Some(part) = state
            .hovered_simulation
            .map(|hit| hit.part)
            .filter(|part| graph.0.dimension_link_id(*part).is_some())
        {
            state.delete_target = Some(DeleteTarget::Part(part));
            state.feedback = Some("Release right mouse to delete Dimension Link".to_owned());
        }
    }
    if actions.just_released(GameAction::Secondary) {
        if let Some(target) = state.delete_target.take() {
            match target {
                DeleteTarget::PlacedBearing(index) => {
                    if suspension_editor::remove_component(&mut graph.0, state, &mut history, index)
                    {
                        return;
                    }
                    if let Some(socket) = state.placed_bearings.get(index).copied() {
                        let previous = EditorSnapshot::capture(&graph.0, state);
                        let attached = graph
                            .0
                            .bearings()
                            .filter_map(|(id, bearing)| {
                                bearing_uses_socket(bearing, socket).then_some(id)
                            })
                            .collect::<Vec<_>>();
                        let targets = bearing_socket_targets(&graph.0, socket)
                            .into_iter()
                            .collect::<HashSet<_>>();
                        let rigid_links = graph
                            .0
                            .rigid_links()
                            .filter_map(|(id, link)| {
                                (targets.contains(&link.first) && targets.contains(&link.second))
                                    .then_some(id)
                            })
                            .collect::<Vec<_>>();
                        let mut staged = graph.0.begin_edit();
                        let commands = rigid_links
                            .iter()
                            .copied()
                            .map(BuildCommand::RemoveRigidLink)
                            .chain(attached.iter().copied().map(BuildCommand::RemoveBearing));
                        match staged.apply_batch(commands) {
                            Ok(_) => {
                                graph.0 = staged.finish();
                                state.placed_bearings.remove(index);
                                history.commit(previous);
                                state.feedback = Some(format!(
                                    "Deleted bearing and {} attachment(s)",
                                    attached.len()
                                ));
                                state.construction_mesh_dirty = true;
                                clear_hover(state);
                            }
                            Err(error) => state.feedback = Some(error.to_string()),
                        }
                    }
                }
                DeleteTarget::Part(part) => {
                    let deleted_link = graph.0.dimension_link_id(part);
                    let previous = EditorSnapshot::capture(&graph.0, state);
                    match stage_part_deletion_preserving_bearings(
                        &graph.0,
                        &state.placed_bearings,
                        &[part],
                    ) {
                        Ok((staged, placed_bearings, migrated)) => {
                            graph.0 = staged;
                            state.placed_bearings = placed_bearings;
                            history.commit(previous);
                            state.feedback = Some(if deleted_link.is_some() {
                                "Deleted Dimension Link and incident connections".to_owned()
                            } else if migrated == 0 {
                                "Deleted cylinder and incident connections".to_owned()
                            } else {
                                format!("Deleted cylinder; moved {migrated} bearing(s)")
                            });
                            state.construction_mesh_dirty = true;
                            if let (Some(id), Some(world_runtime)) =
                                (deleted_link, world_runtime.as_deref_mut())
                            {
                                world_runtime.clear_active_dimension_link_if(id);
                            }
                            clear_hover(state);
                        }
                        Err(error) => state.feedback = Some(error.to_string()),
                    }
                }
            }
        }
        if let Some(drag) = state.delete_drag.take() {
            if let Some(error) = drag.error {
                state.feedback = Some(error.to_string());
                return;
            }
            let previous = EditorSnapshot::capture(&graph.0, state);
            match stage_part_deletion_preserving_bearings(
                &graph.0,
                &state.placed_bearings,
                &drag.parts,
            ) {
                Ok((staged, placed_bearings, migrated)) => {
                    graph.0 = staged;
                    state.placed_bearings = placed_bearings;
                    history.commit(previous);
                    state.feedback = Some(if migrated == 0 {
                        format!(
                            "Deleted {} cuboid(s) and incident connections",
                            drag.parts.len()
                        )
                    } else {
                        format!(
                            "Deleted {} cuboid(s); moved {migrated} bearing(s) to remaining support",
                            drag.parts.len()
                        )
                    });
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        return;
    }
    if empty_hand_link_delete {
        return;
    }
    if tool == Tool::Connector {
        handle_connector_actions(&actions, &mut graph.0, state, &mut history);
        return;
    }
    if matches!(tool, Tool::Block | Tool::Cylinder)
        && suspension_editor::attach(&mut graph.0, state, &mut history, tool, &actions)
    {
        return;
    }
    if matches!(tool, Tool::Spring | Tool::Shock) {
        suspension_editor::drag_actions(&mut graph.0, state, &mut history, &actions);
        return;
    }
    if tool == Tool::Piston {
        piston_editor::drag_actions(&mut graph.0, state, &mut history, &actions);
        return;
    }
    if tool == Tool::Cylinder
        && (state.suspension.insertion.is_some() || state.suspension.drag.is_some())
    {
        suspension_editor::drag_actions(&mut graph.0, state, &mut history, &actions);
        return;
    }
    if tool == Tool::Block {
        handle_block_actions(&actions, &mut graph.0, state, &mut history);
        return;
    }
    if tool == Tool::Layer {
        handle_layer_actions(&actions, &mut graph.0, state, &mut history);
        return;
    }
    if tool == Tool::Cylinder {
        if state.pipe_bend_active()
            && (actions.just_pressed(GameAction::ZoomIn)
                || actions.just_pressed(GameAction::ZoomOut))
        {
            let direction = i8::from(actions.just_pressed(GameAction::ZoomIn))
                - i8::from(actions.just_pressed(GameAction::ZoomOut));
            let (span, message) = adjust_pipe_bend_span(&graph.0, state, direction);
            cylinder_settings.bend_span = span;
            state.feedback = Some(message);
            return;
        }
        if actions.just_pressed(GameAction::Primary) {
            let Some(candidate) = state.cylinder_preview else {
                state.feedback = Some("Point at a flat face or compatible bearing".to_owned());
                return;
            };
            let attachment = if let Some(socket) = state.linear_attachment {
                if let Some(error) = state.preview_error.as_ref() {
                    state.feedback = Some(error.to_string());
                    return;
                }
                BlockAttachment::Linear { socket }
            } else if let Some(index) = state.attachment_bearing {
                let Some(bearing) = state.placed_bearings.get(index).copied() else {
                    state.feedback = Some("Bearing is no longer available".to_owned());
                    return;
                };
                if let Err(error) = stage_bearing_cylinder_in_bounds(
                    &graph.0,
                    candidate,
                    bearing.source,
                    bearing.anchor,
                    bearing.dimensions,
                    &bearing_socket_targets(&graph.0, bearing),
                    state.placement_bounds,
                ) {
                    state.feedback = Some(error.to_string());
                    return;
                }
                BlockAttachment::Bearing {
                    source: bearing.source,
                    anchor: bearing.anchor,
                    dimensions: bearing.dimensions,
                }
            } else {
                if let Err(error) = validate_cylinder_candidate_in_bounds(
                    &graph.0,
                    candidate,
                    state.placement_bounds,
                ) {
                    state.feedback = Some(error.to_string());
                    return;
                }
                match candidate.support {
                    PlacementSupport::Surface(source) => BlockAttachment::AutoWeld { source },
                    PlacementSupport::Free => BlockAttachment::Free,
                    PlacementSupport::Bearing => {
                        state.feedback = Some("Bearing is no longer available".to_owned());
                        return;
                    }
                }
            };
            let Some((ray_origin, ray_direction)) = state.pointer_ray else {
                state.feedback = Some("Pointer ray is unavailable".to_owned());
                return;
            };
            let Some(cursor) = state.pointer_position else {
                state.feedback = Some("Pointer position is unavailable".to_owned());
                return;
            };
            let direction = candidate.spec.pose.rotation.quaternion() * Vec3::Y;
            let start = candidate.spec.pose.translation()
                - direction * candidate.spec.dimensions.axial_length() * 0.5;
            let endpoint = start + direction * candidate.spec.dimensions.axial_length();
            let bearing_offset = matches!(attachment, BlockAttachment::Bearing { .. }).then_some(
                BearingOffsetDrag {
                    start,
                    endpoint,
                    normal: direction,
                },
            );
            let pieces = vec![PipeRunPiece {
                spec: PartSpec::Cylinder(candidate.spec),
                inlet: mechanic_core::FaceKind::NegativeY,
                outlet: mechanic_core::FaceKind::PositiveY,
            }];
            state.pipe_drag = Some(PipeDrag {
                attachment,
                start,
                corners: Vec::new(),
                endpoint,
                directions: vec![direction],
                nodes: Vec::new(),
                branch: state.pipe_branch_preview.filter(|branch| {
                    candidate.support
                        == PlacementSupport::Surface(FaceOwner::Part(branch.site.part()))
                }),
                pending_span: constrained_pipe_bend_span(
                    candidate.spec.dimensions.outer_diameter(),
                    i16::from(cylinder_settings.bend_span),
                ),
                dimensions: candidate.spec.dimensions,
                material: candidate.spec.material,
                appearance: candidate.spec.appearance,
                mode: PipeEditMode::Length,
                bearing_offset,
                choosing_direction: false,
                press: PointerSample {
                    cursor,
                    ray_origin,
                    ray_direction,
                },
                anchor_endpoint: endpoint,
                anchor_dimensions: candidate.spec.dimensions,
                pieces,
                error: None,
            });
            state.feedback = Some(if bearing_offset.is_some() {
                "Bearing and pipe centred — drag to offset, release to commit; R edits length"
                    .to_owned()
            } else {
                "Dragging pipe length — R cycles dimensions, F adds a 90° bend, release commits"
                    .to_owned()
            });
            return;
        }
        if actions.just_released(GameAction::Primary) {
            let Some(drag) = state.pipe_drag.take() else {
                return;
            };
            cylinder_settings.dimensions = drag.dimensions;
            cylinder_settings.bend_span = drag.pending_span;
            if drag.choosing_direction {
                state.feedback = Some(
                    "Pipe run not placed: choose a turn direction before releasing".to_owned(),
                );
                clear_hover(state);
                return;
            }
            if let Some(error) = drag.error {
                state.feedback = Some(error.to_string());
                clear_hover(state);
                return;
            }
            let previous = EditorSnapshot::capture(&graph.0, state);
            let staged = match drag.attachment {
                BlockAttachment::Linear { socket } => {
                    let targets = bearing_socket_targets(&graph.0, socket);
                    stage_pipe_run_in_bounds(
                        &graph.0,
                        &drag.pieces,
                        PipeRunAttachment::Linear(linear_editor::attachment(socket, &targets)),
                        state.placement_bounds,
                    )
                }

                BlockAttachment::AutoWeld { source } => match drag.branch {
                    Some(branch) => crate::builder::apply_pipe_branch(&graph.0, branch).and_then(
                        |(split, junction)| {
                            stage_pipe_run_in_bounds(
                                &split,
                                &drag.pieces,
                                PipeRunAttachment::AutoWeld {
                                    source: FaceOwner::Part(junction),
                                },
                                state.placement_bounds,
                            )
                        },
                    ),
                    None => stage_pipe_run_in_bounds(
                        &graph.0,
                        &drag.pieces,
                        PipeRunAttachment::AutoWeld { source },
                        state.placement_bounds,
                    ),
                },
                BlockAttachment::Free => stage_pipe_run_in_bounds(
                    &graph.0,
                    &drag.pieces,
                    PipeRunAttachment::Free,
                    state.placement_bounds,
                ),
                BlockAttachment::Bearing {
                    source,
                    anchor,
                    dimensions,
                } => {
                    let socket = PlacedBearing {
                        kind: mechanic_core::JointKind::Rotational,
                        axis: Vec3::ZERO,
                        source,
                        anchor,
                        dimensions,
                    };
                    let rigid_targets = bearing_socket_targets(&graph.0, socket);
                    stage_pipe_run_in_bounds(
                        &graph.0,
                        &drag.pieces,
                        PipeRunAttachment::Bearing {
                            source,
                            anchor,
                            dimensions,
                            rigid_targets: &rigid_targets,
                        },
                        state.placement_bounds,
                    )
                }
            };
            match staged {
                Ok(staged) => {
                    let count = drag.pieces.len();
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!(
                        "Placed pipe run with {count} piece(s) and {} bend(s){}",
                        drag.nodes.len(),
                        if drag.branch.is_some() {
                            ", branching through a junction"
                        } else {
                            ""
                        }
                    ));
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        return;
    }
    if actions.pressed(GameAction::Secondary) || !actions.just_pressed(GameAction::Primary) {
        return;
    }

    match tool {
        Tool::Shape => unreachable!("shape actions are handled by handle_shape_actions"),
        Tool::Block => unreachable!("block actions are handled before this match"),
        Tool::Cylinder => unreachable!("cylinder actions are handled before this match"),
        Tool::Layer => unreachable!("layer actions are handled before this match"),
        Tool::Weld => unreachable!("weld actions are handled by weld_tool"),
        Tool::LinearBearing => linear_editor::place(&graph.0, state, &mut history),
        Tool::Piston => unreachable!("piston actions are handled before this match"),
        Tool::Spring | Tool::Shock => suspension_editor::place(&mut graph.0, state, &mut history),
        Tool::Bearing => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Point at a cuboid face".to_owned());
                return;
            };
            let anchor = state.bearing_preview_anchor.or_else(|| {
                bearing_anchor_from_hit_with_grid(
                    &graph.0,
                    hit,
                    state.placement_grid,
                    state.placement_bounds,
                )
                .ok()
            });
            match anchor {
                Some(anchor) => {
                    let Some(source) = bearing_support_face(
                        &graph.0,
                        hit.face,
                        anchor,
                        bearing_settings.dimensions,
                    ) else {
                        state.feedback = Some(
                            "The bearing ring must overlap at least one supporting block"
                                .to_owned(),
                        );
                        return;
                    };
                    let duplicate = bearing_location_occupied(
                        &graph.0,
                        &state.placed_bearings,
                        hit.face,
                        anchor,
                    );
                    if duplicate {
                        state.feedback = Some("A bearing is already placed here".to_owned());
                    } else {
                        let previous = EditorSnapshot::capture(&graph.0, state);
                        state.placed_bearings.push(PlacedBearing {
                            kind: mechanic_core::JointKind::Rotational,
                            axis: Vec3::ZERO,
                            source,
                            anchor,
                            dimensions: bearing_settings.dimensions,
                        });
                        history.commit(previous);
                        state.feedback = Some(
                            "Bearing placed — select Blocker Placer and hover it to attach"
                                .to_owned(),
                        );
                        state.construction_mesh_dirty = true;
                    }
                }
                None => state.feedback = Some("Bearing anchor is invalid".to_owned()),
            }
        }
        Tool::Hammer => {
            state.feedback = Some("Hammer is available in the live World".to_owned());
        }
        Tool::Controller => {
            if let Some(hit) = state.hovered
                && let FaceOwner::Part(part) = hit.face.owner
                && graph.0.is_controller(part)
            {
                state.selected_controller = Some(part);
                let wires = graph.0.controller_links(part).count();
                state.feedback = Some(format!(
                    "Selected control block — {wires} wired, press E to program it"
                ));
                return;
            }
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at a face or into free Garage space".to_owned());
                return;
            };
            let previous = EditorSnapshot::capture(&graph.0, state);
            let existing = graph.0.parts().map(|(part, _)| part).collect::<Vec<_>>();
            match stage_controller_in_bounds(&graph.0, candidate, state.placement_bounds) {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.selected_controller = graph
                        .0
                        .parts()
                        .find(|(part, spec)| {
                            matches!(spec, PartSpec::Controller(_)) && !existing.contains(part)
                        })
                        .map(|(part, _)| part);
                    state.feedback = Some(
                        "Placed control block — with the Connector, drag it to a bearing, then press E"
                            .to_owned(),
                    );
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        tool @ (Tool::GasEngine | Tool::ElectricEngine) => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at a face or into free Garage space".to_owned());
                return;
            };
            let kind = if tool == Tool::GasEngine {
                EngineKind::Gas
            } else {
                EngineKind::Electric
            };
            let previous = EditorSnapshot::capture(&graph.0, state);
            match stage_engine_in_bounds(&graph.0, candidate, kind, state.placement_bounds) {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!("Placed {}", tool.label().to_lowercase()));
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Transmission => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Point at an engine or transmission +Z output".to_owned());
                return;
            };
            let (parent, candidate) = match transmission_candidate_from_hit_in_bounds(
                &graph.0,
                hit,
                state.placement_bounds,
            ) {
                Ok(candidate) => candidate,
                Err(error) => {
                    state.feedback = Some(error.to_string());
                    return;
                }
            };
            let kind = match graph.0.part(parent) {
                Some(PartSpec::Engine(engine)) => Some(engine.kind),
                Some(PartSpec::Transmission(_)) => graph.0.transmission_kind(parent),
                _ => None,
            };
            let previous = EditorSnapshot::capture(&graph.0, state);
            match stage_transmission(&graph.0, parent, candidate) {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!(
                        "Placed {} transmission",
                        match kind {
                            Some(EngineKind::Gas) => "gas",
                            Some(EngineKind::Electric) => "electric",
                            None => "",
                        }
                    ));
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Servo | Tool::Seat | Tool::Input => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at a face or into free Garage space".to_owned());
                return;
            };
            let previous = EditorSnapshot::capture(&graph.0, state);
            let staged = match tool {
                Tool::Servo => stage_servo_in_bounds(&graph.0, candidate, state.placement_bounds),
                Tool::Seat => stage_seat_in_bounds(&graph.0, candidate, state.placement_bounds),
                Tool::Input => stage_input_in_bounds(&graph.0, candidate, state.placement_bounds),
                _ => unreachable!(),
            };
            match staged {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!("Placed {}", tool.label()));
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::DimensionLink => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at a face or into free Garage space".to_owned());
                return;
            };
            let Some(world_runtime) = world_runtime.as_deref_mut() else {
                state.feedback = Some("World state is unavailable".to_owned());
                return;
            };
            let id = world_runtime.allocate_dimension_link_id();
            let previous = EditorSnapshot::capture(&graph.0, state);
            match stage_dimension_link_in_bounds(&graph.0, candidate, id, state.placement_bounds) {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!("Placed Dimension Link {}", id.0));
                    state.construction_mesh_dirty = true;
                    clear_hover(state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Connector => unreachable!("connector actions are handled before this match"),
        Tool::Chroma => unreachable!("Chroma actions are handled before this match"),
    }
    refresh_tool_preview_with_cylinder(
        &graph.0,
        state,
        tool,
        cylinder_settings.dimensions,
        bearing_settings.dimensions,
        selected_material
            .as_deref()
            .map_or(ConstructionMaterial::Steel, |value| value.0),
        chroma_brush.appearance,
    );
}

pub(crate) fn simulation_part_is_static(simulation: &AppSimulation, part: PartId) -> bool {
    let Some(creation) = simulation.creation.as_ref() else {
        return false;
    };
    creation
        .part_to_compound
        .iter()
        .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
        .is_some_and(|compound| creation.compounds[compound as usize].is_static)
}

pub(crate) fn editor_part_is_static_or_pending(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    part: PartId,
) -> bool {
    graph.part(part).is_some()
        && (simulation.published_graph.part(part).is_none()
            || simulation_part_is_static(simulation, part))
}

pub(crate) fn bearing_location_occupied(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    face: mechanic_core::FaceRef,
    anchor: Vec3,
) -> bool {
    let same_surface = |candidate: mechanic_core::FaceRef| {
        let selected = face_geometry_from_ref(face, Some(graph));
        let candidate = face_geometry_from_ref(candidate, Some(graph));
        selected.normal.dot(candidate.normal) > 1.0 - 1.0e-5
            && (selected.center - candidate.center)
                .dot(selected.normal)
                .abs()
                <= 1.0e-5
    };
    placed_bearings
        .iter()
        .any(|bearing| same_surface(bearing.source) && bearing.anchor.abs_diff_eq(anchor, 1.0e-5))
        || graph.bearings().any(|(_, bearing)| {
            (same_surface(bearing.source) || bearing.target.is_some_and(same_surface))
                && bearing.shared_anchor.abs_diff_eq(anchor, 1.0e-5)
        })
}

/// Graph bearings backing one placed socket. A socket can carry several rows
/// when more than one rotor group is attached through the same ring.
pub(crate) fn socket_bearings(graph: &ConstructionGraph, socket: PlacedBearing) -> Vec<BearingId> {
    graph
        .bearings()
        .filter_map(|(id, bearing)| bearing_uses_socket(bearing, socket).then_some(id))
        .collect()
}

pub(crate) fn bearing_uses_socket(
    bearing: &mechanic_core::BearingSpec,
    socket: PlacedBearing,
) -> bool {
    bearing.source == socket.source
        && bearing.shared_anchor.abs_diff_eq(socket.anchor, 1.0e-5)
        && bearing.dimensions == socket.dimensions
        && match (bearing.kind, socket.kind) {
            (mechanic_core::JointKind::Rotational, mechanic_core::JointKind::Rotational) => true,
            (mechanic_core::JointKind::Suspension(a), mechanic_core::JointKind::Suspension(b)) => {
                a == b
            }
            (mechanic_core::JointKind::Linear(a), mechanic_core::JointKind::Linear(b)) => {
                a.dimensions == b.dimensions
                    && a.mount_normal.abs_diff_eq(b.mount_normal, 1.0e-5)
                    && bearing.axis.abs_diff_eq(socket.axis, 1.0e-5)
            }
            (mechanic_core::JointKind::Piston(a), mechanic_core::JointKind::Piston(b)) => {
                a == b && bearing.axis.abs_diff_eq(socket.axis, 1.0e-5)
            }
            _ => false,
        }
}

pub(crate) fn bearing_socket_targets(
    graph: &ConstructionGraph,
    socket: PlacedBearing,
) -> Vec<PartId> {
    graph
        .bearings()
        .filter_map(|(_, bearing)| {
            if !bearing_uses_socket(bearing, socket) {
                return None;
            }
            match bearing.target?.owner {
                FaceOwner::Part(part) => Some(part),
                FaceOwner::Ground => None,
            }
        })
        .collect()
}

pub(crate) fn stage_part_deletion_preserving_bearings(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    deleted_parts: &[PartId],
) -> Result<(ConstructionGraph, Vec<PlacedBearing>, usize), mechanic_core::GraphError> {
    let deleted = deleted_parts.iter().copied().collect::<HashSet<_>>();
    let mut next_bearings = Vec::with_capacity(placed_bearings.len());
    let mut migrations = Vec::<(
        PlacedBearing,
        Vec<(mechanic_core::FaceRef, mechanic_core::JointKind)>,
    )>::new();
    let mut unsupported_target_sets = Vec::<HashSet<PartId>>::new();

    for &socket in placed_bearings {
        let FaceOwner::Part(source_part) = socket.source.owner else {
            continue;
        };
        if !deleted.contains(&source_part) {
            next_bearings.push(socket);
            continue;
        }

        let targets = graph
            .bearings()
            .filter_map(|(_, bearing)| {
                if !bearing_uses_socket(bearing, socket) {
                    return None;
                }
                let target = bearing.target?;
                match target.owner {
                    FaceOwner::Part(part) if !deleted.contains(&part) => {
                        Some((target, bearing.kind))
                    }
                    FaceOwner::Part(_) | FaceOwner::Ground => None,
                }
            })
            .collect::<Vec<_>>();
        let replacement = match socket.kind {
            mechanic_core::JointKind::Rotational | mechanic_core::JointKind::Suspension(_) => {
                bearing_support_face_excluding(
                    graph,
                    socket.source,
                    socket.anchor,
                    socket.dimensions,
                    &deleted,
                )
            }
            mechanic_core::JointKind::Linear(rail) => builder::linear_support_face_excluding(
                graph,
                socket.source,
                socket.anchor,
                rail,
                socket.axis,
                &deleted,
            ),
            // A piston goes with its support; it has no other face to move to.
            mechanic_core::JointKind::Piston(_) => None,
        };
        if let Some(source) = replacement {
            let migrated = PlacedBearing { source, ..socket };
            next_bearings.push(migrated);
            migrations.push((migrated, targets));
        } else {
            unsupported_target_sets
                .push(bearing_socket_targets(graph, socket).into_iter().collect());
        }
    }

    let rigid_links = graph
        .rigid_links()
        .filter_map(|(id, link)| {
            unsupported_target_sets
                .iter()
                .any(|targets| targets.contains(&link.first) && targets.contains(&link.second))
                .then_some(id)
        })
        .collect::<Vec<_>>();
    let mut staged = graph.begin_edit();
    staged.apply_batch(
        rigid_links
            .into_iter()
            .map(BuildCommand::RemoveRigidLink)
            .chain(deleted_parts.iter().copied().map(BuildCommand::Remove)),
    )?;

    let migrated_count = migrations.len();
    let replacement_bearings = migrations
        .into_iter()
        .flat_map(|(socket, targets)| {
            let axis = match socket.kind {
                mechanic_core::JointKind::Rotational => {
                    face_geometry_from_ref(socket.source, Some(&staged)).normal
                }
                mechanic_core::JointKind::Linear(_)
                | mechanic_core::JointKind::Suspension(_)
                | mechanic_core::JointKind::Piston(_) => socket.axis,
            };
            targets.into_iter().map(move |(target, kind)| {
                BuildCommand::AddBearing(
                    mechanic_core::BearingSpec::new(socket.source, target, socket.anchor, axis)
                        .with_dimensions(socket.dimensions)
                        .with_kind(kind),
                )
            })
        })
        .collect::<Vec<_>>();
    staged.apply_batch(replacement_bearings)?;

    Ok((staged.finish(), next_bearings, migrated_count))
}

pub(crate) fn visible_bearing_count(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
) -> usize {
    placed_bearings.len()
        + graph
            .bearings()
            .filter(|(_, bearing)| {
                !placed_bearings
                    .iter()
                    .any(|&socket| bearing_uses_socket(bearing, socket))
            })
            .count()
}

#[expect(
    clippy::too_many_lines,
    reason = "click, drag, and bearing attachment share one transaction"
)]
pub(crate) fn handle_block_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    if actions.just_pressed(GameAction::Primary) {
        let Some(candidate) = state.preview else {
            state.feedback = Some("Point at the platform or a cuboid face".to_owned());
            return;
        };
        let (attachment, normal) = if let Some(socket) = state.linear_attachment {
            if let Some(error) = state.preview_error.as_ref() {
                state.feedback = Some(error.to_string());
                return;
            }
            let mechanic_core::JointKind::Linear(rail) = socket.kind else {
                unreachable!()
            };
            (
                BlockAttachment::Linear { socket },
                rail.rotation(socket.axis).expect("validated frame") * rail.face.normal(),
            )
        } else if let Some(index) = state.attachment_bearing {
            let Some(bearing) = state.placed_bearings.get(index).copied() else {
                state.feedback = Some("Bearing is no longer available".to_owned());
                return;
            };
            if let Some(error) = stage_bearing_attachment_in_bounds(
                graph,
                candidate,
                bearing.source,
                bearing.anchor,
                bearing.dimensions,
                state.placement_bounds,
            )
            .err()
            {
                state.feedback = Some(error.to_string());
                return;
            }
            (
                BlockAttachment::Bearing {
                    source: bearing.source,
                    anchor: bearing.anchor,
                    dimensions: bearing.dimensions,
                },
                face_geometry_from_ref(bearing.source, Some(graph)).normal,
            )
        } else {
            if let Some(error) = validate_block_batch_in_bounds(
                graph,
                candidate,
                &[candidate.spec],
                state.placement_bounds,
            )
            .err()
            {
                state.feedback = Some(error.to_string());
                return;
            }
            match candidate.support {
                PlacementSupport::Surface(source) => {
                    let hit = state
                        .hovered
                        .expect("surface preview originates from a hit");
                    (
                        BlockAttachment::AutoWeld { source },
                        face_geometry_from_ref(hit.face, Some(graph)).normal,
                    )
                }
                PlacementSupport::Free => {
                    let (_, direction) = state
                        .pointer_ray
                        .expect("free preview originates from a pointer ray");
                    (BlockAttachment::Free, -direction)
                }
                PlacementSupport::Bearing => {
                    state.feedback = Some("Bearing is no longer available".to_owned());
                    return;
                }
            }
        };
        let plane = PlacementPlane::from_normal(normal);
        let Some((ray_origin, ray_direction)) = state.pointer_ray else {
            state.feedback = Some("Pointer ray is unavailable".to_owned());
            return;
        };
        let Some(cursor) = state.pointer_position else {
            state.feedback = Some("Pointer position is unavailable".to_owned());
            return;
        };
        state.block_drag = Some(BlockDrag {
            start: candidate,
            attachment,
            start_guides: state.smart_guides.clone(),
            press: PointerSample {
                cursor,
                ray_origin,
                ray_direction,
            },
            plane,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            volume: BlockVolume::new(candidate.spec, IVec3::ZERO)
                .expect("one block is a valid volume"),
            error: None,
        });
        state.block_preview_revision = state.block_preview_revision.wrapping_add(1);
        state.feedback = Some(if matches!(attachment, BlockAttachment::Bearing { .. }) {
            format!(
                "Attaching green blocks through bearing on {} plane — release to place",
                plane.label()
            )
        } else {
            format!(
                "Dragging blocks on {} plane — release to place, Rotate changes plane",
                plane.label()
            )
        });
        return;
    }

    if !actions.just_released(GameAction::Primary) {
        return;
    }
    let Some(drag) = state.block_drag.take() else {
        return;
    };
    if let Some(error) = drag.error {
        state.feedback = Some(error.to_string());
        return;
    }
    let count = drag.volume.count();
    let previous = EditorSnapshot::capture(graph, state);
    let publication_generation = state.construction_publication_generation.wrapping_add(1);
    let staged = match drag.attachment {
        BlockAttachment::Linear { socket } => {
            let targets = bearing_socket_targets(graph, socket);
            builder::stage_linear_block_volume_in_bounds(
                graph,
                &state.snap_index,
                drag.start,
                drag.volume,
                linear_editor::attachment(socket, &targets),
                state.placement_bounds,
                publication_generation,
            )
        }

        BlockAttachment::AutoWeld { source } => stage_block_volume_in_bounds(
            graph,
            &state.snap_index,
            drag.start,
            drag.volume,
            None,
            Some(source),
            state.placement_bounds,
            publication_generation,
        ),
        BlockAttachment::Free => stage_block_volume_in_bounds(
            graph,
            &state.snap_index,
            drag.start,
            drag.volume,
            None,
            None,
            state.placement_bounds,
            publication_generation,
        ),
        BlockAttachment::Bearing {
            source,
            anchor,
            dimensions,
            ..
        } => {
            let socket = PlacedBearing {
                kind: mechanic_core::JointKind::Rotational,
                axis: Vec3::ZERO,
                source,
                anchor,
                dimensions,
            };
            let rigid_targets = bearing_socket_targets(graph, socket);
            stage_block_volume_in_bounds(
                graph,
                &state.snap_index,
                drag.start,
                drag.volume,
                Some((source, anchor, dimensions, &rigid_targets)),
                None,
                state.placement_bounds,
                publication_generation,
            )
        }
    };
    match staged {
        Ok(staged) => {
            let weld_count = staged.weld_count;
            debug_assert_eq!(staged.new_parts.len(), count);
            debug_assert_eq!(staged.bounds, drag.volume.bounds());
            state.construction_publication_generation = staged.publication_generation;
            *graph = staged.graph;
            history.commit(previous);
            state.feedback = Some(format!(
                "Placed {count} block(s); added {weld_count} weld(s){}",
                if matches!(drag.attachment, BlockAttachment::Bearing { .. }) {
                    " through bearing; socket remains available"
                } else {
                    ""
                }
            ));
            state.construction_mesh_dirty = true;
            clear_hover(state);
        }
        Err(error) => state.feedback = Some(error.to_string()),
    }
}
