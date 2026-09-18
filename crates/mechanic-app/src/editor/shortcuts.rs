//! Tool shortcuts, the pipette, and orientation cycling.

use crate::builder::{SurfaceHit, raycast_construction};
use crate::camera::{MainCamera, MaterialWheelState, PlayerState};
use crate::chroma::ChromaBrush;
use crate::control_panel::ControlPanelState;
use crate::controls::GameAction;
use crate::creation_menu::CreationMenuState;
use crate::editor::build_actions::{PlacedBearing, appearance_target, target_appearance};
use crate::editor::dimensions::{BearingToolSettings, CylinderToolSettings};
use crate::editor::hammer::HammerInteraction;
use crate::editor::history::cancel_transient_editor_state;
use crate::editor::hover::PointerSample;
use crate::editor::pipe::begin_pipe_node;
use crate::editor::raycast::{
    hovered_part, raycast_placed_bearings, raycast_simulation, raycast_simulation_bearings,
};
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedMaterial, SelectedTool, Tool};
use crate::pause_menu::PauseMenuState;
use crate::render::authored::{AUTHORED_ORIENTATION_COUNT, AUTHORED_ORIENTATIONS};
use crate::render::mesh::drive::axis_tangents;
use crate::simulation::state::AppSimulation;
use crate::{camera, linear_editor, live_edit, suspension_editor, suspension_render, ui};
use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, Res, ResMut, Single, Vec2, Vec3, Window, With, format,
};
use bevy::window::PrimaryWindow;
use mechanic_core::{
    BearingDimensions, ConstructionGraph, ConstructionMaterial, CylinderDimensions, EngineKind,
    FaceOwner, PartId, PartSpec,
};

/// Opens the aimed-at control-block panel with `E`.
///
/// Editor and live simulation hits both work. Remembered wiring selection must
/// not take this shared interaction key away from seat entry or exit.
#[expect(clippy::too_many_arguments)]
pub(crate) fn handle_control_panel_shortcut(
    actions: Res<ButtonInput<GameAction>>,
    menu: Res<CreationMenuState>,
    selection: Res<SelectedTool>,
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut panel: ResMut<ControlPanelState>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    pause: Res<PauseMenuState>,
) {
    if menu.is_open() || pause.blocks_world_input() {
        return;
    }
    if panel.is_open() {
        return;
    }
    if !actions.just_pressed(GameAction::Interact)
        || !player.world_input_active()
        || player.seat.is_some()
        || wheel.open
    {
        return;
    }
    if let Some(index) = state.hovered_bearing
        && state
            .placed_bearings
            .get(index)
            .is_some_and(|s| matches!(s.kind, mechanic_core::BearingKind::Suspension(_)))
    {
        if selection.active_editor_tool() == Some(Tool::Connector) {
            let socket = state.placed_bearings[index];
            let component = state.suspension.picked_component.map_or(0, |(_, c)| c);
            state.suspension.controls.dismissed = None;
            state.suspension.controls.select(socket, component);
        } else {
            state.feedback = Some("Equip Connector to adjust suspension".into());
        }
        return;
    }
    if hovered_part(state.hovered).is_some_and(|part| graph.0.dimension_link_id(part).is_some())
        || state
            .hovered_simulation
            .is_some_and(|hit| graph.0.dimension_link_id(hit.part).is_some())
    {
        return;
    }
    let target = hovered_part(state.hovered)
        .filter(|&part| graph.0.is_controller(part))
        .or_else(|| {
            state
                .hovered_simulation
                .map(|hit| hit.part)
                .filter(|&part| graph.0.is_controller(part))
        });
    let Some(controller) = target else {
        return;
    };
    state.selected_controller = Some(controller);
    panel.open(controller);
}

#[expect(clippy::too_many_arguments)]
pub(crate) fn handle_shortcuts(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut selection: ResMut<SelectedTool>,
    simulation: Res<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    mut material: ResMut<SelectedMaterial>,
    mut chroma_brush: ResMut<ChromaBrush>,
    mut bearing_settings: ResMut<BearingToolSettings>,
    mut cylinder_settings: ResMut<CylinderToolSettings>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
) {
    if overlay.blocks_keyboard() || !player.world_input_active() || wheel.open {
        return;
    }
    for (action, tool) in GameAction::TOOL_ACTIONS {
        if actions.just_pressed(action) {
            selection.select_tool(tool);
            break;
        }
    }
    for (action, mode) in GameAction::MODE_ACTIONS {
        if actions.just_pressed(action) {
            selection.select_mode(mode);
            break;
        }
    }
    if actions.just_pressed(GameAction::ClearPipette) {
        if selection.active_editor_tool() == Some(Tool::Chroma) {
            if let Some(appearance) = suspension_editor::sample_appearance(&graph.0, &state) {
                chroma_brush.appearance = appearance;
                state.feedback = Some("Sampled suspension appearance".into());
                return;
            }
            match appearance_target(&graph.0, &state)
                .and_then(|target| target_appearance(&graph.0, target))
            {
                Some(appearance) => {
                    chroma_brush.appearance = appearance;
                    state.feedback = Some("Sampled construction appearance".to_owned());
                }
                None => state.feedback = Some("Point at construction to sample it".to_owned()),
            }
        } else if selection.tool.is_some() {
            let mut view = live_edit::EditorView::new(&mut graph, &mut state);
            let (graph, state) = view.parts();
            clear_held_tool(&mut graph.0, state, &mut selection, &mut hammer);
        } else {
            let cursor = camera::viewport_center(Vec2::new(window.width(), window.height()));
            let ray = camera.0.viewport_to_world(camera.1, cursor).ok();
            let setup = ray.and_then(|ray| {
                pipette_at_ray(
                    &graph.0,
                    &state,
                    &simulation,
                    ray.origin,
                    ray.direction.as_vec3(),
                )
            });
            if let Some(setup) = setup {
                apply_pipette_setup(
                    setup,
                    &graph.0,
                    &mut state,
                    &mut selection,
                    &mut material,
                    &mut bearing_settings,
                    &mut cylinder_settings,
                );
            } else {
                state.feedback = Some("Nothing to pick up".to_owned());
            }
        }
    }
    if actions.just_pressed(GameAction::Rotate)
        && let Some(tool) = selection.active_editor_tool()
        && tool != Tool::Weld
    {
        state.feedback = Some(cycle_orientation(&mut state, tool));
    }
    if selection.active_editor_tool() == Some(Tool::Cylinder)
        && actions.just_pressed(GameAction::PipeTurn)
    {
        state.feedback = Some(begin_pipe_node(&graph.0, &mut state));
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PipetteSetup {
    Ground,
    Bearing(BearingDimensions),
    Linear(mechanic_core::LinearBearing, Vec3),
    Suspension(mechanic_core::SuspensionSpec, usize),
    Part(PartId),
}

pub(crate) fn pipette_socket(socket: PlacedBearing) -> PipetteSetup {
    match socket.kind {
        mechanic_core::BearingKind::Rotational => PipetteSetup::Bearing(socket.dimensions),
        mechanic_core::BearingKind::Linear(rail) => PipetteSetup::Linear(rail, socket.axis),
        mechanic_core::BearingKind::Suspension(spec) => {
            PipetteSetup::Suspension(spec, usize::from(spec.spring().is_none()))
        }
    }
}

pub(crate) fn clear_held_tool(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    selection: &mut SelectedTool,
    hammer: &mut HammerInteraction,
) {
    cancel_transient_editor_state(graph, state);
    state.wire_drag = None;
    state.vertex_drag = None;
    state.paint_selecting = false;
    state.selected_vertices.clear();
    state.active_region = None;
    *hammer = HammerInteraction::default();
    selection.clear();
    state.construction_mesh_dirty = true;
    state.feedback =
        Some("Hand cleared — Clear / Pipette picks the object under the reticle".to_owned());
}

pub(crate) fn pipette_at_ray(
    graph: &ConstructionGraph,
    state: &EditorState,
    simulation: &AppSimulation,
    origin: Vec3,
    direction: Vec3,
) -> Option<PipetteSetup> {
    if simulation.creation.is_some() {
        let creation = simulation.creation.as_ref()?;
        let graph = &simulation.published_graph;
        let part = raycast_simulation(graph, creation, &simulation.transforms, origin, direction);
        let bearing = raycast_simulation_bearings(
            graph,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
            origin,
            direction,
        );
        if let Some((index, distance, owner)) = suspension_render::raycast_scene_component(
            graph,
            Some(simulation),
            &state.placed_bearings,
            origin,
            direction,
        ) && part.is_none_or(|part| distance < part.distance)
            && bearing.is_none_or(|(_, ring_distance)| distance < ring_distance)
            && let mechanic_core::BearingKind::Suspension(spec) = state.placed_bearings[index].kind
        {
            return Some(PipetteSetup::Suspension(
                spec,
                suspension_editor::component_index(spec, owner),
            ));
        }
        if let Some((index, distance)) = linear_editor::raycast_scene(
            graph,
            Some(simulation),
            &state.placed_bearings,
            origin,
            direction,
        ) && part.is_none_or(|part| distance < part.distance)
            && bearing.is_none_or(|(_, ring_distance)| distance < ring_distance)
        {
            return Some(pipette_socket(state.placed_bearings[index]));
        }
        return match (part, bearing) {
            (Some(part), Some((dimensions, distance))) if distance < part.distance => {
                Some(PipetteSetup::Bearing(dimensions))
            }
            (Some(part), _) => Some(PipetteSetup::Part(part.part)),
            (None, Some((dimensions, _))) => Some(PipetteSetup::Bearing(dimensions)),
            (None, None) => None,
        };
    }
    let part = raycast_construction(graph, origin, direction);
    let bearing = raycast_placed_bearings(
        graph,
        None,
        &state.placed_bearings,
        origin,
        direction,
        crate::editor::raycast::BearingPick::Ring,
    )
    .and_then(|(index, distance)| Some((*state.placed_bearings.get(index)?, distance)));
    if let Some((index, distance, owner)) = suspension_render::raycast_scene_component(
        graph,
        None,
        &state.placed_bearings,
        origin,
        direction,
    ) && part.is_none_or(|part| distance < part.distance)
        && bearing.is_none_or(|(_, ring_distance)| distance <= ring_distance + 1.0e-6)
        && let mechanic_core::BearingKind::Suspension(spec) = state.placed_bearings[index].kind
    {
        return Some(PipetteSetup::Suspension(
            spec,
            suspension_editor::component_index(spec, owner),
        ));
    }
    match (part, bearing) {
        (Some(hit), Some((dimensions, distance))) if distance < hit.distance => {
            Some(pipette_socket(dimensions))
        }
        (
            Some(SurfaceHit {
                face:
                    mechanic_core::FaceRef {
                        owner: FaceOwner::Ground,
                        ..
                    },
                ..
            }),
            _,
        ) => Some(PipetteSetup::Ground),
        (Some(hit), _) => hovered_part(Some(hit)).map(PipetteSetup::Part),
        (None, Some((dimensions, _))) => Some(pipette_socket(dimensions)),
        (None, None) => None,
    }
}

pub(crate) fn apply_pipette_setup(
    setup: PipetteSetup,
    graph: &ConstructionGraph,
    state: &mut EditorState,
    selection: &mut SelectedTool,
    material: &mut SelectedMaterial,
    bearing_settings: &mut BearingToolSettings,
    cylinder_settings: &mut CylinderToolSettings,
) {
    state.active_region = None;
    let tool = match setup {
        PipetteSetup::Ground => Tool::Block,
        PipetteSetup::Suspension(spec, component) => {
            state.suspension.spring = spec.spring().unwrap_or_default();
            state.suspension.shock = spec.shock().unwrap_or_default();
            match component {
                0 => Tool::Spring,
                2 => {
                    if let Some(stop) = spec.bump_stop() {
                        state.suspension.stop = stop;
                        material.0 = ConstructionMaterial::Rubber;
                    }
                    Tool::Cylinder
                }
                _ => Tool::Shock,
            }
        }
        PipetteSetup::Linear(rail, axis) => {
            state.linear.dimensions = rail.dimensions;
            let (u, v) = axis_tangents(rail.mount_normal);
            state.linear.turns = [u, v, -u, -v]
                .iter()
                .position(|candidate| candidate.abs_diff_eq(axis, 1.0e-5))
                .and_then(|index| u8::try_from(index).ok())
                .unwrap_or(0);
            Tool::LinearBearing
        }
        PipetteSetup::Bearing(dimensions) => {
            bearing_settings.dimensions = dimensions;
            Tool::Bearing
        }
        PipetteSetup::Part(part) => {
            if let Some(region) = graph
                .region_of(part)
                .and_then(|id| graph.region(id).map(|region| (id, region)))
            {
                state.active_region = Some(region.0);
                material.0 = region.1.material();
                Tool::Shape
            } else {
                let Some(spec) = graph.part(part).copied() else {
                    state.feedback = Some("Nothing to pick up".to_owned());
                    return;
                };
                state.authored_orientation = AUTHORED_ORIENTATIONS
                    .iter()
                    .position(|rotation| *rotation == spec.pose().rotation)
                    .and_then(|index| u8::try_from(index).ok())
                    .unwrap_or_default();
                match spec {
                    PartSpec::Cuboid(spec) => {
                        material.0 = spec.material;
                        Tool::Block
                    }
                    PartSpec::Cylinder(spec) => {
                        material.0 = spec.material;
                        cylinder_settings.dimensions = spec.dimensions;
                        Tool::Cylinder
                    }
                    PartSpec::PipeJunction(spec) => {
                        material.0 = spec.material;
                        cylinder_settings.dimensions = CylinderDimensions::new(
                            spec.dimensions.outer_diameter(),
                            spec.dimensions.inner_diameter(),
                            cylinder_settings.dimensions.axial_length(),
                        )
                        .expect("stored junction cross-section is valid for a cylinder");
                        Tool::Cylinder
                    }
                    PartSpec::PipeBend(spec) => {
                        material.0 = spec.material;
                        cylinder_settings.bend_span = spec.dimensions.span_blocks();
                        cylinder_settings.dimensions = CylinderDimensions::new(
                            spec.dimensions.outer_diameter(),
                            spec.dimensions.inner_diameter(),
                            cylinder_settings.dimensions.axial_length(),
                        )
                        .expect("stored bend cross-section is valid for a cylinder");
                        Tool::Cylinder
                    }
                    PartSpec::Controller(_) => Tool::Controller,
                    PartSpec::Engine(spec) => match spec.kind {
                        EngineKind::Gas => Tool::GasEngine,
                        EngineKind::Electric => Tool::ElectricEngine,
                    },
                    PartSpec::Transmission(_) => Tool::Transmission,
                    PartSpec::Servo(_) => Tool::Servo,
                    PartSpec::Seat(_) => Tool::Seat,
                    PartSpec::Input(_) => Tool::Input,
                    PartSpec::DimensionLink(_) => Tool::DimensionLink,
                }
            }
        }
    };
    selection.select_editor_tool(tool);
    state.construction_mesh_dirty = true;
    state.feedback = Some(format!("Picked up {}", tool.label()));
}

/// What Rotate does: rotate whichever drag plane or vertex axis is open, or step
/// an authored part's orientation, reporting what to say about it.
pub(crate) fn cycle_orientation(state: &mut EditorState, tool: Tool) -> String {
    if let Some(message) = suspension_editor::cycle_drag(state, tool) {
        return message;
    }
    let sample = state.pointer_position.zip(state.pointer_ray).map(
        |(cursor, (ray_origin, ray_direction))| PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        },
    );
    if let Some(drag) = state.pipe_drag.as_mut() {
        let leaving_bearing_offset = drag.bearing_offset.take().is_some();
        if !leaving_bearing_offset {
            drag.mode = drag.mode.next();
        }
        drag.anchor_endpoint = drag.endpoint;
        drag.anchor_dimensions = drag.dimensions;
        if let Some(press) = sample {
            drag.press = press;
        }
        return format!("Pipe edit mode: {}", drag.mode.label());
    }
    if let Some(drag) = state.vertex_drag.as_mut() {
        let Some(pointer) = sample else {
            return "Move the pointer back over the world to change the shape axis".to_owned();
        };
        drag.cycle_axis(pointer.ray_origin, pointer.ray_direction);
        return format!("Shape axis: {}", drag.axis_label());
    }
    // Both drags freeze what has been dragged so far and measure from here, so
    // a rectangle plus a rotation extrudes into a box instead of starting over.
    if let Some(drag) = state.block_drag.as_mut() {
        drag.plane = drag.plane.cycle();
        drag.anchor_span = drag.span;
        if let Some(press) = sample {
            drag.press = press;
        }
        drag.last_span = None;
        return format!("Drag plane: {}", drag.plane.label());
    }
    if let Some(drag) = state.region_drag.as_mut() {
        drag.plane = drag.plane.cycle();
        drag.anchor_span = drag.span;
        if let Some(press) = sample {
            drag.press = press;
        }
        drag.last_span = None;
        return format!("Area plane: {}", drag.plane.label());
    }
    if let Some(drag) = state.delete_drag.as_mut() {
        drag.plane = drag.plane.cycle();
        drag.anchor_span = drag.span;
        if let Some(press) = sample {
            drag.press = press;
        }
        drag.last_span = None;
        return format!("Delete plane: {}", drag.plane.label());
    }
    if tool == Tool::Cylinder && state.pipe_branch_preview.is_some() {
        state.pipe_branch_turn.1 = state.pipe_branch_turn.1.wrapping_add(1);
        return "Branch turned to the next free direction".to_owned();
    }
    if matches!(
        tool,
        Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
    ) {
        state.authored_orientation = (state.authored_orientation + 1) % AUTHORED_ORIENTATION_COUNT;
        return format!(
            "{} orientation: {}/{}",
            tool.label(),
            state.authored_orientation + 1,
            AUTHORED_ORIENTATION_COUNT,
        );
    }
    "Rotate cycles machine, Seat, and Input orientations, or changes an active drag plane"
        .to_owned()
}
