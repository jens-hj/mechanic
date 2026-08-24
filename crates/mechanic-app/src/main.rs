//! Construction prototype with a GPU-physics preview.

#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

use std::{
    collections::{HashSet, VecDeque},
    error::Error,
    path::Path,
};

mod builder;
mod camera;
mod control_panel;
mod creation_menu;
mod creation_store;
mod hotbar;
mod sequencer;
mod showcase;
mod ui;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::{NoFrustumCulling, RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    image::ImageLoaderSettings,
    input::keyboard::Key,
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::PrimitiveTopology,
        renderer::{RenderDevice, RenderQueue},
    },
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use builder::{
    BEARING_DEPTH, BLOCK_SIZE_METERS, CylinderPlacementCandidate, GROUND_HALF_SIZE,
    PlacementCandidate, PlacementError, PlacementPlane, SurfaceHit, bearing_anchor_from_hit,
    bearing_attachment_candidate, bearing_overlaps_candidate, bearing_overlaps_cylinder_candidate,
    bearing_support_face, bearing_support_face_excluding, begin_weld,
    block_sheet_endpoint_from_rays, block_sheet_specs, candidate_from_hit,
    cylinder_candidate_from_hit, face_geometry_from_ref, oriented_cuboid_candidate_from_hit,
    part_world_bounds, raycast_construction, raycast_construction_for_annulus,
    raycast_oriented_cuboid, raycast_placement_plane, rigid_body_parts, stage_bearing_attachment,
    stage_bearing_block_batch, stage_bearing_cylinder, stage_block_batch_from_source,
    stage_controller_from_source, stage_cylinder_from_source, stage_engine_from_source,
    stage_input_from_source, stage_seat_from_source, stage_servo_from_source, stage_weld_objects,
    try_face_geometry_from_ref, validate_block_batch, validate_cylinder_candidate,
};
use camera::{OrbitCamera, SeatedView, seated_view_rotation};
use control_panel::ControlPanelState;
use creation_menu::{CreationMenuState, CreationRequest};
use creation_store::CreationStore;
use hotbar::{SelectedTool, Tool, shortcut_tool};
use mechanic_core::{
    ActuatorAssignment, BearingDimensions, BearingId, BearingSocket, BuildCommand,
    CYLINDER_SWEEP_STEP_DEGREES, CompiledCreation, ConstructionGraph, ControllerSpec,
    CreationDocument, CuboidSpec, CylinderDimensions, DriveLinkSpec, DriveState, DriveTarget,
    EngineKind, FaceOwner, GridRotation, InputSeatLinkSpec, InputSpec, MAX_BEARING_OUTER_DIAMETER,
    MAX_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_SWEEP_DEGREES, MIN_BEARING_DIAMETER_GAP,
    MIN_BEARING_OUTER_DIAMETER, MIN_CYLINDER_DIAMETER_GAP, MIN_CYLINDER_OUTER_DIAMETER,
    MIN_CYLINDER_SWEEP_DEGREES, PartId, PartSpec, PendingOperation, SeatControllerLinkSpec,
    SeatSpec, ServoSpec, TopologyError,
};
use mechanic_gpu::{
    FIXED_DT_SECONDS, FixedStepScheduler, GpuPhysics, GpuPhysicsConfig, GpuTransform,
};
use sequencer::{DriveKeyState, DriveSequencer, gpu_drive_rows};

const SIMULATION_VISUAL_TICK_INTERVAL: u32 = 2;
const HAMMER_CHARGE_SECONDS: f32 = 1.5;
const HAMMER_MIN_IMPULSE: f32 = 25.0;
const HAMMER_MAX_IMPULSE: f32 = 4_000.0;
const HAMMER_MAX_POINT_TRAVEL_PER_TICK: f32 = 0.05;
const HAMMER_MAX_DELIVERY_TICKS: u16 = 12;
const HISTORY_CAPACITY: usize = 64;
const BEARING_DIAMETER_STEP: f32 = 0.05;
const CYLINDER_DIAMETER_STEP: f32 = 0.05;
const CYLINDER_LENGTH_STEP: f32 = 0.25;
const CONTROLLER_SURFACE_COLOR: Color = Color::srgb(0.10, 0.78, 0.68);
const BLOCK_DRAG_DEAD_ZONE_PIXELS: f32 = 5.0;

#[derive(Resource, Default)]
struct EditorGraph(ConstructionGraph);

/// Display name of the creation currently open, when one was saved or loaded.
/// It prefills the modal's name field so re-saving keeps the same file.
#[derive(Resource, Default)]
struct CurrentCreation(Option<String>);

#[derive(Resource, Default)]
struct AppSimulation {
    gpu: Option<GpuPhysics>,
    creation: Option<CompiledCreation>,
    paused: bool,
    scheduler: FixedStepScheduler,
    next_tick: u64,
    transforms: Vec<GpuTransform>,
    visual_ticks_since_publish: u32,
    static_mesh_dirty: bool,
    render_dirty: bool,
}

#[derive(Resource, Default)]
struct HammerInteraction {
    charging: Option<HammerCharge>,
    pending: Option<HammerImpact>,
}

#[derive(Clone, Copy, Debug)]
struct HammerCharge {
    body_index: u32,
    local_point: Vec3,
    direction: Vec3,
    elapsed_seconds: f32,
}

#[derive(Clone, Copy, Debug)]
struct HammerImpact {
    body_index: u32,
    local_point: Vec3,
    impulse_per_tick: Vec3,
    remaining_ticks: u16,
}

#[derive(Clone, Copy, Debug)]
struct SimulationHit {
    part: PartId,
    body_index: u32,
    distance: f32,
    point: Vec3,
}

#[derive(Clone, Debug)]
struct BlockDrag {
    start: PlacementCandidate,
    attachment: BlockAttachment,
    press: PointerSample,
    plane: PlacementPlane,
    last_endpoint: Option<(PlacementPlane, IVec3)>,
    specs: Vec<CuboidSpec>,
    error: Option<PlacementError>,
}

#[derive(Clone, Copy, Debug)]
struct PointerSample {
    cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
enum BlockAttachment {
    AutoWeld {
        source: FaceOwner,
    },
    Bearing {
        source: mechanic_core::FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
    },
}

#[derive(Clone, Debug)]
struct DeleteDrag {
    start: CuboidSpec,
    plane: PlacementPlane,
    last_endpoint: Option<(PlacementPlane, IVec3)>,
    parts: Vec<PartId>,
    error: Option<PlacementError>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlacedBearing {
    source: mechanic_core::FaceRef,
    anchor: Vec3,
    dimensions: BearingDimensions,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
struct BearingToolSettings {
    dimensions: BearingDimensions,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
struct CylinderToolSettings {
    dimensions: CylinderDimensions,
}

#[derive(Clone, Debug)]
struct EditorSnapshot {
    graph: ConstructionGraph,
    placed_bearings: Vec<PlacedBearing>,
}

impl EditorSnapshot {
    fn capture(graph: &ConstructionGraph, state: &EditorState) -> Self {
        let mut graph = graph.clone();
        if graph.pending().is_some() {
            graph
                .apply(BuildCommand::CancelPending)
                .expect("captured pending editor operation can be cancelled");
        }
        Self {
            graph,
            placed_bearings: state.placed_bearings.clone(),
        }
    }
}

#[derive(Resource, Default)]
struct EditorHistory {
    undo: VecDeque<EditorSnapshot>,
    redo: VecDeque<EditorSnapshot>,
}

impl EditorHistory {
    fn commit(&mut self, previous: EditorSnapshot) {
        self.redo.clear();
        if self.undo.len() == HISTORY_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(previous);
    }

    fn undo(&mut self, current: EditorSnapshot) -> Option<EditorSnapshot> {
        let previous = self.undo.pop_back()?;
        self.redo.push_back(current);
        Some(previous)
    }

    fn redo(&mut self, current: EditorSnapshot) -> Option<EditorSnapshot> {
        let next = self.redo.pop_back()?;
        self.undo.push_back(current);
        Some(next)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HistoryAction {
    Undo,
    Redo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimulationShortcut {
    TogglePlayback,
    Restart,
}

#[derive(Clone, Copy, Debug)]
enum DeleteTarget {
    PlacedBearing(usize),
    Part(PartId),
}

#[allow(clippy::too_many_arguments)] // Bevy system resources are explicit parameters.
fn handle_simulation_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    overlay: Res<ui::UiInput>,
) {
    if overlay.blocks_keyboard() {
        return;
    }
    if keyboard.just_pressed(KeyCode::Escape) && simulation.is_running() {
        *simulation = AppSimulation::default();
        *hammer = HammerInteraction::default();
        state.construction_mesh_dirty = true;
        state.feedback = Some("Returned to build mode".to_owned());
        return;
    }
    let Some(shortcut) = requested_simulation_shortcut(&keyboard) else {
        return;
    };
    let restarting = simulation.is_running();
    if restarting && shortcut == SimulationShortcut::TogglePlayback {
        simulation.paused = !simulation.paused;
        *hammer = HammerInteraction::default();
        state.feedback = Some(if simulation.paused {
            "Simulation paused — Space resumes, Shift+Space restarts".to_owned()
        } else {
            "Simulation resumed".to_owned()
        });
        return;
    }
    state.block_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    *hammer = HammerInteraction::default();

    if graph.0.pending().is_some() {
        let _ = graph.0.apply(BuildCommand::CancelPending);
    }
    let creation = match graph.0.compile() {
        Ok(creation) => creation,
        Err(error) => {
            state.feedback = Some(format!("Cannot start simulation: {error}"));
            return;
        }
    };
    let physics_config = GpuPhysicsConfig {
        mechanism_self_collisions: !showcase::uses_reduced_collision_mode(&graph.0),
        ..GpuPhysicsConfig::default()
    };
    let gpu = match GpuPhysics::new_with_config(
        render_device.wgpu_device(),
        &render_queue,
        &creation,
        physics_config,
    ) {
        Ok(gpu) => gpu,
        Err(error) => {
            state.feedback = Some(format!("Cannot start simulation: {error}"));
            return;
        }
    };
    let transforms = creation
        .compounds
        .iter()
        .map(|compound| {
            let position = compound.root_translation;
            let rotation = compound.root_rotation;
            GpuTransform {
                position: [position.x, position.y, position.z, 0.0],
                rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
            }
        })
        .collect();
    *simulation = AppSimulation {
        gpu: Some(gpu),
        creation: Some(creation),
        paused: false,
        scheduler: FixedStepScheduler::new(),
        next_tick: 1,
        transforms,
        visual_ticks_since_publish: 0,
        static_mesh_dirty: true,
        render_dirty: true,
    };
    state.feedback = Some(if restarting {
        "Simulation restarted".to_owned()
    } else {
        "Simulation running (throttled mesh preview)".to_owned()
    });
}

fn requested_simulation_shortcut(keyboard: &ButtonInput<KeyCode>) -> Option<SimulationShortcut> {
    keyboard.just_pressed(KeyCode::Space).then(|| {
        if keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            SimulationShortcut::Restart
        } else {
            SimulationShortcut::TogglePlayback
        }
    })
}

/// Whether the primary modifier plus `S` was pressed this frame.
///
/// A modifier is required because a bare letter binds to a drive state, so
/// plain `S` belongs to a machine rather than to the editor.
fn save_shortcut_requested(keyboard: &ButtonInput<KeyCode>) -> bool {
    keyboard.any_pressed([
        KeyCode::ControlLeft,
        KeyCode::ControlRight,
        KeyCode::SuperLeft,
        KeyCode::SuperRight,
    ]) && keyboard.just_pressed(KeyCode::KeyS)
}

/// Opens the creations modal with `P`, or with the primary modifier and `S`.
///
/// While it is open the modal owns the keyboard, so neither key reaches here:
/// `p` and `s` type into its name field, and Escape is its own to handle. The
/// control-block panel owns the keyboard the same way, and the two must never
/// both be typing, so neither can open over the other.
#[allow(clippy::too_many_arguments)] // Bevy system resources are explicit parameters.
fn handle_creation_menu_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
    store: Res<CreationStore>,
    current: Res<CurrentCreation>,
    panel: Res<ControlPanelState>,
    mut menu: ResMut<CreationMenuState>,
) {
    if menu.is_open() || panel.blocks_keyboard() {
        return;
    }
    let saving = save_shortcut_requested(&keyboard);
    if !saving && !keyboard.just_pressed(KeyCode::KeyP) {
        return;
    }
    if simulation.is_running() {
        state.feedback =
            Some("Creations are saved and opened in build mode — press Escape first".to_owned());
        return;
    }
    cancel_transient_editor_state(&mut graph.0, &mut state);
    menu.open(
        store.list(),
        current.0.clone().unwrap_or_default(),
        store.directory().to_path_buf(),
    );
    state.feedback = Some(if saving {
        "Type a name, then Enter to save".to_owned()
    } else {
        "Open a creation, or type a name to save this one".to_owned()
    });
}

/// Opens or closes the control-block panel with `E`.
///
/// The panel targets the hovered control block, falling back to the selected
/// one, so it opens both from the world and from whatever was last wired.
fn handle_control_panel_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    menu: Res<CreationMenuState>,
    overlay: Res<ui::UiInput>,
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut panel: ResMut<ControlPanelState>,
    simulation: Res<AppSimulation>,
) {
    if menu.is_open() {
        return;
    }
    if panel.is_open() {
        // Escape backs out of a value being typed or a key being bound before
        // it closes the panel; the panel itself handles those, seeing the same
        // press later in the frame.
        if keyboard.just_pressed(KeyCode::Escape) && !overlay.escape_is_consumed() {
            panel.close();
            state.feedback = Some("Control block panel closed".to_owned());
        }
        return;
    }
    if simulation.is_running() {
        return;
    }
    if !keyboard.just_pressed(KeyCode::KeyE) {
        return;
    }
    let target = hovered_part(state.hovered)
        .filter(|&part| graph.0.is_controller(part))
        .or_else(|| {
            state
                .selected_controller
                .filter(|&part| graph.0.is_controller(part))
        });
    let Some(controller) = target else {
        state.feedback = Some("Point at a control block, or select one, then press E".to_owned());
        return;
    };
    state.selected_controller = Some(controller);
    panel.open(controller);
}

#[allow(clippy::too_many_arguments)]
fn handle_seat_interaction(
    keyboard: Res<ButtonInput<KeyCode>>,
    motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    overlay: Res<ui::UiInput>,
    graph: Res<EditorGraph>,
    simulation: Res<AppSimulation>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut camera: Single<(&Camera, &GlobalTransform, &mut Transform), With<OrbitCamera>>,
    mut seated: ResMut<SeatedView>,
    mut state: ResMut<EditorState>,
) {
    if !simulation.is_running() {
        if seated.seat.take().is_some() {
            cursor.grab_mode = CursorGrabMode::None;
            cursor.visible = true;
        }
        return;
    }

    if keyboard.just_pressed(KeyCode::KeyE) && !overlay.blocks_keyboard() {
        if seated.seat.is_some() {
            seated.leave();
            cursor.grab_mode = CursorGrabMode::None;
            cursor.visible = true;
            state.feedback = Some("Left Seat".to_owned());
            return;
        }
        let (camera_component, camera_global, _) = &mut *camera;
        let cursor_position = Vec2::new(window.width() * 0.5, window.height() * 0.5);
        let hit = camera_component
            .viewport_to_world(camera_global, cursor_position)
            .ok()
            .and_then(|ray| {
                raycast_simulation(
                    &graph.0,
                    simulation
                        .creation
                        .as_ref()
                        .expect("running simulation has a compiled creation"),
                    &simulation.transforms,
                    ray.origin,
                    ray.direction.as_vec3(),
                )
            });
        if let Some(hit) = hit
            && graph.0.is_seat(hit.part)
        {
            seated.seat = Some(hit.part);
            seated.yaw = 0.0;
            seated.pitch = 0.0;
            cursor.grab_mode = CursorGrabMode::Locked;
            cursor.visible = false;
            state.feedback = Some("Seated — mouse looks around, E leaves the Seat".to_owned());
        } else {
            state.feedback = Some("Look at a Seat and press E".to_owned());
        }
    }

    let Some(seat) = seated.seat else {
        return;
    };
    let Some(creation) = simulation.creation.as_ref() else {
        return;
    };
    let Some((_, compound)) = creation
        .part_to_compound
        .iter()
        .find(|(part, _)| *part == seat)
    else {
        seated.leave();
        return;
    };
    let Some(snapshot) = simulation.transforms.get(*compound as usize) else {
        return;
    };
    let Some(PartSpec::Seat(spec)) = graph.0.part(seat).copied() else {
        seated.leave();
        return;
    };
    seated.yaw -= motion.delta.x * 0.0025;
    seated.pitch = (seated.pitch - motion.delta.y * 0.0025).clamp(
        -core::f32::consts::FRAC_PI_2 + 0.08,
        core::f32::consts::FRAC_PI_2 - 0.08,
    );
    let root_position = Vec3::from_slice(&snapshot.position[..3]);
    let root_rotation = Quat::from_array(snapshot.rotation);
    let local_rotation = spec.pose.rotation.quaternion();
    let seat_rotation = root_rotation * local_rotation;
    let seat_center = root_position
        + root_rotation
            * (spec.pose.translation() - creation.compounds[*compound as usize].root_translation);
    let (_, _, transform) = &mut *camera;
    transform.translation = seat_center + seat_rotation * (Vec3::Y * 0.475);
    transform.rotation = seated_view_rotation(seat_rotation, seated.yaw, seated.pitch);
}

/// Advances every driven bearing's program and pushes changed rows to the GPU.
///
/// Runs immediately before the tick is dispatched, so a state entered this
/// frame takes effect in the same tick rather than the next one.
fn run_drive_sequencer(
    keyboard: Res<ButtonInput<KeyCode>>,
    graph: Res<EditorGraph>,
    overlay: Res<ui::UiInput>,
    simulation: Res<AppSimulation>,
    mut sequencer: ResMut<DriveSequencer>,
    mut state: ResMut<EditorState>,
    seated: Res<SeatedView>,
) {
    if !simulation.is_running() {
        if sequencer.is_started() {
            sequencer.stop();
        }
        return;
    }
    if !sequencer.is_started() {
        let Some(creation) = simulation.creation.as_ref() else {
            return;
        };
        sequencer.start(creation, &graph.0);
        state.drive_rows_dirty = true;
    }
    if simulation.is_paused() {
        return;
    }
    let keys = DriveKeyState::from_keyboard(&keyboard, overlay.blocks_keyboard());
    let keyboard_controller = seated
        .seat
        .filter(|seat| graph.0.seat_input(*seat).is_some())
        .and_then(|seat| graph.0.seat_controller(seat));
    if sequencer.step(&graph.0, &keys, keyboard_controller, simulation.next_tick) {
        state.drive_rows_dirty = true;
    }
}

/// Applies whatever the creations modal decided this frame.
#[allow(clippy::too_many_arguments)] // Bevy system resources are explicit parameters.
fn handle_creation_request(
    mut menu: ResMut<CreationMenuState>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    mut current: ResMut<CurrentCreation>,
    store: Res<CreationStore>,
    mut camera: Single<(&mut OrbitCamera, &mut Transform)>,
) {
    let Some(request) = menu.take_request() else {
        return;
    };
    match request {
        CreationRequest::LoadPreset(preset) => {
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match showcase::build_preset(preset).and_then(|candidate| {
                install_editor_graph(&mut graph.0, candidate).map_err(showcase::ShowcaseError::from)
            }) {
                Ok(creation) => {
                    debug_assert_eq!(creation.compounds.len(), preset.body_count());
                    history.commit(previous);
                    current.0 = None;
                    adopt_loaded_creation(&mut state, &mut camera, &graph.0, Vec::new());
                    state.feedback = Some(format!(
                        "Opened {}: {} welds, {} bearings, {} bodies — Space to simulate",
                        preset.label(),
                        graph.0.weld_count(),
                        graph.0.bearing_count(),
                        creation.compounds.len(),
                    ));
                }
                Err(error) => {
                    state.feedback = Some(format!("Could not open creation: {error}"));
                }
            }
        }
        CreationRequest::Load(path) => {
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match load_creation(&mut graph.0, &path) {
                Ok((name, sockets, creation)) => {
                    history.commit(previous);
                    let bodies = creation.compounds.len();
                    let placed = sockets
                        .into_iter()
                        .map(|socket| PlacedBearing {
                            source: socket.source,
                            anchor: socket.anchor,
                            dimensions: socket.dimensions,
                        })
                        .collect();
                    adopt_loaded_creation(&mut state, &mut camera, &graph.0, placed);
                    state.feedback = Some(format!(
                        "Opened \"{name}\": {} parts, {} bearings, {bodies} bodies — Space to simulate",
                        graph.0.part_count(),
                        graph.0.bearing_count(),
                    ));
                    current.0 = Some(name);
                }
                Err(error) => {
                    state.feedback = Some(format!("Could not open creation: {error}"));
                }
            }
        }
        CreationRequest::Save(name) => {
            // Saving reads the graph and writes a file; it changes no
            // construction, so it commits no undo entry.
            let document = capture_creation(&graph.0, &state, &name);
            match store.save(&document) {
                Ok(path) => {
                    current.0 = Some(name.clone());
                    state.feedback = Some(format!("Saved \"{name}\" to {}", path.display()));
                }
                Err(error) => {
                    state.feedback = Some(format!("Could not save creation: {error}"));
                }
            }
        }
        CreationRequest::Delete(path) => match creation_store::delete(&path) {
            Ok(()) => {
                state.feedback = Some(format!("Deleted {}", path.display()));
                if menu.is_open() {
                    menu.set_entries(store.list());
                }
            }
            Err(error) => {
                state.feedback = Some(format!("Could not delete creation: {error}"));
                if menu.is_open() {
                    menu.notify(format!("Could not delete: {error}"));
                }
            }
        },
    }
}

/// Captures everything a creation is: the construction, plus the bearing rings
/// the editor is holding that no part hangs from yet.
fn capture_creation(
    graph: &ConstructionGraph,
    state: &EditorState,
    name: &str,
) -> CreationDocument {
    let snapshot = EditorSnapshot::capture(graph, state);
    let sockets = snapshot
        .placed_bearings
        .iter()
        .map(|bearing| BearingSocket {
            source: bearing.source,
            anchor: bearing.anchor,
            dimensions: bearing.dimensions,
        })
        .collect::<Vec<_>>();
    CreationDocument::from_graph(&snapshot.graph, name, &sockets)
}

/// Reads a creation file and installs it, compiling before it commits.
fn load_creation(
    current: &mut ConstructionGraph,
    path: &Path,
) -> Result<(String, Vec<BearingSocket>, CompiledCreation), Box<dyn Error>> {
    let loaded = creation_store::read_document(path)?.into_graph()?;
    let creation = install_editor_graph(current, loaded.graph)?;
    Ok((loaded.name, loaded.sockets, creation))
}

/// Clears the transient editing state a freshly opened creation invalidates,
/// then frames the camera on what arrived.
fn adopt_loaded_creation(
    state: &mut EditorState,
    camera: &mut Single<(&mut OrbitCamera, &mut Transform)>,
    graph: &ConstructionGraph,
    placed_bearings: Vec<PlacedBearing>,
) {
    clear_hover(state);
    state.block_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    state.selected_controller = None;
    state.placed_bearings = placed_bearings;
    state.construction_mesh_dirty = true;
    if let Some((minimum, maximum)) = graph_bounds(graph) {
        let (orbit, transform) = &mut **camera;
        orbit.frame_bounds(minimum, maximum);
        **transform = orbit.transform();
    }
}

fn install_editor_graph(
    current: &mut ConstructionGraph,
    candidate: ConstructionGraph,
) -> Result<CompiledCreation, TopologyError> {
    let creation = candidate.compile()?;
    *current = candidate;
    Ok(creation)
}

fn graph_bounds(graph: &ConstructionGraph) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for (_, spec) in graph.parts() {
        let (part_minimum, part_maximum) = part_world_bounds(*spec);
        minimum = minimum.min(part_minimum);
        maximum = maximum.max(part_maximum);
    }
    minimum.is_finite().then_some((minimum, maximum))
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]
fn advance_simulation(
    time: Res<Time>,
    graph: Res<EditorGraph>,
    sequencer: Res<DriveSequencer>,
    selection: Res<SelectedTool>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visibility: Single<
        &mut Visibility,
        (
            With<ConstructionVisual>,
            Without<BearingVisual>,
            Without<SimulationVisual>,
        ),
    >,
    mut bearing_visibility: Single<
        &mut Visibility,
        (
            With<BearingVisual>,
            Without<ConstructionVisual>,
            Without<SimulationVisual>,
        ),
    >,
    mut authored_visuals: Query<
        (&AuthoredPartVisual, &mut Visibility),
        (
            Without<ConstructionVisual>,
            Without<BearingVisual>,
            Without<SimulationVisual>,
        ),
    >,
    mut simulation_visibility: Single<
        &mut Visibility,
        (
            With<SimulationVisual>,
            Without<ConstructionVisual>,
            Without<BearingVisual>,
        ),
    >,
) {
    if !simulation.is_running() {
        return;
    }

    if state.drive_rows_dirty {
        state.drive_rows_dirty = false;
        if let (Some(gpu), Some(creation)) = (simulation.gpu.as_ref(), simulation.creation.as_ref())
            && let Err(error) = gpu.write_mechanism_drives(
                &render_queue,
                &gpu_drive_rows(creation, &graph.0, &sequencer),
            )
        {
            stop_failed_simulation(&mut simulation, &mut state, error.to_string());
            return;
        }
    }

    let tick = {
        let AppSimulation {
            paused,
            scheduler,
            next_tick,
            ..
        } = &mut *simulation;
        next_simulation_tick(scheduler, next_tick, time.delta(), *paused)
    };
    if let Some(tick) = tick {
        if let Err(error) =
            apply_pending_hammer_impact(&simulation, &mut hammer, &render_device, &render_queue)
        {
            hammer.pending = None;
            stop_failed_simulation(&mut simulation, &mut state, error);
            return;
        }
        let publishing =
            simulation.visual_ticks_since_publish + 1 >= SIMULATION_VISUAL_TICK_INTERVAL;
        let diagnostics = {
            let gpu = simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state");
            gpu.dispatch_tick(render_device.wgpu_device(), &render_queue, tick);
            // Reading diagnostics drains the whole queue, so it is only done on
            // the ticks that already stall to publish a snapshot. The kernels of
            // every other tick overlap with rendering instead, and a failure is
            // still caught on the next published tick.
            publishing.then(|| {
                gpu.read_last_tick(render_device.wgpu_device())
                    .map_err(|error| error.to_string())
            })
        };
        match diagnostics {
            None => {}
            Some(Ok(diagnostics)) if diagnostics.error_flags == 0 => {}
            Some(Ok(diagnostics)) => {
                stop_failed_simulation(
                    &mut simulation,
                    &mut state,
                    format!("physics kernel reported flags {}", diagnostics.error_flags),
                );
                return;
            }
            Some(Err(error)) => {
                stop_failed_simulation(&mut simulation, &mut state, error);
                return;
            }
        }

        simulation.visual_ticks_since_publish += 1;
        if simulation.visual_ticks_since_publish >= SIMULATION_VISUAL_TICK_INTERVAL {
            let transforms = simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state")
                .read_snapshot_transforms(
                    render_device.wgpu_device(),
                    &render_queue,
                    u8::try_from(tick % 3).unwrap_or(0),
                )
                .map_err(|error| error.to_string());
            match transforms {
                Ok(transforms) => {
                    simulation.transforms = transforms;
                    simulation.visual_ticks_since_publish = 0;
                    simulation.render_dirty = true;
                }
                Err(error) => {
                    stop_failed_simulation(&mut simulation, &mut state, error);
                    return;
                }
            }
        }
    }

    if simulation.static_mesh_dirty {
        let creation = simulation
            .creation
            .as_ref()
            .expect("running simulation has compiled creation");
        if let Some(mut mesh) = meshes.get_mut(&visuals.construction_mesh) {
            *mesh = renderable_mesh(combined_simulation_mesh(
                &graph.0,
                creation,
                &simulation.transforms,
                SimulationMeshKind::Static,
            ));
        }
        **construction_visibility = Visibility::Visible;
        simulation.static_mesh_dirty = false;
    }

    if !simulation.render_dirty {
        return;
    }
    let creation = simulation
        .creation
        .as_ref()
        .expect("running simulation has compiled creation");
    if let Some(mut mesh) = meshes.get_mut(&visuals.simulation_mesh) {
        *mesh = renderable_mesh(combined_simulation_mesh(
            &graph.0,
            creation,
            &simulation.transforms,
            SimulationMeshKind::Dynamic,
        ));
    }
    // Every mesh below is written only while its own visual is on screen. A
    // hidden mesh has no slab allocation, so writing to one both wastes the
    // rebuild and makes the renderer log a use-after-free every frame.
    let bearings_visible = graph.0.bearing_count() > 0 || !state.placed_bearings.is_empty();
    for appearance in AuthoredPart::ALL {
        let visible = graph.0.parts().any(|(_, spec)| appearance.matches(*spec));
        if visible && let Some(mut mesh) = meshes.get_mut(visuals.authored_mesh(appearance)) {
            *mesh = combined_simulation_authored_mesh(
                &graph.0,
                creation,
                &simulation.transforms,
                appearance,
            );
        }
        for (visual, mut visibility) in &mut authored_visuals {
            if visual.0 == appearance {
                *visibility = if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
    if bearings_visible && let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
        *mesh = combined_simulation_bearing_mesh(
            &graph.0,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
        );
    }
    // The drive overlay follows the bodies while they move, so it is rebuilt
    // from the same published snapshot -- but only while it is on screen. A
    // hidden mesh has no slab allocation, so writing to it every frame both
    // wastes the rebuild and makes the renderer log a use-after-free.
    if drive_xray_is_visible(selection.0, control_link_count(&graph.0))
        && let Some(mut mesh) = meshes.get_mut(&visuals.drive_xray_mesh)
    {
        *mesh = combined_simulation_drive_xray_mesh(
            &graph.0,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
            &sequencer,
        );
    }
    **bearing_visibility = if bearings_visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    **simulation_visibility = Visibility::Visible;
    simulation.render_dirty = false;
}

fn next_simulation_tick(
    scheduler: &mut FixedStepScheduler,
    next_tick: &mut u64,
    elapsed: std::time::Duration,
    paused: bool,
) -> Option<u64> {
    if paused {
        return None;
    }
    if scheduler.advance(elapsed).count() == 0 {
        return None;
    }
    let tick = *next_tick;
    *next_tick = next_tick.saturating_add(1);
    Some(tick)
}

fn stop_failed_simulation(simulation: &mut AppSimulation, state: &mut EditorState, error: String) {
    *simulation = AppSimulation::default();
    state.construction_mesh_dirty = true;
    state.feedback = Some(format!("Simulation stopped: {error}"));
}

impl AppSimulation {
    const fn is_running(&self) -> bool {
        self.gpu.is_some()
    }

    const fn is_paused(&self) -> bool {
        self.is_running() && self.paused
    }
}

#[derive(Resource, Default)]
struct EditorState {
    hovered: Option<SurfaceHit>,
    /// Unattached bearing surface directly hit by the pointer ray.
    hovered_bearing: Option<usize>,
    /// Unattached bearing that would claim the current block preview.
    attachment_bearing: Option<usize>,
    preview: Option<PlacementCandidate>,
    cylinder_preview: Option<CylinderPlacementCandidate>,
    preview_error: Option<PlacementError>,
    /// One of the 24 grid-aligned orientations used by authored parts.
    authored_orientation: u8,
    feedback: Option<String>,
    construction_mesh_dirty: bool,
    delete_target: Option<DeleteTarget>,
    block_drag: Option<BlockDrag>,
    block_preview_revision: u64,
    delete_drag: Option<DeleteDrag>,
    delete_preview_revision: u64,
    placed_bearings: Vec<PlacedBearing>,
    /// Control block the panel edits, and the one a new wire starts from.
    selected_controller: Option<PartId>,
    /// Drive rows changed and a running simulation still holds the old ones.
    drive_rows_dirty: bool,
    /// Drive wire the pointer is dragging out.
    wire_drag: Option<WireDrag>,
    /// Latest pointer ray, so a dragged wire can follow the cursor.
    pointer_ray: Option<(Vec3, Vec3)>,
    /// Latest pointer position paired with [`Self::pointer_ray`].
    pointer_position: Option<Vec2>,
}

#[derive(Resource)]
struct EditorVisuals {
    construction_mesh: Handle<Mesh>,
    simulation_mesh: Handle<Mesh>,
    bearing_mesh: Handle<Mesh>,
    joint_xray_mesh: Handle<Mesh>,
    controller_mesh: Handle<Mesh>,
    gas_engine_mesh: Handle<Mesh>,
    electric_engine_mesh: Handle<Mesh>,
    servo_mesh: Handle<Mesh>,
    seat_mesh: Handle<Mesh>,
    input_mesh: Handle<Mesh>,
    authored_preview_meshes: [Handle<Mesh>; 6],
    authored_preview_materials: [Handle<StandardMaterial>; 6],
    invalid_authored_preview_materials: [Handle<StandardMaterial>; 6],
    drive_xray_mesh: Handle<Mesh>,
    wire_drag_mesh: Handle<Mesh>,
    wire_hover_mesh: Handle<Mesh>,
    cube_preview_mesh: Handle<Mesh>,
    cylinder_preview_mesh: Handle<Mesh>,
    bearing_preview_mesh: Handle<Mesh>,
    white_preview_material: Handle<StandardMaterial>,
    green_preview_material: Handle<StandardMaterial>,
    red_preview_material: Handle<StandardMaterial>,
    block_drag_preview_mesh: Handle<Mesh>,
    delete_drag_preview_mesh: Handle<Mesh>,
    weld_hover_preview_mesh: Handle<Mesh>,
    weld_selection_preview_mesh: Handle<Mesh>,
}

impl EditorVisuals {
    fn authored_mesh(&self, appearance: AuthoredPart) -> &Handle<Mesh> {
        match appearance {
            AuthoredPart::Controller => &self.controller_mesh,
            AuthoredPart::GasEngine => &self.gas_engine_mesh,
            AuthoredPart::ElectricEngine => &self.electric_engine_mesh,
            AuthoredPart::Servo => &self.servo_mesh,
            AuthoredPart::Seat => &self.seat_mesh,
            AuthoredPart::Input => &self.input_mesh,
        }
    }

    fn authored_preview_mesh(&self, appearance: AuthoredPart) -> &Handle<Mesh> {
        &self.authored_preview_meshes[appearance.index()]
    }

    fn authored_preview_material(
        &self,
        appearance: AuthoredPart,
        invalid: bool,
    ) -> &Handle<StandardMaterial> {
        let materials = if invalid {
            &self.invalid_authored_preview_materials
        } else {
            &self.authored_preview_materials
        };
        &materials[appearance.index()]
    }
}

#[derive(Component)]
struct ActionPreview;

#[derive(Component)]
struct SelectionPreview;

#[derive(Component)]
struct DeletePreview;

#[derive(Component)]
struct ConstructionVisual;

#[derive(Component)]
struct SimulationVisual;

#[derive(Component)]
struct BearingVisual;

#[derive(Component)]
struct JointXrayVisual;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthoredPart {
    Controller,
    GasEngine,
    ElectricEngine,
    Servo,
    Seat,
    Input,
}

impl AuthoredPart {
    const ALL: [Self; 6] = [
        Self::Controller,
        Self::GasEngine,
        Self::ElectricEngine,
        Self::Servo,
        Self::Seat,
        Self::Input,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Controller => 0,
            Self::GasEngine => 1,
            Self::ElectricEngine => 2,
            Self::Servo => 3,
            Self::Seat => 4,
            Self::Input => 5,
        }
    }

    const fn from_tool(tool: Tool) -> Option<Self> {
        match tool {
            Tool::Controller => Some(Self::Controller),
            Tool::GasEngine => Some(Self::GasEngine),
            Tool::ElectricEngine => Some(Self::ElectricEngine),
            Tool::Servo => Some(Self::Servo),
            Tool::Seat => Some(Self::Seat),
            Tool::Input => Some(Self::Input),
            _ => None,
        }
    }

    fn matches(self, spec: PartSpec) -> bool {
        matches!(
            (self, spec),
            (Self::Controller, PartSpec::Controller(_))
                | (
                    Self::GasEngine,
                    PartSpec::Engine(mechanic_core::EngineSpec {
                        kind: EngineKind::Gas,
                        ..
                    }),
                )
                | (Self::Servo, PartSpec::Servo(_))
                | (Self::Seat, PartSpec::Seat(_))
                | (Self::Input, PartSpec::Input(_))
                | (
                    Self::ElectricEngine,
                    PartSpec::Engine(mechanic_core::EngineSpec {
                        kind: EngineKind::Electric,
                        ..
                    }),
                )
        )
    }
}

const AUTHORED_ORIENTATION_COUNT: u8 = 24;
const AUTHORED_ORIENTATIONS: [GridRotation; 24] = [
    GridRotation::new(0, 0, 0),
    GridRotation::new(0, 1, 0),
    GridRotation::new(0, 2, 0),
    GridRotation::new(0, 3, 0),
    GridRotation::new(0, 0, 1),
    GridRotation::new(0, 0, 2),
    GridRotation::new(0, 0, 3),
    GridRotation::new(0, 1, 1),
    GridRotation::new(0, 1, 2),
    GridRotation::new(0, 1, 3),
    GridRotation::new(0, 2, 1),
    GridRotation::new(0, 2, 2),
    GridRotation::new(0, 2, 3),
    GridRotation::new(0, 3, 1),
    GridRotation::new(0, 3, 2),
    GridRotation::new(0, 3, 3),
    GridRotation::new(1, 0, 0),
    GridRotation::new(1, 0, 1),
    GridRotation::new(1, 0, 2),
    GridRotation::new(1, 0, 3),
    GridRotation::new(1, 2, 0),
    GridRotation::new(1, 2, 1),
    GridRotation::new(1, 2, 2),
    GridRotation::new(1, 2, 3),
];

fn authored_orientation(index: u8) -> GridRotation {
    AUTHORED_ORIENTATIONS[usize::from(index) % AUTHORED_ORIENTATIONS.len()]
}

#[derive(Component)]
struct AuthoredPartVisual(AuthoredPart);

#[derive(Component)]
struct DriveXrayVisual;

/// The wire the pointer is currently dragging between a block and a bearing.
#[derive(Component)]
struct WireDragVisual;

/// The joint or control block the pointer would wire, drawn oversized.
#[derive(Component)]
struct WireHoverVisual;

const BEARING_RENDER_DEPTH_BIAS: f32 = 2.0;
const BEARING_RENDER_RADIAL_SKIN: f32 = 0.001;
const PREVIEW_RENDER_DEPTH_BIAS: f32 = 1.0;
/// Matches the 0.992 scale of a single 0.25 m block preview without making
/// large sheet previews shrink in proportion to their full width.
const BLOCK_SHEET_PREVIEW_INSET_METERS: f32 = 0.001;

fn bearing_surface_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        metallic: 0.35,
        perceptual_roughness: 0.55,
        depth_bias: BEARING_RENDER_DEPTH_BIAS,
        ..default()
    }
}

fn preview_material(base_color: Color) -> StandardMaterial {
    StandardMaterial {
        base_color,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    }
}

fn authored_part_material(asset_server: &AssetServer, stem: &str) -> StandardMaterial {
    let linear_texture = |suffix: &str| {
        asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| settings.is_srgb = false)
            .load(format!("{stem}_{suffix}.png"))
    };
    let orm = linear_texture("orm");
    StandardMaterial {
        base_color_texture: Some(asset_server.load(format!("{stem}_base_color.png"))),
        metallic: 1.0,
        perceptual_roughness: 1.0,
        metallic_roughness_texture: Some(orm.clone()),
        occlusion_texture: Some(orm),
        normal_map_texture: Some(linear_texture("normal")),
        emissive: LinearRgba::WHITE,
        emissive_texture: Some(asset_server.load(format!("{stem}_emissive.png"))),
        ..default()
    }
}

fn authored_preview_material(
    mut material: StandardMaterial,
    base_color: Color,
) -> StandardMaterial {
    material.base_color = base_color;
    material.alpha_mode = AlphaMode::Blend;
    material.cull_mode = None;
    material.depth_bias = PREVIEW_RENDER_DEPTH_BIAS;
    material
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mechanic — construction and simulation prototype".to_owned(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        // After DefaultPlugins: the overlay's render pass installs into the
        // render sub-app, which does not exist until RenderPlugin has run.
        .add_plugins(bevy_mosaic::MosaicPlugin)
        .init_resource::<EditorGraph>()
        .init_resource::<EditorState>()
        .init_resource::<EditorHistory>()
        .init_resource::<CreationMenuState>()
        .init_resource::<CreationStore>()
        .init_resource::<CurrentCreation>()
        .init_resource::<AppSimulation>()
        .init_resource::<HammerInteraction>()
        .init_resource::<BearingToolSettings>()
        .init_resource::<ControlPanelState>()
        .init_resource::<DriveSequencer>()
        .init_resource::<CylinderToolSettings>()
        .init_resource::<SelectedTool>()
        .init_resource::<SeatedView>()
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.75, 0.80, 0.90),
            brightness: 350.0,
            ..default()
        })
        .add_systems(Startup, (setup, ui::mount).chain())
        .add_systems(
            Update,
            (
                (
                    handle_creation_menu_shortcut,
                    handle_control_panel_shortcut,
                    ui::drain,
                    ui::push,
                    ui::push_help,
                    ui::push_markers,
                    ui::sync_input,
                    handle_history_shortcut,
                    handle_creation_request,
                )
                    .chain(),
                camera::update_orbit_camera,
                handle_simulation_shortcut,
                handle_seat_interaction,
                handle_shortcuts,
                (
                    handle_bearing_dimension_shortcuts,
                    handle_cylinder_dimension_shortcuts,
                )
                    .chain(),
                handle_tool_change,
                update_hover,
                handle_build_actions,
                ui::push_dimensions,
                handle_hammer_actions,
                update_joint_xray,
                sync_visual_meshes,
                update_wire_drag_preview,
                update_wire_hover_preview,
                run_drive_sequencer,
                advance_simulation,
                update_previews,
            )
                .chain(),
        )
        .run();
}

#[allow(clippy::too_many_lines)] // One-time Bevy scene composition is clearest in declaration order.
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let construction_mesh = meshes.add(Cuboid::default());
    let simulation_mesh = meshes.add(Cuboid::default());
    let bearing_mesh = meshes.add(Cuboid::default());
    let joint_xray_mesh = meshes.add(Cuboid::default());
    let controller_mesh = meshes.add(Cuboid::default());
    let gas_engine_mesh = meshes.add(Cuboid::default());
    let electric_engine_mesh = meshes.add(Cuboid::default());
    let servo_mesh = meshes.add(Cuboid::default());
    let seat_mesh = meshes.add(Cuboid::default());
    let input_mesh = meshes.add(Cuboid::default());
    let authored_preview_meshes =
        AuthoredPart::ALL.map(|appearance| meshes.add(single_authored_part_mesh(appearance)));
    let drive_xray_mesh = meshes.add(Cuboid::default());
    let wire_drag_mesh = meshes.add(wire_drag_preview_mesh(Vec3::ZERO, Vec3::ZERO));
    let wire_hover_mesh = meshes.add(degenerate_overlay_mesh());
    let cube_preview_mesh = meshes.add(Cuboid::default());
    let cylinder_preview_mesh = meshes.add(single_cylinder_mesh(CylinderDimensions::default()));
    let bearing_preview_mesh = meshes.add(single_bearing_mesh(BearingDimensions::default()));
    let block_drag_preview_mesh = meshes.add(Cuboid::default());
    let delete_drag_preview_mesh = meshes.add(Cuboid::default());
    let weld_hover_preview_mesh = meshes.add(Cuboid::default());
    let weld_selection_preview_mesh = meshes.add(Cuboid::default());
    let construction_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.48, 0.78),
        perceptual_roughness: 0.8,
        ..default()
    });
    let bearing_material = materials.add(bearing_surface_material());
    let authored_materials = [
        authored_part_material(&asset_server, "machines/controller/controller"),
        authored_part_material(&asset_server, "machines/gas_engine/gas_engine"),
        authored_part_material(&asset_server, "machines/electric_engine/electric_engine"),
        authored_part_material(&asset_server, "machines/servo/servo"),
        authored_part_material(&asset_server, "machines/seat/seat"),
        authored_part_material(&asset_server, "machines/input/input"),
    ];
    let authored_preview_materials = std::array::from_fn(|index| {
        materials.add(authored_preview_material(
            authored_materials[index].clone(),
            Color::srgba(1.0, 1.0, 1.0, 0.46),
        ))
    });
    let invalid_authored_preview_materials = std::array::from_fn(|index| {
        materials.add(authored_preview_material(
            authored_materials[index].clone(),
            Color::srgba(1.0, 0.18, 0.16, 0.52),
        ))
    });
    let authored_materials = authored_materials.map(|material| materials.add(material));
    let drive_xray_material = materials.add(StandardMaterial {
        base_color: CONTROLLER_SURFACE_COLOR,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let wire_drag_material = drive_xray_material.clone();
    let wire_hover_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.86, 0.99, 1.0),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let joint_xray_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let white_preview_material = materials.add(preview_material(Color::srgba(1.0, 1.0, 1.0, 0.34)));
    let red_preview_material = materials.add(preview_material(Color::srgba(1.0, 0.06, 0.04, 0.46)));
    let green_preview_material =
        materials.add(preview_material(Color::srgba(0.12, 1.0, 0.28, 0.52)));

    commands.insert_resource(EditorVisuals {
        construction_mesh: construction_mesh.clone(),
        simulation_mesh: simulation_mesh.clone(),
        bearing_mesh: bearing_mesh.clone(),
        joint_xray_mesh: joint_xray_mesh.clone(),
        controller_mesh: controller_mesh.clone(),
        gas_engine_mesh: gas_engine_mesh.clone(),
        electric_engine_mesh: electric_engine_mesh.clone(),
        servo_mesh: servo_mesh.clone(),
        seat_mesh: seat_mesh.clone(),
        input_mesh: input_mesh.clone(),
        authored_preview_meshes,
        authored_preview_materials,
        invalid_authored_preview_materials,
        drive_xray_mesh: drive_xray_mesh.clone(),
        wire_drag_mesh: wire_drag_mesh.clone(),
        wire_hover_mesh: wire_hover_mesh.clone(),
        cube_preview_mesh: cube_preview_mesh.clone(),
        cylinder_preview_mesh,
        bearing_preview_mesh,
        white_preview_material: white_preview_material.clone(),
        green_preview_material,
        red_preview_material: red_preview_material.clone(),
        block_drag_preview_mesh,
        delete_drag_preview_mesh,
        weld_hover_preview_mesh,
        weld_selection_preview_mesh,
    });

    commands.spawn((
        Name::new("Ground platform"),
        Mesh3d(
            meshes.add(
                Plane3d::default()
                    .mesh()
                    .size(GROUND_HALF_SIZE * 2.0, GROUND_HALF_SIZE * 2.0),
            ),
        ),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.19, 0.22),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));
    commands.spawn((
        Name::new("Construction mesh"),
        Mesh3d(construction_mesh),
        MeshMaterial3d(construction_material.clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        ConstructionVisual,
    ));
    commands.spawn((
        Name::new("Simulation mesh"),
        Mesh3d(simulation_mesh),
        MeshMaterial3d(construction_material),
        NoFrustumCulling,
        Visibility::Hidden,
        SimulationVisual,
    ));
    commands.spawn((
        Name::new("Bearing mesh"),
        Mesh3d(bearing_mesh.clone()),
        MeshMaterial3d(bearing_material),
        NoFrustumCulling,
        Visibility::Hidden,
        BearingVisual,
    ));
    commands.spawn((
        Name::new("Control block mesh"),
        Mesh3d(controller_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Controller.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Controller),
    ));
    commands.spawn((
        Name::new("Gas engine mesh"),
        Mesh3d(gas_engine_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::GasEngine.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::GasEngine),
    ));
    commands.spawn((
        Name::new("Electric engine mesh"),
        Mesh3d(electric_engine_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::ElectricEngine.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::ElectricEngine),
    ));
    commands.spawn((
        Name::new("Servo mesh"),
        Mesh3d(servo_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Servo.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Servo),
    ));
    commands.spawn((
        Name::new("Seat mesh"),
        Mesh3d(seat_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Seat.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Seat),
    ));
    commands.spawn((
        Name::new("Input mesh"),
        Mesh3d(input_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Input.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Input),
    ));
    commands.spawn((
        Name::new("Joint x-ray mesh"),
        Mesh3d(joint_xray_mesh),
        MeshMaterial3d(joint_xray_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        JointXrayVisual,
    ));
    commands.spawn((
        Name::new("Drive x-ray mesh"),
        Mesh3d(drive_xray_mesh),
        MeshMaterial3d(drive_xray_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        DriveXrayVisual,
    ));
    // Kept visible with a degenerate mesh while idle: a hidden mesh has no slab
    // allocation, so writing the first frame of a drag into it would log a
    // use-after-free.
    commands.spawn((
        Name::new("Drive wire drag"),
        Mesh3d(wire_drag_mesh),
        MeshMaterial3d(wire_drag_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Visible,
        WireDragVisual,
    ));
    commands.spawn((
        Name::new("Drive wire hover"),
        Mesh3d(wire_hover_mesh),
        MeshMaterial3d(wire_hover_material),
        Transform::default(),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Visible,
        WireHoverVisual,
    ));
    commands.spawn((
        Name::new("Action preview"),
        Mesh3d(cube_preview_mesh.clone()),
        MeshMaterial3d(white_preview_material.clone()),
        Transform::default(),
        Visibility::Hidden,
        ActionPreview,
    ));
    commands.spawn((
        Name::new("Selection preview"),
        Mesh3d(cube_preview_mesh.clone()),
        MeshMaterial3d(white_preview_material),
        Transform::default(),
        Visibility::Hidden,
        SelectionPreview,
    ));
    commands.spawn((
        Name::new("Delete preview"),
        Mesh3d(cube_preview_mesh),
        MeshMaterial3d(red_preview_material),
        Transform::default(),
        Visibility::Hidden,
        DeletePreview,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let orbit = OrbitCamera::default();
    commands
        .spawn((
            Name::new("Orbital camera"),
            Camera3d::default(),
            Tonemapping::None,
            orbit.transform(),
            orbit,
        ))
        .with_children(|camera| {
            camera.spawn((
                Name::new("Joint x-ray camera"),
                Camera3d::default(),
                // The overlay rides the camera that draws last. This pass loads
                // rather than clears, so an overlay painted before it is drawn
                // over by the joints showing through.
                bevy_mosaic::MosaicCamera,
                Camera {
                    order: 1,
                    clear_color: ClearColorConfig::None,
                    ..default()
                },
                Tonemapping::None,
                RenderLayers::layer(1),
                Transform::default(),
            ));
        });
}

fn help_toggle_requested(keyboard: &ButtonInput<Key>) -> bool {
    keyboard
        .get_just_pressed()
        .any(|key| matches!(key, Key::Character(character) if character == "?"))
}

fn requested_history_action(keyboard: &ButtonInput<KeyCode>) -> Option<HistoryAction> {
    let primary_modifier = keyboard.any_pressed([
        KeyCode::ControlLeft,
        KeyCode::ControlRight,
        KeyCode::SuperLeft,
        KeyCode::SuperRight,
    ]);
    if !primary_modifier || !keyboard.just_pressed(KeyCode::KeyZ) {
        return None;
    }
    Some(
        if keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            HistoryAction::Redo
        } else {
            HistoryAction::Undo
        },
    )
}

fn handle_history_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    simulation: Res<AppSimulation>,
    overlay: Res<ui::UiInput>,
) {
    if overlay.blocks_keyboard() {
        return;
    }
    let Some(action) = requested_history_action(&keyboard) else {
        return;
    };
    apply_history_action(
        action,
        &mut graph.0,
        &mut state,
        &mut history,
        simulation.is_running(),
    );
}

fn apply_history_action(
    action: HistoryAction,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    simulation_running: bool,
) -> bool {
    if simulation_running {
        state.feedback = Some(format!(
            "{} is available in build mode — press Escape first",
            match action {
                HistoryAction::Undo => "Undo",
                HistoryAction::Redo => "Redo",
            }
        ));
        return false;
    }

    let current = EditorSnapshot::capture(graph, state);
    let restored = match action {
        HistoryAction::Undo => history.undo(current),
        HistoryAction::Redo => history.redo(current),
    };
    let Some(restored) = restored else {
        state.feedback = Some(match action {
            HistoryAction::Undo => "Nothing to undo".to_owned(),
            HistoryAction::Redo => "Nothing to redo".to_owned(),
        });
        return false;
    };

    *graph = restored.graph;
    state.placed_bearings = restored.placed_bearings;
    cancel_transient_editor_state(graph, state);
    state.construction_mesh_dirty = true;
    state.feedback = Some(match action {
        HistoryAction::Undo => "Undid construction edit".to_owned(),
        HistoryAction::Redo => "Redid construction edit".to_owned(),
    });
    true
}

fn cancel_transient_editor_state(graph: &mut ConstructionGraph, state: &mut EditorState) {
    if graph.pending().is_some() {
        graph
            .apply(BuildCommand::CancelPending)
            .expect("restored pending editor operation can be cancelled");
    }
    state.block_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    clear_hover(state);
}

fn handle_shortcuts(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut selection: ResMut<SelectedTool>,
    simulation: Res<AppSimulation>,
    overlay: Res<ui::UiInput>,
) {
    if overlay.blocks_keyboard() {
        return;
    }
    for key in [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
        KeyCode::Digit0,
        KeyCode::Minus,
        KeyCode::Equal,
    ] {
        if keyboard.just_pressed(key) {
            selection.0 = shortcut_tool(key).expect("tool shortcut has a mapping");
            break;
        }
    }
    if simulation.is_running() {
        return;
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        if state.block_drag.take().is_some() {
            clear_hover(&mut state);
            state.feedback = Some("Block drag cancelled".to_owned());
        } else if state.delete_drag.take().is_some() {
            clear_hover(&mut state);
            state.feedback = Some("Delete drag cancelled".to_owned());
        } else if graph.0.pending().is_some() {
            let _ = graph.0.apply(BuildCommand::CancelPending);
            state.feedback = Some("Selection cancelled".to_owned());
        } else {
            selection.0 = Tool::Block;
            state.feedback = None;
        }
    }
    if keyboard.just_pressed(KeyCode::KeyQ) {
        if let Some(drag) = state.block_drag.as_mut() {
            drag.plane = drag.plane.cycle();
            state.feedback = Some(format!("Drag plane: {}", drag.plane.label()));
        } else if let Some(drag) = state.delete_drag.as_mut() {
            drag.plane = drag.plane.cycle();
            state.feedback = Some(format!("Delete plane: {}", drag.plane.label()));
        } else if matches!(
            selection.0,
            Tool::Controller
                | Tool::GasEngine
                | Tool::ElectricEngine
                | Tool::Servo
                | Tool::Seat
                | Tool::Input
        ) {
            state.authored_orientation =
                (state.authored_orientation + 1) % AUTHORED_ORIENTATION_COUNT;
            state.feedback = Some(format!(
                "{} orientation: {}/{}",
                selection.0.label(),
                state.authored_orientation + 1,
                AUTHORED_ORIENTATION_COUNT,
            ));
        } else {
            state.feedback = Some(
                "Q cycles machine, Seat, and Input orientations, or changes an active drag plane"
                    .to_owned(),
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BearingDimensionTarget {
    Outer,
    Inner,
}

fn requested_bearing_dimension_adjustment(
    keyboard: &ButtonInput<KeyCode>,
    tool: Tool,
    simulating: bool,
    menu_blocks_input: bool,
) -> Option<(BearingDimensionTarget, i8)> {
    if tool != Tool::Bearing || simulating || menu_blocks_input {
        return None;
    }
    let direction = if keyboard.just_pressed(KeyCode::ArrowLeft) {
        -1
    } else if keyboard.just_pressed(KeyCode::ArrowRight) {
        1
    } else {
        return None;
    };
    let target = if keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        BearingDimensionTarget::Inner
    } else {
        BearingDimensionTarget::Outer
    };
    Some((target, direction))
}

fn adjusted_bearing_dimensions(
    dimensions: BearingDimensions,
    target: BearingDimensionTarget,
    direction: i8,
) -> BearingDimensions {
    let step = f32::from(direction) * BEARING_DIAMETER_STEP;
    let stepped =
        |diameter: f32| ((diameter + step) / BEARING_DIAMETER_STEP).round() * BEARING_DIAMETER_STEP;
    let (outer, inner) = match target {
        BearingDimensionTarget::Outer => {
            let outer = stepped(dimensions.outer_diameter())
                .clamp(MIN_BEARING_OUTER_DIAMETER, MAX_BEARING_OUTER_DIAMETER);
            let inner = dimensions
                .inner_diameter()
                .min(outer - MIN_BEARING_DIAMETER_GAP);
            (outer, inner)
        }
        BearingDimensionTarget::Inner => {
            let inner = stepped(dimensions.inner_diameter())
                .clamp(0.0, dimensions.outer_diameter() - MIN_BEARING_DIAMETER_GAP);
            (dimensions.outer_diameter(), inner)
        }
    };
    BearingDimensions::new(outer, inner)
        .expect("clamped bearing tool settings satisfy core dimensions")
}

fn handle_bearing_dimension_shortcuts(
    keyboard: Res<ButtonInput<KeyCode>>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut settings: ResMut<BearingToolSettings>,
    mut state: ResMut<EditorState>,
) {
    let Some((target, direction)) = requested_bearing_dimension_adjustment(
        &keyboard,
        selection.0,
        simulation.is_running(),
        menu.blocks_keyboard(),
    ) else {
        return;
    };
    settings.dimensions = adjusted_bearing_dimensions(settings.dimensions, target, direction);
    state.feedback = Some(format!(
        "Bearing outer {:.2} m, inner {:.2} m",
        settings.dimensions.outer_diameter(),
        settings.dimensions.inner_diameter()
    ));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CylinderDimensionTarget {
    Outer,
    Inner,
    Length,
    Sweep,
}

fn requested_cylinder_dimension_adjustment(
    keyboard: &ButtonInput<KeyCode>,
    tool: Tool,
    simulating: bool,
    menu_blocks_input: bool,
) -> Option<(CylinderDimensionTarget, i8)> {
    if tool != Tool::Cylinder || simulating || menu_blocks_input {
        return None;
    }
    let shift = keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    if keyboard.just_pressed(KeyCode::ArrowDown) {
        return Some((
            if shift {
                CylinderDimensionTarget::Sweep
            } else {
                CylinderDimensionTarget::Length
            },
            -1,
        ));
    }
    if keyboard.just_pressed(KeyCode::ArrowUp) {
        return Some((
            if shift {
                CylinderDimensionTarget::Sweep
            } else {
                CylinderDimensionTarget::Length
            },
            1,
        ));
    }
    let direction = if keyboard.just_pressed(KeyCode::ArrowLeft) {
        -1
    } else if keyboard.just_pressed(KeyCode::ArrowRight) {
        1
    } else {
        return None;
    };
    let target = if shift {
        CylinderDimensionTarget::Inner
    } else {
        CylinderDimensionTarget::Outer
    };
    Some((target, direction))
}

fn adjusted_cylinder_dimensions(
    dimensions: CylinderDimensions,
    target: CylinderDimensionTarget,
    direction: i8,
) -> CylinderDimensions {
    if target == CylinderDimensionTarget::Sweep {
        let sweep = (i32::from(dimensions.sweep_angle_degrees())
            + i32::from(direction) * i32::from(CYLINDER_SWEEP_STEP_DEGREES))
        .clamp(
            i32::from(MIN_CYLINDER_SWEEP_DEGREES),
            i32::from(MAX_CYLINDER_SWEEP_DEGREES),
        );
        return dimensions
            .with_sweep_angle_degrees(u16::try_from(sweep).expect("clamped sweep fits u16"))
            .expect("clamped cylinder sweep is valid");
    }
    let step_diameter = f32::from(direction) * CYLINDER_DIAMETER_STEP;
    let stepped_diameter = |value: f32| {
        ((value + step_diameter) / CYLINDER_DIAMETER_STEP).round() * CYLINDER_DIAMETER_STEP
    };
    let (outer, inner, length) = match target {
        CylinderDimensionTarget::Outer => {
            let outer = stepped_diameter(dimensions.outer_diameter())
                .clamp(MIN_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_OUTER_DIAMETER);
            (
                outer,
                dimensions
                    .inner_diameter()
                    .min(outer - MIN_CYLINDER_DIAMETER_GAP),
                dimensions.axial_length(),
            )
        }
        CylinderDimensionTarget::Inner => (
            dimensions.outer_diameter(),
            stepped_diameter(dimensions.inner_diameter())
                .clamp(0.0, dimensions.outer_diameter() - MIN_CYLINDER_DIAMETER_GAP),
            dimensions.axial_length(),
        ),
        CylinderDimensionTarget::Length => (
            dimensions.outer_diameter(),
            dimensions.inner_diameter(),
            (dimensions.axial_length() + f32::from(direction) * CYLINDER_LENGTH_STEP)
                .clamp(0.25, 8.0),
        ),
        CylinderDimensionTarget::Sweep => unreachable!("sweep adjustment returned above"),
    };
    CylinderDimensions::new(outer, inner, length)
        .expect("clamped cylinder tool settings satisfy core dimensions")
        .with_sweep_angle_degrees(dimensions.sweep_angle_degrees())
        .expect("existing cylinder sweep remains valid")
}

fn handle_cylinder_dimension_shortcuts(
    keyboard: Res<ButtonInput<KeyCode>>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut settings: ResMut<CylinderToolSettings>,
    mut state: ResMut<EditorState>,
) {
    let Some((target, direction)) = requested_cylinder_dimension_adjustment(
        &keyboard,
        selection.0,
        simulation.is_running(),
        menu.blocks_keyboard(),
    ) else {
        return;
    };
    settings.dimensions = adjusted_cylinder_dimensions(settings.dimensions, target, direction);
    state.feedback = Some(format!(
        "Cylinder outer {:.2} m, inner {:.2} m, length {:.2} m, sweep {}°",
        settings.dimensions.outer_diameter(),
        settings.dimensions.inner_diameter(),
        settings.dimensions.axial_length(),
        settings.dimensions.sweep_angle_degrees(),
    ));
}

fn handle_tool_change(
    selection: Res<SelectedTool>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
) {
    if !selection.is_changed() {
        return;
    }
    if graph.0.pending().is_some() {
        let _ = graph.0.apply(BuildCommand::CancelPending);
    }
    clear_hover(&mut state);
    state.block_drag = None;
    state.delete_drag = None;
    state.wire_drag = None;
    if !selection.0.edits_drives() {
        state.selected_controller = None;
    }
    state.feedback = None;
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn update_hover(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    cylinder_settings: Res<CylinderToolSettings>,
    overlay: Res<ui::UiInput>,
) {
    if simulation.is_running() || overlay.blocks_pointer() {
        clear_hover(&mut state);
        return;
    }
    if camera::camera_input_active(&mouse_buttons, &keyboard) {
        if state.block_drag.take().is_some() {
            state.feedback = Some("Block drag cancelled while moving camera".to_owned());
        }
        if state.delete_drag.take().is_some() {
            state.feedback = Some("Delete drag cancelled while moving camera".to_owned());
        }
        clear_hover(&mut state);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            selection.0,
            cylinder_settings.dimensions,
        );
        return;
    }
    if overlay.blocks_pointer() {
        clear_hover(&mut state);
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        state.pointer_position = None;
        if state.block_drag.is_some() {
            invalidate_block_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.delete_drag.is_some() {
            invalidate_delete_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        clear_hover(&mut state);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            selection.0,
            cylinder_settings.dimensions,
        );
        return;
    };
    let (camera, camera_transform) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        state.pointer_position = None;
        if state.block_drag.is_some() {
            invalidate_block_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.delete_drag.is_some() {
            invalidate_delete_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        clear_hover(&mut state);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            selection.0,
            cylinder_settings.dimensions,
        );
        return;
    };
    state.pointer_position = Some(cursor);
    state.pointer_ray = Some((ray.origin, ray.direction.as_vec3()));
    if state.block_drag.is_some() {
        refresh_block_drag(
            &graph.0,
            &mut state,
            cursor,
            ray.origin,
            ray.direction.as_vec3(),
        );
        return;
    }
    if state.delete_drag.is_some() {
        refresh_delete_drag(&graph.0, &mut state, ray.origin, ray.direction.as_vec3());
        return;
    }
    let ray_direction = ray.direction.as_vec3();
    let construction_hit = if mouse_buttons.pressed(MouseButton::Right) {
        raycast_construction(&graph.0, ray.origin, ray_direction)
    } else {
        match selection.0 {
            Tool::Bearing => raycast_construction_for_annulus(
                &graph.0,
                ray.origin,
                ray_direction,
                bearing_settings.dimensions.inner_diameter(),
                bearing_settings.dimensions.outer_diameter(),
            ),
            Tool::Cylinder => raycast_construction_for_annulus(
                &graph.0,
                ray.origin,
                ray_direction,
                cylinder_settings.dimensions.inner_diameter(),
                cylinder_settings.dimensions.outer_diameter(),
            ),
            Tool::Block
            | Tool::Weld
            | Tool::Hammer
            | Tool::Controller
            | Tool::Connector
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input => raycast_construction(&graph.0, ray.origin, ray_direction),
        }
    };
    // Wiring aims at the whole joint, hole and pin included, so a wire can be
    // dropped on a bearing without having to hit its thin ring.
    let wiring = selection.0 == Tool::Connector;
    let bearing_hit = if wiring {
        raycast_placed_bearing_discs(&graph.0, &state.placed_bearings, ray.origin, ray_direction)
            .or_else(|| {
                raycast_placed_bearings(&graph.0, &state.placed_bearings, ray.origin, ray_direction)
            })
    } else if matches!(selection.0, Tool::Block | Tool::Cylinder)
        || mouse_buttons.pressed(MouseButton::Right)
    {
        raycast_placed_bearings(&graph.0, &state.placed_bearings, ray.origin, ray_direction)
    } else {
        None
    };
    // A joint is usually buried under the parts it carries. The overlay draws it
    // through them, so wiring picks it through them too -- otherwise a bearing
    // is only clickable from the one angle where nothing covers it.
    if let Some((bearing, distance)) = bearing_hit
        && (wiring || construction_hit.is_none_or(|hit| distance <= hit.distance))
    {
        state.hovered = construction_hit;
        state.hovered_bearing = Some(bearing);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            selection.0,
            cylinder_settings.dimensions,
        );
        return;
    }
    let Some(hit) = construction_hit else {
        clear_hover(&mut state);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            selection.0,
            cylinder_settings.dimensions,
        );
        return;
    };
    state.hovered_bearing = None;
    state.hovered = Some(hit);
    refresh_tool_preview_with_cylinder(
        &graph.0,
        &mut state,
        selection.0,
        cylinder_settings.dimensions,
    );
}

fn refresh_block_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, press, plane, last_endpoint) = {
        let drag = state
            .block_drag
            .as_ref()
            .expect("block drag was checked by caller");
        (drag.start, drag.press, drag.plane, drag.last_endpoint)
    };
    let endpoint = if cursor.distance(press.cursor) <= BLOCK_DRAG_DEAD_ZONE_PIXELS {
        start.spec.pose.translation_half_units()
    } else {
        let Some(endpoint) = block_sheet_endpoint_from_rays(
            start.spec,
            plane,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            invalidate_block_drag(state, PlacementError::DragPlaneUnavailable);
            return;
        };
        endpoint
    };
    if last_endpoint == Some((plane, endpoint)) {
        return;
    }
    let result = block_sheet_specs(start.spec, endpoint, plane).and_then(|specs| {
        validate_block_batch(graph, start, &specs)?;
        Ok(specs)
    });
    let drag = state
        .block_drag
        .as_mut()
        .expect("block drag remains active while refreshing");
    drag.last_endpoint = Some((plane, endpoint));
    match result {
        Ok(specs) => {
            drag.specs = specs;
            drag.error = None;
            state.preview_error = None;
        }
        Err(error) => {
            drag.error = Some(error.clone());
            state.preview_error = Some(error);
        }
    }
    state.block_preview_revision = state.block_preview_revision.wrapping_add(1);
}

fn invalidate_block_drag(state: &mut EditorState, error: PlacementError) {
    let drag = state
        .block_drag
        .as_mut()
        .expect("block drag was checked by caller");
    if drag.error.as_ref() != Some(&error) {
        drag.error = Some(error.clone());
        state.preview_error = Some(error);
    }
}

fn refresh_delete_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, plane, last_endpoint) = {
        let drag = state
            .delete_drag
            .as_ref()
            .expect("delete drag was checked by caller");
        (drag.start, drag.plane, drag.last_endpoint)
    };
    let Some(endpoint) = raycast_placement_plane(ray_origin, ray_direction, start, plane) else {
        invalidate_delete_drag(state, PlacementError::DragPlaneUnavailable);
        return;
    };
    if last_endpoint == Some((plane, endpoint)) {
        return;
    }
    let result = delete_sheet_parts(graph, start, endpoint, plane);
    let drag = state
        .delete_drag
        .as_mut()
        .expect("delete drag remains active while refreshing");
    drag.last_endpoint = Some((plane, endpoint));
    match result {
        Ok(parts) => {
            drag.parts = parts;
            drag.error = None;
        }
        Err(error) => drag.error = Some(error),
    }
    state.delete_preview_revision = state.delete_preview_revision.wrapping_add(1);
}

fn invalidate_delete_drag(state: &mut EditorState, error: PlacementError) {
    let drag = state
        .delete_drag
        .as_mut()
        .expect("delete drag was checked by caller");
    if drag.error.as_ref() != Some(&error) {
        drag.error = Some(error);
    }
}

fn delete_sheet_parts(
    graph: &ConstructionGraph,
    start: CuboidSpec,
    endpoint: IVec3,
    plane: PlacementPlane,
) -> Result<Vec<PartId>, PlacementError> {
    let centers = block_sheet_specs(start, endpoint, plane)?
        .into_iter()
        .map(|spec| spec.pose.translation_half_units())
        .collect::<HashSet<_>>();
    Ok(graph
        .parts()
        .filter_map(|(part, spec)| {
            matches!(spec, PartSpec::Cuboid(_))
                .then(|| centers.contains(&spec.pose().translation_half_units()))
                .unwrap_or(false)
                .then_some(part)
        })
        .collect())
}

fn clear_hover(state: &mut EditorState) {
    state.hovered = None;
    state.hovered_bearing = None;
    state.attachment_bearing = None;
    state.preview = None;
    state.cylinder_preview = None;
    state.preview_error = None;
}

#[allow(clippy::too_many_lines)]
fn refresh_tool_preview_with_cylinder(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    tool: Tool,
    cylinder_dimensions: CylinderDimensions,
) {
    state.preview = None;
    state.cylinder_preview = None;
    state.attachment_bearing = None;
    state.preview_error = match (tool, graph.pending()) {
        (Tool::Block, _) => {
            let surface_candidate = state.hovered.and_then(|hit| {
                try_face_geometry_from_ref(hit.face, Some(graph))
                    .is_some()
                    .then(|| candidate_from_hit(graph, hit))
            });
            let direct_bearing = state.hovered_bearing.filter(|&index| {
                state.placed_bearings.get(index).is_some_and(|bearing| {
                    surface_candidate.is_none_or(|candidate| {
                        bearing_overlaps_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            let bearing_index = direct_bearing.or_else(|| {
                surface_candidate.and_then(|candidate| {
                    state.placed_bearings.iter().position(|bearing| {
                        bearing_overlaps_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            state.attachment_bearing = bearing_index;
            if let Some(bearing) =
                bearing_index.and_then(|index| state.placed_bearings.get(index).copied())
            {
                let candidate = surface_candidate.unwrap_or_else(|| {
                    bearing_attachment_candidate(graph, bearing.source, bearing.anchor)
                });
                let error = stage_bearing_attachment(
                    graph,
                    candidate,
                    bearing.source,
                    bearing.anchor,
                    bearing.dimensions,
                )
                .err();
                state.preview = Some(candidate);
                error
            } else {
                surface_candidate.and_then(|candidate| {
                    let error = validate_block_batch(graph, candidate, &[candidate.spec]).err();
                    state.preview = Some(candidate);
                    error
                })
            }
        }
        (Tool::Weld, Some(PendingOperation::Weld(first))) => state
            .hovered
            .and_then(|hit| stage_weld_objects(graph, first.owner, hit.face.owner).err()),
        (Tool::Cylinder, _) => {
            let surface_candidate = state
                .hovered
                .and_then(|hit| cylinder_candidate_from_hit(graph, hit, cylinder_dimensions).ok());
            let direct_bearing = state.hovered_bearing.filter(|&index| {
                state.placed_bearings.get(index).is_some_and(|bearing| {
                    surface_candidate.is_none_or(|candidate| {
                        bearing_overlaps_cylinder_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            let bearing_index = direct_bearing.or_else(|| {
                surface_candidate.and_then(|candidate| {
                    state.placed_bearings.iter().position(|bearing| {
                        bearing_overlaps_cylinder_candidate(
                            graph,
                            bearing.source,
                            bearing.anchor,
                            bearing.dimensions,
                            candidate,
                        )
                    })
                })
            });
            state.attachment_bearing = bearing_index;
            let candidate = if let Some(bearing) =
                bearing_index.and_then(|index| state.placed_bearings.get(index).copied())
            {
                surface_candidate.or_else(|| {
                    let hit = SurfaceHit {
                        distance: 0.0,
                        point: bearing.anchor,
                        face: bearing.source,
                    };
                    cylinder_candidate_from_hit(graph, hit, cylinder_dimensions).ok()
                })
            } else {
                surface_candidate
            };
            candidate.and_then(|candidate| {
                let error = validate_cylinder_candidate(graph, candidate).err();
                state.cylinder_preview = Some(candidate);
                error
            })
        }
        (
            tool @ (Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input),
            _,
        ) => state
            .hovered
            .filter(|hit| try_face_geometry_from_ref(hit.face, Some(graph)).is_some())
            .and_then(|hit| {
                let dimensions = match tool {
                    Tool::Controller => ControllerSpec::GRID_UNITS,
                    Tool::GasEngine => EngineKind::Gas.grid_units(),
                    Tool::ElectricEngine => EngineKind::Electric.grid_units(),
                    Tool::Servo => ServoSpec::GRID_UNITS,
                    Tool::Seat => SeatSpec::GRID_UNITS,
                    Tool::Input => InputSpec::GRID_UNITS,
                    _ => unreachable!(),
                };
                let candidate = oriented_cuboid_candidate_from_hit(
                    graph,
                    hit,
                    dimensions,
                    authored_orientation(state.authored_orientation),
                );
                let error = validate_block_batch(graph, candidate, &[candidate.spec]).err();
                state.preview = Some(candidate);
                error
            }),
        (Tool::Weld | Tool::Hammer | Tool::Connector, _) => None,
        (Tool::Bearing, _) => state.hovered.and_then(|hit| {
            if try_face_geometry_from_ref(hit.face, Some(graph)).is_none() {
                Some(PlacementError::CurvedSurface)
            } else {
                bearing_anchor_from_hit(graph, hit).err()
            }
        }),
    };
}

#[cfg(test)]
fn refresh_tool_preview(graph: &ConstructionGraph, state: &mut EditorState, tool: Tool) {
    refresh_tool_preview_with_cylinder(graph, state, tool, CylinderDimensions::default());
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
// Tool-specific input flows remain readable together.
fn handle_build_actions(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    cylinder_settings: Res<CylinderToolSettings>,
    overlay: Res<ui::UiInput>,
) {
    if simulation.is_running() || overlay.blocks_pointer() {
        return;
    }
    if overlay.blocks_pointer() {
        if mouse.just_released(MouseButton::Left) && state.block_drag.take().is_some() {
            clear_hover(&mut state);
            state.feedback = Some("Block drag cancelled over hotbar".to_owned());
        }
        if mouse.just_released(MouseButton::Right) && state.delete_drag.take().is_some() {
            clear_hover(&mut state);
            state.feedback = Some("Delete drag cancelled over hotbar".to_owned());
        }
        return;
    }
    if camera::camera_input_active(&mouse, &keyboard) {
        return;
    }
    if mouse.just_pressed(MouseButton::Right) && state.block_drag.take().is_some() {
        clear_hover(&mut state);
        state.feedback = Some("Block drag cancelled".to_owned());
        return;
    }
    if selection.0 == Tool::Connector && mouse.just_pressed(MouseButton::Right) {
        if state.wire_drag.take().is_some() {
            state.feedback = Some("Drive wire cancelled".to_owned());
            return;
        }
        state.feedback = Some(disconnect_connector_links(
            &mut graph.0,
            &mut state,
            &mut history,
        ));
        return;
    }
    if mouse.just_pressed(MouseButton::Right) {
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
                    state.delete_drag = Some(DeleteDrag {
                        start: spec,
                        plane,
                        last_endpoint: None,
                        parts: vec![part],
                        error: None,
                    });
                    state.delete_preview_revision = state.delete_preview_revision.wrapping_add(1);
                    state.feedback = Some(format!(
                        "Dragging delete on {} plane — release to remove, Q changes plane",
                        plane.label()
                    ));
                }
                PartSpec::Cylinder(_) => {
                    state.delete_target = Some(DeleteTarget::Part(part));
                    state.feedback = Some("Release right mouse to delete cylinder".to_owned());
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
            }
        }
    }
    if mouse.just_released(MouseButton::Right) {
        if let Some(target) = state.delete_target.take() {
            match target {
                DeleteTarget::PlacedBearing(index) => {
                    if let Some(socket) = state.placed_bearings.get(index).copied() {
                        let previous = EditorSnapshot::capture(&graph.0, &state);
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
                        let mut staged = graph.0.clone();
                        let commands = rigid_links
                            .iter()
                            .copied()
                            .map(BuildCommand::RemoveRigidLink)
                            .chain(attached.iter().copied().map(BuildCommand::RemoveBearing));
                        match staged.apply_batch(commands) {
                            Ok(_) => {
                                graph.0 = staged;
                                state.placed_bearings.remove(index);
                                history.commit(previous);
                                state.feedback = Some(format!(
                                    "Deleted bearing and {} attachment(s)",
                                    attached.len()
                                ));
                                state.construction_mesh_dirty = true;
                                clear_hover(&mut state);
                            }
                            Err(error) => state.feedback = Some(error.to_string()),
                        }
                    }
                }
                DeleteTarget::Part(part) => {
                    let previous = EditorSnapshot::capture(&graph.0, &state);
                    match stage_part_deletion_preserving_bearings(
                        &graph.0,
                        &state.placed_bearings,
                        &[part],
                    ) {
                        Ok((staged, placed_bearings, migrated)) => {
                            graph.0 = staged;
                            state.placed_bearings = placed_bearings;
                            history.commit(previous);
                            state.feedback = Some(if migrated == 0 {
                                "Deleted cylinder and incident connections".to_owned()
                            } else {
                                format!("Deleted cylinder; moved {migrated} bearing(s)")
                            });
                            state.construction_mesh_dirty = true;
                            clear_hover(&mut state);
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
            let previous = EditorSnapshot::capture(&graph.0, &state);
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
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        return;
    }
    if selection.0 == Tool::Connector {
        handle_connector_actions(&mouse, &mut graph.0, &mut state, &mut history);
        return;
    }
    if selection.0 == Tool::Block {
        handle_block_actions(&mouse, &mut graph.0, &mut state, &mut history);
        return;
    }
    if selection.0 == Tool::Cylinder {
        if mouse.just_pressed(MouseButton::Left) {
            let Some(candidate) = state.cylinder_preview else {
                state.feedback = Some("Point at a flat face or compatible bearing".to_owned());
                return;
            };
            let previous = EditorSnapshot::capture(&graph.0, &state);
            let staged = if let Some(index) = state.attachment_bearing {
                let Some(bearing) = state.placed_bearings.get(index).copied() else {
                    state.feedback = Some("Bearing is no longer available".to_owned());
                    return;
                };
                let rigid_targets = bearing_socket_targets(&graph.0, bearing);
                stage_bearing_cylinder(
                    &graph.0,
                    candidate,
                    bearing.source,
                    bearing.anchor,
                    bearing.dimensions,
                    &rigid_targets,
                )
            } else {
                let Some(hit) = state.hovered else {
                    return;
                };
                stage_cylinder_from_source(&graph.0, candidate, hit.face.owner)
            };
            match staged {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!(
                        "Placed cylinder {:.2}/{:.2} m × {:.2} m at {}°",
                        cylinder_settings.dimensions.outer_diameter(),
                        cylinder_settings.dimensions.inner_diameter(),
                        cylinder_settings.dimensions.axial_length(),
                        cylinder_settings.dimensions.sweep_angle_degrees(),
                    ));
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        return;
    }
    if mouse.pressed(MouseButton::Right) || !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    match selection.0 {
        Tool::Block => unreachable!("block actions are handled before this match"),
        Tool::Cylinder => unreachable!("cylinder actions are handled before this match"),
        Tool::Weld => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Select an object".to_owned());
                return;
            };
            if let Some(PendingOperation::Weld(first)) = graph.0.pending() {
                match stage_weld_objects(&graph.0, first.owner, hit.face.owner) {
                    Ok(staged) => {
                        let previous = EditorSnapshot::capture(&graph.0, &state);
                        graph.0 = staged;
                        history.commit(previous);
                        state.feedback = Some("Welded the two objects".to_owned());
                    }
                    Err(error) => state.feedback = Some(error.to_string()),
                }
            } else {
                let selected_face = if try_face_geometry_from_ref(hit.face, Some(&graph.0))
                    .is_some()
                {
                    hit.face
                } else {
                    match hit.face.owner {
                        FaceOwner::Part(part) => {
                            mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::PositiveY)
                        }
                        FaceOwner::Ground => hit.face,
                    }
                };
                match begin_weld(&mut graph.0, selected_face) {
                    Ok(()) => {
                        state.feedback =
                            Some("First object selected; choose a touching object".to_owned());
                    }
                    Err(error) => state.feedback = Some(error.to_string()),
                }
            }
        }
        Tool::Bearing => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Point at a cuboid face".to_owned());
                return;
            };
            match bearing_anchor_from_hit(&graph.0, hit) {
                Ok(anchor) => {
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
                        let previous = EditorSnapshot::capture(&graph.0, &state);
                        state.placed_bearings.push(PlacedBearing {
                            source,
                            anchor,
                            dimensions: bearing_settings.dimensions,
                        });
                        history.commit(previous);
                        state.feedback =
                            Some("Bearing placed — select Block and hover it to attach".to_owned());
                        state.construction_mesh_dirty = true;
                    }
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Hammer => {
            state.feedback = Some("Hammer is available while simulating — press Space".to_owned());
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
                state.feedback = Some("Point at the platform or a face".to_owned());
                return;
            };
            let Some(hit) = state.hovered else {
                return;
            };
            let previous = EditorSnapshot::capture(&graph.0, &state);
            let existing = graph.0.parts().map(|(part, _)| part).collect::<Vec<_>>();
            match stage_controller_from_source(&graph.0, candidate, hit.face.owner) {
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
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        tool @ (Tool::GasEngine | Tool::ElectricEngine) => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at the platform or a face".to_owned());
                return;
            };
            let Some(hit) = state.hovered else {
                return;
            };
            let kind = if tool == Tool::GasEngine {
                EngineKind::Gas
            } else {
                EngineKind::Electric
            };
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match stage_engine_from_source(&graph.0, candidate, hit.face.owner, kind) {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!("Placed {}", tool.label().to_lowercase()));
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Servo | Tool::Seat | Tool::Input => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at the platform or a face".to_owned());
                return;
            };
            let Some(hit) = state.hovered else {
                return;
            };
            let previous = EditorSnapshot::capture(&graph.0, &state);
            let staged = match selection.0 {
                Tool::Servo => stage_servo_from_source(&graph.0, candidate, hit.face.owner),
                Tool::Seat => stage_seat_from_source(&graph.0, candidate, hit.face.owner),
                Tool::Input => stage_input_from_source(&graph.0, candidate, hit.face.owner),
                _ => unreachable!(),
            };
            match staged {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!("Placed {}", selection.0.label()));
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Connector => unreachable!("connector actions are handled before this match"),
    }
    refresh_tool_preview_with_cylinder(
        &graph.0,
        &mut state,
        selection.0,
        cylinder_settings.dimensions,
    );
}

fn bearing_location_occupied(
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
            (same_surface(bearing.source) || same_surface(bearing.target))
                && bearing.shared_anchor.abs_diff_eq(anchor, 1.0e-5)
        })
}

/// Graph bearings backing one placed socket. A socket can carry several rows
/// when more than one rotor group is attached through the same ring.
fn socket_bearings(graph: &ConstructionGraph, socket: PlacedBearing) -> Vec<BearingId> {
    graph
        .bearings()
        .filter_map(|(id, bearing)| bearing_uses_socket(bearing, socket).then_some(id))
        .collect()
}

/// One-line description of a wire's envelope and its first state.
fn drive_summary(spec: &DriveLinkSpec) -> String {
    let actuator = match spec.actuator {
        ActuatorAssignment::Unpowered => "unpowered".to_owned(),
        ActuatorAssignment::Servo => "Servo".to_owned(),
        ActuatorAssignment::Motor {
            electric_percent,
            gas_percent,
        } => format!("motor E{electric_percent}% / G{gas_percent}%"),
    };
    let states = spec.program.len();
    format!(
        "{actuator}, {states} state{}",
        if states == 1 { "" } else { "s" }
    )
}

/// Advances the two-click connector. Returns the feedback line to display.
/// One end of a drive wire while it is being dragged out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WireEnd {
    Controller(PartId),
    Input(PartId),
    Seat(PartId),
    /// Index into [`EditorState::placed_bearings`].
    Bearing(usize),
}

impl WireEnd {
    /// Resolves a supported logical connection from two ends, in either order.
    const fn paired_with(self, other: Self) -> Option<WireConnection> {
        match (self, other) {
            (Self::Controller(controller), Self::Bearing(bearing))
            | (Self::Bearing(bearing), Self::Controller(controller)) => {
                Some(WireConnection::Drive {
                    controller,
                    bearing,
                })
            }
            (Self::Input(input), Self::Seat(seat)) | (Self::Seat(seat), Self::Input(input)) => {
                Some(WireConnection::InputSeat { input, seat })
            }
            (Self::Seat(seat), Self::Controller(controller))
            | (Self::Controller(controller), Self::Seat(seat)) => {
                Some(WireConnection::SeatController { seat, controller })
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WireConnection {
    Drive { controller: PartId, bearing: usize },
    InputSeat { input: PartId, seat: PartId },
    SeatController { seat: PartId, controller: PartId },
}

/// A drive wire the pointer is dragging out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WireDrag {
    from: WireEnd,
    /// The pointer was released back on `from`, so the wire is waiting for a
    /// second click instead of a drag.
    armed: bool,
}

/// What one pointer press or release does to a wire drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WireDragStep {
    /// Nothing in progress and nothing to start.
    Idle,
    /// Nothing wirable under the pointer.
    Miss,
    Begin(WireEnd),
    Connect(WireConnection),
    /// Keep the started end, so a plain click can be finished by a second one.
    Arm,
    Cancel,
}

/// Wiring is symmetric: press either end and release on the other. Pressing and
/// releasing on the same end leaves the wire armed, so click-then-click works
/// as well as drag-and-drop.
fn wire_drag_step(drag: Option<WireDrag>, under: Option<WireEnd>, pressed: bool) -> WireDragStep {
    let Some(drag) = drag else {
        if !pressed {
            return WireDragStep::Idle;
        }
        return under.map_or(WireDragStep::Miss, WireDragStep::Begin);
    };
    if let Some(under) = under
        && let Some(connection) = drag.from.paired_with(under)
    {
        return WireDragStep::Connect(connection);
    }
    if pressed {
        // A press somewhere else restarts the wire there, or drops it.
        return under.map_or(WireDragStep::Cancel, WireDragStep::Begin);
    }
    if under == Some(drag.from) {
        WireDragStep::Arm
    } else {
        WireDragStep::Cancel
    }
}

/// The wire end the pointer is over, if any. A bearing wins over the block
/// behind it, which is what the hover raycast already resolves.
fn wire_end_under_cursor(graph: &ConstructionGraph, state: &EditorState) -> Option<WireEnd> {
    if let Some(index) = state.hovered_bearing {
        return Some(WireEnd::Bearing(index));
    }
    let FaceOwner::Part(part) = state.hovered?.face.owner else {
        return None;
    };
    match graph.part(part) {
        Some(PartSpec::Controller(_)) => Some(WireEnd::Controller(part)),
        Some(PartSpec::Input(_)) => Some(WireEnd::Input(part)),
        Some(PartSpec::Seat(_)) => Some(WireEnd::Seat(part)),
        _ => None,
    }
}

fn wire_end_position(graph: &ConstructionGraph, state: &EditorState, end: WireEnd) -> Option<Vec3> {
    match end {
        WireEnd::Controller(part) | WireEnd::Input(part) | WireEnd::Seat(part) => {
            Some(graph.part(part)?.pose().translation())
        }
        WireEnd::Bearing(index) => Some(state.placed_bearings.get(index)?.anchor),
    }
}

/// Both ends of the wire being dragged: where it started, and either the joint
/// it would land on or the pointer itself.
fn wire_drag_endpoints(graph: &ConstructionGraph, state: &EditorState) -> Option<(Vec3, Vec3)> {
    let drag = state.wire_drag?;
    let from = wire_end_position(graph, state, drag.from)?;
    let target = wire_end_under_cursor(graph, state)
        .filter(|end| drag.from.paired_with(*end).is_some())
        .and_then(|end| wire_end_position(graph, state, end));
    if let Some(target) = target {
        return Some((from, target));
    }
    // No target yet, so the loose end follows the pointer at the depth the
    // wire started from.
    let (origin, direction) = state.pointer_ray?;
    let direction = direction.normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    Some((
        from,
        origin + direction * (from - origin).dot(direction).max(0.1),
    ))
}

/// Press-and-drag wiring for the Connector tool.
fn handle_connector_actions(
    mouse: &ButtonInput<MouseButton>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let pressed = mouse.just_pressed(MouseButton::Left);
    if !pressed && !mouse.just_released(MouseButton::Left) {
        return;
    }
    let under = wire_end_under_cursor(graph, state);
    match wire_drag_step(state.wire_drag, under, pressed) {
        WireDragStep::Idle => {}
        WireDragStep::Miss => {
            state.feedback =
                Some("Drag Controller↔Bearing, Input↔Seat, or Seat↔Controller".to_owned());
        }
        WireDragStep::Begin(from) => {
            state.wire_drag = Some(WireDrag { from, armed: false });
            state.feedback = Some(match from {
                WireEnd::Controller(controller) => {
                    state.selected_controller = Some(controller);
                    "Drag to a bearing or Seat".to_owned()
                }
                WireEnd::Bearing(_) => "Drag to a control block to wire it".to_owned(),
                WireEnd::Input(_) => "Drag to a Seat".to_owned(),
                WireEnd::Seat(_) => "Drag to an Input or Controller".to_owned(),
            });
        }
        WireDragStep::Connect(connection) => {
            state.wire_drag = None;
            state.feedback = Some(match connection {
                WireConnection::Drive {
                    controller,
                    bearing,
                } => connect_drive_wire(graph, state, history, controller, bearing),
                WireConnection::InputSeat { input, seat } => connect_control_link(
                    graph,
                    state,
                    history,
                    BuildCommand::AddInputSeatLink(InputSeatLinkSpec { input, seat }),
                    "Linked Input to Seat",
                ),
                WireConnection::SeatController { seat, controller } => connect_control_link(
                    graph,
                    state,
                    history,
                    BuildCommand::AddSeatControllerLink(SeatControllerLinkSpec {
                        seat,
                        controller,
                    }),
                    "Linked Seat to Controller",
                ),
            });
        }
        WireDragStep::Arm => {
            if let Some(drag) = state.wire_drag.as_mut() {
                drag.armed = true;
            }
            state.feedback = Some("Now click the other end to finish the wire".to_owned());
        }
        WireDragStep::Cancel => {
            state.wire_drag = None;
            state.feedback = Some("Drive wire cancelled".to_owned());
        }
    }
}

fn connect_control_link(
    graph: &mut ConstructionGraph,
    state: &EditorState,
    history: &mut EditorHistory,
    command: BuildCommand,
    success: &str,
) -> String {
    let previous = EditorSnapshot::capture(graph, state);
    match graph.apply(command) {
        Ok(_) => {
            history.commit(previous);
            success.to_owned()
        }
        Err(error) => error.to_string(),
    }
}

/// Wires `controller` to every bearing row of one placed socket, or reverses
/// those rows when the pair is already wired. Returns the feedback line.
fn connect_drive_wire(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    controller: PartId,
    socket_index: usize,
) -> String {
    let Some(socket) = state.placed_bearings.get(socket_index).copied() else {
        return "That bearing is no longer there".to_owned();
    };
    let bearings = socket_bearings(graph, socket);
    if bearings.is_empty() {
        return "Attach a part through this bearing before wiring it".to_owned();
    }

    let existing = graph
        .drive_links()
        .filter(|(_, link)| link.controller == controller && bearings.contains(&link.bearing))
        .map(|(id, link)| (id, *link))
        .collect::<Vec<_>>();
    let previous = EditorSnapshot::capture(graph, state);
    let commands = if existing.is_empty() {
        bearings
            .iter()
            .map(|&bearing| BuildCommand::AddDriveLink(DriveLinkSpec::new(controller, bearing)))
            .collect::<Vec<_>>()
    } else {
        // Clicking a bearing already wired to this block flips its direction.
        existing
            .iter()
            .map(|&(id, _)| BuildCommand::RemoveDriveLink(id))
            .chain(existing.iter().map(|&(_, link)| {
                BuildCommand::AddDriveLink(DriveLinkSpec {
                    reversed: !link.reversed,
                    ..link
                })
            }))
            .collect::<Vec<_>>()
    };
    let reversing = !existing.is_empty();

    let mut staged = graph.clone();
    match staged.apply_batch(commands) {
        Ok(_) => {
            *graph = staged;
            history.commit(previous);
            state.selected_controller = Some(controller);
            state.construction_mesh_dirty = true;
            if reversing {
                "Reversed this bearing's direction".to_owned()
            } else {
                let summary = graph
                    .bearing_drive_link(bearings[0])
                    .map_or_else(String::new, |(_, link)| {
                        format!(" — {}", drive_summary(link))
                    });
                format!(
                    "Wired {} bearing row(s){summary}. Press E to program it",
                    bearings.len()
                )
            }
        }
        Err(error) => error.to_string(),
    }
}

/// Removes every drive wire on the hovered bearing. Returns the feedback line.
fn disconnect_connector_links(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) -> String {
    if graph.pending().is_some() {
        let _ = graph.apply(BuildCommand::CancelPending);
        return "Drive wire cancelled".to_owned();
    }
    if let Some(socket) = state
        .hovered_bearing
        .and_then(|index| state.placed_bearings.get(index).copied())
    {
        return disconnect_drive_wires(graph, state, history, socket);
    }
    let Some(part) = hovered_part(state.hovered) else {
        return "Right click a linked bearing, Input, Seat, or Controller".to_owned();
    };
    let commands = graph
        .input_seat_links()
        .filter_map(|(id, link)| {
            (link.input == part || link.seat == part)
                .then_some(BuildCommand::RemoveInputSeatLink(id))
        })
        .chain(graph.seat_controller_links().filter_map(|(id, link)| {
            (link.seat == part || link.controller == part)
                .then_some(BuildCommand::RemoveSeatControllerLink(id))
        }))
        .collect::<Vec<_>>();
    if commands.is_empty() {
        return "That part has no Input-chain links".to_owned();
    }
    let previous = EditorSnapshot::capture(graph, state);
    match graph.apply_batch(commands) {
        Ok(_) => {
            history.commit(previous);
            state.construction_mesh_dirty = true;
            "Removed Input-chain link(s)".to_owned()
        }
        Err(error) => error.to_string(),
    }
}

fn disconnect_drive_wires(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    socket: PlacedBearing,
) -> String {
    let bearings = socket_bearings(graph, socket);
    let links = graph
        .drive_links()
        .filter_map(|(id, link)| bearings.contains(&link.bearing).then_some(id))
        .collect::<Vec<_>>();
    if links.is_empty() {
        return "That bearing is not wired to a control block".to_owned();
    }
    let previous = EditorSnapshot::capture(graph, state);
    let mut staged = graph.clone();
    match staged.apply_batch(links.iter().copied().map(BuildCommand::RemoveDriveLink)) {
        Ok(_) => {
            *graph = staged;
            history.commit(previous);
            state.construction_mesh_dirty = true;
            "Removed this bearing's drive wire".to_owned()
        }
        Err(error) => error.to_string(),
    }
}

fn bearing_uses_socket(bearing: &mechanic_core::BearingSpec, socket: PlacedBearing) -> bool {
    bearing.source == socket.source
        && bearing.shared_anchor.abs_diff_eq(socket.anchor, 1.0e-5)
        && bearing.dimensions == socket.dimensions
}

fn bearing_socket_targets(graph: &ConstructionGraph, socket: PlacedBearing) -> Vec<PartId> {
    graph
        .bearings()
        .filter_map(|(_, bearing)| {
            if !bearing_uses_socket(bearing, socket) {
                return None;
            }
            match bearing.target.owner {
                FaceOwner::Part(part) => Some(part),
                FaceOwner::Ground => None,
            }
        })
        .collect()
}

fn stage_part_deletion_preserving_bearings(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    deleted_parts: &[PartId],
) -> Result<(ConstructionGraph, Vec<PlacedBearing>, usize), mechanic_core::GraphError> {
    let deleted = deleted_parts.iter().copied().collect::<HashSet<_>>();
    let mut next_bearings = Vec::with_capacity(placed_bearings.len());
    let mut migrations = Vec::<(PlacedBearing, Vec<mechanic_core::FaceRef>)>::new();
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
                match bearing.target.owner {
                    FaceOwner::Part(part) if !deleted.contains(&part) => Some(bearing.target),
                    FaceOwner::Part(_) | FaceOwner::Ground => None,
                }
            })
            .collect::<Vec<_>>();
        if let Some(source) = bearing_support_face_excluding(
            graph,
            socket.source,
            socket.anchor,
            socket.dimensions,
            &deleted,
        ) {
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
    let mut staged = graph.clone();
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
            let axis = face_geometry_from_ref(socket.source, Some(&staged)).normal;
            targets.into_iter().map(move |target| {
                BuildCommand::AddBearing(
                    mechanic_core::BearingSpec::new(socket.source, target, socket.anchor, axis)
                        .with_dimensions(socket.dimensions),
                )
            })
        })
        .collect::<Vec<_>>();
    staged.apply_batch(replacement_bearings)?;

    Ok((staged, next_bearings, migrated_count))
}

fn visible_bearing_count(graph: &ConstructionGraph, placed_bearings: &[PlacedBearing]) -> usize {
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

#[allow(clippy::too_many_lines)] // Click, drag, and bearing attachment share one transaction.
fn handle_block_actions(
    mouse: &ButtonInput<MouseButton>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    if mouse.just_pressed(MouseButton::Left) {
        let Some(candidate) = state.preview else {
            state.feedback = Some("Point at the platform or a cuboid face".to_owned());
            return;
        };
        let (attachment, normal) = if let Some(index) = state.attachment_bearing {
            let Some(bearing) = state.placed_bearings.get(index).copied() else {
                state.feedback = Some("Bearing is no longer available".to_owned());
                return;
            };
            if let Some(error) = stage_bearing_attachment(
                graph,
                candidate,
                bearing.source,
                bearing.anchor,
                bearing.dimensions,
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
            if let Some(error) = validate_block_batch(graph, candidate, &[candidate.spec]).err() {
                state.feedback = Some(error.to_string());
                return;
            }
            let hit = state.hovered.expect("block preview originates from a hit");
            (
                BlockAttachment::AutoWeld {
                    source: hit.face.owner,
                },
                face_geometry_from_ref(hit.face, Some(graph)).normal,
            )
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
            press: PointerSample {
                cursor,
                ray_origin,
                ray_direction,
            },
            plane,
            last_endpoint: None,
            specs: vec![candidate.spec],
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
                "Dragging blocks on {} plane — release to place, Q changes plane",
                plane.label()
            )
        });
        return;
    }

    if !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(drag) = state.block_drag.take() else {
        return;
    };
    if let Some(error) = drag.error {
        state.feedback = Some(error.to_string());
        return;
    }
    let count = drag.specs.len();
    let previous = EditorSnapshot::capture(graph, state);
    let staged = match drag.attachment {
        BlockAttachment::AutoWeld { source } => {
            stage_block_batch_from_source(graph, drag.start, &drag.specs, source)
        }
        BlockAttachment::Bearing {
            source,
            anchor,
            dimensions,
            ..
        } => {
            let socket = PlacedBearing {
                source,
                anchor,
                dimensions,
            };
            let rigid_targets = bearing_socket_targets(graph, socket);
            stage_bearing_block_batch(
                graph,
                drag.start,
                &drag.specs,
                source,
                anchor,
                dimensions,
                &rigid_targets,
            )
        }
    };
    match staged {
        Ok(staged) => {
            let weld_count = staged.weld_count().saturating_sub(graph.weld_count());
            *graph = staged;
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

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn handle_hammer_actions(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    simulation: Res<AppSimulation>,
    graph: Res<EditorGraph>,
    mut hammer: ResMut<HammerInteraction>,
    mut state: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    overlay: Res<ui::UiInput>,
) {
    if !simulation.is_running() {
        hammer.charging = None;
        hammer.pending = None;
        return;
    }
    if simulation.is_paused() || overlay.blocks_pointer() {
        hammer.charging = None;
        hammer.pending = None;
        return;
    }
    if !selection.0.works_in_mode(true) {
        hammer.charging = None;
        if mouse.just_pressed(MouseButton::Left) && !overlay.blocks_pointer() {
            state.feedback = Some(format!(
                "{} is available in build mode — press Escape first",
                selection.0.label()
            ));
        }
        return;
    }
    if overlay.blocks_pointer() && hammer.charging.is_none() {
        return;
    }
    if camera::camera_input_active(&mouse, &keyboard) {
        if hammer.charging.take().is_some() {
            state.feedback = Some("Hammer charge cancelled while moving camera".to_owned());
        }
        return;
    }

    if mouse.just_pressed(MouseButton::Left) {
        hammer.charging = None;
        let hit = window
            .cursor_position()
            .and_then(|cursor| {
                let (camera, camera_transform) = *camera;
                camera.viewport_to_world(camera_transform, cursor).ok()
            })
            .and_then(|ray| {
                let creation = simulation
                    .creation
                    .as_ref()
                    .expect("running simulation has compiled creation");
                raycast_simulation(
                    &graph.0,
                    creation,
                    &simulation.transforms,
                    ray.origin,
                    ray.direction.as_vec3(),
                )
                .map(|hit| (hit, ray.direction.as_vec3()))
            });
        match hit {
            Some((hit, direction)) => {
                let creation = simulation
                    .creation
                    .as_ref()
                    .expect("running simulation has compiled creation");
                if creation.compounds[hit.body_index as usize].is_static {
                    state.feedback = Some("The fixed structure cannot be struck loose".to_owned());
                } else {
                    let transform = simulation.transforms[hit.body_index as usize];
                    let position = Vec3::from_slice(&transform.position[..3]);
                    let rotation = Quat::from_array(transform.rotation);
                    hammer.charging = Some(HammerCharge {
                        body_index: hit.body_index,
                        local_point: rotation.inverse() * (hit.point - position),
                        direction: direction.normalize(),
                        elapsed_seconds: 0.0,
                    });
                    state.feedback =
                        Some("Charging hammer — release left mouse to strike".to_owned());
                }
            }
            None => state.feedback = Some("Point at a moving cuboid to use the hammer".to_owned()),
        }
    }

    if mouse.pressed(MouseButton::Left)
        && let Some(charge) = hammer.charging.as_mut()
    {
        charge.elapsed_seconds =
            (charge.elapsed_seconds + time.delta_secs()).min(HAMMER_CHARGE_SECONDS);
    }

    if !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(charge) = hammer.charging.take() else {
        return;
    };
    let transform = simulation.transforms[charge.body_index as usize];
    let magnitude = hammer_impulse_magnitude(charge.elapsed_seconds);
    let impulse = charge.direction * magnitude;
    let (delivery_ticks, impulse_per_tick) = hammer_delivery(
        simulation
            .creation
            .as_ref()
            .expect("running simulation has compiled creation"),
        transform,
        charge.body_index,
        charge.local_point,
        impulse,
    );
    hammer.pending = Some(HammerImpact {
        body_index: charge.body_index,
        local_point: charge.local_point,
        impulse_per_tick,
        remaining_ticks: delivery_ticks,
    });
    let delivered_magnitude = impulse_per_tick.length() * f32::from(delivery_ticks);
    state.feedback = Some(if delivered_magnitude + f32::EPSILON < magnitude {
        format!("Hammer strike: {delivered_magnitude:.0} N·s (stability limited)")
    } else {
        format!("Hammer strike: {magnitude:.0} N·s")
    });
}

fn apply_pending_hammer_impact(
    simulation: &AppSimulation,
    hammer: &mut HammerInteraction,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) -> Result<(), String> {
    let Some(impact) = hammer.pending.as_mut() else {
        return Ok(());
    };
    let transform = simulation
        .transforms
        .get(impact.body_index as usize)
        .ok_or_else(|| "hammer target no longer exists".to_owned())?;
    let position = Vec3::from_slice(&transform.position[..3]);
    let rotation = Quat::from_array(transform.rotation);
    let world_point = position + rotation * impact.local_point;
    simulation
        .gpu
        .as_ref()
        .expect("running simulation has GPU state")
        .apply_impulse(
            render_device.wgpu_device(),
            render_queue,
            impact.body_index,
            world_point,
            impact.impulse_per_tick,
        )
        .map_err(|error| format!("Hammer strike failed: {error}"))?;
    impact.remaining_ticks -= 1;
    if impact.remaining_ticks == 0 {
        hammer.pending = None;
    }
    Ok(())
}

fn hammer_impulse_magnitude(elapsed_seconds: f32) -> f32 {
    let charge = (elapsed_seconds / HAMMER_CHARGE_SECONDS).clamp(0.0, 1.0);
    HAMMER_MIN_IMPULSE + (HAMMER_MAX_IMPULSE - HAMMER_MIN_IMPULSE) * charge * charge
}

fn hammer_delivery(
    creation: &CompiledCreation,
    transform: GpuTransform,
    body_index: u32,
    local_point: Vec3,
    impulse: Vec3,
) -> (u16, Vec3) {
    let point_travel = hammer_point_travel(creation, transform, body_index, local_point, impulse);
    let maximum_travel = HAMMER_MAX_POINT_TRAVEL_PER_TICK * f32::from(HAMMER_MAX_DELIVERY_TICKS);
    let delivered_impulse = if point_travel > maximum_travel {
        impulse * (maximum_travel / point_travel)
    } else {
        impulse
    };
    let delivered_travel = point_travel.min(maximum_travel);
    let mut ticks = 1_u16;
    while delivered_travel > HAMMER_MAX_POINT_TRAVEL_PER_TICK * f32::from(ticks)
        && ticks < HAMMER_MAX_DELIVERY_TICKS
    {
        ticks += 1;
    }
    (ticks, delivered_impulse / f32::from(ticks))
}

fn hammer_point_travel(
    creation: &CompiledCreation,
    transform: GpuTransform,
    body_index: u32,
    local_point: Vec3,
    impulse: Vec3,
) -> f32 {
    let compound = &creation.compounds[body_index as usize];
    let mass = &compound.mass_properties;
    let rotation = Quat::from_array(transform.rotation);
    let arm = rotation * local_point;
    let local_torque = rotation.inverse() * arm.cross(impulse);
    let angular_delta = rotation * (mass.inverse_inertia * local_torque);
    let linear_delta = impulse * mass.inverse_mass;
    let maximum_radius = creation
        .colliders
        .iter()
        .filter(|collider| collider.compound_index == body_index)
        .map(|collider| collider.local_center.length() + collider.half_extents.length())
        .fold(0.0_f32, f32::max);
    (linear_delta.length() + angular_delta.length() * maximum_radius) * FIXED_DT_SECONDS
}

fn raycast_simulation(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    origin: Vec3,
    direction: Vec3,
) -> Option<SimulationHit> {
    creation
        .part_to_compound
        .iter()
        .filter_map(|&(part, body_index)| {
            let transform = transforms.get(body_index as usize)?;
            let position = Vec3::from_slice(&transform.position[..3]);
            let rotation = Quat::from_array(transform.rotation);
            let initial = &creation.compounds[body_index as usize];
            let spec = *graph.part(part)?;
            let center =
                position + rotation * (spec.pose().translation() - initial.root_translation);
            let part_rotation = rotation * spec.pose().rotation.quaternion();
            let (distance, point) = match spec {
                PartSpec::Cuboid(_)
                | PartSpec::Controller(_)
                | PartSpec::Engine(_)
                | PartSpec::Servo(_)
                | PartSpec::Seat(_)
                | PartSpec::Input(_) => {
                    let hit = raycast_oriented_cuboid(
                        origin,
                        direction,
                        center,
                        part_rotation,
                        spec.size_meters() * 0.5,
                    )?;
                    (hit.distance, hit.point)
                }
                PartSpec::Cylinder(spec) => {
                    let distance = raycast_cylinder_shape(
                        origin,
                        direction,
                        center,
                        part_rotation,
                        spec.dimensions,
                    )?;
                    (distance, origin + direction.normalize() * distance)
                }
            };
            Some(SimulationHit {
                part,
                body_index,
                distance,
                point,
            })
        })
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

fn raycast_cylinder_shape(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    rotation: Quat,
    dimensions: CylinderDimensions,
) -> Option<f32> {
    let direction = direction.normalize();
    let local_origin = rotation.inverse() * (origin - center);
    let local_direction = rotation.inverse() * direction;
    let outer_radius = dimensions.outer_diameter() * 0.5;
    let inner_radius = dimensions.inner_diameter() * 0.5;
    let half_length = dimensions.axial_length() * 0.5;
    let mut nearest = f32::INFINITY;
    if local_direction.y.abs() > f32::EPSILON {
        for y in [-half_length, half_length] {
            let distance = (y - local_origin.y) / local_direction.y;
            let point = local_origin + local_direction * distance;
            let radius_squared = point.x.mul_add(point.x, point.z * point.z);
            if distance >= 0.0
                && radius_squared <= outer_radius * outer_radius
                && radius_squared >= inner_radius * inner_radius
                && point_in_cylinder_slice(point.x, point.z, dimensions)
            {
                nearest = nearest.min(distance);
            }
        }
    }
    for radius in [outer_radius, inner_radius] {
        if radius <= 0.0 {
            continue;
        }
        let a = local_direction
            .x
            .mul_add(local_direction.x, local_direction.z * local_direction.z);
        if a <= f32::EPSILON {
            continue;
        }
        let b = 2.0
            * local_origin
                .x
                .mul_add(local_direction.x, local_origin.z * local_direction.z);
        let c = local_origin
            .x
            .mul_add(local_origin.x, local_origin.z * local_origin.z)
            - radius * radius;
        let discriminant = b.mul_add(b, -4.0 * a * c);
        if discriminant < 0.0 {
            continue;
        }
        let root = discriminant.sqrt();
        for distance in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
            let y = local_origin.y + local_direction.y * distance;
            let point = local_origin + local_direction * distance;
            if distance >= 0.0
                && y.abs() <= half_length
                && point_in_cylinder_slice(point.x, point.z, dimensions)
            {
                nearest = nearest.min(distance);
            }
        }
    }
    if dimensions.sweep_angle_degrees() < 360 {
        let half_sweep = dimensions.sweep_angle_radians() * 0.5;
        for (angle, outward) in [(-half_sweep, -1.0_f32), (half_sweep, 1.0_f32)] {
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let angular = Vec3::new(-angle.sin(), 0.0, angle.cos()) * outward;
            let denominator = local_direction.dot(angular);
            if denominator.abs() <= f32::EPSILON {
                continue;
            }
            let distance = -local_origin.dot(angular) / denominator;
            if distance < 0.0 {
                continue;
            }
            let point = local_origin + local_direction * distance;
            let radius = point.dot(radial);
            if point.y.abs() <= half_length && radius >= inner_radius && radius <= outer_radius {
                nearest = nearest.min(distance);
            }
        }
    }
    nearest.is_finite().then_some(nearest)
}

fn point_in_cylinder_slice(x: f32, z: f32, dimensions: CylinderDimensions) -> bool {
    dimensions.sweep_angle_degrees() == 360
        || z.atan2(x).abs() <= dimensions.sweep_angle_radians() * 0.5 + 1.0e-5
}

fn hovered_part(hit: Option<SurfaceHit>) -> Option<PartId> {
    match hit?.face.owner {
        FaceOwner::Part(part) => Some(part),
        FaceOwner::Ground => None,
    }
}

fn raycast_placed_bearings(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    bearings
        .iter()
        .enumerate()
        .filter_map(|(index, bearing)| {
            let normal = face_geometry_from_ref(bearing.source, Some(graph)).normal;
            let distance = raycast_bearing_annulus(
                origin,
                direction,
                bearing.anchor,
                normal,
                bearing.dimensions,
            )?;
            Some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
}

/// Bearing pick used for drive wiring. Unlike [`raycast_placed_bearings`] this
/// accepts the whole disc, including the hole and whatever is threaded through
/// it, because a wire is aimed at a joint rather than at its ring.
fn raycast_placed_bearing_discs(
    graph: &ConstructionGraph,
    bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(usize, f32)> {
    if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < f32::EPSILON {
        return None;
    }
    let direction = direction.normalize();
    bearings
        .iter()
        .enumerate()
        .filter_map(|(index, bearing)| {
            let axis = face_geometry_from_ref(bearing.source, Some(graph))
                .normal
                .normalize();
            let slope = direction.dot(axis);
            if slope.abs() < 1.0e-6 {
                return None;
            }
            let distance = (bearing.anchor - origin).dot(axis) / slope;
            if distance <= 0.0 {
                return None;
            }
            let radius = (origin + direction * distance - bearing.anchor).length();
            (radius <= bearing.dimensions.outer_diameter() * 0.5).then_some((index, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
}

fn raycast_bearing_annulus(
    origin: Vec3,
    direction: Vec3,
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
) -> Option<f32> {
    let axis = axis.normalize();
    let direction = direction.normalize();
    let offset = origin - anchor;
    let axial_origin = offset.dot(axis);
    let axial_direction = direction.dot(axis);
    let radial_origin = offset - axis * axial_origin;
    let radial_direction = direction - axis * axial_direction;
    let half_depth = BEARING_DEPTH * 0.5;
    let outer_radius = dimensions.outer_diameter() * 0.5;
    let inner_radius = dimensions.inner_diameter() * 0.5;
    let mut nearest = f32::INFINITY;

    let radial_a = radial_direction.length_squared();
    if radial_a > f32::EPSILON {
        for radius in [outer_radius, inner_radius] {
            if radius <= 0.0 {
                continue;
            }
            let radial_b = 2.0 * radial_origin.dot(radial_direction);
            let radial_c = radial_origin.length_squared() - radius * radius;
            let discriminant = radial_b.mul_add(radial_b, -4.0 * radial_a * radial_c);
            if discriminant < 0.0 {
                continue;
            }
            let root = discriminant.sqrt();
            for distance in [
                (-radial_b - root) / (2.0 * radial_a),
                (-radial_b + root) / (2.0 * radial_a),
            ] {
                let depth = axial_origin + axial_direction * distance;
                if distance >= 0.0 && depth.abs() <= half_depth + 1.0e-6 {
                    nearest = nearest.min(distance);
                }
            }
        }
    }

    if axial_direction.abs() > f32::EPSILON {
        for depth in [-half_depth, half_depth] {
            let distance = (depth - axial_origin) / axial_direction;
            if distance < 0.0 {
                continue;
            }
            let radial = radial_origin + radial_direction * distance;
            let radius_squared = radial.length_squared();
            if radius_squared <= outer_radius * outer_radius + f32::EPSILON
                && radius_squared >= inner_radius * inner_radius
            {
                nearest = nearest.min(distance);
            }
        }
    }

    nearest.is_finite().then_some(nearest)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sync_visual_meshes(
    graph: Res<EditorGraph>,
    sequencer: Res<DriveSequencer>,
    selection: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    mut state: ResMut<EditorState>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visibility: Single<
        &mut Visibility,
        (
            With<ConstructionVisual>,
            Without<BearingVisual>,
            Without<SimulationVisual>,
        ),
    >,
    mut bearing_visibility: Single<
        &mut Visibility,
        (
            With<BearingVisual>,
            Without<ConstructionVisual>,
            Without<SimulationVisual>,
        ),
    >,
    mut simulation_visibility: Single<
        &mut Visibility,
        (
            With<SimulationVisual>,
            Without<ConstructionVisual>,
            Without<BearingVisual>,
        ),
    >,
    mut authored_visuals: Query<
        (&AuthoredPartVisual, &mut Visibility),
        (
            Without<ConstructionVisual>,
            Without<BearingVisual>,
            Without<SimulationVisual>,
        ),
    >,
) {
    if !state.construction_mesh_dirty {
        return;
    }
    if graph.0.part_count() == 0 {
        **construction_visibility = Visibility::Hidden;
    } else {
        if let Some(mut mesh) = meshes.get_mut(&visuals.construction_mesh) {
            *mesh = renderable_mesh(combined_construction_mesh(&graph.0));
        }
        **construction_visibility = Visibility::Visible;
    }
    **simulation_visibility = Visibility::Hidden;
    for appearance in AuthoredPart::ALL {
        let visible = graph.0.parts().any(|(_, spec)| appearance.matches(*spec));
        if visible && let Some(mut mesh) = meshes.get_mut(visuals.authored_mesh(appearance)) {
            *mesh = combined_authored_construction_mesh(&graph.0, appearance);
        }
        for (visual, mut visibility) in &mut authored_visuals {
            if visual.0 == appearance {
                *visibility = if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
    if graph.0.bearing_count() == 0 && state.placed_bearings.is_empty() {
        **bearing_visibility = Visibility::Hidden;
    } else {
        let rings = combined_bearing_mesh(&graph.0, &state.placed_bearings);
        if joint_xray_is_visible(
            selection.0,
            simulation.is_running(),
            visible_bearing_count(&graph.0, &state.placed_bearings),
        ) && let Some(mut mesh) = meshes.get_mut(&visuals.joint_xray_mesh)
        {
            *mesh = rings.clone();
        }
        if let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
            *mesh = rings;
        }
        **bearing_visibility = Visibility::Visible;
    }
    if drive_xray_is_visible(selection.0, control_link_count(&graph.0))
        && let Some(mut mesh) = meshes.get_mut(&visuals.drive_xray_mesh)
    {
        *mesh = combined_drive_xray_mesh(&graph.0, &state.placed_bearings, &sequencer);
    }
    state.construction_mesh_dirty = false;
}

/// The bearing x-ray is also shown while wiring, so a drive wire can be traced
/// back through the construction to the block that owns it.
fn joint_xray_is_visible(tool: Tool, simulating: bool, bearing_count: usize) -> bool {
    matches!(tool, Tool::Controller | Tool::Connector) && !simulating && bearing_count > 0
}

/// The drive overlay additionally stays up while simulating, so the joint a key
/// is driving can be seen moving. Its meshes are rebuilt from each published
/// snapshot, so the arcs and wires track the running bodies.
fn drive_xray_is_visible(tool: Tool, driven_count: usize) -> bool {
    matches!(tool, Tool::Controller | Tool::Connector) && driven_count > 0
}

/// Number and world position of every driven joint.
///
/// Numbering comes from [`control_panel::panel_rows`], the same grouping the
/// panel lists, so `Joint 3` in the table is the joint wearing a floating `3`.
/// Two wires on one physical joint share a row, and so share one label.
fn joint_number_labels(
    graph: &ConstructionGraph,
    anchor_of: impl Fn(&mechanic_core::BearingSpec) -> Option<Vec3>,
) -> Vec<(usize, Vec3)> {
    let mut labels = Vec::new();
    for (controller, _) in graph.parts().filter(|(id, _)| graph.is_controller(*id)) {
        for (index, row) in control_panel::panel_rows(graph, controller)
            .iter()
            .enumerate()
        {
            let Some(bearing) = graph
                .drive_link(row.primary)
                .and_then(|link| graph.bearing(link.bearing))
            else {
                continue;
            };
            if let Some(anchor) = anchor_of(bearing) {
                labels.push((index + 1, anchor));
            }
        }
    }
    labels
}

fn driven_bearing_count(graph: &ConstructionGraph) -> usize {
    graph
        .drive_links()
        .filter(|(_, link)| graph.is_controller(link.controller))
        .count()
}

fn control_link_count(graph: &ConstructionGraph) -> usize {
    driven_bearing_count(graph)
        + graph.input_seat_links().count()
        + graph.seat_controller_links().count()
}

#[allow(clippy::type_complexity)]
fn update_joint_xray(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    selection: Res<SelectedTool>,
    mut drive_visibility: Single<
        &mut Visibility,
        (With<DriveXrayVisual>, Without<JointXrayVisual>),
    >,
    mut visibility: Single<&mut Visibility, (With<JointXrayVisual>, Without<DriveXrayVisual>)>,
) {
    let drive_visible = drive_xray_is_visible(selection.0, control_link_count(&graph.0));
    let joint_visible = joint_xray_is_visible(
        selection.0,
        simulation.is_running(),
        visible_bearing_count(&graph.0, &state.placed_bearings),
    );
    // An overlay's mesh is left alone while it is hidden, so it has to be
    // rebuilt on the frame it comes back. This system runs ahead of both mesh
    // builders, so the request is served without a stale frame.
    if (drive_visible && **drive_visibility == Visibility::Hidden)
        || (joint_visible && **visibility == Visibility::Hidden)
    {
        state.construction_mesh_dirty = true;
        if simulation.is_running() {
            simulation.render_dirty = true;
        }
    }
    **drive_visibility = if drive_visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    **visibility = if joint_visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
}

#[allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn update_previews(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    selected_tool: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    cylinder_settings: Res<CylinderToolSettings>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rendered_block_revision: Local<u64>,
    mut rendered_delete_revision: Local<u64>,
    mut rendered_bearing_dimensions: Local<BearingDimensions>,
    mut rendered_weld_hover: Local<Option<PartId>>,
    mut rendered_weld_selection: Local<Option<PartId>>,
    mut action: Single<
        (
            &mut Mesh3d,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        (
            With<ActionPreview>,
            Without<SelectionPreview>,
            Without<DeletePreview>,
        ),
    >,
    mut selection: Single<
        (
            &mut Mesh3d,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        (
            With<SelectionPreview>,
            Without<ActionPreview>,
            Without<DeletePreview>,
        ),
    >,
    mut delete: Single<
        (
            &mut Mesh3d,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        (
            With<DeletePreview>,
            Without<ActionPreview>,
            Without<SelectionPreview>,
        ),
    >,
) {
    hide_preview(&mut action.2);
    hide_preview(&mut selection.2);
    hide_preview(&mut delete.2);

    if selected_tool.0 == Tool::Weld {
        if hovered_part(state.hovered).is_none() {
            *rendered_weld_hover = None;
        }
        if !matches!(graph.0.pending(), Some(PendingOperation::Weld(_))) {
            *rendered_weld_selection = None;
        }
    } else {
        *rendered_weld_hover = None;
        *rendered_weld_selection = None;
    }

    if simulation.is_running() {
        return;
    }

    if let Some(drag) = state.delete_drag.as_ref() {
        if *rendered_delete_revision != state.delete_preview_revision {
            let specs = drag
                .parts
                .iter()
                .filter_map(|&part| graph.0.part(part).copied())
                .collect::<Vec<_>>();
            if let Some(mut mesh) = meshes.get_mut(&visuals.delete_drag_preview_mesh) {
                *mesh = combined_parts_mesh_scaled(&specs, 1.0);
            }
            *rendered_delete_revision = state.delete_preview_revision;
        }
        delete.0.0 = visuals.delete_drag_preview_mesh.clone();
        *delete.1 = Transform::default();
        delete.3.0 = visuals.red_preview_material.clone();
        *delete.2 = Visibility::Visible;
        return;
    }

    if let Some(target) = state.delete_target {
        match target {
            DeleteTarget::PlacedBearing(index) => {
                if let Some(bearing) = state.placed_bearings.get(index) {
                    let normal = face_geometry_from_ref(bearing.source, Some(&graph.0)).normal;
                    update_bearing_preview_mesh(
                        &mut meshes,
                        &visuals.bearing_preview_mesh,
                        &mut rendered_bearing_dimensions,
                        bearing.dimensions,
                    );
                    show_bearing_preview(
                        &mut delete,
                        &visuals.bearing_preview_mesh,
                        &visuals.red_preview_material,
                        bearing.anchor,
                        normal,
                    );
                }
            }
            DeleteTarget::Part(part) => {
                if let Some(spec) = graph.0.part(part).copied() {
                    if let Some(mut mesh) = meshes.get_mut(&visuals.delete_drag_preview_mesh) {
                        *mesh = combined_parts_mesh_scaled(&[spec], 1.015);
                    }
                    delete.0.0 = visuals.delete_drag_preview_mesh.clone();
                    *delete.1 = Transform::default();
                    delete.3.0 = visuals.red_preview_material.clone();
                    *delete.2 = Visibility::Visible;
                }
            }
        }
        return;
    }

    let bearing_attachment_highlighted = bearing_attachment_is_highlighted(
        selected_tool.0,
        state.attachment_bearing,
        state.preview_error.as_ref(),
    );
    let action_material = if state.preview_error.is_some() {
        &visuals.red_preview_material
    } else if bearing_attachment_highlighted {
        &visuals.green_preview_material
    } else {
        &visuals.white_preview_material
    };
    if let Some(bearing) = state
        .attachment_bearing
        .and_then(|index| state.placed_bearings.get(index))
    {
        let normal = face_geometry_from_ref(bearing.source, Some(&graph.0)).normal;
        update_bearing_preview_mesh(
            &mut meshes,
            &visuals.bearing_preview_mesh,
            &mut rendered_bearing_dimensions,
            bearing.dimensions,
        );
        show_bearing_preview(
            &mut selection,
            &visuals.bearing_preview_mesh,
            if bearing_attachment_highlighted {
                &visuals.green_preview_material
            } else {
                &visuals.red_preview_material
            },
            bearing.anchor,
            normal,
        );
        selection.1.scale = Vec3::splat(1.12);
    }
    match (selected_tool.0, graph.0.pending()) {
        (Tool::Block, _) => {
            if let Some(drag) = state.block_drag.as_ref() {
                if *rendered_block_revision != state.block_preview_revision {
                    if let Some(mut mesh) = meshes.get_mut(&visuals.block_drag_preview_mesh) {
                        *mesh = block_sheet_preview_mesh(&drag.specs);
                    }
                    *rendered_block_revision = state.block_preview_revision;
                }
                action.0.0 = visuals.block_drag_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            } else if let Some(candidate) = state.preview {
                show_cuboid_preview(
                    &mut action,
                    &visuals.cube_preview_mesh,
                    action_material,
                    candidate.spec,
                    0.992,
                );
            }
        }
        (Tool::Cylinder, _) => {
            if let Some(candidate) = state.cylinder_preview {
                if let Some(mut mesh) = meshes.get_mut(&visuals.cylinder_preview_mesh) {
                    *mesh = single_cylinder_mesh(cylinder_settings.dimensions);
                }
                show_cylinder_preview(
                    &mut action,
                    &visuals.cylinder_preview_mesh,
                    action_material,
                    candidate.spec,
                );
            }
        }
        (Tool::Weld, pending) => {
            if let Some(part) = hovered_part(state.hovered) {
                if *rendered_weld_hover != Some(part) || graph.is_changed() {
                    let specs = rigid_body_parts(&graph.0, part)
                        .into_iter()
                        .filter_map(|member| graph.0.part(member).copied())
                        .collect::<Vec<_>>();
                    if let Some(mut mesh) = meshes.get_mut(&visuals.weld_hover_preview_mesh) {
                        *mesh = combined_parts_mesh_scaled(&specs, 1.018);
                    }
                    *rendered_weld_hover = Some(part);
                }
                action.0.0 = visuals.weld_hover_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            }
            if let Some(PendingOperation::Weld(first)) = pending
                && let FaceOwner::Part(part) = first.owner
            {
                if *rendered_weld_selection != Some(part) || graph.is_changed() {
                    let specs = rigid_body_parts(&graph.0, part)
                        .into_iter()
                        .filter_map(|member| graph.0.part(member).copied())
                        .collect::<Vec<_>>();
                    if let Some(mut mesh) = meshes.get_mut(&visuals.weld_selection_preview_mesh) {
                        *mesh = combined_parts_mesh_scaled(&specs, 1.028);
                    }
                    *rendered_weld_selection = Some(part);
                }
                selection.0.0 = visuals.weld_selection_preview_mesh.clone();
                *selection.1 = Transform::default();
                selection.3.0 = visuals.white_preview_material.clone();
                *selection.2 = Visibility::Visible;
            }
        }
        (Tool::Bearing, _) => {
            if let Some(hit) = state.hovered
                && let Some(face) = try_face_geometry_from_ref(hit.face, Some(&graph.0))
            {
                let anchor = bearing_anchor_from_hit(&graph.0, hit).unwrap_or(hit.point);
                update_bearing_preview_mesh(
                    &mut meshes,
                    &visuals.bearing_preview_mesh,
                    &mut rendered_bearing_dimensions,
                    bearing_settings.dimensions,
                );
                show_bearing_preview(
                    &mut action,
                    &visuals.bearing_preview_mesh,
                    action_material,
                    anchor,
                    face.normal,
                );
            }
        }
        (
            Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input,
            _,
        ) => {
            if let (Some(candidate), Some(appearance)) =
                (state.preview, AuthoredPart::from_tool(selected_tool.0))
            {
                show_cuboid_preview(
                    &mut action,
                    visuals.authored_preview_mesh(appearance),
                    visuals.authored_preview_material(appearance, state.preview_error.is_some()),
                    candidate.spec,
                    0.992,
                );
            }
        }
        (Tool::Hammer | Tool::Connector, _) => {}
    }
}

fn update_bearing_preview_mesh(
    meshes: &mut Assets<Mesh>,
    mesh_handle: &Handle<Mesh>,
    rendered_dimensions: &mut BearingDimensions,
    dimensions: BearingDimensions,
) {
    if !bearing_preview_dimensions_changed(rendered_dimensions, dimensions) {
        return;
    }
    if let Some(mut mesh) = meshes.get_mut(mesh_handle) {
        *mesh = single_bearing_mesh(dimensions);
    }
}

fn bearing_preview_dimensions_changed(
    rendered_dimensions: &mut BearingDimensions,
    dimensions: BearingDimensions,
) -> bool {
    if *rendered_dimensions == dimensions {
        false
    } else {
        *rendered_dimensions = dimensions;
        true
    }
}

fn bearing_attachment_is_highlighted(
    tool: Tool,
    attachment_bearing: Option<usize>,
    preview_error: Option<&PlacementError>,
) -> bool {
    matches!(tool, Tool::Block | Tool::Cylinder)
        && attachment_bearing.is_some()
        && preview_error.is_none()
}

type PreviewItem<'a> = (
    Mut<'a, Mesh3d>,
    Mut<'a, Transform>,
    Mut<'a, Visibility>,
    Mut<'a, MeshMaterial3d<StandardMaterial>>,
);

fn hide_preview(visibility: &mut Visibility) {
    *visibility = Visibility::Hidden;
}

fn show_cuboid_preview(
    preview: &mut PreviewItem<'_>,
    mesh_handle: &Handle<Mesh>,
    material_handle: &Handle<StandardMaterial>,
    spec: CuboidSpec,
    scale_factor: f32,
) {
    preview.0.0 = mesh_handle.clone();
    *preview.1 = Transform::from_translation(spec.pose.translation())
        .with_rotation(spec.pose.rotation.quaternion())
        .with_scale(spec.size_meters() * scale_factor);
    preview.3.0 = material_handle.clone();
    *preview.2 = Visibility::Visible;
}

fn show_bearing_preview(
    preview: &mut PreviewItem<'_>,
    mesh_handle: &Handle<Mesh>,
    material_handle: &Handle<StandardMaterial>,
    anchor: Vec3,
    normal: Vec3,
) {
    preview.0.0 = mesh_handle.clone();
    *preview.1 =
        Transform::from_translation(anchor).with_rotation(Quat::from_rotation_arc(Vec3::Y, normal));
    preview.3.0 = material_handle.clone();
    *preview.2 = Visibility::Visible;
}

fn show_cylinder_preview(
    preview: &mut PreviewItem<'_>,
    mesh_handle: &Handle<Mesh>,
    material_handle: &Handle<StandardMaterial>,
    spec: mechanic_core::CylinderSpec,
) {
    preview.0.0 = mesh_handle.clone();
    *preview.1 = Transform::from_translation(spec.pose.translation())
        .with_rotation(spec.pose.rotation.quaternion())
        .with_scale(Vec3::splat(0.992));
    preview.3.0 = material_handle.clone();
    *preview.2 = Visibility::Visible;
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
// Tool-specific guidance is kept together with its HUD layout.
fn tool_status_line(
    tool: Tool,
    bearing_dimensions: BearingDimensions,
    cylinder_dimensions: CylinderDimensions,
    selected_wires: Option<usize>,
) -> String {
    match tool {
        Tool::Connector => format!(
            "Tool: Connector    Drag a block to a bearing, or a bearing to a block    {}    Right click a wired bearing removes it",
            selected_wires.map_or_else(
                || "No block selected".to_owned(),
                |wires| format!(
                    "Selected block: {wires} bearing{} wired",
                    if wires == 1 { "" } else { "s" }
                )
            ),
        ),
        Tool::Controller => format!(
            "Tool: {}    Q Rotate 90°    {}    E opens its program",
            tool.label(),
            selected_wires.map_or_else(
                || "No block selected — click one to select it".to_owned(),
                |wires| format!(
                    "Selected block: {wires} bearing{} wired",
                    if wires == 1 { "" } else { "s" }
                )
            ),
        ),
        Tool::Block => format!("Tool: Block    Block size: {BLOCK_SIZE_METERS:.2} m"),
        Tool::Bearing => format!(
            "Tool: Bearing    Outer: {:.2} m ←/→  Inner: {:.2} m Shift+←/→",
            bearing_dimensions.outer_diameter(),
            bearing_dimensions.inner_diameter(),
        ),
        Tool::Cylinder => format!(
            "Tool: Cylinder    Outer: {:.2} m ←/→  Inner: {:.2} m Shift+←/→  Length: {:.2} m ↓/↑  Sweep: {}° Shift+↓/↑",
            cylinder_dimensions.outer_diameter(),
            cylinder_dimensions.inner_diameter(),
            cylinder_dimensions.axial_length(),
            cylinder_dimensions.sweep_angle_degrees(),
        ),
        Tool::GasEngine | Tool::ElectricEngine | Tool::Servo | Tool::Seat | Tool::Input => {
            format!("Tool: {}    Q Rotate 90°", tool.label())
        }
        _ => format!("Tool: {}", tool.label()),
    }
}

const CUBE_POSITIONS: [[f32; 3]; 24] = [
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
];
const CUBE_NORMALS: [[f32; 3]; 24] = [
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
];
const CUBE_INDICES: [u32; 36] = [
    0, 3, 1, 1, 3, 2, 4, 5, 7, 5, 6, 7, 8, 11, 9, 9, 11, 10, 12, 13, 15, 13, 14, 15, 16, 19, 17,
    17, 19, 18, 20, 21, 23, 21, 22, 23,
];

// Authored machine GLBs use +X, -X, +Y, -Y, +Z, -Z face order. Keeping that
// template here lets their UV atlases stay exact while placed parts remain in
// the app's batched meshes rather than becoming one entity per part.
const AUTHORED_CUBE_POSITIONS: [[f32; 3]; 24] = [
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
];
const AUTHORED_CUBE_NORMALS: [[f32; 3]; 24] = [
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
];
const AUTHORED_CUBE_TANGENTS: [[f32; 4]; 24] = [
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0, -1.0],
];
const AUTHORED_CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18,
    16, 18, 19, 20, 21, 22, 20, 22, 23,
];

const CONTROLLER_UVS: [[f32; 2]; 24] = [
    [0.0, 0.5],
    [0.0, 0.0],
    [0.25, 0.0],
    [0.25, 0.5],
    [0.25, 0.5],
    [0.25, 0.0],
    [0.5, 0.0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.25],
    [1.0, 0.25],
    [1.0, 0.5],
    [0.5, 0.25],
    [0.5, 0.0],
    [1.0, 0.0],
    [1.0, 0.25],
    [0.0, 1.0],
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 1.0],
    [0.5, 1.0],
    [0.5, 0.5],
    [1.0, 0.5],
    [1.0, 1.0],
];
const GAS_ENGINE_UVS: [[f32; 2]; 24] = [
    [0.0, 1.0],
    [0.0, 0.666_667],
    [0.5, 0.666_667],
    [0.5, 1.0],
    [0.5, 1.0],
    [0.5, 0.666_667],
    [1.0, 0.666_667],
    [1.0, 1.0],
    [0.0, 0.666_667],
    [0.0, 0.166_667],
    [0.333_333, 0.166_667],
    [0.333_333, 0.666_667],
    [0.333_333, 0.666_667],
    [0.333_333, 0.166_667],
    [0.666_667, 0.166_667],
    [0.666_667, 0.666_667],
    [0.666_667, 0.666_667],
    [0.666_667, 0.333_333],
    [1.0, 0.333_333],
    [1.0, 0.666_667],
    [0.666_667, 0.333_333],
    [0.666_667, 0.0],
    [1.0, 0.0],
    [1.0, 0.333_333],
];
const ELECTRIC_ENGINE_UVS: [[f32; 2]; 24] = [
    [0.666_667, 1.0],
    [0.666_667, 0.5],
    [1.0, 0.5],
    [1.0, 1.0],
    [0.0, 0.5],
    [0.0, 0.0],
    [0.333_333, 0.0],
    [0.333_333, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 0.0],
    [0.666_667, 0.0],
    [0.666_667, 0.5],
    [0.666_667, 0.5],
    [0.666_667, 0.0],
    [1.0, 0.0],
    [1.0, 0.5],
    [0.0, 1.0],
    [0.0, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 1.0],
    [0.333_333, 1.0],
    [0.333_333, 0.5],
    [0.666_667, 0.5],
    [0.666_667, 1.0],
];
const SERVO_UVS: [[f32; 2]; 24] = [
    [0.666_667, 0.5],
    [0.666_667, 1.0],
    [1.0, 1.0],
    [1.0, 0.5],
    [0.0, 0.0],
    [0.0, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 0.0],
    [0.333_333, 0.0],
    [0.333_333, 0.5],
    [0.666_667, 0.5],
    [0.666_667, 0.0],
    [0.666_667, 0.0],
    [0.666_667, 0.5],
    [1.0, 0.5],
    [1.0, 0.0],
    [0.0, 0.5],
    [0.0, 1.0],
    [0.333_333, 1.0],
    [0.333_333, 0.5],
    [0.333_333, 0.5],
    [0.333_333, 1.0],
    [0.666_667, 1.0],
    [0.666_667, 0.5],
];
const SEAT_UVS: [[f32; 2]; 24] = [
    [0.5, 0.25],
    [0.5, 0.5],
    [1.0, 0.5],
    [1.0, 0.25],
    [0.5, 0.0],
    [0.5, 0.25],
    [1.0, 0.25],
    [1.0, 0.0],
    [0.0, 0.5],
    [0.0, 1.0],
    [0.5, 1.0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 1.0],
    [1.0, 1.0],
    [1.0, 0.5],
    [0.0, 0.25],
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 0.25],
    [0.0, 0.0],
    [0.0, 0.25],
    [0.5, 0.25],
    [0.5, 0.0],
];
const INPUT_UVS: [[f32; 2]; 24] = [
    [0.0, 0.25],
    [0.0, 0.5],
    [0.25, 0.5],
    [0.25, 0.25],
    [0.25, 0.25],
    [0.25, 0.5],
    [0.5, 0.5],
    [0.5, 0.25],
    [0.0, 0.5],
    [0.0, 0.75],
    [0.5, 0.75],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.75],
    [1.0, 0.75],
    [1.0, 0.5],
    [0.0, 0.75],
    [0.0, 1.0],
    [0.5, 1.0],
    [0.5, 0.75],
    [0.5, 0.75],
    [0.5, 1.0],
    [1.0, 1.0],
    [1.0, 0.75],
];

fn authored_uvs(appearance: AuthoredPart) -> [[f32; 2]; 24] {
    let assimp_uvs = match appearance {
        AuthoredPart::Controller => CONTROLLER_UVS,
        AuthoredPart::GasEngine => GAS_ENGINE_UVS,
        AuthoredPart::ElectricEngine => ELECTRIC_ENGINE_UVS,
        AuthoredPart::Servo => SERVO_UVS,
        AuthoredPart::Seat => SEAT_UVS,
        AuthoredPart::Input => INPUT_UVS,
    };
    // Assimp's dump uses an OpenGL-style bottom-left texture origin. Bevy
    // samples the PNG atlases from the top left, matching the original glTF
    // accessor, so restore that V coordinate before building the runtime mesh.
    assimp_uvs.map(|[u, v]| [u, 1.0 - v])
}

fn single_authored_part_mesh(appearance: AuthoredPart) -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, AUTHORED_CUBE_POSITIONS.to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, AUTHORED_CUBE_NORMALS.to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, authored_uvs(appearance).to_vec())
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, AUTHORED_CUBE_TANGENTS.to_vec())
    .with_inserted_indices(Indices::U32(AUTHORED_CUBE_INDICES.to_vec()))
}

fn combined_construction_mesh(graph: &ConstructionGraph) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for (_, spec) in graph
        .parts()
        .filter(|(_, spec)| matches!(spec, PartSpec::Cuboid(_) | PartSpec::Cylinder(_)))
    {
        append_part(*spec, 1.0, &mut positions, &mut normals, &mut indices);
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// Control blocks render as their own teal mesh so they stand out from the
/// construction they steer.
#[cfg(test)]
fn combined_controller_mesh(graph: &ConstructionGraph) -> Mesh {
    combined_authored_construction_mesh(graph, AuthoredPart::Controller)
}

fn combined_authored_construction_mesh(
    graph: &ConstructionGraph,
    appearance: AuthoredPart,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for (_, spec) in graph.parts().filter(|(_, spec)| appearance.matches(**spec)) {
        let cuboid = spec
            .as_cuboid()
            .expect("authored machine appearances have cuboid envelopes");
        append_authored_cuboid(
            cuboid.pose.translation(),
            cuboid.pose.rotation.quaternion(),
            cuboid.size_meters(),
            appearance,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_parts_mesh_scaled(specs: &[PartSpec], scale_factor: f32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for &spec in specs {
        append_part(
            spec,
            scale_factor,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// Exact world bounds of a block sheet, including the outer half-block skin.
pub(crate) fn block_sheet_bounds(specs: &[CuboidSpec]) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for &spec in specs {
        let (part_minimum, part_maximum) = part_world_bounds(PartSpec::Cuboid(spec));
        minimum = minimum.min(part_minimum);
        maximum = maximum.max(part_maximum);
    }
    minimum.is_finite().then_some((minimum, maximum))
}

/// One exterior cuboid spanning the live sheet preview, inset just enough to
/// keep its contact faces from being coplanar with opaque construction.
///
/// Built directly from bounds because a valid 4,096-block sheet can be wider
/// than the construction API's per-cuboid dimension limit.
fn block_sheet_preview_mesh(specs: &[CuboidSpec]) -> Mesh {
    let (minimum, maximum) = block_sheet_bounds(specs).expect("a block drag contains a block");
    let visual_minimum = minimum + Vec3::splat(BLOCK_SHEET_PREVIEW_INSET_METERS);
    let visual_maximum = maximum - Vec3::splat(BLOCK_SHEET_PREVIEW_INSET_METERS);
    let mut positions = Vec::with_capacity(CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(CUBE_INDICES.len());
    append_transformed_cuboid(
        (visual_minimum + visual_maximum) * 0.5,
        Quat::IDENTITY,
        visual_maximum - visual_minimum,
        &mut positions,
        &mut normals,
        &mut indices,
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

#[derive(Clone, Copy)]
enum SimulationMeshKind {
    Static,
    Dynamic,
}

fn combined_simulation_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
) -> Mesh {
    let parts = creation
        .part_to_compound
        .iter()
        .filter(|(part, compound_index)| {
            let is_authored = graph.part(*part).is_some_and(|spec| {
                matches!(
                    spec,
                    PartSpec::Controller(_)
                        | PartSpec::Engine(_)
                        | PartSpec::Servo(_)
                        | PartSpec::Seat(_)
                        | PartSpec::Input(_)
                )
            });
            let is_static = creation.compounds[*compound_index as usize].is_static;
            match kind {
                SimulationMeshKind::Static => is_static && !is_authored,
                SimulationMeshKind::Dynamic => !is_static && !is_authored,
            }
        });
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for &(part, compound_index) in parts {
        let transform = transforms[compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        let initial = &creation.compounds[compound_index as usize];
        let spec = *graph.part(part).expect("compiled source remains in graph");
        let local_center = spec.pose().translation() - initial.root_translation;
        let world_spec = match spec {
            PartSpec::Cuboid(cuboid) => {
                append_transformed_cuboid(
                    root_translation + root_rotation * local_center,
                    root_rotation * cuboid.pose.rotation.quaternion(),
                    cuboid.size_meters(),
                    &mut positions,
                    &mut normals,
                    &mut indices,
                );
                continue;
            }
            PartSpec::Controller(_)
            | PartSpec::Engine(_)
            | PartSpec::Servo(_)
            | PartSpec::Seat(_)
            | PartSpec::Input(_) => {
                unreachable!("authored parts render in their material-specific mesh")
            }
            PartSpec::Cylinder(cylinder) => cylinder,
        };
        append_cylinder_shape(
            root_translation + root_rotation * local_center,
            root_rotation * world_spec.pose.rotation.quaternion(),
            world_spec.dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_simulation_authored_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    appearance: AuthoredPart,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for &(part, compound_index) in creation.part_to_compound.iter().filter(|(part, _)| {
        graph
            .part(*part)
            .is_some_and(|spec| appearance.matches(*spec))
    }) {
        let transform = transforms[compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        let initial = &creation.compounds[compound_index as usize];
        let spec = *graph.part(part).expect("compiled source remains in graph");
        let local_center = spec.pose().translation() - initial.root_translation;
        let cuboid = spec
            .as_cuboid()
            .expect("authored machine appearances have cuboid envelopes");
        append_authored_cuboid(
            root_translation + root_rotation * local_center,
            root_rotation * cuboid.pose.rotation.quaternion(),
            cuboid.size_meters(),
            appearance,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_bearing_mesh(graph: &ConstructionGraph, placed_bearings: &[PlacedBearing]) -> Mesh {
    const SEGMENTS: usize = 24;
    let vertices_per_bearing = SEGMENTS * 8;
    let indices_per_bearing = SEGMENTS * 24;
    let bearing_count = visible_bearing_count(graph, placed_bearings);
    let mut positions = Vec::with_capacity(bearing_count * vertices_per_bearing);
    let mut normals = Vec::with_capacity(bearing_count * vertices_per_bearing);
    let mut indices = Vec::with_capacity(bearing_count * indices_per_bearing);
    for (_, bearing) in graph.bearings().filter(|(_, bearing)| {
        !placed_bearings
            .iter()
            .any(|&socket| bearing_uses_socket(bearing, socket))
    }) {
        append_bearing_cylinder(
            bearing.shared_anchor,
            bearing.axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }
    for bearing in placed_bearings {
        let axis = face_geometry_from_ref(bearing.source, Some(graph)).normal;
        append_bearing_cylinder(
            bearing.anchor,
            axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_simulation_bearing_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
) -> Mesh {
    const SEGMENTS: usize = 24;
    let bearing_count = creation.bearings.len() + placed_bearings.len();
    let mut positions = Vec::with_capacity(bearing_count * SEGMENTS * 8);
    let mut normals = Vec::with_capacity(bearing_count * SEGMENTS * 8);
    let mut indices = Vec::with_capacity(bearing_count * SEGMENTS * 24);

    for compiled in &creation.bearings {
        let bearing = graph
            .bearing(compiled.source_bearing)
            .expect("compiled bearing source remains in graph");
        if placed_bearings
            .iter()
            .any(|&socket| bearing_uses_socket(bearing, socket))
        {
            continue;
        }
        let (anchor, axis) = transform_bearing_pose(
            transforms[compiled.compound_a as usize],
            compiled.local_anchor_a,
            compiled.local_axis_a,
        );
        append_bearing_cylinder(
            anchor,
            axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    for bearing in placed_bearings {
        let FaceOwner::Part(source_part) = bearing.source.owner else {
            continue;
        };
        let Some(compound_index) = creation
            .part_to_compound
            .iter()
            .find_map(|&(part, index)| (part == source_part).then_some(index))
        else {
            continue;
        };
        let initial = &creation.compounds[compound_index as usize];
        let inverse_initial_rotation = initial.root_rotation.inverse();
        let local_anchor = inverse_initial_rotation * (bearing.anchor - initial.root_translation);
        let source_axis = face_geometry_from_ref(bearing.source, Some(graph)).normal;
        let local_axis = inverse_initial_rotation * source_axis;
        let (anchor, axis) = transform_bearing_pose(
            transforms[compound_index as usize],
            local_anchor,
            local_axis,
        );
        append_bearing_cylinder(
            anchor,
            axis,
            bearing.dimensions,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// Where one graph bearing sits in simulation space.
///
/// The published snapshot moves compounds, so a running mechanism's joint is
/// found through its compiled row rather than through its build pose.
fn simulation_bearing_pose(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    bearing: &mechanic_core::BearingSpec,
) -> Option<(Vec3, Vec3)> {
    let compiled = creation
        .bearings
        .iter()
        .find(|compiled| graph.bearing(compiled.source_bearing) == Some(bearing))?;
    Some(transform_bearing_pose(
        *transforms.get(compiled.compound_a as usize)?,
        compiled.local_anchor_a,
        compiled.local_axis_a,
    ))
}

fn transform_bearing_pose(
    transform: GpuTransform,
    local_anchor: Vec3,
    local_axis: Vec3,
) -> (Vec3, Vec3) {
    let translation = Vec3::new(
        transform.position[0],
        transform.position[1],
        transform.position[2],
    );
    let rotation = Quat::from_array(transform.rotation);
    (translation + rotation * local_anchor, rotation * local_axis)
}

fn single_bearing_mesh(dimensions: BearingDimensions) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_bearing_cylinder(
        Vec3::ZERO,
        Vec3::Y,
        dimensions,
        &mut positions,
        &mut normals,
        &mut indices,
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

#[allow(clippy::too_many_lines)] // Solid and annular surfaces share one indexed mesh layout.
fn append_bearing_cylinder(
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let inner_diameter = if dimensions.inner_diameter() > 0.0 {
        dimensions.inner_diameter() + BEARING_RENDER_RADIAL_SKIN * 2.0
    } else {
        0.0
    };
    append_annular_cylinder(
        anchor,
        axis,
        dimensions.outer_diameter() - BEARING_RENDER_RADIAL_SKIN * 2.0,
        inner_diameter,
        BEARING_DEPTH,
        positions,
        normals,
        indices,
    );
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn append_cylinder_shape(
    center: Vec3,
    rotation: Quat,
    dimensions: CylinderDimensions,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    if dimensions.sweep_angle_degrees() == 360 {
        append_annular_cylinder(
            center,
            rotation * Vec3::Y,
            dimensions.outer_diameter() * scale,
            dimensions.inner_diameter() * scale,
            dimensions.axial_length() * scale,
            positions,
            normals,
            indices,
        );
        return;
    }

    let axis = rotation * Vec3::Y;
    let tangent_u = rotation * Vec3::X;
    let tangent_v = rotation * Vec3::Z;
    let outer = dimensions.outer_diameter() * scale * 0.5;
    let inner = dimensions.inner_diameter() * scale * 0.5;
    let half_length = dimensions.axial_length() * scale * 0.5;
    let sweep = dimensions.sweep_angle_radians();
    let segment_count = dimensions.sweep_angle_degrees() / 15;
    let lower = center - axis * half_length;
    let upper = center + axis * half_length;
    let radial = |angle: f32| tangent_u * angle.cos() + tangent_v * angle.sin();
    let angular = |angle: f32| -tangent_u * angle.sin() + tangent_v * angle.cos();

    for segment in 0..segment_count {
        let first_angle = -sweep * 0.5 + sweep * f32::from(segment) / f32::from(segment_count);
        let second_angle = -sweep * 0.5 + sweep * f32::from(segment + 1) / f32::from(segment_count);
        let first = radial(first_angle);
        let second = radial(second_angle);
        append_mesh_quad(
            [
                lower + first * outer,
                upper + first * outer,
                upper + second * outer,
                lower + second * outer,
            ],
            (first + second).normalize(),
            positions,
            normals,
            indices,
        );
        if inner > 0.0 {
            append_mesh_quad(
                [
                    lower + second * inner,
                    upper + second * inner,
                    upper + first * inner,
                    lower + first * inner,
                ],
                -(first + second).normalize(),
                positions,
                normals,
                indices,
            );
            append_mesh_quad(
                [
                    lower + first * inner,
                    lower + first * outer,
                    lower + second * outer,
                    lower + second * inner,
                ],
                -axis,
                positions,
                normals,
                indices,
            );
            append_mesh_quad(
                [
                    upper + second * inner,
                    upper + second * outer,
                    upper + first * outer,
                    upper + first * inner,
                ],
                axis,
                positions,
                normals,
                indices,
            );
        } else {
            append_mesh_triangle(
                [lower, lower + first * outer, lower + second * outer],
                -axis,
                positions,
                normals,
                indices,
            );
            append_mesh_triangle(
                [upper, upper + second * outer, upper + first * outer],
                axis,
                positions,
                normals,
                indices,
            );
        }
    }

    for (angle, normal, reverse) in [
        (-sweep * 0.5, -angular(-sweep * 0.5), false),
        (sweep * 0.5, angular(sweep * 0.5), true),
    ] {
        let direction = radial(angle);
        let inner_lower = lower + direction * inner;
        let inner_upper = upper + direction * inner;
        let outer_lower = lower + direction * outer;
        let outer_upper = upper + direction * outer;
        let vertices = if reverse {
            [inner_lower, outer_lower, outer_upper, inner_upper]
        } else {
            [inner_lower, inner_upper, outer_upper, outer_lower]
        };
        append_mesh_quad(vertices, normal, positions, normals, indices);
    }
}

/// Radius multiplier placing the spin arc just outside a driven bearing's ring.
const DRIVE_ARC_RADIUS_SCALE: f32 = 1.4;
/// Half-thickness of every drive overlay ribbon, in metres.
const DRIVE_OVERLAY_HALF_WIDTH: f32 = 0.012;
/// Arc sweep of the spin-direction indicator, in radians.
const DRIVE_ARC_SWEEP: f32 = core::f32::consts::PI * 1.25;

/// Orthonormal pair spanning the plane perpendicular to `axis`. Matches the
/// basis every bearing ring is already built from.
fn axis_tangents(axis: Vec3) -> (Vec3, Vec3) {
    let axis = axis.normalize();
    let tangent_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    (tangent_u, axis.cross(tangent_u))
}

/// Flat ribbon between two points, kept broadside to the viewer-independent
/// `face` normal so it stays visible in the unlit x-ray pass.
fn append_overlay_segment(
    start: Vec3,
    end: Vec3,
    face: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let along = end - start;
    if along.length_squared() <= f32::EPSILON {
        return;
    }
    let side = along.normalize().cross(face.normalize());
    if side.length_squared() <= f32::EPSILON {
        return;
    }
    let offset = side.normalize() * DRIVE_OVERLAY_HALF_WIDTH;
    append_mesh_quad(
        [start - offset, end - offset, end + offset, start + offset],
        face.normalize(),
        positions,
        normals,
        indices,
    );
}

/// Spin arc, arrow head, and optional angle-limit ticks for one driven bearing.
#[allow(clippy::too_many_arguments)] // Overlay builders thread three mesh buffers.
fn append_drive_indicator(
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
    state: DriveState,
    travel: Option<(f32, f32)>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const ARC_SEGMENTS: usize = 20;
    let axis = axis.normalize();
    let (tangent_u, tangent_v) = axis_tangents(axis);
    let radius = dimensions.outer_diameter() * 0.5 * DRIVE_ARC_RADIUS_SCALE;
    let radial = |angle: f32| tangent_u * angle.cos() + tangent_v * angle.sin();
    // The arc sweeps the way the joint is being asked to turn.
    let signed = match state.target() {
        DriveTarget::Angle(angle) => angle,
        DriveTarget::Speed(speed) => speed,
    };
    let winding = if signed < 0.0 { -1.0 } else { 1.0 };

    let mut previous = anchor + radial(0.0) * radius;
    for segment in 1..=ARC_SEGMENTS {
        let fraction = f32::from(u8::try_from(segment).unwrap_or(u8::MAX))
            / f32::from(u8::try_from(ARC_SEGMENTS).unwrap_or(u8::MAX));
        let angle = winding * DRIVE_ARC_SWEEP * fraction;
        let point = anchor + radial(angle) * radius;
        append_overlay_segment(previous, point, axis, positions, normals, indices);
        previous = point;
    }

    let tip_angle = winding * DRIVE_ARC_SWEEP;
    let tangent = (radial(tip_angle + winding * 0.01) - radial(tip_angle)).normalize_or_zero();
    if tangent != Vec3::ZERO {
        let outward = radial(tip_angle);
        let head = radius * 0.32;
        append_mesh_triangle(
            [
                previous + tangent * head,
                previous - outward * head * 0.5,
                previous + outward * head * 0.5,
            ],
            axis,
            positions,
            normals,
            indices,
        );
    }

    if let Some((minimum, maximum)) = travel {
        for angle in [minimum, maximum] {
            let direction = radial(angle);
            append_overlay_segment(
                anchor + direction * radius * 0.9,
                anchor + direction * radius * 1.25,
                axis,
                positions,
                normals,
                indices,
            );
        }
    }
}

/// Straight wire from a driven bearing to the control block steering it.
fn append_drive_wire(
    anchor: Vec3,
    controller_center: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let along = controller_center - anchor;
    if along.length_squared() <= f32::EPSILON {
        return;
    }
    // Two crossed ribbons so the wire reads from any camera angle.
    let (tangent_u, tangent_v) = axis_tangents(along);
    for face in [tangent_u, tangent_v] {
        append_overlay_segment(anchor, controller_center, face, positions, normals, indices);
    }
}

/// Mesh for the wire being dragged out by the pointer.
///
/// A wire with no length still yields one degenerate triangle so the visual
/// always has vertex data to allocate, which keeps it renderable-but-invisible
/// while no drag is in progress.
fn wire_drag_preview_mesh(from: Vec3, to: Vec3) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_drive_wire(from, to, &mut positions, &mut normals, &mut indices);
    if positions.is_empty() {
        return degenerate_overlay_mesh();
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// A single zero-area triangle: nothing to see, but still vertex data.
///
/// Overlays that come and go stay visible and swap to this instead of hiding,
/// because a hidden mesh has no slab allocation and writing to one makes the
/// renderer log a use-after-free.
fn degenerate_overlay_mesh() -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_mesh_triangle(
        [Vec3::ZERO; 3],
        Vec3::Y,
        &mut positions,
        &mut normals,
        &mut indices,
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

/// Keeps an empty logical batch allocated in Bevy's GPU mesh slabs.
///
/// Bevy 0.19 frees a modified mesh's previous allocation before discovering
/// that a zero-vertex replacement needs no new allocation, then still attempts
/// to upload its vertex and index data. A zero-area triangle is invisible but
/// preserves both allocations and avoids that renderer use-after-free path.
fn renderable_mesh(mesh: Mesh) -> Mesh {
    if mesh.count_vertices() == 0 {
        degenerate_overlay_mesh()
    } else {
        mesh
    }
}

/// How much bigger a wirable joint or block is drawn while the pointer is on
/// it. The ring is thin, so it needs more than the solid block does.
const WIRE_HOVER_BEARING_SCALE: f32 = 1.3;
const WIRE_HOVER_BLOCK_SCALE: f32 = 1.14;

/// Draws the joint or control block the pointer is over, slightly oversized, so
/// what a wire would land on is visible before the button goes down.
#[allow(clippy::too_many_arguments)]
fn update_wire_hover_preview(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    selection: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut drawn: Local<Option<WireEnd>>,
    mut transform: Single<&mut Transform, With<WireHoverVisual>>,
) {
    let hovered = if selection.0 == Tool::Connector && !simulation.is_running() {
        wire_end_under_cursor(&graph.0, &state)
    } else {
        None
    };
    let placement = match hovered {
        Some(WireEnd::Bearing(index)) => state.placed_bearings.get(index).map(|socket| {
            let normal = face_geometry_from_ref(socket.source, Some(&graph.0)).normal;
            Transform::from_translation(socket.anchor)
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, normal))
                .with_scale(Vec3::splat(WIRE_HOVER_BEARING_SCALE))
        }),
        Some(WireEnd::Controller(part) | WireEnd::Input(part) | WireEnd::Seat(part)) => graph
            .0
            .part(part)
            .and_then(|spec| spec.as_cuboid())
            .map(|block| {
                Transform::from_translation(block.pose.translation())
                    .with_rotation(block.pose.rotation.quaternion())
                    .with_scale(block.size_meters() * WIRE_HOVER_BLOCK_SCALE)
            }),
        None => None,
    };
    let hovered = placement.is_some().then_some(hovered).flatten();
    if *drawn != hovered
        && let Some(mut mesh) = meshes.get_mut(&visuals.wire_hover_mesh)
    {
        *mesh = match hovered {
            Some(WireEnd::Bearing(index)) => state
                .placed_bearings
                .get(index)
                .map_or_else(degenerate_overlay_mesh, |socket| {
                    single_bearing_mesh(socket.dimensions)
                }),
            Some(WireEnd::Controller(_) | WireEnd::Input(_) | WireEnd::Seat(_)) => {
                Cuboid::default().into()
            }
            None => degenerate_overlay_mesh(),
        };
    }
    *drawn = hovered;
    **transform = placement.unwrap_or_default();
}

/// Draws the wire from the end it was started on to the pointer, snapping to a
/// joint or block once the pointer is over one that would complete it.
fn update_wire_drag_preview(
    graph: Res<EditorGraph>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut state: ResMut<EditorState>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut drawn: Local<bool>,
) {
    // The release that ends a drag is swallowed when it lands on the camera or
    // the hotbar, so a wire that is neither held nor armed is dropped here
    // rather than left following the pointer forever.
    if state
        .wire_drag
        .is_some_and(|drag| !drag.armed && !mouse.pressed(MouseButton::Left))
    {
        state.wire_drag = None;
    }
    let endpoints = wire_drag_endpoints(&graph.0, &state);
    if endpoints.is_none() && !*drawn {
        return;
    }
    let (from, to) = endpoints.unwrap_or((Vec3::ZERO, Vec3::ZERO));
    if let Some(mut mesh) = meshes.get_mut(&visuals.wire_drag_mesh) {
        *mesh = wire_drag_preview_mesh(from, to);
    }
    *drawn = endpoints.is_some();
}

fn combined_drive_xray_mesh(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    sequencer: &DriveSequencer,
) -> Mesh {
    drive_xray_mesh(
        graph,
        placed_bearings,
        sequencer,
        |bearing, controller| {
            Some((
                bearing.shared_anchor,
                bearing.axis,
                graph.part(controller)?.pose().translation(),
            ))
        },
        |part| Some(graph.part(part)?.pose().translation()),
    )
}

/// The same overlay in simulation space, following the bodies as they move.
///
/// Bearings and control blocks are read from the published snapshot rather than
/// the build pose, so the arcs and wires stay attached to a running mechanism.
fn combined_simulation_drive_xray_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
    sequencer: &DriveSequencer,
) -> Mesh {
    let compound_of = |part: PartId| {
        creation
            .part_to_compound
            .iter()
            .find_map(|(candidate, compound)| (*candidate == part).then_some(*compound))
    };
    let part_position = |part: PartId| {
        let compound = compound_of(part)?;
        let block = *transforms.get(compound as usize)?;
        Some(
            Vec3::new(block.position[0], block.position[1], block.position[2])
                + Quat::from_array(block.rotation)
                    * (graph.part(part)?.pose().translation()
                        - creation.compounds[compound as usize].root_translation),
        )
    };
    drive_xray_mesh(
        graph,
        placed_bearings,
        sequencer,
        |bearing, controller| {
            let (anchor, axis) = simulation_bearing_pose(graph, creation, transforms, bearing)?;
            Some((anchor, axis, part_position(controller)?))
        },
        part_position,
    )
}

/// Shared overlay builder. `pose` resolves one bearing and its control block to
/// world space, which is the only thing that differs between build and
/// simulation.
fn drive_xray_mesh(
    graph: &ConstructionGraph,
    placed_bearings: &[PlacedBearing],
    sequencer: &DriveSequencer,
    pose: impl Fn(&mechanic_core::BearingSpec, PartId) -> Option<(Vec3, Vec3, Vec3)>,
    part_position: impl Fn(PartId) -> Option<Vec3>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut drawn = Vec::new();

    for (id, bearing) in graph.bearings() {
        let Some((link, spec)) = graph.bearing_drive_link(id) else {
            continue;
        };
        // While simulating, the arc shows the state the bearing is actually in.
        let active = sequencer.active_state(link).unwrap_or(0);
        let Some(state) = spec.program.state(active) else {
            continue;
        };
        let state = if spec.reversed {
            state
                .with_target(state.target().reversed())
                .unwrap_or(state)
        } else {
            state
        };
        let travel = spec.limits.angle_limits();
        // A socket carrying several rotor rows would otherwise draw its arc more
        // than once at the same place.
        if drawn
            .iter()
            .any(|previous: &Vec3| previous.abs_diff_eq(bearing.shared_anchor, 1.0e-5))
        {
            continue;
        }
        drawn.push(bearing.shared_anchor);

        let Some((anchor, axis, controller_center)) = pose(bearing, spec.controller) else {
            continue;
        };
        let dimensions = placed_bearings
            .iter()
            .find(|socket| bearing_uses_socket(bearing, **socket))
            .map_or(bearing.dimensions, |socket| socket.dimensions);
        append_drive_indicator(
            anchor,
            axis,
            dimensions,
            state,
            travel,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        append_drive_wire(
            anchor,
            controller_center,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    for (_, link) in graph.input_seat_links() {
        if let (Some(input), Some(seat)) = (part_position(link.input), part_position(link.seat)) {
            append_drive_wire(input, seat, &mut positions, &mut normals, &mut indices);
        }
    }
    for (_, link) in graph.seat_controller_links() {
        if let (Some(seat), Some(controller)) =
            (part_position(link.seat), part_position(link.controller))
        {
            append_drive_wire(seat, controller, &mut positions, &mut normals, &mut indices);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn append_mesh_triangle(
    vertices: [Vec3; 3],
    normal: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(vertices.map(|vertex| vertex.to_array()));
    normals.extend([normal.to_array(); 3]);
    indices.extend([base, base + 1, base + 2]);
}

fn append_mesh_quad(
    vertices: [Vec3; 4],
    normal: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(vertices.map(|vertex| vertex.to_array()));
    normals.extend([normal.to_array(); 4]);
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn append_annular_cylinder(
    anchor: Vec3,
    axis: Vec3,
    outer_diameter: f32,
    inner_diameter: f32,
    axial_length: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const SEGMENTS: u16 = 24;
    let axis = axis.normalize();
    let tangent_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    let tangent_v = axis.cross(tangent_u);
    let outer_radius = outer_diameter * 0.5;
    let inner_radius = inner_diameter * 0.5;
    let half_depth = axial_length * 0.5;
    let lower = anchor - axis * half_depth;
    let upper = anchor + axis * half_depth;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");

    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * outer_radius).to_array());
        positions.push((upper + radial * outer_radius).to_array());
        normals.push(radial.to_array());
        normals.push(radial.to_array());
    }
    for segment in 0..SEGMENTS {
        let next = (segment + 1) % SEGMENTS;
        let lower_current = base + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = base + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        indices.extend([
            lower_current,
            lower_next,
            upper_current,
            upper_current,
            lower_next,
            upper_next,
        ]);
    }

    if inner_radius == 0.0 {
        let lower_center = u32::try_from(positions.len()).unwrap();
        positions.push(lower.to_array());
        normals.push((-axis).to_array());
        let upper_center = u32::try_from(positions.len()).unwrap();
        positions.push(upper.to_array());
        normals.push(axis.to_array());
        let lower_ring = u32::try_from(positions.len()).unwrap();
        for segment in 0..SEGMENTS {
            let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
            let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
            positions.push((lower + radial * outer_radius).to_array());
            normals.push((-axis).to_array());
        }
        let upper_ring = u32::try_from(positions.len()).unwrap();
        for segment in 0..SEGMENTS {
            let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
            let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
            positions.push((upper + radial * outer_radius).to_array());
            normals.push(axis.to_array());
        }
        for segment in 0..SEGMENTS {
            let next = (segment + 1) % SEGMENTS;
            let current = u32::from(segment);
            let next = u32::from(next);
            indices.extend([
                lower_center,
                lower_ring + next,
                lower_ring + current,
                upper_center,
                upper_ring + current,
                upper_ring + next,
            ]);
        }
        return;
    }

    let inner_side = u32::try_from(positions.len()).unwrap();
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * inner_radius).to_array());
        positions.push((upper + radial * inner_radius).to_array());
        normals.push((-radial).to_array());
        normals.push((-radial).to_array());
    }
    for segment in 0..SEGMENTS {
        let next = (segment + 1) % SEGMENTS;
        let lower_current = inner_side + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = inner_side + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        indices.extend([
            lower_current,
            upper_current,
            lower_next,
            upper_current,
            upper_next,
            lower_next,
        ]);
    }

    let lower_outer = append_bearing_face_ring(
        lower,
        -axis,
        outer_radius,
        tangent_u,
        tangent_v,
        positions,
        normals,
    );
    let lower_inner = append_bearing_face_ring(
        lower,
        -axis,
        inner_radius,
        tangent_u,
        tangent_v,
        positions,
        normals,
    );
    let upper_outer = append_bearing_face_ring(
        upper,
        axis,
        outer_radius,
        tangent_u,
        tangent_v,
        positions,
        normals,
    );
    let upper_inner = append_bearing_face_ring(
        upper,
        axis,
        inner_radius,
        tangent_u,
        tangent_v,
        positions,
        normals,
    );
    for segment in 0..SEGMENTS {
        let next = (segment + 1) % SEGMENTS;
        let current = u32::from(segment);
        let next = u32::from(next);
        indices.extend([
            lower_outer + current,
            lower_inner + current,
            lower_outer + next,
            lower_outer + next,
            lower_inner + current,
            lower_inner + next,
            upper_outer + current,
            upper_outer + next,
            upper_inner + current,
            upper_outer + next,
            upper_inner + next,
            upper_inner + current,
        ]);
    }
}

fn append_bearing_face_ring(
    center: Vec3,
    normal: Vec3,
    radius: f32,
    tangent_u: Vec3,
    tangent_v: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
) -> u32 {
    const SEGMENTS: u16 = 24;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((center + radial * radius).to_array());
        normals.push(normal.to_array());
    }
    base
}

fn append_part(
    spec: PartSpec,
    scale_factor: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    match spec {
        PartSpec::Cuboid(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Controller(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Engine(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Servo(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Seat(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Input(spec) => append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.cuboid().size_meters() * scale_factor,
            positions,
            normals,
            indices,
        ),
        PartSpec::Cylinder(spec) => append_cylinder_shape(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.dimensions,
            scale_factor,
            positions,
            normals,
            indices,
        ),
    }
}

fn single_cylinder_mesh(dimensions: CylinderDimensions) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    append_cylinder_shape(
        Vec3::ZERO,
        Quat::IDENTITY,
        dimensions,
        1.0,
        &mut positions,
        &mut normals,
        &mut indices,
    );
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn append_transformed_cuboid(
    translation: Vec3,
    rotation: Quat,
    size: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base_index = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(
        CUBE_POSITIONS.map(|position| {
            (translation + rotation * (Vec3::from_array(position) * size)).to_array()
        }),
    );
    normals.extend(CUBE_NORMALS.map(|normal| (rotation * Vec3::from_array(normal)).to_array()));
    indices.extend(CUBE_INDICES.map(|index| base_index + index));
}

#[allow(clippy::too_many_arguments)] // Appends every vertex stream of one authored cuboid.
fn append_authored_cuboid(
    translation: Vec3,
    rotation: Quat,
    size: Vec3,
    appearance: AuthoredPart,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let base_index = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(
        AUTHORED_CUBE_POSITIONS.map(|position| {
            (translation + rotation * (Vec3::from_array(position) * size)).to_array()
        }),
    );
    normals.extend(
        AUTHORED_CUBE_NORMALS.map(|normal| (rotation * Vec3::from_array(normal)).to_array()),
    );
    uvs.extend(authored_uvs(appearance));
    tangents.extend(AUTHORED_CUBE_TANGENTS.map(|tangent| {
        let tangent_xyz = rotation * Vec3::from_array(tangent[..3].try_into().unwrap());
        [tangent_xyz.x, tangent_xyz.y, tangent_xyz.z, tangent[3]]
    }));
    indices.extend(AUTHORED_CUBE_INDICES.map(|index| base_index + index));
}

#[cfg(test)]
mod rendering_tests {
    use bevy::{
        mesh::VertexAttributeValues,
        prelude::{AlphaMode, Color, Handle, IVec3, Image, Mesh, Quat, StandardMaterial, Vec3},
    };
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec, DriveLimits, DriveLinkSpec,
        DriveProgram, DriveState, DriveTarget, EngineKind, EngineSpec, FaceKind, FaceRef,
        GridRotation, InputSpec, SeatSpec, ServoSpec,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        AuthoredPart, BEARING_DEPTH, BEARING_RENDER_RADIAL_SKIN, BLOCK_SHEET_PREVIEW_INSET_METERS,
        PlacedBearing, SimulationMeshKind, append_bearing_cylinder, append_cylinder_shape,
        authored_preview_material, authored_uvs, bearing_preview_dimensions_changed,
        bearing_surface_material, block_sheet_bounds, block_sheet_preview_mesh, block_sheet_specs,
        combined_authored_construction_mesh, combined_bearing_mesh, combined_controller_mesh,
        combined_drive_xray_mesh, combined_simulation_bearing_mesh, combined_simulation_mesh,
        drive_xray_is_visible, joint_xray_is_visible, preview_material, renderable_mesh,
        single_authored_part_mesh, single_bearing_mesh, single_cylinder_mesh,
    };
    use crate::PlacementPlane;
    use crate::hotbar::Tool;
    use crate::sequencer::DriveSequencer;

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        let Some(VertexAttributeValues::Float32x3(values)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("mesh must have float3 positions")
        };
        values.iter().copied().map(Vec3::from_array).collect()
    }

    #[test]
    fn bearing_mesh_insets_radial_surfaces_without_changing_depth() {
        let anchor = Vec3::new(2.0, 3.0, 4.0);
        let axis = Vec3::X;
        let dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            anchor,
            axis,
            dimensions,
            &mut positions,
            &mut normals,
            &mut indices,
        );

        let offsets = positions
            .iter()
            .map(|position| Vec3::from_array(*position) - anchor)
            .collect::<Vec<_>>();
        let minimum_depth = offsets
            .iter()
            .map(|offset| offset.dot(axis))
            .fold(f32::INFINITY, f32::min);
        let maximum_depth = offsets
            .iter()
            .map(|offset| offset.dot(axis))
            .fold(f32::NEG_INFINITY, f32::max);
        let maximum_radius = offsets
            .iter()
            .map(|offset| (*offset - axis * offset.dot(axis)).length())
            .fold(0.0, f32::max);
        let minimum_radius = offsets
            .iter()
            .map(|offset| (*offset - axis * offset.dot(axis)).length())
            .fold(f32::INFINITY, f32::min);

        assert!((minimum_depth + BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!((maximum_depth - BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!(
            (maximum_radius - (dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
        assert!(
            (minimum_radius - (dimensions.inner_diameter() * 0.5 + BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn bearing_material_biases_coplanar_surfaces_toward_the_camera() {
        assert!(bearing_surface_material().depth_bias > 0.0);
    }

    #[test]
    fn every_preview_material_biases_coplanar_surfaces_toward_the_camera() {
        for color in [
            Color::srgba(1.0, 1.0, 1.0, 0.34),
            Color::srgba(1.0, 0.06, 0.04, 0.46),
            Color::srgba(0.12, 1.0, 0.28, 0.52),
        ] {
            assert!(preview_material(color).depth_bias > 0.0);
        }
    }

    #[test]
    fn block_sheet_preview_is_one_cuboid_inset_from_every_logical_contact_plane() {
        let start = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
        )
        .unwrap();
        let endpoint = start.pose.translation_half_units() + IVec3::new(10, 0, -6);
        let specs = block_sheet_specs(start, endpoint, PlacementPlane::Xz).unwrap();
        let expected = block_sheet_bounds(&specs).unwrap();
        let mesh = block_sheet_preview_mesh(&specs);
        let vertices = positions(&mesh);
        let actual_minimum = vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::INFINITY), Vec3::min);
        let actual_maximum = vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);

        assert_eq!(vertices.len(), 24);
        assert_eq!(mesh.indices().unwrap().len(), 36);
        let inset = Vec3::splat(BLOCK_SHEET_PREVIEW_INSET_METERS);
        assert!(actual_minimum.abs_diff_eq(expected.0 + inset, 1.0e-6));
        assert!(actual_maximum.abs_diff_eq(expected.1 - inset, 1.0e-6));
    }

    #[test]
    fn solid_bearing_remains_closed_when_its_outer_radius_is_inset() {
        let mesh = single_bearing_mesh(BearingDimensions::new(0.4, 0.0).unwrap());
        assert!(positions(&mesh).iter().any(|position| {
            position.x.hypot(position.z) < f32::EPSILON && position.y.abs() <= BEARING_DEPTH * 0.5
        }));
    }

    #[test]
    fn cylinder_mesh_uses_exact_radii_and_variable_axial_length() {
        let mesh = single_cylinder_mesh(CylinderDimensions::new(1.2, 0.4, 2.0).unwrap());
        let positions = positions(&mesh);
        let maximum_radius = positions
            .iter()
            .map(|position| position.x.hypot(position.z))
            .fold(0.0_f32, f32::max);
        let maximum_y = positions
            .iter()
            .map(|position| position.y.abs())
            .fold(0.0_f32, f32::max);
        assert!((maximum_radius - 0.6).abs() < 1.0e-5);
        assert!((maximum_y - 1.0).abs() < 1.0e-5);
        assert!(
            positions
                .iter()
                .any(|position| { (position.x.hypot(position.z) - 0.2).abs() < 1.0e-5 })
        );
    }

    #[test]
    fn cylinder_sector_mesh_has_cut_walls_and_outward_winding() {
        let dimensions = CylinderDimensions::new(1.0, 0.5, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_cylinder_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );

        assert!(positions.iter().all(|position| position[0] >= -1.0e-6));
        assert!(normals.iter().any(|normal| normal[1].abs() < 1.0e-6
            && normal[0].abs() > 0.5
            && normal[2].abs() > 0.5));
        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric_normal = (b - a).cross(c - a);
            let expected_normal = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(geometric_normal.dot(expected_normal) > 0.0);
        }
    }

    #[test]
    fn simulation_renders_one_cylinder_despite_sixteen_physical_colliders() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
                BuildPose::default(),
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        assert_eq!(
            creation.colliders.len(),
            mechanic_core::CYLINDER_COLLIDER_COUNT
        );
        let transforms = [GpuTransform {
            position: [0.0, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];
        let mesh =
            combined_simulation_mesh(&graph, &creation, &transforms, SimulationMeshKind::Dynamic);
        assert_eq!(
            mesh.count_vertices(),
            single_cylinder_mesh(CylinderDimensions::new(1.0, 0.5, 1.0).unwrap()).count_vertices()
        );
    }

    #[test]
    fn zero_inner_diameter_generates_a_solid_disc_with_outward_winding() {
        let dimensions = BearingDimensions::new(0.50, 0.0).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            Vec3::ZERO,
            Vec3::Y,
            dimensions,
            &mut positions,
            &mut normals,
            &mut indices,
        );

        assert!(positions.iter().any(|position| {
            let position = Vec3::from_array(*position);
            position.x.abs() < 1.0e-6 && position.z.abs() < 1.0e-6
        }));
        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric_normal = (b - a).cross(c - a);
            let expected_normal = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(geometric_normal.dot(expected_normal) > 0.0);
        }
    }

    #[test]
    fn annular_mesh_inner_wall_and_faces_have_outward_winding() {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            Vec3::ZERO,
            Vec3::Y,
            BearingDimensions::default(),
            &mut positions,
            &mut normals,
            &mut indices,
        );

        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric_normal = (b - a).cross(c - a);
            let expected_normal = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(geometric_normal.dot(expected_normal) > 0.0);
        }
    }

    #[test]
    fn unattached_bearing_is_included_in_the_visible_bearing_mesh() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };

        let mesh = combined_bearing_mesh(&graph, &[bearing]);

        assert!(mesh.count_vertices() > 0);
        assert_eq!(graph.bearing_count(), 0);
    }

    #[test]
    fn combined_bearing_mesh_preserves_each_bearings_dimensions() {
        let mut graph = ConstructionGraph::new();
        let specs = [IVec3::new(0, 2, 0), IVec3::new(4, 2, 0)].map(|center| {
            CuboidSpec::new([4, 4, 4], BuildPose::new(center, GridRotation::default())).unwrap()
        });
        let parts = specs.map(|spec| {
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let attached_dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    FaceRef::part(parts[0], FaceKind::PositiveX),
                    FaceRef::part(parts[1], FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, 0.0),
                    Vec3::X,
                )
                .with_dimensions(attached_dimensions),
            ))
            .unwrap();
        let placed_dimensions = BearingDimensions::new(0.40, 0.0).unwrap();
        let placed = PlacedBearing {
            source: FaceRef::part(parts[1], FaceKind::PositiveY),
            anchor: Vec3::new(1.0, 1.0, 0.0),
            dimensions: placed_dimensions,
        };

        let mesh = combined_bearing_mesh(&graph, &[placed]);
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("bearing mesh positions use Float32x3")
        };
        let attached_vertices = 24 * 8;
        let attached_radius = positions[..attached_vertices]
            .iter()
            .map(|position| {
                let offset = Vec3::from_array(*position) - Vec3::new(0.5, 0.5, 0.0);
                (offset - Vec3::X * offset.x).length()
            })
            .fold(0.0, f32::max);
        let placed_radius = positions[attached_vertices..]
            .iter()
            .map(|position| {
                let offset = Vec3::from_array(*position) - placed.anchor;
                (offset - Vec3::Y * offset.y).length()
            })
            .fold(0.0, f32::max);
        assert!(
            (attached_radius
                - (attached_dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
        assert!(
            (placed_radius
                - (placed_dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn reusable_socket_with_multiple_attachments_renders_as_one_ring() {
        const VERTICES_PER_BEARING: usize = 24 * 8;

        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support)).unwrap()
        else {
            unreachable!()
        };
        let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let socket = PlacedBearing {
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        for target in targets {
            graph
                .apply(BuildCommand::AddBearing(
                    mechanic_core::BearingSpec::new(
                        socket.source,
                        FaceRef::part(target, FaceKind::NegativeY),
                        socket.anchor,
                        Vec3::Y,
                    )
                    .with_dimensions(socket.dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(mechanic_core::RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();

        let build_mesh = combined_bearing_mesh(&graph, &[socket]);
        assert_eq!(build_mesh.count_vertices(), VERTICES_PER_BEARING);

        let creation = graph.compile().unwrap();
        assert_eq!(creation.bearings.len(), 1);
        let transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: [
                    compound.root_translation.x,
                    compound.root_translation.y,
                    compound.root_translation.z,
                    0.0,
                ],
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        let simulation_mesh =
            combined_simulation_bearing_mesh(&graph, &creation, &transforms, &[socket]);
        assert_eq!(simulation_mesh.count_vertices(), VERTICES_PER_BEARING);
    }

    #[test]
    fn simulation_bearing_mesh_follows_attached_and_unattached_source_bodies() {
        const VERTICES_PER_BEARING: usize = 24 * 8;

        let mut graph = ConstructionGraph::new();
        let specs = [IVec3::new(0, 2, 0), IVec3::new(4, 2, 0)].map(|center| {
            CuboidSpec::new([4, 4, 4], BuildPose::new(center, GridRotation::default())).unwrap()
        });
        let parts = specs.map(|spec| {
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let attached_dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    FaceRef::part(parts[0], FaceKind::PositiveX),
                    FaceRef::part(parts[1], FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, 0.0),
                    Vec3::X,
                )
                .with_dimensions(attached_dimensions),
            ))
            .unwrap();
        let placed = PlacedBearing {
            source: FaceRef::part(parts[0], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 1.0, 0.0),
            dimensions: BearingDimensions::new(0.40, 0.10).unwrap(),
        };
        let creation = graph.compile().unwrap();
        let source_compound = creation
            .part_to_compound
            .iter()
            .find_map(|&(part, compound)| (part == parts[0]).then_some(compound))
            .unwrap();
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let mut transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: [
                    compound.root_translation.x,
                    compound.root_translation.y,
                    compound.root_translation.z,
                    0.0,
                ],
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        transforms[source_compound as usize] = GpuTransform {
            position: [3.0, 4.0, 5.0, 0.0],
            rotation: rotation.to_array(),
        };

        let mesh = combined_simulation_bearing_mesh(&graph, &creation, &transforms, &[placed]);
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("bearing mesh positions use Float32x3")
        };
        let attached_anchor = Vec3::new(3.0, 4.5, 5.0);
        let placed_anchor = Vec3::new(2.5, 4.0, 5.0);
        for (vertices, expected_anchor) in [
            (&positions[..VERTICES_PER_BEARING], attached_anchor),
            (&positions[VERTICES_PER_BEARING..], placed_anchor),
        ] {
            let centroid = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .sum::<Vec3>()
                / 192.0;
            assert!(centroid.abs_diff_eq(expected_anchor, 1.0e-5));
        }
    }

    #[test]
    fn joint_xray_is_build_only_and_requires_a_bearing() {
        assert!(joint_xray_is_visible(Tool::Controller, false, 1));
        assert!(joint_xray_is_visible(Tool::Connector, false, 1));
        assert!(!joint_xray_is_visible(Tool::Controller, true, 1));
        assert!(!joint_xray_is_visible(Tool::Controller, false, 0));
        assert!(!joint_xray_is_visible(Tool::Block, false, 1));
    }

    fn hinged_pair_with_control_block(reversed: bool) -> ConstructionGraph {
        let mut graph = ConstructionGraph::new();
        let spawn = |graph: &mut ConstructionGraph, x: i32| {
            let BuildOutcome::Spawned(id) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 4, 4],
                        BuildPose::new(bevy::prelude::IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            id
        };
        let left = spawn(&mut graph, 0);
        let right = spawn(&mut graph, 4);
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                FaceRef::part(left, FaceKind::PositiveX),
                FaceRef::part(right, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(bevy::prelude::IVec3::new(0, 12, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut link = DriveLinkSpec::new(controller, bearing);
        link.reversed = reversed;
        link.limits = DriveLimits::new(2.0, 10.0, Some((-0.5, 0.5))).unwrap();
        link.program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(2.0)).unwrap()], false).unwrap();
        graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
        graph
    }

    #[test]
    fn authored_parts_render_in_textured_meshes_not_the_construction_mesh() {
        let mut graph = hinged_pair_with_control_block(false);
        for (kind, x) in [(EngineKind::Gas, 20), (EngineKind::Electric, 24)] {
            graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    kind,
                    BuildPose::new(IVec3::new(x, 12, 0), GridRotation::default()),
                )))
                .unwrap();
        }
        graph
            .apply(BuildCommand::SpawnServo(ServoSpec::new(BuildPose::new(
                IVec3::new(28, 12, 0),
                GridRotation::default(),
            ))))
            .unwrap();
        graph
            .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::new(
                IVec3::new(32, 12, 0),
                GridRotation::default(),
            ))))
            .unwrap();
        graph
            .apply(BuildCommand::SpawnInput(InputSpec::new(BuildPose::new(
                IVec3::new(36, 12, 0),
                GridRotation::default(),
            ))))
            .unwrap();
        let construction = super::combined_construction_mesh(&graph);
        let controllers = combined_controller_mesh(&graph);
        let gas = combined_authored_construction_mesh(&graph, AuthoredPart::GasEngine);
        let electric = combined_authored_construction_mesh(&graph, AuthoredPart::ElectricEngine);
        let servo = combined_authored_construction_mesh(&graph, AuthoredPart::Servo);
        let seat = combined_authored_construction_mesh(&graph, AuthoredPart::Seat);
        let input = combined_authored_construction_mesh(&graph, AuthoredPart::Input);

        // Two hinged blocks remain in the construction mesh; every authored part
        // has an independent batch so its material can use its own texture set.
        assert_eq!(positions(&construction).len(), 24 * 2);
        assert_eq!(positions(&controllers).len(), 24);
        assert_eq!(positions(&gas).len(), 24);
        assert_eq!(positions(&electric).len(), 24);
        assert_eq!(positions(&servo).len(), 24);
        assert_eq!(positions(&seat).len(), 24);
        assert_eq!(positions(&input).len(), 24);
        for mesh in [&controllers, &gas, &electric, &servo, &seat, &input] {
            assert_eq!(
                mesh.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().len(),
                positions(mesh).len()
            );
            assert_eq!(
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap().len(),
                positions(mesh).len()
            );
        }
    }

    #[test]
    fn authored_preview_keeps_the_machine_uvs_and_texture_maps() {
        let mesh = single_authored_part_mesh(AuthoredPart::GasEngine);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("authored preview mesh must have float2 UVs")
        };
        assert_eq!(uvs, &authored_uvs(AuthoredPart::GasEngine));

        let texture = Handle::<Image>::default();
        let material = authored_preview_material(
            StandardMaterial {
                base_color_texture: Some(texture.clone()),
                normal_map_texture: Some(texture.clone()),
                metallic_roughness_texture: Some(texture.clone()),
                emissive_texture: Some(texture.clone()),
                ..Default::default()
            },
            Color::srgba(1.0, 1.0, 1.0, 0.46),
        );
        assert_eq!(material.alpha_mode, AlphaMode::Blend);
        assert_eq!(material.base_color_texture, Some(texture.clone()));
        assert_eq!(material.normal_map_texture, Some(texture.clone()));
        assert_eq!(material.metallic_roughness_texture, Some(texture.clone()));
        assert_eq!(material.emissive_texture, Some(texture));
    }

    #[test]
    fn authored_uvs_assign_each_controller_atlas_tile_to_its_named_face() {
        let uvs = authored_uvs(AuthoredPart::Controller);
        let bounds = |face: usize| {
            uvs[face * 4..face * 4 + 4].iter().fold(
                ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
                |(minimum, maximum), uv| {
                    (
                        [minimum[0].min(uv[0]), minimum[1].min(uv[1])],
                        [maximum[0].max(uv[0]), maximum[1].max(uv[1])],
                    )
                },
            )
        };

        // Vertex groups are +X, -X, +Y, -Y, +Z, -Z. These rectangles are the
        // labelled tiles in controller_reference.png.
        assert_eq!(bounds(0), ([0.0, 0.5], [0.25, 1.0]));
        assert_eq!(bounds(1), ([0.25, 0.5], [0.5, 1.0]));
        assert_eq!(bounds(2), ([0.5, 0.5], [1.0, 0.75]));
        assert_eq!(bounds(3), ([0.5, 0.75], [1.0, 1.0]));
        assert_eq!(bounds(4), ([0.0, 0.0], [0.5, 0.5]));
        assert_eq!(bounds(5), ([0.5, 0.0], [1.0, 0.5]));

        let approximately = |actual: [f32; 2], expected: [f32; 2]| {
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-6)
        };
        assert!(approximately(
            authored_uvs(AuthoredPart::GasEngine)[0],
            [0.0, 0.0]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::ElectricEngine)[0],
            [0.666_667, 0.0]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::Servo)[0],
            [0.666_667, 0.5]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::Seat)[0],
            [0.5, 0.75]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::Input)[0],
            [0.0, 0.75]
        ));
    }

    #[test]
    fn input_uvs_do_not_fold_either_triangle_of_a_face() {
        let uvs = authored_uvs(AuthoredPart::Input);
        let signed_area = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        };

        for face in uvs.chunks_exact(4) {
            let first = signed_area(face[0], face[1], face[2]);
            let second = signed_area(face[0], face[2], face[3]);
            assert!(
                first * second > 0.0,
                "both triangles must map the same way around the atlas tile"
            );
        }
    }

    #[test]
    fn empty_logical_batches_keep_an_invisible_gpu_allocation() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap();

        let logical = super::combined_construction_mesh(&graph);
        assert_eq!(logical.count_vertices(), 0);

        let allocated = renderable_mesh(logical);
        assert_eq!(allocated.count_vertices(), 3);
        assert!(
            positions(&allocated)
                .iter()
                .all(|position| position.length_squared() <= f32::EPSILON)
        );
        assert_eq!(allocated.indices().unwrap().len(), 3);
    }

    #[test]
    fn drive_overlay_is_empty_without_a_wire_and_mirrors_the_spin_direction() {
        let mut graph = ConstructionGraph::new();
        assert!(
            positions(&combined_drive_xray_mesh(
                &graph,
                &[],
                &DriveSequencer::default()
            ))
            .is_empty()
        );

        graph = hinged_pair_with_control_block(false);
        let forward = positions(&combined_drive_xray_mesh(
            &graph,
            &[],
            &DriveSequencer::default(),
        ));
        assert!(!forward.is_empty());

        let reversed = positions(&combined_drive_xray_mesh(
            &hinged_pair_with_control_block(true),
            &[],
            &DriveSequencer::default(),
        ));
        assert_eq!(forward.len(), reversed.len());
        // The arc sweeps the other way, so the two overlays are not identical.
        assert!(
            forward
                .iter()
                .zip(&reversed)
                .any(|(left, right)| !left.abs_diff_eq(*right, 1.0e-4))
        );
    }

    #[test]
    fn drive_overlay_shows_for_the_control_block_tools() {
        for tool in [Tool::Controller, Tool::Connector] {
            assert!(joint_xray_is_visible(tool, false, 1));
            // Unlike the bearing rings, the drive overlay does not depend on
            // the mode: it stays up while the simulation runs so a driven joint
            // can be watched moving.
            assert!(!joint_xray_is_visible(tool, true, 1));
            assert!(drive_xray_is_visible(tool, 1));
            assert!(!drive_xray_is_visible(tool, 0));
        }
        assert!(!drive_xray_is_visible(Tool::Block, 1));
    }

    #[test]
    fn unchanged_bearing_preview_dimensions_do_not_rebuild_the_mesh() {
        let mut rendered = BearingDimensions::default();
        assert!(!bearing_preview_dimensions_changed(
            &mut rendered,
            BearingDimensions::default(),
        ));
        let custom = BearingDimensions::new(0.80, 0.20).unwrap();
        assert!(bearing_preview_dimensions_changed(&mut rendered, custom));
        assert!(!bearing_preview_dimensions_changed(&mut rendered, custom));
    }
}

#[cfg(test)]
mod interaction_tests {
    use bevy::{
        input::keyboard::Key,
        prelude::{App, ButtonInput, IVec3, KeyCode, MouseButton, Update, Vec2, Vec3},
    };
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec, DriveLinkSpec, FaceKind,
        FaceOwner, FaceRef, GridRotation, PartId, PendingOperation, RigidLinkSpec,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        AUTHORED_ORIENTATION_COUNT, AUTHORED_ORIENTATIONS, AppSimulation, BearingDimensionTarget,
        BearingToolSettings, BlockAttachment, BlockDrag, CylinderDimensionTarget,
        CylinderToolSettings, EditorGraph, EditorHistory, EditorState, HAMMER_CHARGE_SECONDS,
        HAMMER_MAX_IMPULSE, HAMMER_MIN_IMPULSE, HistoryAction, PlacedBearing, PlacementPlane,
        PointerSample, SelectedTool, SimulationShortcut, SurfaceHit, Tool,
        adjusted_bearing_dimensions, adjusted_cylinder_dimensions, apply_history_action,
        bearing_attachment_candidate, bearing_attachment_is_highlighted, block_sheet_specs,
        candidate_from_hit, connect_drive_wire, delete_sheet_parts, disconnect_drive_wires,
        hammer_delivery, hammer_impulse_magnitude, hammer_point_travel, handle_block_actions,
        handle_build_actions, handle_shortcuts, handle_tool_change, help_toggle_requested,
        raycast_construction, raycast_placed_bearing_discs, raycast_placed_bearings,
        raycast_simulation, refresh_block_drag, refresh_tool_preview,
        requested_bearing_dimension_adjustment, requested_cylinder_dimension_adjustment,
        requested_simulation_shortcut, rigid_body_parts, stage_part_deletion_preserving_bearings,
        tool_status_line, wire_drag_step,
    };
    use crate::{WireConnection, WireDrag, WireDragStep, WireEnd, ui::UiInput};

    fn pointer_sample(cursor: Vec2, ray_origin: Vec3, ray_direction: Vec3) -> PointerSample {
        PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        }
    }

    #[test]
    fn question_mark_is_what_asks_for_the_help_panel() {
        let mut keyboard = ButtonInput::default();
        assert!(!help_toggle_requested(&keyboard));

        keyboard.press(Key::Character("?".into()));
        assert!(help_toggle_requested(&keyboard));

        keyboard.clear();
        keyboard.press(Key::Character("a".into()));
        assert!(
            !help_toggle_requested(&keyboard),
            "an ordinary letter is not a request for help",
        );
    }

    #[test]
    fn q_cycles_every_authored_tool_through_all_grid_orientations() {
        for tool in [Tool::Controller, Tool::GasEngine, Tool::ElectricEngine] {
            let mut app = App::new();
            app.init_resource::<ButtonInput<KeyCode>>()
                .init_resource::<EditorGraph>()
                .init_resource::<EditorState>()
                .init_resource::<AppSimulation>()
                .init_resource::<UiInput>()
                .insert_resource(SelectedTool(tool))
                .add_systems(Update, handle_shortcuts);

            for expected in (1..AUTHORED_ORIENTATION_COUNT).chain(std::iter::once(0)) {
                app.world_mut()
                    .resource_mut::<ButtonInput<KeyCode>>()
                    .press(KeyCode::KeyQ);
                app.update();
                assert_eq!(
                    app.world().resource::<EditorState>().authored_orientation,
                    expected,
                    "{} should rotate on Q",
                    tool.label()
                );
                app.world_mut()
                    .resource_mut::<ButtonInput<KeyCode>>()
                    .reset(KeyCode::KeyQ);
            }
        }
    }

    #[test]
    fn authored_orientation_cycle_contains_all_24_cube_orientations_once() {
        let mut signatures = Vec::new();
        for rotation in AUTHORED_ORIENTATIONS {
            let quaternion = rotation.quaternion();
            let signature = [Vec3::X, Vec3::Y, Vec3::Z]
                .map(|axis| (quaternion * axis).round().as_ivec3().to_array());
            assert!(!signatures.contains(&signature));
            signatures.push(signature);
        }
        assert_eq!(signatures.len(), 24);
    }

    #[test]
    fn authored_preview_uses_a_tipped_rotation_selected_with_q() {
        let graph = ConstructionGraph::new();
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            }),
            authored_orientation: 16,
            ..Default::default()
        };

        refresh_tool_preview(&graph, &mut state, Tool::GasEngine);

        let preview = state.preview.expect("gas engine has a ground preview");
        assert_eq!(preview.spec.pose.rotation.quarter_turns_xyz(), [1, 0, 0]);
        let (minimum, maximum) = super::part_world_bounds(preview.spec.into());
        assert!((maximum.y - minimum.y - 0.75).abs() < 1.0e-6);
        assert!((maximum.z - minimum.z - 0.50).abs() < 1.0e-6);
    }

    #[test]
    fn space_toggles_playback_and_shift_space_restarts() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(KeyCode::Space);
        assert_eq!(
            requested_simulation_shortcut(&keyboard),
            Some(SimulationShortcut::TogglePlayback)
        );

        keyboard.press(KeyCode::ShiftLeft);
        assert_eq!(
            requested_simulation_shortcut(&keyboard),
            Some(SimulationShortcut::Restart)
        );
    }

    #[test]
    fn bearing_shortcuts_are_gated_and_adjust_the_requested_diameter() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(KeyCode::ArrowRight);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Bearing, false, false),
            Some((BearingDimensionTarget::Outer, 1))
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Block, false, false),
            None
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Bearing, true, false),
            None
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Bearing, false, true),
            None
        );

        keyboard.reset_all();
        keyboard.press(KeyCode::ArrowUp);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Bearing, false, false),
            None
        );

        let increased = adjusted_bearing_dimensions(
            BearingDimensions::default(),
            BearingDimensionTarget::Outer,
            1,
        );
        assert!((increased.outer_diameter() - 0.30).abs() < 1.0e-6);
        assert!((increased.inner_diameter() - 0.10).abs() < f32::EPSILON);

        keyboard.reset_all();
        keyboard.press(KeyCode::ArrowRight);
        keyboard.press(KeyCode::ShiftLeft);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Bearing, false, false),
            Some((BearingDimensionTarget::Inner, 1))
        );
    }

    #[test]
    fn cylinder_shortcuts_adjust_and_clamp_without_graph_history() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(KeyCode::ArrowRight);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Tool::Cylinder, false, false),
            Some((CylinderDimensionTarget::Outer, 1))
        );
        keyboard.press(KeyCode::ShiftLeft);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Tool::Cylinder, false, false),
            Some((CylinderDimensionTarget::Inner, 1))
        );
        keyboard.release(KeyCode::ArrowRight);
        keyboard.press(KeyCode::ArrowDown);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Tool::Cylinder, false, false),
            Some((CylinderDimensionTarget::Sweep, -1))
        );
        assert!(
            requested_cylinder_dimension_adjustment(&keyboard, Tool::Block, false, false).is_none()
        );

        let dimensions = CylinderDimensions::new(0.25, 0.20, 0.25).unwrap();
        let reduced = adjusted_cylinder_dimensions(dimensions, CylinderDimensionTarget::Outer, -1);
        assert!((reduced.outer_diameter() - 0.20).abs() < 1.0e-6);
        assert!((reduced.inner_diameter() - 0.15).abs() < 1.0e-6);
        let minimum = adjusted_cylinder_dimensions(reduced, CylinderDimensionTarget::Length, -1);
        assert!((minimum.axial_length() - 0.25).abs() < f32::EPSILON);
        let slice = adjusted_cylinder_dimensions(minimum, CylinderDimensionTarget::Sweep, -1);
        assert_eq!(slice.sweep_angle_degrees(), 345);
        let minimum_sweep = (0..30).fold(slice, |dimensions, _| {
            adjusted_cylinder_dimensions(dimensions, CylinderDimensionTarget::Sweep, -1)
        });
        assert_eq!(minimum_sweep.sweep_angle_degrees(), 15);

        let graph = ConstructionGraph::new();
        let history = EditorHistory::default();
        let settings = CylinderToolSettings {
            dimensions: minimum,
        };
        assert_eq!(graph.part_count(), 0);
        assert!(history.undo.is_empty() && history.redo.is_empty());
        assert_eq!(settings.dimensions, minimum);
    }

    #[test]
    fn bearing_adjustments_clamp_and_remain_outside_history() {
        let mut settings = BearingToolSettings {
            dimensions: BearingDimensions::new(0.20, 0.15).unwrap(),
        };
        let history = EditorHistory::default();
        settings.dimensions =
            adjusted_bearing_dimensions(settings.dimensions, BearingDimensionTarget::Outer, -1);
        assert!((settings.dimensions.outer_diameter() - 0.15).abs() < 1.0e-6);
        assert!((settings.dimensions.inner_diameter() - 0.10).abs() < 1.0e-6);
        assert!(history.undo.is_empty());
        assert!(history.redo.is_empty());

        let minimum = adjusted_bearing_dimensions(
            BearingDimensions::new(0.05, 0.0).unwrap(),
            BearingDimensionTarget::Outer,
            -1,
        );
        assert_eq!(minimum, BearingDimensions::new(0.05, 0.0).unwrap());
        let maximum_inner = adjusted_bearing_dimensions(
            BearingDimensions::default(),
            BearingDimensionTarget::Inner,
            1,
        );
        assert!((maximum_inner.inner_diameter() - 0.15).abs() < 1.0e-6);

        let hud = tool_status_line(
            Tool::Bearing,
            settings.dimensions,
            CylinderDimensions::default(),
            None,
        );
        assert!(hud.contains("Outer: 0.15 m"));
        assert!(hud.contains("Inner: 0.10 m"));
        assert!(hud.contains("Shift+←/→"));
    }

    #[test]
    fn hammer_charge_is_monotonic_and_clamped() {
        let tap = hammer_impulse_magnitude(0.0);
        let half = hammer_impulse_magnitude(HAMMER_CHARGE_SECONDS * 0.5);
        let full = hammer_impulse_magnitude(HAMMER_CHARGE_SECONDS);
        assert!((tap - HAMMER_MIN_IMPULSE).abs() < f32::EPSILON);
        assert!(tap < half && half < full);
        assert!((full - HAMMER_MAX_IMPULSE).abs() < f32::EPSILON);
        assert!((full - 4_000.0).abs() < f32::EPSILON);
        assert!((hammer_impulse_magnitude(100.0) - HAMMER_MAX_IMPULSE).abs() < f32::EPSILON);
    }

    #[test]
    fn hard_hammer_impulses_are_delivered_in_collision_safe_steps() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(0, 1, 0), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let creation = graph.compile().unwrap();
        let root = creation.compounds[0].root_translation;
        let transform = GpuTransform {
            position: [root.x, root.y, root.z, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        let impulse = Vec3::X * HAMMER_MAX_IMPULSE;
        let local_point = Vec3::new(0.0, 0.125, 0.0);

        let (ticks, impulse_per_tick) =
            hammer_delivery(&creation, transform, 0, local_point, impulse);

        assert!(ticks > 1);
        assert!(ticks <= super::HAMMER_MAX_DELIVERY_TICKS);
        assert!(impulse_per_tick.length() * f32::from(ticks) < impulse.length());
        assert!(
            hammer_point_travel(&creation, transform, 0, local_point, impulse_per_tick)
                <= super::HAMMER_MAX_POINT_TRAVEL_PER_TICK + f32::EPSILON
        );

        let mut heavy_graph = ConstructionGraph::new();
        let heavy = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        heavy_graph.apply(BuildCommand::Spawn(heavy)).unwrap();
        let heavy_creation = heavy_graph.compile().unwrap();
        let heavy_root = heavy_creation.compounds[0].root_translation;
        let heavy_transform = GpuTransform {
            position: [heavy_root.x, heavy_root.y, heavy_root.z, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        let (heavy_ticks, heavy_impulse_per_tick) =
            hammer_delivery(&heavy_creation, heavy_transform, 0, Vec3::ZERO, impulse);
        assert!(
            (heavy_impulse_per_tick.length() * f32::from(heavy_ticks) - impulse.length()).abs()
                < 1.0e-3
        );
    }

    #[test]
    fn hammer_raycast_uses_the_current_simulated_pose() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(_) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let transforms = [GpuTransform {
            position: [5.0, 1.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        let hit = raycast_simulation(
            &graph,
            &creation,
            &transforms,
            Vec3::new(5.0, 1.0, 5.0),
            Vec3::NEG_Z,
        )
        .unwrap();
        assert_eq!(hit.body_index, 0);
        assert!(hit.point.abs_diff_eq(Vec3::new(5.0, 1.0, 0.5), 1.0e-5));
        assert!(
            raycast_simulation(
                &graph,
                &creation,
                &transforms,
                Vec3::new(0.0, 1.0, 5.0),
                Vec3::NEG_Z,
            )
            .is_none()
        );
    }

    #[test]
    fn hammer_raycast_respects_a_cylinder_slice() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let spec = CylinderSpec::new(
            dimensions,
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        );
        let BuildOutcome::Spawned(_) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
        else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let root = creation.compounds[0].root_translation;
        let transforms = [GpuTransform {
            position: [root.x, root.y, root.z, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        assert!(
            raycast_simulation(
                &graph,
                &creation,
                &transforms,
                Vec3::new(0.3, 2.0, 0.0),
                Vec3::NEG_Y,
            )
            .is_some()
        );
        assert!(
            raycast_simulation(
                &graph,
                &creation,
                &transforms,
                Vec3::new(-0.3, 2.0, 0.0),
                Vec3::NEG_Y,
            )
            .is_none()
        );
    }

    #[test]
    fn selecting_a_tool_cancels_pending_editor_state() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::BeginPending(PendingOperation::Weld(
                FaceRef::part(part, FaceKind::PositiveY),
            )))
            .unwrap();

        let mut app = App::new();
        app.insert_resource(EditorGraph(graph))
            .insert_resource(EditorState::default())
            .insert_resource(SelectedTool(Tool::Bearing))
            .add_systems(Update, handle_tool_change);

        app.update();

        assert!(app.world().resource::<EditorGraph>().0.pending().is_none());
    }

    #[test]
    fn weld_highlight_contains_the_entire_rigid_body_only() {
        let mut graph = ConstructionGraph::new();
        let specs = [
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
            )
            .unwrap(),
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 6, 0), GridRotation::default()),
            )
            .unwrap(),
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 10, 0), GridRotation::default()),
            )
            .unwrap(),
        ];
        let parts = specs.map(|spec| {
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        graph
            .apply(BuildCommand::Weld(mechanic_core::WeldSpec {
                first: FaceRef::part(parts[0], FaceKind::PositiveY),
                second: FaceRef::part(parts[1], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                FaceRef::part(parts[1], FaceKind::PositiveY),
                FaceRef::part(parts[2], FaceKind::NegativeY),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::Y,
            )))
            .unwrap();

        assert_eq!(rigid_body_parts(&graph, parts[0]), parts[..2]);
        assert_eq!(rigid_body_parts(&graph, parts[2]), vec![parts[2]]);
    }

    #[test]
    fn block_click_places_on_release_through_drag_path() {
        let mut graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let press = pointer_sample(
            Vec2::new(320.0, 240.0),
            Vec3::new(-0.3, 2.0, -0.2),
            Vec3::new(0.2, -1.0, 0.3),
        );
        let mut state = EditorState {
            hovered: Some(hit),
            preview: Some(candidate),
            pointer_position: Some(press.cursor),
            pointer_ray: Some((press.ray_origin, press.ray_direction)),
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();

        mouse.press(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 0);
        assert!(state.block_drag.is_some());
        refresh_block_drag(
            &graph,
            &mut state,
            press.cursor,
            press.ray_origin,
            press.ray_direction,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().specs.len(), 1);

        mouse.clear();
        mouse.release(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 1);
        assert!(state.block_drag.is_none());
        assert_eq!(history.undo.len(), 1);
    }

    #[test]
    fn block_drag_dead_zone_and_motion_are_relative_to_the_press() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y);
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                press,
                plane: PlacementPlane::Xz,
                last_endpoint: None,
                specs: vec![candidate.spec],
                error: None,
            }),
            ..Default::default()
        };

        refresh_block_drag(
            &graph,
            &mut state,
            Vec2::new(4.99, 0.0),
            Vec3::new(4.0, 2.0, 4.0),
            Vec3::NEG_Y,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().specs.len(), 1);

        for (target, expected) in [
            (Vec3::new(0.50, 2.0, 0.25), 6),
            (Vec3::new(-0.50, 2.0, -0.25), 6),
            (Vec3::new(0.50, 2.0, 0.0), 3),
            (Vec3::new(0.0, 2.0, -0.50), 3),
        ] {
            refresh_block_drag(
                &graph,
                &mut state,
                Vec2::new(10.0, 0.0),
                target,
                Vec3::NEG_Y,
            );
            assert_eq!(state.block_drag.as_ref().unwrap().specs.len(), expected);
        }
    }

    #[test]
    fn cycling_the_plane_without_pointer_motion_stays_one_by_one() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let press = pointer_sample(
            Vec2::new(100.0, 100.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::new(0.0, -1.0, -1.0),
        );
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                press,
                plane: PlacementPlane::Xz,
                last_endpoint: None,
                specs: vec![candidate.spec],
                error: None,
            }),
            ..Default::default()
        };

        let drag = state.block_drag.as_mut().unwrap();
        drag.plane = drag.plane.cycle();
        assert_eq!(drag.plane, PlacementPlane::Xy);
        refresh_block_drag(
            &graph,
            &mut state,
            press.cursor,
            press.ray_origin,
            press.ray_direction,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().specs.len(), 1);

        refresh_block_drag(
            &graph,
            &mut state,
            Vec2::new(110.0, 100.0),
            press.ray_origin + Vec3::new(0.5, 0.0, 0.0),
            press.ray_direction,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().specs.len(), 3);
    }

    #[test]
    fn dragged_placement_is_one_atomic_history_step() {
        let mut graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let endpoint = candidate.spec.pose.translation_half_units() + IVec3::new(4, 0, 2);
        let specs = block_sheet_specs(candidate.spec, endpoint, PlacementPlane::Xz).unwrap();
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
                plane: PlacementPlane::Xz,
                last_endpoint: Some((PlacementPlane::Xz, endpoint)),
                specs,
                error: None,
            }),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();
        mouse.press(MouseButton::Left);
        mouse.clear();
        mouse.release(MouseButton::Left);

        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(graph.part_count(), 6);
        assert_eq!(graph.weld_count(), 13);
        assert_eq!(history.undo.len(), 1);
        apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );
        assert_eq!(graph.part_count(), 0);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn wiring_picks_a_bearing_through_the_hole_the_ring_pick_misses() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };

        // Straight down the axis passes through the hole, and whatever is
        // threaded through it, so the ring pick finds nothing there.
        let axis = Vec3::new(0.0, 3.0, 0.0);
        assert!(raycast_placed_bearings(&graph, &[bearing], axis, Vec3::NEG_Y).is_none());
        assert_eq!(
            raycast_placed_bearing_discs(&graph, &[bearing], axis, Vec3::NEG_Y).map(|hit| hit.0),
            Some(0)
        );

        // Past the rim it still misses, so the disc does not swallow the block.
        let outside = Vec3::new(bearing.dimensions.outer_diameter(), 3.0, 0.0);
        assert!(raycast_placed_bearing_discs(&graph, &[bearing], outside, Vec3::NEG_Y).is_none());
    }

    #[test]
    fn placed_bearing_is_picked_before_support_and_attaches_on_release() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };
        let mut state = EditorState {
            placed_bearings: vec![bearing],
            hovered_bearing: Some(0),
            attachment_bearing: Some(0),
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::new(0.1, 3.0, 0.0), Vec3::NEG_Y)),
            preview: Some(bearing_attachment_candidate(
                &graph,
                bearing.source,
                bearing.anchor,
            )),
            ..Default::default()
        };

        let origin = Vec3::new(0.1, 3.0, 0.0);
        let (_, bearing_distance) =
            raycast_placed_bearings(&graph, &state.placed_bearings, origin, Vec3::NEG_Y).unwrap();
        let support_distance = raycast_construction(&graph, origin, Vec3::NEG_Y)
            .unwrap()
            .distance;
        assert!(bearing_distance < support_distance);
        assert!(
            raycast_placed_bearings(
                &graph,
                &state.placed_bearings,
                Vec3::new(0.0, 3.0, 0.0),
                Vec3::NEG_Y,
            )
            .is_none()
        );
        let tiny_hole = PlacedBearing {
            dimensions: BearingDimensions::new(0.25, 0.001).unwrap(),
            ..bearing
        };
        assert!(
            raycast_placed_bearings(&graph, &[tiny_hole], Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y,)
                .is_none()
        );
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);

        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();
        mouse.press(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings.len(), 1);
        assert!(state.block_drag.is_some());

        mouse.clear();
        mouse.release(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(state.placed_bearings, vec![bearing]);
        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn oversized_bearing_claims_offset_block_preview_and_highlights_attachment() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.36, 1.0, 0.0),
                face: bearing.source,
            }),
            placed_bearings: vec![bearing],
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::new(0.36, 3.0, 0.0), Vec3::NEG_Y)),
            ..Default::default()
        };

        refresh_tool_preview(&graph, &mut state, Tool::Block);

        assert_eq!(state.hovered_bearing, None);
        assert_eq!(state.attachment_bearing, Some(0));
        assert!(state.preview_error.is_none());
        assert!(bearing_attachment_is_highlighted(
            Tool::Block,
            state.attachment_bearing,
            state.preview_error.as_ref(),
        ));
        let preview = state.preview.unwrap();
        assert!((preview.spec.pose.translation().x - 0.25).abs() < 1.0e-6);

        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();
        mouse.press(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        mouse.clear();
        mouse.release(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(state.placed_bearings, vec![bearing]);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            bearing.dimensions
        );
    }

    #[test]
    fn right_click_through_bearing_hole_deletes_block_but_keeps_bearing() {
        let mut graph = ConstructionGraph::new();
        let parts = [IVec3::new(0, 1, 0), IVec3::new(2, 1, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let center_face = FaceRef::part(parts[0], FaceKind::PositiveY);
        let bearing = PlacedBearing {
            source: FaceRef::part(parts[1], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 0.25, 0.0),
            dimensions: BearingDimensions::new(0.75, 0.40).unwrap(),
        };
        let state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: bearing.anchor,
                face: center_face,
            }),
            hovered_bearing: None,
            attachment_bearing: Some(0),
            placed_bearings: vec![bearing],
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        mouse.press(MouseButton::Right);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let state = app.world().resource::<EditorState>();
            assert!(state.delete_target.is_none());
            assert!(state.delete_drag.is_some());
        }
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
            mouse.clear();
            mouse.release(MouseButton::Right);
        }
        app.update();

        let graph = app.world().resource::<EditorGraph>();
        let state = app.world().resource::<EditorState>();
        assert!(graph.0.part(parts[0]).is_none());
        assert!(graph.0.part(parts[1]).is_some());
        assert_eq!(state.placed_bearings, vec![bearing]);
    }

    #[test]
    fn deleting_current_support_rehomes_bearing_to_remaining_ring_support() {
        let mut graph = ConstructionGraph::new();
        let supports = [IVec3::new(-1, 1, 0), IVec3::new(1, 1, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let target_spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(target) = graph.apply(BuildCommand::Spawn(target_spec)).unwrap()
        else {
            unreachable!()
        };
        let socket = PlacedBearing {
            source: FaceRef::part(supports[0], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 0.25, 0.0),
            dimensions: BearingDimensions::new(0.50, 0.10).unwrap(),
        };
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    socket.source,
                    FaceRef::part(target, FaceKind::NegativeY),
                    socket.anchor,
                    Vec3::Y,
                )
                .with_dimensions(socket.dimensions),
            ))
            .unwrap();

        let (graph, sockets, migrated) =
            stage_part_deletion_preserving_bearings(&graph, &[socket], &[supports[0]]).unwrap();

        assert_eq!(migrated, 1);
        assert!(graph.part(supports[0]).is_none());
        assert!(graph.part(supports[1]).is_some());
        assert_eq!(sockets.len(), 1);
        assert_eq!(
            sockets[0].source,
            FaceRef::part(supports[1], FaceKind::PositiveY)
        );
        let bearing = graph.bearings().next().unwrap().1;
        assert_eq!(bearing.source, sockets[0].source);
        assert_eq!(bearing.target, FaceRef::part(target, FaceKind::NegativeY));
        assert_eq!(graph.compile().unwrap().bearings.len(), 1);

        let (graph, sockets, migrated) =
            stage_part_deletion_preserving_bearings(&graph, &sockets, &[supports[1]]).unwrap();
        assert_eq!(migrated, 0);
        assert!(sockets.is_empty());
        assert_eq!(graph.bearing_count(), 0);
        assert!(graph.part(target).is_some());
    }

    #[test]
    fn deleting_reusable_socket_removes_all_of_its_joint_attachments() {
        let mut graph = ConstructionGraph::new();
        let support_spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) =
            graph.apply(BuildCommand::Spawn(support_spec)).unwrap()
        else {
            unreachable!()
        };
        let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
            let target_spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(target) =
                graph.apply(BuildCommand::Spawn(target_spec)).unwrap()
            else {
                unreachable!()
            };
            target
        });
        let socket = PlacedBearing {
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        for target in targets {
            graph
                .apply(BuildCommand::AddBearing(
                    mechanic_core::BearingSpec::new(
                        socket.source,
                        FaceRef::part(target, FaceKind::NegativeY),
                        socket.anchor,
                        Vec3::Y,
                    )
                    .with_dimensions(socket.dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();
        let state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        mouse.press(MouseButton::Right);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<MouseButton>>();
            mouse.clear();
            mouse.release(MouseButton::Right);
        }
        app.update();

        let graph = app.world().resource::<EditorGraph>();
        let state = app.world().resource::<EditorState>();
        assert_eq!(graph.0.part_count(), 3);
        assert_eq!(graph.0.bearing_count(), 0);
        assert_eq!(graph.0.rigid_link_count(), 0);
        assert!(state.placed_bearings.is_empty());
    }

    #[test]
    fn delete_drag_selects_only_the_rectangular_plane() {
        let mut graph = ConstructionGraph::new();
        let centers = [
            IVec3::new(1, 1, 1),
            IVec3::new(3, 1, 1),
            IVec3::new(1, 1, 3),
            IVec3::new(3, 1, 3),
            IVec3::new(1, 3, 1),
        ];
        let mut parts = Vec::new();
        for center in centers {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        let start = graph.part(parts[0]).copied().unwrap().as_cuboid().unwrap();

        let selected =
            delete_sheet_parts(&graph, start, IVec3::new(3, 1, 3), PlacementPlane::Xz).unwrap();

        assert_eq!(selected.len(), 4);
        assert!(!selected.contains(&parts[4]));
    }

    fn wired_socket_graph() -> (ConstructionGraph, PlacedBearing, PartId) {
        let mut graph = ConstructionGraph::new();
        let spawn = |graph: &mut ConstructionGraph, x: i32| {
            let BuildOutcome::Spawned(id) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 4, 4],
                        BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            id
        };
        let left = spawn(&mut graph, 0);
        let right = spawn(&mut graph, 4);
        let source = FaceRef::part(left, FaceKind::PositiveX);
        let anchor = Vec3::new(0.5, 0.5, 0.0);
        graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                source,
                FaceRef::part(right, FaceKind::NegativeX),
                anchor,
                Vec3::X,
            )))
            .unwrap();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let socket = PlacedBearing {
            source,
            anchor,
            dimensions: BearingDimensions::default(),
        };
        (graph, socket, controller)
    }

    #[test]
    fn dragging_a_control_block_onto_a_bearing_wires_every_row_of_that_socket() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let block = WireEnd::Controller(controller);

        assert_eq!(
            wire_drag_step(None, Some(block), true),
            WireDragStep::Begin(block)
        );
        assert_eq!(
            wire_drag_step(
                Some(WireDrag {
                    from: block,
                    armed: false
                }),
                Some(WireEnd::Bearing(0)),
                false
            ),
            WireDragStep::Connect(WireConnection::Drive {
                controller,
                bearing: 0
            })
        );

        let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
        assert!(message.contains("Wired"), "{message}");
        assert_eq!(graph.drive_link_count(), 1);
        assert!(state.construction_mesh_dirty);
    }

    #[test]
    fn a_wire_can_be_dragged_from_the_bearing_end_as_well() {
        let (_, _, controller) = wired_socket_graph();
        let drag = Some(WireDrag {
            from: WireEnd::Bearing(0),
            armed: false,
        });
        assert_eq!(
            wire_drag_step(drag, Some(WireEnd::Controller(controller)), false),
            WireDragStep::Connect(WireConnection::Drive {
                controller,
                bearing: 0
            })
        );
        // Two ends of the same kind never pair up.
        assert_eq!(
            wire_drag_step(drag, Some(WireEnd::Bearing(1)), false),
            WireDragStep::Cancel
        );
    }

    #[test]
    fn releasing_where_the_wire_started_leaves_it_armed_for_a_second_click() {
        let (_, _, controller) = wired_socket_graph();
        let block = WireEnd::Controller(controller);
        let drag = WireDrag {
            from: block,
            armed: false,
        };

        assert_eq!(
            wire_drag_step(Some(drag), Some(block), false),
            WireDragStep::Arm
        );
        assert_eq!(
            wire_drag_step(
                Some(WireDrag {
                    armed: true,
                    ..drag
                }),
                Some(WireEnd::Bearing(0)),
                true
            ),
            WireDragStep::Connect(WireConnection::Drive {
                controller,
                bearing: 0
            })
        );
        // Letting go over empty space drops the wire instead.
        assert_eq!(
            wire_drag_step(Some(drag), None, false),
            WireDragStep::Cancel
        );
    }

    #[test]
    fn clicking_a_wired_bearing_again_reverses_it_and_right_click_removes_the_wire() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let bearing = graph.bearings().next().unwrap().0;
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap();
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();

        let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
        assert!(message.contains("Reversed"), "{message}");
        assert_eq!(graph.drive_link_count(), 1);
        assert!(graph.drive_links().next().unwrap().1.reversed);
        assert!(
            graph
                .bearing_drive_link(bearing)
                .is_some_and(|(_, link)| link.reversed)
        );

        let message = disconnect_drive_wires(&mut graph, &mut state, &mut history, socket);
        assert!(message.contains("Removed"), "{message}");
        assert_eq!(graph.drive_link_count(), 0);
    }

    #[test]
    fn wiring_an_unattached_socket_reports_that_it_has_no_joint_yet() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(block) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![PlacedBearing {
                source: FaceRef::part(block, FaceKind::PositiveX),
                anchor: Vec3::new(0.5, 0.5, 0.0),
                dimensions: BearingDimensions::default(),
            }],
            ..Default::default()
        };
        let mut history = EditorHistory::default();

        let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
        assert!(message.contains("Attach a part"), "{message}");
        assert_eq!(graph.drive_link_count(), 0);
    }

    #[test]
    fn control_block_status_line_reports_how_many_bearings_are_wired() {
        let selected = tool_status_line(
            Tool::Controller,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            Some(2),
        );
        assert!(selected.contains("2 bearings wired"), "{selected}");
        assert!(selected.contains("E opens its program"), "{selected}");

        let single = tool_status_line(
            Tool::Connector,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            Some(1),
        );
        assert!(single.contains("1 bearing wired"), "{single}");

        let none = tool_status_line(
            Tool::Controller,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            None,
        );
        assert!(none.contains("No block selected"), "{none}");
    }

    #[test]
    fn the_panel_opens_on_a_hovered_control_block_and_blocks_the_keyboard() {
        let (graph, _, controller) = wired_socket_graph();
        let mut panel = crate::control_panel::ControlPanelState::default();
        assert!(!panel.is_open());
        assert!(!panel.blocks_keyboard());

        panel.open(controller);
        assert_eq!(panel.controller(), Some(controller));
        assert!(panel.blocks_keyboard(), "typing must not fire shortcuts");

        // One row per wired bearing, and none until the block is wired.
        assert!(crate::control_panel::panel_rows(&graph, controller).is_empty());

        panel.close();
        assert!(!panel.is_open());
    }

    #[test]
    fn every_wire_of_one_socket_is_written_by_a_single_row_edit() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);

        let rows = crate::control_panel::panel_rows(&graph, controller);
        assert_eq!(rows.len(), 1, "one socket is one joint row");
        let commands = crate::control_panel::set_row_commands(
            &rows[0],
            mechanic_core::DriveLimits::new(2.0, 30.0, None).unwrap(),
            mechanic_core::DriveProgram::default(),
            mechanic_core::DriveName::new("Tipper arm"),
            mechanic_core::ActuatorAssignment::Unpowered,
        );
        assert_eq!(commands.len(), rows[0].links.len());
        graph.apply_batch(commands).unwrap();
        for (_, link) in graph.controller_links(controller) {
            assert!((link.limits.max_torque_newton_meters() - 30.0).abs() < f32::EPSILON);
        }
    }
}

#[cfg(test)]
mod history_tests {
    use bevy::prelude::{ButtonInput, IVec3, KeyCode, Vec2, Vec3};
    use mechanic_core::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        CuboidSpec, FaceKind, FaceRef, GridRotation, PendingOperation, WeldSpec,
    };

    use super::{
        BlockAttachment, BlockDrag, DeleteDrag, DeleteTarget, EditorHistory, EditorSnapshot,
        EditorState, HISTORY_CAPACITY, HistoryAction, PlacedBearing, PlacementPlane, PointerSample,
        SurfaceHit, apply_history_action, bearing_attachment_candidate, requested_history_action,
        stage_bearing_attachment,
    };

    fn spawn_cube(graph: &mut ConstructionGraph, center: IVec3) -> mechanic_core::PartId {
        let spec =
            CuboidSpec::new([4; 3], BuildPose::new(center, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    }

    #[test]
    fn control_and_command_z_choose_undo_and_shift_redo() {
        for modifier in [
            KeyCode::ControlLeft,
            KeyCode::ControlRight,
            KeyCode::SuperLeft,
            KeyCode::SuperRight,
        ] {
            let mut keyboard = ButtonInput::default();
            keyboard.press(modifier);
            keyboard.press(KeyCode::KeyZ);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Undo)
            );

            keyboard.press(KeyCode::ShiftLeft);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Redo)
            );
        }

        let mut keyboard = ButtonInput::default();
        keyboard.press(KeyCode::ControlLeft);
        keyboard.press(KeyCode::KeyY);
        assert_eq!(requested_history_action(&keyboard), None);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn bearing_attachment_round_trips_exact_ids_and_cancels_transients() {
        let mut graph = ConstructionGraph::new();
        let support = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let socket = PlacedBearing {
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.70, 0.35).unwrap(),
        };
        let mut state = EditorState {
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);
        let candidate = bearing_attachment_candidate(&graph, socket.source, socket.anchor);
        graph = stage_bearing_attachment(
            &graph,
            candidate,
            socket.source,
            socket.anchor,
            socket.dimensions,
        )
        .unwrap();
        history.commit(previous);
        let attached_parts = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        let attached_bearings = graph.bearings().map(|(id, _)| id).collect::<Vec<_>>();
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            socket.dimensions
        );

        graph
            .apply(BuildCommand::BeginPending(PendingOperation::Weld(
                FaceRef::part(support, FaceKind::PositiveX),
            )))
            .unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::Y,
            face: FaceRef::part(support, FaceKind::PositiveY),
        };
        state.hovered = Some(hit);
        state.preview = Some(candidate);
        state.block_drag = Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: hit.face.owner,
            },
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            last_endpoint: None,
            specs: vec![candidate.spec],
            error: None,
        });
        state.delete_drag = Some(DeleteDrag {
            start: graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            plane: PlacementPlane::Xz,
            last_endpoint: None,
            parts: vec![support],
            error: None,
        });
        state.delete_target = Some(DeleteTarget::PlacedBearing(0));

        apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings, vec![socket]);
        assert!(graph.pending().is_none());
        assert!(state.block_drag.is_none());
        assert!(state.delete_drag.is_none());
        assert!(state.delete_target.is_none());
        assert!(state.hovered.is_none());
        assert!(state.preview.is_none());
        assert!(state.construction_mesh_dirty);

        apply_history_action(
            HistoryAction::Redo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );

        assert_eq!(
            graph.parts().map(|(id, _)| id).collect::<Vec<_>>(),
            attached_parts
        );
        assert_eq!(
            graph.bearings().map(|(id, _)| id).collect::<Vec<_>>(),
            attached_bearings
        );
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            socket.dimensions
        );
        assert_eq!(state.placed_bearings, vec![socket]);
    }

    #[test]
    fn dragged_deletion_restores_connections_atomically() {
        let mut graph = ConstructionGraph::new();
        let parts = [
            spawn_cube(&mut graph, IVec3::new(0, 2, 0)),
            spawn_cube(&mut graph, IVec3::new(0, 6, 0)),
            spawn_cube(&mut graph, IVec3::new(0, 10, 0)),
        ];
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::ground(),
                second: FaceRef::part(parts[0], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(parts[0], FaceKind::PositiveY),
                second: FaceRef::part(parts[1], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[1], FaceKind::PositiveY),
                FaceRef::part(parts[2], FaceKind::NegativeY),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::Y,
            )))
            .unwrap();
        let original_part_ids = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        let original_weld_ids = graph.welds().map(|(id, _)| id).collect::<Vec<_>>();
        let original_bearing_ids = graph.bearings().map(|(id, _)| id).collect::<Vec<_>>();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);

        graph
            .apply_batch(parts[..2].iter().copied().map(BuildCommand::Remove))
            .unwrap();
        history.commit(previous);
        assert_eq!(history.undo.len(), 1);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(graph.bearing_count(), 0);

        apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );
        assert_eq!(
            graph.parts().map(|(id, _)| id).collect::<Vec<_>>(),
            original_part_ids
        );
        assert_eq!(
            graph.welds().map(|(id, _)| id).collect::<Vec<_>>(),
            original_weld_ids
        );
        assert_eq!(
            graph.bearings().map(|(id, _)| id).collect::<Vec<_>>(),
            original_bearing_ids
        );
    }

    #[test]
    fn history_is_bounded_and_new_edits_clear_only_redo() {
        let graph = ConstructionGraph::new();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        for _ in 0..=HISTORY_CAPACITY {
            history.commit(EditorSnapshot::capture(&graph, &state));
        }
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);

        apply_history_action(
            HistoryAction::Undo,
            &mut graph.clone(),
            &mut state,
            &mut history,
            false,
        );
        assert_eq!(history.redo.len(), 1);
        state.feedback = Some("camera and tool changes are transient".to_owned());
        assert_eq!(history.redo.len(), 1);

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.redo.is_empty());
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);
    }

    #[test]
    fn simulation_and_empty_stacks_report_guidance_without_mutation() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
            true,
        );
        assert_eq!(graph.parts().next().unwrap().0, part);
        assert!(state.feedback.as_deref().unwrap().contains("build mode"));

        apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );
        assert_eq!(state.feedback.as_deref(), Some("Nothing to undo"));
        apply_history_action(
            HistoryAction::Redo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );
        assert_eq!(state.feedback.as_deref(), Some("Nothing to redo"));
    }
}

#[cfg(test)]
mod showcase_loading_tests {
    use std::time::Duration;

    use bevy::prelude::IVec3;
    use mechanic_core::{BuildCommand, BuildPose, CuboidSpec, GridRotation, TopologyError};
    use mechanic_gpu::FixedStepScheduler;

    use super::{
        ConstructionGraph, EditorHistory, EditorSnapshot, EditorState, HistoryAction,
        apply_history_action, install_editor_graph, next_simulation_tick, showcase,
    };

    #[test]
    fn app_simulation_drops_catch_up_backlog() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 1;

        assert_eq!(
            next_simulation_tick(
                &mut scheduler,
                &mut next_tick,
                Duration::from_secs(1),
                false,
            ),
            Some(1)
        );
        assert_eq!(scheduler.next_tick(), 61);
        assert_eq!(
            next_simulation_tick(
                &mut scheduler,
                &mut next_tick,
                Duration::from_millis(17),
                false,
            ),
            Some(2)
        );
    }

    #[test]
    fn paused_simulation_does_not_advance_or_accumulate_time() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 7;
        let scheduler_tick = scheduler.next_tick();

        assert_eq!(
            next_simulation_tick(
                &mut scheduler,
                &mut next_tick,
                Duration::from_secs(10),
                true,
            ),
            None
        );
        assert_eq!(next_tick, 7);
        assert_eq!(scheduler.next_tick(), scheduler_tick);
    }

    #[test]
    fn selected_creation_replaces_editor_and_round_trips_history() {
        let mut graph = ConstructionGraph::new();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);
        let preset = showcase::CreationPreset::PendulumGarden256;
        let creation =
            install_editor_graph(&mut graph, showcase::build_preset(preset).unwrap()).unwrap();
        history.commit(previous);
        assert_eq!(graph.part_count(), preset.part_count());
        assert_eq!(creation.compounds.len(), preset.part_count());
        assert!(preset.matches(&graph));

        apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );
        assert_eq!(graph.part_count(), 0);
        apply_history_action(
            HistoryAction::Redo,
            &mut graph,
            &mut state,
            &mut history,
            false,
        );
        assert!(preset.matches(&graph));
    }

    #[test]
    fn failed_install_preserves_the_current_graph() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [2, 2, 2],
            BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();

        let result = install_editor_graph(&mut graph, ConstructionGraph::new());
        assert!(matches!(result, Err(TopologyError::EmptyConstruction)));
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.parts().next().unwrap().1, &spec);
    }
}

#[cfg(test)]
mod joint_number_tests {
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec,
        CuboidSpec, DriveLinkSpec, FaceKind, FaceRef, GridRotation, PartId,
    };

    use bevy::prelude::*;

    use super::{
        AppSimulation, EditorGraph, control_panel, drive_xray_is_visible, driven_bearing_count,
        joint_number_labels,
    };
    use crate::hotbar::Tool;
    use crate::ui::markers;

    fn spawned(outcome: BuildOutcome) -> PartId {
        match outcome {
            BuildOutcome::Spawned(part) => part,
            other => panic!("expected a spawn, got {other:?}"),
        }
    }

    fn cuboid(dimensions: [u8; 3], units: IVec3) -> CuboidSpec {
        CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
            .expect("test dimensions are in range")
    }

    /// One control block driving two joints, each on its own rotor.
    fn two_driven_joints() -> (ConstructionGraph, PartId, [Vec3; 2]) {
        let mut graph = ConstructionGraph::new();
        let base = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([16, 2, 4], IVec3::new(0, 1, 0))))
                .expect("the base spawns"),
        );
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
                )))
                .expect("the control block spawns"),
        );
        let mut anchors = Vec::new();
        for offset in [-6_i8, 6_i8] {
            let rotor = spawned(
                graph
                    .apply(BuildCommand::Spawn(cuboid(
                        [2, 2, 2],
                        IVec3::new(i32::from(offset), 3, 0),
                    )))
                    .expect("the rotor spawns"),
            );
            let anchor = Vec3::new(f32::from(offset) * 0.25, 0.5, 0.0);
            let BuildOutcome::BearingAdded(bearing) = graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(base, FaceKind::PositiveY),
                    FaceRef::part(rotor, FaceKind::NegativeY),
                    anchor,
                    Vec3::Y,
                )))
                .expect("the bearing is added")
            else {
                panic!("expected a bearing outcome");
            };
            graph
                .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                    controller, bearing,
                )))
                .expect("the wire is added");
            anchors.push(anchor);
        }
        (graph, controller, [anchors[0], anchors[1]])
    }

    #[test]
    fn floating_numbers_match_the_rows_the_panel_lists() {
        let (graph, controller, anchors) = two_driven_joints();
        let rows = control_panel::panel_rows(&graph, controller);
        let labels = joint_number_labels(&graph, |bearing| Some(bearing.shared_anchor));

        assert_eq!(rows.len(), 2, "each joint gets its own panel row");
        assert_eq!(
            labels,
            vec![(1, anchors[0]), (2, anchors[1])],
            "the number floating over a joint is the row number the panel shows"
        );
    }

    #[test]
    fn two_wires_on_one_joint_share_a_single_number() {
        let (mut graph, controller, anchors) = two_driven_joints();
        // A second group hung from the first joint's socket adds a wire
        // describing the same physical joint, which the panel folds into one
        // row and which therefore earns one number, not two.
        let extra = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(-6, 3, 0))))
                .expect("the extra rotor spawns"),
        );
        let base = graph.parts().next().expect("the base is the first part").0;
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(extra, FaceKind::NegativeY),
                anchors[0],
                Vec3::Y,
            )))
            .expect("the second bearing is added")
        else {
            panic!("expected a bearing outcome");
        };
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .expect("the second wire is added");

        assert_eq!(driven_bearing_count(&graph), 3, "three wires exist");
        assert_eq!(
            control_panel::panel_rows(&graph, controller).len(),
            2,
            "but they describe two joints"
        );
        assert_eq!(
            joint_number_labels(&graph, |bearing| Some(bearing.shared_anchor)),
            vec![(1, anchors[0]), (2, anchors[1])],
            "one joint carries one number no matter how many wires reach it"
        );
    }

    /// Builds an app running `update_joint_numbers` over the given graph.
    ///
    /// The camera has no viewport, so every projection fails and the labels
    /// stay hidden. That is deliberate: this exercises the spawn and despawn
    /// bookkeeping, which is what breaks, without needing a render target.
    /// What the overlay would number, with the connector in hand.
    fn numbered(graph: &ConstructionGraph) -> Vec<usize> {
        markers::wanted(
            &EditorGraph(graph.clone()),
            &AppSimulation::default(),
            Tool::Connector,
        )
        .into_iter()
        .map(|(number, _)| number)
        .collect()
    }

    #[test]
    fn every_driven_joint_is_numbered_once() {
        let (graph, _, _) = two_driven_joints();
        assert_eq!(numbered(&graph), vec![1, 2], "each joint gets one number");
    }

    #[test]
    fn numbers_track_joints_appearing_and_disappearing() {
        let (graph, _, _) = two_driven_joints();
        assert!(
            numbered(&ConstructionGraph::new()).is_empty(),
            "nothing driven, nothing numbered",
        );
        assert_eq!(numbered(&graph).len(), 2);
        assert!(
            numbered(&ConstructionGraph::new()).is_empty(),
            "removing the joints clears them",
        );
    }

    #[test]
    fn putting_away_the_connector_clears_the_numbers() {
        let (graph, _, _) = two_driven_joints();
        assert_eq!(numbered(&graph).len(), 2);
        assert!(
            markers::wanted(&EditorGraph(graph), &AppSimulation::default(), Tool::Hammer,)
                .is_empty(),
            "the numbers belong to the tools that show the wires",
        );
    }

    #[test]
    fn numbers_show_with_the_tools_that_show_the_wires() {
        for tool in [Tool::Connector, Tool::Controller] {
            assert!(
                drive_xray_is_visible(tool, 1),
                "{tool:?} should show joint numbers"
            );
        }
        for tool in [Tool::Block, Tool::Bearing, Tool::Weld, Tool::Hammer] {
            assert!(
                !drive_xray_is_visible(tool, 1),
                "{tool:?} should not show joint numbers"
            );
        }
        assert!(
            !drive_xray_is_visible(Tool::Connector, 0),
            "nothing driven means nothing to number"
        );
    }
}

#[cfg(test)]
mod creation_file_tests {
    use bevy::prelude::Vec3;
    use mechanic_core::{BearingDimensions, FaceKind, FaceRef};

    use super::{
        ConstructionGraph, EditorState, PlacedBearing, capture_creation,
        creation_store::{CreationStore, read_document},
        install_editor_graph, showcase,
    };

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("mechanic-creations-{}-{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A preset construction plus one bearing ring the editor is still holding.
    fn editor_with_a_loose_ring() -> (ConstructionGraph, EditorState) {
        let graph = showcase::build_preset(showcase::CreationPreset::PendulumGarden256)
            .expect("the preset builds");
        let (part, _) = graph.parts().next().expect("the preset has parts");
        let mut state = EditorState::default();
        state.placed_bearings.push(PlacedBearing {
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::new(0.25, 1.5, -0.75),
            dimensions: BearingDimensions::new(0.4, 0.15).expect("the ring is in range"),
        });
        (graph, state)
    }

    #[test]
    fn saving_then_opening_restores_the_construction_and_its_loose_rings() {
        let temporary = TempDir::new("round-trip");
        let store = CreationStore::new(&temporary.0);
        let (graph, state) = editor_with_a_loose_ring();

        let path = store
            .save(&capture_creation(&graph, &state, "Pendulum Rig"))
            .expect("the creation is written");

        let loaded = read_document(&path)
            .expect("the file parses")
            .into_graph()
            .expect("the document rebuilds");
        let mut installed = ConstructionGraph::new();
        let creation =
            install_editor_graph(&mut installed, loaded.graph).expect("the rebuild compiles");

        assert_eq!(loaded.name, "Pendulum Rig");
        assert_eq!(installed.part_count(), graph.part_count());
        assert_eq!(installed.weld_count(), graph.weld_count());
        assert_eq!(installed.bearing_count(), graph.bearing_count());
        assert_eq!(
            creation.compounds.len(),
            graph
                .compile()
                .expect("the original compiles")
                .compounds
                .len()
        );

        let restored = loaded.sockets.first().expect("the loose ring comes back");
        let original = &state.placed_bearings[0];
        assert_eq!(restored.anchor, original.anchor);
        assert_eq!(restored.dimensions, original.dimensions);
        assert_eq!(restored.source.face, original.source.face);
        assert_eq!(
            installed.parts().next().map(|(id, _)| id),
            match restored.source.owner {
                mechanic_core::FaceOwner::Part(part) => Some(part),
                mechanic_core::FaceOwner::Ground => None,
            },
            "the ring still hangs off the first part"
        );
    }

    #[test]
    fn the_listing_summarises_what_a_saved_creation_holds() {
        let temporary = TempDir::new("listing");
        let store = CreationStore::new(&temporary.0);
        let (graph, state) = editor_with_a_loose_ring();
        store
            .save(&capture_creation(&graph, &state, "Pendulum Rig"))
            .expect("the creation is written");

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Pendulum Rig");
        assert_eq!(listed[0].part_count, graph.part_count());
        assert_eq!(listed[0].joint_count, graph.bearing_count());
    }
}
