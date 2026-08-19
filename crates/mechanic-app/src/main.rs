//! Construction prototype with a GPU-physics preview.

#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

use std::collections::{HashSet, VecDeque};

mod builder;
mod camera;
mod creation_menu;
mod hotbar;
mod showcase;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::{NoFrustumCulling, RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    input::keyboard::Key,
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::PrimitiveTopology,
        renderer::{RenderDevice, RenderQueue},
    },
};
use builder::{
    BEARING_DEPTH, BLOCK_SIZE_METERS, GROUND_HALF_SIZE, PlacementCandidate, PlacementError,
    PlacementPlane, SurfaceHit, bearing_anchor_from_hit, bearing_attachment_candidate,
    bearing_overlaps_candidate, bearing_support_face, bearing_support_face_excluding, begin_weld,
    block_sheet_specs, candidate_from_hit, face_geometry_from_ref, raycast_construction,
    raycast_oriented_cuboid, raycast_placement_plane, rigid_body_parts, stage_bearing_attachment,
    stage_bearing_block_batch, stage_block_batch_from_source, stage_weld_objects,
    validate_block_batch,
};
use camera::OrbitCamera;
use creation_menu::CreationMenuState;
use hotbar::{HotbarPointerCapture, SelectedTool, Tool, shortcut_tool};
use mechanic_core::{
    BearingDimensions, BuildCommand, CompiledCreation, ConstructionGraph, CuboidSpec, FaceOwner,
    MAX_BEARING_OUTER_DIAMETER, MIN_BEARING_DIAMETER_GAP, MIN_BEARING_OUTER_DIAMETER, PartId,
    PendingOperation, TopologyError,
};
use mechanic_gpu::{FixedStepScheduler, GpuPhysics, GpuPhysicsConfig, GpuTransform};

const SIMULATION_VISUAL_TICK_INTERVAL: u32 = 2;
const HAMMER_CHARGE_SECONDS: f32 = 1.5;
const HAMMER_MIN_IMPULSE: f32 = 25.0;
const HAMMER_MAX_IMPULSE: f32 = 4_000.0;
const HISTORY_CAPACITY: usize = 64;
const BEARING_DIAMETER_STEP: f32 = 0.05;
const HELP_TEXT_COLOR: Color = Color::srgb(0.88, 0.92, 0.96);
const HELP_MUTED_COLOR: Color = Color::srgb(0.58, 0.66, 0.73);
const HELP_BLUE_COLOR: Color = Color::srgb(0.30, 0.78, 1.0);
const HELP_GREEN_COLOR: Color = Color::srgb(0.35, 0.93, 0.60);
const HELP_YELLOW_COLOR: Color = Color::srgb(1.0, 0.76, 0.28);
const HELP_RED_COLOR: Color = Color::srgb(1.0, 0.40, 0.36);
const HELP_ORANGE_COLOR: Color = Color::srgb(1.0, 0.65, 0.20);

#[derive(Resource, Default)]
struct EditorGraph(ConstructionGraph);

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
}

#[derive(Clone, Copy, Debug)]
struct HammerCharge {
    body_index: u32,
    local_point: Vec3,
    direction: Vec3,
    elapsed_seconds: f32,
}

#[derive(Clone, Copy, Debug)]
struct SimulationHit {
    body_index: u32,
    distance: f32,
    point: Vec3,
}

#[derive(Clone, Debug)]
struct BlockDrag {
    start: PlacementCandidate,
    attachment: BlockAttachment,
    plane: PlacementPlane,
    last_endpoint: Option<(PlacementPlane, IVec3)>,
    specs: Vec<CuboidSpec>,
    error: Option<PlacementError>,
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
}

fn handle_simulation_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut menu: ResMut<CreationMenuState>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if keyboard.just_pressed(KeyCode::Escape) && simulation.is_running() && !menu.is_open() {
        *simulation = AppSimulation::default();
        state.construction_mesh_dirty = true;
        state.feedback = Some("Returned to build mode".to_owned());
        return;
    }
    let Some(shortcut) = requested_simulation_shortcut(&keyboard) else {
        return;
    };
    if menu.is_open() {
        menu.close();
        state.feedback = Some("Creation menu closed".to_owned());
        return;
    }
    let restarting = simulation.is_running();
    if restarting && shortcut == SimulationShortcut::TogglePlayback {
        simulation.paused = !simulation.paused;
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

fn handle_creation_menu_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
    mut menu: ResMut<CreationMenuState>,
) {
    menu.begin_frame();
    if menu.is_open() && keyboard.just_pressed(KeyCode::Escape) {
        menu.close();
        state.feedback = Some("Creation menu closed".to_owned());
    } else if keyboard.just_pressed(KeyCode::KeyP) {
        if simulation.is_running() {
            state.feedback =
                Some("Creations can be opened in build mode — press Escape first".to_owned());
        } else if menu.is_open() {
            menu.close();
            state.feedback = Some("Creation menu closed".to_owned());
        } else {
            cancel_transient_editor_state(&mut graph.0, &mut state);
            menu.open();
            state.feedback = Some("Choose a creation to open".to_owned());
        }
    }
}

fn handle_creation_request(
    mut menu: ResMut<CreationMenuState>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    mut camera: Single<(&mut OrbitCamera, &mut Transform)>,
) {
    let Some(preset) = menu.take_request() else {
        return;
    };
    let previous = EditorSnapshot::capture(&graph.0, &state);
    let result = showcase::build_preset(preset).and_then(|candidate| {
        install_editor_graph(&mut graph.0, candidate).map_err(showcase::ShowcaseError::from)
    });
    match result {
        Ok(creation) => {
            history.commit(previous);
            debug_assert_eq!(creation.compounds.len(), preset.body_count());
            clear_hover(&mut state);
            state.block_drag = None;
            state.delete_drag = None;
            state.delete_target = None;
            state.placed_bearings.clear();
            state.construction_mesh_dirty = true;
            state.feedback = Some(format!(
                "Opened {}: {} welds, {} bearings, {} bodies — Space to simulate",
                preset.label(),
                graph.0.weld_count(),
                graph.0.bearing_count(),
                creation.compounds.len(),
            ));
            if let Some((minimum, maximum)) = graph_bounds(&graph.0) {
                let (orbit, transform) = &mut *camera;
                orbit.frame_bounds(minimum, maximum);
                **transform = orbit.transform();
            }
        }
        Err(error) => {
            state.feedback = Some(format!("Could not open creation: {error}"));
        }
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
        let center = spec.pose.translation();
        let half = spec.size_meters() * 0.5;
        let rotation = spec.pose.rotation.quaternion();
        for x in [-half.x, half.x] {
            for y in [-half.y, half.y] {
                for z in [-half.z, half.z] {
                    let corner = center + rotation * Vec3::new(x, y, z);
                    minimum = minimum.min(corner);
                    maximum = maximum.max(corner);
                }
            }
        }
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
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
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
        let diagnostics = {
            let gpu = simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state");
            gpu.dispatch_tick(render_device.wgpu_device(), &render_queue, tick);
            gpu.read_last_tick(render_device.wgpu_device())
                .map_err(|error| error.to_string())
        };
        match diagnostics {
            Ok(diagnostics) if diagnostics.error_flags == 0 => {}
            Ok(diagnostics) => {
                stop_failed_simulation(
                    &mut simulation,
                    &mut state,
                    format!("physics kernel reported flags {}", diagnostics.error_flags),
                );
                return;
            }
            Err(error) => {
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
            *mesh = combined_simulation_mesh(
                &graph.0,
                creation,
                &simulation.transforms,
                SimulationMeshKind::Static,
            );
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
        *mesh = combined_simulation_mesh(
            &graph.0,
            creation,
            &simulation.transforms,
            SimulationMeshKind::Dynamic,
        );
    }
    if let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
        *mesh = combined_simulation_bearing_mesh(
            &graph.0,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
        );
    }
    **bearing_visibility = if graph.0.bearing_count() == 0 && state.placed_bearings.is_empty() {
        Visibility::Hidden
    } else {
        Visibility::Visible
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
    preview_error: Option<PlacementError>,
    feedback: Option<String>,
    construction_mesh_dirty: bool,
    delete_target: Option<DeleteTarget>,
    block_drag: Option<BlockDrag>,
    block_preview_revision: u64,
    delete_drag: Option<DeleteDrag>,
    delete_preview_revision: u64,
    placed_bearings: Vec<PlacedBearing>,
}

#[derive(Resource)]
struct EditorVisuals {
    construction_mesh: Handle<Mesh>,
    simulation_mesh: Handle<Mesh>,
    bearing_mesh: Handle<Mesh>,
    cube_preview_mesh: Handle<Mesh>,
    bearing_preview_mesh: Handle<Mesh>,
    white_preview_material: Handle<StandardMaterial>,
    green_preview_material: Handle<StandardMaterial>,
    red_preview_material: Handle<StandardMaterial>,
    block_drag_preview_mesh: Handle<Mesh>,
    delete_drag_preview_mesh: Handle<Mesh>,
    weld_hover_preview_mesh: Handle<Mesh>,
    weld_selection_preview_mesh: Handle<Mesh>,
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

#[derive(Component)]
struct HelpText;

#[derive(Clone, Copy, Component)]
enum HelpLine {
    Title,
    PrimaryControls,
    EditControls,
    PointerControls,
    Tool,
    Counts,
    Hint,
    Status,
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
        .init_resource::<EditorGraph>()
        .init_resource::<EditorState>()
        .init_resource::<EditorHistory>()
        .init_resource::<CreationMenuState>()
        .init_resource::<AppSimulation>()
        .init_resource::<HammerInteraction>()
        .init_resource::<BearingToolSettings>()
        .init_resource::<SelectedTool>()
        .init_resource::<HotbarPointerCapture>()
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.75, 0.80, 0.90),
            brightness: 350.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_creation_menu_shortcut,
                toggle_help_text,
                handle_history_shortcut,
                creation_menu::update,
                handle_creation_request,
                hotbar::update,
                camera::update_orbit_camera,
                handle_simulation_shortcut,
                handle_shortcuts,
                handle_bearing_dimension_shortcuts,
                handle_tool_change,
                update_hover,
                handle_build_actions,
                handle_hammer_actions,
                sync_visual_meshes,
                update_joint_xray,
                advance_simulation,
                update_previews,
                update_help_text,
            )
                .chain(),
        )
        .run();
}

#[allow(clippy::too_many_lines)] // One-time Bevy scene composition is clearest in declaration order.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let construction_mesh = meshes.add(Cuboid::default());
    let simulation_mesh = meshes.add(Cuboid::default());
    let bearing_mesh = meshes.add(Cuboid::default());
    let cube_preview_mesh = meshes.add(Cuboid::default());
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
    let bearing_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        metallic: 0.35,
        perceptual_roughness: 0.55,
        ..default()
    });
    let joint_xray_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let white_preview_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.34),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let red_preview_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.06, 0.04, 0.46),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let green_preview_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.12, 1.0, 0.28, 0.52),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });

    commands.insert_resource(EditorVisuals {
        construction_mesh: construction_mesh.clone(),
        simulation_mesh: simulation_mesh.clone(),
        bearing_mesh: bearing_mesh.clone(),
        cube_preview_mesh: cube_preview_mesh.clone(),
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
        Name::new("Joint x-ray mesh"),
        Mesh3d(bearing_mesh),
        MeshMaterial3d(joint_xray_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        JointXrayVisual,
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

    commands
        .spawn((
            Name::new("Help and status panel"),
            HelpText,
            Node {
                position_type: PositionType::Absolute,
                top: px(16),
                left: px(16),
                width: px(720),
                max_width: percent(94),
                padding: UiRect::all(px(16)),
                flex_direction: FlexDirection::Column,
                row_gap: px(7),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(10)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.015, 0.022, 0.032, 0.94)),
            BorderColor::all(Color::srgba(0.24, 0.38, 0.48, 0.92)),
            GlobalZIndex(20),
            Visibility::Hidden,
        ))
        .with_children(|panel| {
            for line in [
                HelpLine::Title,
                HelpLine::PrimaryControls,
                HelpLine::EditControls,
                HelpLine::PointerControls,
                HelpLine::Tool,
                HelpLine::Counts,
                HelpLine::Hint,
                HelpLine::Status,
            ] {
                let (font_size, color, margin) = match line {
                    HelpLine::Title => (21.0, HELP_BLUE_COLOR, UiRect::bottom(px(3))),
                    HelpLine::PrimaryControls => (15.0, HELP_TEXT_COLOR, UiRect::ZERO),
                    HelpLine::EditControls | HelpLine::PointerControls => {
                        (14.0, HELP_MUTED_COLOR, UiRect::ZERO)
                    }
                    HelpLine::Tool => (17.0, HELP_BLUE_COLOR, UiRect::top(px(6))),
                    HelpLine::Counts => (14.0, HELP_MUTED_COLOR, UiRect::ZERO),
                    HelpLine::Hint => (15.0, HELP_TEXT_COLOR, UiRect::top(px(4))),
                    HelpLine::Status => (14.0, HELP_GREEN_COLOR, UiRect::ZERO),
                };
                panel.spawn((
                    line,
                    Text::new(""),
                    TextFont {
                        font_size: FontSize::Px(font_size),
                        ..default()
                    },
                    TextColor(color),
                    Node {
                        width: percent(100),
                        margin,
                        ..default()
                    },
                ));
            }
        });
    hotbar::spawn(&mut commands);
    creation_menu::spawn(&mut commands);
}

fn help_toggle_requested(keyboard: &ButtonInput<Key>) -> bool {
    keyboard
        .get_just_pressed()
        .any(|key| matches!(key, Key::Character(character) if character == "?"))
}

fn toggle_help_text(
    keyboard: Res<ButtonInput<Key>>,
    mut visibility: Single<&mut Visibility, With<HelpText>>,
) {
    if !help_toggle_requested(&keyboard) {
        return;
    }
    **visibility = match **visibility {
        Visibility::Hidden => Visibility::Visible,
        _ => Visibility::Hidden,
    };
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
    mut menu: ResMut<CreationMenuState>,
) {
    let Some(action) = requested_history_action(&keyboard) else {
        return;
    };
    let restored = apply_history_action(
        action,
        &mut graph.0,
        &mut state,
        &mut history,
        simulation.is_running(),
    );
    if restored {
        menu.close();
    }
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
    menu: Res<CreationMenuState>,
) {
    if menu.blocks_pointer() {
        return;
    }
    for key in [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
    ] {
        if keyboard.just_pressed(key) {
            selection.0 = shortcut_tool(key).expect("numbered tool key has a mapping");
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
        } else {
            state.feedback = Some("Q changes the plane during a block or delete drag".to_owned());
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
    let direction = if keyboard.just_pressed(KeyCode::BracketLeft) {
        -1
    } else if keyboard.just_pressed(KeyCode::BracketRight) {
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
        menu.blocks_pointer(),
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
    state.feedback = None;
}

#[allow(clippy::too_many_arguments)]
fn update_hover(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    hotbar_capture: Res<HotbarPointerCapture>,
) {
    if simulation.is_running() {
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
        refresh_tool_preview(&graph.0, &mut state, selection.0);
        return;
    }
    if hotbar_capture.active() {
        clear_hover(&mut state);
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        if state.block_drag.is_some() {
            invalidate_block_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.delete_drag.is_some() {
            invalidate_delete_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state, selection.0);
        return;
    };
    let (camera, camera_transform) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        if state.block_drag.is_some() {
            invalidate_block_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.delete_drag.is_some() {
            invalidate_delete_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state, selection.0);
        return;
    };
    if state.block_drag.is_some() {
        refresh_block_drag(&graph.0, &mut state, ray.origin, ray.direction.as_vec3());
        return;
    }
    if state.delete_drag.is_some() {
        refresh_delete_drag(&graph.0, &mut state, ray.origin, ray.direction.as_vec3());
        return;
    }
    let construction_hit = raycast_construction(&graph.0, ray.origin, ray.direction.as_vec3());
    if (selection.0 == Tool::Block || mouse_buttons.pressed(MouseButton::Right))
        && let Some((bearing, distance)) = raycast_placed_bearings(
            &graph.0,
            &state.placed_bearings,
            ray.origin,
            ray.direction.as_vec3(),
        )
        && construction_hit.is_none_or(|hit| distance <= hit.distance)
    {
        state.hovered = construction_hit;
        state.hovered_bearing = Some(bearing);
        refresh_tool_preview(&graph.0, &mut state, selection.0);
        return;
    }
    let Some(hit) = construction_hit else {
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state, selection.0);
        return;
    };
    state.hovered_bearing = None;
    state.hovered = Some(hit);
    refresh_tool_preview(&graph.0, &mut state, selection.0);
}

fn refresh_block_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, plane, last_endpoint) = {
        let drag = state
            .block_drag
            .as_ref()
            .expect("block drag was checked by caller");
        (drag.start, drag.plane, drag.last_endpoint)
    };
    let Some(endpoint) = raycast_placement_plane(ray_origin, ray_direction, start.spec, plane)
    else {
        invalidate_block_drag(state, PlacementError::DragPlaneUnavailable);
        return;
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
            centers
                .contains(&spec.pose.translation_half_units())
                .then_some(part)
        })
        .collect())
}

fn clear_hover(state: &mut EditorState) {
    state.hovered = None;
    state.hovered_bearing = None;
    state.attachment_bearing = None;
    state.preview = None;
    state.preview_error = None;
}

fn refresh_tool_preview(graph: &ConstructionGraph, state: &mut EditorState, tool: Tool) {
    state.preview = None;
    state.attachment_bearing = None;
    state.preview_error = match (tool, graph.pending()) {
        (Tool::Block, _) => {
            let surface_candidate = state.hovered.map(|hit| candidate_from_hit(graph, hit));
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
        (Tool::Weld | Tool::Hammer | Tool::JointXray, _) => None,
        (Tool::Bearing, _) => state
            .hovered
            .and_then(|hit| bearing_anchor_from_hit(graph, hit).err()),
    };
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
    hotbar_capture: Res<HotbarPointerCapture>,
) {
    if simulation.is_running() {
        return;
    }
    if hotbar_capture.active() {
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
    if mouse.just_pressed(MouseButton::Right) {
        if let Some(index) = state.hovered_bearing {
            state.delete_target = Some(DeleteTarget::PlacedBearing(index));
            state.feedback = Some("Release right mouse to delete bearing".to_owned());
        } else if let Some(hit) = state.hovered
            && let FaceOwner::Part(part) = hit.face.owner
            && let Some(spec) = graph.0.part(part).copied()
        {
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
    if selection.0 == Tool::Block {
        handle_block_actions(&mouse, &mut graph.0, &mut state, &mut history);
        return;
    }
    if mouse.pressed(MouseButton::Right) || !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    match selection.0 {
        Tool::Block => unreachable!("block actions are handled before this match"),
        Tool::Weld => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Select an object".to_owned());
                return;
            };
            match graph.0.pending() {
                Some(PendingOperation::Weld(first)) => {
                    match stage_weld_objects(&graph.0, first.owner, hit.face.owner) {
                        Ok(staged) => {
                            let previous = EditorSnapshot::capture(&graph.0, &state);
                            graph.0 = staged;
                            history.commit(previous);
                            state.feedback = Some("Welded the two objects".to_owned());
                        }
                        Err(error) => state.feedback = Some(error.to_string()),
                    }
                }
                _ => match begin_weld(&mut graph.0, hit.face) {
                    Ok(()) => {
                        state.feedback =
                            Some("First object selected; choose a touching object".to_owned());
                    }
                    Err(error) => state.feedback = Some(error.to_string()),
                },
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
        Tool::JointXray => {
            state.feedback =
                Some("Joint X-ray shows every bearing through the creation".to_owned());
        }
    }
    refresh_tool_preview(&graph.0, &mut state, selection.0);
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
        state.block_drag = Some(BlockDrag {
            start: candidate,
            attachment,
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
    mut hammer: ResMut<HammerInteraction>,
    mut state: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    hotbar_capture: Res<HotbarPointerCapture>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if !simulation.is_running() {
        hammer.charging = None;
        return;
    }
    if simulation.is_paused() {
        hammer.charging = None;
        return;
    }
    if !selection.0.works_in_mode(true) {
        hammer.charging = None;
        if mouse.just_pressed(MouseButton::Left) && !hotbar_capture.active() {
            state.feedback = Some(format!(
                "{} is available in build mode — press Escape first",
                selection.0.label()
            ));
        }
        return;
    }
    if hotbar_capture.active() && hammer.charging.is_none() {
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
    let position = Vec3::from_slice(&transform.position[..3]);
    let rotation = Quat::from_array(transform.rotation);
    let world_point = position + rotation * charge.local_point;
    let magnitude = hammer_impulse_magnitude(charge.elapsed_seconds);
    let result = simulation
        .gpu
        .as_ref()
        .expect("running simulation has GPU state")
        .apply_impulse(
            render_device.wgpu_device(),
            &render_queue,
            charge.body_index,
            world_point,
            charge.direction * magnitude,
        );
    state.feedback = Some(match result {
        Ok(_) => format!("Hammer strike: {magnitude:.0} N·s"),
        Err(error) => format!("Hammer strike failed: {error}"),
    });
}

fn hammer_impulse_magnitude(elapsed_seconds: f32) -> f32 {
    let charge = (elapsed_seconds / HAMMER_CHARGE_SECONDS).clamp(0.0, 1.0);
    HAMMER_MIN_IMPULSE + (HAMMER_MAX_IMPULSE - HAMMER_MIN_IMPULSE) * charge * charge
}

fn raycast_simulation(
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    origin: Vec3,
    direction: Vec3,
) -> Option<SimulationHit> {
    creation
        .colliders
        .iter()
        .filter_map(|collider| {
            let body_index = collider.compound_index;
            let transform = transforms.get(body_index as usize)?;
            let position = Vec3::from_slice(&transform.position[..3]);
            let rotation = Quat::from_array(transform.rotation);
            let center = position + rotation * collider.local_center;
            let hit = raycast_oriented_cuboid(
                origin,
                direction,
                center,
                rotation * collider.local_rotation,
                collider.half_extents,
            )?;
            Some(SimulationHit {
                body_index,
                distance: hit.distance,
                point: hit.point,
            })
        })
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
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

#[allow(clippy::type_complexity)]
fn sync_visual_meshes(
    graph: Res<EditorGraph>,
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
) {
    if !state.construction_mesh_dirty {
        return;
    }
    if graph.0.part_count() == 0 {
        **construction_visibility = Visibility::Hidden;
    } else {
        if let Some(mut mesh) = meshes.get_mut(&visuals.construction_mesh) {
            *mesh = combined_construction_mesh(&graph.0);
        }
        **construction_visibility = Visibility::Visible;
    }
    **simulation_visibility = Visibility::Hidden;
    if graph.0.bearing_count() == 0 && state.placed_bearings.is_empty() {
        **bearing_visibility = Visibility::Hidden;
    } else {
        if let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
            *mesh = combined_bearing_mesh(&graph.0, &state.placed_bearings);
        }
        **bearing_visibility = Visibility::Visible;
    }
    state.construction_mesh_dirty = false;
}

fn joint_xray_is_visible(tool: Tool, simulating: bool, bearing_count: usize) -> bool {
    tool == Tool::JointXray && !simulating && bearing_count > 0
}

fn update_joint_xray(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    mut visibility: Single<&mut Visibility, With<JointXrayVisual>>,
) {
    **visibility = if joint_xray_is_visible(
        selection.0,
        simulation.is_running(),
        visible_bearing_count(&graph.0, &state.placed_bearings),
    ) {
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
                *mesh = combined_specs_mesh(&specs);
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
                        *mesh = combined_specs_mesh(&drag.specs);
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
        (Tool::Weld, pending) => {
            if let Some(part) = hovered_part(state.hovered) {
                if *rendered_weld_hover != Some(part) || graph.is_changed() {
                    let specs = rigid_body_parts(&graph.0, part)
                        .into_iter()
                        .filter_map(|member| graph.0.part(member).copied())
                        .collect::<Vec<_>>();
                    if let Some(mut mesh) = meshes.get_mut(&visuals.weld_hover_preview_mesh) {
                        *mesh = combined_specs_mesh_scaled(&specs, 1.018);
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
                        *mesh = combined_specs_mesh_scaled(&specs, 1.028);
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
            if let Some(hit) = state.hovered {
                let face = face_geometry_from_ref(hit.face, Some(&graph.0));
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
        (Tool::Hammer | Tool::JointXray, _) => {}
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
    tool == Tool::Block && attachment_bearing.is_some() && preview_error.is_none()
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

#[allow(clippy::too_many_lines)] // Tool-specific guidance is kept together with its HUD layout.
fn update_help_text(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    hammer: Res<HammerInteraction>,
    selection: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    mut lines: Query<(&HelpLine, &mut Text, &mut TextColor)>,
) {
    let active_error = state
        .delete_drag
        .as_ref()
        .and_then(|drag| drag.error.as_ref())
        .or(state.preview_error.as_ref());
    let status = if let Some(charge) = hammer.charging {
        format!(
            "Hammer charge: {:.0}% — release to strike",
            charge.elapsed_seconds / HAMMER_CHARGE_SECONDS * 100.0
        )
    } else {
        active_error.map_or_else(
            || state.feedback.clone().unwrap_or_else(|| "Ready".to_owned()),
            ToString::to_string,
        )
    };
    let status_color = if hammer.charging.is_some() {
        HELP_YELLOW_COLOR
    } else if active_error.is_some()
        || ["Cannot", "Could not", "Simulation stopped"]
            .iter()
            .any(|prefix| status.starts_with(prefix))
    {
        HELP_RED_COLOR
    } else {
        HELP_GREEN_COLOR
    };
    let tool_hint = if simulation.is_paused() {
        "Simulation paused at the current pose — press Space to resume".to_owned()
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!(
            "Release to delete {} cuboid(s) on the {} plane",
            drag.parts.len(),
            drag.plane.label()
        )
    } else {
        match (
            simulation.is_running(),
            selection.0,
            graph.0.pending(),
            state.block_drag.as_ref(),
            state.attachment_bearing,
        ) {
            (false, Tool::Block, _, Some(drag), _) => {
                if matches!(drag.attachment, BlockAttachment::Bearing { .. }) {
                    format!(
                        "Green bearing attachment active — release to connect {} block(s)",
                        drag.specs.len()
                    )
                } else {
                    format!(
                        "Release to place {} blocks on the {} plane",
                        drag.specs.len(),
                        drag.plane.label()
                    )
                }
            }
            (false, Tool::Block, _, None, Some(_)) => {
                "Green bearing attachment active — click or drag to connect blocks".to_owned()
            }
            (true, Tool::Hammer, _, _, _) => {
                "Hold left mouse on a moving cuboid; release to strike".to_owned()
            }
            (true, _, _, _, _) => {
                "This tool is build-only — press Escape to return to build mode".to_owned()
            }
            (false, Tool::Block, _, _, _) => {
                "Click for one block or drag to place a welded sheet".to_owned()
            }
            (false, Tool::Weld, None, _, _) => "Left click selects the first object".to_owned(),
            (false, Tool::Weld, Some(_), _, _) => "Left click a touching second object".to_owned(),
            (false, Tool::Bearing, _, _, _) => {
                "Left click places a bearing; use Block to attach it".to_owned()
            }
            (false, Tool::Hammer, _, _, _) => {
                "Hammer is available while simulating — press Space to start".to_owned()
            }
            (false, Tool::JointXray, _, _, _) => {
                "All bearings are visible through the construction".to_owned()
            }
        }
    };
    let (mode, title_color, primary_controls) = if simulation.is_paused() {
        (
            "PAUSED",
            HELP_YELLOW_COLOR,
            "SPACE  Resume     SHIFT+SPACE  Restart     ESC  Build mode     ?  Hide help",
        )
    } else if simulation.is_running() {
        (
            "SIMULATING",
            HELP_GREEN_COLOR,
            "SPACE  Pause     SHIFT+SPACE  Restart     ESC  Build mode     ?  Hide help",
        )
    } else {
        (
            "BUILDING",
            HELP_BLUE_COLOR,
            "SPACE  Start simulation     P  Open creations     ?  Hide help",
        )
    };
    let title = format!("MECHANIC  •  {mode}");
    let action_controls = if state.delete_drag.is_some() {
        "RELEASE RIGHT  Delete     ESC  Cancel"
    } else {
        match (
            simulation.is_running(),
            selection.0,
            state.block_drag.is_some(),
        ) {
            (false, Tool::Block, true) => "RELEASE  Place     RIGHT / ESC  Cancel",
            (true, Tool::Hammer, _) if !simulation.is_paused() => "HOLD LEFT  Charge hammer",
            (true, _, _) => "Construction actions unavailable",
            (false, Tool::JointXray, _) => "ORBIT / PAN  Inspect     RIGHT DRAG  Delete",
            (false, _, _) => "LEFT  Action     RIGHT DRAG  Delete",
        }
    };
    let plane_controls = if let Some(drag) = state.block_drag.as_ref() {
        format!("Q  Cycle plane ({})", drag.plane.label())
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!("Q  Cycle delete plane ({})", drag.plane.label())
    } else {
        "Q  Cycle plane while dragging or deleting".to_owned()
    };
    let edit_controls = if simulation.is_running() {
        if simulation.is_paused() {
            "Current pose is frozen; construction editing remains locked".to_owned()
        } else {
            "Physics is live; construction editing remains locked".to_owned()
        }
    } else {
        format!("{plane_controls}     CTRL/CMD+Z  Undo     SHIFT+CTRL/CMD+Z  Redo")
    };
    let pointer_controls =
        format!("{action_controls}     ALT+LEFT  Orbit     SHIFT+LEFT  Pan     WHEEL  Zoom");
    let tool = tool_status_line(selection.0, bearing_settings.dimensions);
    let tool_color = match selection.0 {
        Tool::Bearing => HELP_ORANGE_COLOR,
        Tool::Hammer => HELP_YELLOW_COLOR,
        Tool::Weld => HELP_GREEN_COLOR,
        Tool::Block | Tool::JointXray => HELP_BLUE_COLOR,
    };
    let counts = format!(
        "{} parts  •  {} welds  •  {} bearings",
        graph.0.part_count(),
        graph.0.weld_count(),
        visible_bearing_count(&graph.0, &state.placed_bearings),
    );
    let status = format!("STATUS  •  {status}");

    for (line, mut text, mut color) in &mut lines {
        let (content, line_color): (&str, Color) = match line {
            HelpLine::Title => (&title, title_color),
            HelpLine::PrimaryControls => (primary_controls, HELP_TEXT_COLOR),
            HelpLine::EditControls => (&edit_controls, HELP_MUTED_COLOR),
            HelpLine::PointerControls => (&pointer_controls, HELP_MUTED_COLOR),
            HelpLine::Tool => (&tool, tool_color),
            HelpLine::Counts => (&counts, HELP_MUTED_COLOR),
            HelpLine::Hint => (&tool_hint, HELP_TEXT_COLOR),
            HelpLine::Status => (&status, status_color),
        };
        text.0.clear();
        text.0.push_str(content);
        color.0 = line_color;
    }
}

fn tool_status_line(tool: Tool, bearing_dimensions: BearingDimensions) -> String {
    match tool {
        Tool::Block => format!("Tool: Block    Block size: {BLOCK_SIZE_METERS:.2} m"),
        Tool::Bearing => format!(
            "Tool: Bearing    Outer: {:.2} m  [ / ]    Inner: {:.2} m  Shift+[ / Shift+]",
            bearing_dimensions.outer_diameter(),
            bearing_dimensions.inner_diameter(),
        ),
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

fn combined_construction_mesh(graph: &ConstructionGraph) -> Mesh {
    let mut positions = Vec::with_capacity(graph.part_count() * CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(graph.part_count() * CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(graph.part_count() * CUBE_INDICES.len());
    for (_, spec) in graph.parts() {
        append_cuboid(spec, &mut positions, &mut normals, &mut indices);
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_specs_mesh(specs: &[CuboidSpec]) -> Mesh {
    combined_specs_mesh_scaled(specs, 1.0)
}

fn combined_specs_mesh_scaled(specs: &[CuboidSpec], scale_factor: f32) -> Mesh {
    let mut positions = Vec::with_capacity(specs.len() * CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(specs.len() * CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(specs.len() * CUBE_INDICES.len());
    for spec in specs {
        append_transformed_cuboid(
            spec.pose.translation(),
            spec.pose.rotation.quaternion(),
            spec.size_meters() * scale_factor,
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
    let colliders = creation.colliders.iter().filter(|collider| {
        let is_static = creation.compounds[collider.compound_index as usize].is_static;
        match kind {
            SimulationMeshKind::Static => is_static,
            SimulationMeshKind::Dynamic => !is_static,
        }
    });
    let collider_count = colliders.clone().count();
    let mut positions = Vec::with_capacity(collider_count * CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(collider_count * CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(collider_count * CUBE_INDICES.len());
    for collider in colliders {
        let transform = transforms[collider.compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        let translation = root_translation + root_rotation * collider.local_center;
        let rotation = root_rotation * collider.local_rotation;
        let size = graph
            .part(collider.source_part)
            .expect("compiled collider source remains in graph")
            .size_meters();
        append_transformed_cuboid(
            translation,
            rotation,
            size,
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
    const SEGMENTS: u16 = 24;
    let axis = axis.normalize();
    let tangent_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    let tangent_v = axis.cross(tangent_u);
    let outer_radius = dimensions.outer_diameter() * 0.5;
    let inner_radius = dimensions.inner_diameter() * 0.5;
    let half_depth = BEARING_DEPTH * 0.5;
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

fn append_cuboid(
    spec: &CuboidSpec,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let size = spec.size_meters();
    let rotation = spec.pose.rotation.quaternion();
    let translation = spec.pose.translation();
    append_transformed_cuboid(translation, rotation, size, positions, normals, indices);
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

#[cfg(test)]
mod rendering_tests {
    use bevy::{
        mesh::VertexAttributeValues,
        prelude::{IVec3, Mesh, Quat, Vec3},
    };
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec,
        FaceKind, FaceRef, GridRotation,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        BEARING_DEPTH, PlacedBearing, append_bearing_cylinder, bearing_preview_dimensions_changed,
        combined_bearing_mesh, combined_simulation_bearing_mesh, joint_xray_is_visible,
    };
    use crate::hotbar::Tool;

    #[test]
    fn bearing_mesh_uses_custom_outer_inner_and_fixed_depth() {
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
        assert!((maximum_radius - dimensions.outer_diameter() * 0.5).abs() < 1.0e-6);
        assert!((minimum_radius - dimensions.inner_diameter() * 0.5).abs() < 1.0e-6);
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
        assert!((attached_radius - attached_dimensions.outer_diameter() * 0.5).abs() < 1.0e-6);
        assert!((placed_radius - placed_dimensions.outer_diameter() * 0.5).abs() < 1.0e-6);
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
        assert!(joint_xray_is_visible(Tool::JointXray, false, 1));
        assert!(!joint_xray_is_visible(Tool::JointXray, true, 1));
        assert!(!joint_xray_is_visible(Tool::JointXray, false, 0));
        assert!(!joint_xray_is_visible(Tool::Block, false, 1));
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
        prelude::{App, ButtonInput, IVec3, KeyCode, MouseButton, Update, Vec3, Visibility},
    };
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec,
        FaceKind, FaceOwner, FaceRef, GridRotation, PendingOperation, RigidLinkSpec,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        AppSimulation, BearingDimensionTarget, BearingToolSettings, BlockAttachment, BlockDrag,
        EditorGraph, EditorHistory, EditorState, HAMMER_CHARGE_SECONDS, HAMMER_MAX_IMPULSE,
        HAMMER_MIN_IMPULSE, HelpText, HistoryAction, HotbarPointerCapture, PlacedBearing,
        PlacementPlane, SelectedTool, SimulationShortcut, SurfaceHit, Tool,
        adjusted_bearing_dimensions, apply_history_action, bearing_attachment_candidate,
        bearing_attachment_is_highlighted, block_sheet_specs, candidate_from_hit,
        delete_sheet_parts, hammer_impulse_magnitude, handle_block_actions, handle_build_actions,
        handle_tool_change, help_toggle_requested, raycast_construction, raycast_placed_bearings,
        raycast_simulation, refresh_tool_preview, requested_bearing_dimension_adjustment,
        requested_simulation_shortcut, rigid_body_parts, stage_part_deletion_preserving_bearings,
        toggle_help_text, tool_status_line,
    };

    #[test]
    fn question_mark_toggles_the_hidden_help_overlay() {
        let mut app = App::new();
        app.init_resource::<ButtonInput<Key>>()
            .add_systems(Update, toggle_help_text);
        let help = app.world_mut().spawn((HelpText, Visibility::Hidden)).id();
        let question_mark = Key::Character("?".into());

        app.world_mut()
            .resource_mut::<ButtonInput<Key>>()
            .press(question_mark.clone());
        assert!(help_toggle_requested(
            app.world().resource::<ButtonInput<Key>>()
        ));
        app.update();
        assert_eq!(
            app.world().entity(help).get::<Visibility>(),
            Some(&Visibility::Visible)
        );

        {
            let mut keyboard = app.world_mut().resource_mut::<ButtonInput<Key>>();
            keyboard.release(question_mark.clone());
            keyboard.clear();
            keyboard.press(question_mark);
        }
        app.update();
        assert_eq!(
            app.world().entity(help).get::<Visibility>(),
            Some(&Visibility::Hidden)
        );
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
        keyboard.press(KeyCode::BracketRight);
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

        let increased = adjusted_bearing_dimensions(
            BearingDimensions::default(),
            BearingDimensionTarget::Outer,
            1,
        );
        assert!((increased.outer_diameter() - 0.30).abs() < 1.0e-6);
        assert!((increased.inner_diameter() - 0.10).abs() < f32::EPSILON);

        keyboard.press(KeyCode::ShiftLeft);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Tool::Bearing, false, false),
            Some((BearingDimensionTarget::Inner, 1))
        );
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

        let hud = tool_status_line(Tool::Bearing, settings.dimensions);
        assert!(hud.contains("Outer: 0.15 m"));
        assert!(hud.contains("Inner: 0.10 m"));
        assert!(hud.contains("Shift+["));
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
                &creation,
                &transforms,
                Vec3::new(0.0, 1.0, 5.0),
                Vec3::NEG_Z,
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
        let mut state = EditorState {
            hovered: Some(hit),
            preview: Some(candidate),
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();

        mouse.press(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 0);
        assert!(state.block_drag.is_some());

        mouse.clear();
        mouse.release(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 1);
        assert!(state.block_drag.is_none());
        assert_eq!(history.undo.len(), 1);
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
            .insert_resource(HotbarPointerCapture::default())
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
            .insert_resource(HotbarPointerCapture::default())
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
        let start = *graph.part(parts[0]).unwrap();

        let selected =
            delete_sheet_parts(&graph, start, IVec3::new(3, 1, 3), PlacementPlane::Xz).unwrap();

        assert_eq!(selected.len(), 4);
        assert!(!selected.contains(&parts[4]));
    }
}

#[cfg(test)]
mod history_tests {
    use bevy::prelude::{ButtonInput, IVec3, KeyCode, Vec3};
    use mechanic_core::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        CuboidSpec, FaceKind, FaceRef, GridRotation, PendingOperation, WeldSpec,
    };

    use super::{
        BlockAttachment, BlockDrag, DeleteDrag, DeleteTarget, EditorHistory, EditorSnapshot,
        EditorState, HISTORY_CAPACITY, HistoryAction, PlacedBearing, PlacementPlane, SurfaceHit,
        apply_history_action, bearing_attachment_candidate, requested_history_action,
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
            plane: PlacementPlane::Xz,
            last_endpoint: None,
            specs: vec![candidate.spec],
            error: None,
        });
        state.delete_drag = Some(DeleteDrag {
            start: *graph.part(support).unwrap(),
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
