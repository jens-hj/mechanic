//! Construction prototype with a GPU-physics preview.

#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

use std::collections::HashSet;

mod builder;
mod camera;
mod hotbar;
mod showcase;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    core_pipeline::tonemapping::Tonemapping,
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::PrimitiveTopology,
        renderer::{RenderDevice, RenderQueue},
    },
};
use builder::{
    BEARING_DEPTH, BEARING_DIAMETER, BLOCK_SIZE_METERS, GROUND_HALF_SIZE, PlacementCandidate,
    PlacementError, PlacementPlane, SurfaceHit, bearing_anchor_from_hit,
    bearing_attachment_candidate, begin_weld, block_sheet_specs, candidate_from_hit,
    face_geometry_from_ref, raycast_construction, raycast_oriented_cuboid, raycast_placement_plane,
    stage_bearing_attachment, stage_bearing_block_batch, stage_block_batch, stage_weld_objects,
    validate_block_batch,
};
use camera::OrbitCamera;
use hotbar::{HotbarPointerCapture, SelectedTool, Tool, shortcut_tool};
use mechanic_core::{
    BuildCommand, CompiledCreation, ConstructionGraph, CuboidSpec, FaceOwner, PartId,
    PendingOperation, TopologyError,
};
use mechanic_gpu::{FixedStepScheduler, GpuPhysics, GpuPhysicsConfig, GpuTransform};

const SIMULATION_VISUAL_TICK_INTERVAL: u32 = 2;
const HAMMER_CHARGE_SECONDS: f32 = 1.5;
const HAMMER_MIN_IMPULSE: f32 = 25.0;
const HAMMER_MAX_IMPULSE: f32 = 4_000.0;

#[derive(Resource, Default)]
struct EditorGraph(ConstructionGraph);

#[derive(Resource, Default)]
struct AppSimulation {
    gpu: Option<GpuPhysics>,
    creation: Option<CompiledCreation>,
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
    AutoWeld,
    Bearing {
        index: usize,
        source: mechanic_core::FaceRef,
        anchor: Vec3,
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

#[derive(Clone, Copy, Debug)]
struct PlacedBearing {
    source: mechanic_core::FaceRef,
    anchor: Vec3,
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
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if !keyboard.just_pressed(KeyCode::Space) {
        return;
    }
    state.cancel_showcase_replacement();
    if simulation.is_running() {
        *simulation = AppSimulation::default();
        state.construction_mesh_dirty = true;
        state.feedback = Some("Simulation stopped; returned to build mode".to_owned());
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
        mechanism_self_collisions: !is_showcase_graph(&graph.0),
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
        scheduler: FixedStepScheduler::new(),
        next_tick: 1,
        transforms,
        visual_ticks_since_publish: 0,
        static_mesh_dirty: true,
        render_dirty: true,
    };
    state.feedback = Some("Simulation running (throttled mesh preview)".to_owned());
}

fn is_showcase_graph(graph: &ConstructionGraph) -> bool {
    graph.part_count() == showcase::PART_COUNT
        && graph.weld_count() == showcase::WELD_COUNT
        && graph.bearing_count() == showcase::BEARING_COUNT
}

fn handle_showcase_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
    mut camera: Single<(&mut OrbitCamera, &mut Transform)>,
) {
    if simulation.is_running() || !keyboard.just_pressed(KeyCode::KeyP) {
        return;
    }
    if !showcase_press_requests_load(graph.0.part_count(), &mut state.showcase_replace_armed) {
        state.feedback =
            Some("Showcase replaces the current construction; press P again to confirm".to_owned());
        return;
    }

    let result = showcase::build().and_then(|candidate| {
        install_editor_graph(&mut graph.0, candidate).map_err(showcase::ShowcaseError::from)
    });
    match result {
        Ok(creation) => {
            debug_assert_eq!(creation.compounds.len(), showcase::COMPOUND_COUNT);
            clear_hover(&mut state);
            state.block_drag = None;
            state.delete_drag = None;
            state.delete_target = None;
            state.placed_bearings.clear();
            state.construction_mesh_dirty = true;
            state.feedback = Some(format!(
                "Showcase loaded: {} parts, {} welds, {} bearings, {} bodies — Space to simulate",
                graph.0.part_count(),
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
            state.feedback = Some(format!("Could not load showcase: {error}"));
        }
    }
}

fn showcase_press_requests_load(part_count: usize, armed: &mut bool) -> bool {
    if part_count > 0 && !*armed {
        *armed = true;
        false
    } else {
        *armed = false;
        true
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
            scheduler,
            next_tick,
            ..
        } = &mut *simulation;
        next_simulation_tick(scheduler, next_tick, time.delta())
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
        **bearing_visibility = Visibility::Hidden;
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
    **simulation_visibility = Visibility::Visible;
    simulation.render_dirty = false;
}

fn next_simulation_tick(
    scheduler: &mut FixedStepScheduler,
    next_tick: &mut u64,
    elapsed: std::time::Duration,
) -> Option<u64> {
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
}

#[derive(Resource, Default)]
struct EditorState {
    hovered: Option<SurfaceHit>,
    hovered_bearing: Option<usize>,
    preview: Option<PlacementCandidate>,
    preview_error: Option<PlacementError>,
    feedback: Option<String>,
    construction_mesh_dirty: bool,
    delete_target: Option<DeleteTarget>,
    showcase_replace_armed: bool,
    block_drag: Option<BlockDrag>,
    block_preview_revision: u64,
    delete_drag: Option<DeleteDrag>,
    delete_preview_revision: u64,
    placed_bearings: Vec<PlacedBearing>,
}

impl EditorState {
    fn cancel_showcase_replacement(&mut self) {
        self.showcase_replace_armed = false;
    }
}

#[derive(Resource)]
struct EditorVisuals {
    construction_mesh: Handle<Mesh>,
    simulation_mesh: Handle<Mesh>,
    bearing_mesh: Handle<Mesh>,
    cube_preview_mesh: Handle<Mesh>,
    bearing_preview_mesh: Handle<Mesh>,
    white_preview_material: Handle<StandardMaterial>,
    red_preview_material: Handle<StandardMaterial>,
    block_drag_preview_mesh: Handle<Mesh>,
    delete_drag_preview_mesh: Handle<Mesh>,
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
struct HelpText;

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
        .init_resource::<AppSimulation>()
        .init_resource::<HammerInteraction>()
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
                hotbar::update,
                camera::update_orbit_camera,
                handle_simulation_shortcut,
                handle_showcase_shortcut,
                handle_shortcuts,
                handle_tool_change,
                update_hover,
                handle_build_actions,
                handle_hammer_actions,
                sync_visual_meshes,
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
    let bearing_preview_mesh = meshes.add(Cylinder::new(BEARING_DIAMETER * 0.5, BEARING_DEPTH));
    let block_drag_preview_mesh = meshes.add(Cuboid::default());
    let delete_drag_preview_mesh = meshes.add(Cuboid::default());
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

    commands.insert_resource(EditorVisuals {
        construction_mesh: construction_mesh.clone(),
        simulation_mesh: simulation_mesh.clone(),
        bearing_mesh: bearing_mesh.clone(),
        cube_preview_mesh: cube_preview_mesh.clone(),
        bearing_preview_mesh,
        white_preview_material: white_preview_material.clone(),
        red_preview_material: red_preview_material.clone(),
        block_drag_preview_mesh,
        delete_drag_preview_mesh,
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
        Mesh3d(bearing_mesh),
        MeshMaterial3d(bearing_material),
        NoFrustumCulling,
        Visibility::Hidden,
        BearingVisual,
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
    commands.spawn((
        Name::new("Orbital camera"),
        Camera3d::default(),
        Tonemapping::None,
        orbit.transform(),
        orbit,
    ));

    commands.spawn((
        HelpText,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(17.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: px(14),
            left: px(14),
            padding: UiRect::all(px(10)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.02, 0.025, 0.035, 0.82)),
    ));
    hotbar::spawn(&mut commands);
}

fn handle_shortcuts(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut selection: ResMut<SelectedTool>,
    simulation: Res<AppSimulation>,
) {
    let editor_shortcut = keyboard.any_just_pressed([
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Escape,
        KeyCode::KeyQ,
    ]);
    if editor_shortcut {
        state.cancel_showcase_replacement();
    }
    for key in [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
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

fn handle_tool_change(
    selection: Res<SelectedTool>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
) {
    if !selection.is_changed() {
        return;
    }
    state.cancel_showcase_replacement();
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
        state.hovered = None;
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
    state.preview = None;
    state.preview_error = None;
}

fn refresh_tool_preview(graph: &ConstructionGraph, state: &mut EditorState, tool: Tool) {
    state.preview = None;
    state.preview_error = match (tool, graph.pending()) {
        (Tool::Block, _) => {
            if let Some(bearing) = state
                .hovered_bearing
                .and_then(|index| state.placed_bearings.get(index).copied())
            {
                let candidate = bearing_attachment_candidate(graph, bearing.source, bearing.anchor);
                let error =
                    stage_bearing_attachment(graph, candidate, bearing.source, bearing.anchor)
                        .err();
                state.preview = Some(candidate);
                error
            } else {
                state.hovered.and_then(|hit| {
                    let candidate = candidate_from_hit(graph, hit);
                    let error = validate_block_batch(graph, candidate, &[candidate.spec]).err();
                    state.preview = Some(candidate);
                    error
                })
            }
        }
        (Tool::Weld, Some(PendingOperation::Weld(first))) => state
            .hovered
            .and_then(|hit| stage_weld_objects(graph, first.owner, hit.face.owner).err()),
        (Tool::Weld | Tool::Hammer, _) => None,
        (Tool::Bearing, _) => state
            .hovered
            .and_then(|hit| bearing_anchor_from_hit(graph, hit).err()),
    };
}

#[allow(clippy::too_many_lines)] // Tool-specific input flows remain readable together.
fn handle_build_actions(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
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
    if mouse.any_just_pressed([MouseButton::Left, MouseButton::Right])
        || mouse.just_released(MouseButton::Right)
    {
        state.cancel_showcase_replacement();
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
                    if index < state.placed_bearings.len() {
                        state.placed_bearings.remove(index);
                        state.feedback = Some("Deleted unattached bearing".to_owned());
                        state.construction_mesh_dirty = true;
                        clear_hover(&mut state);
                    }
                }
            }
        }
        if let Some(drag) = state.delete_drag.take() {
            if let Some(error) = drag.error {
                state.feedback = Some(error.to_string());
                return;
            }
            let deleted = drag.parts.iter().copied().collect::<HashSet<_>>();
            match graph
                .0
                .apply_batch(drag.parts.iter().copied().map(BuildCommand::Remove))
            {
                Ok(_) => {
                    state.placed_bearings.retain(|bearing| {
                        !matches!(bearing.source.owner, FaceOwner::Part(owner) if deleted.contains(&owner))
                    });
                    state.feedback = Some(format!(
                        "Deleted {} cuboid(s) and incident connections",
                        drag.parts.len()
                    ));
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        return;
    }
    if selection.0 == Tool::Block {
        handle_block_actions(&mouse, &mut graph.0, &mut state);
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
                            graph.0 = staged;
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
                    let duplicate = state.placed_bearings.iter().any(|bearing| {
                        bearing.source == hit.face && bearing.anchor.abs_diff_eq(anchor, 1.0e-5)
                    });
                    if duplicate {
                        state.feedback = Some("A bearing is already placed here".to_owned());
                    } else {
                        state.placed_bearings.push(PlacedBearing {
                            source: hit.face,
                            anchor,
                        });
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
    }
    refresh_tool_preview(&graph.0, &mut state, selection.0);
}

fn handle_block_actions(
    mouse: &ButtonInput<MouseButton>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
) {
    if mouse.just_pressed(MouseButton::Left) {
        let Some(candidate) = state.preview else {
            state.feedback = Some("Point at the platform or a cuboid face".to_owned());
            return;
        };
        let (attachment, normal) = if let Some(index) = state.hovered_bearing {
            let Some(bearing) = state.placed_bearings.get(index).copied() else {
                state.feedback = Some("Bearing is no longer available".to_owned());
                return;
            };
            if let Some(error) =
                stage_bearing_attachment(graph, candidate, bearing.source, bearing.anchor).err()
            {
                state.feedback = Some(error.to_string());
                return;
            }
            (
                BlockAttachment::Bearing {
                    index,
                    source: bearing.source,
                    anchor: bearing.anchor,
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
                BlockAttachment::AutoWeld,
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
        state.feedback = Some(format!(
            "Dragging blocks on {} plane — release to place, Q changes plane",
            plane.label()
        ));
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
    let staged = match drag.attachment {
        BlockAttachment::AutoWeld => stage_block_batch(graph, drag.start, &drag.specs),
        BlockAttachment::Bearing { source, anchor, .. } => {
            stage_bearing_block_batch(graph, drag.start, &drag.specs, source, anchor)
        }
    };
    match staged {
        Ok(staged) => {
            let weld_count = staged.weld_count().saturating_sub(graph.weld_count());
            *graph = staged;
            if let BlockAttachment::Bearing {
                index,
                source,
                anchor,
            } = drag.attachment
                && state.placed_bearings.get(index).is_some_and(|bearing| {
                    bearing.source == source && bearing.anchor.abs_diff_eq(anchor, 1.0e-5)
                })
            {
                state.placed_bearings.remove(index);
            }
            state.feedback = Some(format!(
                "Placed {count} block(s); added {weld_count} weld(s){}",
                if matches!(drag.attachment, BlockAttachment::Bearing { .. }) {
                    " through bearing"
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
    if !selection.0.works_in_mode(true) {
        hammer.charging = None;
        if mouse.just_pressed(MouseButton::Left) && !hotbar_capture.active() {
            state.feedback = Some(format!(
                "{} is available in build mode — press Space",
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
    bearings
        .iter()
        .enumerate()
        .filter_map(|(index, bearing)| {
            let normal = face_geometry_from_ref(bearing.source, Some(graph)).normal;
            let rotation = Quat::from_rotation_arc(Vec3::Y, normal);
            let hit = raycast_oriented_cuboid(
                origin,
                direction,
                bearing.anchor,
                rotation,
                Vec3::new(
                    BEARING_DIAMETER * 0.5,
                    BEARING_DEPTH * 0.5,
                    BEARING_DIAMETER * 0.5,
                ),
            )?;
            Some((index, hit.distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
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
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rendered_block_revision: Local<u64>,
    mut rendered_delete_revision: Local<u64>,
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

    let action_material = if state.preview_error.is_none() {
        &visuals.white_preview_material
    } else {
        &visuals.red_preview_material
    };
    if let Some(bearing) = state
        .hovered_bearing
        .and_then(|index| state.placed_bearings.get(index))
    {
        let normal = face_geometry_from_ref(bearing.source, Some(&graph.0)).normal;
        show_bearing_preview(
            &mut selection,
            &visuals.bearing_preview_mesh,
            &visuals.white_preview_material,
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
            if let Some(part) = hovered_part(state.hovered)
                && let Some(spec) = graph.0.part(part)
            {
                show_cuboid_preview(
                    &mut action,
                    &visuals.cube_preview_mesh,
                    action_material,
                    *spec,
                    1.018,
                );
            }
            if let Some(PendingOperation::Weld(first)) = pending
                && let FaceOwner::Part(part) = first.owner
                && let Some(spec) = graph.0.part(part)
            {
                show_cuboid_preview(
                    &mut selection,
                    &visuals.cube_preview_mesh,
                    &visuals.white_preview_material,
                    *spec,
                    1.028,
                );
            }
        }
        (Tool::Bearing, _) => {
            if let Some(hit) = state.hovered {
                let face = face_geometry_from_ref(hit.face, Some(&graph.0));
                let anchor = bearing_anchor_from_hit(&graph.0, hit).unwrap_or(hit.point);
                show_bearing_preview(
                    &mut action,
                    &visuals.bearing_preview_mesh,
                    action_material,
                    anchor,
                    face.normal,
                );
            }
        }
        (Tool::Hammer, _) => {}
    }
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
    mut text: Single<&mut Text, With<HelpText>>,
) {
    let status = hammer.charging.map_or_else(
        || {
            state
                .delete_drag
                .as_ref()
                .and_then(|drag| drag.error.as_ref())
                .or(state.preview_error.as_ref())
                .map_or_else(
                    || state.feedback.clone().unwrap_or_else(|| "Ready".to_owned()),
                    ToString::to_string,
                )
        },
        |charge| {
            format!(
                "Hammer charge: {:.0}% — release to strike",
                charge.elapsed_seconds / HAMMER_CHARGE_SECONDS * 100.0
            )
        },
    );
    let tool_hint = if let Some(drag) = state.delete_drag.as_ref() {
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
            state.hovered_bearing,
        ) {
            (false, Tool::Block, _, Some(drag), _) => format!(
                "Release to place {} blocks on the {} plane",
                drag.specs.len(),
                drag.plane.label()
            ),
            (false, Tool::Block, _, None, Some(_)) => {
                "Bearing highlighted — click or drag to attach blocks through it".to_owned()
            }
            (true, Tool::Hammer, _, _, _) => {
                "Hold left mouse on a moving cuboid; release to strike".to_owned()
            }
            (true, _, _, _, _) => {
                "Selected tool is available in build mode — press Space".to_owned()
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
                "Hammer is available while simulating — press Space".to_owned()
            }
        }
    };
    let mode = if simulation.is_running() {
        "SIMULATING"
    } else {
        "BUILDING"
    };
    let action_controls = if state.delete_drag.is_some() {
        "Release Right Delete   Esc Cancel"
    } else {
        match (
            simulation.is_running(),
            selection.0,
            state.block_drag.is_some(),
        ) {
            (false, Tool::Block, true) => "Release Place   Right/Esc Cancel",
            (true, Tool::Hammer, _) => "Hold Left Hammer",
            (true, _, _) => "Left click No action",
            (false, _, _) => "Left click Action   Right drag Delete",
        }
    };
    let tool_line = if selection.0 == Tool::Block {
        format!("Tool: Block    Block size: {BLOCK_SIZE_METERS:.2} m")
    } else {
        format!("Tool: {}", selection.0.label())
    };
    let plane_controls = if let Some(drag) = state.block_drag.as_ref() {
        format!("Q Cycle Plane ({})", drag.plane.label())
    } else if let Some(drag) = state.delete_drag.as_ref() {
        format!("Q Cycle Delete Plane ({})", drag.plane.label())
    } else {
        "Q Cycle Plane While Dragging/Deleting".to_owned()
    };
    text.0 = format!(
        "MECHANIC — {mode}\n\
         {plane_controls}   P Load Showcase   Space Start/Stop\n\
         {action_controls}\n\
         Option+Left drag Orbit   Shift+Left drag Pan   Wheel Zoom\n\n\
         {tool_line}\n\
         Parts: {}   Welds: {}   Bearings: {}\n\
         {tool_hint}\n\
         {status}",
        graph.0.part_count(),
        graph.0.weld_count(),
        graph.0.bearing_count() + state.placed_bearings.len(),
    );
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
    let mut positions = Vec::with_capacity(specs.len() * CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(specs.len() * CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(specs.len() * CUBE_INDICES.len());
    for spec in specs {
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
    let vertices_per_bearing = SEGMENTS * 4 + 2;
    let indices_per_bearing = SEGMENTS * 12;
    let bearing_count = graph.bearing_count() + placed_bearings.len();
    let mut positions = Vec::with_capacity(bearing_count * vertices_per_bearing);
    let mut normals = Vec::with_capacity(bearing_count * vertices_per_bearing);
    let mut indices = Vec::with_capacity(bearing_count * indices_per_bearing);
    for (_, bearing) in graph.bearings() {
        append_bearing_cylinder(
            bearing.shared_anchor,
            bearing.axis,
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

fn append_bearing_cylinder(
    anchor: Vec3,
    axis: Vec3,
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
    let radius = BEARING_DIAMETER * 0.5;
    let half_depth = BEARING_DEPTH * 0.5;
    let lower = anchor - axis * half_depth;
    let upper = anchor + axis * half_depth;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");

    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * radius).to_array());
        positions.push((upper + radial * radius).to_array());
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
        positions.push((lower + radial * radius).to_array());
        normals.push((-axis).to_array());
    }
    let upper_ring = u32::try_from(positions.len()).unwrap();
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((upper + radial * radius).to_array());
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
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind, FaceRef,
        GridRotation,
    };

    use super::{
        BEARING_DEPTH, BEARING_DIAMETER, PlacedBearing, append_bearing_cylinder,
        combined_bearing_mesh,
    };

    #[test]
    fn bearing_visual_is_quarter_metre_wide_and_five_centimetres_each_side() {
        let anchor = Vec3::new(2.0, 3.0, 4.0);
        let axis = Vec3::X;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(anchor, axis, &mut positions, &mut normals, &mut indices);

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

        assert!((minimum_depth + BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!((maximum_depth - BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!((maximum_radius - BEARING_DIAMETER * 0.5).abs() < 1.0e-6);
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
        };

        let mesh = combined_bearing_mesh(&graph, &[bearing]);

        assert!(mesh.count_vertices() > 0);
        assert_eq!(graph.bearing_count(), 0);
    }
}

#[cfg(test)]
mod interaction_tests {
    use bevy::prelude::{App, ButtonInput, IVec3, MouseButton, Update, Vec3};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, FaceKind, FaceRef,
        GridRotation, PendingOperation,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        EditorGraph, EditorState, HAMMER_CHARGE_SECONDS, HAMMER_MAX_IMPULSE, HAMMER_MIN_IMPULSE,
        PlacedBearing, PlacementPlane, SelectedTool, SurfaceHit, Tool,
        bearing_attachment_candidate, candidate_from_hit, delete_sheet_parts,
        hammer_impulse_magnitude, handle_block_actions, handle_tool_change, raycast_construction,
        raycast_placed_bearings, raycast_simulation,
    };

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
            .insert_resource(EditorState {
                showcase_replace_armed: true,
                ..Default::default()
            })
            .insert_resource(SelectedTool(Tool::Bearing))
            .add_systems(Update, handle_tool_change);

        app.update();

        assert!(app.world().resource::<EditorGraph>().0.pending().is_none());
        assert!(!app.world().resource::<EditorState>().showcase_replace_armed);
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

        mouse.press(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state);
        assert_eq!(graph.part_count(), 0);
        assert!(state.block_drag.is_some());

        mouse.clear();
        mouse.release(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state);
        assert_eq!(graph.part_count(), 1);
        assert!(state.block_drag.is_none());
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
        };
        let mut state = EditorState {
            placed_bearings: vec![bearing],
            hovered_bearing: Some(0),
            preview: Some(bearing_attachment_candidate(
                &graph,
                bearing.source,
                bearing.anchor,
            )),
            ..Default::default()
        };

        let origin = Vec3::new(0.0, 3.0, 0.0);
        let (_, bearing_distance) =
            raycast_placed_bearings(&graph, &state.placed_bearings, origin, Vec3::NEG_Y).unwrap();
        let support_distance = raycast_construction(&graph, origin, Vec3::NEG_Y)
            .unwrap()
            .distance;
        assert!(bearing_distance < support_distance);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);

        let mut mouse = ButtonInput::default();
        mouse.press(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings.len(), 1);
        assert!(state.block_drag.is_some());

        mouse.clear();
        mouse.release(MouseButton::Left);
        handle_block_actions(&mouse, &mut graph, &mut state);

        assert!(state.placed_bearings.is_empty());
        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 0);
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
mod showcase_loading_tests {
    use std::time::Duration;

    use bevy::prelude::IVec3;
    use mechanic_core::{BuildCommand, BuildPose, CuboidSpec, GridRotation, TopologyError};
    use mechanic_gpu::FixedStepScheduler;

    use super::{
        ConstructionGraph, EditorState, install_editor_graph, is_showcase_graph,
        next_simulation_tick, showcase, showcase_press_requests_load,
    };

    #[test]
    fn app_simulation_drops_catch_up_backlog() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 1;

        assert_eq!(
            next_simulation_tick(&mut scheduler, &mut next_tick, Duration::from_secs(1)),
            Some(1)
        );
        assert_eq!(scheduler.next_tick(), 61);
        assert_eq!(
            next_simulation_tick(&mut scheduler, &mut next_tick, Duration::from_millis(17)),
            Some(2)
        );
    }

    #[test]
    fn empty_editor_loads_without_confirmation() {
        let mut armed = false;
        assert!(showcase_press_requests_load(0, &mut armed));
        assert!(!armed);

        let mut graph = ConstructionGraph::new();
        let creation = install_editor_graph(&mut graph, showcase::build().unwrap()).unwrap();
        assert_eq!(graph.part_count(), showcase::PART_COUNT);
        assert_eq!(creation.compounds.len(), showcase::COMPOUND_COUNT);
        assert!(is_showcase_graph(&graph));
    }

    #[test]
    fn populated_editor_requires_two_consecutive_presses() {
        let mut armed = false;
        assert!(!showcase_press_requests_load(1, &mut armed));
        assert!(armed);
        assert!(showcase_press_requests_load(1, &mut armed));
        assert!(!armed);
    }

    #[test]
    fn another_editor_action_cancels_replacement() {
        let mut state = EditorState {
            showcase_replace_armed: true,
            ..Default::default()
        };
        state.cancel_showcase_replacement();
        assert!(!state.showcase_replace_armed);
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
