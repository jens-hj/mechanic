//! Construction prototype with a GPU-physics preview.

#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

mod builder;
mod camera;
mod chroma;
mod control_panel;
mod controls;
mod creation_menu;
mod creation_store;
mod garage;
mod hotbar;
mod multitool;
mod pause_menu;
mod performance;
mod sequencer;
mod settings;
mod shape_tool;
mod showcase;
mod ui;
mod world;

use bevy::{
    app::AppExit,
    asset::RenderAssetUsages,
    camera::visibility::{NoFrustumCulling, RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::FrameTimeDiagnosticsPlugin,
    image::{ImageAddressMode, ImageFilterMode, ImageLoaderSettings},
    input::mouse::AccumulatedMouseScroll,
    mesh::Indices,
    pbr::ExtendedMaterial,
    prelude::*,
    render::{
        Render, RenderApp,
        mesh::allocator::MeshAllocatorSettings,
        render_resource::{
            Extent3d, PipelineCache, PrimitiveTopology, TextureDimension, TextureFormat,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderDevice, RenderQueue},
        slab_allocator::SlabAllocatorSettings,
    },
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use builder::{
    BEARING_DEPTH, BLOCK_SIZE_METERS, CylinderPlacementCandidate, PipeRunAttachment, PipeRunPiece,
    PlacementBounds, PlacementCandidate, PlacementError, PlacementGrid, PlacementPlane,
    PlacementSnapIndex, PlacementSupport, SmartGuide, SurfaceHit,
    bearing_anchor_from_hit_with_grid, bearing_attachment_candidate, bearing_overlaps_candidate,
    bearing_overlaps_cylinder_candidate, bearing_support_face, bearing_support_face_excluding,
    begin_weld, block_box_bounds, block_box_specs, block_span_from_rays,
    candidate_from_hit_with_grid, cylinder_candidate_from_hit_with_grid, face_geometry_from_ref,
    free_cuboid_candidate, free_cylinder_candidate, oriented_cuboid_candidate_from_hit_with_grid,
    part_world_bounds, pipe_run_pieces, raycast_construction, raycast_construction_for_annulus,
    raycast_construction_for_annulus_with_ground, raycast_construction_with_ground,
    raycast_oriented_cuboid, raycast_placement_plane_point, rigid_body_parts, smart_snap_anchor,
    smart_snap_block_span, smart_snap_cuboid_candidate, smart_snap_cylinder_candidate,
    smart_snap_free_cuboid_candidate, smart_snap_free_cylinder_candidate,
    stage_bearing_attachment_in_bounds, stage_bearing_block_batch_in_bounds,
    stage_bearing_cylinder_in_bounds, stage_block_batch_in_bounds, stage_controller_in_bounds,
    stage_dimension_link_in_bounds, stage_engine_in_bounds, stage_input_in_bounds,
    stage_pipe_run_in_bounds, stage_seat_in_bounds, stage_servo_in_bounds, stage_transmission,
    stage_weld_objects, transmission_candidate_from_hit_in_bounds, try_face_geometry_from_ref,
    validate_block_batch_in_bounds, validate_cylinder_candidate_in_bounds,
    validate_pipe_run_in_bounds,
};
#[cfg(test)]
use builder::{candidate_from_hit, stage_bearing_attachment};
use camera::{
    MainCamera, MaterialWheelState, PlayerCamera, PlayerState, SEATED_EYE_HEIGHT,
    seated_view_rotation,
};
use chroma::{ChromaBrush, ChromaMaterialExtension, ConstructionRenderMaterial};
use control_panel::ControlPanelState;
use controls::{BindingInput, GameAction, InputChord, Modifiers, WheelDirection};
use creation_menu::{CreationMenuState, CreationRequest};
use creation_store::CreationStore;
use hotbar::{SelectedMaterial, SelectedTerrainMaterial, SelectedTool, Tool};
use mechanic_core::{
    ActuatorAssignment, AppearanceTarget, BearingDimensions, BearingId, BearingSocket,
    BuildCommand, BuildOutcome, CYLINDER_SWEEP_STEP_DEGREES, CageIndex, CellGrid, ColliderShape,
    CompiledCreation, ConstructionEditDelta, ConstructionGraph, ConstructionMaterial,
    ControllerSpec, CreationDocument, CuboidSpec, CylinderDimensions, DimensionLinkSpec,
    DriveLinkSpec, DriveState, DriveTarget, EngineKind, FaceKind, FaceOwner, FaceRef,
    GRID_UNIT_METERS, GridRotation, InputSeatLinkSpec, InputSpec, MAX_BEARING_OUTER_DIAMETER,
    MAX_CYLINDER_OUTER_DIAMETER, MAX_CYLINDER_SWEEP_DEGREES, MIN_BEARING_DIAMETER_GAP,
    MIN_BEARING_OUTER_DIAMETER, MIN_CYLINDER_DIAMETER_GAP, MIN_CYLINDER_OUTER_DIAMETER,
    MIN_CYLINDER_SWEEP_DEGREES, MaterialAppearance, POSITION_TICK_METERS,
    POSITION_TICKS_PER_GRID_UNIT, POSITION_TICKS_PER_HALF_GRID_UNIT, PartId, PartPiece, PartSpec,
    PendingOperation, PipeBendDimensions, RegionId, STEP_METERS, STEPS_PER_CELL,
    SeatControllerLinkSpec, SeatSpec, ServoSpec, ShapeRegion, TopologyError, TransmissionSpec,
    face_neighbour_offset, part_cells,
};
use mechanic_gpu::{
    FIXED_DT_SECONDS, FixedStepScheduler, GpuPhysics, GpuPhysicsConfig, GpuTickReadback,
    GpuTransform, GpuVelocity,
};
use pause_menu::{PauseMenuState, PauseRequest};
use performance::PerformanceMetrics;
use sequencer::{DriveKeyState, DriveSequencer, GearboxRuntime, geared_gpu_drive_rows};
use settings::AppSettings;

const SIMULATION_VISUAL_TICK_INTERVAL: u64 = 2;
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
const DEBUG_FRAME_FREEZE_KEY: KeyCode = KeyCode::F8;

/// Shared signal from the render world once Bevy has populated both filtered
/// environment maps.
#[derive(Resource, Clone, Default)]
struct EnvironmentMapGenerationReady(Arc<AtomicBool>);

/// Retains Bevy's filtered environment map but turns its generator into a
/// one-shot operation.
struct OneShotEnvironmentMapPlugin;

/// Avoids repeatedly reallocating tiny GPU mesh slabs while the terrain
/// horizon publishes thousands of chunks over successive frames.
struct StreamingMeshAllocatorPlugin;

impl Plugin for StreamingMeshAllocatorPlugin {
    fn build(&self, app: &mut App) {
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("the render app exists after DefaultPlugins");
        render_app.insert_resource(MeshAllocatorSettings {
            slab_allocator_settings: SlabAllocatorSettings {
                min_slab_size: 8 * 1024 * 1024,
                growth_factor: 2.0,
                ..default()
            },
            ..default()
        });
    }
}

impl Plugin for OneShotEnvironmentMapPlugin {
    fn build(&self, app: &mut App) {
        let ready = EnvironmentMapGenerationReady::default();
        app.insert_resource(ready.clone())
            .add_systems(Update, retain_generated_environment_map);

        app.get_sub_app_mut(RenderApp)
            .expect("the render app exists after DefaultPlugins")
            .insert_resource(ready)
            .add_systems(
                Render,
                mark_environment_map_generated.after(bevy::pbr::generate::filtering_system),
            );
    }
}

fn mark_environment_map_generated(
    ready: Res<EnvironmentMapGenerationReady>,
    maps: Query<(), With<bevy::pbr::generate::GeneratorBindGroups>>,
    pipelines: Option<Res<bevy::pbr::generate::GeneratorPipelines>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(pipelines) = pipelines else {
        return;
    };
    if maps.is_empty() {
        return;
    }
    let pipeline_ids = [
        pipelines.downsample_first,
        pipelines.downsample_second,
        pipelines.copy,
        pipelines.radiance,
        pipelines.irradiance,
    ];
    if pipeline_ids
        .into_iter()
        .all(|id| pipeline_cache.get_compute_pipeline(id).is_some())
    {
        ready.0.store(true, Ordering::Release);
    }
}

fn retain_generated_environment_map(
    ready: Res<EnvironmentMapGenerationReady>,
    mut commands: Commands,
    maps: Query<
        Entity,
        (
            With<GeneratedEnvironmentMapLight>,
            With<EnvironmentMapLight>,
        ),
    >,
) {
    if !ready.0.swap(false, Ordering::AcqRel) {
        return;
    }
    for entity in &maps {
        commands
            .entity(entity)
            .remove::<GeneratedEnvironmentMapLight>();
    }
}

/// The overlay's cyan, matching the `accent.speed` the panels use. Selected
/// shape corners take it so a selection reads by colour and not only by size.
const SHAPE_SELECTION_COLOR: Color = Color::srgb(0.247, 0.796, 0.878);
/// Roughly the old five-pixel threshold at a typical desktop field of view.
pub(crate) const DRAG_DEAD_ZONE_RADIANS: f32 = 0.004;

#[derive(Resource, Default)]
struct EditorGraph(ConstructionGraph);

/// Display name of the creation currently open, when one was saved or loaded.
/// It prefills the modal's name field so re-saving keeps the same file.
#[derive(Resource, Default)]
struct CurrentCreation(Option<String>);

#[derive(Resource, Debug, Default, PartialEq, Eq)]
struct DebugFrameFreeze {
    active: bool,
    resume_pending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DebugFrameFreezeEffect {
    None,
    PauseTime,
    ResumeTime,
}

impl DebugFrameFreeze {
    fn advance(&mut self, freeze_pressed: bool, primary_pressed: bool) -> DebugFrameFreezeEffect {
        if self.resume_pending {
            self.resume_pending = false;
            return DebugFrameFreezeEffect::ResumeTime;
        }
        if self.active && primary_pressed {
            self.active = false;
            self.resume_pending = true;
            return DebugFrameFreezeEffect::None;
        }
        if !self.active && freeze_pressed {
            self.active = true;
            return DebugFrameFreezeEffect::PauseTime;
        }
        DebugFrameFreezeEffect::None
    }

    const fn blocks_updates(&self) -> bool {
        self.active || self.resume_pending
    }
}

fn update_debug_frame_freeze(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut freeze: ResMut<DebugFrameFreeze>,
    mut virtual_time: ResMut<Time<Virtual>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    match freeze.advance(
        keyboard.just_pressed(DEBUG_FRAME_FREEZE_KEY),
        mouse.just_pressed(MouseButton::Left),
    ) {
        DebugFrameFreezeEffect::PauseTime => virtual_time.pause(),
        DebugFrameFreezeEffect::ResumeTime => virtual_time.unpause(),
        DebugFrameFreezeEffect::None => {}
    }
    if freeze.blocks_updates() {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

fn debug_frame_updates_enabled(freeze: Res<DebugFrameFreeze>) -> bool {
    !freeze.blocks_updates()
}

#[cfg(test)]
mod debug_frame_freeze_tests {
    use super::{DebugFrameFreeze, DebugFrameFreezeEffect};

    #[test]
    fn freeze_stays_active_until_a_click_is_consumed() {
        let mut freeze = DebugFrameFreeze::default();
        assert_eq!(
            freeze.advance(true, false),
            DebugFrameFreezeEffect::PauseTime
        );
        assert!(freeze.blocks_updates());

        assert_eq!(freeze.advance(false, true), DebugFrameFreezeEffect::None);
        assert!(freeze.blocks_updates());

        assert_eq!(
            freeze.advance(false, false),
            DebugFrameFreezeEffect::ResumeTime
        );
        assert!(!freeze.blocks_updates());
    }
}

#[derive(Resource, Default)]
struct AppSimulation {
    gpu: Option<GpuPhysics>,
    creation: Option<CompiledCreation>,
    scheduler: FixedStepScheduler,
    next_tick: u64,
    tick_backlog: u64,
    completed_tick: u64,
    previous_transforms: Vec<GpuTransform>,
    transforms: Vec<GpuTransform>,
    previous_snapshot_tick: u64,
    snapshot_tick: u64,
    static_mesh_dirty: bool,
    render_dirty: bool,
    physics_cpu_ms: Option<f64>,
    last_tick_readback: Option<GpuTickReadback>,
    failure: Option<String>,
    world_revision: Option<(u64, u64)>,
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
    /// Smart guides acquired by the starting block before the gesture began.
    start_guides: Vec<SmartGuide>,
    /// Re-anchored whenever the plane rotates, so motion after Rotate is
    /// measured from that moment rather than from the original press.
    press: PointerSample,
    plane: PlacementPlane,
    /// Span at the last press or plane rotation. The plane's own two axes grow
    /// from here; the third keeps what it already had.
    anchor_span: IVec3,
    /// Blocks beyond the start block along each axis, signed.
    span: IVec3,
    last_span: Option<IVec3>,
    specs: Vec<CuboidSpec>,
    error: Option<PlacementError>,
}

/// A drag that claims an area of existing blocks for the Shape tool, using the
/// same gesture the Block tool places with.
#[derive(Clone, Debug)]
struct RegionDrag {
    /// The block first clicked, which anchors the area.
    start: CuboidSpec,
    /// Re-anchored whenever the plane rotates, so motion after Rotate is
    /// measured from that moment rather than from the original press.
    press: PointerSample,
    plane: PlacementPlane,
    /// Span at the last press or plane rotation.
    anchor_span: IVec3,
    /// Cells beyond the start block along each axis, signed.
    span: IVec3,
    last_span: Option<IVec3>,
    /// The area as it currently stands, whether or not it is claimable.
    region: ShapeRegion,
    /// Why the area cannot be claimed, if it cannot.
    error: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct PointerSample {
    #[cfg_attr(not(test), allow(dead_code))]
    cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
enum BlockAttachment {
    AutoWeld {
        source: FaceOwner,
    },
    Free,
    Bearing {
        source: mechanic_core::FaceRef,
        anchor: Vec3,
        dimensions: BearingDimensions,
    },
}

#[derive(Clone, Debug)]
struct DeleteDrag {
    start: CuboidSpec,
    /// Re-anchored whenever the plane rotates, so motion after Rotate is
    /// measured from that moment rather than from the original press.
    press: PointerSample,
    plane: PlacementPlane,
    /// Span at the last press or plane rotation.
    anchor_span: IVec3,
    /// Blocks beyond the start block along each axis, signed.
    span: IVec3,
    last_span: Option<IVec3>,
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

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
struct CylinderToolSettings {
    dimensions: CylinderDimensions,
    bend_radius: f32,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
struct SmartSnapSettings {
    enabled: bool,
    range: f32,
    scrolled_during_hold: bool,
    pub(crate) range_adjusted_this_frame: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FreePlacementSettings {
    range: f32,
    range_adjusted_this_frame: bool,
}

impl Default for FreePlacementSettings {
    fn default() -> Self {
        Self {
            range: 5.0,
            range_adjusted_this_frame: false,
        }
    }
}

impl FreePlacementSettings {
    fn update(&mut self, range_steps: f32, applicable: bool, object_snap_adjusted: bool) {
        self.range_adjusted_this_frame = false;
        if applicable && !object_snap_adjusted && range_steps != 0.0 {
            self.range = (self.range + range_steps * 0.25).clamp(0.25, 30.0);
            self.range_adjusted_this_frame = true;
        }
    }
}

impl Default for SmartSnapSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            range: 1.0,
            scrolled_during_hold: false,
            range_adjusted_this_frame: false,
        }
    }
}

impl SmartSnapSettings {
    fn update(&mut self, toggle_released: bool, range_steps: f32, used_during_hold: bool) {
        self.range_adjusted_this_frame = false;
        if range_steps != 0.0 {
            self.range = (self.range + range_steps * 0.25).clamp(0.25, 5.0);
            self.scrolled_during_hold = true;
            self.range_adjusted_this_frame = true;
        }
        self.scrolled_during_hold |= used_during_hold;
        if toggle_released {
            if !self.scrolled_during_hold {
                self.enabled = !self.enabled;
            }
            self.scrolled_during_hold = false;
        }
    }
}

fn update_smart_snap_settings(
    actions: Res<ButtonInput<GameAction>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut state: ResMut<EditorState>,
) {
    let adjustment = f32::from(actions.just_pressed(GameAction::ObjectSnapRangeIncrease))
        - f32::from(actions.just_pressed(GameAction::ObjectSnapRangeDecrease));
    let used_during_hold = actions.pressed(GameAction::ToggleObjectSnap)
        && mouse.any_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle]);
    state.smart_snap.update(
        actions.just_released(GameAction::ToggleObjectSnap),
        adjustment,
        used_during_hold,
    );
}

fn update_free_placement_settings(
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    space: Res<State<world::AppSpace>>,
    mut state: ResMut<EditorState>,
) {
    let adjustment = f32::from(actions.just_pressed(GameAction::FreePlacementRangeIncrease))
        - f32::from(actions.just_pressed(GameAction::FreePlacementRangeDecrease));
    let applicable = *space.get() == world::AppSpace::Garage
        && selection
            .active_editor_tool()
            .is_some_and(tool_supports_free_placement)
        && state.block_drag.is_none()
        && state.pipe_drag.is_none();
    let object_snap_adjusted = actions.just_pressed(GameAction::ObjectSnapRangeIncrease)
        || actions.just_pressed(GameAction::ObjectSnapRangeDecrease);
    state
        .free_placement
        .update(adjustment, applicable, object_snap_adjusted);
}

fn rebuild_placement_snap_index(graph: Res<EditorGraph>, mut state: ResMut<EditorState>) {
    if graph.is_changed() {
        state.snap_index.rebuild(&graph.0);
    }
}

fn active_placement_grid(actions: &ButtonInput<GameAction>) -> PlacementGrid {
    PlacementGrid::from_modifiers(
        actions.pressed(GameAction::FinePlacement),
        actions.pressed(GameAction::PrecisionPlacement),
    )
}

const fn tool_supports_free_placement(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::Block
            | Tool::Cylinder
            | Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
            | Tool::DimensionLink
    )
}

fn free_placement_point_on_miss(
    tool: Tool,
    bounds: PlacementBounds,
    origin: Vec3,
    direction: Vec3,
    range: f32,
    secondary_pressed: bool,
) -> Option<Vec3> {
    (bounds == PlacementBounds::GarageBuild
        && tool_supports_free_placement(tool)
        && !secondary_pressed)
        .then_some(origin + direction * range)
}

impl Default for CylinderToolSettings {
    fn default() -> Self {
        Self {
            dimensions: CylinderDimensions::default(),
            bend_radius: PipeBendDimensions::DEFAULT_RADIUS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PipeEditMode {
    #[default]
    Length,
    OuterDiameter,
    InnerDiameter,
}

impl PipeEditMode {
    const fn next(self) -> Self {
        match self {
            Self::Length => Self::OuterDiameter,
            Self::OuterDiameter => Self::InnerDiameter,
            Self::InnerDiameter => Self::Length,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Length => "Length",
            Self::OuterDiameter => "Outer diameter",
            Self::InnerDiameter => "Inner diameter",
        }
    }
}

#[derive(Clone, Debug)]
struct PipeDrag {
    attachment: BlockAttachment,
    start: Vec3,
    corners: Vec<Vec3>,
    endpoint: Vec3,
    directions: Vec<Vec3>,
    bend_radii: Vec<f32>,
    pending_radius: f32,
    dimensions: CylinderDimensions,
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
    mode: PipeEditMode,
    choosing_direction: bool,
    press: PointerSample,
    anchor_endpoint: Vec3,
    anchor_dimensions: CylinderDimensions,
    pieces: Vec<PipeRunPiece>,
    error: Option<PlacementError>,
}

#[derive(Clone, Debug)]
struct EditorSnapshot {
    graph: Arc<ConstructionGraph>,
    placed_bearings: Vec<PlacedBearing>,
    revision: u64,
}

#[derive(Clone, Debug)]
struct ChromaStroke {
    previous: EditorSnapshot,
    targets: HashSet<AppearanceTarget>,
    remove: bool,
    changed: bool,
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
            graph: Arc::new(graph),
            placed_bearings: state.placed_bearings.clone(),
            revision: 0,
        }
    }
}

#[derive(Resource, Default)]
struct EditorHistory {
    undo: VecDeque<EditorSnapshot>,
    redo: VecDeque<EditorSnapshot>,
    current_revision: u64,
    clean_revision: u64,
    next_revision: u64,
}

impl EditorHistory {
    fn commit(&mut self, mut previous: EditorSnapshot) {
        previous.revision = self.current_revision;
        self.redo.clear();
        if self.undo.len() == HISTORY_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(previous);
        self.next_revision = self.next_revision.saturating_add(1);
        self.current_revision = self.next_revision;
    }

    fn undo(&mut self, mut current: EditorSnapshot) -> Option<EditorSnapshot> {
        current.revision = self.current_revision;
        let previous = self.undo.pop_back()?;
        self.current_revision = previous.revision;
        self.redo.push_back(current);
        Some(previous)
    }

    fn redo(&mut self, mut current: EditorSnapshot) -> Option<EditorSnapshot> {
        current.revision = self.current_revision;
        let next = self.redo.pop_back()?;
        self.current_revision = next.revision;
        self.undo.push_back(current);
        Some(next)
    }

    const fn is_dirty(&self) -> bool {
        self.current_revision != self.clean_revision
    }

    fn mark_clean(&mut self) {
        self.clean_revision = self.current_revision;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HistoryAction {
    Undo,
    Redo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EscapeTarget {
    PauseSubmenu,
    PauseMenu,
    ExistingUi,
    ControlPanel,
    WorldState,
    OpenPause,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExitDisposition {
    Exit,
    ConfirmUnsaved,
}

const fn exit_disposition(construction_dirty: bool) -> ExitDisposition {
    if construction_dirty {
        ExitDisposition::ConfirmUnsaved
    } else {
        ExitDisposition::Exit
    }
}

#[allow(clippy::fn_params_excessive_bools)]
// The booleans are independent, ordered input owners; the return value is their priority.
const fn escape_target(
    pause_open: bool,
    submenu_open: bool,
    existing_ui: bool,
    panel_open: bool,
    world_state: bool,
) -> EscapeTarget {
    if pause_open && submenu_open {
        EscapeTarget::PauseSubmenu
    } else if pause_open {
        EscapeTarget::PauseMenu
    } else if existing_ui {
        EscapeTarget::ExistingUi
    } else if panel_open {
        EscapeTarget::ControlPanel
    } else if world_state {
        EscapeTarget::WorldState
    } else {
        EscapeTarget::OpenPause
    }
}

fn begin_pause_frame(mut pause: ResMut<PauseMenuState>) {
    pause.begin_frame();
}

#[allow(clippy::too_many_arguments)]
fn handle_pause_escape(
    keyboard: Res<ButtonInput<KeyCode>>,
    overlay: Res<ui::UiInput>,
    menu: Res<CreationMenuState>,
    mut panel: ResMut<ControlPanelState>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    mut wheel: ResMut<MaterialWheelState>,
    mut pause: ResMut<PauseMenuState>,
    worlds: Res<world::WorldListState>,
) {
    if worlds.is_open() || !keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    if pause.binding_capture().is_some() {
        pause.cancel_binding_capture();
        return;
    }
    let world_state = state.block_drag.is_some()
        || state.pipe_drag.is_some()
        || state.delete_drag.is_some()
        || graph.0.pending().is_some()
        || state.wire_drag.is_some()
        || shape_tool_is_busy(selection.active_editor_tool(), &state);
    let target = escape_target(
        pause.is_open(),
        pause.is_in_submenu(),
        menu.is_open() || overlay.escape_is_consumed(),
        panel.is_open(),
        world_state,
    );
    pause.consume_frame();
    match target {
        EscapeTarget::PauseSubmenu => pause.return_to_main(),
        EscapeTarget::PauseMenu => pause.close(),
        EscapeTarget::ExistingUi => {}
        EscapeTarget::ControlPanel => {
            panel.close();
            state.feedback = Some("Control block panel closed".to_owned());
        }
        EscapeTarget::WorldState => cancel_one_world_escape_owner(&mut graph.0, &mut state),
        EscapeTarget::OpenPause => {
            wheel.close();
            pause.open();
        }
    }
}

fn cancel_one_world_escape_owner(graph: &mut ConstructionGraph, state: &mut EditorState) {
    if state.block_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Block drag cancelled".to_owned());
    } else if state.pipe_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Pipe run cancelled".to_owned());
    } else if state.delete_drag.take().is_some() {
        clear_hover(state);
        state.feedback = Some("Delete drag cancelled".to_owned());
    } else if graph.pending().is_some() {
        let _ = graph.apply(BuildCommand::CancelPending);
        state.feedback = Some("Selection cancelled".to_owned());
    } else if state.wire_drag.take().is_some() {
        state.feedback = Some("Wire drag cancelled".to_owned());
    } else if state.region_drag.take().is_some() {
        state.feedback = Some("Area selection cancelled".to_owned());
    } else if state.vertex_drag.take().is_some() {
        state.construction_mesh_dirty = true;
        state.feedback = Some("Shape drag cancelled".to_owned());
    } else if state.paint_selecting || !state.selected_vertices.is_empty() {
        state.paint_selecting = false;
        state.selected_vertices.clear();
        state.feedback = Some("Selection cleared".to_owned());
    } else if state.active_region.take().is_some() {
        state.construction_mesh_dirty = true;
        state.feedback = Some("Left the region".to_owned());
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_pause_request(
    mut pause: ResMut<PauseMenuState>,
    mut settings: ResMut<AppSettings>,
    history: Res<EditorHistory>,
    space: Res<State<world::AppSpace>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(request) = pause.take_request() else {
        return;
    };
    match request {
        PauseRequest::Continue => pause.close(),
        PauseRequest::OpenOptions => pause.open_options(),
        PauseRequest::OpenControls => pause.open_controls(),
        PauseRequest::Back | PauseRequest::CancelExit => pause.return_to_main(),
        PauseRequest::SetCameraFov(degrees) => {
            if let Err(error) = settings.set_camera_fov_degrees(degrees) {
                warn!("could not save settings: {error}");
            }
        }
        PauseRequest::BeginBindingCapture(_) => {}
        PauseRequest::ClearBinding(action, slot) => {
            if let Err(error) = settings.set_binding(action, slot, None) {
                warn!("could not save settings: {error}");
            }
        }
        PauseRequest::ResetControls => {
            if let Err(error) = settings.reset_controls() {
                warn!("could not save settings: {error}");
            }
        }
        PauseRequest::Exit => {
            match exit_disposition(*space.get() == world::AppSpace::Garage && history.is_dirty()) {
                ExitDisposition::Exit => {
                    exit.write(AppExit::Success);
                }
                ExitDisposition::ConfirmUnsaved => pause.confirm_exit(),
            }
        }
        PauseRequest::ExitWithoutSaving => {
            exit.write(AppExit::Success);
        }
    }
}

fn capture_control_binding(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut pause: ResMut<PauseMenuState>,
    mut settings: ResMut<AppSettings>,
) {
    let Some(capture) = pause.binding_capture() else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    let modifiers = Modifiers::from_keyboard(&keyboard);
    let modifier_key = |key| {
        matches!(
            key,
            KeyCode::ShiftLeft
                | KeyCode::ShiftRight
                | KeyCode::ControlLeft
                | KeyCode::ControlRight
                | KeyCode::AltLeft
                | KeyCode::AltRight
                | KeyCode::SuperLeft
                | KeyCode::SuperRight
        )
    };
    let input = keyboard
        .get_just_pressed()
        .copied()
        .find(|key| !modifier_key(*key))
        .map(BindingInput::Key)
        .or_else(|| {
            mouse
                .get_just_pressed()
                .next()
                .copied()
                .map(BindingInput::Mouse)
        })
        .or_else(|| {
            if scroll.delta.y > 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Up))
            } else if scroll.delta.y < 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Down))
            } else if scroll.delta.x < 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Left))
            } else if scroll.delta.x > 0.0 {
                Some(BindingInput::Wheel(WheelDirection::Right))
            } else {
                None
            }
        });
    let Some(input) = input else {
        return;
    };
    if matches!(input, BindingInput::Wheel(_)) && !capture.action.instantaneous() {
        return;
    }
    let chord = InputChord { input, modifiers };
    if let Err(error) = settings.set_binding(capture.action, capture.slot, Some(chord)) {
        warn!("could not save settings: {error}");
    }
    pause.finish_binding_capture(chord);
}

#[derive(Clone, Copy, Debug)]
enum DeleteTarget {
    PlacedBearing(usize),
    Part(PartId),
}

#[allow(clippy::too_many_arguments)]
fn maintain_space_simulation(
    space: Res<State<world::AppSpace>>,
    worlds: Res<world::WorldListState>,
    graph: Res<EditorGraph>,
    history: Res<EditorHistory>,
    runtime: Res<world::WorldRuntime>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if *space.get() == world::AppSpace::Garage {
        if simulation.gpu.is_some() {
            simulation.gpu = None;
            simulation.world_revision = None;
            *hammer = HammerInteraction::default();
            state.construction_mesh_dirty = true;
        }
        return;
    }
    if worlds.is_open() {
        simulation.gpu = None;
        simulation.world_revision = None;
        return;
    }

    let revision = (history.current_revision, runtime.foundation_revision());
    if simulation.world_revision == Some(revision) {
        return;
    }
    if graph.0.part_count() == 0 {
        *simulation = AppSimulation {
            world_revision: Some(revision),
            ..default()
        };
        return;
    }
    let anchored = runtime.anchored_parts().collect::<Vec<_>>();
    let creation = match graph.0.compile_with_static_parts(anchored) {
        Ok(creation) => creation,
        Err(error) => {
            *simulation = AppSimulation {
                world_revision: Some(revision),
                ..default()
            };
            state.feedback = Some(format!("Cannot update live world physics: {error}"));
            return;
        }
    };
    let (transforms, velocities) = rebuilt_body_states(&creation, &simulation);
    let next_tick = simulation.next_tick.max(1);
    if !creation_requires_live_physics(&creation) {
        *simulation = AppSimulation {
            creation: Some(creation),
            next_tick,
            previous_transforms: transforms.clone(),
            transforms,
            previous_snapshot_tick: next_tick.saturating_sub(1),
            snapshot_tick: next_tick,
            world_revision: Some(revision),
            ..default()
        };
        return;
    }
    let physics_config = GpuPhysicsConfig {
        ground_plane_enabled: false,
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
            *simulation = AppSimulation {
                world_revision: Some(revision),
                ..default()
            };
            state.feedback = Some(format!("Cannot update live world physics: {error}"));
            return;
        }
    };
    gpu.enable_async_readback();
    if let Err(error) = gpu.write_body_states(&render_queue, &transforms, &velocities) {
        state.feedback = Some(format!("Cannot preserve live world body state: {error}"));
    }
    *simulation = AppSimulation {
        gpu: Some(gpu),
        creation: Some(creation),
        scheduler: FixedStepScheduler::new(),
        next_tick,
        tick_backlog: 0,
        completed_tick: next_tick.saturating_sub(1),
        previous_transforms: transforms.clone(),
        transforms,
        previous_snapshot_tick: next_tick.saturating_sub(2),
        snapshot_tick: next_tick.saturating_sub(1),
        static_mesh_dirty: true,
        render_dirty: true,
        physics_cpu_ms: None,
        last_tick_readback: None,
        failure: None,
        world_revision: Some(revision),
    };
}

fn creation_requires_live_physics(creation: &CompiledCreation) -> bool {
    creation
        .compounds
        .iter()
        .any(|compound| !compound.is_static)
}

fn rebuilt_body_states(
    creation: &CompiledCreation,
    previous: &AppSimulation,
) -> (Vec<GpuTransform>, Vec<GpuVelocity>) {
    let mut transforms = creation
        .compounds
        .iter()
        .map(|compound| GpuTransform {
            position: compound.root_translation.extend(0.0).to_array(),
            rotation: compound.root_rotation.to_array(),
        })
        .collect::<Vec<_>>();
    let mut velocities = vec![
        GpuVelocity {
            linear: [0.0; 4],
            angular: [0.0; 4],
        };
        creation.compounds.len()
    ];
    let Some(previous_creation) = previous.creation.as_ref() else {
        return (transforms, velocities);
    };
    let tick_delta = previous
        .snapshot_tick
        .saturating_sub(previous.previous_snapshot_tick);
    let tick_delta = u16::try_from(tick_delta).unwrap_or(u16::MAX);
    let elapsed = f32::from(tick_delta) * FIXED_DT_SECONDS;
    for (new_index, compound) in creation.compounds.iter().enumerate() {
        if compound.is_static {
            continue;
        }
        let Some(old_index) = previous_creation
            .compounds
            .iter()
            .position(|old| !old.is_static && old.source_parts == compound.source_parts)
        else {
            continue;
        };
        let Some(&current) = previous.transforms.get(old_index) else {
            continue;
        };
        transforms[new_index] = current;
        if elapsed <= f32::EPSILON {
            continue;
        }
        let prior = previous
            .previous_transforms
            .get(old_index)
            .copied()
            .unwrap_or(current);
        let current_position = Vec3::from_slice(&current.position[..3]);
        let prior_position = Vec3::from_slice(&prior.position[..3]);
        let current_rotation = Quat::from_array(current.rotation);
        let prior_rotation = Quat::from_array(prior.rotation);
        let delta = (current_rotation * prior_rotation.inverse()).normalize();
        let (axis, angle) = delta.to_axis_angle();
        velocities[new_index] = GpuVelocity {
            linear: ((current_position - prior_position) / elapsed)
                .extend(0.0)
                .to_array(),
            angular: (axis * (angle / elapsed)).extend(0.0).to_array(),
        };
    }
    (transforms, velocities)
}

/// Whether the primary modifier plus `S` was pressed this frame.
///
/// A modifier is required because a bare letter binds to a drive state, so
/// plain `S` belongs to a machine rather than to the editor.
fn save_shortcut_requested(actions: &ButtonInput<GameAction>) -> bool {
    actions.just_pressed(GameAction::Save)
}

/// Opens the creations modal with `P`, or with the primary modifier and `S`.
///
/// While it is open the modal owns the keyboard, so neither key reaches here:
/// `p` and `s` type into its name field, and Escape is its own to handle. The
/// control-block panel owns the keyboard the same way, and the two must never
/// both be typing, so neither can open over the other.
#[allow(clippy::too_many_arguments)] // Bevy system resources are explicit parameters.
fn handle_creation_menu_shortcut(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    space: Res<State<world::AppSpace>>,
    store: Res<CreationStore>,
    current: Res<CurrentCreation>,
    panel: Res<ControlPanelState>,
    pause: Res<PauseMenuState>,
    mut menu: ResMut<CreationMenuState>,
    worlds: Res<world::WorldListState>,
) {
    if worlds.is_open() || menu.is_open() || panel.blocks_keyboard() || pause.blocks_world_input() {
        return;
    }
    let saving = save_shortcut_requested(&actions);
    if !saving && !actions.just_pressed(GameAction::Creations) {
        return;
    }
    if *space.get() == world::AppSpace::World {
        state.feedback =
            Some("Saved creations are managed in the Garage — press F6 first".to_owned());
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

#[allow(clippy::too_many_arguments)] // Bevy system resources are explicit parameters.
fn handle_dimension_link_interaction(
    actions: Res<ButtonInput<GameAction>>,
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut runtime: ResMut<world::WorldRuntime>,
    space: Res<State<world::AppSpace>>,
    player: Res<PlayerState>,
    overlay: Res<ui::UiInput>,
    wheel: Res<MaterialWheelState>,
) {
    if !actions.just_pressed(GameAction::Interact)
        || !player.world_input_active()
        || overlay.blocks_keyboard()
        || wheel.open
    {
        return;
    }
    let aimed = state
        .hovered
        .and_then(|hit| match hit.face.owner {
            FaceOwner::Part(part) => Some((part, hit.distance)),
            FaceOwner::Ground => None,
        })
        .or_else(|| state.hovered_simulation.map(|hit| (hit.part, hit.distance)));
    let Some((part, distance)) = aimed else {
        return;
    };
    let Some(id) = graph.0.dimension_link_id(part) else {
        return;
    };
    if distance > 3.0 {
        state.feedback = Some("Dimension Link is out of interaction range (3 m)".to_owned());
        return;
    }
    state.feedback = Some(
        match runtime.activate_dimension_link(*space.get(), &graph.0, part) {
            Ok(_) => format!("Activated Dimension Link {}", id.0),
            Err(error) => error,
        },
    );
    state.construction_mesh_dirty = true;
}

/// Opens or closes the control-block panel with `E`.
///
/// The panel targets the hovered control block, falling back to the selected
/// one, so it opens both from the world and from whatever was last wired.
#[allow(clippy::too_many_arguments)]
fn handle_control_panel_shortcut(
    actions: Res<ButtonInput<GameAction>>,
    menu: Res<CreationMenuState>,
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut panel: ResMut<ControlPanelState>,
    space: Res<State<world::AppSpace>>,
    simulation: Res<AppSimulation>,
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
    if !actions.just_pressed(GameAction::Interact) || !player.world_input_active() || wheel.open {
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
                .selected_controller
                .filter(|&part| graph.0.is_controller(part))
        });
    let Some(controller) = target else {
        state.feedback = Some("Point at a control block, or select one, then press E".to_owned());
        return;
    };
    if *space.get() == world::AppSpace::World && !simulation_part_is_static(&simulation, controller)
    {
        state.feedback = Some("Moving constructions cannot be programmed".to_owned());
        return;
    }
    state.selected_controller = Some(controller);
    panel.open(controller);
}

#[allow(clippy::too_many_arguments)]
fn handle_seat_interaction(
    actions: Res<ButtonInput<GameAction>>,
    overlay: Res<ui::UiInput>,
    wheel: Res<MaterialWheelState>,
    graph: Res<EditorGraph>,
    simulation: Res<AppSimulation>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut camera: Single<
        (
            &Camera,
            &mut PlayerCamera,
            &mut Transform,
            &mut GlobalTransform,
        ),
        With<MainCamera>,
    >,
    mut player: ResMut<PlayerState>,
    mut state: ResMut<EditorState>,
) {
    if !simulation.is_running() {
        if player.seat.is_some() {
            let camera_position = camera.2.translation;
            player.leave_seat_at(camera_position);
        }
        return;
    }

    if actions.just_pressed(GameAction::Interact)
        && player.world_input_active()
        && !overlay.blocks_keyboard()
        && !wheel.open
    {
        if let Some(seat) = player.seat {
            let beside = seat_world_pose(&graph.0, &simulation, seat)
                .map_or(camera.2.translation, |(centre, rotation)| {
                    centre + rotation * (Vec3::X * 0.9)
                });
            player.leave_seat_at(beside);
            state.feedback = Some("Left Seat".to_owned());
            return;
        }
        let (camera_component, _, _, camera_global) = &mut *camera;
        let cursor_position = camera::viewport_center(Vec2::new(window.width(), window.height()));
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
            && camera::seat_entry_allowed(hit.distance, graph.0.is_seat(hit.part))
        {
            player.seat = Some(hit.part);
            camera.1.yaw = 0.0;
            camera.1.pitch = 0.0;
            state.feedback = Some("Seated — mouse looks around, E leaves the Seat".to_owned());
        } else {
            state.feedback = Some(if hit.is_some() {
                "Seat must be under the reticle and within 3 m".to_owned()
            } else {
                "Aim the reticle at a Seat within 3 m and press E".to_owned()
            });
        }
    }

    let Some(seat) = player.seat else {
        return;
    };
    let Some((seat_center, seat_rotation)) = seat_world_pose(&graph.0, &simulation, seat) else {
        player.seat = None;
        return;
    };
    let (_, view, transform, global) = &mut *camera;
    let rotation = seated_view_rotation(seat_rotation, view.yaw, view.pitch);
    **transform = view.apply_pullback(
        seat_center + seat_rotation * (Vec3::Y * SEATED_EYE_HEIGHT),
        rotation,
    );
    **global = GlobalTransform::from(**transform);
}

pub(crate) fn seat_world_pose(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    seat: PartId,
) -> Option<(Vec3, Quat)> {
    let creation = simulation.creation.as_ref()?;
    let compound = creation
        .part_to_compound
        .iter()
        .find_map(|(part, compound)| (*part == seat).then_some(*compound))?;
    let snapshot = simulation.transforms.get(compound as usize)?;
    let PartSpec::Seat(spec) = graph.part(seat).copied()? else {
        return None;
    };
    let root_position = Vec3::from_slice(&snapshot.position[..3]);
    let root_rotation = Quat::from_array(snapshot.rotation);
    let seat_rotation = root_rotation * spec.pose.rotation.quaternion();
    let seat_center = root_position
        + root_rotation
            * (spec.pose.translation() - creation.compounds[compound as usize].root_translation);
    Some((seat_center, seat_rotation))
}

/// Advances every driven bearing's program and pushes changed rows to the GPU.
///
/// Runs immediately before the tick is dispatched, so a state entered this
/// frame takes effect in the same tick rather than the next one.
#[allow(clippy::too_many_arguments)] // Bevy systems receive each independent resource explicitly.
fn run_drive_sequencer(
    keyboard: Res<ButtonInput<KeyCode>>,
    graph: Res<EditorGraph>,
    overlay: Res<ui::UiInput>,
    simulation: Res<AppSimulation>,
    mut sequencer: ResMut<DriveSequencer>,
    mut gearboxes: ResMut<GearboxRuntime>,
    mut state: ResMut<EditorState>,
    player: Res<PlayerState>,
) {
    if !simulation.is_running() {
        if sequencer.is_started() {
            sequencer.stop();
            gearboxes.stop();
        }
        return;
    }
    if !sequencer.is_started() {
        let Some(creation) = simulation.creation.as_ref() else {
            return;
        };
        sequencer.start(creation, &graph.0);
        gearboxes.start(&graph.0, &sequencer);
        state.drive_rows_dirty = true;
    }
    let keys = DriveKeyState::from_keyboard(&keyboard, overlay.blocks_keyboard());
    let keyboard_controller = player
        .seat
        .filter(|seat| graph.0.seat_input(*seat).is_some())
        .and_then(|seat| graph.0.seat_controller(seat));
    let sequencer_changed =
        sequencer.step(&graph.0, &keys, keyboard_controller, simulation.next_tick);
    let measured_speeds = measured_engine_speeds(&graph.0, &simulation, &sequencer);
    let gearbox_changed = gearboxes.step(
        &graph.0,
        &sequencer,
        &keyboard,
        (!overlay.blocks_keyboard())
            .then_some(keyboard_controller)
            .flatten(),
        simulation.next_tick,
        &measured_speeds,
        false,
    );
    if sequencer_changed || gearbox_changed {
        state.drive_rows_dirty = true;
    }
}

/// Signed joint speeds from the two transform snapshots already read for rendering.
#[allow(clippy::cast_precision_loss)]
fn measured_engine_speeds(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    sequencer: &DriveSequencer,
) -> Vec<(PartId, EngineKind, f32)> {
    let Some(creation) = simulation.creation.as_ref() else {
        return Vec::new();
    };
    let tick_delta = simulation
        .snapshot_tick
        .saturating_sub(simulation.previous_snapshot_tick);
    if tick_delta == 0 || simulation.previous_transforms.len() != simulation.transforms.len() {
        return Vec::new();
    }
    let delta_seconds = tick_delta as f32 * FIXED_DT_SECONDS;
    let mut result = Vec::<(PartId, EngineKind, f32)>::new();
    for row in sequencer.rows() {
        let Some(link) = graph.drive_link(row.link) else {
            continue;
        };
        let Some(bearing) = creation
            .bearings
            .iter()
            .find(|bearing| bearing.coordinate_index == Some(row.coordinate))
        else {
            continue;
        };
        let a = bearing.compound_a as usize;
        let b = bearing.compound_b as usize;
        let (Some(previous_a), Some(previous_b), Some(current_a), Some(current_b)) = (
            simulation.previous_transforms.get(a),
            simulation.previous_transforms.get(b),
            simulation.transforms.get(a),
            simulation.transforms.get(b),
        ) else {
            continue;
        };
        let speed = signed_joint_speed(
            Quat::from_array(previous_a.rotation),
            Quat::from_array(previous_b.rotation),
            Quat::from_array(current_a.rotation),
            Quat::from_array(current_b.rotation),
            bearing.local_axis_a,
            delta_seconds,
        );
        for kind in [EngineKind::Electric, EngineKind::Gas] {
            let powered = match kind {
                EngineKind::Electric => link.actuator.uses_electric(),
                EngineKind::Gas => link.actuator.uses_gas(),
            };
            if !powered {
                continue;
            }
            if let Some(entry) = result.iter_mut().find(|(controller, candidate, _)| {
                *controller == link.controller && *candidate == kind
            }) {
                if speed.abs() > entry.2.abs() {
                    entry.2 = speed;
                }
            } else {
                result.push((link.controller, kind, speed));
            }
        }
    }
    result
}

fn signed_joint_speed(
    previous_a: Quat,
    previous_b: Quat,
    current_a: Quat,
    current_b: Quat,
    local_axis_a: Vec3,
    delta_seconds: f32,
) -> f32 {
    let angular_a = (current_a * previous_a.inverse()).to_scaled_axis() / delta_seconds;
    let angular_b = (current_b * previous_b.inverse()).to_scaled_axis() / delta_seconds;
    (angular_b - angular_a).dot(current_a * local_axis_a)
}

#[cfg(test)]
mod transmission_speed_tests {
    use bevy::prelude::{Quat, Vec3};

    use super::signed_joint_speed;

    #[test]
    fn snapshot_pairs_measure_signed_relative_joint_speed() {
        let delta_seconds = 0.1;
        let positive = signed_joint_speed(
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::from_axis_angle(Vec3::Z, 0.2),
            Vec3::Z,
            delta_seconds,
        );
        let negative = signed_joint_speed(
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::from_axis_angle(Vec3::Z, -0.2),
            Vec3::Z,
            delta_seconds,
        );
        assert!((positive - 2.0).abs() < 1.0e-5);
        assert!((negative + 2.0).abs() < 1.0e-5);
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
    mut player: ResMut<PlayerState>,
    mut camera: Single<(&mut PlayerCamera, &mut Transform, &mut GlobalTransform), With<MainCamera>>,
    mut world_runtime: ResMut<world::WorldRuntime>,
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
                    adopt_loaded_creation(
                        &mut state,
                        &mut player,
                        &mut camera,
                        &graph.0,
                        Vec::new(),
                    );
                    state.feedback = Some(format!(
                        "Opened {}: {} welds, {} bearings, {} bodies — F6 enters the live World",
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
            match load_creation_remapped(&mut graph.0, &path, &mut world_runtime) {
                Ok((name, sockets, creation)) => {
                    history.commit(previous);
                    history.mark_clean();
                    let bodies = creation.as_ref().map(|compiled| compiled.compounds.len());
                    let placed = sockets
                        .into_iter()
                        .map(|socket| PlacedBearing {
                            source: socket.source,
                            anchor: socket.anchor,
                            dimensions: socket.dimensions,
                        })
                        .collect();
                    adopt_loaded_creation(&mut state, &mut player, &mut camera, &graph.0, placed);
                    state.feedback = Some(if let Some(bodies) = bodies {
                        format!(
                            "Opened \"{name}\": {} parts, {} bearings, {bodies} bodies — F6 enters the live World",
                            graph.0.part_count(),
                            graph.0.bearing_count(),
                        )
                    } else {
                        format!(
                            "Opened \"{name}\" — complete matching transmission stacks before entering the World"
                        )
                    });
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
                    history.mark_clean();
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
type LoadedCreation = (String, Vec<BearingSocket>, Option<CompiledCreation>);

fn load_creation_remapped(
    current: &mut ConstructionGraph,
    path: &Path,
    runtime: &mut world::WorldRuntime,
) -> Result<LoadedCreation, Box<dyn Error>> {
    let mut document = creation_store::read_document(path)?;
    runtime.remap_imported_dimension_links(&mut document);
    let loaded = document.into_graph()?;
    match loaded.graph.compile() {
        Ok(creation) => {
            *current = loaded.graph;
            Ok((loaded.name, loaded.sockets, Some(creation)))
        }
        Err(TopologyError::TransmissionDepthMismatch { .. }) => {
            *current = loaded.graph;
            Ok((loaded.name, loaded.sockets, None))
        }
        Err(error) => Err(Box::new(error)),
    }
}

/// Clears the transient editing state a freshly opened creation invalidates,
/// then frames the camera on what arrived.
fn adopt_loaded_creation(
    state: &mut EditorState,
    player: &mut PlayerState,
    camera: &mut Single<
        (&mut PlayerCamera, &mut Transform, &mut GlobalTransform),
        With<MainCamera>,
    >,
    graph: &ConstructionGraph,
    placed_bearings: Vec<PlacedBearing>,
) {
    clear_hover(state);
    state.block_drag = None;
    state.pipe_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    state.selected_controller = None;
    state.placed_bearings = placed_bearings;
    state.construction_mesh_dirty = true;
    if let Some((minimum, maximum)) = graph_bounds(graph) {
        let (view, transform, global) = &mut **camera;
        player.place_outside_bounds(view, minimum, maximum);
        **transform = view.apply_pullback(
            player.position + Vec3::Y * camera::EYE_HEIGHT,
            view.look_rotation(),
        );
        **global = GlobalTransform::from(**transform);
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
    graph_and_world: (Res<EditorGraph>, Res<world::WorldRuntime>),
    sequencer: Res<DriveSequencer>,
    gearboxes: Res<GearboxRuntime>,
    selection: Res<SelectedTool>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    mut hammer: ResMut<HammerInteraction>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visuals: Query<
        (&ConstructionVisual, &mut Visibility),
        (Without<BearingVisual>, Without<SimulationVisual>),
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
    mut simulation_visuals: Query<
        (&SimulationVisual, &mut Visibility),
        (Without<ConstructionVisual>, Without<BearingVisual>),
    >,
) {
    let (graph, world_runtime) = graph_and_world;
    if !simulation.is_running() {
        return;
    }

    // Poll before scheduling more work. This is deliberately non-blocking:
    // completed tick telemetry and transforms arrive whenever the shared GPU
    // queue reaches their staging copies.
    loop {
        let completed = simulation
            .gpu
            .as_ref()
            .expect("running simulation has GPU state")
            .poll_tick_readback(render_device.wgpu_device());
        match completed {
            Ok(Some(completed)) if completed.diagnostics.error_flags == 0 => {
                if completed.tick_index <= simulation.completed_tick {
                    continue;
                }
                simulation.completed_tick = completed.tick_index;
                simulation.last_tick_readback = Some(completed.diagnostics);
                if visual_snapshot_is_due(simulation.snapshot_tick, completed.tick_index) {
                    simulation.previous_transforms =
                        core::mem::replace(&mut simulation.transforms, completed.transforms);
                    simulation.previous_snapshot_tick = simulation.snapshot_tick;
                    simulation.snapshot_tick = completed.tick_index;
                    simulation.render_dirty = true;
                }
            }
            Ok(Some(completed)) => {
                stop_failed_simulation(
                    &mut simulation,
                    &mut state,
                    format!(
                        "physics tick {} reported flags {}",
                        completed.tick_index, completed.diagnostics.error_flags
                    ),
                );
                return;
            }
            Ok(None) => break,
            Err(error) => {
                stop_failed_simulation(&mut simulation, &mut state, error.to_string());
                return;
            }
        }
    }

    if state.drive_rows_dirty {
        state.drive_rows_dirty = false;
        if let (Some(gpu), Some(creation)) = (simulation.gpu.as_ref(), simulation.creation.as_ref())
            && let Err(error) = gpu.write_mechanism_drives(
                &render_queue,
                &geared_gpu_drive_rows(creation, &graph.0, &sequencer, &gearboxes),
            )
        {
            stop_failed_simulation(&mut simulation, &mut state, error.to_string());
            return;
        }
    }

    let ticks = {
        let available = simulation
            .gpu
            .as_ref()
            .expect("running simulation has GPU state")
            .async_readback_slots_available();
        let AppSimulation {
            scheduler,
            next_tick,
            tick_backlog,
            ..
        } = &mut *simulation;
        next_simulation_ticks(
            scheduler,
            next_tick,
            tick_backlog,
            time.delta(),
            false,
            u64::try_from(available).unwrap_or(u64::MAX),
        )
    };
    if !ticks.is_empty() {
        let physics_started = std::time::Instant::now();
        for tick in ticks {
            if let Err(error) =
                apply_pending_hammer_impact(&simulation, &mut hammer, &render_device, &render_queue)
            {
                hammer.pending = None;
                stop_failed_simulation(&mut simulation, &mut state, error);
                return;
            }
            simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state")
                .dispatch_tick(render_device.wgpu_device(), &render_queue, tick);
        }
        simulation.record_performance(physics_started.elapsed(), None);
    }

    if simulation.static_mesh_dirty {
        let creation = simulation
            .creation
            .as_ref()
            .expect("running simulation has compiled creation");
        for material in ConstructionMaterial::ALL {
            let visible = simulation_material_is_present(
                &graph.0,
                creation,
                SimulationMeshKind::Static,
                material,
            );
            if visible
                && let Some(mut asset) =
                    meshes.get_mut(&visuals.construction_meshes[material_index(material)])
            {
                *asset = renderable_mesh(combined_simulation_material_mesh(
                    &graph.0,
                    creation,
                    &simulation.transforms,
                    SimulationMeshKind::Static,
                    material,
                ));
            }
            for (visual, mut visibility) in &mut construction_visuals {
                if visual.0 == material {
                    *visibility = if visible {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    };
                }
            }
        }
        simulation.static_mesh_dirty = false;
    }

    if !simulation.render_dirty {
        return;
    }
    let creation = simulation
        .creation
        .as_ref()
        .expect("running simulation has compiled creation");
    for material in ConstructionMaterial::ALL {
        let visible = simulation_material_is_present(
            &graph.0,
            creation,
            SimulationMeshKind::Dynamic,
            material,
        );
        if visible
            && let Some(mut asset) =
                meshes.get_mut(&visuals.simulation_meshes[material_index(material)])
        {
            *asset = renderable_mesh(combined_simulation_material_mesh(
                &graph.0,
                creation,
                &simulation.transforms,
                SimulationMeshKind::Dynamic,
                material,
            ));
        }
        for (visual, mut visibility) in &mut simulation_visuals {
            if visual.0 == material {
                *visibility = if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
    // Every mesh below is written only while its own visual is on screen. A
    // hidden mesh has no slab allocation, so writing to one both wastes the
    // rebuild and makes the renderer log a use-after-free every frame.
    let bearings_visible = graph.0.bearing_count() > 0 || !state.placed_bearings.is_empty();
    let active_dimension_link = world_runtime.active_dimension_link();
    for appearance in AuthoredPart::ALL {
        let visible = graph
            .0
            .parts()
            .any(|(part, spec)| appearance.matches(&graph.0, part, *spec, active_dimension_link));
        if visible && let Some(mut mesh) = meshes.get_mut(visuals.authored_mesh(appearance)) {
            *mesh = combined_simulation_authored_mesh(
                &graph.0,
                creation,
                &simulation.transforms,
                appearance,
                active_dimension_link,
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
    if drive_xray_is_visible(selection.active_editor_tool(), control_link_count(&graph.0))
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
    simulation.render_dirty = false;
}

fn next_simulation_ticks(
    scheduler: &mut FixedStepScheduler,
    next_tick: &mut u64,
    tick_backlog: &mut u64,
    elapsed: std::time::Duration,
    paused: bool,
    maximum_batch: u64,
) -> std::ops::Range<u64> {
    if paused {
        return *next_tick..*next_tick;
    }
    *tick_backlog = tick_backlog.saturating_add(scheduler.advance(elapsed).count());
    let first = *next_tick;
    let batch = (*tick_backlog).min(maximum_batch);
    *tick_backlog -= batch;
    *next_tick = next_tick.saturating_add(batch);
    first..*next_tick
}

const fn visual_snapshot_is_due(snapshot_tick: u64, completed_tick: u64) -> bool {
    completed_tick.saturating_sub(snapshot_tick) >= SIMULATION_VISUAL_TICK_INTERVAL
}

fn stop_failed_simulation(simulation: &mut AppSimulation, state: &mut EditorState, error: String) {
    simulation.failure = Some(error.clone());
    state.feedback = Some(format!("Simulation stopped: {error}"));
}

impl AppSimulation {
    const fn is_running(&self) -> bool {
        self.gpu.is_some() && self.failure.is_none()
    }

    pub(crate) fn live_part_pose(
        &self,
        graph: &ConstructionGraph,
        part: PartId,
    ) -> Option<(Vec3, Quat)> {
        let spec = *graph.part(part)?;
        let Some(creation) = self.creation.as_ref() else {
            return Some((spec.pose().translation(), spec.pose().rotation.quaternion()));
        };
        let body = creation
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body))?;
        let transform = self.transforms.get(body as usize)?;
        let root_position = Vec3::from_slice(&transform.position[..3]);
        let root_rotation = Quat::from_array(transform.rotation);
        let initial = &creation.compounds[body as usize];
        Some((
            root_position + root_rotation * (spec.pose().translation() - initial.root_translation),
            root_rotation * spec.pose().rotation.quaternion(),
        ))
    }

    fn record_performance(
        &mut self,
        cpu_elapsed: std::time::Duration,
        readback: Option<GpuTickReadback>,
    ) {
        const SMOOTHING: f64 = 0.2;
        let cpu_ms = cpu_elapsed.as_secs_f64() * 1_000.0;
        self.physics_cpu_ms = Some(self.physics_cpu_ms.map_or(cpu_ms, |previous| {
            previous + (cpu_ms - previous) * SMOOTHING
        }));
        if let Some(readback) = readback {
            self.last_tick_readback = Some(readback);
        }
    }
}

#[derive(Resource, Default)]
struct EditorState {
    placement_bounds: PlacementBounds,
    hovered: Option<SurfaceHit>,
    hovered_simulation: Option<SimulationHit>,
    world_edit_blocker: Option<WorldEditBlocker>,
    /// Unattached bearing surface directly hit by the pointer ray.
    hovered_bearing: Option<usize>,
    /// Unattached bearing that would claim the current block preview.
    attachment_bearing: Option<usize>,
    preview: Option<PlacementCandidate>,
    cylinder_preview: Option<CylinderPlacementCandidate>,
    /// Empty-space point offered when an eligible Garage tool misses construction.
    free_placement_point: Option<Vec3>,
    bearing_preview_anchor: Option<Vec3>,
    preview_error: Option<PlacementError>,
    /// A staged action that is allowed but costs something the player should
    /// see first — currently a weld that locks a bearing solid.
    preview_warning: Option<String>,
    placement_grid: PlacementGrid,
    smart_guides: Vec<SmartGuide>,
    smart_snap: SmartSnapSettings,
    free_placement: FreePlacementSettings,
    snap_index: PlacementSnapIndex,
    /// One of the 24 grid-aligned orientations used by authored parts.
    authored_orientation: u8,
    feedback: Option<String>,
    construction_mesh_dirty: bool,
    /// Last immutable graph revision atomically published to construction meshes.
    rendered_graph: ConstructionGraph,
    delete_target: Option<DeleteTarget>,
    block_drag: Option<BlockDrag>,
    block_preview_revision: u64,
    pipe_drag: Option<PipeDrag>,
    pipe_preview_revision: u64,
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
    /// The region the Shape tool is editing. Nothing can be shaped until one is
    /// chosen, and while one is, everything else fades back.
    active_region: Option<RegionId>,
    /// Area being dragged out to become a region.
    region_drag: Option<RegionDrag>,
    /// Cage vertex the pointer is over.
    hovered_vertex: Option<CageIndex>,
    /// Vertex the pointer is dragging.
    vertex_drag: Option<shape_tool::VertexDrag>,
    /// Vertices painted by Shift+left, moved together by one drag.
    selected_vertices: Vec<CageIndex>,
    /// Whether Shift+left is sweeping across cage vertices.
    paint_selecting: bool,
    /// The new cage vertex the pointer is currently being offered.
    edge_offer: Option<shape_tool::EdgeInsertion>,
    /// Construction solid currently focused by Chamfer or Fillet mode.
    feature_focus: Option<mechanic_core::SolidOwner>,
    /// Logical feature edge under the pointer.
    hovered_feature_edge: Option<shape_tool::FeatureEdgeHit>,
    /// Separate logical chains sharing the next feature amount.
    selected_feature_edges: Vec<mechanic_core::EdgeChainRef>,
    /// Chamfer/fillet amount drag in progress.
    feature_drag: Option<shape_tool::FeatureDrag>,
    /// Earlier feature selected through its virtual source overlay.
    selected_shape_feature: Option<mechanic_core::ShapeFeatureId>,
    /// Earlier feature whose dashed source chain is under the pointer.
    hovered_source_feature: Option<mechanic_core::ShapeFeatureId>,
    /// Active paint/remove drag, committed to history as one edit on release.
    chroma_stroke: Option<ChromaStroke>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorldEditBlocker {
    MovingConstruction,
}

impl EditorState {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn world_drag_active(&self) -> bool {
        self.contextual_selector_blocked() || self.active_region.is_some()
    }

    /// Whether a transient gesture owns the pointer strongly enough to block
    /// the contextual hold-Tab selector. A Shape focus by itself is retained
    /// across mode changes and therefore does not block the selector.
    pub(crate) fn contextual_selector_blocked(&self) -> bool {
        self.block_drag.is_some()
            || self.pipe_drag.is_some()
            || self.delete_drag.is_some()
            || self.delete_target.is_some()
            || self.region_drag.is_some()
            || self.vertex_drag.is_some()
            || self.feature_drag.is_some()
            || self.wire_drag.is_some()
            || self.paint_selecting
            || self.chroma_stroke.is_some()
    }

    pub(crate) fn pipe_bend_active(&self) -> bool {
        self.pipe_drag
            .as_ref()
            .is_some_and(|drag| drag.choosing_direction || !drag.bend_radii.is_empty())
    }

    fn cancel_delete_gesture(&mut self) -> bool {
        let cancelled_drag = self.delete_drag.take().is_some();
        let cancelled_target = self.delete_target.take().is_some();
        let cancelled = cancelled_drag || cancelled_target;
        if cancelled {
            clear_hover(self);
        }
        cancelled
    }
}

#[derive(Resource)]
struct EditorVisuals {
    construction_meshes: [Handle<Mesh>; ConstructionMaterial::ALL.len()],
    construction_materials: [Handle<ConstructionRenderMaterial>; ConstructionMaterial::ALL.len()],
    ghost_materials: [Handle<ConstructionRenderMaterial>; ConstructionMaterial::ALL.len()],
    simulation_meshes: [Handle<Mesh>; ConstructionMaterial::ALL.len()],
    bearing_mesh: Handle<Mesh>,
    joint_xray_mesh: Handle<Mesh>,
    shape_node_mesh: Handle<Mesh>,
    shape_selected_mesh: Handle<Mesh>,
    shape_plane_mesh: Handle<Mesh>,
    shape_arrow_mesh: Handle<Mesh>,
    controller_mesh: Handle<Mesh>,
    gas_engine_mesh: Handle<Mesh>,
    electric_engine_mesh: Handle<Mesh>,
    gas_transmission_mesh: Handle<Mesh>,
    electric_transmission_mesh: Handle<Mesh>,
    servo_mesh: Handle<Mesh>,
    seat_mesh: Handle<Mesh>,
    input_mesh: Handle<Mesh>,
    dimension_link_disabled_mesh: Handle<Mesh>,
    dimension_link_enabled_mesh: Handle<Mesh>,
    authored_preview_meshes: [Handle<Mesh>; AuthoredPart::ALL.len()],
    authored_preview_materials: [Handle<StandardMaterial>; AuthoredPart::ALL.len()],
    invalid_authored_preview_materials: [Handle<StandardMaterial>; AuthoredPart::ALL.len()],
    drive_xray_mesh: Handle<Mesh>,
    wire_drag_mesh: Handle<Mesh>,
    wire_hover_mesh: Handle<Mesh>,
    cube_preview_mesh: Handle<Mesh>,
    cylinder_preview_mesh: Handle<Mesh>,
    bearing_preview_mesh: Handle<Mesh>,
    white_preview_material: Handle<StandardMaterial>,
    chroma_preview_material: Handle<StandardMaterial>,
    green_preview_material: Handle<StandardMaterial>,
    red_preview_material: Handle<StandardMaterial>,
    /// Allowed, but with a consequence worth seeing first.
    amber_preview_material: Handle<StandardMaterial>,
    block_drag_preview_mesh: Handle<Mesh>,
    delete_drag_preview_mesh: Handle<Mesh>,
    weld_hover_preview_mesh: Handle<Mesh>,
    weld_selection_preview_mesh: Handle<Mesh>,
}

#[derive(Default)]
struct PreviewMeshRevisions {
    block: u64,
    pipe: u64,
    delete: u64,
}

impl EditorVisuals {
    fn authored_mesh(&self, appearance: AuthoredPart) -> &Handle<Mesh> {
        match appearance {
            AuthoredPart::Controller => &self.controller_mesh,
            AuthoredPart::GasEngine => &self.gas_engine_mesh,
            AuthoredPart::ElectricEngine => &self.electric_engine_mesh,
            AuthoredPart::GasTransmission => &self.gas_transmission_mesh,
            AuthoredPart::ElectricTransmission => &self.electric_transmission_mesh,
            AuthoredPart::Servo => &self.servo_mesh,
            AuthoredPart::Seat => &self.seat_mesh,
            AuthoredPart::Input => &self.input_mesh,
            AuthoredPart::DimensionLinkDisabled => &self.dimension_link_disabled_mesh,
            AuthoredPart::DimensionLinkEnabled => &self.dimension_link_enabled_mesh,
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
struct ConstructionVisual(ConstructionMaterial);

#[derive(Component)]
struct SimulationVisual(ConstructionMaterial);

#[derive(Component)]
struct BearingVisual;

#[derive(Component)]
struct JointXrayVisual;

#[derive(Component)]
struct FovCamera;

#[derive(Component)]
struct ShapeNodeVisual;

#[derive(Component)]
struct ShapeSelectedVisual;

/// The plane a drag is currently sliding along.
#[derive(Component)]
struct ShapePlaneVisual;

/// The arrows naming that plane's two axes.
#[derive(Component)]
struct ShapeArrowVisual;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlacementLatticeKey {
    grid: PlacementGrid,
    low_ticks: IVec3,
    high_ticks: IVec3,
    plane: Option<PlacementPlane>,
}

#[derive(Component, Default)]
struct PlacementLatticeVisual {
    key: Option<PlacementLatticeKey>,
}

#[derive(Component, Default)]
struct SmartGuideVisual {
    guides: Vec<SmartGuide>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SmartSnapRangeKey {
    low_ticks: IVec3,
    high_ticks: IVec3,
    range_ticks: i32,
    plane: Option<PlacementPlane>,
}

#[derive(Component, Default)]
struct SmartSnapRangeVisual {
    key: Option<SmartSnapRangeKey>,
}

#[derive(Component)]
pub(crate) struct PlayerAvatar;

#[derive(Component, Clone)]
pub(crate) struct AvatarPart {
    standing: Transform,
    seated: Transform,
}

#[derive(Resource)]
pub(crate) struct AvatarMaterials {
    clothing: Handle<StandardMaterial>,
    head: Handle<StandardMaterial>,
    boots: Handle<StandardMaterial>,
}

/// One of the Shape tool's three overlay batches, each excluding the others so
/// the three `Single` parameters can be held at once.
type ShapeOverlay<'w, 's, Own, First, Second> =
    Single<'w, 's, &'static mut Visibility, (With<Own>, Without<First>, Without<Second>)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthoredPart {
    Controller,
    GasEngine,
    ElectricEngine,
    GasTransmission,
    ElectricTransmission,
    Servo,
    Seat,
    Input,
    DimensionLinkDisabled,
    DimensionLinkEnabled,
}

impl AuthoredPart {
    const ALL: [Self; 10] = [
        Self::Controller,
        Self::GasEngine,
        Self::ElectricEngine,
        Self::GasTransmission,
        Self::ElectricTransmission,
        Self::Servo,
        Self::Seat,
        Self::Input,
        Self::DimensionLinkDisabled,
        Self::DimensionLinkEnabled,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Controller => 0,
            Self::GasEngine => 1,
            Self::ElectricEngine => 2,
            Self::GasTransmission => 3,
            Self::ElectricTransmission => 4,
            Self::Servo => 5,
            Self::Seat => 6,
            Self::Input => 7,
            Self::DimensionLinkDisabled => 8,
            Self::DimensionLinkEnabled => 9,
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
            Tool::DimensionLink => Some(Self::DimensionLinkDisabled),
            _ => None,
        }
    }

    fn matches(
        self,
        graph: &ConstructionGraph,
        part: PartId,
        spec: PartSpec,
        active: Option<mechanic_core::DimensionLinkId>,
    ) -> bool {
        if let PartSpec::DimensionLink(link) = spec {
            return match self {
                Self::DimensionLinkDisabled => Some(link.id) != active,
                Self::DimensionLinkEnabled => Some(link.id) == active,
                _ => false,
            };
        }
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
        ) || matches!(spec, PartSpec::Transmission(_))
            && match self {
                Self::GasTransmission => graph.transmission_kind(part) == Some(EngineKind::Gas),
                Self::ElectricTransmission => {
                    graph.transmission_kind(part) == Some(EngineKind::Electric)
                }
                _ => false,
            }
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

fn bearing_surface_material(asset_server: &AssetServer) -> StandardMaterial {
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                configure_bearing_texture(settings, is_srgb);
            })
            .load(format!("machines/bearing/bearing_{suffix}.png"))
    };
    bearing_pbr_material(
        texture("base_color", true),
        texture("normal", false),
        texture("orm", false),
    )
}

fn bearing_pbr_material(
    base_color: Handle<Image>,
    normal: Handle<Image>,
    orm: Handle<Image>,
) -> StandardMaterial {
    StandardMaterial {
        base_color_texture: Some(base_color),
        metallic: 1.0,
        perceptual_roughness: 1.0,
        metallic_roughness_texture: Some(orm.clone()),
        occlusion_texture: Some(orm),
        normal_map_texture: Some(normal),
        depth_bias: BEARING_RENDER_DEPTH_BIAS,
        ..default()
    }
}

fn configure_bearing_texture(settings: &mut ImageLoaderSettings, is_srgb: bool) {
    settings.is_srgb = is_srgb;
    let sampler = settings.sampler.get_or_init_descriptor();
    sampler.address_mode_u = ImageAddressMode::Repeat;
    sampler.address_mode_v = ImageAddressMode::ClampToEdge;
    sampler.mag_filter = ImageFilterMode::Linear;
    sampler.min_filter = ImageFilterMode::Linear;
    sampler.mipmap_filter = ImageFilterMode::Linear;
    sampler.anisotropy_clamp = 8;
}

#[derive(Resource)]
struct BearingTextureMipsPending(Vec<Handle<Image>>);

fn prepare_bearing_texture_mips(
    mut images: ResMut<Assets<Image>>,
    mut pending: ResMut<BearingTextureMipsPending>,
) {
    let Some(index) = pending
        .0
        .iter()
        .position(|handle| images.contains(handle.id()))
    else {
        return;
    };
    let handle = pending.0.swap_remove(index);
    let Some(mut image) = images.get_mut(&handle) else {
        return;
    };
    if let Err(error) = world::generate_rgba8_mip_chain(&mut image) {
        warn!("failed to generate bearing texture mipmaps: {error}");
        pending.0.clear();
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
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                configure_authored_texture(settings, is_srgb);
            })
            .load(format!("{stem}_{suffix}.png"))
    };
    let orm = texture("orm", false);
    StandardMaterial {
        base_color_texture: Some(texture("base_color", true)),
        metallic: 1.0,
        perceptual_roughness: 1.0,
        metallic_roughness_texture: Some(orm.clone()),
        occlusion_texture: Some(orm),
        normal_map_texture: Some(texture("normal", false)),
        emissive: LinearRgba::WHITE,
        emissive_texture: Some(texture("emissive", true)),
        ..default()
    }
}

fn configure_authored_texture(settings: &mut ImageLoaderSettings, is_srgb: bool) {
    settings.is_srgb = is_srgb;
    let sampler = settings.sampler.get_or_init_descriptor();
    sampler.address_mode_u = ImageAddressMode::ClampToEdge;
    sampler.address_mode_v = ImageAddressMode::ClampToEdge;
    sampler.mag_filter = ImageFilterMode::Linear;
    sampler.min_filter = ImageFilterMode::Linear;
    sampler.mipmap_filter = ImageFilterMode::Linear;
}

const fn material_index(material: ConstructionMaterial) -> usize {
    match material {
        ConstructionMaterial::Aluminium => 0,
        ConstructionMaterial::CarbonFiber => 1,
        ConstructionMaterial::Concrete => 2,
        ConstructionMaterial::Copper => 3,
        ConstructionMaterial::Dirt => 4,
        ConstructionMaterial::Graphite => 5,
        ConstructionMaterial::Iron => 6,
        ConstructionMaterial::Plastic => 7,
        ConstructionMaterial::Rubber => 8,
        ConstructionMaterial::Sand => 9,
        ConstructionMaterial::Steel => 10,
        ConstructionMaterial::Stone => 11,
        ConstructionMaterial::Wood => 12,
    }
}

const fn construction_tint_mask_path(material: ConstructionMaterial) -> Option<&'static str> {
    match material {
        ConstructionMaterial::Copper => Some("materials/copper/copper_tint.png"),
        ConstructionMaterial::Dirt => Some("materials/dirt/dirt_tint.png"),
        ConstructionMaterial::Aluminium
        | ConstructionMaterial::CarbonFiber
        | ConstructionMaterial::Concrete
        | ConstructionMaterial::Graphite
        | ConstructionMaterial::Iron
        | ConstructionMaterial::Plastic
        | ConstructionMaterial::Rubber
        | ConstructionMaterial::Sand
        | ConstructionMaterial::Steel
        | ConstructionMaterial::Stone
        | ConstructionMaterial::Wood => None,
    }
}

fn construction_material(
    asset_server: &AssetServer,
    material: ConstructionMaterial,
    tint_mask: Handle<Image>,
) -> ConstructionRenderMaterial {
    let stem = match material {
        ConstructionMaterial::Aluminium => "materials/aluminium/aluminium",
        ConstructionMaterial::Graphite => "materials/graphite/graphite",
        ConstructionMaterial::CarbonFiber => "materials/carbon_fiber/carbon_fiber",
        ConstructionMaterial::Concrete => "materials/concrete/concrete",
        ConstructionMaterial::Copper => "materials/copper/copper",
        ConstructionMaterial::Dirt => "materials/dirt/dirt",
        ConstructionMaterial::Iron => "materials/iron/iron",
        ConstructionMaterial::Plastic => "materials/plastic/plastic",
        ConstructionMaterial::Rubber => "materials/rubber/rubber",
        ConstructionMaterial::Sand => "materials/sand/sand",
        ConstructionMaterial::Steel => "materials/steel/steel",
        ConstructionMaterial::Stone => "materials/stone/stone",
        ConstructionMaterial::Wood => "materials/wood/wood",
    };
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                configure_repeating_texture(settings, is_srgb);
            })
            .load(format!("{stem}_{suffix}.png"))
    };
    let orm = texture("orm", false);
    ExtendedMaterial {
        base: StandardMaterial {
            base_color_texture: Some(texture("base_color", true)),
            metallic: 1.0,
            perceptual_roughness: 1.0,
            metallic_roughness_texture: Some(orm.clone()),
            occlusion_texture: Some(orm),
            normal_map_texture: Some(texture("normal", false)),
            ..default()
        },
        extension: ChromaMaterialExtension {
            tint_mask,
            base_lightness: Vec4::new(
                chroma::material_profile(material).mean_oklab_lightness,
                0.0,
                0.0,
                0.0,
            ),
        },
    }
}

fn configure_repeating_texture(settings: &mut ImageLoaderSettings, is_srgb: bool) {
    settings.is_srgb = is_srgb;
    let sampler = settings.sampler.get_or_init_descriptor();
    sampler.address_mode_u = ImageAddressMode::Repeat;
    sampler.address_mode_v = ImageAddressMode::Repeat;
    sampler.mag_filter = ImageFilterMode::Linear;
    sampler.min_filter = ImageFilterMode::Linear;
    sampler.mipmap_filter = ImageFilterMode::Linear;
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

#[allow(clippy::too_many_lines)] // The app schedule is kept in visible execution order.
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mechanic — construction and simulation prototype".to_owned(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..Default::default()
        }))
        .add_plugins((
            FrameTimeDiagnosticsPlugin::new(120),
            bevy::render::diagnostic::RenderDiagnosticsPlugin,
        ))
        .add_plugins(StreamingMeshAllocatorPlugin)
        .add_plugins(OneShotEnvironmentMapPlugin)
        .add_plugins(MaterialPlugin::<world::TerrainRenderMaterial>::default())
        .add_plugins(MaterialPlugin::<ConstructionRenderMaterial>::default())
        // After DefaultPlugins: the overlay's render pass installs into the
        // render sub-app, which does not exist until RenderPlugin has run.
        .add_plugins(bevy_mosaic::MosaicPlugin)
        .init_resource::<EditorGraph>()
        .init_resource::<EditorState>()
        .init_resource::<EditorHistory>()
        .init_resource::<CreationMenuState>()
        .init_resource::<CreationStore>()
        .init_resource::<CurrentCreation>()
        .init_resource::<DebugFrameFreeze>()
        .init_resource::<PauseMenuState>()
        .init_resource::<PerformanceMetrics>()
        .init_resource::<AppSettings>()
        .init_resource::<AppSimulation>()
        .init_resource::<HammerInteraction>()
        .init_resource::<BearingToolSettings>()
        .init_resource::<ControlPanelState>()
        .init_resource::<DriveSequencer>()
        .init_resource::<GearboxRuntime>()
        .init_resource::<CylinderToolSettings>()
        .init_resource::<shape_tool::ShapeMirror>()
        .init_resource::<shape_tool::ShapeSnap>()
        .init_resource::<shape_tool::ShapeEditMode>()
        .init_resource::<SelectedTool>()
        .init_resource::<SelectedMaterial>()
        .init_resource::<ChromaBrush>()
        .init_resource::<SelectedTerrainMaterial>()
        .init_resource::<PlayerState>()
        .init_resource::<MaterialWheelState>()
        .init_resource::<ButtonInput<GameAction>>()
        .add_plugins(world::WorldPrototypePlugin)
        .add_plugins(multitool::MultitoolPlugin)
        // A dim base fill keeps occluded construction readable without
        // overpowering the garage's authored lighting.
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb_u8(43, 60, 76),
            brightness: 20.0,
            ..Default::default()
        })
        .insert_resource(ClearColor(garage::VOID_COLOR))
        .add_systems(Startup, (setup, ui::mount).chain())
        .add_systems(
            Update,
            (
                update_debug_frame_freeze,
                prepare_bearing_texture_mips,
                performance::toggle,
                (
                    (
                        (
                            begin_pause_frame,
                            controls::update_action_state,
                            update_smart_snap_settings,
                            update_free_placement_settings,
                            capture_control_binding,
                            handle_creation_menu_shortcut,
                            handle_dimension_link_interaction,
                            handle_control_panel_shortcut,
                            ui::drain,
                            handle_pause_request,
                            handle_pause_escape,
                        )
                            .chain(),
                        (
                            ui::push,
                            ui::push_help,
                            ui::push_markers,
                            ui::push_player,
                            ui::sync_input,
                            handle_history_shortcut,
                            handle_creation_request,
                        )
                            .chain(),
                    )
                        .chain(),
                    (
                        apply_camera_fov,
                        camera::update_material_wheel,
                        camera::update_player_camera,
                        handle_seat_interaction,
                        handle_shortcuts,
                    )
                        .chain(),
                    (
                        (
                            handle_bearing_dimension_shortcuts,
                            handle_cylinder_dimension_shortcuts,
                        )
                            .chain(),
                        (
                            handle_tool_change,
                            rebuild_placement_snap_index,
                            update_hover,
                            handle_build_actions,
                            handle_shape_actions,
                            ui::push_dimensions,
                            handle_hammer_actions,
                        )
                            .chain(),
                        update_joint_xray,
                        sync_placement_overlays,
                        sync_visual_meshes,
                        (sync_shape_nodes, sync_region_focus, sync_drag_plane).chain(),
                        update_wire_drag_preview,
                        update_wire_hover_preview,
                        maintain_space_simulation.after(world::sync_world_foundations),
                        run_drive_sequencer,
                        advance_simulation.run_if(world::world_playing),
                        sync_player_avatar,
                        update_previews,
                        (performance::sample, ui::push_performance).chain(),
                    )
                        .chain(),
                )
                    .chain()
                    .run_if(debug_frame_updates_enabled),
            )
                .chain(),
        )
        .run();
}

#[allow(clippy::too_many_lines)] // One-time Bevy scene composition is clearest in declaration order.
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    settings: Res<AppSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut construction_render_materials: ResMut<Assets<ConstructionRenderMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let construction_meshes = ConstructionMaterial::ALL.map(|_| meshes.add(Cuboid::default()));
    let simulation_meshes = ConstructionMaterial::ALL.map(|_| meshes.add(Cuboid::default()));
    let bearing_mesh = meshes.add(Cuboid::default());
    let joint_xray_mesh = meshes.add(Cuboid::default());
    let shape_node_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_selected_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_plane_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_arrow_mesh = meshes.add(degenerate_overlay_mesh());
    let placement_lattice_mesh = meshes.add(degenerate_overlay_mesh());
    let smart_guide_mesh = meshes.add(degenerate_overlay_mesh());
    let smart_snap_range_mesh = meshes.add(degenerate_overlay_mesh());
    let controller_mesh = meshes.add(Cuboid::default());
    let gas_engine_mesh = meshes.add(Cuboid::default());
    let electric_engine_mesh = meshes.add(Cuboid::default());
    let gas_transmission_mesh = meshes.add(Cuboid::default());
    let electric_transmission_mesh = meshes.add(Cuboid::default());
    let servo_mesh = meshes.add(Cuboid::default());
    let seat_mesh = meshes.add(Cuboid::default());
    let input_mesh = meshes.add(Cuboid::default());
    let dimension_link_disabled_mesh = meshes.add(Cuboid::default());
    let dimension_link_enabled_mesh = meshes.add(Cuboid::default());
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
    let white_tint_mask = images.add(Image::new_fill(
        Extent3d::default(),
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    ));
    let tint_mask = |material| match construction_tint_mask_path(material) {
        Some(path) => asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                configure_repeating_texture(settings, false);
            })
            .load(path),
        None => white_tint_mask.clone(),
    };
    let construction_materials = ConstructionMaterial::ALL.map(|material| {
        construction_render_materials.add(construction_material(
            &asset_server,
            material,
            tint_mask(material),
        ))
    });
    // Faded copies, swapped in while a region is being edited so the area under
    // the cursor is the only thing that reads as solid.
    let ghost_materials = ConstructionMaterial::ALL.map(|material| {
        let mut ghost = construction_material(&asset_server, material, tint_mask(material));
        ghost.base.base_color = ghost.base.base_color.with_alpha(0.16);
        ghost.base.alpha_mode = AlphaMode::Blend;
        construction_render_materials.add(ghost)
    });
    let bearing_material = bearing_surface_material(&asset_server);
    commands.insert_resource(BearingTextureMipsPending(vec![
        bearing_material
            .base_color_texture
            .clone()
            .expect("the bearing has a base-color map"),
        bearing_material
            .normal_map_texture
            .clone()
            .expect("the bearing has a normal map"),
        bearing_material
            .metallic_roughness_texture
            .clone()
            .expect("the bearing has an ORM map"),
    ]));
    let bearing_material = materials.add(bearing_material);
    let authored_materials = [
        authored_part_material(&asset_server, "machines/controller/controller"),
        authored_part_material(&asset_server, "machines/gas_engine/gas_engine"),
        authored_part_material(&asset_server, "machines/electric_engine/electric_engine"),
        authored_part_material(&asset_server, "machines/transmission_gas/transmission_gas"),
        authored_part_material(
            &asset_server,
            "machines/transmission_electric/transmission_electric",
        ),
        authored_part_material(&asset_server, "machines/servo/servo"),
        authored_part_material(&asset_server, "machines/seat/seat"),
        authored_part_material(&asset_server, "machines/input/input"),
        authored_part_material(
            &asset_server,
            "machines/dimension_link/disabled/dimension_link",
        ),
        authored_part_material(
            &asset_server,
            "machines/dimension_link/enabled/dimension_link",
        ),
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
    let chroma_preview_material =
        materials.add(preview_material(Color::srgba(1.0, 1.0, 1.0, 0.46)));
    let red_preview_material = materials.add(preview_material(Color::srgba(1.0, 0.06, 0.04, 0.46)));
    let amber_preview_material =
        materials.add(preview_material(Color::srgba(1.0, 0.60, 0.06, 0.46)));
    let green_preview_material =
        materials.add(preview_material(Color::srgba(0.12, 1.0, 0.28, 0.52)));
    let placement_lattice_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR.with_alpha(0.24),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });
    let smart_guide_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.86, 0.18),
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });
    let smart_snap_range_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.86, 0.18, 0.34),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });

    spawn_player_avatar(&mut commands, &mut meshes, &mut materials);

    commands.insert_resource(EditorVisuals {
        construction_meshes: construction_meshes.clone(),
        construction_materials: construction_materials.clone(),
        ghost_materials: ghost_materials.clone(),
        simulation_meshes: simulation_meshes.clone(),
        bearing_mesh: bearing_mesh.clone(),
        joint_xray_mesh: joint_xray_mesh.clone(),
        shape_node_mesh: shape_node_mesh.clone(),
        shape_selected_mesh: shape_selected_mesh.clone(),
        shape_plane_mesh: shape_plane_mesh.clone(),
        shape_arrow_mesh: shape_arrow_mesh.clone(),
        controller_mesh: controller_mesh.clone(),
        gas_engine_mesh: gas_engine_mesh.clone(),
        electric_engine_mesh: electric_engine_mesh.clone(),
        gas_transmission_mesh: gas_transmission_mesh.clone(),
        electric_transmission_mesh: electric_transmission_mesh.clone(),
        servo_mesh: servo_mesh.clone(),
        seat_mesh: seat_mesh.clone(),
        input_mesh: input_mesh.clone(),
        dimension_link_disabled_mesh: dimension_link_disabled_mesh.clone(),
        dimension_link_enabled_mesh: dimension_link_enabled_mesh.clone(),
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
        chroma_preview_material,
        green_preview_material,
        red_preview_material: red_preview_material.clone(),
        amber_preview_material,
        block_drag_preview_mesh,
        delete_drag_preview_mesh,
        weld_hover_preview_mesh,
        weld_selection_preview_mesh,
    });

    garage::spawn(&mut commands, &asset_server, &mut meshes, &mut materials);
    for material in ConstructionMaterial::ALL {
        let index = material_index(material);
        commands.spawn((
            Name::new(format!("{} construction mesh", material.label())),
            Mesh3d(construction_meshes[index].clone()),
            MeshMaterial3d(construction_materials[index].clone()),
            NoFrustumCulling,
            Visibility::Hidden,
            ConstructionVisual(material),
        ));
        commands.spawn((
            Name::new(format!("{} simulation mesh", material.label())),
            Mesh3d(simulation_meshes[index].clone()),
            MeshMaterial3d(construction_materials[index].clone()),
            NoFrustumCulling,
            Visibility::Hidden,
            SimulationVisual(material),
        ));
    }
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
        Name::new("Gas transmission mesh"),
        Mesh3d(gas_transmission_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::GasTransmission.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::GasTransmission),
    ));
    commands.spawn((
        Name::new("Electric transmission mesh"),
        Mesh3d(electric_transmission_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::ElectricTransmission.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::ElectricTransmission),
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
        Name::new("Disabled Dimension Link mesh"),
        Mesh3d(dimension_link_disabled_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::DimensionLinkDisabled.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::DimensionLinkDisabled),
    ));
    commands.spawn((
        Name::new("Enabled Dimension Link mesh"),
        Mesh3d(dimension_link_enabled_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::DimensionLinkEnabled.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::DimensionLinkEnabled),
    ));
    commands.spawn((
        Name::new("Joint x-ray mesh"),
        Mesh3d(joint_xray_mesh),
        MeshMaterial3d(joint_xray_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        JointXrayVisual,
    ));
    commands.spawn((
        Name::new("Placement lattice"),
        Mesh3d(placement_lattice_mesh),
        MeshMaterial3d(placement_lattice_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        PlacementLatticeVisual::default(),
    ));
    commands.spawn((
        Name::new("Smart placement guides"),
        Mesh3d(smart_guide_mesh),
        MeshMaterial3d(smart_guide_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        SmartGuideVisual::default(),
    ));
    commands.spawn((
        Name::new("Smart snap range"),
        Mesh3d(smart_snap_range_mesh),
        MeshMaterial3d(smart_snap_range_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        SmartSnapRangeVisual::default(),
    ));
    let shape_selected_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Name::new("Selected shape node markers"),
        Mesh3d(shape_selected_mesh),
        MeshMaterial3d(shape_selected_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeSelectedVisual,
    ));
    let shape_plane_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR.with_alpha(0.14),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Name::new("Drag plane"),
        Mesh3d(shape_plane_mesh),
        MeshMaterial3d(shape_plane_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapePlaneVisual,
    ));
    commands.spawn((
        Name::new("Drag plane arrows"),
        Mesh3d(shape_arrow_mesh),
        MeshMaterial3d(shape_selected_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeArrowVisual,
    ));
    commands.spawn((
        Name::new("Shape node markers"),
        Mesh3d(shape_node_mesh),
        MeshMaterial3d(joint_xray_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeNodeVisual,
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

    // Filter the authored sky once, then retain the resulting diffuse and
    // roughness-aware specular maps without regenerating them every frame.
    let environment_map = images.add(sky_cubemap(SKY_CUBEMAP_SIZE));

    let player_camera = PlayerCamera::default();
    let projection = Projection::Perspective(PerspectiveProjection {
        fov: settings.camera_fov_degrees().to_radians(),
        ..default()
    });
    let camera_transform = player_camera.apply_pullback(
        PlayerState::default().position + Vec3::Y * camera::EYE_HEIGHT,
        player_camera.look_rotation(),
    );
    commands
        .spawn((
            Name::new("Player camera"),
            Camera3d::default(),
            projection.clone(),
            garage::EXPOSURE,
            Tonemapping::SomewhatBoringDisplayTransform,
            garage::fog(),
            GeneratedEnvironmentMapLight {
                environment_map,
                intensity: SKY_ENVIRONMENT_INTENSITY,
                ..default()
            },
            camera_transform,
            player_camera,
            MainCamera,
            FovCamera,
        ))
        .with_children(|camera| {
            camera.spawn((
                Name::new("Joint x-ray camera"),
                Camera3d::default(),
                projection,
                // The overlay rides the camera that draws last. This pass loads
                // rather than clears, so an overlay painted before it is drawn
                // over by the joints showing through.
                bevy_mosaic::MosaicCamera,
                Camera {
                    order: 2,
                    clear_color: ClearColorConfig::None,
                    ..default()
                },
                Tonemapping::None,
                RenderLayers::layer(1),
                FovCamera,
                Transform::default(),
            ));
        });
}

fn apply_camera_fov(
    settings: Res<AppSettings>,
    mut cameras: Query<&mut Projection, With<FovCamera>>,
) {
    if !settings.is_changed() {
        return;
    }
    let fov = settings.camera_fov_degrees().to_radians();
    for mut projection in &mut cameras {
        set_projection_fov(&mut projection, fov);
    }
}

fn set_projection_fov(projection: &mut Projection, fov_radians: f32) {
    if let Projection::Perspective(perspective) = projection {
        perspective.fov = fov_radians;
    }
}

#[cfg(test)]
mod pause_feature_tests {
    use super::*;

    #[test]
    fn existing_escape_owners_take_priority_before_pause() {
        assert_eq!(
            escape_target(false, false, true, true, true),
            EscapeTarget::ExistingUi
        );
        assert_eq!(
            escape_target(false, false, false, true, true),
            EscapeTarget::ControlPanel
        );
        assert_eq!(
            escape_target(false, false, false, false, true),
            EscapeTarget::WorldState
        );
        assert_eq!(
            escape_target(false, false, false, false, false),
            EscapeTarget::OpenPause
        );
    }

    #[test]
    fn pause_confirmation_cancels_before_pause_continues() {
        assert_eq!(
            escape_target(true, true, true, true, true),
            EscapeTarget::PauseSubmenu
        );
        assert_eq!(
            escape_target(true, false, true, true, true),
            EscapeTarget::PauseMenu
        );
    }

    #[test]
    fn revisions_restore_clean_identity_through_undo_and_redo() {
        let graph = ConstructionGraph::default();
        let state = EditorState::default();
        let mut history = EditorHistory::default();
        assert!(
            !history.is_dirty(),
            "the initial blank construction is clean"
        );

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.is_dirty(), "a successful edit creates a revision");
        history.mark_clean();
        assert!(
            !history.is_dirty(),
            "a successful save marks that revision clean"
        );

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.is_dirty(), "a later edit is dirty");
        history
            .undo(EditorSnapshot::capture(&graph, &state))
            .expect("undo reaches the saved revision");
        assert!(!history.is_dirty());
        history
            .redo(EditorSnapshot::capture(&graph, &state))
            .expect("redo reaches the edited revision");
        assert!(history.is_dirty());
    }

    #[test]
    fn clean_exit_is_immediate_and_dirty_exit_requires_confirmation() {
        assert_eq!(exit_disposition(false), ExitDisposition::Exit);
        assert_eq!(exit_disposition(true), ExitDisposition::ConfirmUnsaved);
    }

    #[test]
    fn both_camera_projections_receive_the_same_fov() {
        let mut projections = [
            Projection::Perspective(PerspectiveProjection::default()),
            Projection::Perspective(PerspectiveProjection::default()),
        ];
        let wanted = 80.0_f32.to_radians();
        for projection in &mut projections {
            set_projection_fov(projection, wanted);
        }
        for projection in projections {
            let Projection::Perspective(perspective) = projection else {
                panic!("test projection remains perspective");
            };
            assert!((perspective.fov - wanted).abs() < f32::EPSILON);
        }
    }
}

pub(crate) fn avatar_material(color: Color) -> StandardMaterial {
    StandardMaterial {
        base_color: color.with_alpha(0.0),
        perceptual_roughness: 0.92,
        alpha_mode: AlphaMode::Blend,
        ..default()
    }
}

pub(crate) fn avatar_pose(position: Vec3, scale: Vec3, rotation: Quat) -> Transform {
    Transform::from_translation(position)
        .with_rotation(rotation)
        .with_scale(scale)
}

#[allow(clippy::too_many_lines)]
pub(crate) fn spawn_player_avatar(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let cube = meshes.add(Cuboid::default());
    let avatar_materials = AvatarMaterials {
        clothing: materials.add(avatar_material(Color::srgb(0.08, 0.48, 0.46))),
        head: materials.add(avatar_material(Color::srgb(0.72, 0.58, 0.46))),
        boots: materials.add(avatar_material(Color::srgb(0.055, 0.065, 0.075))),
    };
    let parts = [
        (
            "Head",
            avatar_materials.head.clone(),
            avatar_pose(
                Vec3::new(0.0, 1.53, 0.0),
                Vec3::new(0.30, 0.32, 0.28),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(0.0, 0.54, 0.04),
                Vec3::new(0.30, 0.32, 0.28),
                Quat::IDENTITY,
            ),
        ),
        (
            "Torso",
            avatar_materials.clothing.clone(),
            avatar_pose(
                Vec3::new(0.0, 1.12, 0.0),
                Vec3::new(0.42, 0.52, 0.24),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(0.0, 0.18, -0.04),
                Vec3::new(0.42, 0.46, 0.24),
                Quat::IDENTITY,
            ),
        ),
        (
            "Left arm",
            avatar_materials.clothing.clone(),
            avatar_pose(
                Vec3::new(-0.29, 1.08, 0.0),
                Vec3::new(0.13, 0.55, 0.13),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(-0.29, 0.16, 0.16),
                Vec3::new(0.13, 0.48, 0.13),
                Quat::from_rotation_x(-0.55),
            ),
        ),
        (
            "Right arm",
            avatar_materials.clothing.clone(),
            avatar_pose(
                Vec3::new(0.29, 1.08, 0.0),
                Vec3::new(0.13, 0.55, 0.13),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(0.29, 0.16, 0.16),
                Vec3::new(0.13, 0.48, 0.13),
                Quat::from_rotation_x(-0.55),
            ),
        ),
        (
            "Left leg",
            avatar_materials.clothing.clone(),
            avatar_pose(
                Vec3::new(-0.12, 0.52, 0.0),
                Vec3::new(0.17, 0.68, 0.18),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(-0.12, -0.05, 0.34),
                Vec3::new(0.17, 0.62, 0.18),
                Quat::from_rotation_x(core::f32::consts::FRAC_PI_2),
            ),
        ),
        (
            "Right leg",
            avatar_materials.clothing.clone(),
            avatar_pose(
                Vec3::new(0.12, 0.52, 0.0),
                Vec3::new(0.17, 0.68, 0.18),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(0.12, -0.05, 0.34),
                Vec3::new(0.17, 0.62, 0.18),
                Quat::from_rotation_x(core::f32::consts::FRAC_PI_2),
            ),
        ),
        (
            "Left boot",
            avatar_materials.boots.clone(),
            avatar_pose(
                Vec3::new(-0.12, 0.10, 0.06),
                Vec3::new(0.19, 0.20, 0.31),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(-0.12, -0.05, 0.72),
                Vec3::new(0.19, 0.20, 0.31),
                Quat::IDENTITY,
            ),
        ),
        (
            "Right boot",
            avatar_materials.boots.clone(),
            avatar_pose(
                Vec3::new(0.12, 0.10, 0.06),
                Vec3::new(0.19, 0.20, 0.31),
                Quat::IDENTITY,
            ),
            avatar_pose(
                Vec3::new(0.12, -0.05, 0.72),
                Vec3::new(0.19, 0.20, 0.31),
                Quat::IDENTITY,
            ),
        ),
    ];
    commands
        .spawn((
            Name::new("Player mannequin"),
            Transform::default(),
            Visibility::Hidden,
            PlayerAvatar,
        ))
        .with_children(|avatar| {
            for (name, material, standing, seated) in parts {
                avatar.spawn((
                    Name::new(name),
                    Mesh3d(cube.clone()),
                    MeshMaterial3d(material),
                    standing,
                    AvatarPart { standing, seated },
                ));
            }
        });
    commands.insert_resource(avatar_materials);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sync_player_avatar(
    player: Res<PlayerState>,
    view: Single<&PlayerCamera, With<MainCamera>>,
    graph: Res<EditorGraph>,
    simulation: Res<AppSimulation>,
    handles: Res<AvatarMaterials>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut root: Single<(&mut Transform, &mut Visibility), With<PlayerAvatar>>,
    mut parts: Query<(&AvatarPart, &mut Transform), Without<PlayerAvatar>>,
) {
    let alpha = camera::avatar_alpha(view.current_pullback());
    *root.1 = if alpha > 0.0 {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for handle in [&handles.clothing, &handles.head, &handles.boots] {
        if let Some(mut material) = materials.get_mut(handle) {
            material.base_color = material.base_color.with_alpha(alpha);
        }
    }
    let seated_pose = player
        .seat
        .and_then(|seat| seat_world_pose(&graph.0, &simulation, seat));
    if let Some((centre, rotation)) = seated_pose {
        root.0.translation = centre;
        root.0.rotation = rotation;
    } else {
        root.0.translation = player.position;
        root.0.rotation = Quat::from_rotation_y(view.yaw);
    }
    for (part, mut transform) in &mut parts {
        *transform = if seated_pose.is_some() {
            part.seated
        } else {
            part.standing
        };
    }
}

/// Edge of each source cubemap face, in texels. Bevy filters this once into a
/// 32x32 diffuse map and a roughness-aware specular mip chain.
const SKY_CUBEMAP_SIZE: u32 = 64;

/// Radiance the sky cubemap is scaled to, in cd/m². A uniform hemisphere of
/// radiance `L` delivers `pi * L` lux, so this is roughly two thousand lux of
/// fill — enough to open the shadows up, far short of flattening them.
const SKY_ENVIRONMENT_INTENSITY: f32 = 700.0;

/// Straight up: cool and bright, the way an overcast sky reads.
const SKY_ZENITH: Vec3 = Vec3::new(0.62, 0.74, 1.0);
/// The band around the horizon, paler than the zenith and near neutral.
const SKY_HORIZON: Vec3 = Vec3::new(0.80, 0.82, 0.88);
/// Straight down: dim and warm, standing in for bounce off the platform.
const SKY_GROUND: Vec3 = Vec3::new(0.26, 0.22, 0.18);

/// Builds the garage's sky-and-ground source cubemap for one-time filtering.
fn sky_cubemap(size: u32) -> Image {
    let edge = f32::from(u16::try_from(size).expect("a cubemap face is a modest number of texels"));
    let mut texels: Vec<u8> =
        Vec::with_capacity(usize::try_from(6 * size * size * 8).expect("the sky map fits memory"));
    for face in 0..6_usize {
        for row in 0..size {
            for column in 0..size {
                let along = |index: u32| {
                    let index =
                        f32::from(u16::try_from(index).expect("a texel index is within its face"));
                    2.0f32.mul_add(index + 0.5, -edge) / edge
                };
                let (u, v) = (along(column), along(row));
                let colour = sky_colour(cubemap_direction(face, u, v));
                for channel in [colour.x, colour.y, colour.z, 1.0] {
                    texels.extend_from_slice(&half_bits(channel).to_le_bytes());
                }
            }
        }
    }
    Image {
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        }),
        ..Image::new(
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            texels,
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}

/// The direction a texel of one cubemap face looks along, in the +X, -X, +Y,
/// -Y, +Z, -Z order the graphics API expects.
fn cubemap_direction(face: usize, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

/// Sky above, ground below, meeting at the horizon.
fn sky_colour(direction: Vec3) -> Vec3 {
    let height = direction.y;
    if height >= 0.0 {
        SKY_HORIZON.lerp(SKY_ZENITH, height.sqrt())
    } else {
        SKY_HORIZON.lerp(SKY_GROUND, (-height).powf(0.7))
    }
}

/// Encodes the sky's finite, non-negative values as IEEE 754 binary16.
fn half_bits(value: f32) -> u16 {
    let bits = value.clamp(0.0, 65_504.0).to_bits();
    let exponent = i32::try_from((bits >> 23) & 0xff).expect("a float exponent fits in i32") - 127;
    if exponent < -14 {
        return 0;
    }
    let exponent = u16::try_from(exponent + 15).expect("a clamped exponent is in range");
    let mantissa = u16::try_from((bits & 0x007f_ffff) >> 13).expect("ten mantissa bits fit in u16");
    (exponent << 10) | mantissa
}

fn requested_history_action(actions: &ButtonInput<GameAction>) -> Option<HistoryAction> {
    if actions.just_pressed(GameAction::Redo) {
        Some(HistoryAction::Redo)
    } else if actions.just_pressed(GameAction::Undo) {
        Some(HistoryAction::Undo)
    } else {
        None
    }
}

fn handle_history_shortcut(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    overlay: Res<ui::UiInput>,
) {
    if overlay.blocks_keyboard() {
        return;
    }
    let Some(action) = requested_history_action(&actions) else {
        return;
    };
    apply_history_action(action, &mut graph.0, &mut state, &mut history);
}

fn apply_history_action(
    action: HistoryAction,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) -> bool {
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

    *graph = Arc::unwrap_or_clone(restored.graph);
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
    state.pipe_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    state.region_drag = None;
    clear_hover(state);
}

/// Whether the Shape tool has something of its own for `Escape` to unwind.
fn shape_tool_is_busy(tool: Option<Tool>, state: &EditorState) -> bool {
    tool == Some(Tool::Shape)
        && (state.region_drag.is_some()
            || state.active_region.is_some()
            || state.vertex_drag.is_some()
            || !state.selected_vertices.is_empty())
}

#[allow(clippy::too_many_arguments)]
fn handle_shortcuts(
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
            clear_held_tool(&mut graph.0, &mut state, &mut selection, &mut hammer);
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
    {
        state.feedback = Some(cycle_orientation(&mut state, tool));
    }
    if actions.just_pressed(GameAction::PipeTurn)
        && selection.active_editor_tool() == Some(Tool::Cylinder)
    {
        state.feedback = Some(begin_pipe_turn(&mut state));
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PipetteSetup {
    Ground,
    Bearing(BearingDimensions),
    Part(PartId),
}

fn clear_held_tool(
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

fn pipette_at_ray(
    graph: &ConstructionGraph,
    state: &EditorState,
    simulation: &AppSimulation,
    origin: Vec3,
    direction: Vec3,
) -> Option<PipetteSetup> {
    if simulation.creation.is_some() {
        let creation = simulation.creation.as_ref()?;
        let part = raycast_simulation(graph, creation, &simulation.transforms, origin, direction);
        let bearing = raycast_simulation_bearings(
            graph,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
            origin,
            direction,
        );
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
    let bearing = raycast_placed_bearings(graph, &state.placed_bearings, origin, direction)
        .and_then(|(index, distance)| {
            Some((state.placed_bearings.get(index)?.dimensions, distance))
        });
    match (part, bearing) {
        (Some(hit), Some((dimensions, distance))) if distance < hit.distance => {
            Some(PipetteSetup::Bearing(dimensions))
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
        (None, Some((dimensions, _))) => Some(PipetteSetup::Bearing(dimensions)),
        (None, None) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_pipette_setup(
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
                    PartSpec::PipeBend(spec) => {
                        material.0 = spec.material;
                        cylinder_settings.bend_radius = spec.dimensions.radius();
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
fn cycle_orientation(state: &mut EditorState, tool: Tool) -> String {
    let sample = state.pointer_position.zip(state.pointer_ray).map(
        |(cursor, (ray_origin, ray_direction))| PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        },
    );
    if let Some(drag) = state.pipe_drag.as_mut() {
        drag.mode = drag.mode.next();
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

fn begin_pipe_turn(state: &mut EditorState) -> String {
    let Some(drag) = state.pipe_drag.as_mut() else {
        return "Hold primary while dragging a pipe before adding a bend".to_owned();
    };
    if drag.choosing_direction {
        return "Aim toward one of the four perpendicular turn directions".to_owned();
    }
    if drag.dimensions.sweep_angle_degrees() != 360 {
        return "Partial-cylinder sectors support straight runs only".to_owned();
    }
    let leg_start = drag.corners.last().copied().unwrap_or(drag.start);
    let leg_length = drag.endpoint.distance(leg_start);
    let previous_radius = drag.bend_radii.last().copied().unwrap_or(0.0);
    let required = previous_radius + drag.pending_radius;
    if leg_length + 1.0e-5 < required {
        return format!("Current leg needs {required:.2} m clearance before another bend");
    }
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
    "Endpoint frozen — aim toward a perpendicular arrow; wheel changes radius".to_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BearingDimensionTarget {
    Outer,
    Inner,
}

fn requested_bearing_dimension_adjustment(
    actions: &ButtonInput<GameAction>,
    tool: Option<Tool>,
    menu_blocks_input: bool,
) -> Option<(BearingDimensionTarget, i8)> {
    if tool != Some(Tool::Bearing) || menu_blocks_input {
        return None;
    }
    [
        (
            GameAction::BearingInnerDecrease,
            BearingDimensionTarget::Inner,
            -1,
        ),
        (
            GameAction::BearingInnerIncrease,
            BearingDimensionTarget::Inner,
            1,
        ),
        (
            GameAction::BearingOuterDecrease,
            BearingDimensionTarget::Outer,
            -1,
        ),
        (
            GameAction::BearingOuterIncrease,
            BearingDimensionTarget::Outer,
            1,
        ),
    ]
    .into_iter()
    .find_map(|(action, target, direction)| {
        actions.just_pressed(action).then_some((target, direction))
    })
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
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut settings: ResMut<BearingToolSettings>,
    mut state: ResMut<EditorState>,
) {
    let Some((target, direction)) = requested_bearing_dimension_adjustment(
        &actions,
        selection.active_editor_tool(),
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
    actions: &ButtonInput<GameAction>,
    tool: Option<Tool>,
    menu_blocks_input: bool,
) -> Option<(CylinderDimensionTarget, i8)> {
    if tool != Some(Tool::Cylinder) || menu_blocks_input {
        return None;
    }
    [
        (
            GameAction::CylinderSweepDecrease,
            CylinderDimensionTarget::Sweep,
            -1,
        ),
        (
            GameAction::CylinderSweepIncrease,
            CylinderDimensionTarget::Sweep,
            1,
        ),
        (
            GameAction::CylinderLengthDecrease,
            CylinderDimensionTarget::Length,
            -1,
        ),
        (
            GameAction::CylinderLengthIncrease,
            CylinderDimensionTarget::Length,
            1,
        ),
        (
            GameAction::CylinderInnerDecrease,
            CylinderDimensionTarget::Inner,
            -1,
        ),
        (
            GameAction::CylinderInnerIncrease,
            CylinderDimensionTarget::Inner,
            1,
        ),
        (
            GameAction::CylinderOuterDecrease,
            CylinderDimensionTarget::Outer,
            -1,
        ),
        (
            GameAction::CylinderOuterIncrease,
            CylinderDimensionTarget::Outer,
            1,
        ),
    ]
    .into_iter()
    .find_map(|(action, target, direction)| {
        actions.just_pressed(action).then_some((target, direction))
    })
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
    actions: Res<ButtonInput<GameAction>>,
    selection: Res<SelectedTool>,
    menu: Res<CreationMenuState>,
    mut settings: ResMut<CylinderToolSettings>,
    mut state: ResMut<EditorState>,
) {
    if state.pipe_drag.is_some() {
        return;
    }
    let Some((target, direction)) = requested_cylinder_dimension_adjustment(
        &actions,
        selection.active_editor_tool(),
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
    state.pipe_drag = None;
    state.delete_drag = None;
    state.wire_drag = None;
    if !selection
        .active_editor_tool()
        .is_some_and(Tool::edits_drives)
    {
        state.selected_controller = None;
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn update_hover(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    actions: Res<ButtonInput<GameAction>>,
    simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    bearing_settings: Res<BearingToolSettings>,
    cylinder_settings: Res<CylinderToolSettings>,
    selected_material: Option<Res<SelectedMaterial>>,
    chroma_brush: Res<ChromaBrush>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    space: Res<State<world::AppSpace>>,
    world_runtime: Res<world::WorldRuntime>,
) {
    let placement_bounds = match space.get() {
        world::AppSpace::Garage => PlacementBounds::GarageBuild,
        world::AppSpace::World => PlacementBounds::World {
            origin: world_runtime.horizontal_origin(),
        },
    };
    state.placement_bounds = placement_bounds;
    if state.block_drag.is_none() && state.pipe_drag.is_none() {
        state.placement_grid = active_placement_grid(&actions);
    }
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        clear_hover(&mut state);
        return;
    }
    let cursor = camera::viewport_center(Vec2::new(window.width(), window.height()));
    let (camera, camera_transform) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        state.pointer_position = None;
        if state.block_drag.is_some() {
            invalidate_block_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.pipe_drag.is_some() {
            invalidate_pipe_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        if state.delete_drag.is_some() {
            invalidate_delete_drag(&mut state, PlacementError::DragPlaneUnavailable);
            return;
        }
        clear_hover(&mut state);
        if let Some(tool) = selection.active_editor_tool() {
            refresh_tool_preview_with_cylinder(
                &graph.0,
                &mut state,
                tool,
                cylinder_settings.dimensions,
                bearing_settings.dimensions,
                selected_material
                    .as_deref()
                    .map_or(ConstructionMaterial::Steel, |value| value.0),
                chroma_brush.appearance,
            );
        }
        return;
    };
    state.pointer_position = Some(cursor);
    state.pointer_ray = Some((ray.origin, ray.direction.as_vec3()));
    let ray_direction = ray.direction.as_vec3();
    let terrain_ground = placement_bounds
        .is_world()
        .then(|| world_runtime.raycast_terrain(ray.origin, ray_direction, 64.0))
        .flatten()
        .map(|(point, distance)| SurfaceHit {
            distance,
            point,
            face: FaceRef::ground(),
        });
    let moving_hit = (*space.get() == world::AppSpace::World)
        .then(|| {
            simulation
                .creation
                .as_ref()
                .and_then(|creation| {
                    raycast_simulation(
                        &graph.0,
                        creation,
                        &simulation.transforms,
                        ray.origin,
                        ray_direction,
                    )
                })
                .filter(|hit| {
                    !simulation.creation.as_ref().is_some_and(|creation| {
                        creation.compounds[hit.body_index as usize].is_static
                    })
                })
        })
        .flatten();
    state.hovered_simulation = moving_hit;
    let nearest_editable =
        raycast_construction_with_ground(&graph.0, ray.origin, ray_direction, terrain_ground)
            .filter(|hit| match hit.face.owner {
                FaceOwner::Ground => true,
                FaceOwner::Part(part) => simulation_part_is_static(&simulation, part),
            });
    state.world_edit_blocker = moving_hit
        .is_some_and(|moving| nearest_editable.is_none_or(|hit| moving.distance <= hit.distance))
        .then_some(WorldEditBlocker::MovingConstruction);
    let raycast_surface = |annulus: Option<(f32, f32)>| {
        let hit = match annulus {
            Some((inner, outer)) if placement_bounds.is_world() => {
                raycast_construction_for_annulus_with_ground(
                    &graph.0,
                    ray.origin,
                    ray_direction,
                    inner,
                    outer,
                    terrain_ground,
                )
            }
            Some((inner, outer)) if placement_bounds == PlacementBounds::GarageBuild => {
                raycast_construction_for_annulus_with_ground(
                    &graph.0,
                    ray.origin,
                    ray_direction,
                    inner,
                    outer,
                    None,
                )
            }
            Some((inner, outer)) => {
                raycast_construction_for_annulus(&graph.0, ray.origin, ray_direction, inner, outer)
            }
            None if placement_bounds.is_world() => raycast_construction_with_ground(
                &graph.0,
                ray.origin,
                ray_direction,
                terrain_ground,
            ),
            None if placement_bounds == PlacementBounds::GarageBuild => {
                raycast_construction_with_ground(&graph.0, ray.origin, ray_direction, None)
            }
            None => raycast_construction(&graph.0, ray.origin, ray_direction),
        };
        let hit = hit.filter(|hit| match hit.face.owner {
            FaceOwner::Ground => true,
            FaceOwner::Part(part) if !placement_bounds.is_world() => graph.0.part(part).is_some(),
            FaceOwner::Part(part) => simulation_part_is_static(&simulation, part),
        });
        if moving_hit
            .is_some_and(|moving| hit.is_none_or(|editable| moving.distance <= editable.distance))
        {
            None
        } else {
            hit
        }
    };
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
    if state.pipe_drag.is_some() {
        refresh_pipe_drag(
            &graph.0,
            &mut state,
            cursor,
            ray.origin,
            ray.direction.as_vec3(),
        );
        return;
    }
    if state.delete_drag.is_some() {
        refresh_delete_drag(
            &graph.0,
            &mut state,
            cursor,
            ray.origin,
            ray.direction.as_vec3(),
        );
        return;
    }
    let Some(tool) = selection.active_editor_tool() else {
        let construction_hit = raycast_surface(None);
        let bearing_hit =
            raycast_placed_bearings(&graph.0, &state.placed_bearings, ray.origin, ray_direction);
        if let Some((bearing, distance)) = bearing_hit
            && construction_hit.is_none_or(|hit| distance <= hit.distance)
        {
            state.hovered = construction_hit;
            state.hovered_bearing = Some(bearing);
        } else {
            state.hovered = construction_hit;
            state.hovered_bearing = None;
        }
        state.preview = None;
        state.cylinder_preview = None;
        return;
    };
    let construction_hit = if actions.pressed(GameAction::Secondary) {
        raycast_surface(None)
    } else {
        match tool {
            Tool::Bearing => raycast_surface(Some((
                bearing_settings.dimensions.inner_diameter(),
                bearing_settings.dimensions.outer_diameter(),
            ))),
            Tool::Cylinder => raycast_surface(Some((
                cylinder_settings.dimensions.inner_diameter(),
                cylinder_settings.dimensions.outer_diameter(),
            ))),
            Tool::Block
            | Tool::Weld
            | Tool::Hammer
            | Tool::Controller
            | Tool::Connector
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Transmission
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
            | Tool::DimensionLink
            | Tool::Shape
            | Tool::Chroma => raycast_surface(None),
        }
    };
    // Wiring aims at the whole joint, hole and pin included, so a wire can be
    // dropped on a bearing without having to hit its thin ring.
    let wiring = tool == Tool::Connector;
    let bearing_hit = if wiring {
        raycast_placed_bearing_discs(&graph.0, &state.placed_bearings, ray.origin, ray_direction)
            .or_else(|| {
                raycast_placed_bearings(&graph.0, &state.placed_bearings, ray.origin, ray_direction)
            })
    } else if matches!(tool, Tool::Block | Tool::Cylinder) || actions.pressed(GameAction::Secondary)
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
        state.free_placement_point = None;
        state.hovered = construction_hit;
        state.hovered_bearing = Some(bearing);
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            tool,
            cylinder_settings.dimensions,
            bearing_settings.dimensions,
            selected_material
                .as_deref()
                .map_or(ConstructionMaterial::Steel, |value| value.0),
            chroma_brush.appearance,
        );
        return;
    }
    let Some(hit) = construction_hit else {
        let free_point = free_placement_point_on_miss(
            tool,
            placement_bounds,
            ray.origin,
            ray_direction,
            state.free_placement.range,
            actions.pressed(GameAction::Secondary),
        );
        clear_hover(&mut state);
        state.free_placement_point = free_point;
        refresh_tool_preview_with_cylinder(
            &graph.0,
            &mut state,
            tool,
            cylinder_settings.dimensions,
            bearing_settings.dimensions,
            selected_material
                .as_deref()
                .map_or(ConstructionMaterial::Steel, |value| value.0),
            chroma_brush.appearance,
        );
        return;
    };
    state.free_placement_point = None;
    state.hovered_bearing = None;
    state.hovered = Some(hit);
    refresh_tool_preview_with_cylinder(
        &graph.0,
        &mut state,
        tool,
        cylinder_settings.dimensions,
        bearing_settings.dimensions,
        selected_material
            .as_deref()
            .map_or(ConstructionMaterial::Steel, |value| value.0),
        chroma_brush.appearance,
    );
}

fn refresh_block_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    _cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, start_guides, press, plane, anchor_span, last_span) = {
        let drag = state
            .block_drag
            .as_ref()
            .expect("block drag was checked by caller");
        (
            drag.start,
            drag.start_guides.clone(),
            drag.press,
            drag.plane,
            drag.anchor_span,
            drag.last_span,
        )
    };
    let gridded_span = if camera::ray_drag_started(press.ray_direction, ray_direction) {
        let Some(span) = block_span_from_rays(
            start.spec,
            plane,
            anchor_span,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            invalidate_block_drag(state, PlacementError::DragPlaneUnavailable);
            return;
        };
        span
    } else {
        anchor_span
    };
    let (span, endpoint_guides) = if state.smart_snap.enabled {
        raycast_placement_plane_point(ray_origin, ray_direction, start.spec, plane).map_or(
            (gridded_span, Vec::new()),
            |pointer| {
                let bounds = state.placement_bounds;
                smart_snap_block_span(
                    &state.snap_index,
                    start.spec,
                    plane,
                    gridded_span,
                    pointer,
                    state.smart_snap.range,
                    |guided_span| {
                        block_box_specs(start.spec, guided_span).is_ok_and(|specs| {
                            validate_block_batch_in_bounds(graph, start, &specs, bounds).is_ok()
                        })
                    },
                )
            },
        )
    } else {
        (gridded_span, Vec::new())
    };
    let mut combined_guides = if state.smart_snap.enabled {
        start_guides
    } else {
        Vec::new()
    };
    for guide in endpoint_guides {
        if !combined_guides.contains(&guide) {
            combined_guides.push(guide);
        }
    }
    if last_span == Some(span) && state.smart_guides == combined_guides {
        return;
    }
    let result = block_box_specs(start.spec, span).and_then(|specs| {
        validate_block_batch_in_bounds(graph, start, &specs, state.placement_bounds)?;
        Ok(specs)
    });
    let drag = state
        .block_drag
        .as_mut()
        .expect("block drag remains active while refreshing");
    drag.span = span;
    drag.last_span = Some(span);
    state.smart_guides = combined_guides;
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

fn refresh_pipe_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
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
        let Some(outgoing) = pipe_turn_direction(direction, press.ray_direction, ray_direction)
        else {
            return;
        };
        let drag = state.pipe_drag.as_mut().expect("pipe drag remains active");
        let corner = drag.endpoint;
        drag.corners.push(corner);
        drag.bend_radii.push(drag.pending_radius);
        drag.directions.push(outgoing);
        drag.endpoint = corner + outgoing * drag.dimensions.axial_length();
        drag.anchor_endpoint = drag.endpoint;
        drag.anchor_dimensions = drag.dimensions;
        drag.press = PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        };
        drag.choosing_direction = false;
        rebuild_pipe_drag(graph, state);
        state.feedback = Some(format!(
            "Turn locked; dragging next leg — radius {:.2} m",
            state
                .pipe_drag
                .as_ref()
                .and_then(|drag| drag.bend_radii.last())
                .copied()
                .unwrap_or_default()
        ));
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
            let anchor_length = anchor_endpoint.distance(leg_start);
            let units = ((anchor_length + current_parameter - press_parameter) / GRID_UNIT_METERS)
                .round()
                .clamp(1.0, 32.0);
            drag.endpoint = leg_start + direction * (units * GRID_UNIT_METERS);
        }
        PipeEditMode::OuterDiameter | PipeEditMode::InnerDiameter => {
            let delta = pipe_pointer_delta(press.ray_direction, ray_direction);
            let steps = (delta / 0.005).round();
            let mut outer = anchor_dimensions.outer_diameter();
            let mut inner = anchor_dimensions.inner_diameter();
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
                PipeEditMode::Length => unreachable!(),
            }
            drag.dimensions =
                CylinderDimensions::new(outer, inner, anchor_dimensions.axial_length())
                    .expect("clamped pipe dimensions remain valid")
                    .with_sweep_angle_degrees(anchor_dimensions.sweep_angle_degrees())
                    .expect("the existing sector sweep remains valid");
        }
    }
    rebuild_pipe_drag(graph, state);
}

fn rebuild_pipe_drag(graph: &ConstructionGraph, state: &mut EditorState) {
    let (points, bend_radii, dimensions, material, appearance) = {
        let drag = state.pipe_drag.as_ref().expect("pipe drag remains active");
        let mut points = Vec::with_capacity(drag.corners.len() + 2);
        points.push(drag.start);
        points.extend(drag.corners.iter().copied());
        points.push(drag.endpoint);
        (
            points,
            drag.bend_radii.clone(),
            drag.dimensions,
            drag.material,
            drag.appearance,
        )
    };
    let result =
        pipe_run_pieces(&points, &bend_radii, dimensions, material).and_then(|mut pieces| {
            for piece in &mut pieces {
                piece.spec = ordinary_part_with_appearance(piece.spec, appearance);
            }
            validate_pipe_run_in_bounds(graph, &pieces, state.placement_bounds)?;
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
    state.pipe_preview_revision = state.pipe_preview_revision.wrapping_add(1);
}

fn ordinary_part_with_appearance(spec: PartSpec, appearance: MaterialAppearance) -> PartSpec {
    match spec {
        PartSpec::Cuboid(spec) => PartSpec::Cuboid(spec.with_appearance(appearance)),
        PartSpec::Cylinder(spec) => PartSpec::Cylinder(spec.with_appearance(appearance)),
        PartSpec::PipeBend(spec) => PartSpec::PipeBend(spec.with_appearance(appearance)),
        authored => authored,
    }
}

fn invalidate_pipe_drag(state: &mut EditorState, error: PlacementError) {
    let drag = state
        .pipe_drag
        .as_mut()
        .expect("pipe drag was checked by caller");
    drag.error = Some(error.clone());
    state.preview_error = Some(error);
}

fn closest_axis_parameter(
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

fn pipe_pointer_delta(anchor: Vec3, current: Vec3) -> f32 {
    let pitch = current.y.asin() - anchor.y.asin();
    let yaw = wrap_angle(current.x.atan2(current.z) - anchor.x.atan2(anchor.z));
    if yaw.abs() >= pitch.abs() {
        yaw
    } else {
        -pitch
    }
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn pipe_turn_direction(incoming: Vec3, anchor_ray: Vec3, current_ray: Vec3) -> Option<Vec3> {
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

fn adjust_pipe_bend_radius(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    direction: i8,
) -> (f32, String) {
    let drag = state
        .pipe_drag
        .as_mut()
        .expect("radius adjustment requires an active pipe drag");
    let minimum = PipeBendDimensions::minimum_radius(drag.dimensions.outer_diameter());
    let current = if drag.choosing_direction || drag.bend_radii.is_empty() {
        drag.pending_radius
    } else {
        *drag.bend_radii.last().expect("a latest bend exists")
    };
    let requested = current + f32::from(direction) * GRID_UNIT_METERS;
    let radius = requested.clamp(minimum, mechanic_core::MAX_PIPE_BEND_RADIUS);
    if drag.choosing_direction || drag.bend_radii.is_empty() {
        drag.pending_radius = radius;
    } else {
        *drag.bend_radii.last_mut().expect("a latest bend exists") = radius;
        drag.pending_radius = radius;
    }
    rebuild_pipe_drag(graph, state);
    let message = if (radius - requested).abs() > 1.0e-5 {
        if requested < minimum {
            format!("Bend radius clamped to minimum {minimum:.2} m for this diameter")
        } else {
            format!(
                "Bend radius clamped to maximum {:.2} m",
                mechanic_core::MAX_PIPE_BEND_RADIUS
            )
        }
    } else {
        format!("Bend radius: {radius:.2} m")
    };
    (radius, message)
}

fn refresh_delete_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    _cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, press, plane, anchor_span, last_span) = {
        let drag = state
            .delete_drag
            .as_ref()
            .expect("delete drag was checked by caller");
        (
            drag.start,
            drag.press,
            drag.plane,
            drag.anchor_span,
            drag.last_span,
        )
    };
    let span = if camera::ray_drag_started(press.ray_direction, ray_direction) {
        let Some(span) = block_span_from_rays(
            start,
            plane,
            anchor_span,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            invalidate_delete_drag(state, PlacementError::DragPlaneUnavailable);
            return;
        };
        span
    } else {
        anchor_span
    };
    if last_span == Some(span) {
        return;
    }
    let result = delete_box_parts(graph, start, span);
    let drag = state
        .delete_drag
        .as_mut()
        .expect("delete drag remains active while refreshing");
    drag.span = span;
    drag.last_span = Some(span);
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

/// Every block whose centre falls inside the cuboid a delete drag spans.
fn delete_box_parts(
    graph: &ConstructionGraph,
    start: CuboidSpec,
    span: IVec3,
) -> Result<Vec<PartId>, PlacementError> {
    let centers = block_box_specs(start, span)?
        .into_iter()
        .map(|spec| spec.pose.translation_position_ticks())
        .collect::<HashSet<_>>();
    Ok(graph
        .parts()
        .filter_map(|(part, spec)| {
            matches!(spec, PartSpec::Cuboid(_))
                .then(|| centers.contains(&spec.pose().translation_position_ticks()))
                .unwrap_or(false)
                .then_some(part)
        })
        .collect())
}

/// Describes what a staged weld costs, or nothing when it costs nothing.
fn weld_lockup_warning(before: &ConstructionGraph, after: &ConstructionGraph) -> Option<String> {
    match builder::newly_locked_bearings(before, after) {
        0 => None,
        1 => Some("This weld locks 1 bearing solid".to_owned()),
        count => Some(format!("This weld locks {count} bearings solid")),
    }
}

fn clear_hover(state: &mut EditorState) {
    state.hovered = None;
    state.hovered_simulation = None;
    state.world_edit_blocker = None;
    state.hovered_bearing = None;
    state.attachment_bearing = None;
    state.preview = None;
    state.cylinder_preview = None;
    state.free_placement_point = None;
    state.bearing_preview_anchor = None;
    state.preview_error = None;
    state.preview_warning = None;
    state.smart_guides.clear();
}

#[allow(clippy::too_many_lines)]
fn refresh_tool_preview_with_cylinder(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    tool: Tool,
    cylinder_dimensions: CylinderDimensions,
    bearing_dimensions: BearingDimensions,
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
) {
    let placement_grid = state.placement_grid;
    state.preview = None;
    state.cylinder_preview = None;
    state.bearing_preview_anchor = None;
    state.attachment_bearing = None;
    state.preview_warning = None;
    state.smart_guides.clear();
    // A shaped face is no longer an axis-aligned rectangle, so nothing can sit
    // flush on it until it is flattened back onto the grid.
    if let Some(hit) = state.hovered
        && !builder::face_is_flat(graph, hit.face)
        && !matches!(tool, Tool::Shape | Tool::Hammer)
    {
        state.preview_error = Some(PlacementError::SurfaceNotFlat);
        return;
    }
    state.preview_error = match (tool, graph.pending()) {
        (Tool::Block, _) => {
            let surface_candidate = state.hovered.and_then(|hit| {
                try_face_geometry_from_ref(hit.face, Some(graph))
                    .is_some()
                    .then(|| {
                        candidate_from_hit_with_grid(
                            graph,
                            hit,
                            placement_grid,
                            state.placement_bounds,
                        )
                    })
                    .map(|mut candidate| {
                        candidate.spec = candidate
                            .spec
                            .with_material(material)
                            .with_appearance(appearance);
                        let smart_snap = state.smart_snap;
                        if smart_snap.enabled {
                            let bounds = state.placement_bounds;
                            let (snapped_candidate, active_guides) = smart_snap_cuboid_candidate(
                                graph,
                                &state.snap_index,
                                hit,
                                candidate,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    validate_block_batch_in_bounds(
                                        graph,
                                        guided,
                                        &[guided.spec],
                                        bounds,
                                    )
                                    .is_ok()
                                },
                            );
                            state.smart_guides = active_guides;
                            candidate = snapped_candidate;
                        }
                        candidate
                    })
            });
            let free_candidate = state.free_placement_point.and_then(|point| {
                let (_, direction) = state.pointer_ray?;
                let mut candidate = free_cuboid_candidate(
                    point,
                    direction,
                    [1; 3],
                    GridRotation::default(),
                    placement_grid,
                    state.placement_bounds,
                );
                candidate.spec = candidate
                    .spec
                    .with_material(material)
                    .with_appearance(appearance);
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = smart_snap_free_cuboid_candidate(
                        &state.snap_index,
                        candidate,
                        placement_grid,
                        smart_snap.range,
                        |guided| {
                            validate_block_batch_in_bounds(graph, guided, &[guided.spec], bounds)
                                .is_ok()
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                Some(candidate)
            });
            let placement_candidate = surface_candidate.or(free_candidate);
            let direct_bearing = state.hovered_bearing.filter(|&index| {
                state.placed_bearings.get(index).is_some_and(|bearing| {
                    placement_candidate.is_none_or(|candidate| {
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
                placement_candidate.and_then(|candidate| {
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
                let mut candidate = placement_candidate.unwrap_or_else(|| {
                    let mut candidate =
                        bearing_attachment_candidate(graph, bearing.source, bearing.anchor);
                    candidate.spec = candidate
                        .spec
                        .with_material(material)
                        .with_appearance(appearance);
                    candidate
                });
                candidate.support = PlacementSupport::Bearing;
                let error = stage_bearing_attachment_in_bounds(
                    graph,
                    candidate,
                    bearing.source,
                    bearing.anchor,
                    bearing.dimensions,
                    state.placement_bounds,
                )
                .err();
                state.preview = Some(candidate);
                error
            } else {
                placement_candidate.and_then(|candidate| {
                    let error = validate_block_batch_in_bounds(
                        graph,
                        candidate,
                        &[candidate.spec],
                        state.placement_bounds,
                    )
                    .err();
                    state.preview = Some(candidate);
                    error
                })
            }
        }
        (Tool::Weld, Some(PendingOperation::Weld(first))) => {
            state.hovered.and_then(|hit| {
                match stage_weld_objects(graph, first.owner, hit.face.owner) {
                    // A weld that closes a loop is allowed; if it also leaves a
                    // bearing with both sides in one body, say so before the
                    // click rather than leaving the player to wonder why
                    // nothing turns.
                    Ok(staged) => {
                        state.preview_warning = weld_lockup_warning(graph, &staged);
                        None
                    }
                    Err(error) => Some(error),
                }
            })
        }
        (Tool::Cylinder, _) => {
            let surface_candidate = state.hovered.and_then(|hit| {
                let mut candidate = cylinder_candidate_from_hit_with_grid(
                    graph,
                    hit,
                    cylinder_dimensions,
                    placement_grid,
                    state.placement_bounds,
                )
                .ok()?;
                candidate.spec = candidate
                    .spec
                    .with_material(material)
                    .with_appearance(appearance);
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = smart_snap_cylinder_candidate(
                        graph,
                        &state.snap_index,
                        hit,
                        candidate,
                        placement_grid,
                        smart_snap.range,
                        |guided| {
                            validate_cylinder_candidate_in_bounds(graph, guided, bounds).is_ok()
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                Some(candidate)
            });
            let free_candidate = state.free_placement_point.and_then(|point| {
                let (_, direction) = state.pointer_ray?;
                let mut candidate = free_cylinder_candidate(
                    point,
                    direction,
                    cylinder_dimensions,
                    placement_grid,
                    state.placement_bounds,
                );
                candidate.spec = candidate
                    .spec
                    .with_material(material)
                    .with_appearance(appearance);
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = smart_snap_free_cylinder_candidate(
                        &state.snap_index,
                        candidate,
                        placement_grid,
                        smart_snap.range,
                        |guided| {
                            validate_cylinder_candidate_in_bounds(graph, guided, bounds).is_ok()
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                Some(candidate)
            });
            let placement_candidate = surface_candidate.or(free_candidate);
            let direct_bearing = state.hovered_bearing.filter(|&index| {
                state.placed_bearings.get(index).is_some_and(|bearing| {
                    placement_candidate.is_none_or(|candidate| {
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
                placement_candidate.and_then(|candidate| {
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
                placement_candidate
                    .or_else(|| {
                        let hit = SurfaceHit {
                            distance: 0.0,
                            point: bearing.anchor,
                            face: bearing.source,
                        };
                        cylinder_candidate_from_hit_with_grid(
                            graph,
                            hit,
                            cylinder_dimensions,
                            placement_grid,
                            state.placement_bounds,
                        )
                        .ok()
                        .map(|mut candidate| {
                            candidate.spec = candidate
                                .spec
                                .with_material(material)
                                .with_appearance(appearance);
                            candidate
                        })
                    })
                    .map(|mut candidate| {
                        candidate.support = PlacementSupport::Bearing;
                        candidate
                    })
            } else {
                placement_candidate
            };
            candidate.and_then(|candidate| {
                let error =
                    validate_cylinder_candidate_in_bounds(graph, candidate, state.placement_bounds)
                        .err();
                state.cylinder_preview = Some(candidate);
                error
            })
        }
        (Tool::Transmission, _) => state.hovered.and_then(|hit| {
            match transmission_candidate_from_hit_in_bounds(graph, hit, state.placement_bounds) {
                Ok((_, candidate)) => {
                    state.preview = Some(candidate);
                    None
                }
                Err(error) => {
                    if try_face_geometry_from_ref(hit.face, Some(graph)).is_some() {
                        state.preview = Some(oriented_cuboid_candidate_from_hit_with_grid(
                            graph,
                            hit,
                            TransmissionSpec::GRID_UNITS,
                            GridRotation::default(),
                            placement_grid,
                            state.placement_bounds,
                        ));
                    }
                    Some(error)
                }
            }
        }),
        (
            tool @ (Tool::Controller
            | Tool::GasEngine
            | Tool::ElectricEngine
            | Tool::Servo
            | Tool::Seat
            | Tool::Input
            | Tool::DimensionLink),
            _,
        ) => {
            let dimensions = match tool {
                Tool::Controller => ControllerSpec::GRID_UNITS,
                Tool::GasEngine => EngineKind::Gas.grid_units(),
                Tool::ElectricEngine => EngineKind::Electric.grid_units(),
                Tool::Servo => ServoSpec::GRID_UNITS,
                Tool::Seat => SeatSpec::GRID_UNITS,
                Tool::Input => InputSpec::GRID_UNITS,
                Tool::DimensionLink => DimensionLinkSpec::GRID_UNITS,
                _ => unreachable!(),
            };
            let rotation = authored_orientation(state.authored_orientation);
            let surface = state
                .hovered
                .filter(|hit| try_face_geometry_from_ref(hit.face, Some(graph)).is_some())
                .map(|hit| {
                    (
                        oriented_cuboid_candidate_from_hit_with_grid(
                            graph,
                            hit,
                            dimensions,
                            rotation,
                            placement_grid,
                            state.placement_bounds,
                        ),
                        Some(hit),
                    )
                });
            let free = state.free_placement_point.and_then(|point| {
                let (_, direction) = state.pointer_ray?;
                Some((
                    free_cuboid_candidate(
                        point,
                        direction,
                        dimensions,
                        rotation,
                        placement_grid,
                        state.placement_bounds,
                    ),
                    None,
                ))
            });
            surface.or(free).and_then(|(mut candidate, hit)| {
                let smart_snap = state.smart_snap;
                if smart_snap.enabled {
                    let bounds = state.placement_bounds;
                    let (snapped_candidate, active_guides) = hit.map_or_else(
                        || {
                            smart_snap_free_cuboid_candidate(
                                &state.snap_index,
                                candidate,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    validate_block_batch_in_bounds(
                                        graph,
                                        guided,
                                        &[guided.spec],
                                        bounds,
                                    )
                                    .is_ok()
                                },
                            )
                        },
                        |hit| {
                            smart_snap_cuboid_candidate(
                                graph,
                                &state.snap_index,
                                hit,
                                candidate,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    validate_block_batch_in_bounds(
                                        graph,
                                        guided,
                                        &[guided.spec],
                                        bounds,
                                    )
                                    .is_ok()
                                },
                            )
                        },
                    );
                    state.smart_guides = active_guides;
                    candidate = snapped_candidate;
                }
                let error = validate_block_batch_in_bounds(
                    graph,
                    candidate,
                    &[candidate.spec],
                    state.placement_bounds,
                )
                .err();
                state.preview = Some(candidate);
                error
            })
        }
        // Shaping edits the grid rather than placing anything, so like these
        // it has no placement ghost of its own.
        (Tool::Weld | Tool::Hammer | Tool::Connector | Tool::Shape | Tool::Chroma, _) => None,
        (Tool::Bearing, _) => state.hovered.and_then(|hit| {
            if try_face_geometry_from_ref(hit.face, Some(graph)).is_none() {
                Some(PlacementError::CurvedSurface)
            } else {
                match bearing_anchor_from_hit_with_grid(
                    graph,
                    hit,
                    placement_grid,
                    state.placement_bounds,
                ) {
                    Ok(mut anchor) => {
                        let smart_snap = state.smart_snap;
                        if smart_snap.enabled {
                            let normal_axis = PlacementPlane::from_normal(
                                face_geometry_from_ref(hit.face, Some(graph)).normal,
                            )
                            .normal_axis();
                            let (snapped_anchor, active_guides) = smart_snap_anchor(
                                &state.snap_index,
                                anchor,
                                normal_axis,
                                placement_grid,
                                smart_snap.range,
                                |guided| {
                                    bearing_support_face(
                                        graph,
                                        hit.face,
                                        guided,
                                        bearing_dimensions,
                                    )
                                    .is_some()
                                },
                            );
                            anchor = snapped_anchor;
                            state.smart_guides = active_guides;
                        }
                        state.bearing_preview_anchor = Some(anchor);
                        None
                    }
                    Err(error) => Some(error),
                }
            }
        }),
    };
}

#[cfg(test)]
fn refresh_tool_preview(graph: &ConstructionGraph, state: &mut EditorState, tool: Tool) {
    state.snap_index.rebuild(graph);
    refresh_tool_preview_with_cylinder(
        graph,
        state,
        tool,
        CylinderDimensions::default(),
        BearingDimensions::default(),
        ConstructionMaterial::Steel,
        MaterialAppearance::BAKED,
    );
}

/// Selecting an editable area, then hovering, dragging, and mirroring its cage.
///
/// Shaping edits a region rather than a part, so it runs beside the placement
/// tools rather than through them. A drag previews live by rebuilding the
/// construction mesh each frame and commits one batched command on release,
/// which keeps a whole symmetric edit to a single undo entry.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn handle_shape_actions(
    actions: Res<ButtonInput<GameAction>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    mut mirror: ResMut<shape_tool::ShapeMirror>,
    mut snap: ResMut<shape_tool::ShapeSnap>,
    mode: Res<shape_tool::ShapeEditMode>,
    _simulation: Res<AppSimulation>,
    selection: Res<SelectedTool>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
    camera_transform: Single<&GlobalTransform, With<MainCamera>>,
) {
    if selection.active_editor_tool() != Some(Tool::Shape) {
        if state.vertex_drag.take().is_some() {
            state.construction_mesh_dirty = true;
        }
        leave_region(&mut state);
        leave_feature_shape(&mut state);
        return;
    }
    if mode.is_changed() {
        if state.vertex_drag.take().is_some() || state.feature_drag.take().is_some() {
            state.construction_mesh_dirty = true;
        }
        state.hovered_vertex = None;
        state.edge_offer = None;
        state.paint_selecting = false;
        state.selected_vertices.clear();
        state.hovered_feature_edge = None;
        state.selected_feature_edges.clear();
        state.selected_shape_feature = None;
        if *mode != shape_tool::ShapeEditMode::Vertex {
            *snap = shape_tool::ShapeSnap::feature_default();
        }
        state.feedback = Some(format!("Shape mode: {} — {}", mode.label(), snap.label()));
    }
    // Regions can vanish under the tool when their blocks are deleted.
    if state
        .active_region
        .is_some_and(|id| graph.0.region(id).is_none())
    {
        leave_region(&mut state);
    }
    if *mode != shape_tool::ShapeEditMode::Vertex {
        handle_feature_shape_actions(
            &actions,
            &keys,
            &mut graph.0,
            &mut state,
            &mut history,
            *snap,
            *mode,
            *overlay,
            &player,
            &wheel,
        );
        return;
    }
    if state
        .feature_focus
        .is_some_and(|owner| matches!(owner, mechanic_core::SolidOwner::Part(_)))
        && state.active_region.is_none()
    {
        state.feedback = Some(
            "Vertex editing is unavailable for this solid; choose Chamfer or Fillet".to_owned(),
        );
        return;
    }
    if handle_shape_keyboard(
        &actions,
        &camera_transform,
        &mut graph.0,
        &mut state,
        &mut history,
        &mut mirror,
        &mut snap,
    ) {
        return;
    }
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        return;
    }
    let Some((ray_origin, ray_direction)) = state.pointer_ray else {
        return;
    };
    let pointer_position = state.pointer_position;

    if actions.just_pressed(GameAction::Secondary) {
        if state.region_drag.take().is_some() {
            state.feedback = Some("Area selection cancelled".to_owned());
        } else if state.vertex_drag.take().is_some() {
            state.construction_mesh_dirty = true;
            state.feedback = Some("Shape drag cancelled".to_owned());
        } else if state.paint_selecting || !state.selected_vertices.is_empty() {
            state.paint_selecting = false;
            state.selected_vertices.clear();
            state.feedback = Some("Selection cleared".to_owned());
        } else if state.active_region.take().is_some() {
            state.construction_mesh_dirty = true;
            state.feedback = Some("Left the region".to_owned());
        }
        return;
    }

    // Without a region in hand the tool is a chooser: the same drag the Block
    // tool uses, claiming an area instead of filling one.
    let Some(region_id) = state.active_region else {
        choose_region(
            &actions,
            &mut graph.0,
            &mut state,
            &mut history,
            pointer_position,
            ray_origin,
            ray_direction,
        );
        return;
    };
    let Some(region) = graph.0.region(region_id).cloned() else {
        return;
    };

    if let Some(drag) = state.vertex_drag.as_mut() {
        let offset = shape_tool::drag_offset(&region, drag, *snap, ray_origin, ray_direction);
        if offset != drag.offset {
            drag.offset = offset;
            state.construction_mesh_dirty = true;
        }
        if actions.just_released(GameAction::Primary) {
            let drag = state.vertex_drag.take().expect("a drag is in progress");
            if drag.offset == drag.start_offset {
                select_clicked_vertex(&mut state, drag.index, shift_held(&actions));
            } else {
                commit_vertex_drag(
                    &mut graph.0,
                    &mut state,
                    &mut history,
                    region_id,
                    &region,
                    &drag,
                    *mirror,
                );
            }
        }
        return;
    }

    state.hovered_vertex = shape_tool::hovered_vertex(&region, ray_origin, ray_direction);
    state.edge_offer = state
        .hovered_vertex
        .is_none()
        .then(|| shape_tool::edge_insertion(&region, ray_origin, ray_direction))
        .flatten();

    if state.paint_selecting {
        if actions.just_released(GameAction::Primary)
            || !actions.pressed(GameAction::Primary)
            || !shift_held(&actions)
        {
            state.paint_selecting = false;
            state.feedback = Some(format!(
                "Selected {} corners",
                state.selected_vertices.len()
            ));
        } else if let Some(index) = state.hovered_vertex
            && !state.selected_vertices.contains(&index)
        {
            state.selected_vertices.push(index);
        }
        return;
    }

    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    if shift_held(&actions) {
        state.paint_selecting = true;
        if let Some(index) = state.hovered_vertex
            && !state.selected_vertices.contains(&index)
        {
            state.selected_vertices.push(index);
        }
        return;
    }
    if let Some(index) = state.hovered_vertex {
        let drag = shape_tool::begin_group_drag(
            &region,
            index,
            &state.selected_vertices,
            ray_origin,
            ray_direction,
        );
        // Grabbing a vertex outside the selection abandons it, which is what
        // makes starting over cost nothing.
        if drag.group.is_empty() {
            state.selected_vertices.clear();
        }
        state.feedback = Some(format!(
            "Moving on {} axis — Rotate changes axis",
            drag.axis_label()
        ));
        state.vertex_drag = Some(drag);
    } else if let Some(offer) = state.edge_offer {
        subdivide_region(&mut graph.0, &mut state, &mut history, region_id, offer);
    }
}

/// Drops everything that only makes sense while a region is being edited.
fn leave_region(state: &mut EditorState) {
    if state.active_region.take().is_some() {
        state.construction_mesh_dirty = true;
    }
    state.hovered_vertex = None;
    state.paint_selecting = false;
    state.edge_offer = None;
    state.region_drag = None;
    state.selected_vertices.clear();
}

fn leave_feature_shape(state: &mut EditorState) {
    state.feature_focus = None;
    state.hovered_feature_edge = None;
    state.selected_feature_edges.clear();
    state.feature_drag = None;
    state.selected_shape_feature = None;
    state.hovered_source_feature = None;
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn handle_feature_shape_actions(
    actions: &ButtonInput<GameAction>,
    keys: &ButtonInput<KeyCode>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    snap: shape_tool::ShapeSnap,
    mode: shape_tool::ShapeEditMode,
    overlay: ui::UiInput,
    player: &PlayerState,
    wheel: &MaterialWheelState,
) {
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        return;
    }
    let Some((ray_origin, ray_direction)) = state.pointer_ray else {
        return;
    };

    if let Some(feature) = state.selected_shape_feature {
        if keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::Backspace) {
            let snapshot = EditorSnapshot::capture(graph, state);
            match graph.apply(BuildCommand::RemoveShapeFeature(feature)) {
                Ok(_) => {
                    history.commit(snapshot);
                    state.selected_shape_feature = None;
                    state.selected_feature_edges.clear();
                    state.construction_mesh_dirty = true;
                    state.feedback = Some("Removed feature".to_owned());
                }
                Err(error) => {
                    state.feedback = Some(format!("Cannot remove feature: {error}"));
                }
            }
            return;
        }
        let direction = if actions.just_pressed(GameAction::NudgeRight)
            || actions.just_pressed(GameAction::NudgeUp)
        {
            1_i64
        } else if actions.just_pressed(GameAction::NudgeLeft)
            || actions.just_pressed(GameAction::NudgeDown)
        {
            -1_i64
        } else {
            0
        };
        if direction != 0
            && let Some(existing) = graph.shape_feature(feature).cloned()
        {
            let amount = (i64::from(existing.amount_ticks) + direction * i64::from(snap.steps))
                .max(i64::from(snap.steps));
            let amount = u32::try_from(amount).unwrap_or(u32::MAX);
            let snapshot = EditorSnapshot::capture(graph, state);
            match graph.apply(BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: amount,
            }) {
                Ok(_) => {
                    history.commit(snapshot);
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(feature_amount_label(existing.treatment, amount));
                }
                Err(error) => state.feedback = Some(format!("Cannot adjust feature: {error}")),
            }
            return;
        }
    }

    if actions.just_pressed(GameAction::Secondary) {
        if state.feature_drag.take().is_some() {
            state.construction_mesh_dirty = true;
            state.feedback = Some("Feature drag cancelled".to_owned());
        } else if !state.selected_feature_edges.is_empty()
            || state.selected_shape_feature.take().is_some()
        {
            state.selected_feature_edges.clear();
            state.feedback = Some("Edge selection cleared".to_owned());
        } else if state.feature_focus.take().is_some() {
            state.feedback = Some("Left the solid".to_owned());
        }
        return;
    }

    if let Some(drag) = state.feature_drag.as_mut() {
        let proposed = drag.proposed_amount(snap, ray_origin, ray_direction);
        let clamped = if proposed == drag.amount_ticks {
            drag.amount_ticks
        } else {
            clamp_feature_amount(graph, drag, proposed, snap.steps.cast_unsigned())
        };
        if clamped != proposed {
            drag.discard_rejected_excess(clamped);
        }
        if clamped != drag.amount_ticks {
            drag.amount_ticks = clamped;
            state.construction_mesh_dirty = true;
            state.feedback = Some(feature_amount_label(drag.treatment, clamped));
        }
        if actions.just_released(GameAction::Primary) {
            let drag = state.feature_drag.take().expect("feature drag is active");
            if drag.amount_ticks == 0 {
                state.feedback = Some("Edges selected — drag inward to add a feature".to_owned());
                return;
            }
            let snapshot = EditorSnapshot::capture(graph, state);
            let command = if let Some(feature) = drag.feature {
                BuildCommand::SetShapeFeatureAmount {
                    feature,
                    amount_ticks: drag.amount_ticks,
                }
            } else {
                BuildCommand::AddShapeFeature(mechanic_core::ShapeFeature::new(
                    drag.targets.clone(),
                    drag.treatment,
                    drag.amount_ticks,
                ))
            };
            match graph.apply(command) {
                Ok(BuildOutcome::ShapeFeatureAdded(feature)) => {
                    history.commit(snapshot);
                    state.feature_focus = graph
                        .shape_feature(feature)
                        .and_then(|feature| feature.targets.first())
                        .map(|target| target.owner);
                    state.selected_shape_feature = None;
                    state.selected_feature_edges.clear();
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(format!(
                        "Added {}",
                        feature_amount_label(drag.treatment, drag.amount_ticks)
                    ));
                }
                Ok(BuildOutcome::ShapeFeatureUpdated) => {
                    history.commit(snapshot);
                    state.selected_shape_feature = None;
                    state.selected_feature_edges.clear();
                    state.construction_mesh_dirty = true;
                    state.feedback = Some(format!(
                        "Updated {}",
                        feature_amount_label(drag.treatment, drag.amount_ticks)
                    ));
                }
                Ok(_) => unreachable!("feature edits report a feature outcome"),
                Err(error) => state.feedback = Some(format!("Cannot apply feature: {error}")),
            }
        }
        return;
    }

    let pointed_owner = state.hovered.and_then(|hit| match hit.face.owner {
        FaceOwner::Part(part) => Some(graph.region_of(part).map_or(
            mechanic_core::SolidOwner::Part(part),
            mechanic_core::SolidOwner::Region,
        )),
        FaceOwner::Ground => None,
    });
    let owner = pointed_owner.or(state.feature_focus);
    state.hovered_source_feature = None;
    let treatment = match mode {
        shape_tool::ShapeEditMode::Chamfer => mechanic_core::EdgeTreatment::Chamfer,
        shape_tool::ShapeEditMode::Fillet => mechanic_core::EdgeTreatment::Fillet,
        shape_tool::ShapeEditMode::Vertex => unreachable!(),
    };
    let virtual_hit = owner.and_then(|owner| {
        graph
            .shape_features()
            .filter(|(_, feature)| feature.treatment == treatment)
            .filter_map(|(feature_id, feature)| {
                let solid = graph.evaluated_solid_before(owner, feature_id).ok()?;
                feature
                    .targets
                    .iter()
                    .copied()
                    .filter(|target| target.owner == owner)
                    .filter_map(|target| {
                        shape_tool::hovered_source_edge(&solid, target, ray_origin, ray_direction)
                    })
                    .min_by(|left, right| left.distance.total_cmp(&right.distance))
                    .map(|hit| (feature_id, hit))
            })
            .min_by(|left, right| left.1.distance.total_cmp(&right.1.distance))
    });
    if let Some((feature, hit)) = virtual_hit {
        state.hovered_source_feature = Some(feature);
        state.hovered_feature_edge = Some(hit);
    } else {
        state.hovered_feature_edge = owner.and_then(|owner| {
            let solid = graph.evaluated_solid(owner).ok()?;
            shape_tool::hovered_feature_edge(&solid, owner, ray_origin, ray_direction)
        });
    }

    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    let Some(hit) = state.hovered_feature_edge else {
        if pointed_owner.is_some() {
            state.selected_feature_edges.clear();
            state.feature_focus = pointed_owner;
            state.feedback = Some("Aim at a highlighted logical edge".to_owned());
        } else {
            state.feedback = Some("Aim at a construction solid".to_owned());
        }
        return;
    };

    if let Some(feature_id) = state.hovered_source_feature {
        let Some(feature) = graph.shape_feature(feature_id).cloned() else {
            return;
        };
        state.selected_shape_feature = Some(feature_id);
        state.selected_feature_edges.clone_from(&feature.targets);
        state.feature_focus = Some(hit.target.owner);
        state.feature_drag = Some(shape_tool::FeatureDrag::begin(
            hit,
            feature.targets,
            feature.treatment,
            Some(feature_id),
            feature.amount_ticks,
            ray_origin,
            ray_direction,
        ));
        state.feedback = Some(format!(
            "Adjusting {} — drag or use arrows; Delete removes",
            feature_amount_label(feature.treatment, feature.amount_ticks)
        ));
        return;
    }

    if shift_held(actions) {
        if !state.selected_feature_edges.is_empty()
            && !shape_owners_connected(
                graph,
                state.selected_feature_edges[0].owner,
                hit.target.owner,
            )
        {
            state.selected_feature_edges.clear();
        }
        let chain = tangent_feature_chain(graph, hit.target);
        if chain
            .iter()
            .all(|target| state.selected_feature_edges.contains(target))
        {
            state
                .selected_feature_edges
                .retain(|target| !chain.contains(target));
            state.feedback = Some(format!(
                "Selected {} edge chain(s)",
                state.selected_feature_edges.len()
            ));
            return;
        }
        for target in chain {
            if !state.selected_feature_edges.contains(&target) {
                state.selected_feature_edges.push(target);
            }
        }
    } else if !state.selected_feature_edges.contains(&hit.target) {
        state.selected_feature_edges = tangent_feature_chain(graph, hit.target);
    }
    state.feature_focus = Some(hit.target.owner);
    state.selected_shape_feature = None;
    state.feature_drag = Some(shape_tool::FeatureDrag::begin(
        hit,
        state.selected_feature_edges.clone(),
        treatment,
        None,
        0,
        ray_origin,
        ray_direction,
    ));
    state.feedback = Some(format!(
        "Selected {} edge chain(s) — drag inward",
        state.selected_feature_edges.len()
    ));
}

fn clamp_feature_amount(
    graph: &ConstructionGraph,
    drag: &shape_tool::FeatureDrag,
    proposed: u32,
    increment: u32,
) -> u32 {
    let mut amount = proposed;
    while amount > 0 {
        let mut preview = graph.clone();
        let command = if let Some(feature) = drag.feature {
            BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: amount,
            }
        } else {
            BuildCommand::AddShapeFeature(mechanic_core::ShapeFeature::new(
                drag.targets.clone(),
                drag.treatment,
                amount,
            ))
        };
        if preview.apply(command).is_ok() {
            return amount;
        }
        amount = amount.saturating_sub(increment.max(1));
    }
    0
}

fn feature_amount_label(treatment: mechanic_core::EdgeTreatment, amount_ticks: u32) -> String {
    let metres = f64::from(amount_ticks) * f64::from(POSITION_TICK_METERS);
    let name = match treatment {
        mechanic_core::EdgeTreatment::Chamfer => "setback",
        mechanic_core::EdgeTreatment::Fillet => "radius",
    };
    if metres < 0.1 {
        format!("{name}: {:.1} mm", metres * 1000.0)
    } else {
        format!("{name}: {metres:.3} m")
    }
}

fn shape_owners_connected(
    graph: &ConstructionGraph,
    first: mechanic_core::SolidOwner,
    second: mechanic_core::SolidOwner,
) -> bool {
    weld_connected_shape_owners(graph, first).contains(&second)
}

/// Finds the complete weld component with one indexed traversal.
///
/// Edge selection used to repeat a full weld scan for every construction
/// owner. Dense creations made that cubic in practice: selecting one edge in
/// the 504-part fillet test creation performed hundreds of millions of weld
/// checks before tangent matching even began.
fn weld_connected_shape_owners(
    graph: &ConstructionGraph,
    initial: mechanic_core::SolidOwner,
) -> HashSet<mechanic_core::SolidOwner> {
    let starts = match initial {
        mechanic_core::SolidOwner::Part(part) => vec![part],
        mechanic_core::SolidOwner::Region(region) => graph
            .parts()
            .filter_map(|(part, _)| (graph.region_of(part) == Some(region)).then_some(part))
            .collect(),
    };
    if starts.is_empty() {
        return HashSet::new();
    }

    let mut neighbours = HashMap::<PartId, Vec<PartId>>::new();
    for (_, weld) in graph.welds() {
        let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        else {
            continue;
        };
        neighbours.entry(first).or_default().push(second);
        neighbours.entry(second).or_default().push(first);
    }

    let mut reached = starts.iter().copied().collect::<HashSet<_>>();
    let mut pending = starts;
    while let Some(part) = pending.pop() {
        if let Some(adjacent) = neighbours.get(&part) {
            for &next in adjacent {
                if reached.insert(next) {
                    pending.push(next);
                }
            }
        }
    }
    reached
        .into_iter()
        .map(|part| {
            graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            )
        })
        .collect()
}

fn tangent_feature_chain(
    graph: &ConstructionGraph,
    initial: mechanic_core::EdgeChainRef,
) -> Vec<mechanic_core::EdgeChainRef> {
    let mut candidates = Vec::<(mechanic_core::EdgeChainRef, Vec<(Vec3, Vec3)>)>::new();
    let connected = weld_connected_shape_owners(graph, initial.owner);
    let mut owners = graph
        .parts()
        .filter_map(|(part, spec)| {
            ordinary_material(*spec)?;
            let owner = graph.region_of(part).map_or(
                mechanic_core::SolidOwner::Part(part),
                mechanic_core::SolidOwner::Region,
            );
            connected.contains(&owner).then_some(owner)
        })
        .collect::<Vec<_>>();
    owners.sort_unstable();
    owners.dedup();
    for owner in owners {
        let Ok(solid) = graph.evaluated_solid(owner) else {
            continue;
        };
        for logical in &solid.logical_edges {
            if !logical.convex {
                continue;
            }
            let target = mechanic_core::EdgeChainRef {
                owner,
                edge: logical.key,
            };
            let endpoints = logical_chain_endpoints(&solid, logical);
            candidates.push((target, endpoints));
        }
    }
    let mut selected = vec![initial];
    loop {
        let mut additions = Vec::new();
        for selected_target in selected.clone() {
            let Some((_, endpoints)) = candidates
                .iter()
                .find(|(target, _)| *target == selected_target)
            else {
                continue;
            };
            for &(point, tangent) in endpoints {
                let matches = candidates
                    .iter()
                    .filter(|(target, _)| !selected.contains(target))
                    .filter(|(_, candidate_endpoints)| {
                        candidate_endpoints
                            .iter()
                            .any(|(candidate, candidate_tangent)| {
                                point.distance(*candidate) <= mechanic_core::ANCHOR_TOLERANCE_METERS
                                    && tangent.dot(*candidate_tangent).abs() >= 1.0 - 1.0e-4
                            })
                    })
                    .map(|(target, _)| *target)
                    .collect::<Vec<_>>();
                if matches.len() == 1 && !additions.contains(&matches[0]) {
                    additions.push(matches[0]);
                }
            }
        }
        if additions.is_empty() {
            break;
        }
        selected.extend(additions);
    }
    selected
}

fn logical_chain_endpoints(
    solid: &mechanic_core::EvaluatedSolid,
    logical: &mechanic_core::LogicalEdge,
) -> Vec<(Vec3, Vec3)> {
    let mut occurrences = Vec::<(Vec3, Vec3)>::new();
    for &edge_index in &logical.half_edges {
        let edge = solid.half_edges[edge_index as usize];
        let next = solid.half_edges[edge.next as usize];
        let start = solid.vertices[edge.origin as usize].position;
        let end = solid.vertices[next.origin as usize].position;
        let tangent = (end - start).normalize_or_zero();
        occurrences.push((start, tangent));
        occurrences.push((end, tangent));
    }
    occurrences
        .iter()
        .copied()
        .filter(|(point, _)| {
            occurrences
                .iter()
                .filter(|(candidate, _)| {
                    point.distance(*candidate) <= mechanic_core::ANCHOR_TOLERANCE_METERS
                })
                .count()
                == 1
        })
        .collect()
}

/// The area a drag covers: the block it started on, grown by `span` cells.
fn region_area(start: CuboidSpec, span: IVec3) -> ShapeRegion {
    let cells = part_cells(start);
    ShapeRegion::from_origin_steps(
        cells.corner_steps(IVec3::ZERO, 0) + span.min(IVec3::ZERO) * STEPS_PER_CELL,
        cells.counts() + span.abs(),
        start.material,
    )
    .expect("a drag area is at least the block it started on")
}

/// Drags an area of blocks out and claims it as an editable region, using the
/// same gesture the Block tool places with — Rotate included.
fn choose_region(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    cursor: Option<Vec2>,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    if state.region_drag.is_some() {
        if let Some(cursor) = cursor {
            refresh_region_drag(graph, state, cursor, ray_origin, ray_direction);
        }
        if actions.just_released(GameAction::Primary) {
            commit_region_drag(graph, state, history);
        }
        return;
    }
    if !actions.just_pressed(GameAction::Primary) {
        return;
    }
    let Some(hit) = state.hovered else {
        state.feedback = Some("Aim at a block to choose an area".to_owned());
        return;
    };
    let FaceOwner::Part(part) = hit.face.owner else {
        state.feedback = Some("The ground cannot be shaped".to_owned());
        return;
    };
    // Clicking a block already inside a region reopens it rather than refusing.
    if let Some(existing) = graph.region_of(part) {
        state.active_region = Some(existing);
        state.construction_mesh_dirty = true;
        state.feedback = Some("Editing region — drag its corners".to_owned());
        return;
    }
    let Some(start) = graph.part(part).and_then(|spec| spec.as_cuboid()) else {
        state.feedback = Some("Only blocks can be shaped".to_owned());
        return;
    };
    let Some(cursor) = cursor else {
        state.feedback = Some("Pointer position is unavailable".to_owned());
        return;
    };
    let plane =
        PlacementPlane::from_normal(builder::face_geometry_from_ref(hit.face, Some(graph)).normal);
    let region = region_area(start, IVec3::ZERO);
    let error = graph
        .check_region_area(&region)
        .err()
        .map(|error| error.to_string());
    state.region_drag = Some(RegionDrag {
        start,
        press: PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        },
        plane,
        anchor_span: IVec3::ZERO,
        span: IVec3::ZERO,
        last_span: Some(IVec3::ZERO),
        region,
        error,
    });
    state.feedback = Some(format!(
        "Choosing an area on {} plane — release to shape it, Rotate changes plane",
        plane.label()
    ));
}

/// Re-measures the dragged area against the pointer and re-checks the rules.
fn refresh_region_drag(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    _cursor: Vec2,
    ray_origin: Vec3,
    ray_direction: Vec3,
) {
    let (start, press, plane, anchor_span, last_span) = {
        let drag = state
            .region_drag
            .as_ref()
            .expect("a region drag was checked by the caller");
        (
            drag.start,
            drag.press,
            drag.plane,
            drag.anchor_span,
            drag.last_span,
        )
    };
    let span = if camera::ray_drag_started(press.ray_direction, ray_direction) {
        // A plane the pointer cannot reach leaves the last good area standing
        // rather than collapsing the drag.
        let Some(span) = builder::block_span_from_rays(
            start,
            plane,
            anchor_span,
            press.ray_origin,
            press.ray_direction,
            ray_origin,
            ray_direction,
        ) else {
            return;
        };
        span
    } else {
        anchor_span
    };
    if last_span == Some(span) {
        return;
    }
    let region = region_area(start, span);
    let cells = region.size_cells().element_product();
    let error = if cells > i32::try_from(builder::MAX_DRAG_BLOCKS).expect("the cap fits in i32") {
        Some(format!(
            "an area is limited to {} blocks",
            builder::MAX_DRAG_BLOCKS
        ))
    } else {
        graph
            .check_region_area(&region)
            .err()
            .map(|error| error.to_string())
    };
    let drag = state
        .region_drag
        .as_mut()
        .expect("the region drag stays open while refreshing");
    drag.span = span;
    drag.last_span = Some(span);
    drag.region = region;
    drag.error = error;
}

/// Claims the dragged area, if it broke no rule.
fn commit_region_drag(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let drag = state
        .region_drag
        .take()
        .expect("a region drag was checked by the caller");
    if let Some(error) = drag.error {
        state.feedback = Some(format!("Cannot shape: {error}"));
        return;
    }
    let cells = drag.region.size_cells();
    let snapshot = EditorSnapshot::capture(graph, state);
    match graph.apply(BuildCommand::AddRegion(drag.region)) {
        Ok(BuildOutcome::RegionAdded(id)) => {
            history.commit(snapshot);
            state.active_region = Some(id);
            state.construction_mesh_dirty = true;
            state.feedback = Some(format!(
                "Editing {}x{}x{} region — drag its corners",
                cells.x, cells.y, cells.z
            ));
        }
        Ok(_) => unreachable!("adding a region reports the region it added"),
        Err(error) => state.feedback = Some(format!("Cannot shape: {error}")),
    }
}

/// Inserts a cage plane where the pointer offered one.
fn subdivide_region(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    region: RegionId,
    offer: shape_tool::EdgeInsertion,
) {
    let snapshot = EditorSnapshot::capture(graph, state);
    match graph.apply(BuildCommand::SubdivideRegion {
        region,
        axis: offer.axis,
        position: offer.position,
    }) {
        Ok(_) => {
            history.commit(snapshot);
            state.construction_mesh_dirty = true;
            state.edge_offer = None;
            state.feedback = Some("Added a cage vertex".to_owned());
        }
        Err(error) => state.feedback = Some(format!("Cannot subdivide: {error}")),
    }
}

/// The Shape tool's keyboard: mirror planes, the step size, and nudging.
///
/// Returns whether it consumed the frame, which a nudge does so the pointer
/// does not also act on the same input.
#[allow(clippy::too_many_arguments)]
fn handle_shape_keyboard(
    actions: &ButtonInput<GameAction>,
    camera_transform: &GlobalTransform,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    mirror: &mut shape_tool::ShapeMirror,
    snap: &mut shape_tool::ShapeSnap,
) -> bool {
    if actions.just_pressed(GameAction::ShapeMirrorX) {
        mirror.x = !mirror.x;
        state.feedback = Some(mirror.label());
    }
    if actions.just_pressed(GameAction::ShapeMirrorZ) {
        mirror.z = !mirror.z;
        state.feedback = Some(mirror.label());
    }
    if actions.just_pressed(GameAction::ShapeSnap) {
        snap.cycle();
        state.feedback = Some(snap.label());
    }
    let Some((axis, direction)) = nudge_request(actions, camera_transform) else {
        return false;
    };
    nudge_selection(graph, state, history, axis, direction, *snap, *mirror);
    true
}

/// Which way the arrow keys are asking the selection to move.
///
/// The keys read as screen directions and resolve to whichever world axis lies
/// nearest, so a nudge goes where it looks like it should while still landing
/// on the grid. Depth is deliberately absent: orbiting the camera is how the
/// third axis is reached.
fn nudge_request(
    actions: &ButtonInput<GameAction>,
    camera_transform: &GlobalTransform,
) -> Option<(usize, i32)> {
    let (right, up) = (
        camera_transform.right().as_vec3(),
        camera_transform.up().as_vec3(),
    );
    let (basis, sign) = if actions.just_pressed(GameAction::NudgeRight) {
        (right, 1)
    } else if actions.just_pressed(GameAction::NudgeLeft) {
        (right, -1)
    } else if actions.just_pressed(GameAction::NudgeUp) {
        (up, 1)
    } else if actions.just_pressed(GameAction::NudgeDown) {
        (up, -1)
    } else {
        return None;
    };
    let (axis, axis_sign) = shape_tool::screen_axis(basis);
    Some((axis, axis_sign * sign))
}

/// Moves every selected cage vertex one increment, as one undo entry.
fn nudge_selection(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    axis: usize,
    direction: i32,
    snap: shape_tool::ShapeSnap,
    mirror: shape_tool::ShapeMirror,
) {
    let Some(region_id) = state.active_region else {
        return;
    };
    if state.selected_vertices.is_empty() {
        state.feedback = Some("Select a corner first — click one or drag a box".to_owned());
        return;
    }
    let Some(region) = graph.region(region_id).cloned() else {
        return;
    };
    let edits = shape_tool::nudge_edits(
        &region,
        &state.selected_vertices,
        axis,
        direction,
        snap,
        mirror,
    );
    if edits.is_empty() {
        state.feedback = Some("Corner is already as far as it goes".to_owned());
        return;
    }
    let snapshot = EditorSnapshot::capture(graph, state);
    match graph.apply(BuildCommand::SetRegionVertices {
        region: region_id,
        vertices: edits,
    }) {
        Ok(_) => {
            history.commit(snapshot);
            state.construction_mesh_dirty = true;
            state.feedback = Some(format!(
                "Nudged {} corner(s) — {}",
                state.selected_vertices.len(),
                snap.label()
            ));
        }
        Err(error) => state.feedback = Some(format!("Cannot shape: {error}")),
    }
}

/// A click on a vertex picks it for the keyboard; holding shift builds a set up
/// one corner at a time.
fn select_clicked_vertex(state: &mut EditorState, index: CageIndex, extend: bool) {
    if extend {
        if let Some(at) = state
            .selected_vertices
            .iter()
            .position(|&other| other == index)
        {
            state.selected_vertices.remove(at);
        } else {
            state.selected_vertices.push(index);
        }
    } else {
        state.selected_vertices = vec![index];
    }
    state.feedback = Some(match state.selected_vertices.len() {
        0 => "Selection cleared".to_owned(),
        1 => "Corner selected — arrows nudge it".to_owned(),
        count => format!("Selected {count} corners"),
    });
}

fn shift_held(actions: &ButtonInput<GameAction>) -> bool {
    actions.pressed(GameAction::SelectionModifier)
}

/// Commits a finished drag, expanded across the active mirror planes.
fn commit_vertex_drag(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    region_id: RegionId,
    region: &ShapeRegion,
    drag: &shape_tool::VertexDrag,
    mirror: shape_tool::ShapeMirror,
) {
    state.construction_mesh_dirty = true;
    if drag.offset == drag.start_offset {
        return;
    }
    let snapshot = EditorSnapshot::capture(graph, state);
    let edits = shape_tool::drag_edits(region, drag, mirror);
    let count = edits.len();
    match graph.apply(BuildCommand::SetRegionVertices {
        region: region_id,
        vertices: edits,
    }) {
        Ok(_) => {
            history.commit(snapshot);
            state.feedback = Some(if count > 1 {
                format!("Shaped {count} corners")
            } else {
                "Shaped a corner".to_owned()
            });
        }
        Err(error) => {
            state.feedback = Some(format!("Cannot shape: {error}"));
        }
    }
}

/// Fades everything outside the region being edited.
///
/// With a region in hand the rest of the build drops back to a ghost so the
/// area under the cursor is the only thing reading as solid. Leaving the region
/// puts every material back.
fn sync_region_focus(
    state: Res<EditorState>,
    mode: Res<shape_tool::ShapeEditMode>,
    selection: Res<SelectedTool>,
    visuals: Res<EditorVisuals>,
    mut construction_visuals: Query<(
        &ConstructionVisual,
        &mut MeshMaterial3d<ConstructionRenderMaterial>,
    )>,
) {
    let editing = region_focus_is_active(
        selection.active_editor_tool(),
        *mode,
        state.active_region.is_some(),
    );
    for (visual, mut material) in &mut construction_visuals {
        let index = material_index(visual.0);
        let wanted = if editing {
            &visuals.ghost_materials[index]
        } else {
            &visuals.construction_materials[index]
        };
        if material.0.id() != wanted.id() {
            material.0 = wanted.clone();
        }
    }
}

fn region_focus_is_active(
    tool: Option<Tool>,
    mode: shape_tool::ShapeEditMode,
    has_active_region: bool,
) -> bool {
    tool == Some(Tool::Shape)
        && matches!(mode, shape_tool::ShapeEditMode::Vertex)
        && has_active_region
}

/// Draws the active region's cage: its vertices, the edges between them, and
/// the new vertex the pointer is being offered.
///
/// Only vertices near the pointer appear, so choosing the tool does not bury the
/// build in handles.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn sync_shape_nodes(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    mirror: Res<shape_tool::ShapeMirror>,
    mode: Res<shape_tool::ShapeEditMode>,
    selection: Res<SelectedTool>,
    _simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut markers: ShapeOverlay<ShapeNodeVisual, ShapeSelectedVisual, ShapePlaneVisual>,
    mut selected_markers: ShapeOverlay<ShapeSelectedVisual, ShapeNodeVisual, ShapePlaneVisual>,
) {
    let hide = selection.active_editor_tool() != Some(Tool::Shape);

    // Two batches rather than one: a selected corner reads by colour as well as
    // by size, which size alone was not carrying.
    let mut plain = OverlayGeometry::default();
    let mut chosen = OverlayGeometry::default();

    if !hide && *mode != shape_tool::ShapeEditMode::Vertex {
        let owner = state
            .hovered_feature_edge
            .map(|hit| hit.target.owner)
            .or(state.feature_focus);
        if let Some(owner) = owner
            && let Ok(solid) = graph.0.evaluated_solid(owner)
        {
            for logical in &solid.logical_edges {
                if !logical.convex {
                    continue;
                }
                let target = mechanic_core::EdgeChainRef {
                    owner,
                    edge: logical.key,
                };
                let selected = state.selected_feature_edges.contains(&target);
                let hovered = state
                    .hovered_feature_edge
                    .is_some_and(|hit| hit.target == target);
                let geometry = if selected { &mut chosen } else { &mut plain };
                let thickness = if hovered {
                    0.022
                } else if selected {
                    0.017
                } else {
                    0.010
                };
                for &edge_index in &logical.half_edges {
                    let edge = solid.half_edges[edge_index as usize];
                    let next = solid.half_edges[edge.next as usize];
                    let start = solid.vertices[edge.origin as usize].position;
                    let end = solid.vertices[next.origin as usize].position;
                    append_overlay_bar(start, end, thickness, geometry);
                }
            }
            let treatment = match *mode {
                shape_tool::ShapeEditMode::Chamfer => mechanic_core::EdgeTreatment::Chamfer,
                shape_tool::ShapeEditMode::Fillet => mechanic_core::EdgeTreatment::Fillet,
                shape_tool::ShapeEditMode::Vertex => unreachable!(),
            };
            for (feature_id, feature) in graph
                .0
                .shape_features()
                .filter(|(_, feature)| feature.treatment == treatment)
            {
                let Ok(source) = graph.0.evaluated_solid_before(owner, feature_id) else {
                    continue;
                };
                let selected = state.selected_shape_feature == Some(feature_id);
                let geometry = if selected { &mut chosen } else { &mut plain };
                for target in feature
                    .targets
                    .iter()
                    .filter(|target| target.owner == owner)
                {
                    let Some(logical) = source.logical_edge(target.edge) else {
                        continue;
                    };
                    for &edge_index in &logical.half_edges {
                        let edge = source.half_edges[edge_index as usize];
                        let next = source.half_edges[edge.next as usize];
                        append_dashed_overlay_bar(
                            source.vertices[edge.origin as usize].position,
                            source.vertices[next.origin as usize].position,
                            if selected { 0.014 } else { 0.008 },
                            geometry,
                        );
                    }
                }
            }
        }
        **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
        **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
        return;
    }

    // The area being dragged out, cyan while it is claimable and plain while a
    // rule refuses it, so the outline itself carries the verdict.
    if !hide && let Some(drag) = state.region_drag.as_ref() {
        let target = if drag.error.is_some() {
            &mut plain
        } else {
            &mut chosen
        };
        append_region_outline(&drag.region, target);
    }

    let Some((_, region)) = (if hide {
        None
    } else {
        preview_region(&graph.0, &state, *mirror)
    }) else {
        **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
        **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
        return;
    };
    let Some((ray_origin, ray_direction)) = state.pointer_ray else {
        **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
        **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
        return;
    };
    let dragged = state.vertex_drag.as_ref().map(|drag| drag.index);
    for (index, position, distance) in
        shape_tool::revealed_vertices(&region, ray_origin, ray_direction)
    {
        let selected = state.selected_vertices.contains(&index);
        let mut size = shape_tool::vertex_marker_size(distance);
        if selected {
            size *= 1.5;
        }
        if state.hovered_vertex == Some(index) || dragged == Some(index) {
            size *= 1.8;
        }
        let target = if selected { &mut chosen } else { &mut plain };
        append_transformed_cuboid(
            position,
            Quat::IDENTITY,
            Vec3::splat(size),
            &mut target.positions,
            &mut target.normals,
            &mut target.indices,
        );
    }
    // The vertex the pointer is being offered on an edge, shown in the same
    // cyan as a selection because taking it is what it becomes.
    if let Some(offer) = state.edge_offer {
        append_transformed_cuboid(
            offer.at,
            Quat::IDENTITY,
            Vec3::splat(0.024),
            &mut chosen.positions,
            &mut chosen.normals,
            &mut chosen.indices,
        );
    }
    **markers = write_overlay(&mut meshes, &visuals.shape_node_mesh, plain);
    **selected_markers = write_overlay(&mut meshes, &visuals.shape_selected_mesh, chosen);
}

/// Draws the guide for an open drag: a placement plane or one vertex axis.
///
/// Shared by placing blocks and choosing a shape area, because both measure the
/// pointer against a plane and both rotate it with Rotate.
fn sync_drag_plane(
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut plane_marker: ShapeOverlay<ShapePlaneVisual, ShapeArrowVisual, ShapeSelectedVisual>,
    mut arrow_marker: ShapeOverlay<ShapeArrowVisual, ShapePlaneVisual, ShapeSelectedVisual>,
) {
    let mut sheet = OverlayGeometry::default();
    let mut arrows = OverlayGeometry::default();
    if let Some(hit) = state.hovered_feature_edge {
        append_feature_pull_arrow(hit.point, hit.bisector, &mut arrows);
    } else if let Some(drag) = state.pipe_drag.as_ref() {
        let direction = *drag
            .directions
            .last()
            .expect("a pipe run has one direction");
        let absolute = direction.abs();
        let incoming_axis = if absolute.x >= absolute.y && absolute.x >= absolute.z {
            0
        } else if absolute.y >= absolute.z {
            1
        } else {
            2
        };
        if drag.choosing_direction || drag.mode != PipeEditMode::Length {
            for axis in 0..3 {
                if axis != incoming_axis {
                    append_axis_arrows(drag.endpoint, axis, &mut arrows);
                }
            }
        } else {
            append_axis_arrows(drag.endpoint, incoming_axis, &mut arrows);
        }
    } else if let Some((low, high, plane)) = active_drag_plane(&state, &simulation) {
        append_drag_plane(low, high, plane, &mut sheet);
        append_plane_arrows(low, high, plane, &mut arrows);
    } else if let Some(drag) = state.vertex_drag.as_ref() {
        append_axis_arrows(drag.position(), drag.axis, &mut arrows);
    }
    **plane_marker = write_overlay(&mut meshes, &visuals.shape_plane_mesh, sheet);
    **arrow_marker = write_overlay(&mut meshes, &visuals.shape_arrow_mesh, arrows);
}

/// Draws the increasing-amount direction for an edge treatment. Most of the
/// shaft stays outside the solid while the head lands at the edge, so the
/// inward 45-degree direction remains visible instead of disappearing behind
/// the preview mesh.
fn append_feature_pull_arrow(at: Vec3, direction: Vec3, geometry: &mut OverlayGeometry) {
    const OUTSIDE_REACH: f32 = 0.16;
    const TIP_INSET: f32 = 0.015;
    const SHAFT_HALF_WIDTH: f32 = 0.008;
    const HEAD_HALF_WIDTH: f32 = 0.026;
    const HEAD_LENGTH: f32 = 0.06;

    let Some(along) = direction.try_normalize() else {
        return;
    };
    let first_across = along.any_orthonormal_vector();
    let second_across = along.cross(first_across).normalize_or_zero();
    let base = at - along * OUTSIDE_REACH;
    let tip = at + along * TIP_INSET;
    let neck = tip - along * HEAD_LENGTH;
    for across in [first_across, second_across] {
        let normal = along.cross(across);
        append_mesh_quad(
            [
                base - across * SHAFT_HALF_WIDTH,
                base + across * SHAFT_HALF_WIDTH,
                neck + across * SHAFT_HALF_WIDTH,
                neck - across * SHAFT_HALF_WIDTH,
            ],
            normal,
            &mut geometry.positions,
            &mut geometry.normals,
            &mut geometry.indices,
        );
        append_mesh_triangle(
            [
                neck - across * HEAD_HALF_WIDTH,
                neck + across * HEAD_HALF_WIDTH,
                tip,
            ],
            normal,
            &mut geometry.positions,
            &mut geometry.normals,
            &mut geometry.indices,
        );
    }
}

/// The bounds and plane of whichever drag is open, if one is.
fn active_drag_plane(
    state: &EditorState,
    _simulation: &AppSimulation,
) -> Option<(Vec3, Vec3, PlacementPlane)> {
    if let Some(drag) = state.region_drag.as_ref() {
        let (low, high) = region_world_bounds(&drag.region);
        return Some((low, high, drag.plane));
    }
    if let Some(drag) = state.delete_drag.as_ref() {
        let (low, high) = block_box_bounds(drag.start, drag.span);
        return Some((low, high, drag.plane));
    }
    let drag = state.block_drag.as_ref()?;
    let (low, high) = block_sheet_bounds(&drag.specs)?;
    Some((low, high, drag.plane))
}

/// One overlay batch being assembled.
#[derive(Default)]
struct OverlayGeometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

/// Writes one overlay batch into its mesh, reporting whether it has anything to
/// draw.
fn write_overlay(
    meshes: &mut Assets<Mesh>,
    handle: &Handle<Mesh>,
    geometry: OverlayGeometry,
) -> Visibility {
    if geometry.positions.is_empty() {
        return Visibility::Hidden;
    }
    if let Some(mut mesh) = meshes.get_mut(handle) {
        *mesh = renderable_mesh(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, geometry.positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals)
            .with_inserted_indices(Indices::U32(geometry.indices)),
        );
    }
    Visibility::Visible
}

#[allow(clippy::type_complexity, clippy::too_many_lines)]
fn sync_placement_overlays(
    state: Res<EditorState>,
    selection: Res<SelectedTool>,
    actions: Res<ButtonInput<GameAction>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut lattice: Single<
        (&Mesh3d, &mut Visibility, &mut PlacementLatticeVisual),
        (
            With<PlacementLatticeVisual>,
            Without<SmartGuideVisual>,
            Without<SmartSnapRangeVisual>,
        ),
    >,
    mut guides: Single<
        (&Mesh3d, &mut Visibility, &mut SmartGuideVisual),
        (
            With<SmartGuideVisual>,
            Without<PlacementLatticeVisual>,
            Without<SmartSnapRangeVisual>,
        ),
    >,
    mut range: Single<
        (&Mesh3d, &mut Visibility, &mut SmartSnapRangeVisual),
        (
            With<SmartSnapRangeVisual>,
            Without<PlacementLatticeVisual>,
            Without<SmartGuideVisual>,
        ),
    >,
) {
    let tool = selection.active_editor_tool();
    let placing = matches!(
        tool,
        Some(
            Tool::Block
                | Tool::Cylinder
                | Tool::Bearing
                | Tool::Controller
                | Tool::GasEngine
                | Tool::ElectricEngine
                | Tool::Transmission
                | Tool::Servo
                | Tool::Seat
                | Tool::Input
                | Tool::DimensionLink
        )
    );
    let target = state
        .block_drag
        .as_ref()
        .and_then(|drag| {
            block_sheet_bounds(&drag.specs).map(|(low, high)| (low, high, Some(drag.plane)))
        })
        .or_else(|| {
            state
                .pipe_drag
                .as_ref()
                .map(|drag| (drag.endpoint, drag.endpoint, None))
        })
        .or_else(|| {
            state.preview.map(|candidate| {
                let (low, high) = part_world_bounds(PartSpec::Cuboid(candidate.spec));
                (low, high, None)
            })
        })
        .or_else(|| {
            state.cylinder_preview.map(|candidate| {
                let (low, high) = part_world_bounds(PartSpec::Cylinder(candidate.spec));
                (low, high, None)
            })
        })
        .or_else(|| {
            (tool == Some(Tool::Bearing))
                .then_some(state.bearing_preview_anchor?)
                .map(|anchor| (anchor, anchor, None))
        });
    let Some((low, high, plane)) = target.filter(|_| placing) else {
        *lattice.1 = Visibility::Hidden;
        *guides.1 = Visibility::Hidden;
        *range.1 = Visibility::Hidden;
        return;
    };

    let origin = placement_origin_meters(state.placement_bounds);
    let low_ticks = position_ticks(low + origin);
    let high_ticks = position_ticks(high + origin);
    let key = PlacementLatticeKey {
        grid: state.placement_grid,
        low_ticks,
        high_ticks,
        plane,
    };
    if lattice.2.key == Some(key) {
        *lattice.1 = Visibility::Visible;
    } else {
        let geometry = placement_lattice_geometry(
            low_ticks.as_vec3() * POSITION_TICK_METERS - origin,
            high_ticks.as_vec3() * POSITION_TICK_METERS - origin,
            origin,
            state.placement_grid,
            plane,
        );
        *lattice.1 = write_overlay(&mut meshes, &lattice.0.0, geometry);
        lattice.2.key = Some(key);
    }

    if guides.2.guides == state.smart_guides {
        *guides.1 = if state.smart_guides.is_empty() {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    } else {
        let geometry = smart_guide_geometry(&state.smart_guides);
        *guides.1 = write_overlay(&mut meshes, &guides.0.0, geometry);
        guides.2.guides.clone_from(&state.smart_guides);
    }

    if actions.pressed(GameAction::ToggleObjectSnap) {
        let range_ticks = position_tick(state.smart_snap.range);
        let key = SmartSnapRangeKey {
            low_ticks,
            high_ticks,
            range_ticks,
            plane,
        };
        if range.2.key == Some(key) {
            *range.1 = Visibility::Visible;
        } else {
            let geometry = smart_snap_range_geometry(
                low_ticks.as_vec3() * POSITION_TICK_METERS - origin,
                high_ticks.as_vec3() * POSITION_TICK_METERS - origin,
                state.smart_snap.range,
                plane,
            );
            *range.1 = write_overlay(&mut meshes, &range.0.0, geometry);
            range.2.key = Some(key);
        }
    } else {
        *range.1 = Visibility::Hidden;
    }
}

#[allow(clippy::cast_possible_truncation)]
fn placement_origin_meters(bounds: PlacementBounds) -> Vec3 {
    match bounds {
        PlacementBounds::Garage | PlacementBounds::GarageBuild => Vec3::ZERO,
        PlacementBounds::World { origin } => Vec3::new(origin.x as f32, 0.0, origin.y as f32),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn position_ticks(position: Vec3) -> IVec3 {
    (position / POSITION_TICK_METERS).round().as_ivec3()
}

#[allow(clippy::cast_possible_truncation)]
fn position_tick(position: f32) -> i32 {
    (position / POSITION_TICK_METERS).round() as i32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn placement_lattice_geometry(
    selection_low: Vec3,
    selection_high: Vec3,
    origin: Vec3,
    grid: PlacementGrid,
    plane: Option<PlacementPlane>,
) -> OverlayGeometry {
    let mut geometry = OverlayGeometry::default();
    let step = grid.step_ticks() as f32 * POSITION_TICK_METERS;
    let mut low = selection_low - Vec3::splat(step);
    let mut high = selection_high + Vec3::splat(step);
    if let Some(plane) = plane {
        let normal = plane.normal_axis();
        low[normal] = (selection_low[normal] + selection_high[normal]) * 0.5;
        high[normal] = low[normal];
        append_planar_lattice(
            selection_low,
            selection_high,
            low,
            high,
            origin,
            grid,
            plane,
            &mut geometry,
        );
        return geometry;
    }

    let coordinates: [Vec<f32>; 3] = core::array::from_fn(|axis| {
        lattice_coordinates(
            low[axis] + origin[axis],
            high[axis] + origin[axis],
            axis,
            grid,
        )
        .into_iter()
        .map(|global| global - origin[axis])
        .collect::<Vec<_>>()
    });
    for direction in 0..3 {
        let first = (direction + 1) % 3;
        let second = (direction + 2) % 3;
        for &a in &coordinates[first] {
            for &b in &coordinates[second] {
                if coordinate_inside(a, selection_low[first], selection_high[first])
                    && coordinate_inside(b, selection_low[second], selection_high[second])
                {
                    continue;
                }
                let mut at = (low + high) * 0.5;
                at[first] = a;
                at[second] = b;
                let first_tick = ((a + origin[first]) / POSITION_TICK_METERS).round() as i32;
                let second_tick = ((b + origin[second]) / POSITION_TICK_METERS).round() as i32;
                let thickness = lattice_thickness(first, first_tick)
                    .max(lattice_thickness(second, second_tick));
                append_lattice_line(
                    low[direction],
                    high[direction],
                    direction,
                    at,
                    thickness,
                    &mut geometry,
                );
            }
        }
    }
    geometry
}

#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn append_planar_lattice(
    selection_low: Vec3,
    selection_high: Vec3,
    low: Vec3,
    high: Vec3,
    origin: Vec3,
    grid: PlacementGrid,
    plane: PlacementPlane,
    geometry: &mut OverlayGeometry,
) {
    let [first, second] = plane.tangent_axes();
    for (direction, cross) in [(first, second), (second, first)] {
        let coordinates = lattice_coordinates(
            low[cross] + origin[cross],
            high[cross] + origin[cross],
            cross,
            grid,
        );
        for global_coordinate in coordinates {
            let coordinate = global_coordinate - origin[cross];
            let tick = (global_coordinate / POSITION_TICK_METERS).round() as i32;
            let thickness = lattice_thickness(cross, tick);
            let mut at = (low + high) * 0.5;
            at[cross] = coordinate;
            if coordinate_inside(coordinate, selection_low[cross], selection_high[cross]) {
                append_lattice_line(
                    low[direction],
                    selection_low[direction],
                    direction,
                    at,
                    thickness,
                    geometry,
                );
                append_lattice_line(
                    selection_high[direction],
                    high[direction],
                    direction,
                    at,
                    thickness,
                    geometry,
                );
            } else {
                append_lattice_line(
                    low[direction],
                    high[direction],
                    direction,
                    at,
                    thickness,
                    geometry,
                );
            }
        }
    }
}

fn coordinate_inside(coordinate: f32, low: f32, high: f32) -> bool {
    const TOLERANCE: f32 = POSITION_TICK_METERS * 0.25;
    coordinate >= low - TOLERANCE && coordinate <= high + TOLERANCE
}

fn append_lattice_line(
    low: f32,
    high: f32,
    direction: usize,
    mut at: Vec3,
    thickness: f32,
    geometry: &mut OverlayGeometry,
) {
    if high - low <= f32::EPSILON {
        return;
    }
    at[direction] = (low + high) * 0.5;
    let mut half = Vec3::splat(thickness * 0.5);
    half[direction] = (high - low) * 0.5;
    append_transformed_cuboid(
        at,
        Quat::IDENTITY,
        half,
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn lattice_coordinates(low: f32, high: f32, axis: usize, grid: PlacementGrid) -> Vec<f32> {
    let step = grid.step_ticks();
    let phase = if axis == 1 {
        0
    } else {
        POSITION_TICKS_PER_HALF_GRID_UNIT.rem_euclid(step)
    };
    let low_tick = (low / POSITION_TICK_METERS).ceil() as i32;
    let high_tick = (high / POSITION_TICK_METERS).floor() as i32;
    let mut tick = low_tick + (phase - low_tick).rem_euclid(step);
    let mut coordinates = Vec::new();
    while tick <= high_tick {
        coordinates.push(tick as f32 * POSITION_TICK_METERS);
        tick = tick.saturating_add(step);
    }
    coordinates
}

fn lattice_thickness(axis: usize, tick: i32) -> f32 {
    let major_phase = if axis == 1 {
        0
    } else {
        POSITION_TICKS_PER_HALF_GRID_UNIT
    };
    if (tick - major_phase).rem_euclid(POSITION_TICKS_PER_GRID_UNIT) == 0 {
        0.004
    } else if (tick - major_phase).rem_euclid(20) == 0 {
        0.002
    } else {
        0.0008
    }
}

fn smart_guide_geometry(guides: &[SmartGuide]) -> OverlayGeometry {
    let mut geometry = OverlayGeometry::default();
    for guide in guides {
        append_overlay_bar(guide.from, guide.to, 0.007, &mut geometry);
        for point in [guide.from, guide.to] {
            append_transformed_cuboid(
                point,
                Quat::IDENTITY,
                Vec3::splat(0.012),
                &mut geometry.positions,
                &mut geometry.normals,
                &mut geometry.indices,
            );
        }
    }
    geometry
}

fn smart_snap_range_geometry(
    selection_low: Vec3,
    selection_high: Vec3,
    range: f32,
    plane: Option<PlacementPlane>,
) -> OverlayGeometry {
    let mut geometry = OverlayGeometry::default();
    if let Some(plane) = plane {
        let [first, second] = plane.tangent_axes();
        append_snap_range_outline(
            selection_low,
            selection_high,
            range,
            first,
            second,
            plane.normal_axis(),
            &mut geometry,
        );
    } else {
        for (first, second, normal) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
            append_snap_range_outline(
                selection_low,
                selection_high,
                range,
                first,
                second,
                normal,
                &mut geometry,
            );
        }
    }
    geometry
}

#[allow(clippy::too_many_arguments)]
fn append_snap_range_outline(
    selection_low: Vec3,
    selection_high: Vec3,
    range: f32,
    first: usize,
    second: usize,
    normal: usize,
    geometry: &mut OverlayGeometry,
) {
    const CORNER_SEGMENTS: u8 = 8;
    const THICKNESS: f32 = 0.004;

    let normal_coordinate = (selection_low[normal] + selection_high[normal]) * 0.5;
    let point = |first_coordinate: f32, second_coordinate: f32| {
        let mut point = Vec3::ZERO;
        point[first] = first_coordinate;
        point[second] = second_coordinate;
        point[normal] = normal_coordinate;
        point
    };

    for second_coordinate in [
        selection_low[second] - range,
        selection_high[second] + range,
    ] {
        append_overlay_bar(
            point(selection_low[first], second_coordinate),
            point(selection_high[first], second_coordinate),
            THICKNESS,
            geometry,
        );
    }
    for first_coordinate in [selection_low[first] - range, selection_high[first] + range] {
        append_overlay_bar(
            point(first_coordinate, selection_low[second]),
            point(first_coordinate, selection_high[second]),
            THICKNESS,
            geometry,
        );
    }

    for (center_first, center_second, start_angle) in [
        (selection_high[first], selection_high[second], 0.0),
        (
            selection_low[first],
            selection_high[second],
            core::f32::consts::FRAC_PI_2,
        ),
        (
            selection_low[first],
            selection_low[second],
            core::f32::consts::PI,
        ),
        (
            selection_high[first],
            selection_low[second],
            3.0 * core::f32::consts::FRAC_PI_2,
        ),
    ] {
        let mut previous = point(
            center_first + range * start_angle.cos(),
            center_second + range * start_angle.sin(),
        );
        for segment in 1..=CORNER_SEGMENTS {
            let angle = start_angle
                + core::f32::consts::FRAC_PI_2 * f32::from(segment) / f32::from(CORNER_SEGMENTS);
            let next = point(
                center_first + range * angle.cos(),
                center_second + range * angle.sin(),
            );
            append_overlay_bar(previous, next, THICKNESS, geometry);
            previous = next;
        }
    }
}

fn append_overlay_bar(from: Vec3, to: Vec3, thickness: f32, geometry: &mut OverlayGeometry) {
    let delta = to - from;
    let length = delta.length().max(thickness);
    let rotation = if delta.length_squared() <= f32::EPSILON {
        Quat::IDENTITY
    } else {
        Quat::from_rotation_arc(Vec3::X, delta.normalize())
    };
    append_transformed_cuboid(
        (from + to) * 0.5,
        rotation,
        Vec3::new(length * 0.5, thickness * 0.5, thickness * 0.5),
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

fn append_dashed_overlay_bar(from: Vec3, to: Vec3, thickness: f32, geometry: &mut OverlayGeometry) {
    let delta = to - from;
    let length = delta.length();
    if length <= f32::EPSILON {
        return;
    }
    let direction = delta / length;
    let dash = (thickness * 4.0).max(0.018);
    let gap = dash * 0.7;
    let mut start = 0.0;
    while start < length {
        let end = (start + dash).min(length);
        append_overlay_bar(
            from + direction * start,
            from + direction * end,
            thickness,
            geometry,
        );
        start += dash + gap;
    }
}

/// Draws a region's bounding box as twelve thin bars, so a dragged area reads
/// as a volume rather than a face.
fn append_region_outline(region: &ShapeRegion, geometry: &mut OverlayGeometry) {
    const THICKNESS: f32 = 0.012;
    let (low_steps, high_steps) = region.bounds_steps();
    let low = low_steps.as_vec3() * STEP_METERS;
    let high = high_steps.as_vec3() * STEP_METERS;
    let centre = (low + high) * 0.5;
    let extent = high - low;
    for axis in 0..3 {
        let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut size = Vec3::splat(THICKNESS);
        // The bar runs the full length of its axis and overshoots at the ends
        // by its own width, which is what closes the corners.
        size[axis] = extent[axis] + THICKNESS;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let mut at = centre;
            at[first] += a * extent[first] * 0.5;
            at[second] += b * extent[second] * 0.5;
            append_transformed_cuboid(
                at,
                Quat::IDENTITY,
                size * 0.5,
                &mut geometry.positions,
                &mut geometry.normals,
                &mut geometry.indices,
            );
        }
    }
}

/// Draws the plane an area drag is sliding along, as a translucent sheet through
/// the block the drag started on — the same plane the pointer is measured
/// against, so Rotate visibly rotates it.
fn append_drag_plane(low: Vec3, high: Vec3, plane: PlacementPlane, geometry: &mut OverlayGeometry) {
    const THICKNESS: f32 = 0.004;
    /// Overhang past the area, so the sheet reads as a plane rather than a lid.
    const MARGIN: f32 = GRID_UNIT_METERS;
    let normal_axis = plane.normal_axis();
    let mut size = (high - low) + Vec3::splat(MARGIN * 2.0);
    size[normal_axis] = THICKNESS;
    append_transformed_cuboid(
        (low + high) * 0.5,
        Quat::IDENTITY,
        size * 0.5,
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

/// Draws an arrow along each of the drag plane's four cardinal directions, so
/// the plane says which two axes the pointer is driving.
fn append_plane_arrows(
    low: Vec3,
    high: Vec3,
    plane: PlacementPlane,
    geometry: &mut OverlayGeometry,
) {
    /// Clear of the sheet's own slab, so the arrows never fight it for depth.
    const LIFT: f32 = 0.005;
    const SHAFT_HALF_WIDTH: f32 = 0.008;
    const HEAD_HALF_WIDTH: f32 = 0.026;
    const HEAD_LENGTH: f32 = 0.06;
    /// How far an arrow reaches, kept between these so it reads as a gizmo on a
    /// single block and does not span the whole sheet on a large area.
    const MIN_REACH: f32 = 0.14;
    const MAX_REACH: f32 = 0.55;

    let centre = (low + high) * 0.5;
    let extents = (high - low) * 0.5;
    let normal_axis = plane.normal_axis();
    let normal = Vec3::AXES[normal_axis];
    for (index, axis) in plane.tangent_axes().into_iter().enumerate() {
        let along = Vec3::AXES[axis];
        let across = Vec3::AXES[plane.tangent_axes()[1 - index]];
        let reach = (extents[axis] + GRID_UNIT_METERS * 0.5).clamp(MIN_REACH, MAX_REACH);
        let shaft = (reach - HEAD_LENGTH).max(HEAD_LENGTH * 0.5);
        for direction in [1.0_f32, -1.0] {
            let tip = along * (direction * reach);
            let neck = along * (direction * shaft);
            // One copy either side of the sheet, so the arrow reads whichever
            // face of the plane the camera is looking at.
            for side in [1.0_f32, -1.0] {
                let base = centre + normal * (side * LIFT);
                let facing = normal * side;
                append_mesh_quad(
                    [
                        base - across * SHAFT_HALF_WIDTH,
                        base + across * SHAFT_HALF_WIDTH,
                        base + neck + across * SHAFT_HALF_WIDTH,
                        base + neck - across * SHAFT_HALF_WIDTH,
                    ],
                    facing,
                    &mut geometry.positions,
                    &mut geometry.normals,
                    &mut geometry.indices,
                );
                append_mesh_triangle(
                    [
                        base + neck - across * HEAD_HALF_WIDTH,
                        base + neck + across * HEAD_HALF_WIDTH,
                        base + tip,
                    ],
                    facing,
                    &mut geometry.positions,
                    &mut geometry.normals,
                    &mut geometry.indices,
                );
            }
        }
    }
}

/// Draws a two-headed arrow along the one axis a cage vertex may currently
/// move. Two crossed profiles keep it readable from any camera angle.
fn append_axis_arrows(at: Vec3, axis: usize, geometry: &mut OverlayGeometry) {
    const GAP: f32 = 0.035;
    const SHAFT_HALF_WIDTH: f32 = 0.008;
    const HEAD_HALF_WIDTH: f32 = 0.026;
    const HEAD_LENGTH: f32 = 0.06;
    const REACH: f32 = 0.18;

    let along = Vec3::AXES[axis];
    let perpendicular = [(axis + 1) % 3, (axis + 2) % 3];
    for across_axis in perpendicular {
        let across = Vec3::AXES[across_axis];
        let normal = along.cross(across);
        for direction in [1.0_f32, -1.0] {
            let base = at + along * (direction * GAP);
            let neck = at + along * (direction * (REACH - HEAD_LENGTH));
            let tip = at + along * (direction * REACH);
            append_mesh_quad(
                [
                    base - across * SHAFT_HALF_WIDTH,
                    base + across * SHAFT_HALF_WIDTH,
                    neck + across * SHAFT_HALF_WIDTH,
                    neck - across * SHAFT_HALF_WIDTH,
                ],
                normal,
                &mut geometry.positions,
                &mut geometry.normals,
                &mut geometry.indices,
            );
            append_mesh_triangle(
                [
                    neck - across * HEAD_HALF_WIDTH,
                    neck + across * HEAD_HALF_WIDTH,
                    tip,
                ],
                normal,
                &mut geometry.positions,
                &mut geometry.normals,
                &mut geometry.indices,
            );
        }
    }
}

/// A region's bounding box in world metres.
fn region_world_bounds(region: &ShapeRegion) -> (Vec3, Vec3) {
    let (low, high) = region.bounds_steps();
    (low.as_vec3() * STEP_METERS, high.as_vec3() * STEP_METERS)
}

fn appearance_target(graph: &ConstructionGraph, state: &EditorState) -> Option<AppearanceTarget> {
    let hit = state.hovered?;
    let FaceOwner::Part(part) = hit.face.owner else {
        return None;
    };
    graph.part(part)?.appearance()?;
    Some(
        graph
            .region_of(part)
            .map_or(AppearanceTarget::Part(part), AppearanceTarget::Region),
    )
}

fn target_appearance(
    graph: &ConstructionGraph,
    target: AppearanceTarget,
) -> Option<MaterialAppearance> {
    match target {
        AppearanceTarget::Part(part) => graph.part(part)?.appearance(),
        AppearanceTarget::Region(region) => Some(graph.region(region)?.appearance()),
    }
}

fn handle_chroma_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    brush: MaterialAppearance,
) {
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

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
// Tool-specific input flows remain readable together.
fn handle_build_actions(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    _simulation: Res<AppSimulation>,
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
    if state.world_edit_blocker == Some(WorldEditBlocker::MovingConstruction)
        && (actions.just_pressed(GameAction::Primary)
            || actions.just_pressed(GameAction::Secondary))
    {
        state.feedback = Some(
            "Moving constructions cannot be edited; anchor them before changing parts".to_owned(),
        );
        return;
    }
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        if actions.just_released(GameAction::Primary) && state.block_drag.take().is_some() {
            clear_hover(&mut state);
            state.feedback = Some("Block drag cancelled over hotbar".to_owned());
        }
        if actions.just_released(GameAction::Primary) && state.pipe_drag.take().is_some() {
            clear_hover(&mut state);
            state.feedback = Some("Pipe run cancelled over hotbar".to_owned());
        }
        if actions.just_released(GameAction::Secondary) && state.cancel_delete_gesture() {
            state.feedback = Some("Delete drag cancelled over hotbar".to_owned());
        }
        return;
    }
    let Some(tool) = selection.active_editor_tool() else {
        return;
    };
    if tool == Tool::Shape {
        return;
    }
    if tool == Tool::Chroma {
        handle_chroma_actions(
            &actions,
            &mut graph.0,
            &mut state,
            &mut history,
            chroma_brush.appearance,
        );
        return;
    }
    if actions.just_pressed(GameAction::Secondary) && state.block_drag.take().is_some() {
        clear_hover(&mut state);
        state.feedback = Some("Block drag cancelled".to_owned());
        return;
    }
    if actions.just_pressed(GameAction::Secondary) && state.pipe_drag.take().is_some() {
        clear_hover(&mut state);
        state.feedback = Some("Pipe run cancelled".to_owned());
        return;
    }
    if tool == Tool::Connector && actions.just_pressed(GameAction::Secondary) {
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
        }
    }
    if actions.just_released(GameAction::Secondary) {
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
                                clear_hover(&mut state);
                            }
                            Err(error) => state.feedback = Some(error.to_string()),
                        }
                    }
                }
                DeleteTarget::Part(part) => {
                    let deleted_link = graph.0.dimension_link_id(part);
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
                            if let (Some(id), Some(world_runtime)) =
                                (deleted_link, world_runtime.as_deref_mut())
                            {
                                world_runtime.clear_active_dimension_link_if(id);
                            }
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
    if tool == Tool::Connector {
        handle_connector_actions(&actions, &mut graph.0, &mut state, &mut history);
        return;
    }
    if tool == Tool::Block {
        handle_block_actions(&actions, &mut graph.0, &mut state, &mut history);
        return;
    }
    if tool == Tool::Cylinder {
        if state.pipe_bend_active()
            && (actions.just_pressed(GameAction::ZoomIn)
                || actions.just_pressed(GameAction::ZoomOut))
        {
            let direction = i8::from(actions.just_pressed(GameAction::ZoomIn))
                - i8::from(actions.just_pressed(GameAction::ZoomOut));
            let (radius, message) = adjust_pipe_bend_radius(&graph.0, &mut state, direction);
            cylinder_settings.bend_radius = radius;
            state.feedback = Some(message);
            return;
        }
        if actions.just_pressed(GameAction::Primary) {
            let Some(candidate) = state.cylinder_preview else {
                state.feedback = Some("Point at a flat face or compatible bearing".to_owned());
                return;
            };
            let attachment = if let Some(index) = state.attachment_bearing {
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
                bend_radii: Vec::new(),
                pending_radius: cylinder_settings.bend_radius.max(
                    PipeBendDimensions::minimum_radius(candidate.spec.dimensions.outer_diameter()),
                ),
                dimensions: candidate.spec.dimensions,
                material: candidate.spec.material,
                appearance: candidate.spec.appearance,
                mode: PipeEditMode::Length,
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
            state.pipe_preview_revision = state.pipe_preview_revision.wrapping_add(1);
            state.feedback = Some(
                "Dragging pipe length — R cycles dimensions, F adds a 90° bend, release commits"
                    .to_owned(),
            );
            return;
        }
        if actions.just_released(GameAction::Primary) {
            let Some(drag) = state.pipe_drag.take() else {
                return;
            };
            cylinder_settings.dimensions = drag.dimensions;
            cylinder_settings.bend_radius = drag
                .bend_radii
                .last()
                .copied()
                .unwrap_or(drag.pending_radius);
            if drag.choosing_direction {
                state.feedback = Some(
                    "Pipe run not placed: choose a turn direction before releasing".to_owned(),
                );
                clear_hover(&mut state);
                return;
            }
            if let Some(error) = drag.error {
                state.feedback = Some(error.to_string());
                clear_hover(&mut state);
                return;
            }
            let previous = EditorSnapshot::capture(&graph.0, &state);
            let staged = match drag.attachment {
                BlockAttachment::AutoWeld { source } => stage_pipe_run_in_bounds(
                    &graph.0,
                    &drag.pieces,
                    PipeRunAttachment::AutoWeld { source },
                    state.placement_bounds,
                ),
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
                        "Placed pipe run with {count} piece(s) and {} bend(s)",
                        drag.bend_radii.len()
                    ));
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
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
        Tool::Weld => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Select an object".to_owned());
                return;
            };
            if let Some(PendingOperation::Weld(first)) = graph.0.pending() {
                match stage_weld_objects(&graph.0, first.owner, hit.face.owner) {
                    Ok(staged) => {
                        let lockup = weld_lockup_warning(&graph.0, &staged);
                        let previous = EditorSnapshot::capture(&graph.0, &state);
                        graph.0 = staged;
                        history.commit(previous);
                        state.feedback = Some(lockup.map_or_else(
                            || "Welded the two objects".to_owned(),
                            |warning| format!("Welded the two objects — {warning}"),
                        ));
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
                        let previous = EditorSnapshot::capture(&graph.0, &state);
                        state.placed_bearings.push(PlacedBearing {
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
            let previous = EditorSnapshot::capture(&graph.0, &state);
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
                    clear_hover(&mut state);
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
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match stage_engine_in_bounds(&graph.0, candidate, kind, state.placement_bounds) {
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
            let previous = EditorSnapshot::capture(&graph.0, &state);
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
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Servo | Tool::Seat | Tool::Input => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at a face or into free Garage space".to_owned());
                return;
            };
            let previous = EditorSnapshot::capture(&graph.0, &state);
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
                    clear_hover(&mut state);
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
            let previous = EditorSnapshot::capture(&graph.0, &state);
            match stage_dimension_link_in_bounds(&graph.0, candidate, id, state.placement_bounds) {
                Ok(staged) => {
                    graph.0 = staged;
                    history.commit(previous);
                    state.feedback = Some(format!("Placed Dimension Link {}", id.0));
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        Tool::Connector => unreachable!("connector actions are handled before this match"),
        Tool::Chroma => unreachable!("Chroma actions are handled before this match"),
    }
    refresh_tool_preview_with_cylinder(
        &graph.0,
        &mut state,
        tool,
        cylinder_settings.dimensions,
        bearing_settings.dimensions,
        selected_material
            .as_deref()
            .map_or(ConstructionMaterial::Steel, |value| value.0),
        chroma_brush.appearance,
    );
}

fn simulation_part_is_static(simulation: &AppSimulation, part: PartId) -> bool {
    let Some(creation) = simulation.creation.as_ref() else {
        return false;
    };
    creation
        .part_to_compound
        .iter()
        .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
        .is_some_and(|compound| creation.compounds[compound as usize].is_static)
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
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let pressed = actions.just_pressed(GameAction::Primary);
    if !pressed && !actions.just_released(GameAction::Primary) {
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
    state: &mut EditorState,
    history: &mut EditorHistory,
    command: BuildCommand,
    success: &str,
) -> String {
    let previous = EditorSnapshot::capture(graph, state);
    match graph.apply(command) {
        Ok(_) => {
            history.commit(previous);
            // The new link is a line in the drive overlay, and that overlay is
            // only rebuilt on request. Without this the wire stays invisible
            // until something else happens to dirty the mesh.
            state.construction_mesh_dirty = true;
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

    let mut staged = graph.begin_edit();
    match staged.apply_batch(commands) {
        Ok(_) => {
            *graph = staged.finish();
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
    let mut staged = graph.begin_edit();
    match staged.apply_batch(links.iter().copied().map(BuildCommand::RemoveDriveLink)) {
        Ok(_) => {
            *graph = staged.finish();
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

    Ok((staged.finish(), next_bearings, migrated_count))
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
        let (attachment, normal) = if let Some(index) = state.attachment_bearing {
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
    let count = drag.specs.len();
    let previous = EditorSnapshot::capture(graph, state);
    let staged = match drag.attachment {
        BlockAttachment::AutoWeld { .. } | BlockAttachment::Free => {
            stage_block_batch_in_bounds(graph, drag.start, &drag.specs, state.placement_bounds)
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
            stage_bearing_block_batch_in_bounds(
                graph,
                drag.start,
                &drag.specs,
                source,
                anchor,
                dimensions,
                &rigid_targets,
                state.placement_bounds,
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
    actions: Res<ButtonInput<GameAction>>,
    time: Res<Time>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    simulation: Res<AppSimulation>,
    graph: Res<EditorGraph>,
    mut hammer: ResMut<HammerInteraction>,
    mut state: ResMut<EditorState>,
    selection: Res<SelectedTool>,
    overlay: Res<ui::UiInput>,
    player: Res<PlayerState>,
    wheel: Res<MaterialWheelState>,
) {
    if !simulation.is_running() {
        hammer.charging = None;
        hammer.pending = None;
        return;
    }
    if overlay.blocks_pointer() || !player.world_input_active() || wheel.open {
        hammer.charging = None;
        hammer.pending = None;
        return;
    }
    let Some(tool) = selection.active_editor_tool() else {
        hammer.charging = None;
        return;
    };
    if tool != Tool::Hammer {
        hammer.charging = None;
        return;
    }
    if overlay.blocks_pointer() && hammer.charging.is_none() {
        return;
    }
    if actions.just_pressed(GameAction::Primary) {
        hammer.charging = None;
        let cursor = camera::viewport_center(Vec2::new(window.width(), window.height()));
        let hit = {
            let (camera, camera_transform) = *camera;
            camera.viewport_to_world(camera_transform, cursor).ok()
        }
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

    if actions.pressed(GameAction::Primary)
        && let Some(charge) = hammer.charging.as_mut()
    {
        charge.elapsed_seconds =
            (charge.elapsed_seconds + time.delta_secs()).min(HAMMER_CHARGE_SECONDS);
    }

    if !actions.just_released(GameAction::Primary) {
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
        .map(|collider| collider.local_center.length() + collider_reach(collider))
        .fold(0.0_f32, f32::max);
    (linear_delta.length() + angular_delta.length() * maximum_radius) * FIXED_DT_SECONDS
}

/// How far one collider extends from its own centre.
fn collider_reach(collider: &mechanic_core::LocalCollider) -> f32 {
    match &collider.shape {
        mechanic_core::ColliderShape::Cuboid { half_extents, .. } => half_extents.length(),
        mechanic_core::ColliderShape::Convex(convex) => convex
            .vertices
            .iter()
            .map(|vertex| (*vertex - collider.local_center).length())
            .fold(0.0_f32, f32::max),
    }
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
                | PartSpec::Transmission(_)
                | PartSpec::Servo(_)
                | PartSpec::Seat(_)
                | PartSpec::Input(_)
                | PartSpec::DimensionLink(_) => {
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
                PartSpec::PipeBend(_) => {
                    let hit = creation
                        .colliders
                        .iter()
                        .filter(|collider| collider.source_part == part)
                        .filter_map(|collider| {
                            let ColliderShape::Cuboid {
                                local_rotation,
                                half_extents,
                            } = &collider.shape
                            else {
                                return None;
                            };
                            raycast_oriented_cuboid(
                                origin,
                                direction,
                                position + rotation * collider.local_center,
                                rotation * *local_rotation,
                                *half_extents,
                            )
                        })
                        .min_by(|left, right| left.distance.total_cmp(&right.distance))?;
                    (hit.distance, hit.point)
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

#[allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn sync_visual_meshes(
    graph: Res<EditorGraph>,
    world_runtime: Res<world::WorldRuntime>,
    mirror: Res<shape_tool::ShapeMirror>,
    sequencer: Res<DriveSequencer>,
    selection: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    mut state: ResMut<EditorState>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visuals: Query<
        (&ConstructionVisual, &mut Visibility),
        (Without<BearingVisual>, Without<SimulationVisual>),
    >,
    mut bearing_visibility: Single<
        &mut Visibility,
        (
            With<BearingVisual>,
            Without<ConstructionVisual>,
            Without<SimulationVisual>,
        ),
    >,
    mut simulation_visuals: Query<
        (&SimulationVisual, &mut Visibility),
        (Without<ConstructionVisual>, Without<BearingVisual>),
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
    let edit_delta = ConstructionEditDelta::between(&state.rendered_graph, &graph.0);
    let rebuild_all = edit_delta.is_empty();
    let affected_parts = edit_delta.affected_parts();
    let mut dirty_materials = affected_parts
        .iter()
        .filter_map(|&part| {
            graph
                .0
                .part(part)
                .or_else(|| state.rendered_graph.part(part))
                .copied()
                .and_then(ordinary_material)
        })
        .collect::<HashSet<_>>();
    for &region in &edit_delta.region_owned_geometry {
        if let Some(material) = graph
            .0
            .region(region)
            .or_else(|| state.rendered_graph.region(region))
            .map(ShapeRegion::material)
        {
            dirty_materials.insert(material);
        }
    }
    let feature_preview_graph = state.feature_drag.as_ref().and_then(|drag| {
        if drag.amount_ticks == 0 {
            return None;
        }
        let mut preview = graph.0.clone();
        let command = if let Some(feature) = drag.feature {
            BuildCommand::SetShapeFeatureAmount {
                feature,
                amount_ticks: drag.amount_ticks,
            }
        } else {
            BuildCommand::AddShapeFeature(mechanic_core::ShapeFeature::new(
                drag.targets.clone(),
                drag.treatment,
                drag.amount_ticks,
            ))
        };
        preview.apply(command).ok().map(|_| preview)
    });
    let mesh_graph = feature_preview_graph.as_ref().unwrap_or(&graph.0);
    let preview = preview_region(&graph.0, &state, *mirror);
    let active_dimension_link = world_runtime.active_dimension_link();
    for material in ConstructionMaterial::ALL {
        if !rebuild_all && !dirty_materials.contains(&material) {
            continue;
        }
        let mesh = combined_material_construction_mesh(mesh_graph, preview.as_ref(), material);
        let visible = mesh.count_vertices() > 0;
        if let Some(mut asset) =
            meshes.get_mut(&visuals.construction_meshes[material_index(material)])
        {
            *asset = renderable_mesh(mesh);
        }
        for (visual, mut visibility) in &mut construction_visuals {
            if visual.0 == material {
                *visibility = if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
    for (_, mut visibility) in &mut simulation_visuals {
        *visibility = Visibility::Hidden;
    }
    for appearance in AuthoredPart::ALL {
        let appearance_changed = rebuild_all
            || affected_parts.iter().any(|&part| {
                graph.0.part(part).is_some_and(|spec| {
                    appearance.matches(&graph.0, part, *spec, active_dimension_link)
                }) || state.rendered_graph.part(part).is_some_and(|spec| {
                    appearance.matches(&state.rendered_graph, part, *spec, active_dimension_link)
                })
            });
        if !appearance_changed {
            continue;
        }
        let visible = graph
            .0
            .parts()
            .any(|(part, spec)| appearance.matches(&graph.0, part, *spec, active_dimension_link));
        if visible && let Some(mut mesh) = meshes.get_mut(visuals.authored_mesh(appearance)) {
            *mesh =
                combined_authored_construction_mesh(&graph.0, appearance, active_dimension_link);
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
            selection.active_editor_tool(),
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
    if drive_xray_is_visible(selection.active_editor_tool(), control_link_count(&graph.0))
        && let Some(mut mesh) = meshes.get_mut(&visuals.drive_xray_mesh)
    {
        *mesh = combined_drive_xray_mesh(&graph.0, &state.placed_bearings, &sequencer);
    }
    state.rendered_graph = graph.0.clone();
    state.construction_mesh_dirty = false;
}

/// The bearing x-ray is also shown while wiring, so a drive wire can be traced
/// back through the construction to the block that owns it.
fn joint_xray_is_visible(
    tool: impl Into<Option<Tool>>,
    simulating: bool,
    bearing_count: usize,
) -> bool {
    matches!(tool.into(), Some(Tool::Controller | Tool::Connector))
        && !simulating
        && bearing_count > 0
}

/// The drive overlay additionally stays up while simulating, so the joint a key
/// is driving can be seen moving. Its meshes are rebuilt from each published
/// snapshot, so the arcs and wires track the running bodies.
fn drive_xray_is_visible(tool: impl Into<Option<Tool>>, driven_count: usize) -> bool {
    matches!(tool.into(), Some(Tool::Controller | Tool::Connector)) && driven_count > 0
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
    let drive_visible =
        drive_xray_is_visible(selection.active_editor_tool(), control_link_count(&graph.0));
    let joint_visible = joint_xray_is_visible(
        selection.active_editor_tool(),
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

#[derive(bevy::ecs::system::SystemParam)]
struct ChromaPreviewParams<'w> {
    selected_material: Res<'w, SelectedMaterial>,
    brush: Res<'w, ChromaBrush>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
}

#[allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn update_previews(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    _simulation: Res<AppSimulation>,
    selected_tool: Res<SelectedTool>,
    mut chroma: ChromaPreviewParams,
    bearing_settings: Res<BearingToolSettings>,
    cylinder_settings: Res<CylinderToolSettings>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rendered_revisions: Local<PreviewMeshRevisions>,
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

    if selected_tool.active_editor_tool() == Some(Tool::Weld) {
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

    if let Some(drag) = state.delete_drag.as_ref() {
        if rendered_revisions.delete != state.delete_preview_revision {
            let specs = drag
                .parts
                .iter()
                .filter_map(|&part| graph.0.part(part).copied())
                .collect::<Vec<_>>();
            if let Some(mut mesh) = meshes.get_mut(&visuals.delete_drag_preview_mesh) {
                *mesh = combined_parts_mesh_scaled(&specs, 1.0);
            }
            rendered_revisions.delete = state.delete_preview_revision;
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

    let bearing_attachment_highlighted = selected_tool.active_editor_tool().is_some_and(|tool| {
        bearing_attachment_is_highlighted(
            tool,
            state.attachment_bearing,
            state.preview_error.as_ref(),
        )
    });
    let chroma_preview = matches!(
        selected_tool.active_editor_tool(),
        Some(Tool::Block | Tool::Cylinder)
    ) && chroma.brush.appearance != MaterialAppearance::BAKED;
    if chroma_preview {
        let [r, g, b] =
            chroma::representative_srgb(chroma.selected_material.0, chroma.brush.appearance);
        if let Some(mut material) = chroma.materials.get_mut(&visuals.chroma_preview_material) {
            material.base_color = Color::srgb_u8(r, g, b).with_alpha(0.46);
        }
    }
    let action_material = if state.preview_error.is_some() {
        &visuals.red_preview_material
    } else if state.preview_warning.is_some() {
        &visuals.amber_preview_material
    } else if bearing_attachment_highlighted {
        &visuals.green_preview_material
    } else if chroma_preview {
        &visuals.chroma_preview_material
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
    match (selected_tool.active_editor_tool(), graph.0.pending()) {
        (None | Some(Tool::Shape | Tool::Chroma), _) => {
            *action.2 = Visibility::Hidden;
        }
        (Some(Tool::Block), _) => {
            if let Some(drag) = state.block_drag.as_ref() {
                if rendered_revisions.block != state.block_preview_revision {
                    if let Some(mut mesh) = meshes.get_mut(&visuals.block_drag_preview_mesh) {
                        *mesh = block_sheet_preview_mesh(&drag.specs);
                    }
                    rendered_revisions.block = state.block_preview_revision;
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
        (Some(Tool::Cylinder), _) => {
            if let Some(drag) = state.pipe_drag.as_ref() {
                if rendered_revisions.pipe != state.pipe_preview_revision {
                    let specs = drag
                        .pieces
                        .iter()
                        .map(|piece| piece.spec)
                        .collect::<Vec<_>>();
                    if let Some(mut mesh) = meshes.get_mut(&visuals.block_drag_preview_mesh) {
                        *mesh = combined_parts_mesh_scaled(&specs, 1.0);
                    }
                    rendered_revisions.pipe = state.pipe_preview_revision;
                }
                action.0.0 = visuals.block_drag_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            } else if let Some(candidate) = state.cylinder_preview {
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
        (Some(Tool::Weld), pending) => {
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
        (Some(Tool::Bearing), _) => {
            if let Some(hit) = state.hovered
                && let Some(face) = try_face_geometry_from_ref(hit.face, Some(&graph.0))
            {
                let anchor = state.bearing_preview_anchor.unwrap_or_else(|| {
                    bearing_anchor_from_hit_with_grid(
                        &graph.0,
                        hit,
                        state.placement_grid,
                        state.placement_bounds,
                    )
                    .unwrap_or(hit.point)
                });
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
            Some(
                Tool::Controller
                | Tool::GasEngine
                | Tool::ElectricEngine
                | Tool::Servo
                | Tool::Seat
                | Tool::Input
                | Tool::DimensionLink,
            ),
            _,
        ) => {
            if let (Some(candidate), Some(appearance)) = (
                state.preview,
                selected_tool
                    .active_editor_tool()
                    .and_then(AuthoredPart::from_tool),
            ) {
                show_cuboid_preview(
                    &mut action,
                    visuals.authored_preview_mesh(appearance),
                    visuals.authored_preview_material(appearance, state.preview_error.is_some()),
                    candidate.spec,
                    0.992,
                );
            }
        }
        (Some(Tool::Transmission), _) => {
            if let Some(candidate) = state.preview {
                let kind = state.hovered.and_then(|hit| match hit.face.owner {
                    FaceOwner::Part(part) => match graph.0.part(part) {
                        Some(PartSpec::Engine(engine)) => Some(engine.kind),
                        Some(PartSpec::Transmission(_)) => graph.0.transmission_kind(part),
                        _ => None,
                    },
                    FaceOwner::Ground => None,
                });
                let appearance = match kind.unwrap_or(EngineKind::Electric) {
                    EngineKind::Gas => AuthoredPart::GasTransmission,
                    EngineKind::Electric => AuthoredPart::ElectricTransmission,
                };
                show_cuboid_preview(
                    &mut action,
                    visuals.authored_preview_mesh(appearance),
                    visuals.authored_preview_material(appearance, state.preview_error.is_some()),
                    candidate.spec,
                    0.992,
                );
            }
        }
        (Some(Tool::Hammer | Tool::Connector), _) => {}
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
    tool: impl Into<Option<Tool>>,
    bearing_dimensions: BearingDimensions,
    cylinder_dimensions: CylinderDimensions,
    selected_wires: Option<usize>,
    material: ConstructionMaterial,
) -> String {
    let Some(tool) = tool.into() else {
        return "Hand: Empty    Clear / Pipette picks the object under the reticle".to_owned();
    };
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
            "Tool: {}    Rotate action: 90°    {}    Interact opens its program",
            tool.label(),
            selected_wires.map_or_else(
                || "No block selected — click one to select it".to_owned(),
                |wires| format!(
                    "Selected block: {wires} bearing{} wired",
                    if wires == 1 { "" } else { "s" }
                )
            ),
        ),
        Tool::Block => format!(
            "Tool: Blocker Placer    Material: {}    Block size: {BLOCK_SIZE_METERS:.2} m",
            material.label(),
        ),
        Tool::Bearing => format!(
            "Tool: Bearing    Outer: {:.2} m ←/→  Inner: {:.2} m Shift+←/→",
            bearing_dimensions.outer_diameter(),
            bearing_dimensions.inner_diameter(),
        ),
        Tool::Cylinder => format!(
            "Tool: Pipe / Cylinder    Material: {}    Outer: {:.2} m ←/→  Inner: {:.2} m Shift+←/→  Length: {:.2} m ↓/↑  Sweep: {}° Shift+↓/↑    Hold primary to drag; R mode; F bend; wheel radius",
            material.label(),
            cylinder_dimensions.outer_diameter(),
            cylinder_dimensions.inner_diameter(),
            cylinder_dimensions.axial_length(),
            cylinder_dimensions.sweep_angle_degrees(),
        ),
        Tool::GasEngine | Tool::ElectricEngine | Tool::Servo | Tool::Seat | Tool::Input => {
            format!("Tool: {}    Rotate action: 90°", tool.label())
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
// Both transmission GLBs use this same six-tile atlas layout. The imported
// vertex order differs within each face, but remapping it onto
// `AUTHORED_CUBE_POSITIONS` produces the controller-style ordering below.
const TRANSMISSION_UVS: [[f32; 2]; 24] = CONTROLLER_UVS;
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
const DIMENSION_LINK_UVS: [[f32; 2]; 24] = [
    [0.0, 0.5],
    [0.0, 0.25],
    [0.25, 0.25],
    [0.25, 0.5],
    [0.25, 0.5],
    [0.25, 0.25],
    [0.5, 0.25],
    [0.5, 0.5],
    [0.0, 0.75],
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 0.75],
    [0.5, 0.75],
    [0.5, 0.5],
    [1.0, 0.5],
    [1.0, 0.75],
    [0.0, 1.0],
    [0.0, 0.75],
    [0.5, 0.75],
    [0.5, 1.0],
    [0.5, 1.0],
    [0.5, 0.75],
    [1.0, 0.75],
    [1.0, 1.0],
];

fn authored_uvs(appearance: AuthoredPart) -> [[f32; 2]; 24] {
    let assimp_uvs = match appearance {
        AuthoredPart::Controller => CONTROLLER_UVS,
        AuthoredPart::GasEngine => GAS_ENGINE_UVS,
        AuthoredPart::ElectricEngine => ELECTRIC_ENGINE_UVS,
        AuthoredPart::GasTransmission | AuthoredPart::ElectricTransmission => TRANSMISSION_UVS,
        AuthoredPart::Servo => SERVO_UVS,
        AuthoredPart::Seat => SEAT_UVS,
        AuthoredPart::Input => INPUT_UVS,
        AuthoredPart::DimensionLinkDisabled | AuthoredPart::DimensionLinkEnabled => {
            DIMENSION_LINK_UVS
        }
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

/// The region as it would look with the current cage drag applied.
///
/// A drag is not in the graph until it is released, so previewing it means
/// folding the proposed offsets — mirrors included — into a copy.
fn preview_region(
    graph: &ConstructionGraph,
    state: &EditorState,
    mirror: shape_tool::ShapeMirror,
) -> Option<(RegionId, ShapeRegion)> {
    let id = state.active_region?;
    let mut region = graph.region(id)?.clone();
    if let Some(drag) = state.vertex_drag.as_ref() {
        for (index, offset) in shape_tool::drag_edits(&region, drag, mirror) {
            // A drag that would leave the box simply does not preview; the
            // command would reject it anyway.
            let _ = region.set_offset(index, offset);
        }
    }
    Some((id, region))
}

#[cfg(test)]
fn combined_construction_mesh(graph: &ConstructionGraph) -> Mesh {
    combined_construction_mesh_filtered(graph, None, None)
}

fn combined_material_construction_mesh(
    graph: &ConstructionGraph,
    preview: Option<&(RegionId, ShapeRegion)>,
    material: ConstructionMaterial,
) -> Mesh {
    combined_construction_mesh_filtered(graph, preview, Some(material))
}

/// Builds the construction mesh, substituting `preview` for the region it names
/// so a cage drag can be seen before it is committed.
fn combined_construction_mesh_filtered(
    graph: &ConstructionGraph,
    preview: Option<&(RegionId, ShapeRegion)>,
    material: Option<ConstructionMaterial>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let pipe_texture_offsets = pipe_texture_offsets(graph);
    let welded_pipe_ends = welded_pipe_ends(graph);
    // A part inside a region hands its surface to that region, so drawing both
    // would render the same material twice.
    for (part, spec) in graph.parts().filter(|(_, spec)| {
        ordinary_material(**spec)
            .is_some_and(|part_material| material.is_none_or(|wanted| wanted == part_material))
    }) {
        if graph.region_of(part).is_some() {
            continue;
        }
        let texture_offset = pipe_texture_offsets.get(&part).copied().unwrap_or_default();
        let first_vertex = positions.len();
        if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
            let solid = graph
                .evaluated_solid(mechanic_core::SolidOwner::Part(part))
                .expect("committed feature geometry replays");
            append_evaluated_solid(
                &solid,
                BuildTransform::IDENTITY,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        } else {
            append_textured_part(
                *spec,
                spec.pose().translation(),
                spec.pose().rotation.quaternion(),
                BuildTransform::IDENTITY,
                texture_offset,
                pipe_end_faces(part, &welded_pipe_ends),
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        }
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(spec.appearance().expect("ordinary parts have appearances")),
            positions.len() - first_vertex,
        ));
    }
    for (id, region) in graph.regions() {
        if material.is_some_and(|wanted| wanted != region.material()) {
            continue;
        }
        let shown = match preview {
            Some((preview_id, previewed)) if *preview_id == id => previewed,
            _ => region,
        };
        let first_vertex = positions.len();
        if preview.is_none()
            && graph.owner_has_shape_features(mechanic_core::SolidOwner::Region(id))
        {
            let solid = graph
                .evaluated_solid(mechanic_core::SolidOwner::Region(id))
                .expect("committed region feature geometry replays");
            append_evaluated_solid(
                &solid,
                BuildTransform::IDENTITY,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        } else {
            append_region(
                shown,
                BuildTransform::IDENTITY,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        }
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(shown.appearance()),
            positions.len() - first_vertex,
        ));
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

const fn ordinary_material(spec: PartSpec) -> Option<ConstructionMaterial> {
    match spec {
        PartSpec::Cuboid(cuboid) => Some(cuboid.material),
        PartSpec::Cylinder(cylinder) => Some(cylinder.material),
        PartSpec::PipeBend(bend) => Some(bend.material),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => None,
    }
}

fn pipe_endpoint_texture_u(spec: PartSpec, face: FaceKind) -> Option<f32> {
    match (spec, face) {
        (PartSpec::Cylinder(cylinder), FaceKind::NegativeY) => {
            Some(-cylinder.dimensions.axial_length() * 0.5)
        }
        (PartSpec::Cylinder(cylinder), FaceKind::PositiveY) => {
            Some(cylinder.dimensions.axial_length() * 0.5)
        }
        (PartSpec::PipeBend(bend), FaceKind::NegativeX) => {
            Some(-std::f32::consts::FRAC_PI_2 * bend.dimensions.radius())
        }
        (PartSpec::PipeBend(_), FaceKind::PositiveY) => Some(0.0),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PipeTextureOffset {
    u: f32,
    v_angle: f32,
}

#[derive(Clone, Copy, Debug)]
struct PipeEndpointTextureFrame {
    direction: Vec3,
    radial_zero: Vec3,
}

fn pipe_endpoint_texture_frame(spec: PartSpec, face: FaceKind) -> Option<PipeEndpointTextureFrame> {
    let rotation = spec.pose().rotation.quaternion();
    match (spec, face) {
        (PartSpec::Cylinder(_), FaceKind::NegativeY | FaceKind::PositiveY) => {
            Some(PipeEndpointTextureFrame {
                direction: rotation * Vec3::Y,
                radial_zero: rotation * Vec3::X,
            })
        }
        (PartSpec::PipeBend(_), FaceKind::NegativeX) => Some(PipeEndpointTextureFrame {
            direction: rotation * Vec3::X,
            radial_zero: rotation * Vec3::NEG_Y,
        }),
        (PartSpec::PipeBend(_), FaceKind::PositiveY) => Some(PipeEndpointTextureFrame {
            direction: rotation * Vec3::Y,
            radial_zero: rotation * Vec3::X,
        }),
        _ => None,
    }
}

/// Carries both lengthwise and circumferential texture phase across welded pipe ends.
fn pipe_texture_offsets(graph: &ConstructionGraph) -> HashMap<PartId, PipeTextureOffset> {
    let mut neighbours = HashMap::<PartId, Vec<(PartId, PipeTextureOffset)>>::new();
    for (_, weld) in graph.welds() {
        let (FaceOwner::Part(first), FaceOwner::Part(second)) =
            (weld.first.owner, weld.second.owner)
        else {
            continue;
        };
        let Some(first_u) = graph
            .part(first)
            .copied()
            .and_then(|spec| pipe_endpoint_texture_u(spec, weld.first.face))
        else {
            continue;
        };
        let Some(second_u) = graph
            .part(second)
            .copied()
            .and_then(|spec| pipe_endpoint_texture_u(spec, weld.second.face))
        else {
            continue;
        };
        let first_spec = graph
            .part(first)
            .copied()
            .expect("welded part remains in graph");
        let second_spec = graph
            .part(second)
            .copied()
            .expect("welded part remains in graph");
        let first_frame = pipe_endpoint_texture_frame(first_spec, weld.first.face)
            .expect("pipe endpoint exposes a texture frame");
        let second_frame = pipe_endpoint_texture_frame(second_spec, weld.second.face)
            .expect("pipe endpoint exposes a texture frame");
        let v_angle = if first_frame.direction.dot(second_frame.direction) > 1.0 - 1.0e-4 {
            let second_angular = -second_frame.direction.cross(second_frame.radial_zero);
            -first_frame
                .radial_zero
                .dot(second_angular)
                .atan2(first_frame.radial_zero.dot(second_frame.radial_zero))
        } else {
            0.0
        };
        let second_from_first = PipeTextureOffset {
            u: first_u - second_u,
            v_angle,
        };
        neighbours
            .entry(first)
            .or_default()
            .push((second, second_from_first));
        neighbours.entry(second).or_default().push((
            first,
            PipeTextureOffset {
                u: -second_from_first.u,
                v_angle: -second_from_first.v_angle,
            },
        ));
    }

    let mut offsets = HashMap::new();
    let mut pending = VecDeque::new();
    for (root, spec) in graph.parts() {
        if !matches!(spec, PartSpec::Cylinder(_) | PartSpec::PipeBend(_))
            || offsets.contains_key(&root)
        {
            continue;
        }
        offsets.insert(root, PipeTextureOffset::default());
        pending.push_back(root);
        while let Some(part) = pending.pop_front() {
            let offset = offsets[&part];
            for &(neighbour, delta) in neighbours.get(&part).into_iter().flatten() {
                if let std::collections::hash_map::Entry::Vacant(entry) = offsets.entry(neighbour) {
                    entry.insert(PipeTextureOffset {
                        u: offset.u + delta.u,
                        v_angle: offset.v_angle + delta.v_angle,
                    });
                    pending.push_back(neighbour);
                }
            }
        }
    }
    offsets
}

#[derive(Clone, Copy, Debug)]
struct PipeEndFaces {
    inlet: bool,
    outlet: bool,
}

impl PipeEndFaces {
    const ALL: Self = Self {
        inlet: true,
        outlet: true,
    };
}

fn welded_pipe_ends(graph: &ConstructionGraph) -> HashSet<FaceRef> {
    let mut ends = HashSet::new();
    for (_, weld) in graph.welds() {
        let pipe_endpoint = |face: FaceRef| {
            let FaceOwner::Part(part) = face.owner else {
                return false;
            };
            graph
                .part(part)
                .copied()
                .is_some_and(|spec| pipe_endpoint_texture_u(spec, face.face).is_some())
        };
        if pipe_endpoint(weld.first) && pipe_endpoint(weld.second) {
            ends.insert(weld.first);
            ends.insert(weld.second);
        }
    }
    ends
}

fn pipe_end_faces(part: PartId, welded_ends: &HashSet<FaceRef>) -> PipeEndFaces {
    let hidden = |face| welded_ends.contains(&FaceRef::part(part, face));
    PipeEndFaces {
        inlet: !hidden(FaceKind::NegativeY) && !hidden(FaceKind::NegativeX),
        outlet: !hidden(FaceKind::PositiveY),
    }
}

/// Control blocks render as their own teal mesh so they stand out from the
/// construction they steer.
#[cfg(test)]
fn combined_controller_mesh(graph: &ConstructionGraph) -> Mesh {
    combined_authored_construction_mesh(graph, AuthoredPart::Controller, None)
}

fn combined_authored_construction_mesh(
    graph: &ConstructionGraph,
    appearance: AuthoredPart,
    active_dimension_link: Option<mechanic_core::DimensionLinkId>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for (_, spec) in graph
        .parts()
        .filter(|(part, spec)| appearance.matches(graph, *part, **spec, active_dimension_link))
    {
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

#[cfg(test)]
fn combined_simulation_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
) -> Mesh {
    combined_simulation_mesh_filtered(graph, creation, transforms, kind, None)
}

fn combined_simulation_material_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
    material: ConstructionMaterial,
) -> Mesh {
    combined_simulation_mesh_filtered(graph, creation, transforms, kind, Some(material))
}

fn simulation_material_is_present(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    kind: SimulationMeshKind,
    material: ConstructionMaterial,
) -> bool {
    creation.part_to_compound.iter().any(|&(part, compound)| {
        let is_static = creation.compounds[compound as usize].is_static;
        let right_motion = match kind {
            SimulationMeshKind::Static => is_static,
            SimulationMeshKind::Dynamic => !is_static,
        };
        right_motion
            && graph
                .part(part)
                .copied()
                .and_then(ordinary_material)
                .is_some_and(|candidate| candidate == material)
    })
}

#[allow(clippy::too_many_lines)]
fn combined_simulation_mesh_filtered(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    kind: SimulationMeshKind,
    material: Option<ConstructionMaterial>,
) -> Mesh {
    let pipe_texture_offsets = pipe_texture_offsets(graph);
    let welded_pipe_ends = welded_pipe_ends(graph);
    let parts = creation
        .part_to_compound
        .iter()
        .filter(|(part, compound_index)| {
            let part_material = graph.part(*part).copied().and_then(ordinary_material);
            let is_static = creation.compounds[*compound_index as usize].is_static;
            let right_motion = match kind {
                SimulationMeshKind::Static => is_static,
                SimulationMeshKind::Dynamic => !is_static,
            };
            right_motion
                && part_material.is_some_and(|part_material| {
                    material.is_none_or(|wanted| wanted == part_material)
                })
        });
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();
    let mut drawn_regions: Vec<mechanic_core::RegionId> = Vec::new();
    for &(part, compound_index) in parts {
        let transform = transforms[compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        let initial = &creation.compounds[compound_index as usize];
        let spec = *graph.part(part).expect("compiled source remains in graph");
        let placement = BuildTransform {
            origin: initial.root_translation,
            rotation: root_rotation,
            translation: root_translation,
        };
        // A part inside a region is drawn once, as its region.
        if let Some(id) = graph.region_of(part) {
            if drawn_regions.contains(&id) {
                continue;
            }
            drawn_regions.push(id);
            if let Some(region) = graph.region(id) {
                let first_vertex = positions.len();
                if graph.owner_has_shape_features(mechanic_core::SolidOwner::Region(id)) {
                    let solid = graph
                        .evaluated_solid(mechanic_core::SolidOwner::Region(id))
                        .expect("compiled region feature geometry replays");
                    append_evaluated_solid(
                        &solid,
                        placement,
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        &mut tangents,
                        &mut indices,
                    );
                } else {
                    append_region(
                        region,
                        placement,
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        &mut tangents,
                        &mut indices,
                    );
                }
                colors.extend(std::iter::repeat_n(
                    chroma::encode_appearance(region.appearance()),
                    positions.len() - first_vertex,
                ));
            }
            continue;
        }
        let local_center = spec.pose().translation() - initial.root_translation;
        let texture_offset = pipe_texture_offsets.get(&part).copied().unwrap_or_default();
        let first_vertex = positions.len();
        if graph.owner_has_shape_features(mechanic_core::SolidOwner::Part(part)) {
            let solid = graph
                .evaluated_solid(mechanic_core::SolidOwner::Part(part))
                .expect("compiled feature geometry replays");
            append_evaluated_solid(
                &solid,
                placement,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        } else {
            append_textured_part(
                spec,
                root_translation + root_rotation * local_center,
                root_rotation * spec.pose().rotation.quaternion(),
                placement,
                texture_offset,
                pipe_end_faces(part, &welded_pipe_ends),
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
        }
        colors.extend(std::iter::repeat_n(
            chroma::encode_appearance(spec.appearance().expect("ordinary parts have appearances")),
            positions.len() - first_vertex,
        ));
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, tangents)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_simulation_authored_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    appearance: AuthoredPart,
    active_dimension_link: Option<mechanic_core::DimensionLinkId>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    for &(part, compound_index) in creation.part_to_compound.iter().filter(|(part, _)| {
        graph
            .part(*part)
            .is_some_and(|spec| appearance.matches(graph, *part, *spec, active_dimension_link))
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
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
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
            &mut uvs,
            &mut tangents,
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

fn combined_simulation_bearing_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();

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
            &mut uvs,
            &mut tangents,
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

fn simulation_placed_bearing_pose(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    bearing: PlacedBearing,
) -> Option<(Vec3, Vec3)> {
    let FaceOwner::Part(source_part) = bearing.source.owner else {
        return None;
    };
    let compound_index = creation
        .part_to_compound
        .iter()
        .find_map(|&(part, index)| (part == source_part).then_some(index))?;
    let initial = creation.compounds.get(compound_index as usize)?;
    let inverse_initial_rotation = initial.root_rotation.inverse();
    let local_anchor = inverse_initial_rotation * (bearing.anchor - initial.root_translation);
    let local_axis =
        inverse_initial_rotation * face_geometry_from_ref(bearing.source, Some(graph)).normal;
    Some(transform_bearing_pose(
        *transforms.get(compound_index as usize)?,
        local_anchor,
        local_axis,
    ))
}

fn raycast_simulation_bearings(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
    placed_bearings: &[PlacedBearing],
    origin: Vec3,
    direction: Vec3,
) -> Option<(BearingDimensions, f32)> {
    let graph_bearings = graph.bearings().filter_map(|(_, bearing)| {
        let (anchor, axis) = simulation_bearing_pose(graph, creation, transforms, bearing)?;
        let distance =
            raycast_bearing_annulus(origin, direction, anchor, axis, bearing.dimensions)?;
        Some((bearing.dimensions, distance))
    });
    let placed = placed_bearings.iter().filter_map(|&bearing| {
        let (anchor, axis) = simulation_placed_bearing_pose(graph, creation, transforms, bearing)?;
        let distance =
            raycast_bearing_annulus(origin, direction, anchor, axis, bearing.dimensions)?;
        Some((bearing.dimensions, distance))
    });
    graph_bearings
        .chain(placed)
        .min_by(|left, right| left.1.total_cmp(&right.1))
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
    let mut uvs = Vec::new();
    let mut tangents = Vec::new();
    let mut indices = Vec::new();
    append_bearing_cylinder(
        Vec3::ZERO,
        Vec3::Y,
        dimensions,
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut tangents,
        &mut indices,
    );
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

const BEARING_SEGMENTS: u16 = 24;
const BEARING_ATLAS_PIXELS: f32 = 1_024.0;
const BEARING_ARC_METERS_PER_TILE: f32 = 0.05;
const BEARING_LAND_METERS: f32 = 0.008;
const BEARING_LIP_METERS: f32 = 0.006;
const BEARING_RELIEF_MIN_METERS: f32 = 0.005;
const BEARING_RELIEF_MAX_METERS: f32 = 0.040;
const BEARING_RELIEF_WALL_FRACTION: f32 = 0.10;
const BEARING_TERRACE_NOMINAL_METERS: f32 = 0.014;
const BEARING_TURN_METERS: f32 = 0.009;
const BEARING_STEP_SPLIT: f32 = 0.66;

#[derive(Clone, Copy, Debug)]
struct BearingProfilePlan {
    steps: u8,
    terrace_meters: f32,
    turns: u16,
    relief_meters: f32,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bearing_profile_plan(outer_radius: f32, inner_radius: f32) -> BearingProfilePlan {
    let wall = outer_radius - inner_radius;
    let middle = (wall - BEARING_LAND_METERS - BEARING_LIP_METERS).max(0.0005);
    let steps = (middle / BEARING_TERRACE_NOMINAL_METERS)
        .round()
        .clamp(1.0, 4.0) as u8;
    let unit = middle / f32::from(steps);
    let relief_meters = (wall * BEARING_RELIEF_WALL_FRACTION)
        .clamp(BEARING_RELIEF_MIN_METERS, BEARING_RELIEF_MAX_METERS);
    let terrace_meters = (unit - relief_meters).max(0.0015);
    let turns = (terrace_meters / BEARING_TURN_METERS).round().max(1.0) as u16;
    BearingProfilePlan {
        steps,
        terrace_meters,
        turns,
        relief_meters,
    }
}

fn bearing_band_v(start_row: f32, end_row: f32, t: f32) -> f32 {
    1.0 - (start_row + (end_row - start_row) * t) / BEARING_ATLAS_PIXELS
}

fn bearing_u_repeat(radius: f32) -> f32 {
    (std::f32::consts::TAU * radius.max(0.02) / BEARING_ARC_METERS_PER_TILE)
        .round()
        .max(1.0)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn append_bearing_cylinder(
    anchor: Vec3,
    axis: Vec3,
    dimensions: BearingDimensions,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    const LAND_TS: [f32; 5] = [0.0, 0.17, 0.34, 0.62, 1.0];
    const RELIEF_TS: [f32; 7] = [0.0, 0.12, 0.235, 0.40, 0.55, 0.75, 1.0];
    const LIP_TS: [f32; 7] = [0.0, 0.24, 0.46, 0.60, 0.74, 0.87, 1.0];

    let axis = axis.normalize();
    let radial_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    let radial_v = axis.cross(radial_u);
    let outer_radius = dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN;
    let inner_radius = if dimensions.inner_diameter() > 0.0 {
        dimensions.inner_diameter() * 0.5 + BEARING_RENDER_RADIAL_SKIN
    } else {
        0.0
    };
    let plan = bearing_profile_plan(outer_radius, inner_radius);
    let repeat = bearing_u_repeat(outer_radius);

    for (center, normal, front) in [
        (anchor + axis * BEARING_DEPTH * 0.5, axis, true),
        (anchor - axis * BEARING_DEPTH * 0.5, -axis, false),
    ] {
        let mut radius = outer_radius;
        let land = LAND_TS.map(|t| {
            (
                radius - BEARING_LAND_METERS * t,
                bearing_band_v(192.0, 320.0, t),
            )
        });
        append_bearing_face_strip(
            center, normal, front, radial_u, radial_v, repeat, &land, positions, normals, uvs,
            tangents, indices,
        );
        radius -= BEARING_LAND_METERS;

        let terrace_tile = plan.terrace_meters / f32::from(plan.turns);
        for _ in 0..plan.steps {
            for _ in 0..plan.turns {
                let terrace = [
                    (radius, bearing_band_v(320.0, 704.0, 0.0)),
                    (
                        radius - terrace_tile,
                        bearing_band_v(320.0, 704.0, BEARING_STEP_SPLIT),
                    ),
                ];
                append_bearing_face_strip(
                    center, normal, front, radial_u, radial_v, repeat, &terrace, positions,
                    normals, uvs, tangents, indices,
                );
                radius -= terrace_tile;
            }
            let relief = RELIEF_TS.map(|t| {
                (
                    radius - plan.relief_meters * t,
                    bearing_band_v(
                        320.0,
                        704.0,
                        BEARING_STEP_SPLIT + t * (1.0 - BEARING_STEP_SPLIT),
                    ),
                )
            });
            append_bearing_face_strip(
                center, normal, front, radial_u, radial_v, repeat, &relief, positions, normals,
                uvs, tangents, indices,
            );
            radius -= plan.relief_meters;
        }

        let lip_span = radius - inner_radius;
        let lip = LIP_TS.map(|t| (radius - lip_span * t, bearing_band_v(704.0, 832.0, t)));
        append_bearing_face_strip(
            center, normal, front, radial_u, radial_v, repeat, &lip, positions, normals, uvs,
            tangents, indices,
        );
    }

    let upper = anchor + axis * BEARING_DEPTH * 0.5;
    let lower = anchor - axis * BEARING_DEPTH * 0.5;
    let outer_upper = append_bearing_profile_ring(
        upper,
        outer_radius,
        radial_u,
        radial_v,
        repeat,
        bearing_band_v(0.0, 192.0, 0.0),
        BearingRingNormal::Radial(1.0),
        1.0,
        positions,
        normals,
        uvs,
        tangents,
    );
    let outer_lower = append_bearing_profile_ring(
        lower,
        outer_radius,
        radial_u,
        radial_v,
        repeat,
        bearing_band_v(0.0, 192.0, 1.0),
        BearingRingNormal::Radial(1.0),
        1.0,
        positions,
        normals,
        uvs,
        tangents,
    );
    stitch_bearing_side(outer_upper, outer_lower, true, indices);

    if inner_radius > 0.0 {
        let bore_repeat = bearing_u_repeat(inner_radius);
        let inner_upper = append_bearing_profile_ring(
            upper,
            inner_radius,
            radial_u,
            radial_v,
            bore_repeat,
            bearing_band_v(832.0, 1_024.0, 0.0),
            BearingRingNormal::Radial(-1.0),
            -1.0,
            positions,
            normals,
            uvs,
            tangents,
        );
        let inner_lower = append_bearing_profile_ring(
            lower,
            inner_radius,
            radial_u,
            radial_v,
            bore_repeat,
            bearing_band_v(832.0, 1_024.0, 1.0),
            BearingRingNormal::Radial(-1.0),
            -1.0,
            positions,
            normals,
            uvs,
            tangents,
        );
        stitch_bearing_side(inner_upper, inner_lower, false, indices);
    }
}

#[derive(Clone, Copy)]
enum BearingRingNormal {
    Face(Vec3),
    Radial(f32),
}

#[allow(clippy::too_many_arguments)]
fn append_bearing_face_strip(
    center: Vec3,
    normal: Vec3,
    front: bool,
    radial_u: Vec3,
    radial_v: Vec3,
    repeat: f32,
    rings: &[(f32, f32)],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let handedness = if front { -1.0 } else { 1.0 };
    let mut previous = None;
    for &(radius, v) in rings {
        let current = append_bearing_profile_ring(
            center,
            radius.max(0.0),
            radial_u,
            radial_v,
            repeat,
            v,
            BearingRingNormal::Face(normal),
            handedness,
            positions,
            normals,
            uvs,
            tangents,
        );
        if let Some((previous_start, previous_radius)) = previous {
            stitch_bearing_face(
                previous_start,
                current,
                front,
                radius <= f32::EPSILON && previous_radius > f32::EPSILON,
                indices,
            );
        }
        previous = Some((current, radius));
    }
}

#[allow(clippy::too_many_arguments)]
fn append_bearing_profile_ring(
    center: Vec3,
    radius: f32,
    radial_u: Vec3,
    radial_v: Vec3,
    repeat: f32,
    v: f32,
    ring_normal: BearingRingNormal,
    handedness: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) -> u32 {
    let start = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    for segment in 0..=BEARING_SEGMENTS {
        let phase = f32::from(segment) / f32::from(BEARING_SEGMENTS);
        let angle = std::f32::consts::TAU * phase;
        let radial = radial_u * angle.cos() + radial_v * angle.sin();
        let tangent = -radial_u * angle.sin() + radial_v * angle.cos();
        let normal = match ring_normal {
            BearingRingNormal::Face(normal) => normal,
            BearingRingNormal::Radial(sign) => radial * sign,
        };
        positions.push((center + radial * radius).to_array());
        normals.push(normal.to_array());
        uvs.push([phase * repeat, v]);
        tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
    start
}

fn stitch_bearing_face(
    outer: u32,
    inner: u32,
    front: bool,
    inner_is_center: bool,
    indices: &mut Vec<u32>,
) {
    for segment in 0..BEARING_SEGMENTS {
        let current = u32::from(segment);
        let next = current + 1;
        if front {
            indices.extend([outer + current, outer + next, inner + current]);
            if !inner_is_center {
                indices.extend([outer + next, inner + next, inner + current]);
            }
        } else {
            indices.extend([outer + current, inner + current, outer + next]);
            if !inner_is_center {
                indices.extend([outer + next, inner + current, inner + next]);
            }
        }
    }
}

fn stitch_bearing_side(upper: u32, lower: u32, outward: bool, indices: &mut Vec<u32>) {
    for segment in 0..BEARING_SEGMENTS {
        let current = u32::from(segment);
        let next = current + 1;
        if outward {
            indices.extend([
                upper + current,
                lower + current,
                upper + next,
                upper + next,
                lower + current,
                lower + next,
            ]);
        } else {
            indices.extend([
                upper + current,
                upper + next,
                lower + current,
                upper + next,
                lower + next,
                lower + current,
            ]);
        }
    }
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
    append_cylinder_shape_with_end_faces(
        center,
        rotation,
        dimensions,
        scale,
        PipeEndFaces::ALL,
        0.0,
        positions,
        normals,
        indices,
    );
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn append_cylinder_shape_with_end_faces(
    center: Vec3,
    rotation: Quat,
    dimensions: CylinderDimensions,
    scale: f32,
    end_faces: PipeEndFaces,
    v_angle_offset: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    if dimensions.sweep_angle_degrees() == 360 {
        append_annular_cylinder_with_end_faces(
            center,
            rotation * Vec3::Y,
            dimensions.outer_diameter() * scale,
            dimensions.inner_diameter() * scale,
            dimensions.axial_length() * scale,
            end_faces,
            true,
            Some(rotation * (Vec3::X * v_angle_offset.cos() - Vec3::Z * v_angle_offset.sin())),
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

#[allow(clippy::too_many_arguments)]
fn append_pipe_bend_shape(
    corner: Vec3,
    rotation: Quat,
    dimensions: PipeBendDimensions,
    scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    append_pipe_bend_shape_with_end_faces(
        corner,
        rotation,
        dimensions,
        scale,
        PipeEndFaces::ALL,
        0.0,
        positions,
        normals,
        indices,
    );
}

#[allow(clippy::too_many_arguments)]
fn append_pipe_bend_shape_with_end_faces(
    corner: Vec3,
    rotation: Quat,
    dimensions: PipeBendDimensions,
    scale: f32,
    end_faces: PipeEndFaces,
    v_angle_offset: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const ARC_SLICES: u16 = 12;
    const RADIAL_SIDES: u16 = 24;
    let radius = dimensions.radius() * scale;
    let outer = dimensions.outer_diameter() * scale * 0.5;
    let inner = dimensions.inner_diameter() * scale * 0.5;
    let curve_center = Vec3::new(-radius, radius, 0.0);
    let point = |theta: f32, phi: f32, tube_radius: f32| {
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        curve_center
            + radial * (radius + tube_radius * phi.cos())
            + Vec3::Z * (tube_radius * phi.sin())
    };
    let surface_normal = |theta: f32, phi: f32| {
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        rotation * (radial * phi.cos() + Vec3::Z * phi.sin())
    };
    let world = |local: Vec3| corner + rotation * local;

    for arc in 0..ARC_SLICES {
        let theta0 = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::FRAC_PI_2 * f32::from(arc) / f32::from(ARC_SLICES);
        let theta1 = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::FRAC_PI_2 * f32::from(arc + 1) / f32::from(ARC_SLICES);
        for side in 0..RADIAL_SIDES {
            let phi0 =
                std::f32::consts::TAU * f32::from(side) / f32::from(RADIAL_SIDES) - v_angle_offset;
            let phi1 = std::f32::consts::TAU * f32::from(side + 1) / f32::from(RADIAL_SIDES)
                - v_angle_offset;
            append_mesh_quad_with_normals(
                [
                    world(point(theta0, phi0, outer)),
                    world(point(theta1, phi0, outer)),
                    world(point(theta1, phi1, outer)),
                    world(point(theta0, phi1, outer)),
                ],
                [
                    surface_normal(theta0, phi0),
                    surface_normal(theta1, phi0),
                    surface_normal(theta1, phi1),
                    surface_normal(theta0, phi1),
                ],
                positions,
                normals,
                indices,
            );
            if inner > 0.0 {
                append_mesh_quad_with_normals(
                    [
                        world(point(theta0, phi1, inner)),
                        world(point(theta1, phi1, inner)),
                        world(point(theta1, phi0, inner)),
                        world(point(theta0, phi0, inner)),
                    ],
                    [
                        -surface_normal(theta0, phi1),
                        -surface_normal(theta1, phi1),
                        -surface_normal(theta1, phi0),
                        -surface_normal(theta0, phi0),
                    ],
                    positions,
                    normals,
                    indices,
                );
            }
        }
    }

    append_pipe_bend_end_faces(
        corner,
        rotation,
        dimensions,
        scale,
        end_faces,
        v_angle_offset,
        positions,
        normals,
        indices,
    );
}

#[allow(clippy::too_many_arguments)]
fn append_pipe_bend_end_faces(
    corner: Vec3,
    rotation: Quat,
    dimensions: PipeBendDimensions,
    scale: f32,
    end_faces: PipeEndFaces,
    v_angle_offset: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const RADIAL_SIDES: u16 = 24;
    let radius = dimensions.radius() * scale;
    let outer = dimensions.outer_diameter() * scale * 0.5;
    let inner = dimensions.inner_diameter() * scale * 0.5;
    let curve_center = Vec3::new(-radius, radius, 0.0);
    let point = |theta: f32, phi: f32, tube_radius: f32| {
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        curve_center
            + radial * (radius + tube_radius * phi.cos())
            + Vec3::Z * (tube_radius * phi.sin())
    };
    let world = |local: Vec3| corner + rotation * local;

    for (theta, end_normal, reverse, visible) in [
        (
            -std::f32::consts::FRAC_PI_2,
            Vec3::NEG_X,
            false,
            end_faces.inlet,
        ),
        (0.0, Vec3::Y, true, end_faces.outlet),
    ] {
        if !visible {
            continue;
        }
        let centerline = point(theta, 0.0, 0.0);
        for side in 0..RADIAL_SIDES {
            let phi0 =
                std::f32::consts::TAU * f32::from(side) / f32::from(RADIAL_SIDES) - v_angle_offset;
            let phi1 = std::f32::consts::TAU * f32::from(side + 1) / f32::from(RADIAL_SIDES)
                - v_angle_offset;
            if inner > 0.0 {
                let vertices = if reverse {
                    [
                        world(point(theta, phi1, inner)),
                        world(point(theta, phi1, outer)),
                        world(point(theta, phi0, outer)),
                        world(point(theta, phi0, inner)),
                    ]
                } else {
                    [
                        world(point(theta, phi0, inner)),
                        world(point(theta, phi0, outer)),
                        world(point(theta, phi1, outer)),
                        world(point(theta, phi1, inner)),
                    ]
                };
                append_mesh_quad(vertices, rotation * end_normal, positions, normals, indices);
            } else {
                let vertices = if reverse {
                    [
                        world(centerline),
                        world(point(theta, phi1, outer)),
                        world(point(theta, phi0, outer)),
                    ]
                } else {
                    [
                        world(centerline),
                        world(point(theta, phi0, outer)),
                        world(point(theta, phi1, outer)),
                    ]
                };
                append_mesh_triangle(vertices, rotation * end_normal, positions, normals, indices);
            }
        }
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
        if mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some() {
            degenerate_textured_mesh()
        } else {
            degenerate_overlay_mesh()
        }
    } else {
        mesh
    }
}

fn degenerate_textured_mesh() -> Mesh {
    degenerate_overlay_mesh()
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; 3])
        .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 3])
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
    _simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut drawn: Local<Option<WireEnd>>,
    mut transform: Single<&mut Transform, With<WireHoverVisual>>,
) {
    let hovered = if selection.active_editor_tool() == Some(Tool::Connector) {
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
    actions: Res<ButtonInput<GameAction>>,
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
        .is_some_and(|drag| !drag.armed && !actions.pressed(GameAction::Primary))
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

fn append_mesh_quad_with_normals(
    vertices: [Vec3; 4],
    vertex_normals: [Vec3; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(vertices.map(|vertex| vertex.to_array()));
    normals.extend(vertex_normals.map(|normal| normal.to_array()));
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn append_annular_cylinder_with_end_faces(
    anchor: Vec3,
    axis: Vec3,
    outer_diameter: f32,
    inner_diameter: f32,
    axial_length: f32,
    end_faces: PipeEndFaces,
    duplicate_side_seam: bool,
    circumference_start: Option<Vec3>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const SEGMENTS: u16 = 24;
    let axis = axis.normalize();
    let (tangent_u, tangent_v, positive_circumference) = if let Some(start) = circumference_start {
        let tangent_u = start.normalize();
        (tangent_u, -axis.cross(tangent_u), true)
    } else {
        let tangent_u = if axis.y.abs() < 0.9 {
            axis.cross(Vec3::Y).normalize()
        } else {
            axis.cross(Vec3::X).normalize()
        };
        (tangent_u, axis.cross(tangent_u), false)
    };
    let outer_radius = outer_diameter * 0.5;
    let inner_radius = inner_diameter * 0.5;
    let half_depth = axial_length * 0.5;
    let lower = anchor - axis * half_depth;
    let upper = anchor + axis * half_depth;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");

    let side_ring_vertices = if duplicate_side_seam {
        SEGMENTS + 1
    } else {
        SEGMENTS
    };
    for segment in 0..side_ring_vertices {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * outer_radius).to_array());
        positions.push((upper + radial * outer_radius).to_array());
        normals.push(radial.to_array());
        normals.push(radial.to_array());
    }
    for segment in 0..SEGMENTS {
        let next = if duplicate_side_seam {
            segment + 1
        } else {
            (segment + 1) % SEGMENTS
        };
        let lower_current = base + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = base + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        if positive_circumference {
            indices.extend([
                lower_current,
                upper_current,
                lower_next,
                upper_current,
                upper_next,
                lower_next,
            ]);
        } else {
            indices.extend([
                lower_current,
                lower_next,
                upper_current,
                upper_current,
                lower_next,
                upper_next,
            ]);
        }
    }

    if inner_radius == 0.0 {
        for (center, normal, visible, reverse) in [
            (lower, -axis, end_faces.inlet, !positive_circumference),
            (upper, axis, end_faces.outlet, positive_circumference),
        ] {
            if !visible {
                continue;
            }
            let center_index = u32::try_from(positions.len()).unwrap();
            positions.push(center.to_array());
            normals.push(normal.to_array());
            let ring = append_bearing_face_ring(
                center,
                normal,
                outer_radius,
                tangent_u,
                tangent_v,
                positions,
                normals,
            );
            for segment in 0..SEGMENTS {
                let next = (segment + 1) % SEGMENTS;
                let current = u32::from(segment);
                let next = u32::from(next);
                if reverse {
                    indices.extend([center_index, ring + next, ring + current]);
                } else {
                    indices.extend([center_index, ring + current, ring + next]);
                }
            }
        }
        return;
    }

    let inner_side = u32::try_from(positions.len()).unwrap();
    for segment in 0..side_ring_vertices {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * inner_radius).to_array());
        positions.push((upper + radial * inner_radius).to_array());
        normals.push((-radial).to_array());
        normals.push((-radial).to_array());
    }
    for segment in 0..SEGMENTS {
        let next = if duplicate_side_seam {
            segment + 1
        } else {
            (segment + 1) % SEGMENTS
        };
        let lower_current = inner_side + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = inner_side + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        if positive_circumference {
            indices.extend([
                lower_current,
                lower_next,
                upper_current,
                upper_current,
                lower_next,
                upper_next,
            ]);
        } else {
            indices.extend([
                lower_current,
                upper_current,
                lower_next,
                upper_current,
                upper_next,
                lower_next,
            ]);
        }
    }

    for (center, normal, visible, reverse) in [
        (lower, -axis, end_faces.inlet, !positive_circumference),
        (upper, axis, end_faces.outlet, positive_circumference),
    ] {
        if !visible {
            continue;
        }
        let outer = append_bearing_face_ring(
            center,
            normal,
            outer_radius,
            tangent_u,
            tangent_v,
            positions,
            normals,
        );
        let inner = append_bearing_face_ring(
            center,
            normal,
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
            if reverse {
                indices.extend([
                    outer + current,
                    inner + current,
                    outer + next,
                    outer + next,
                    inner + current,
                    inner + next,
                ]);
            } else {
                indices.extend([
                    outer + current,
                    outer + next,
                    inner + current,
                    outer + next,
                    inner + next,
                    inner + current,
                ]);
            }
        }
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
        PartSpec::Transmission(spec) => append_transformed_cuboid(
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
        PartSpec::DimensionLink(spec) => append_transformed_cuboid(
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
        PartSpec::PipeBend(spec) => append_pipe_bend_shape(
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

/// Maps build-space geometry into the space a mesh is being drawn in.
///
/// The construction view draws parts where they were authored; the simulation
/// view draws them where their compound has moved to. Shaped geometry is
/// generated in build space either way, so it needs this to follow a body.
#[derive(Clone, Copy)]
struct BuildTransform {
    /// Build-space point that maps onto `translation`.
    origin: Vec3,
    /// Rotation applied about `origin`.
    rotation: Quat,
    /// Where `origin` ends up.
    translation: Vec3,
}

impl BuildTransform {
    /// Draws build-space geometry exactly where it was authored.
    const IDENTITY: Self = Self {
        origin: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    fn point(self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (point - self.origin)
    }

    fn direction(self, direction: Vec3) -> Vec3 {
        self.rotation * direction
    }
}

/// Emits the exterior surface of a shaped region, with UVs and tangents.
///
/// The geometry comes from the same decomposition the colliders come from, so
/// what is drawn and what is collided against cannot drift apart. Faces
/// interior to the region are dropped: a cell face whose neighbour is also part
/// of the region is inside the solid, and a piece face with no grid provenance
/// is interior to its own cell.
#[allow(clippy::too_many_arguments)]
fn append_region(
    region: &ShapeRegion,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let first = positions.len();
    append_region_surface(region, placement, positions, normals, indices);
    // The same triplanar projection ordinary blocks use, so a shaped face keeps
    // the material's scale.
    for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
        let position = Vec3::from_array(position);
        let normal = Vec3::from_array(normal);
        let absolute = normal.abs();
        let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
            ([position.x, position.z], Vec3::X)
        } else if absolute.x >= absolute.z {
            ([position.z, position.y], Vec3::Z)
        } else {
            ([position.x, position.y], Vec3::X)
        };
        uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
        tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
    }
}

/// Emits an evaluated feature boundary. Tessellation seams are retained only
/// as triangle edges; surface provenance controls hard versus smooth normals,
/// while the ordinary triplanar projection preserves material scale.
#[allow(clippy::too_many_arguments)]
fn append_evaluated_solid(
    solid: &mechanic_core::EvaluatedSolid,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let projection_normals = solid
        .surfaces
        .iter()
        .fold(HashMap::new(), |mut normals, surface| {
            normals.entry(surface.key).or_insert(surface.normal);
            normals
        });
    for surface in &solid.surfaces {
        let mut loop_edges = Vec::new();
        let mut edge = surface.half_edge;
        loop {
            loop_edges.push(edge);
            edge = solid.half_edges[edge as usize].next;
            if edge == surface.half_edge {
                break;
            }
        }
        if loop_edges.len() < 3 {
            continue;
        }
        let base = u32::try_from(positions.len()).expect("construction mesh fits 32-bit indices");
        for &half_edge in &loop_edges {
            let vertex_index = solid.half_edges[half_edge as usize].origin;
            let position = solid.vertices[vertex_index as usize].position;
            let normal = if surface.smoothing_group == 0 {
                surface.normal
            } else {
                solid
                    .surfaces
                    .iter()
                    .enumerate()
                    .filter(|(_, candidate)| candidate.smoothing_group == surface.smoothing_group)
                    .filter(|(candidate_index, _)| {
                        solid.half_edges.iter().any(|edge| {
                            edge.face as usize == *candidate_index && edge.origin == vertex_index
                        })
                    })
                    .map(|(_, candidate)| candidate.normal)
                    .sum::<Vec3>()
                    .normalize_or_zero()
            };
            let world_position = placement.point(position);
            let world_normal = placement.direction(normal).normalize_or_zero();
            // A rounded patch keeps the projection of its originating face.
            // Choosing from the smoothed vertex normal would change dominant
            // axes halfway through a 90-degree fillet and rotate the material.
            let projection_normal = projection_normals
                .get(&surface.uv_provenance)
                .copied()
                .unwrap_or(surface.normal);
            let absolute = placement.direction(projection_normal).abs();
            let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
                ([world_position.x, world_position.z], Vec3::X)
            } else if absolute.x >= absolute.z {
                ([world_position.z, world_position.y], Vec3::Z)
            } else {
                ([world_position.x, world_position.y], Vec3::X)
            };
            positions.push(world_position.to_array());
            normals.push(world_normal.to_array());
            uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
            tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
        }
        for step in 1..loop_edges.len() - 1 {
            let step = u32::try_from(step).expect("surface polygons fit u32");
            indices.extend([base, base + step, base + step + 1]);
        }
    }
}

/// Emits just the positions, normals, and indices of a region's surface.
fn append_region_surface(
    region: &ShapeRegion,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let grid = region.grid();
    for piece in builder::region_pieces(region) {
        match piece {
            PartPiece::Cuboid {
                center,
                half_extents,
                cell_min,
                cell_span,
                ..
            } => {
                for (axis, sign) in (0..3).flat_map(|axis| [(axis, 1_i32), (axis, -1_i32)]) {
                    if box_face_is_interior(&grid, cell_min, cell_span, axis, sign) {
                        continue;
                    }
                    append_axis_quad(
                        center,
                        half_extents,
                        axis,
                        sign,
                        placement,
                        positions,
                        normals,
                        indices,
                    );
                }
            }
            PartPiece::Convex(convex) => {
                for face in &convex.faces {
                    let Some(gridface) = face.grid_face else {
                        continue;
                    };
                    if grid.contains(gridface.cell + face_neighbour_offset(gridface.face)) {
                        continue;
                    }
                    let base = u32::try_from(positions.len())
                        .expect("construction mesh fits 32-bit indices");
                    for &index in &face.indices {
                        positions.push(placement.point(convex.vertices[index as usize]).to_array());
                        normals.push(placement.direction(face.normal).to_array());
                    }
                    for step in 1..face.indices.len() - 1 {
                        let step = u32::try_from(step).expect("a piece face has few vertices");
                        indices.extend([base, base + step, base + step + 1]);
                    }
                }
            }
        }
    }
}

/// Whether every cell across one side of a box-cover box is still inside the
/// part, which makes that whole side interior geometry.
fn box_face_is_interior(
    grid: &CellGrid,
    cell_min: IVec3,
    cell_span: IVec3,
    axis: usize,
    sign: i32,
) -> bool {
    let mut neighbour = cell_min;
    neighbour[axis] += if sign > 0 { cell_span[axis] } else { -1 };
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    (0..cell_span[tangents[0]]).all(|first| {
        (0..cell_span[tangents[1]]).all(|second| {
            let mut cell = neighbour;
            cell[tangents[0]] += first;
            cell[tangents[1]] += second;
            grid.contains(cell)
        })
    })
}

/// Emits one axis-aligned face of a box, wound outward.
#[allow(clippy::too_many_arguments)]
fn append_axis_quad(
    center: Vec3,
    half_extents: Vec3,
    axis: usize,
    sign: i32,
    placement: BuildTransform,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let mut normal = Vec3::ZERO;
    normal[axis] = if sign > 0 { 1.0 } else { -1.0 };
    let tangents = [(axis + 1) % 3, (axis + 2) % 3];
    let mut first = Vec3::ZERO;
    first[tangents[0]] = half_extents[tangents[0]];
    let mut second = Vec3::ZERO;
    second[tangents[1]] = half_extents[tangents[1]];
    // Flip the winding on negative faces so every quad faces outward.
    if sign < 0 {
        core::mem::swap(&mut first, &mut second);
    }
    let anchor = center + normal * half_extents[axis];
    let corners = [
        anchor - first - second,
        anchor + first - second,
        anchor + first + second,
        anchor - first + second,
    ];
    let base = u32::try_from(positions.len()).expect("construction mesh fits 32-bit indices");
    let world_normal = placement.direction(normal).to_array();
    for corner in corners {
        positions.push(placement.point(corner).to_array());
        normals.push(world_normal);
    }
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

const MATERIAL_TEXTURE_PIXELS_PER_SIDE: f32 = 3_072.0;
const MATERIAL_TEXTURE_PIXELS_PER_BLOCK: f32 = 512.0;
const MATERIAL_TEXTURE_METERS_PER_REPEAT: f32 =
    GRID_UNIT_METERS * MATERIAL_TEXTURE_PIXELS_PER_SIDE / MATERIAL_TEXTURE_PIXELS_PER_BLOCK;

#[allow(clippy::too_many_arguments)]
fn append_textured_part(
    spec: PartSpec,
    translation: Vec3,
    rotation: Quat,
    placement: BuildTransform,
    texture_offset: PipeTextureOffset,
    end_faces: PipeEndFaces,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let _ = placement;
    let first = positions.len();
    match spec {
        PartSpec::Cuboid(cuboid) => append_transformed_cuboid(
            translation,
            rotation,
            cuboid.size_meters(),
            positions,
            normals,
            indices,
        ),
        PartSpec::Cylinder(cylinder) => append_cylinder_shape_with_end_faces(
            translation,
            rotation,
            cylinder.dimensions,
            1.0,
            end_faces,
            texture_offset.v_angle,
            positions,
            normals,
            indices,
        ),
        PartSpec::PipeBend(bend) => append_pipe_bend_shape_with_end_faces(
            translation,
            rotation,
            bend.dimensions,
            1.0,
            end_faces,
            texture_offset.v_angle,
            positions,
            normals,
            indices,
        ),
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => {
            unreachable!("authored parts render in their own texture batches")
        }
    }

    match spec {
        PartSpec::Cuboid(_) => {
            for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
                let position = Vec3::from_array(position);
                let normal = Vec3::from_array(normal);
                let absolute = normal.abs();
                let (uv, tangent) = if absolute.y >= absolute.x && absolute.y >= absolute.z {
                    ([position.x, position.z], Vec3::X)
                } else if absolute.x >= absolute.z {
                    ([position.z, position.y], Vec3::Z)
                } else {
                    ([position.x, position.y], Vec3::X)
                };
                uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
                tangents.push([tangent.x, tangent.y, tangent.z, 1.0]);
            }
        }
        PartSpec::Cylinder(_) => {
            append_cylinder_texture_coordinates(
                translation,
                rotation,
                first,
                positions,
                normals,
                texture_offset.v_angle,
                uvs,
                tangents,
            );
        }
        PartSpec::PipeBend(bend) => {
            append_pipe_bend_texture_coordinates(
                bend.dimensions,
                translation,
                rotation,
                first,
                positions,
                normals,
                texture_offset.v_angle,
                uvs,
                tangents,
            );
        }
        PartSpec::Controller(_)
        | PartSpec::Engine(_)
        | PartSpec::Transmission(_)
        | PartSpec::Servo(_)
        | PartSpec::Seat(_)
        | PartSpec::Input(_)
        | PartSpec::DimensionLink(_) => unreachable!(),
    }
    for uv in &mut uvs[first..] {
        uv[0] += texture_offset.u / MATERIAL_TEXTURE_METERS_PER_REPEAT;
    }
}

#[allow(clippy::too_many_arguments)]
fn append_cylinder_texture_coordinates(
    translation: Vec3,
    rotation: Quat,
    first: usize,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    v_angle_offset: f32,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) {
    let inverse = rotation.inverse();
    // Full pipe rings start at the transported UV cut. Seeding the unwrap at
    // that exact angle disambiguates atan2's equivalent -PI/+PI result.
    let mut outer_angle = Some(-v_angle_offset);
    let mut inner_angle = Some(-v_angle_offset);
    for (&position, &normal) in positions[first..].iter().zip(&normals[first..]) {
        let local = inverse * (Vec3::from_array(position) - translation);
        let local_normal = inverse * Vec3::from_array(normal);
        let radial = Vec3::new(local.x, 0.0, local.z);
        let radial_direction = radial.normalize_or_zero();
        let (uv, local_tangent, local_bitangent) = if local_normal.y.abs() > 0.9 {
            ([local.x, local.z], Vec3::X, Vec3::Z)
        } else if radial_direction != Vec3::ZERO && local_normal.dot(radial_direction).abs() > 0.5 {
            let mut angle = local.z.atan2(local.x);
            let previous = if local_normal.dot(radial_direction) > 0.0 {
                &mut outer_angle
            } else {
                &mut inner_angle
            };
            if let Some(previous) = *previous {
                while angle - previous > std::f32::consts::PI {
                    angle -= std::f32::consts::TAU;
                }
                while angle - previous < -std::f32::consts::PI {
                    angle += std::f32::consts::TAU;
                }
            }
            *previous = Some(angle);
            (
                [local.y, (angle + v_angle_offset) * radial.length()],
                Vec3::Y,
                Vec3::new(-angle.sin(), 0.0, angle.cos()),
            )
        } else {
            ([local.y, radial.length()], Vec3::Y, radial_direction)
        };
        uvs.push(uv.map(|value| value / MATERIAL_TEXTURE_METERS_PER_REPEAT));
        let tangent = rotation * local_tangent;
        let bitangent = rotation * local_bitangent;
        let normal = Vec3::from_array(normal);
        let handedness = if tangent.cross(normal).dot(bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_pipe_bend_texture_coordinates(
    dimensions: PipeBendDimensions,
    translation: Vec3,
    rotation: Quat,
    first: usize,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    v_angle_offset: f32,
    uvs: &mut Vec<[f32; 2]>,
    tangents: &mut Vec<[f32; 4]>,
) {
    const ARC_SLICES: usize = 12;
    const RADIAL_SIDES: usize = 24;
    const CURVED_VERTICES_PER_SECTOR: usize = 8;
    let inverse = rotation.inverse();
    let radius = dimensions.radius();
    for (vertex, (&position, &normal)) in
        positions[first..].iter().zip(&normals[first..]).enumerate()
    {
        let local = inverse * (Vec3::from_array(position) - translation);
        let from_curve_center = Vec2::new(local.x + radius, local.y - radius);
        let theta = from_curve_center.y.atan2(from_curve_center.x);
        let cross_x = from_curve_center.length() - radius;
        let phi = if vertex < ARC_SLICES * RADIAL_SIDES * CURVED_VERTICES_PER_SECTOR {
            let side = (vertex / CURVED_VERTICES_PER_SECTOR) % RADIAL_SIDES;
            let within_sector = vertex % CURVED_VERTICES_PER_SECTOR;
            let boundary = if matches!(within_sector, 0 | 1 | 6 | 7) {
                side
            } else {
                side + 1
            };
            std::f32::consts::TAU * f32::from(u16::try_from(boundary).unwrap())
                / f32::from(u16::try_from(RADIAL_SIDES).unwrap())
                - v_angle_offset
        } else {
            local.z.atan2(cross_x)
        };
        let surface_radius = cross_x.hypot(local.z);
        uvs.push([
            theta * radius / MATERIAL_TEXTURE_METERS_PER_REPEAT,
            (phi + v_angle_offset) * surface_radius / MATERIAL_TEXTURE_METERS_PER_REPEAT,
        ]);
        let tangent = rotation * Vec3::new(-theta.sin(), theta.cos(), 0.0);
        let radial = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let bitangent = rotation * (-radial * phi.sin() + Vec3::Z * phi.cos());
        let normal = Vec3::from_array(normal);
        let handedness = if tangent.cross(normal).dot(bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
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
        image::{ImageAddressMode, ImageFilterMode, ImageLoaderSettings},
        mesh::VertexAttributeValues,
        prelude::{
            AlphaMode, App, Color, EnvironmentMapLight, GeneratedEnvironmentMapLight, Handle,
            IVec3, Image, Mesh, Quat, StandardMaterial, Update, Vec2, Vec3,
        },
    };
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
        DimensionLinkId, DimensionLinkSpec, DriveLimits, DriveLinkSpec, DriveProgram, DriveState,
        DriveTarget, EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind, FaceOwner,
        FaceRef, GridRotation, InputSpec, MaterialAppearance, MaterialColor, MaterialDye,
        MaterialFinish, MaterialShift, PartSpec, PipeBendDimensions, SeatSpec, ServoSpec,
        ShapeFeature, SolidOwner,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        AuthoredPart, BEARING_DEPTH, BEARING_RENDER_RADIAL_SKIN, BLOCK_SHEET_PREVIEW_INSET_METERS,
        MATERIAL_TEXTURE_METERS_PER_REPEAT, MATERIAL_TEXTURE_PIXELS_PER_BLOCK,
        MATERIAL_TEXTURE_PIXELS_PER_SIDE, PlacedBearing, SimulationMeshKind,
        append_bearing_cylinder, append_cylinder_shape, append_pipe_bend_shape,
        append_pipe_bend_texture_coordinates, authored_preview_material, authored_uvs,
        bearing_pbr_material, bearing_preview_dimensions_changed, bearing_profile_plan,
        bearing_u_repeat, block_sheet_bounds, block_sheet_preview_mesh,
        combined_authored_construction_mesh, combined_bearing_mesh, combined_controller_mesh,
        combined_drive_xray_mesh, combined_material_construction_mesh,
        combined_simulation_bearing_mesh, combined_simulation_material_mesh,
        combined_simulation_mesh, configure_authored_texture, configure_bearing_texture,
        configure_repeating_texture, construction_tint_mask_path, drive_xray_is_visible,
        joint_xray_is_visible, preview_material, renderable_mesh, simulation_material_is_present,
        single_authored_part_mesh, single_bearing_mesh, single_cylinder_mesh,
    };
    use super::{
        EnvironmentMapGenerationReady, OverlayGeometry, append_axis_arrows, append_drag_plane,
        append_feature_pull_arrow, append_plane_arrows, region_focus_is_active,
        region_world_bounds, retain_generated_environment_map, sky_cubemap,
    };
    use crate::PlacementPlane;
    use crate::builder::block_sheet_specs;
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
    fn only_vertex_mode_uses_region_focus_ghosting() {
        assert!(region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Vertex,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Chamfer,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Fillet,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Chroma),
            crate::shape_tool::ShapeEditMode::Vertex,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Vertex,
            false,
        ));
    }

    #[test]
    fn only_the_active_dimension_link_uses_the_enabled_batch() {
        let mut graph = ConstructionGraph::new();
        let spec = DimensionLinkSpec::new(DimensionLinkId(7), BuildPose::default());
        let BuildOutcome::Spawned(part) =
            graph.apply(BuildCommand::SpawnDimensionLink(spec)).unwrap()
        else {
            unreachable!()
        };
        let part_spec = *graph.part(part).unwrap();
        assert!(AuthoredPart::DimensionLinkDisabled.matches(&graph, part, part_spec, None));
        assert!(!AuthoredPart::DimensionLinkEnabled.matches(&graph, part, part_spec, None));
        assert!(AuthoredPart::DimensionLinkEnabled.matches(
            &graph,
            part,
            part_spec,
            Some(DimensionLinkId(7))
        ));
        assert!(!AuthoredPart::DimensionLinkDisabled.matches(
            &graph,
            part,
            part_spec,
            Some(DimensionLinkId(7))
        ));
    }

    /// A region offset from the origin, so a centred overlay cannot pass by
    /// sitting at zero.
    fn offset_region() -> mechanic_core::ShapeRegion {
        mechanic_core::ShapeRegion::new(
            IVec3::new(2, 4, 0),
            IVec3::new(3, 2, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap()
    }

    fn overlay_bounds(geometry: &OverlayGeometry) -> (Vec3, Vec3) {
        geometry.positions.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(low, high), position| {
                let at = Vec3::from_array(*position);
                (low.min(at), high.max(at))
            },
        )
    }

    #[test]
    fn the_drag_plane_sits_in_the_middle_of_the_area() {
        let region = offset_region();
        // The area runs y = 0.50 m to 1.00 m, so its middle is 0.75 m.
        for (plane, normal_axis, middle) in [
            (PlacementPlane::Xz, 1, 0.75),
            (PlacementPlane::Xy, 2, 0.125),
            (PlacementPlane::Yz, 0, 0.625),
        ] {
            let (low, high) = region_world_bounds(&region);
            let mut geometry = OverlayGeometry::default();
            append_drag_plane(low, high, plane, &mut geometry);
            let (low, high) = overlay_bounds(&geometry);
            assert!(
                (f32::midpoint(low[normal_axis], high[normal_axis]) - middle).abs() < 1.0e-5,
                "{plane:?} sheet is centred on the area"
            );
            assert!(
                high[normal_axis] - low[normal_axis] < 0.01,
                "{plane:?} sheet is a sheet, not a slab"
            );
        }
    }

    #[test]
    fn the_drag_plane_points_along_both_of_its_axes() {
        let region = offset_region();
        for plane in [PlacementPlane::Xy, PlacementPlane::Xz, PlacementPlane::Yz] {
            let (low, high) = region_world_bounds(&region);
            let mut geometry = OverlayGeometry::default();
            append_plane_arrows(low, high, plane, &mut geometry);
            let (low, high) = overlay_bounds(&geometry);
            let centre = (
                f32::midpoint(0.25, 1.0),
                f32::midpoint(0.5, 1.0),
                f32::midpoint(0.0, 0.25),
            );
            let centre = Vec3::new(centre.0, centre.1, centre.2);
            for axis in plane.tangent_axes() {
                assert!(
                    low[axis] < centre[axis] - 0.1 && high[axis] > centre[axis] + 0.1,
                    "{plane:?} reaches out along axis {axis} in both directions"
                );
            }
            let normal_axis = plane.normal_axis();
            assert!(
                high[normal_axis] - low[normal_axis] < 0.02,
                "{plane:?} arrows lie flat on the plane"
            );
            assert!(
                (f32::midpoint(low[normal_axis], high[normal_axis]) - centre[normal_axis]).abs()
                    < 1.0e-5,
                "{plane:?} arrows are centred on the area with the sheet"
            );
        }
    }

    #[test]
    fn the_vertex_axis_guide_points_only_along_its_active_axis() {
        let at = Vec3::new(0.25, 0.5, 0.75);
        for axis in 0..3 {
            let mut geometry = OverlayGeometry::default();
            append_axis_arrows(at, axis, &mut geometry);
            let (low, high) = overlay_bounds(&geometry);
            assert!(low[axis] < at[axis] - 0.17);
            assert!(high[axis] > at[axis] + 0.17);
            for other in [0, 1, 2].into_iter().filter(|&other| other != axis) {
                assert!(
                    low[other] > at[other] - 0.03 && high[other] < at[other] + 0.03,
                    "axis {axis} guide must stay narrow on axis {other}"
                );
            }
        }
    }

    #[test]
    fn the_feature_guide_points_inward_along_the_cross_section_bisector() {
        let at = Vec3::new(0.25, 0.5, 0.75);
        let direction = Vec3::new(-1.0, -1.0, 0.0).normalize();
        let mut geometry = OverlayGeometry::default();
        append_feature_pull_arrow(at, direction, &mut geometry);
        let (minimum, maximum) = geometry.positions.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), position| {
                let along = (Vec3::from_array(*position) - at).dot(direction);
                (minimum.min(along), maximum.max(along))
            },
        );

        assert!(minimum < -0.15, "the shaft stays visible outside the edge");
        assert!(maximum > 0.014, "the head points inward through the edge");
    }

    #[test]
    fn the_source_sky_map_is_a_cube_of_the_requested_size() {
        let map = sky_cubemap(8);
        assert_eq!(map.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(map.texture_descriptor.size.width, 8);
        assert_eq!(map.texture_descriptor.mip_level_count, 1);
    }

    #[test]
    fn completed_generation_keeps_the_filtered_map_and_removes_its_generator() {
        let ready = EnvironmentMapGenerationReady::default();
        ready.0.store(true, std::sync::atomic::Ordering::Release);
        let mut app = App::new();
        app.insert_resource(ready)
            .add_systems(Update, retain_generated_environment_map);
        let entity = app
            .world_mut()
            .spawn((
                GeneratedEnvironmentMapLight::default(),
                EnvironmentMapLight::default(),
            ))
            .id();

        app.update();

        let entity = app.world().entity(entity);
        assert!(!entity.contains::<GeneratedEnvironmentMapLight>());
        assert!(entity.contains::<EnvironmentMapLight>());
    }

    #[test]
    fn ordinary_construction_uses_one_textured_batch_per_material() {
        let mut graph = ConstructionGraph::new();
        for (index, material) in ConstructionMaterial::ALL.into_iter().enumerate() {
            let x = i32::try_from(index).unwrap() * 12;
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap()
                    .with_material(material)
                    .with_appearance(MaterialAppearance::new(
                        MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.2).unwrap()),
                        MaterialFinish::Painted,
                    )),
                ))
                .unwrap();
            graph
                .apply(BuildCommand::SpawnCylinder(
                    CylinderSpec::new(
                        CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
                        BuildPose::new(IVec3::new(x, 2, 8), GridRotation::default()),
                    )
                    .with_material(material)
                    .with_appearance(MaterialAppearance::new(
                        MaterialColor::Shift(MaterialShift::new(45.0, 1.3, 0.9).unwrap()),
                        MaterialFinish::Anodised,
                    )),
                ))
                .unwrap();
        }

        let creation = graph.compile().unwrap();
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
        for material in ConstructionMaterial::ALL {
            let build = combined_material_construction_mesh(&graph, None, material);
            let simulated = combined_simulation_material_mesh(
                &graph,
                &creation,
                &transforms,
                SimulationMeshKind::Dynamic,
                material,
            );
            assert!(build.count_vertices() > 24);
            assert_eq!(simulated.count_vertices(), build.count_vertices());
            assert_eq!(
                build.attribute(Mesh::ATTRIBUTE_COLOR).unwrap().len(),
                build.count_vertices(),
            );
            assert_eq!(
                simulated.attribute(Mesh::ATTRIBUTE_COLOR),
                build.attribute(Mesh::ATTRIBUTE_COLOR),
                "editor and simulation encode identical appearance payloads",
            );
            assert_eq!(
                build.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().len(),
                build.count_vertices(),
            );
            assert_eq!(
                build.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap().len(),
                build.count_vertices(),
            );
        }
    }

    #[test]
    fn cuboid_uvs_give_each_quarter_metre_block_512_texture_pixels() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(40, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let mesh = combined_material_construction_mesh(&graph, None, ConstructionMaterial::Steel);
        let positions = positions(&mesh);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("material UVs use Float32x2")
        };
        assert!((uvs[1][0] - positions[1].x / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-6);
        assert!((uvs[0][0] - positions[0].x / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-6);
        let expected_span =
            4.0 * MATERIAL_TEXTURE_PIXELS_PER_BLOCK / MATERIAL_TEXTURE_PIXELS_PER_SIDE;
        assert!(((uvs[1][0] - uvs[0][0]).abs() - expected_span).abs() < 1.0e-6);
    }

    #[test]
    fn material_maps_use_repeat_sampling_and_explicit_color_spaces() {
        let mut base_color = ImageLoaderSettings::default();
        configure_repeating_texture(&mut base_color, true);
        assert!(base_color.is_srgb);
        let base_sampler = base_color.sampler.get_or_init_descriptor();
        assert_eq!(base_sampler.address_mode_u, ImageAddressMode::Repeat);
        assert_eq!(base_sampler.address_mode_v, ImageAddressMode::Repeat);

        let mut data_map = ImageLoaderSettings::default();
        configure_repeating_texture(&mut data_map, false);
        assert!(!data_map.is_srgb);
        let data_sampler = data_map.sampler.get_or_init_descriptor();
        assert_eq!(data_sampler.address_mode_u, ImageAddressMode::Repeat);
        assert_eq!(data_sampler.address_mode_v, ImageAddressMode::Repeat);
    }

    #[test]
    fn only_copper_and_dirt_route_material_tint_masks() {
        for material in ConstructionMaterial::ALL {
            let path = construction_tint_mask_path(material);
            match material {
                ConstructionMaterial::Copper => {
                    assert_eq!(path, Some("materials/copper/copper_tint.png"));
                }
                ConstructionMaterial::Dirt => {
                    assert_eq!(path, Some("materials/dirt/dirt_tint.png"));
                }
                _ => assert_eq!(path, None),
            }
        }
    }

    #[test]
    fn authored_maps_clamp_and_use_their_declared_color_spaces() {
        let mut color_map = ImageLoaderSettings::default();
        configure_authored_texture(&mut color_map, true);
        assert!(color_map.is_srgb);
        let color_sampler = color_map.sampler.get_or_init_descriptor();
        assert_eq!(color_sampler.address_mode_u, ImageAddressMode::ClampToEdge);
        assert_eq!(color_sampler.address_mode_v, ImageAddressMode::ClampToEdge);
        assert_eq!(color_sampler.mag_filter, ImageFilterMode::Linear);
        assert_eq!(color_sampler.min_filter, ImageFilterMode::Linear);
        assert_eq!(color_sampler.mipmap_filter, ImageFilterMode::Linear);

        let mut data_map = ImageLoaderSettings::default();
        configure_authored_texture(&mut data_map, false);
        assert!(!data_map.is_srgb);
        let data_sampler = data_map.sampler.get_or_init_descriptor();
        assert_eq!(data_sampler.address_mode_u, ImageAddressMode::ClampToEdge);
        assert_eq!(data_sampler.address_mode_v, ImageAddressMode::ClampToEdge);
    }

    #[test]
    fn bearing_mesh_insets_radial_surfaces_without_changing_depth() {
        let anchor = Vec3::new(2.0, 3.0, 4.0);
        let axis = Vec3::X;
        let dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            anchor,
            axis,
            dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
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
        let material = bearing_pbr_material(
            Handle::<Image>::default(),
            Handle::<Image>::default(),
            Handle::<Image>::default(),
        );
        assert!(material.depth_bias > 0.0);
    }

    #[test]
    fn bearing_maps_repeat_around_the_ring_and_clamp_across_the_profile() {
        let mut settings = ImageLoaderSettings::default();
        configure_bearing_texture(&mut settings, false);
        assert!(!settings.is_srgb);
        let sampler = settings.sampler.get_or_init_descriptor();
        assert_eq!(sampler.address_mode_u, ImageAddressMode::Repeat);
        assert_eq!(sampler.address_mode_v, ImageAddressMode::ClampToEdge);
        assert_eq!(sampler.mag_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.min_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.anisotropy_clamp, 8);
    }

    #[test]
    fn bearing_profile_fit_rule_matches_the_authored_sanity_sizes() {
        let minimum_wall = bearing_profile_plan(0.050, 0.025);
        assert_eq!(minimum_wall.steps, 1);
        assert!((minimum_wall.terrace_meters - 0.006).abs() < 1.0e-6);
        assert!((minimum_wall.relief_meters - 0.005).abs() < 1.0e-6);
        assert_eq!(minimum_wall.turns, 1);

        let common_ring = bearing_profile_plan(0.120, 0.050);
        assert_eq!(common_ring.steps, 4);
        assert!((common_ring.terrace_meters - 0.007).abs() < 1.0e-6);
        assert!((common_ring.relief_meters - 0.007).abs() < 1.0e-6);
        assert_eq!(common_ring.turns, 1);

        let wide_solid = bearing_profile_plan(0.200, 0.0);
        assert_eq!(wide_solid.steps, 4);
        assert!((wide_solid.terrace_meters - 0.0265).abs() < 1.0e-6);
        assert!((wide_solid.relief_meters - 0.020).abs() < 1.0e-6);
        assert_eq!(wide_solid.turns, 3);
    }

    #[test]
    fn bearing_mesh_carries_profile_uvs_and_normal_map_tangents() {
        let dimensions = BearingDimensions::default();
        let mesh = single_bearing_mesh(dimensions);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("bearing UVs use Float32x2")
        };
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("bearing tangents use Float32x4")
        };
        assert_eq!(uvs.len(), mesh.count_vertices());
        assert_eq!(tangents.len(), mesh.count_vertices());

        let outer_radius = dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN;
        let repeat = bearing_u_repeat(outer_radius);
        let seam = usize::from(super::BEARING_SEGMENTS);
        assert!(uvs[0][0].abs() < f32::EPSILON);
        assert!((uvs[0][1] - 0.8125).abs() < f32::EPSILON);
        assert!((uvs[seam][0] - repeat).abs() < f32::EPSILON);
        assert!((uvs[0][1] - uvs[seam][1]).abs() < f32::EPSILON);
        assert!(uvs.iter().any(|uv| uv[1].abs() < f32::EPSILON));
        assert!(uvs.iter().any(|uv| (uv[1] - 1.0).abs() < f32::EPSILON));
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
    fn pipe_bend_mesh_has_exact_bounds_bore_and_outward_winding() {
        let dimensions = PipeBendDimensions::new(0.25, 0.10, 0.50).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_pipe_bend_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        let minimum = positions
            .iter()
            .map(|position| Vec3::from_array(*position))
            .fold(Vec3::splat(f32::INFINITY), Vec3::min);
        let maximum = positions
            .iter()
            .map(|position| Vec3::from_array(*position))
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
        assert!(minimum.abs_diff_eq(Vec3::new(-0.5, -0.125, -0.125), 1.0e-5));
        assert!(maximum.abs_diff_eq(Vec3::new(0.125, 0.5, 0.125), 1.0e-5));
        assert!(positions.iter().any(|position| {
            let point = Vec3::from_array(*position);
            let from_curve_center = Vec2::new(point.x + 0.5, point.y - 0.5).length();
            ((from_curve_center - 0.5).hypot(point.z) - 0.05).abs() < 1.0e-5
        }));
        for (triangle_index, triangle) in indices.chunks_exact(3).enumerate() {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric = (b - a).cross(c - a);
            let expected = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(
                geometric.dot(expected) > 0.0,
                "triangle {triangle_index} has reversed winding"
            );
        }
    }

    #[test]
    fn pipe_bend_curved_vertices_share_analytic_smooth_normals() {
        let dimensions = PipeBendDimensions::new(0.25, 0.10, 0.50).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_pipe_bend_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );

        let theta = -std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_2 / 12.0;
        let expected_normal = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let expected_position = Vec3::new(-0.5, 0.5, 0.0) + expected_normal * (0.5 + 0.125);
        let seam_normals = positions
            .iter()
            .zip(&normals)
            .filter_map(|(&position, &normal)| {
                Vec3::from_array(position)
                    .abs_diff_eq(expected_position, 1.0e-5)
                    .then_some(Vec3::from_array(normal))
            })
            .collect::<Vec<_>>();

        assert!(seam_normals.len() >= 4);
        assert!(
            seam_normals
                .iter()
                .all(|normal| normal.abs_diff_eq(expected_normal, 1.0e-5))
        );
    }

    #[test]
    fn pipe_uvs_keep_texture_u_lengthwise_through_straights_and_bends() {
        let cylinder_dimensions = CylinderDimensions::new(0.25, 0.10, 1.0).unwrap();
        let mut straight_positions = Vec::new();
        let mut straight_normals = Vec::new();
        let mut straight_uvs = Vec::new();
        let mut straight_tangents = Vec::new();
        let mut straight_indices = Vec::new();
        super::append_textured_part(
            PartSpec::Cylinder(CylinderSpec::new(cylinder_dimensions, BuildPose::default())),
            Vec3::ZERO,
            Quat::IDENTITY,
            super::BuildTransform::IDENTITY,
            super::PipeTextureOffset::default(),
            super::PipeEndFaces::ALL,
            &mut straight_positions,
            &mut straight_normals,
            &mut straight_uvs,
            &mut straight_tangents,
            &mut straight_indices,
        );
        let straight_length =
            cylinder_dimensions.axial_length() / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        assert!((straight_uvs[1][0] - straight_uvs[0][0] - straight_length).abs() < 1.0e-6);
        assert!((straight_uvs[1][1] - straight_uvs[0][1]).abs() < 1.0e-6);
        assert!(
            Vec3::from_array(straight_tangents[0][..3].try_into().unwrap())
                .abs_diff_eq(Vec3::Y, 1.0e-6)
        );
        let straight_circumference_step =
            std::f32::consts::TAU / 24.0 * cylinder_dimensions.outer_diameter() * 0.5
                / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        for segment in 0..24 {
            let current = segment * 2;
            let next = (segment + 1) * 2;
            assert!(
                ((straight_uvs[next][1] - straight_uvs[current][1]).abs()
                    - straight_circumference_step)
                    .abs()
                    < 1.0e-6
            );
        }

        let dimensions = PipeBendDimensions::new(0.25, 0.10, 0.50).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_pipe_bend_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        append_pipe_bend_texture_coordinates(
            dimensions,
            Vec3::ZERO,
            Quat::IDENTITY,
            0,
            &positions,
            &normals,
            0.0,
            &mut uvs,
            &mut tangents,
        );

        let arc_step = std::f32::consts::FRAC_PI_2 / 12.0 * dimensions.radius()
            / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        let circumference_step = std::f32::consts::TAU / 24.0 * dimensions.outer_diameter() * 0.5
            / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        assert!((uvs[1][0] - uvs[0][0] - arc_step).abs() < 1.0e-6);
        assert!((uvs[1][1] - uvs[0][1]).abs() < 1.0e-6);
        assert!((uvs[2][0] - uvs[1][0]).abs() < 1.0e-6);
        assert!((uvs[2][1] - uvs[1][1] - circumference_step).abs() < 1.0e-6);

        let tangent = Vec3::from_array(tangents[0][..3].try_into().unwrap());
        assert!(tangent.abs_diff_eq(Vec3::X, 1.0e-6));

        let curved_vertex_count = 12 * 24 * 8;
        let inner_circumference_step =
            std::f32::consts::TAU / 24.0 * dimensions.inner_diameter() * 0.5
                / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        for (quad_uvs, quad_tangents) in uvs[..curved_vertex_count]
            .chunks_exact(8)
            .zip(tangents[..curved_vertex_count].chunks_exact(8))
        {
            assert!((quad_uvs[2][1] - quad_uvs[0][1] - circumference_step).abs() < 1.0e-6);
            assert!((quad_uvs[4][1] - quad_uvs[6][1] - inner_circumference_step).abs() < 1.0e-6);
            assert!(
                quad_tangents[..4]
                    .iter()
                    .all(|tangent| (tangent[3] + 1.0).abs() < f32::EPSILON)
            );
            assert!(
                quad_tangents[4..]
                    .iter()
                    .all(|tangent| (tangent[3] - 1.0).abs() < f32::EPSILON)
            );
        }
    }

    fn assert_pipe_circumference_phase(
        graph: &ConstructionGraph,
        part: mechanic_core::PartId,
        spec: PartSpec,
        offset: super::PipeTextureOffset,
    ) {
        let face = FaceKind::PositiveY;
        let frame = super::pipe_endpoint_texture_frame(spec, face).unwrap();
        let local_angle = std::f32::consts::FRAC_PI_2 - offset.v_angle;
        let radial = frame.radial_zero * local_angle.cos()
            - frame.direction.cross(frame.radial_zero) * local_angle.sin();
        let endpoint =
            crate::builder::face_geometry_from_ref(FaceRef::part(part, face), Some(graph));
        let target = endpoint.center + radial * 0.125;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        super::append_textured_part(
            spec,
            spec.pose().translation(),
            spec.pose().rotation.quaternion(),
            super::BuildTransform::IDENTITY,
            offset,
            super::PipeEndFaces {
                inlet: false,
                outlet: false,
            },
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
        let matching_v = positions
            .iter()
            .zip(&uvs)
            .filter_map(|(&position, uv)| {
                Vec3::from_array(position)
                    .abs_diff_eq(target, 1.0e-5)
                    .then_some(uv[1])
            })
            .collect::<Vec<_>>();
        let expected_v =
            std::f32::consts::FRAC_PI_2 * 0.125 / super::MATERIAL_TEXTURE_METERS_PER_REPEAT;
        assert!(!matching_v.is_empty());
        assert!(
            matching_v.iter().all(|&v| (v - expected_v).abs() < 1.0e-5),
            "part {part:?} circumference phase {matching_v:?} did not match {expected_v}"
        );
    }

    #[test]
    fn welded_pipe_pieces_share_texture_phase_and_hide_internal_caps() {
        let pieces = crate::builder::pipe_run_pieces(
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            ],
            &[0.25],
            CylinderDimensions::new(0.25, 0.10, 1.0).unwrap(),
            ConstructionMaterial::Wood,
        )
        .unwrap();
        let graph = crate::builder::stage_pipe_run(
            &ConstructionGraph::new(),
            &pieces,
            crate::builder::PipeRunAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
        )
        .unwrap();
        let offsets = super::pipe_texture_offsets(&graph);
        let welded_ends = super::welded_pipe_ends(&graph);
        let mut raw_length_mismatch_seen = false;
        let mut raw_circumference_mismatch_seen = false;
        let mut checked = 0;

        for (_, weld) in graph.welds() {
            let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                (weld.first.owner, weld.second.owner)
            else {
                continue;
            };
            let Some(first_u) = graph
                .part(first)
                .copied()
                .and_then(|spec| super::pipe_endpoint_texture_u(spec, weld.first.face))
            else {
                continue;
            };
            let Some(second_u) = graph
                .part(second)
                .copied()
                .and_then(|spec| super::pipe_endpoint_texture_u(spec, weld.second.face))
            else {
                continue;
            };
            let first_frame =
                super::pipe_endpoint_texture_frame(*graph.part(first).unwrap(), weld.first.face)
                    .unwrap();
            let second_frame =
                super::pipe_endpoint_texture_frame(*graph.part(second).unwrap(), weld.second.face)
                    .unwrap();
            let angular = -second_frame.direction.cross(second_frame.radial_zero);
            let second_angle = first_frame
                .radial_zero
                .dot(angular)
                .atan2(first_frame.radial_zero.dot(second_frame.radial_zero));

            raw_length_mismatch_seen |= (first_u - second_u).abs() > 1.0e-6;
            raw_circumference_mismatch_seen |= second_angle.abs() > 1.0e-6;
            assert!((first_u + offsets[&first].u - second_u - offsets[&second].u).abs() < 1.0e-6);
            assert!(
                (offsets[&first].v_angle - second_angle - offsets[&second].v_angle).abs() < 1.0e-6
            );
            assert!(welded_ends.contains(&weld.first));
            assert!(welded_ends.contains(&weld.second));
            checked += 1;
        }

        assert!(raw_length_mismatch_seen);
        assert!(raw_circumference_mismatch_seen);
        assert_eq!(checked, 2);
        assert_eq!(welded_ends.len(), 4);

        for (part, spec) in graph.parts() {
            match spec {
                PartSpec::Cylinder(_) | PartSpec::PipeBend(_) => {
                    assert_pipe_circumference_phase(&graph, part, *spec, offsets[&part]);
                }
                _ => {}
            }
        }

        let mesh = super::combined_construction_mesh(&graph);
        let positions = positions(&mesh);
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("mesh must have float3 normals")
        };
        for (_, weld) in graph.welds() {
            let FaceOwner::Part(_) = weld.first.owner else {
                continue;
            };
            if !welded_ends.contains(&weld.first) {
                continue;
            }
            let face = crate::builder::face_geometry_from_ref(weld.first, Some(&graph));
            assert!(!positions.iter().zip(normals).any(|(position, normal)| {
                (position - face.center).dot(face.normal).abs() < 1.0e-5
                    && Vec3::from_array(*normal).dot(face.normal).abs() > 0.9
            }));
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
    fn simulation_renders_the_same_feature_boundary_as_construction() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let transforms = [GpuTransform {
            position: [2.0, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        let construction = super::combined_construction_mesh(&graph);
        let simulation =
            combined_simulation_mesh(&graph, &creation, &transforms, SimulationMeshKind::Dynamic);
        assert_eq!(simulation.count_vertices(), construction.count_vertices());
        let construction_min_x = positions(&construction)
            .into_iter()
            .map(|position| position.x)
            .fold(f32::INFINITY, f32::min);
        let simulation_min_x = positions(&simulation)
            .into_iter()
            .map(|position| position.x)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (simulation_min_x - construction_min_x - 2.0).abs() < 1.0e-3,
            "construction {construction_min_x}, simulation {simulation_min_x}"
        );
    }

    #[test]
    fn fillet_keeps_one_texture_projection_through_its_profile() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let solid = graph.evaluated_solid(owner).unwrap();
        let edge = solid
            .logical_edges
            .iter()
            .find(|logical| {
                let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                let twin = solid.half_edges[half_edge.twin as usize];
                let patches = [
                    solid.surfaces[half_edge.face as usize].key.local,
                    solid.surfaces[twin.face as usize].key.local,
                ];
                patches.contains(&1) && patches.contains(&3)
            })
            .expect("the positive-X/positive-Y edge exists")
            .key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();

        let mesh = super::combined_construction_mesh(&graph);
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("mesh must have float3 normals")
        };
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("mesh must have float4 tangents")
        };
        let fillet_tangents = normals
            .iter()
            .zip(tangents)
            .filter_map(|(normal, tangent)| {
                let normal = Vec3::from_array(*normal).abs();
                (normal.x > 0.01 && normal.y > 0.01)
                    .then_some(Vec3::from_array(tangent[..3].try_into().unwrap()).abs())
            })
            .collect::<Vec<_>>();

        assert!(!fillet_tangents.is_empty());
        assert!(
            fillet_tangents
                .iter()
                .all(|tangent| tangent.abs_diff_eq(fillet_tangents[0], 1.0e-6)),
            "fillet changed texture projection through its profile: {fillet_tangents:?}"
        );
    }

    #[test]
    fn simulation_publication_only_touches_materials_in_each_motion_family() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(anchored) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default())
                    .unwrap()
                    .with_material(ConstructionMaterial::Steel),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(4, 0, 0), GridRotation::default()),
                )
                .unwrap()
                .with_material(ConstructionMaterial::Wood),
            ))
            .unwrap();
        let creation = graph.compile_with_static_parts([anchored]).unwrap();

        assert!(simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Static,
            ConstructionMaterial::Steel,
        ));
        assert!(!simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Dynamic,
            ConstructionMaterial::Steel,
        ));
        assert!(simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Dynamic,
            ConstructionMaterial::Wood,
        ));
        assert!(!simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Static,
            ConstructionMaterial::Wood,
        ));
    }

    #[test]
    fn zero_inner_diameter_generates_a_solid_disc_with_outward_winding() {
        let dimensions = BearingDimensions::new(0.50, 0.0).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            Vec3::ZERO,
            Vec3::Y,
            dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
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
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            Vec3::ZERO,
            Vec3::Y,
            BearingDimensions::default(),
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
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
        let attached_vertices = single_bearing_mesh(attached_dimensions).count_vertices();
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
        let expected_vertices = single_bearing_mesh(socket.dimensions).count_vertices();
        assert_eq!(build_mesh.count_vertices(), expected_vertices);

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
        assert_eq!(simulation_mesh.count_vertices(), expected_vertices);
    }

    #[test]
    fn simulation_bearing_mesh_follows_attached_and_unattached_source_bodies() {
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
        let attached_vertices = single_bearing_mesh(attached_dimensions).count_vertices();
        for (vertices, expected_anchor) in [
            (&positions[..attached_vertices], attached_anchor),
            (&positions[attached_vertices..], placed_anchor),
        ] {
            let minimum = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let maximum = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            assert!(((minimum + maximum) * 0.5).abs_diff_eq(expected_anchor, 1.0e-5));
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
        let mut engines = Vec::new();
        for (kind, x) in [(EngineKind::Gas, 20), (EngineKind::Electric, 24)] {
            let BuildOutcome::Spawned(engine) = graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    kind,
                    BuildPose::new(IVec3::new(x, 12, 0), GridRotation::default()),
                )))
                .unwrap()
            else {
                unreachable!()
            };
            engines.push(engine);
        }
        for engine in engines {
            let spec = graph.next_transmission_spec(engine).unwrap();
            graph
                .apply(BuildCommand::AttachTransmission {
                    parent: engine,
                    spec,
                })
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
        let gas = combined_authored_construction_mesh(&graph, AuthoredPart::GasEngine, None);
        let electric =
            combined_authored_construction_mesh(&graph, AuthoredPart::ElectricEngine, None);
        let gas_transmission =
            combined_authored_construction_mesh(&graph, AuthoredPart::GasTransmission, None);
        let electric_transmission =
            combined_authored_construction_mesh(&graph, AuthoredPart::ElectricTransmission, None);
        let servo = combined_authored_construction_mesh(&graph, AuthoredPart::Servo, None);
        let seat = combined_authored_construction_mesh(&graph, AuthoredPart::Seat, None);
        let input = combined_authored_construction_mesh(&graph, AuthoredPart::Input, None);

        // Two hinged blocks remain in the construction mesh; every authored part
        // has an independent batch so its material can use its own texture set.
        assert_eq!(positions(&construction).len(), 24 * 2);
        assert_eq!(positions(&controllers).len(), 24);
        assert_eq!(positions(&gas).len(), 24);
        assert_eq!(positions(&electric).len(), 24);
        assert_eq!(positions(&gas_transmission).len(), 24);
        assert_eq!(positions(&electric_transmission).len(), 24);
        assert_eq!(positions(&servo).len(), 24);
        assert_eq!(positions(&seat).len(), 24);
        assert_eq!(positions(&input).len(), 24);
        for mesh in [
            &controllers,
            &gas,
            &electric,
            &gas_transmission,
            &electric_transmission,
            &servo,
            &seat,
            &input,
        ] {
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
        assert_eq!(
            authored_uvs(AuthoredPart::GasTransmission),
            authored_uvs(AuthoredPart::ElectricTransmission),
            "both imported transmission GLBs carry the same atlas UV layout",
        );
        assert_eq!(
            authored_uvs(AuthoredPart::GasTransmission),
            authored_uvs(AuthoredPart::Controller),
            "the extracted transmission atlas maps onto the shared authored cuboid ordering",
        );
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
    fn dimension_link_states_share_the_archive_atlas_layout() {
        let uvs = authored_uvs(AuthoredPart::DimensionLinkDisabled);
        assert_eq!(
            uvs,
            authored_uvs(AuthoredPart::DimensionLinkEnabled),
            "state changes swap maps without changing the mesh atlas",
        );
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

        // +X, -X, +Y, -Y, +Z, -Z in the archive's 4x4 atlas.
        assert_eq!(bounds(0), ([0.0, 0.5], [0.25, 0.75]));
        assert_eq!(bounds(1), ([0.25, 0.5], [0.5, 0.75]));
        assert_eq!(bounds(2), ([0.0, 0.25], [0.5, 0.5]));
        assert_eq!(bounds(3), ([0.5, 0.25], [1.0, 0.5]));
        assert_eq!(bounds(4), ([0.0, 0.0], [0.5, 0.25]));
        assert_eq!(bounds(5), ([0.5, 0.0], [1.0, 0.25]));
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
    use bevy::prelude::{App, ButtonInput, IVec3, KeyCode, Quat, Update, Vec2, Vec3};
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
        DriveLinkSpec, EdgeChainRef, EdgeTreatment, FaceKind, FaceOwner, FaceRef, GridRotation,
        MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish, PartId, PartSpec,
        PendingOperation, RigidLinkSpec, STEP_METERS, ShapeRegion, SolidOwner, WeldSpec,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        AUTHORED_ORIENTATION_COUNT, AUTHORED_ORIENTATIONS, AppSimulation, BearingDimensionTarget,
        BearingToolSettings, BlockAttachment, BlockDrag, CylinderDimensionTarget,
        CylinderToolSettings, EditorGraph, EditorHistory, EditorState, HAMMER_CHARGE_SECONDS,
        HAMMER_MAX_IMPULSE, HAMMER_MIN_IMPULSE, HistoryAction, MaterialWheelState, PipeDrag,
        PipeEditMode, PlacedBearing, PlacementPlane, PlayerState, PointerSample, SelectedTool,
        SurfaceHit, Tool, active_drag_plane, adjusted_bearing_dimensions,
        adjusted_cylinder_dimensions, apply_history_action, bearing_attachment_candidate,
        bearing_attachment_is_highlighted, block_sheet_bounds, candidate_from_hit, choose_region,
        closest_axis_parameter, connect_control_link, connect_drive_wire, cycle_orientation,
        delete_box_parts, disconnect_drive_wires, hammer_delivery, hammer_impulse_magnitude,
        hammer_point_travel, handle_block_actions, handle_build_actions, handle_chroma_actions,
        handle_feature_shape_actions, handle_tool_change, pipe_pointer_delta, pipe_turn_direction,
        raycast_construction, raycast_placed_bearing_discs, raycast_placed_bearings,
        raycast_simulation, refresh_block_drag, refresh_region_drag, refresh_tool_preview,
        requested_bearing_dimension_adjustment, requested_cylinder_dimension_adjustment,
        rigid_body_parts, stage_part_deletion_preserving_bearings, tangent_feature_chain,
        tool_status_line, weld_connected_shape_owners, wire_drag_step,
    };
    use super::{RegionDrag, commit_region_drag, region_area};
    use crate::builder::{SmartGuide, block_sheet_specs};
    use crate::controls::GameAction;
    use crate::{WireConnection, WireDrag, WireDragStep, WireEnd};

    fn pointer_sample(cursor: Vec2, ray_origin: Vec3, ray_direction: Vec3) -> PointerSample {
        PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        }
    }

    #[test]
    fn blocked_delete_release_clears_the_target_that_would_block_tab() {
        let mut state = EditorState {
            delete_target: Some(super::DeleteTarget::PlacedBearing(0)),
            ..Default::default()
        };

        assert!(state.world_drag_active());
        assert!(state.cancel_delete_gesture());
        assert!(!state.world_drag_active());
    }

    #[test]
    fn committing_a_feature_clears_its_edge_selection() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let target = EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        };
        let hit = crate::shape_tool::FeatureEdgeHit {
            target,
            point: Vec3::ZERO,
            tangent: Vec3::Z,
            bisector: Vec3::X,
            distance: 0.0,
        };
        let ray_origin = Vec3::Y;
        let ray_direction = Vec3::NEG_Y;
        let mut state = EditorState {
            pointer_ray: Some((ray_origin, ray_direction)),
            feature_focus: Some(owner),
            selected_feature_edges: vec![target],
            feature_drag: Some(crate::shape_tool::FeatureDrag::begin(
                hit,
                vec![target],
                EdgeTreatment::Fillet,
                None,
                20,
                ray_origin,
                ray_direction,
            )),
            ..Default::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);
        actions.clear();
        actions.release(GameAction::Primary);
        let keys = ButtonInput::<KeyCode>::default();
        let mut history = EditorHistory::default();
        let player = PlayerState {
            input_captured: true,
            ..Default::default()
        };

        handle_feature_shape_actions(
            &actions,
            &keys,
            &mut graph,
            &mut state,
            &mut history,
            crate::shape_tool::ShapeSnap::feature_default(),
            crate::shape_tool::ShapeEditMode::Fillet,
            crate::ui::UiInput::default(),
            &player,
            &MaterialWheelState::default(),
        );

        assert_eq!(graph.shape_features().count(), 1);
        assert!(state.selected_feature_edges.is_empty());
        assert_eq!(state.selected_shape_feature, None);
        assert_eq!(history.undo.len(), 1);
    }

    #[test]
    fn tangent_edge_selection_traverses_one_weld_component() {
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for x in 0..4 {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1; 3],
                        BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        for pair in parts[..3].windows(2) {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(pair[0], FaceKind::PositiveX),
                    second: FaceRef::part(pair[1], FaceKind::NegativeX),
                }))
                .unwrap();
        }

        let target_for = |part| {
            let owner = SolidOwner::Part(part);
            let solid = graph.evaluated_solid(owner).unwrap();
            let edge = solid
                .logical_edges
                .iter()
                .find(|logical| {
                    let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                    let twin = solid.half_edges[half_edge.twin as usize];
                    let patches = [
                        solid.surfaces[half_edge.face as usize].key.local,
                        solid.surfaces[twin.face as usize].key.local,
                    ];
                    patches.contains(&3) && patches.contains(&4)
                })
                .expect("the positive-Y/negative-Z edge exists")
                .key;
            EdgeChainRef { owner, edge }
        };
        let initial = target_for(parts[0]);

        let connected = weld_connected_shape_owners(&graph, initial.owner);
        assert_eq!(connected.len(), 3);
        assert!(!connected.contains(&SolidOwner::Part(parts[3])));
        assert_eq!(
            tangent_feature_chain(&graph, initial),
            parts[..3]
                .iter()
                .copied()
                .map(target_for)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn chroma_drag_paints_each_crossed_part_as_one_undoable_stroke() {
        let mut graph = ConstructionGraph::new();
        let mut ids = Vec::new();
        for x in [0, 4] {
            let BuildOutcome::Spawned(id) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            ids.push(id);
        }
        let paint = MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.0).unwrap()),
            MaterialFinish::Painted,
        );
        let hit = |part| SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::part(part, FaceKind::PositiveY),
        };
        let mut state = EditorState {
            hovered: Some(hit(ids[0])),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();

        mouse.press(GameAction::Primary);
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
        mouse.clear();
        state.hovered = Some(hit(ids[1]));
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
        mouse.clear();
        mouse.release(GameAction::Primary);
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);

        assert_eq!(history.undo.len(), 1);
        assert!(
            ids.into_iter()
                .all(|id| graph.part(id).unwrap().appearance() == Some(paint))
        );
        assert!(apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
        ));
        assert!(
            graph
                .parts()
                .all(|(_, part)| part.appearance() == Some(MaterialAppearance::BAKED))
        );
    }

    #[test]
    fn chroma_remove_restores_baked_once_and_a_noop_stroke_adds_no_history() {
        let paint = MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([224, 86, 31], 1.0).unwrap()),
            MaterialFinish::Anodised,
        );
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default())
                    .unwrap()
                    .with_appearance(paint),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::part(part, FaceKind::PositiveY),
            }),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();

        for expected_history in [1, 1] {
            mouse.press(GameAction::Secondary);
            handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
            mouse.clear();
            mouse.release(GameAction::Secondary);
            handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
            mouse.clear();
            assert_eq!(history.undo.len(), expected_history);
            assert_eq!(
                graph.part(part).unwrap().appearance(),
                Some(MaterialAppearance::BAKED)
            );
        }
    }

    #[test]
    fn rotate_cycles_every_authored_tool_through_all_grid_orientations() {
        for tool in [Tool::Controller, Tool::GasEngine, Tool::ElectricEngine] {
            let mut state = EditorState::default();
            for expected in (1..AUTHORED_ORIENTATION_COUNT).chain(std::iter::once(0)) {
                cycle_orientation(&mut state, tool);
                assert_eq!(
                    state.authored_orientation,
                    expected,
                    "{} should rotate",
                    tool.label()
                );
            }
        }
    }

    #[test]
    fn rotate_cycles_the_axis_of_an_active_vertex_drag() {
        let region =
            ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        let start = crate::shape_tool::vertex_position(&region, [0, 0, 0]).unwrap();
        let ray_origin = start + Vec3::Z * 2.0;
        let ray_direction = Vec3::NEG_Z;
        let drag =
            crate::shape_tool::begin_group_drag(&region, [0, 0, 0], &[], ray_origin, ray_direction);
        let mut state = EditorState {
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((ray_origin, ray_direction)),
            vertex_drag: Some(drag),
            ..Default::default()
        };

        assert_eq!(cycle_orientation(&mut state, Tool::Shape), "Shape axis: Y");
        assert_eq!(state.vertex_drag.as_ref().unwrap().axis, 1);
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
    fn bearing_shortcuts_are_gated_and_adjust_the_requested_diameter() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::BearingOuterIncrease);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
            Some((BearingDimensionTarget::Outer, 1))
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Block), false),
            None
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), true),
            None
        );

        keyboard.reset_all();
        keyboard.press(GameAction::NudgeUp);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
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
        keyboard.press(GameAction::BearingInnerIncrease);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
            Some((BearingDimensionTarget::Inner, 1))
        );
    }

    #[test]
    fn cylinder_shortcuts_adjust_and_clamp_without_graph_history() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::CylinderOuterIncrease);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
            Some((CylinderDimensionTarget::Outer, 1))
        );
        keyboard.reset_all();
        keyboard.press(GameAction::CylinderInnerIncrease);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
            Some((CylinderDimensionTarget::Inner, 1))
        );
        keyboard.reset_all();
        keyboard.press(GameAction::CylinderSweepDecrease);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
            Some((CylinderDimensionTarget::Sweep, -1))
        );
        assert!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Block), false).is_none()
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
            ..Default::default()
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
            mechanic_core::ConstructionMaterial::Steel,
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

    /// A solid, welded slab of one-cell blocks with its minimum corner at the
    /// origin, which is what a region drag needs underneath it.
    fn welded_slab(size: IVec3) -> ConstructionGraph {
        let mut graph = ConstructionGraph::new();
        let mut previous: Option<PartId> = None;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let spec = CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_half_grid(
                            IVec3::ONE + IVec3::new(x, y, z) * 2,
                            GridRotation::default(),
                        ),
                    )
                    .unwrap();
                    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                    else {
                        unreachable!()
                    };
                    if let Some(first) = previous {
                        graph
                            .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second: id }))
                            .unwrap();
                    }
                    previous = Some(id);
                }
            }
        }
        graph
    }

    fn region_drag_on(
        graph: &ConstructionGraph,
        plane: PlacementPlane,
        press: PointerSample,
    ) -> RegionDrag {
        let (_, spec) = graph.parts().next().expect("the slab has blocks");
        let start = spec.as_cuboid().expect("the slab is made of blocks");
        RegionDrag {
            start,
            press,
            plane,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            region: region_area(start, IVec3::ZERO),
            error: None,
        }
    }

    #[test]
    fn shape_selection_uses_the_world_resolved_hover_hit() {
        let mut graph = welded_slab(IVec3::ONE);
        let ray_origin = Vec3::new(0.125, 2.0, 0.125);
        let ray_direction = Vec3::NEG_Y;
        let hit = raycast_construction(&graph, ray_origin, ray_direction)
            .expect("the world-resolved ray hits the block");
        let mut state = EditorState {
            hovered: Some(hit),
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((ray_origin, ray_direction)),
            ..EditorState::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);

        choose_region(
            &actions,
            &mut graph,
            &mut state,
            &mut EditorHistory::default(),
            Some(Vec2::ZERO),
            ray_origin,
            ray_direction,
        );

        assert!(state.region_drag.is_some());
    }

    #[test]
    fn shape_selection_preserves_a_fine_placed_blocks_origin() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_position_ticks(IVec3::new(60, 50, 50), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let ray_origin = Vec3::new(0.15, 2.0, 0.125);
        let ray_direction = Vec3::NEG_Y;
        let hit = raycast_construction(&graph, ray_origin, ray_direction)
            .expect("the ray hits the fine-placed block");
        let FaceOwner::Part(part) = hit.face.owner else {
            unreachable!("the ray hit a block")
        };
        let mut state = EditorState {
            hovered: Some(hit),
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((ray_origin, ray_direction)),
            ..EditorState::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);

        choose_region(
            &actions,
            &mut graph,
            &mut state,
            &mut EditorHistory::default(),
            Some(Vec2::ZERO),
            ray_origin,
            ray_direction,
        );
        commit_region_drag(&mut graph, &mut state, &mut EditorHistory::default());

        let region = state.active_region.and_then(|id| graph.region(id)).unwrap();
        assert_eq!(region.origin_steps(), IVec3::new(10, 0, 0));
        assert!(
            (region.bounds_steps().0.as_vec3() * STEP_METERS)
                .abs_diff_eq(Vec3::new(0.025, 0.0, 0.0), 1.0e-7)
        );
        assert_eq!(graph.region_of(part), state.active_region);
    }

    #[test]
    fn dragging_across_blocks_claims_all_of_them_as_one_region() {
        let graph = welded_slab(IVec3::new(3, 2, 1));
        // Straight down onto the top of the first block, which is the XZ plane
        // the pointer then slides along.
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };

        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
        );

        let drag = state.region_drag.as_ref().unwrap();
        assert_eq!(drag.span, IVec3::new(2, 0, 0));
        assert_eq!(drag.region.size_cells(), IVec3::new(3, 1, 1));
        assert_eq!(drag.error, None, "three welded blocks are a valid area");
    }

    #[test]
    fn rotate_mid_area_drag_extrudes_the_selection_into_a_box() {
        let graph = welded_slab(IVec3::new(3, 2, 1));
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };
        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
        );

        // What Rotate does: keep the rectangle already dragged and re-anchor here.
        let rotated = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 0.125, 2.0), Vec3::NEG_Z);
        {
            let drag = state.region_drag.as_mut().unwrap();
            drag.plane = drag.plane.cycle();
            assert_eq!(drag.plane, PlacementPlane::Xy);
            drag.anchor_span = drag.span;
            drag.press = rotated;
            drag.last_span = None;
        }

        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(0.0, 100.0),
            rotated.ray_origin,
            (Vec3::new(0.125, 0.375, 0.125) - rotated.ray_origin).normalize(),
        );

        let drag = state.region_drag.as_ref().unwrap();
        assert_eq!(
            drag.span,
            IVec3::new(2, 1, 0),
            "the rotation keeps the extent and grows the third axis"
        );
        assert_eq!(drag.region.size_cells(), IVec3::new(3, 2, 1));
        assert_eq!(drag.error, None);
    }

    #[test]
    fn releasing_a_valid_area_opens_it_for_editing() {
        let mut graph = welded_slab(IVec3::new(2, 1, 1));
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };
        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.375, 0.0, 0.125) - press.ray_origin).normalize(),
        );
        let mut history = EditorHistory::default();

        commit_region_drag(&mut graph, &mut state, &mut history);

        assert!(state.region_drag.is_none());
        let region = state.active_region.and_then(|id| graph.region(id)).unwrap();
        assert_eq!(region.size_cells(), IVec3::new(2, 1, 1));
        assert_eq!(history.undo.len(), 1);
    }

    #[test]
    fn an_area_reaching_past_the_blocks_is_refused_rather_than_claimed() {
        let mut graph = welded_slab(IVec3::new(2, 1, 1));
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };
        // Three cells wide over a two-block slab: the far cell is empty.
        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
        );
        assert!(state.region_drag.as_ref().unwrap().error.is_some());

        let mut history = EditorHistory::default();
        commit_region_drag(&mut graph, &mut state, &mut history);

        assert_eq!(graph.regions().count(), 0);
        assert!(state.active_region.is_none());
        assert!(history.undo.is_empty());
    }

    #[test]
    fn placing_blocks_shows_the_same_plane_as_choosing_an_area() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let specs =
            block_sheet_specs(candidate.spec, IVec3::new(4, 1, 2), PlacementPlane::Xz).unwrap();
        let state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                start_guides: Vec::new(),
                press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::new(2, 0, 1),
                last_span: Some(IVec3::new(2, 0, 1)),
                specs: specs.clone(),
                error: None,
            }),
            ..Default::default()
        };

        let (low, high, plane) =
            active_drag_plane(&state, &AppSimulation::default()).expect("a block drag has a plane");
        assert_eq!(plane, PlacementPlane::Xz);
        // Centred on the blocks about to be placed, exactly as an area is.
        let (sheet_low, sheet_high) = block_sheet_bounds(&specs).unwrap();
        assert!(low.abs_diff_eq(sheet_low, 1.0e-6));
        assert!(high.abs_diff_eq(sheet_high, 1.0e-6));

        assert!(
            active_drag_plane(&EditorState::default(), &AppSimulation::default()).is_none(),
            "no drag, no plane"
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
            .insert_resource(SelectedTool::from_editor_tool(Tool::Bearing))
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

        mouse.press(GameAction::Primary);
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
        mouse.release(GameAction::Primary);
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
        let start_guide = SmartGuide {
            axis: 0,
            coordinate: 0.0,
            from: Vec3::ZERO,
            to: Vec3::Z,
        };
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                start_guides: vec![start_guide],
                press,
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::ZERO,
                last_span: None,
                specs: vec![candidate.spec],
                error: None,
            }),
            smart_guides: vec![start_guide],
            ..Default::default()
        };

        refresh_block_drag(
            &graph,
            &mut state,
            Vec2::new(4.99, 0.0),
            press.ray_origin,
            Quat::from_rotation_z(0.003) * Vec3::NEG_Y,
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
                press.ray_origin,
                (Vec3::new(target.x, 0.0, target.z) - press.ray_origin).normalize(),
            );
            assert_eq!(state.block_drag.as_ref().unwrap().specs.len(), expected);
            assert!(state.smart_guides.contains(&start_guide));
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
                start_guides: Vec::new(),
                press,
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::ZERO,
                last_span: None,
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
            press.ray_origin,
            (Vec3::new(0.5, 0.0, 0.0) - press.ray_origin).normalize(),
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
                start_guides: Vec::new(),
                press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::new(2, 0, 1),
                last_span: Some(IVec3::new(2, 0, 1)),
                specs,
                error: None,
            }),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();
        mouse.press(GameAction::Primary);
        mouse.clear();
        mouse.release(GameAction::Primary);

        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(graph.part_count(), 6);
        assert_eq!(graph.weld_count(), 13);
        assert_eq!(history.undo.len(), 1);
        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
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
        mouse.press(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings.len(), 1);
        assert!(state.block_drag.is_some());

        mouse.clear();
        mouse.release(GameAction::Primary);
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
        assert!((preview.spec.pose.translation().x - 0.375).abs() < 1.0e-6);

        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();
        mouse.press(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        mouse.clear();
        mouse.release(GameAction::Primary);
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
            // A delete drag anchors on the press, so it needs the pointer.
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::Y, Vec3::NEG_Y)),
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        mouse.press(GameAction::Secondary);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let state = app.world().resource::<EditorState>();
            assert!(state.delete_target.is_none());
            assert!(state.delete_drag.is_some());
        }
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            mouse.clear();
            mouse.release(GameAction::Secondary);
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
        mouse.press(GameAction::Secondary);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            mouse.clear();
            mouse.release(GameAction::Secondary);
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
    fn delete_drag_selects_only_the_box_it_spans() {
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

        // A flat span is the four in that plane; the block above is untouched.
        let flat = delete_box_parts(&graph, start, IVec3::new(1, 0, 1)).unwrap();
        assert_eq!(flat.len(), 4);
        assert!(!flat.contains(&parts[4]));

        // Rotating into the third axis reaches the one above too.
        let boxed = delete_box_parts(&graph, start, IVec3::new(1, 1, 1)).unwrap();
        assert_eq!(boxed.len(), 5);
        assert!(boxed.contains(&parts[4]));
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
    fn linking_an_input_to_a_seat_asks_for_the_wire_overlay_to_be_redrawn() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(input) = graph
            .apply(BuildCommand::SpawnInput(mechanic_core::InputSpec::new(
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(seat) = graph
            .apply(BuildCommand::SpawnSeat(mechanic_core::SeatSpec::new(
                BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        let message = connect_control_link(
            &mut graph,
            &mut state,
            &mut history,
            BuildCommand::AddInputSeatLink(mechanic_core::InputSeatLinkSpec { input, seat }),
            "Linked Input to Seat",
        );

        assert_eq!(message, "Linked Input to Seat");
        assert_eq!(graph.input_seat_links().count(), 1);
        // The wire is a line in the drive overlay, and that overlay is only
        // rebuilt on request: without the flag it stays invisible until some
        // unrelated edit happens to dirty the mesh.
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
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(selected.contains("2 bearings wired"), "{selected}");
        assert!(
            selected.contains("Interact opens its program"),
            "{selected}"
        );

        let single = tool_status_line(
            Tool::Connector,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            Some(1),
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(single.contains("1 bearing wired"), "{single}");

        let none = tool_status_line(
            Tool::Controller,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            None,
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(none.contains("No block selected"), "{none}");
    }

    #[test]
    fn pipette_copies_material_dimensions_authored_orientation_and_bearing_setup() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(cuboid) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
                )
                .unwrap()
                .with_material(ConstructionMaterial::Wood),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let cylinder_dimensions = CylinderDimensions::new(0.75, 0.25, 1.5).unwrap();
        let BuildOutcome::Spawned(cylinder) = graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    cylinder_dimensions,
                    BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
                )
                .with_material(ConstructionMaterial::Concrete),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let orientation = AUTHORED_ORIENTATIONS[17];
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(16, 2, 0), orientation),
            )))
            .unwrap()
        else {
            unreachable!()
        };

        let shaped_spec = CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(24, 1, 0), GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Concrete);
        let BuildOutcome::Spawned(shaped) = graph.apply(BuildCommand::Spawn(shaped_spec)).unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::RegionAdded(region) = graph
            .apply(BuildCommand::AddRegion(super::region_area(
                shaped_spec,
                IVec3::ZERO,
            )))
            .unwrap()
        else {
            unreachable!()
        };

        let mut state = EditorState::default();
        let mut selection = SelectedTool::default();
        selection.clear();
        let mut material = super::SelectedMaterial(ConstructionMaterial::Steel);
        let mut bearing = BearingToolSettings::default();
        let mut cylinder_settings = CylinderToolSettings::default();
        macro_rules! apply {
            ($setup:expr) => {
                super::apply_pipette_setup(
                    $setup,
                    &graph,
                    &mut state,
                    &mut selection,
                    &mut material,
                    &mut bearing,
                    &mut cylinder_settings,
                )
            };
        }

        apply!(super::PipetteSetup::Part(cuboid));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Block));
        assert_eq!(material.0, ConstructionMaterial::Wood);

        apply!(super::PipetteSetup::Part(cylinder));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Cylinder));
        assert_eq!(material.0, ConstructionMaterial::Concrete);
        assert_eq!(cylinder_settings.dimensions, cylinder_dimensions);

        apply!(super::PipetteSetup::Part(controller));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Controller));
        assert_eq!(state.authored_orientation, 17);

        let dimensions = BearingDimensions::new(0.9, 0.4).unwrap();
        apply!(super::PipetteSetup::Bearing(dimensions));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Bearing));
        assert_eq!(bearing.dimensions, dimensions);

        material.0 = ConstructionMaterial::Wood;
        apply!(super::PipetteSetup::Ground);
        assert_eq!(selection.active_editor_tool(), Some(Tool::Block));
        assert_eq!(material.0, ConstructionMaterial::Wood);

        apply!(super::PipetteSetup::Part(shaped));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Shape));
        assert_eq!(state.active_region, Some(region));
        assert_eq!(material.0, ConstructionMaterial::Concrete);
    }

    #[test]
    fn pipette_uses_simulation_space_and_reports_no_target() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let simulation = AppSimulation {
            transforms: vec![GpuTransform {
                position: [10.0, 0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
            creation: Some(creation),
            ..Default::default()
        };
        assert_eq!(
            super::pipette_at_ray(
                &graph,
                &EditorState::default(),
                &simulation,
                Vec3::new(10.0, 0.0, 5.0),
                Vec3::NEG_Z,
            ),
            Some(super::PipetteSetup::Part(part)),
        );
        assert_eq!(
            super::pipette_at_ray(
                &ConstructionGraph::new(),
                &EditorState::default(),
                &AppSimulation::default(),
                Vec3::Y,
                Vec3::Y,
            ),
            None,
        );
    }

    #[test]
    fn clearing_the_hand_cancels_pending_edits_and_hammer_charge() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        super::begin_weld(&mut graph, FaceRef::part(part, FaceKind::PositiveY)).unwrap();
        let mut state = EditorState::default();
        let mut selection = SelectedTool::from_editor_tool(Tool::Weld);
        let mut hammer = super::HammerInteraction {
            charging: Some(super::HammerCharge {
                body_index: 0,
                local_point: Vec3::ZERO,
                direction: Vec3::Y,
                elapsed_seconds: 1.0,
            }),
            pending: None,
        };
        super::clear_held_tool(&mut graph, &mut state, &mut selection, &mut hammer);
        assert_eq!(selection.active_editor_tool(), None);
        assert!(graph.pending().is_none());
        assert!(hammer.charging.is_none() && hammer.pending.is_none());
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

    #[test]
    fn pipe_dimension_modes_cycle_without_mutating_the_current_value() {
        let endpoint = Vec3::new(0.0, 1.0, 0.0);
        let dimensions = CylinderDimensions::new(0.50, 0.25, 1.0).unwrap();
        let mut mode = PipeEditMode::Length;
        for expected in [
            PipeEditMode::OuterDiameter,
            PipeEditMode::InnerDiameter,
            PipeEditMode::Length,
        ] {
            mode = mode.next();
            assert_eq!(mode, expected);
            assert_eq!(endpoint, Vec3::new(0.0, 1.0, 0.0));
            assert_eq!(
                dimensions,
                CylinderDimensions::new(0.50, 0.25, 1.0).unwrap()
            );
        }
    }

    #[test]
    fn pipe_turn_chooser_locks_only_perpendicular_aim_beyond_the_dead_zone() {
        let anchor = Vec3::Z;
        assert_eq!(
            pipe_turn_direction(Vec3::X, anchor, (Vec3::Z + Vec3::Y * 0.1).normalize()),
            Some(Vec3::Y)
        );
        assert!(
            pipe_turn_direction(Vec3::X, anchor, (Vec3::Z + Vec3::X * 0.1).normalize()).is_none(),
            "aim along the incoming axis cannot select a perpendicular direction"
        );
        assert!(pipe_turn_direction(Vec3::X, anchor, anchor).is_none());
    }

    #[test]
    fn pipe_drag_reanchors_length_and_diameter_measurements() {
        let axis_origin = Vec3::ZERO;
        let direction = Vec3::Y;
        let camera = Vec3::new(0.0, 0.0, -4.0);
        let first_ray = (Vec3::new(0.0, 1.0, 0.0) - camera).normalize();
        let second_ray = (Vec3::new(0.0, 1.5, 0.0) - camera).normalize();
        let first = closest_axis_parameter(axis_origin, direction, camera, first_ray).unwrap();
        let second = closest_axis_parameter(axis_origin, direction, camera, second_ray).unwrap();
        assert!((first - 1.0).abs() < 1.0e-5);
        assert!((second - 1.5).abs() < 1.0e-5);
        assert!(pipe_pointer_delta(Vec3::Z, (Vec3::Z + Vec3::Y * 0.05).normalize()).abs() > 0.0);
    }

    #[test]
    fn bend_activity_owns_wheel_only_after_turning_starts() {
        let dimensions = CylinderDimensions::default();
        let cylinder = CylinderSpec::new(dimensions, BuildPose::default());
        let make_drag = || PipeDrag {
            attachment: BlockAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
            start: Vec3::ZERO,
            corners: Vec::new(),
            endpoint: Vec3::Y * 0.25,
            directions: vec![Vec3::Y],
            bend_radii: Vec::new(),
            pending_radius: 0.25,
            dimensions,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            mode: PipeEditMode::Length,
            choosing_direction: false,
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::ZERO,
                ray_direction: Vec3::Z,
            },
            anchor_endpoint: Vec3::Y * 0.25,
            anchor_dimensions: dimensions,
            pieces: vec![crate::builder::PipeRunPiece {
                spec: PartSpec::Cylinder(cylinder),
                inlet: FaceKind::NegativeY,
                outlet: FaceKind::PositiveY,
            }],
            error: None,
        };
        let mut state = EditorState {
            pipe_drag: Some(make_drag()),
            ..Default::default()
        };
        assert!(!state.pipe_bend_active());
        state.pipe_drag.as_mut().unwrap().choosing_direction = true;
        assert!(state.pipe_bend_active());
        state.pipe_drag.as_mut().unwrap().choosing_direction = false;
        state.pipe_drag.as_mut().unwrap().bend_radii.push(0.25);
        assert!(state.pipe_bend_active());
    }
}

#[cfg(test)]
mod history_tests {
    use bevy::prelude::{ButtonInput, IVec3, Vec2, Vec3};
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
    use crate::controls::GameAction;

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
        for _ in 0..2 {
            let mut keyboard = ButtonInput::default();
            keyboard.press(GameAction::Undo);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Undo)
            );

            keyboard.reset_all();
            keyboard.press(GameAction::Redo);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Redo)
            );
        }

        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::Save);
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
            start_guides: Vec::new(),
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            specs: vec![candidate.spec],
            error: None,
        });
        state.delete_drag = Some(DeleteDrag {
            start: graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            parts: vec![support],
            error: None,
        });
        state.delete_target = Some(DeleteTarget::PlacedBearing(0));

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);

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

        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);

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

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
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
        );
        assert_eq!(history.redo.len(), 1);
        state.feedback = Some("camera and tool changes are transient".to_owned());
        assert_eq!(history.redo.len(), 1);

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.redo.is_empty());
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);
    }

    #[test]
    fn empty_history_stacks_report_guidance_without_mutation() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(graph.parts().next().unwrap().0, part);
        assert_eq!(state.feedback.as_deref(), Some("Nothing to undo"));
        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);
        assert_eq!(state.feedback.as_deref(), Some("Nothing to redo"));
    }
}

#[cfg(test)]
mod showcase_loading_tests {
    use std::time::Duration;

    use bevy::prelude::IVec3;
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, CuboidSpec, GridRotation, TopologyError,
    };
    use mechanic_gpu::FixedStepScheduler;

    use super::{
        ConstructionGraph, EditorHistory, EditorSnapshot, EditorState, HistoryAction,
        apply_history_action, creation_requires_live_physics, install_editor_graph,
        next_simulation_ticks, showcase, visual_snapshot_is_due,
    };

    #[test]
    fn terrain_anchored_construction_does_not_start_live_physics() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };

        assert!(creation_requires_live_physics(&graph.compile().unwrap()));
        assert!(!creation_requires_live_physics(
            &graph.compile_with_static_parts([part]).unwrap()
        ));
    }

    #[test]
    fn app_simulation_stages_catch_up_ticks_without_dropping_backlog() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 1;
        let mut backlog = 0;

        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                Duration::from_secs(1),
                false,
                3,
            ),
            1..4
        );
        assert_eq!(scheduler.next_tick(), 61);
        assert_eq!(backlog, 57);
        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                Duration::from_millis(17),
                false,
                3,
            ),
            4..7
        );
        assert_eq!(backlog, 55);
        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                Duration::ZERO,
                false,
                u64::MAX,
            ),
            7..62
        );
        assert_eq!(next_tick, 62);
        assert_eq!(backlog, 0);
    }

    #[test]
    fn paused_simulation_does_not_advance_or_accumulate_time() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 7;
        let mut backlog = 5;
        let scheduler_tick = scheduler.next_tick();

        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                Duration::from_secs(10),
                true,
                3,
            ),
            7..7
        );
        assert_eq!(next_tick, 7);
        assert_eq!(backlog, 5);
        assert_eq!(scheduler.next_tick(), scheduler_tick);
    }

    #[test]
    fn prototype_meshes_publish_every_second_completed_physics_tick() {
        assert!(!visual_snapshot_is_due(10, 10));
        assert!(!visual_snapshot_is_due(10, 11));
        assert!(visual_snapshot_is_due(10, 12));
        assert!(visual_snapshot_is_due(10, 14));
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

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 0);
        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);
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
mod placement_snap_tests {
    use bevy::math::DVec2;

    use super::*;

    #[test]
    fn modifier_precedence_selects_precision_only_with_shift_and_control() {
        assert_eq!(
            PlacementGrid::from_modifiers(false, false),
            PlacementGrid::Centimetres25
        );
        assert_eq!(
            PlacementGrid::from_modifiers(false, true),
            PlacementGrid::Centimetres25
        );
        assert_eq!(
            PlacementGrid::from_modifiers(true, false),
            PlacementGrid::Centimetres5
        );
        assert_eq!(
            PlacementGrid::from_modifiers(true, true),
            PlacementGrid::Centimetres1
        );
    }

    #[test]
    fn alt_tap_toggles_but_range_adjustment_does_not() {
        let mut settings = SmartSnapSettings::default();
        settings.update(true, 0.0, false);
        assert!(!settings.enabled);

        settings.update(false, 1.0, false);
        assert!((settings.range - 1.25).abs() <= f32::EPSILON);
        assert!(settings.range_adjusted_this_frame);
        settings.update(true, 0.0, false);
        assert!(!settings.enabled);

        settings.update(false, 100.0, false);
        assert!((settings.range - 5.0).abs() <= f32::EPSILON);
        settings.update(false, -100.0, false);
        assert!((settings.range - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn free_range_adjustment_is_contextual_clamped_and_yields_to_object_snap() {
        let mut settings = FreePlacementSettings::default();
        settings.update(1.0, false, false);
        assert!((settings.range - 5.0).abs() <= f32::EPSILON);
        assert!(!settings.range_adjusted_this_frame);

        settings.update(1.0, true, false);
        assert!((settings.range - 5.25).abs() <= f32::EPSILON);
        assert!(settings.range_adjusted_this_frame);

        settings.update(1.0, true, true);
        assert!((settings.range - 5.25).abs() <= f32::EPSILON);
        assert!(!settings.range_adjusted_this_frame);

        settings.update(1000.0, true, false);
        assert!((settings.range - 30.0).abs() <= f32::EPSILON);
        settings.update(-1000.0, true, false);
        assert!((settings.range - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn free_fallback_is_garage_only_and_requires_an_eligible_tool() {
        let origin = Vec3::new(1.0, 6.0, 2.0);
        let direction = Vec3::NEG_Z;
        assert_eq!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                false,
            ),
            Some(Vec3::new(1.0, 6.0, -3.0))
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Bearing,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                false,
            )
            .is_none()
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::World {
                    origin: DVec2::ZERO,
                },
                origin,
                direction,
                5.0,
                false,
            )
            .is_none()
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                true,
            )
            .is_none()
        );
    }

    #[test]
    fn lattice_coordinates_keep_global_phase_and_emphasis_hierarchy() {
        let coordinates = lattice_coordinates(-0.25, 0.25, 0, PlacementGrid::Centimetres1);
        assert_eq!(coordinates.len(), 50);
        let mut expected = -0.245;
        for coordinate in coordinates {
            assert!((coordinate - expected).abs() < 1.0e-5);
            expected += 0.01;
        }
        assert!(lattice_thickness(0, 50) > lattice_thickness(0, 10));
        assert!(lattice_thickness(0, 10) > lattice_thickness(0, 2));
    }

    #[test]
    fn lattice_wraps_only_one_cell_beyond_the_preview() {
        let selection_low = Vec3::ZERO;
        let selection_high = Vec3::splat(0.25);
        let geometry = placement_lattice_geometry(
            selection_low,
            selection_high,
            Vec3::ZERO,
            PlacementGrid::Centimetres5,
            None,
        );

        let centers = lattice_line_centers(&geometry);
        assert!(!centers.is_empty());
        for center in centers {
            assert!(
                center.cmpge(Vec3::splat(-0.050_01)).all()
                    && center.cmple(Vec3::splat(0.300_01)).all(),
                "line centre {center:?} escaped the one-cell envelope"
            );
            assert!(
                !(0..3).all(|axis| {
                    coordinate_inside(center[axis], selection_low[axis], selection_high[axis])
                }),
                "line centre {center:?} crossed the preview interior"
            );
        }
    }

    #[test]
    fn dragging_shows_only_a_one_cell_border_on_the_whole_active_plane() {
        let selection_low = Vec3::ZERO;
        let selection_high = Vec3::new(0.75, 0.25, 0.5);
        let geometry = placement_lattice_geometry(
            selection_low,
            selection_high,
            Vec3::ZERO,
            PlacementGrid::Centimetres5,
            Some(PlacementPlane::Xz),
        );

        let centers = lattice_line_centers(&geometry);
        assert!(!centers.is_empty());
        assert!(centers.iter().all(|center| {
            (center.y - 0.125).abs() < 1.0e-6
                && center.x >= -0.05
                && center.x <= 0.80
                && center.z >= -0.05
                && center.z <= 0.55
                && !(coordinate_inside(center.x, selection_low.x, selection_high.x)
                    && coordinate_inside(center.z, selection_low.z, selection_high.z))
        }));

        for vertices in geometry.positions.chunks_exact(CUBE_POSITIONS.len()) {
            let low = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let high = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            let extent = high - low;
            assert!(extent.y <= 0.004_1, "no line may run normal to XZ");
            assert!(extent.x > 0.01 || extent.z > 0.01);
        }
    }

    #[test]
    fn snap_range_wraps_the_whole_selection_on_the_active_plane() {
        let geometry = smart_snap_range_geometry(
            Vec3::ZERO,
            Vec3::new(0.75, 0.25, 0.5),
            1.0,
            Some(PlacementPlane::Xz),
        );

        let centers = lattice_line_centers(&geometry);
        assert_eq!(centers.len(), 36);
        assert!(
            centers
                .iter()
                .all(|center| (center.y - 0.125).abs() < 1.0e-6)
        );
        assert!(centers.iter().any(|center| center.x < -0.99));
        assert!(centers.iter().any(|center| center.x > 1.74));
        assert!(centers.iter().any(|center| center.z < -0.99));
        assert!(centers.iter().any(|center| center.z > 1.49));
    }

    #[test]
    fn free_preview_snap_range_uses_three_orthogonal_outlines() {
        let geometry = smart_snap_range_geometry(Vec3::ZERO, Vec3::splat(0.25), 0.5, None);

        let centers = lattice_line_centers(&geometry);
        assert_eq!(centers.len(), 108);
        for axis in 0..3 {
            assert!(
                centers
                    .iter()
                    .filter(|center| (center[axis] - 0.125).abs() < 1.0e-6)
                    .count()
                    >= 36
            );
        }
    }

    fn lattice_line_centers(geometry: &OverlayGeometry) -> Vec<Vec3> {
        geometry
            .positions
            .chunks_exact(CUBE_POSITIONS.len())
            .map(|vertices| {
                let low = vertices
                    .iter()
                    .map(|position| Vec3::from_array(*position))
                    .fold(Vec3::splat(f32::INFINITY), Vec3::min);
                let high = vertices
                    .iter()
                    .map(|position| Vec3::from_array(*position))
                    .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
                (low + high) * 0.5
            })
            .collect()
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
