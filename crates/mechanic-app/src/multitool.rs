//! First-person rendering for the Claude Design multitool asset pack.
//!
//! The supplied asset is a procedural Three.js rig, not a baked mesh. This is
//! its native Bevy representation: the authored dimensions, hierarchy, folds,
//! deployment beats, flip, and use states remain data-driven here.

use core::f32::consts::{PI, TAU};

use bevy::{
    camera::visibility::RenderLayers, core_pipeline::tonemapping::Tonemapping,
    ecs::system::SystemParam, light::NotShadowCaster, prelude::*,
};

use crate::{
    EditorGraph, EditorState, FovCamera, HammerInteraction,
    camera::{AVATAR_HIDDEN_PULLBACK, MainCamera, PlayerCamera, PlayerState},
    controls::GameAction,
    garage,
    hotbar::{MainTool, SelectedTool},
    hovered_part, wire_end_under_cursor,
};

const VIEWMODEL_LAYER: usize = 2;
const PANEL_LENGTH: f32 = 0.280;
const PANEL_THICKNESS: f32 = 0.011;
const PANEL_WIDTH: f32 = 0.052;
const HEX_APOTHEM: f32 = 0.050;
const PRISM_ROOT: f32 = 0.600;
const DEPLOY_SECONDS: f32 = 0.75;
const CONNECTOR_DEPLOY_SECONDS: f32 = 0.38;
const TUCK_SECONDS: f32 = 0.18;
const FLIP_SECONDS: f32 = 0.30;
const POSE_SECONDS: f32 = 0.24;
const AIM_SECONDS: f32 = 0.13;
const VIEWMODEL_SCALE: f32 = 1.38;
const ACTIVE_SCALE: f32 = 1.60;
const CARRY_TRANSLATION: Vec3 = Vec3::new(0.78, -1.20, -1.04);
const READY_TRANSLATION: Vec3 = Vec3::new(0.30, -0.67, -1.04);
const TUCKED_TRANSLATION: Vec3 = Vec3::new(0.80, -0.84, -0.76);
const AIM_TRANSLATION: Vec3 = Vec3::new(0.62, -0.77, -1.08);
const PANEL_PHASES: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

pub(crate) struct MultitoolPlugin;

impl Plugin for MultitoolPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn.after(crate::setup))
            .add_systems(
                Update,
                update
                    .after(crate::handle_hammer_actions)
                    .run_if(crate::debug_frame_updates_enabled),
            );
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum MultitoolKind {
    #[default]
    Matter,
    Welder,
    Connector,
    Sledge,
}

impl MultitoolKind {
    const ALL: [Self; 4] = [Self::Matter, Self::Welder, Self::Connector, Self::Sledge];

    const fn from_main_tool(tool: MainTool) -> Self {
        match tool {
            MainTool::MatterManipulator => Self::Matter,
            MainTool::Welder => Self::Welder,
            MainTool::Connector => Self::Connector,
            MainTool::Hammer => Self::Sledge,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelRole {
    Finger,
    Collar,
    Tine,
    Shield,
    Fin,
    Rib,
    Lamina,
    Strike,
}

#[derive(Clone, Copy, Debug)]
struct Fold {
    radial: f32,
    slide: f32,
    hinge: f32,
    yaw: f32,
    thickness: f32,
    spin: f32,
    role: PanelRole,
}

impl Fold {
    fn for_panel(tool: MultitoolKind, panel: usize) -> Self {
        match tool {
            MultitoolKind::Matter if panel.is_multiple_of(2) => Self {
                radial: 0.004,
                slide: -0.055,
                hinge: 0.20,
                yaw: 0.0,
                thickness: 1.0,
                spin: 0.0,
                role: PanelRole::Finger,
            },
            MultitoolKind::Matter => Self {
                radial: 0.0,
                slide: -0.255,
                hinge: 0.04,
                yaw: 0.0,
                thickness: 1.0,
                spin: 0.0,
                role: PanelRole::Collar,
            },
            MultitoolKind::Welder if panel == 0 || panel == 3 => Self {
                radial: -0.006,
                slide: 0.0,
                hinge: 0.0,
                yaw: 0.0,
                thickness: 1.0,
                spin: 0.0,
                role: PanelRole::Tine,
            },
            MultitoolKind::Welder if panel == 1 || panel == 5 => Self {
                radial: 0.006,
                slide: -0.190,
                hinge: 1.15,
                yaw: 0.0,
                thickness: 1.0,
                spin: 0.0,
                role: PanelRole::Shield,
            },
            MultitoolKind::Welder => Self {
                radial: 0.002,
                slide: -0.262,
                hinge: 0.07,
                yaw: 0.0,
                thickness: 1.0,
                spin: 0.0,
                role: PanelRole::Fin,
            },
            MultitoolKind::Connector => Self {
                radial: 0.006,
                slide: -0.100,
                hinge: 0.44 + if panel.is_multiple_of(2) { 0.0 } else { 0.06 },
                yaw: 0.0,
                thickness: 1.0,
                spin: 0.0,
                role: PanelRole::Rib,
            },
            MultitoolKind::Sledge => sledge_fold(panel),
        }
    }
}

fn sledge_fold(panel: usize) -> Fold {
    let (radial, yaw, spin, role) = match panel {
        0 | 3 => (-0.010, 0.0, 0.0, PanelRole::Lamina),
        1 => (0.014, -PI / 3.0, 0.0, PanelRole::Lamina),
        2 => (0.014, PI / 3.0, 0.0, PanelRole::Lamina),
        4 => (0.038, -PI / 3.0, PI, PanelRole::Strike),
        5 => (0.038, PI / 3.0, PI, PanelRole::Strike),
        _ => unreachable!("a multitool end always has six panels"),
    };
    Fold {
        radial,
        slide: 0.0,
        hinge: 0.0,
        yaw,
        thickness: 2.0,
        spin,
        role,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum FlipPhase {
    #[default]
    Idle,
    Holstering,
    Stowing,
    Rotating,
    Deploying,
}

#[derive(Clone, Copy, Debug, Default)]
struct InteractionFrame {
    using: bool,
    highlighted: bool,
    activation: Activation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Activation {
    #[default]
    None,
    MatterShot,
    HammerStrike,
}

#[derive(SystemParam)]
struct MultitoolInput<'w> {
    selection: Res<'w, SelectedTool>,
    actions: Res<'w, ButtonInput<GameAction>>,
    player: Res<'w, PlayerState>,
    graph: Res<'w, EditorGraph>,
    editor: Res<'w, EditorState>,
    hammer: Res<'w, HammerInteraction>,
}

#[derive(Resource, Debug)]
struct MultitoolState {
    active_end: usize,
    destination_end: usize,
    flip_phase: FlipPhase,
    tools: [MultitoolKind; 2],
    progress: [f32; 2],
    targets: [f32; 2],
    spin: f32,
    spin_target: f32,
    tuck: f32,
    ready: f32,
    aim: f32,
    highlight: f32,
    carry_requested: bool,
    hammer_charge: f32,
    hammer_swing_time: f32,
    elapsed: f32,
    shot_time: [f32; 2],
    matter_parity: [bool; 2],
    panel_blend: [[f32; 6]; 2],
    connector_wind: [f32; 2],
    connector_blend: [f32; 2],
}

impl Default for MultitoolState {
    fn default() -> Self {
        Self {
            active_end: 0,
            destination_end: 0,
            flip_phase: FlipPhase::Idle,
            tools: [MultitoolKind::Matter, MultitoolKind::Connector],
            progress: [0.0, 0.0],
            targets: [1.0, 0.0],
            spin: 0.0,
            spin_target: 0.0,
            tuck: 0.0,
            ready: 0.0,
            aim: 0.0,
            highlight: 0.0,
            carry_requested: false,
            hammer_charge: 0.0,
            hammer_swing_time: 9.0,
            elapsed: 0.0,
            shot_time: [9.0; 2],
            matter_parity: [true; 2],
            panel_blend: [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; 2],
            connector_wind: [0.0; 2],
            connector_blend: [0.0; 2],
        }
    }
}

impl MultitoolState {
    fn select(&mut self, tool: MultitoolKind) {
        self.carry_requested = false;
        if self.flip_phase == FlipPhase::Holstering {
            self.flip_phase = FlipPhase::Idle;
        }
        if matches!(self.flip_phase, FlipPhase::Stowing | FlipPhase::Rotating) {
            self.tools[self.destination_end] = tool;
            return;
        }
        if self.tools[self.active_end] == tool {
            self.targets[self.active_end] = 1.0;
            if self.progress[self.active_end] < 1.0 {
                self.flip_phase = FlipPhase::Deploying;
            }
            return;
        }
        self.destination_end = 1 - self.active_end;
        self.tools[self.destination_end] = tool;
        self.targets = [0.0; 2];
        self.spin_target += PI;
        self.flip_phase = FlipPhase::Stowing;
    }

    fn holster(&mut self) {
        if self.carry_requested {
            return;
        }
        if self.flip_phase == FlipPhase::Rotating {
            self.spin = self.spin_target;
            self.active_end = self.destination_end;
        }
        self.carry_requested = true;
        self.targets = [0.0; 2];
        self.flip_phase = FlipPhase::Holstering;
    }

    fn set_selection(&mut self, tool: Option<MultitoolKind>) {
        if let Some(tool) = tool {
            self.select(tool);
        } else {
            self.holster();
        }
    }

    fn advance(&mut self, delta_seconds: f32, interaction: InteractionFrame) {
        let delta_seconds = delta_seconds.min(0.05);
        self.elapsed += delta_seconds;
        let tuck_target = match self.flip_phase {
            FlipPhase::Idle | FlipPhase::Holstering | FlipPhase::Deploying => 0.0,
            FlipPhase::Stowing | FlipPhase::Rotating => 1.0,
        };
        self.tuck = move_towards(self.tuck, tuck_target, delta_seconds / TUCK_SECONDS);
        let ready_target =
            if !self.carry_requested || self.progress.iter().any(|progress| *progress > 0.0) {
                1.0
            } else {
                0.0
            };
        self.ready = move_towards(self.ready, ready_target, delta_seconds / POSE_SECONDS);
        let active_tool = self.tools[self.active_end];
        let aim_target = if interaction.using
            && !matches!(active_tool, MultitoolKind::Sledge)
            && self.progress[self.active_end] > 0.96
        {
            1.0
        } else {
            0.0
        };
        self.aim = move_towards(self.aim, aim_target, delta_seconds / AIM_SECONDS);
        let highlight_target = if interaction.highlighted { 1.0 } else { 0.0 };
        self.highlight += (highlight_target - self.highlight) * (delta_seconds * 12.0).min(1.0);
        let charge_target = if interaction.using
            && active_tool == MultitoolKind::Sledge
            && self.progress[self.active_end] > 0.96
        {
            1.0
        } else {
            0.0
        };
        self.hammer_charge += (charge_target - self.hammer_charge) * (delta_seconds * 4.5).min(1.0);
        if interaction.activation == Activation::HammerStrike
            && active_tool == MultitoolKind::Sledge
        {
            self.hammer_swing_time = 0.0;
            self.hammer_charge = 0.0;
        } else {
            self.hammer_swing_time += delta_seconds;
        }
        if interaction.activation == Activation::MatterShot && active_tool == MultitoolKind::Matter
        {
            self.shot_time[self.active_end] = 0.0;
            self.matter_parity[self.active_end] = !self.matter_parity[self.active_end];
        }
        self.advance_heads(delta_seconds, interaction.using);
        self.advance_flip(delta_seconds);
    }

    fn advance_heads(&mut self, delta_seconds: f32, using: bool) {
        for end in 0..2 {
            let deploy_seconds = if self.tools[end] == MultitoolKind::Connector {
                CONNECTOR_DEPLOY_SECONDS
            } else {
                DEPLOY_SECONDS
            };
            self.progress[end] = move_towards(
                self.progress[end],
                self.targets[end],
                delta_seconds / deploy_seconds,
            );
            let use_now = using && end == self.active_end && self.progress[end] > 0.96;
            if self.tools[end] == MultitoolKind::Matter {
                if self.shot_time[end] < 9.0 {
                    self.shot_time[end] += delta_seconds;
                }
                for panel in 0_usize..6 {
                    let target = if panel.is_multiple_of(2) == self.matter_parity[end] {
                        1.0
                    } else {
                        0.0
                    };
                    self.panel_blend[end][panel] +=
                        (target - self.panel_blend[end][panel]) * (delta_seconds * 2.0).min(1.0);
                }
            }
            let connector_target = if use_now && self.tools[end] == MultitoolKind::Connector {
                1.0
            } else {
                0.0
            };
            self.connector_blend[end] +=
                (connector_target - self.connector_blend[end]) * (delta_seconds * 2.4).min(1.0);
            if connector_target > 0.5 {
                self.connector_wind[end] += delta_seconds * 3.8;
            }
        }
    }

    fn advance_flip(&mut self, delta_seconds: f32) {
        match self.flip_phase {
            FlipPhase::Idle => {}
            FlipPhase::Holstering => {
                if self.progress.iter().all(|progress| *progress <= 0.0) {
                    self.flip_phase = FlipPhase::Idle;
                }
            }
            FlipPhase::Stowing => {
                if self.progress[self.active_end] <= 0.0 && self.tuck >= 1.0 {
                    self.flip_phase = FlipPhase::Rotating;
                }
            }
            FlipPhase::Rotating => {
                self.spin = move_towards(
                    self.spin,
                    self.spin_target,
                    delta_seconds * PI / FLIP_SECONDS,
                );
                if self.spin >= self.spin_target {
                    self.active_end = self.destination_end;
                    self.targets[self.active_end] = 1.0;
                    self.flip_phase = FlipPhase::Deploying;
                }
            }
            FlipPhase::Deploying => {
                if self.progress[self.active_end] >= 1.0 && self.tuck <= 0.0 {
                    self.flip_phase = FlipPhase::Idle;
                }
            }
        }
    }
}

fn move_towards(current: f32, target: f32, maximum_delta: f32) -> f32 {
    current + (target - current).clamp(-maximum_delta, maximum_delta)
}

#[derive(Component)]
struct MultitoolRoot;

#[derive(Component)]
struct MultitoolPivot;

#[derive(Component, Clone, Copy)]
struct ToolCore {
    end: usize,
    tool: Option<MultitoolKind>,
}

#[derive(Component, Clone, Copy)]
struct PanelNode {
    end: usize,
    panel: usize,
    node: NodeKind,
}

#[derive(Clone, Copy)]
enum NodeKind {
    Carrier,
    Radial,
    Slide,
    Hinge,
    Spinner,
}

#[derive(Component, Clone, Copy)]
struct PanelDecoration {
    end: usize,
    panel: usize,
    decoration: Decoration,
}

#[derive(Clone, Copy)]
enum Decoration {
    Tip,
    Wing,
    Fins,
    Strike,
    InnerFace,
    Accent(MultitoolKind),
}

#[derive(Component, Clone, Copy)]
enum CoreAnimation {
    MatterPayload(usize),
    MatterRing(usize),
    WelderArc(usize),
}

type DecorationFilter = (
    Without<PanelNode>,
    Without<MultitoolPivot>,
    Without<MultitoolRoot>,
    Without<ToolCore>,
);
type PanelNodeFilter = (Without<MultitoolPivot>, Without<MultitoolRoot>);
type CoreAnimationFilter = (
    Without<PanelNode>,
    Without<PanelDecoration>,
    Without<MultitoolPivot>,
    Without<MultitoolRoot>,
);

#[derive(Clone)]
struct MultitoolMeshes {
    cube: Handle<Mesh>,
    cylinder: Handle<Mesh>,
    hex: Handle<Mesh>,
    sphere: Handle<Mesh>,
    grip_ring: Handle<Mesh>,
    core_ring: Handle<Mesh>,
    coil: Handle<Mesh>,
    matter_ring: Handle<Mesh>,
    taper: Handle<Mesh>,
}

#[derive(Resource, Clone)]
struct MultitoolMaterials {
    steel: Handle<StandardMaterial>,
    aluminium: Handle<StandardMaterial>,
    mid: Handle<StandardMaterial>,
    dark: Handle<StandardMaterial>,
    black: Handle<StandardMaterial>,
    bright: Handle<StandardMaterial>,
    grip: Handle<StandardMaterial>,
    accents: [Handle<StandardMaterial>; 4],
    payload: Handle<StandardMaterial>,
}

impl MultitoolMaterials {
    fn accent(&self, tool: MultitoolKind) -> Handle<StandardMaterial> {
        self.accents[tool as usize].clone()
    }
}

fn spawn(
    mut commands: Commands,
    camera: Single<(Entity, &Projection, &GeneratedEnvironmentMapLight), With<MainCamera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let model_meshes = multitool_meshes(&mut meshes);
    let model_materials = multitool_materials(&mut materials);
    let (camera_entity, projection, environment_map) = *camera;
    let mut viewmodel_projection = projection.clone();
    if let Projection::Perspective(perspective) = &mut viewmodel_projection {
        perspective.near = 0.01;
    }

    commands.entity(camera_entity).with_children(|camera| {
        camera.spawn((
            Name::new("Multitool viewmodel camera"),
            Camera3d::default(),
            Camera {
                order: 1,
                clear_color: ClearColorConfig::None,
                ..default()
            },
            viewmodel_projection,
            garage::EXPOSURE,
            Tonemapping::SomewhatBoringDisplayTransform,
            environment_map.clone(),
            RenderLayers::layer(VIEWMODEL_LAYER),
            FovCamera,
        ));
        camera.spawn((
            Name::new("Multitool key light"),
            PointLight {
                intensity: 1_800.0,
                range: 4.0,
                shadow_maps_enabled: false,
                ..default()
            },
            Transform::from_xyz(-0.8, 1.1, 0.4),
            RenderLayers::layer(VIEWMODEL_LAYER),
        ));
        camera
            .spawn((
                Name::new("Multitool viewmodel"),
                MultitoolRoot,
                Visibility::Inherited,
                carry_pose(),
            ))
            .with_children(|root| {
                root.spawn((
                    Name::new("Multitool pivot"),
                    MultitoolPivot,
                    Transform::default(),
                    Visibility::Inherited,
                ))
                .with_children(|pivot| {
                    spawn_shaft(pivot, &model_meshes, &model_materials);
                    spawn_end(pivot, 0, &model_meshes, &model_materials);
                    spawn_end(pivot, 1, &model_meshes, &model_materials);
                });
            });
    });
    commands.insert_resource(model_materials);
    commands.insert_resource(MultitoolState::default());
}

fn multitool_meshes(meshes: &mut Assets<Mesh>) -> MultitoolMeshes {
    MultitoolMeshes {
        cube: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        cylinder: meshes.add(Cylinder::new(1.0, 1.0)),
        hex: meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(6)),
        sphere: meshes.add(Sphere::new(1.0)),
        grip_ring: meshes.add(Torus::new(0.027, 0.0314)),
        core_ring: meshes.add(Torus::new(0.0275, 0.0355)),
        coil: meshes.add(Torus::new(0.008, 0.014)),
        matter_ring: meshes.add(Torus::new(0.0905, 0.1055)),
        taper: meshes.add(ConicalFrustum {
            radius_top: 0.008,
            radius_bottom: 0.022,
            height: 0.080,
        }),
    }
}

fn multitool_materials(materials: &mut Assets<StandardMaterial>) -> MultitoolMaterials {
    let flat = |color, roughness, metallic| StandardMaterial {
        base_color: color,
        perceptual_roughness: roughness,
        metallic,
        ..default()
    };
    let accent = |color: Color, roughness, metallic, intensity| StandardMaterial {
        base_color: color,
        perceptual_roughness: roughness,
        metallic,
        emissive: color.to_linear() * intensity,
        ..default()
    };
    let mint = Color::srgb_u8(47, 216, 180);
    MultitoolMaterials {
        steel: materials.add(flat(Color::srgb_u8(126, 139, 148), 0.42, 0.95)),
        aluminium: materials.add(flat(Color::srgb_u8(90, 107, 118), 0.50, 0.90)),
        mid: materials.add(flat(Color::srgb_u8(65, 79, 90), 0.55, 0.85)),
        dark: materials.add(flat(Color::srgb_u8(42, 52, 60), 0.62, 0.70)),
        black: materials.add(flat(Color::srgb_u8(27, 35, 42), 0.78, 0.30)),
        bright: materials.add(flat(Color::srgb_u8(157, 170, 178), 0.30, 1.0)),
        grip: materials.add(flat(Color::srgb_u8(20, 25, 30), 0.95, 0.05)),
        accents: [
            materials.add(accent(mint, 0.40, 0.30, 1.10)),
            materials.add(accent(Color::srgb_u8(226, 86, 90), 0.40, 0.30, 1.20)),
            materials.add(accent(Color::srgb_u8(47, 168, 216), 0.40, 0.30, 1.00)),
            materials.add(accent(Color::srgb_u8(201, 138, 52), 0.45, 0.80, 0.55)),
        ],
        payload: materials.add(StandardMaterial {
            base_color: mint.with_alpha(0.45),
            emissive: mint.to_linear() * 0.8,
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.40,
            metallic: 0.30,
            ..default()
        }),
    }
}

fn mesh_part(
    name: impl Into<String>,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    transform: Transform,
) -> impl Bundle {
    (
        Name::new(name.into()),
        Mesh3d(mesh),
        MeshMaterial3d(material),
        transform,
        RenderLayers::layer(VIEWMODEL_LAYER),
        NotShadowCaster,
    )
}

fn box_transform(position: Vec3, size: Vec3) -> Transform {
    Transform::from_translation(position).with_scale(size)
}

fn cylinder_transform(position: Vec3, radius: f32, height: f32) -> Transform {
    Transform::from_translation(position).with_scale(Vec3::new(radius, height, radius))
}

fn spawn_shaft(
    parent: &mut ChildSpawnerCommands,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent.spawn(mesh_part(
        "Shaft tube",
        meshes.cylinder.clone(),
        materials.mid.clone(),
        cylinder_transform(Vec3::ZERO, 0.0235, 1.240),
    ));
    parent.spawn(mesh_part(
        "Shaft flute",
        meshes.hex.clone(),
        materials.aluminium.clone(),
        cylinder_transform(Vec3::ZERO, 0.0255, 0.720),
    ));
    for direction in [-1.0, 1.0] {
        parent.spawn(mesh_part(
            format!("Shaft collar {direction}"),
            meshes.cylinder.clone(),
            materials.aluminium.clone(),
            cylinder_transform(Vec3::Y * direction * 0.415, 0.0305, 0.022),
        ));
        parent.spawn(mesh_part(
            format!("Shaft collar lip {direction}"),
            meshes.cylinder.clone(),
            materials.bright.clone(),
            cylinder_transform(Vec3::Y * direction * 0.404, 0.0325, 0.006),
        ));
        parent.spawn(mesh_part(
            format!("Prism root {direction}"),
            meshes.hex.clone(),
            materials.aluminium.clone(),
            cylinder_transform(Vec3::Y * direction * 0.584, 0.0345, 0.030),
        ));
    }
    parent.spawn(mesh_part(
        "Grip",
        meshes.cylinder.clone(),
        materials.grip.clone(),
        cylinder_transform(Vec3::ZERO, 0.0285, 0.360),
    ));
    for (index, y) in [
        -0.165, -0.132, -0.099, -0.066, -0.033, 0.0, 0.033, 0.066, 0.099, 0.132, 0.165,
    ]
    .into_iter()
    .enumerate()
    {
        parent.spawn(mesh_part(
            format!("Grip rib {index}"),
            meshes.grip_ring.clone(),
            materials.black.clone(),
            Transform::from_xyz(0.0, y, 0.0),
        ));
    }
    parent.spawn(mesh_part(
        "Grip band",
        meshes.cylinder.clone(),
        materials.aluminium.clone(),
        cylinder_transform(Vec3::Y * 0.196, 0.0295, 0.012),
    ));
}

fn spawn_end(
    parent: &mut ChildSpawnerCommands,
    end_index: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    let transform = if end_index == 0 {
        Transform::from_xyz(0.0, PRISM_ROOT, 0.0)
    } else {
        Transform::from_xyz(0.0, -PRISM_ROOT, 0.0).with_rotation(Quat::from_rotation_x(PI))
    };
    parent
        .spawn((
            Name::new(format!("Multitool end {end_index}")),
            transform,
            Visibility::Inherited,
        ))
        .with_children(|end| {
            spawn_core(end, end_index, meshes, materials);
            for panel in 0..6 {
                spawn_panel(end, end_index, panel, meshes, materials);
            }
        });
}

fn spawn_core(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn((
            Name::new(format!("End {end} core")),
            ToolCore { end, tool: None },
            Transform::default(),
            Visibility::Hidden,
        ))
        .with_children(|core| {
            core.spawn(mesh_part(
                "Core spine",
                meshes.cylinder.clone(),
                materials.dark.clone(),
                cylinder_transform(Vec3::Y * 0.135, 0.026, 0.270),
            ));
            for (index, y) in [0.045, 0.107, 0.169, 0.231].into_iter().enumerate() {
                core.spawn(mesh_part(
                    format!("Core ring {index}"),
                    meshes.core_ring.clone(),
                    materials.aluminium.clone(),
                    Transform::from_xyz(0.0, y, 0.0),
                ));
            }
            core.spawn(mesh_part(
                "Core base",
                meshes.hex.clone(),
                materials.aluminium.clone(),
                cylinder_transform(Vec3::Y * 0.012, 0.040, 0.024),
            ));
        });
    spawn_sledge_core(parent, end, meshes, materials);
    spawn_matter_core(parent, end, meshes, materials);
    spawn_welder_core(parent, end, meshes, materials);
    spawn_connector_core(parent, end, meshes, materials);
}

fn tool_core_bundle(end: usize, tool: MultitoolKind) -> impl Bundle {
    (
        Name::new(format!("End {end} {tool:?} core")),
        ToolCore {
            end,
            tool: Some(tool),
        },
        Transform::default(),
        Visibility::Hidden,
    )
}

fn spawn_sledge_core(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(tool_core_bundle(end, MultitoolKind::Sledge))
        .with_children(|core| {
            core.spawn(mesh_part(
                "Sledge anvil",
                meshes.cube.clone(),
                materials.black.clone(),
                box_transform(Vec3::Y * 0.160, Vec3::new(0.078, 0.200, 0.060)),
            ));
            core.spawn(mesh_part(
                "Sledge shoulder",
                meshes.cube.clone(),
                materials.aluminium.clone(),
                box_transform(Vec3::Y * 0.272, Vec3::new(0.092, 0.034, 0.062)),
            ));
            core.spawn(mesh_part(
                "Sledge crown",
                meshes.hex.clone(),
                materials.steel.clone(),
                cylinder_transform(Vec3::Y * 0.300, 0.031, 0.030),
            ));
            for (index, y) in [0.104, 0.162, 0.220].into_iter().enumerate() {
                core.spawn(mesh_part(
                    format!("Sledge rib {index}"),
                    meshes.cube.clone(),
                    materials.aluminium.clone(),
                    box_transform(Vec3::Y * y, Vec3::new(0.086, 0.012, 0.064)),
                ));
            }
            core.spawn(mesh_part(
                "Sledge heel",
                meshes.cube.clone(),
                materials.aluminium.clone(),
                box_transform(Vec3::Y * 0.062, Vec3::new(0.092, 0.028, 0.062)),
            ));
        });
}

fn spawn_matter_core(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(tool_core_bundle(end, MultitoolKind::Matter))
        .with_children(|core| {
            core.spawn(mesh_part(
                "Matter hopper",
                meshes.cube.clone(),
                materials.mid.clone(),
                box_transform(Vec3::Y * 0.085, Vec3::new(0.058, 0.090, 0.058)),
            ));
            core.spawn(mesh_part(
                "Matter window",
                meshes.cube.clone(),
                materials.accent(MultitoolKind::Matter),
                box_transform(Vec3::new(0.0, 0.085, 0.031), Vec3::new(0.030, 0.052, 0.004)),
            ));
            core.spawn(mesh_part(
                "Matter neck",
                meshes.cylinder.clone(),
                materials.aluminium.clone(),
                cylinder_transform(Vec3::Y * 0.165, 0.026, 0.070),
            ));
            core.spawn(mesh_part(
                "Matter nozzle",
                meshes.taper.clone(),
                materials.steel.clone(),
                Transform::from_xyz(0.0, 0.225, 0.0).with_scale(Vec3::new(1.0, 0.6875, 1.0)),
            ));
            core.spawn(mesh_part(
                "Matter lens",
                meshes.cylinder.clone(),
                materials.accent(MultitoolKind::Matter),
                cylinder_transform(Vec3::Y * 0.253, 0.012, 0.008),
            ));
            core.spawn((
                mesh_part(
                    "Matter ring",
                    meshes.matter_ring.clone(),
                    materials.accent(MultitoolKind::Matter),
                    Transform::from_xyz(0.0, 0.335, 0.0),
                ),
                CoreAnimation::MatterRing(end),
            ));
            core.spawn((
                mesh_part(
                    "Matter payload",
                    meshes.cube.clone(),
                    materials.payload.clone(),
                    box_transform(Vec3::Y * 0.335, Vec3::splat(0.072)),
                ),
                CoreAnimation::MatterPayload(end),
            ));
        });
}

fn spawn_welder_core(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(tool_core_bundle(end, MultitoolKind::Welder))
        .with_children(|core| {
            core.spawn(mesh_part(
                "Welder body",
                meshes.cube.clone(),
                materials.mid.clone(),
                box_transform(Vec3::Y * 0.095, Vec3::new(0.056, 0.120, 0.056)),
            ));
            core.spawn(mesh_part(
                "Welder band",
                meshes.cube.clone(),
                materials.accent(MultitoolKind::Welder),
                box_transform(Vec3::Y * 0.150, Vec3::new(0.060, 0.012, 0.060)),
            ));
            core.spawn(mesh_part(
                "Welder feed",
                meshes.cylinder.clone(),
                materials.aluminium.clone(),
                cylinder_transform(Vec3::Y * 0.205, 0.020, 0.090),
            ));
            for x in [-0.030, 0.030] {
                core.spawn(mesh_part(
                    format!("Welder gas feed {x}"),
                    meshes.cylinder.clone(),
                    materials.dark.clone(),
                    cylinder_transform(Vec3::new(x, 0.200, 0.020), 0.008, 0.070),
                ));
            }
            core.spawn((
                mesh_part(
                    "Welder arc",
                    meshes.sphere.clone(),
                    materials.accent(MultitoolKind::Welder),
                    Transform::from_xyz(0.0, 0.300, 0.0).with_scale(Vec3::splat(0.016)),
                ),
                CoreAnimation::WelderArc(end),
            ));
        });
}

fn spawn_connector_core(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(tool_core_bundle(end, MultitoolKind::Connector))
        .with_children(|core| {
            core.spawn(mesh_part(
                "Connector base",
                meshes.hex.clone(),
                materials.aluminium.clone(),
                cylinder_transform(Vec3::Y * 0.045, 0.032, 0.055),
            ));
            core.spawn(mesh_part(
                "Connector counter",
                meshes.cube.clone(),
                materials.accent(MultitoolKind::Connector),
                box_transform(Vec3::new(0.0, 0.062, 0.030), Vec3::new(0.044, 0.018, 0.010)),
            ));
            core.spawn(mesh_part(
                "Connector mast",
                meshes.cylinder.clone(),
                materials.mid.clone(),
                cylinder_transform(Vec3::Y * 0.185, 0.011, 0.230),
            ));
            core.spawn(mesh_part(
                "Connector collar",
                meshes.cylinder.clone(),
                materials.aluminium.clone(),
                cylinder_transform(Vec3::Y * 0.145, 0.020, 0.014),
            ));
            core.spawn(mesh_part(
                "Connector lamp",
                meshes.sphere.clone(),
                materials.accent(MultitoolKind::Connector),
                Transform::from_xyz(0.0, 0.312, 0.0).with_scale(Vec3::splat(0.017)),
            ));
        });
}

fn spawn_panel(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn((
            Name::new(format!("End {end} panel {panel} carrier")),
            PanelNode {
                end,
                panel,
                node: NodeKind::Carrier,
            },
            Transform::default(),
            Visibility::Inherited,
        ))
        .with_children(|carrier| {
            carrier
                .spawn(PanelNode {
                    end,
                    panel,
                    node: NodeKind::Radial,
                })
                .insert((Transform::default(), Visibility::Inherited))
                .with_children(|radial| {
                    radial
                        .spawn(PanelNode {
                            end,
                            panel,
                            node: NodeKind::Slide,
                        })
                        .insert((Transform::default(), Visibility::Inherited))
                        .with_children(|slide| {
                            slide
                                .spawn(PanelNode {
                                    end,
                                    panel,
                                    node: NodeKind::Hinge,
                                })
                                .insert((Transform::default(), Visibility::Inherited))
                                .with_children(|hinge| {
                                    hinge
                                        .spawn(PanelNode {
                                            end,
                                            panel,
                                            node: NodeKind::Spinner,
                                        })
                                        .insert((Transform::default(), Visibility::Inherited))
                                        .with_children(|spinner| {
                                            spawn_panel_shell(
                                                spinner, end, panel, meshes, materials,
                                            );
                                            spawn_panel_roles(
                                                spinner, end, panel, meshes, materials,
                                            );
                                        });
                                });
                        });
                });
        });
}

fn spawn_panel_shell(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent.spawn(mesh_part(
        format!("Panel {panel}"),
        meshes.cube.clone(),
        materials.aluminium.clone(),
        box_transform(
            Vec3::Y * (PANEL_LENGTH / 2.0),
            Vec3::new(PANEL_WIDTH, PANEL_LENGTH, PANEL_THICKNESS),
        ),
    ));
    parent.spawn(mesh_part(
        format!("Panel {panel} rail"),
        meshes.cube.clone(),
        materials.dark.clone(),
        box_transform(
            Vec3::new(0.0, PANEL_LENGTH / 2.0, -PANEL_THICKNESS / 2.0),
            Vec3::new(PANEL_WIDTH - 0.014, PANEL_LENGTH - 0.030, 0.004),
        ),
    ));
    for (index, (y, material)) in [
        (PANEL_LENGTH - 0.012, materials.bright.clone()),
        (0.012, materials.dark.clone()),
    ]
    .into_iter()
    .enumerate()
    {
        parent.spawn(mesh_part(
            format!("Panel {panel} lip {index}"),
            meshes.cube.clone(),
            material,
            box_transform(
                Vec3::Y * y,
                Vec3::new(PANEL_WIDTH - 0.003, 0.014, PANEL_THICKNESS - 0.002),
            ),
        ));
    }
    parent
        .spawn((
            Name::new(format!("Panel {panel} inner face")),
            PanelDecoration {
                end,
                panel,
                decoration: Decoration::InnerFace,
            },
            Transform::default(),
            Visibility::Inherited,
        ))
        .with_children(|face| {
            face.spawn(mesh_part(
                "Machined inner face",
                meshes.cube.clone(),
                materials.steel.clone(),
                box_transform(
                    Vec3::new(0.0, PANEL_LENGTH / 2.0, -PANEL_THICKNESS / 2.0 - 0.007),
                    Vec3::new(PANEL_WIDTH - 0.006, PANEL_LENGTH - 0.050, 0.004),
                ),
            ));
        });
    for tool in MultitoolKind::ALL {
        parent
            .spawn((
                Name::new(format!("Panel {panel} {tool:?} accents")),
                PanelDecoration {
                    end,
                    panel,
                    decoration: Decoration::Accent(tool),
                },
                Transform::default(),
                Visibility::Hidden,
            ))
            .with_children(|accent| {
                for y in [0.072, PANEL_LENGTH - 0.072] {
                    accent.spawn(mesh_part(
                        "Panel accent band",
                        meshes.cube.clone(),
                        materials.accent(tool),
                        box_transform(
                            Vec3::new(0.0, y, PANEL_THICKNESS / 2.0 + 0.002),
                            Vec3::new(PANEL_WIDTH - 0.010, 0.014, 0.004),
                        ),
                    ));
                }
            });
    }
}

fn spawn_panel_roles(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    spawn_tip(parent, end, panel, meshes, materials);
    spawn_wing(parent, end, panel, meshes, materials);
    spawn_fins(parent, end, panel, meshes, materials);
    spawn_strike(parent, end, panel, meshes, materials);
}

fn decoration_bundle(end: usize, panel: usize, decoration: Decoration) -> impl Bundle {
    (
        PanelDecoration {
            end,
            panel,
            decoration,
        },
        Visibility::Hidden,
        Transform::default(),
    )
}

fn spawn_tip(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(decoration_bundle(end, panel, Decoration::Tip))
        .insert((
            Name::new(format!("Panel {panel} tip")),
            Transform::from_xyz(0.0, PANEL_LENGTH, 0.0),
        ))
        .with_children(|tip| {
            tip.spawn(mesh_part(
                "Tip taper",
                meshes.taper.clone(),
                materials.aluminium.clone(),
                Transform::from_xyz(0.0, 0.040, 0.0),
            ));
            tip.spawn(mesh_part(
                "Tip coil",
                meshes.coil.clone(),
                materials.aluminium.clone(),
                Transform::from_xyz(0.0, 0.066, 0.0),
            ));
            for tool in [MultitoolKind::Matter, MultitoolKind::Connector] {
                tip.spawn((
                    mesh_part(
                        format!("{tool:?} tip emitter"),
                        meshes.sphere.clone(),
                        materials.accent(tool),
                        Transform::from_xyz(0.0, 0.086, 0.0).with_scale(Vec3::splat(0.010)),
                    ),
                    PanelDecoration {
                        end,
                        panel,
                        decoration: Decoration::Accent(tool),
                    },
                ));
            }
        });
}

fn spawn_wing(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(decoration_bundle(end, panel, Decoration::Wing))
        .insert(Name::new(format!("Panel {panel} blast shield")))
        .with_children(|wing| {
            for (name, position, size, material) in [
                (
                    "plate",
                    Vec3::new(0.0, 0.210, 0.010),
                    Vec3::new(0.150, 0.150, 0.007),
                    materials.aluminium.clone(),
                ),
                (
                    "edge",
                    Vec3::new(0.0, 0.285, 0.010),
                    Vec3::new(0.150, 0.012, 0.014),
                    materials.dark.clone(),
                ),
                (
                    "scorch",
                    Vec3::new(0.0, 0.250, 0.015),
                    Vec3::new(0.130, 0.010, 0.004),
                    materials.accent(MultitoolKind::Welder),
                ),
                (
                    "strut",
                    Vec3::new(0.0, 0.165, 0.012),
                    Vec3::new(0.010, 0.090, 0.030),
                    materials.dark.clone(),
                ),
            ] {
                wing.spawn(mesh_part(
                    format!("Shield {name}"),
                    meshes.cube.clone(),
                    material,
                    box_transform(position, size),
                ));
            }
        });
}

fn spawn_fins(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(decoration_bundle(end, panel, Decoration::Fins))
        .insert(Name::new(format!("Panel {panel} heatsink")))
        .with_children(|fins| {
            for (index, y) in [0.050, 0.090, 0.130, 0.170, 0.210, 0.250]
                .into_iter()
                .enumerate()
            {
                fins.spawn(mesh_part(
                    format!("Heatsink fin {index}"),
                    meshes.cube.clone(),
                    materials.aluminium.clone(),
                    box_transform(
                        Vec3::new(0.0, y, 0.016),
                        Vec3::new(PANEL_WIDTH + 0.014, 0.008, 0.026),
                    ),
                ));
            }
            fins.spawn(mesh_part(
                "Heatsink root",
                meshes.cube.clone(),
                materials.accent(MultitoolKind::Welder),
                box_transform(Vec3::new(0.0, 0.150, 0.006), Vec3::new(0.058, 0.230, 0.008)),
            ));
        });
}

fn spawn_strike(
    parent: &mut ChildSpawnerCommands,
    end: usize,
    panel: usize,
    meshes: &MultitoolMeshes,
    materials: &MultitoolMaterials,
) {
    parent
        .spawn(decoration_bundle(end, panel, Decoration::Strike))
        .insert(Name::new(format!("Panel {panel} strike face")))
        .with_children(|strike| {
            for (name, position, size, material) in [
                (
                    "pad",
                    Vec3::new(0.0, 0.140, -0.0225),
                    Vec3::new(0.096, 0.266, 0.034),
                    materials.steel.clone(),
                ),
                (
                    "face",
                    Vec3::new(0.0, 0.140, -0.0425),
                    Vec3::new(0.102, 0.240, 0.010),
                    materials.bright.clone(),
                ),
                (
                    "band a",
                    Vec3::new(0.0, 0.232, -0.0315),
                    Vec3::new(0.108, 0.018, 0.050),
                    materials.accent(MultitoolKind::Sledge),
                ),
                (
                    "band b",
                    Vec3::new(0.0, 0.048, -0.0315),
                    Vec3::new(0.108, 0.018, 0.050),
                    materials.accent(MultitoolKind::Sledge),
                ),
            ] {
                strike.spawn(mesh_part(
                    format!("Strike {name}"),
                    meshes.cube.clone(),
                    material,
                    box_transform(position, size),
                ));
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn update(
    time: Res<Time>,
    input: MultitoolInput,
    model_materials: Res<MultitoolMaterials>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera: Single<&PlayerCamera, With<MainCamera>>,
    mut state: ResMut<MultitoolState>,
    mut root: Single<(&mut Visibility, &mut Transform), With<MultitoolRoot>>,
    mut pivot: Single<&mut Transform, (With<MultitoolPivot>, Without<MultitoolRoot>)>,
    mut cores: Query<(&ToolCore, &mut Visibility), Without<MultitoolRoot>>,
    mut panel_nodes: Query<(&PanelNode, &mut Transform), PanelNodeFilter>,
    mut decorations: Query<
        (&PanelDecoration, &mut Visibility, Option<&mut Transform>),
        DecorationFilter,
    >,
    mut core_animations: Query<(&CoreAnimation, &mut Transform), CoreAnimationFilter>,
) {
    let selected = input.selection.tool.map(MultitoolKind::from_main_tool);
    state.set_selection(selected);
    let interaction = interaction_frame(&input, selected);
    state.advance(time.delta_secs(), interaction);
    update_accent_lighting(&model_materials, &mut materials, &state, interaction);
    *root.0 = if camera.current_pullback() < AVATAR_HIDDEN_PULLBACK {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    *root.1 = viewmodel_transform(&state);
    pivot.rotation = Quat::from_rotation_z(state.spin);

    for (core, mut visibility) in &mut cores {
        let run_down = deployment_beats(state.progress[core.end]).1;
        *visibility = if core.tool.map_or(run_down > 0.02, |tool| {
            run_down > 0.15 && state.tools[core.end] == tool
        }) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    for (node, mut transform) in &mut panel_nodes {
        animate_panel(&state, node, &mut transform, interaction.using);
    }
    for (decoration, mut visibility, transform) in &mut decorations {
        *visibility = decoration_visibility(&state, decoration);
        if let Some(mut transform) = transform
            && matches!(
                decoration.decoration,
                Decoration::Tip | Decoration::Wing | Decoration::Strike
            )
        {
            let deploy = deployment_beats(state.progress[decoration.end]).2;
            transform.scale = Vec3::splat(deploy.max(0.001));
        }
    }
    for (animation, mut transform) in &mut core_animations {
        match *animation {
            CoreAnimation::MatterPayload(end) => {
                let fold = deployment_beats(state.progress[end]).2;
                transform.rotate_y(time.delta_secs() * 0.6);
                transform.rotate_x(time.delta_secs() * 0.25);
                transform.scale = Vec3::splat(0.072 * (0.4 + 0.6 * fold));
            }
            CoreAnimation::MatterRing(end) => {
                let fold = deployment_beats(state.progress[end]).2;
                transform.rotate_y(time.delta_secs() * 0.4);
                transform.scale = Vec3::splat(fold.max(0.001));
            }
            CoreAnimation::WelderArc(end) => {
                let fold = deployment_beats(state.progress[end]).2;
                let hot = if interaction.using && end == state.active_end {
                    1.0
                } else {
                    0.0
                };
                let hover = if end == state.active_end {
                    state.highlight
                } else {
                    0.0
                };
                let pulse = 0.7 + 0.3 * (state.elapsed * 25.0).sin() + hover * 0.5 + hot * 0.9;
                transform.scale = Vec3::splat(0.016 * pulse * fold.max(0.001));
            }
        }
    }
}

fn interaction_frame(
    input: &MultitoolInput<'_>,
    selected: Option<MultitoolKind>,
) -> InteractionFrame {
    let input_active = input.player.world_input_active();
    let primary_pressed = input_active && input.actions.pressed(GameAction::Primary);
    let weldable =
        selected == Some(MultitoolKind::Welder) && hovered_part(input.editor.hovered).is_some();
    let connectable = selected == Some(MultitoolKind::Connector)
        && wire_end_under_cursor(&input.graph.0, &input.editor).is_some();
    match selected {
        Some(MultitoolKind::Matter) => InteractionFrame {
            using: primary_pressed
                || (input_active && input.actions.pressed(GameAction::Secondary)),
            activation: if input_active
                && (input.actions.just_released(GameAction::Primary)
                    || input.actions.just_released(GameAction::Secondary))
            {
                Activation::MatterShot
            } else {
                Activation::None
            },
            ..default()
        },
        Some(MultitoolKind::Welder) => InteractionFrame {
            using: weldable && primary_pressed,
            highlighted: weldable,
            ..default()
        },
        Some(MultitoolKind::Connector) => InteractionFrame {
            using: input.editor.wire_drag.is_some() || (connectable && primary_pressed),
            highlighted: connectable,
            ..default()
        },
        Some(MultitoolKind::Sledge) => InteractionFrame {
            using: input.hammer.charging.is_some(),
            activation: if input_active
                && input.actions.just_released(GameAction::Primary)
                && input.hammer.pending.is_some()
            {
                Activation::HammerStrike
            } else {
                Activation::None
            },
            ..default()
        },
        None => InteractionFrame::default(),
    }
}

fn animate_panel(
    state: &MultitoolState,
    panel: &PanelNode,
    transform: &mut Transform,
    using: bool,
) {
    let tool = state.tools[panel.end];
    let mut fold = Fold::for_panel(tool, panel.panel);
    let (unlatch, run_down, deploy) = deployment_beats(state.progress[panel.end]);
    let use_now = using && panel.end == state.active_end && state.progress[panel.end] > 0.96;
    let mut extra_yaw = 0.0;
    if tool == MultitoolKind::Matter {
        let blend = state.panel_blend[panel.end][panel.panel];
        fold.radial = 0.004 * blend;
        fold.slide = -0.255 + 0.200 * blend;
        fold.hinge = 0.04 + 0.16 * blend;
        if state.shot_time[panel.end] < 0.26 && blend > 0.5 {
            fold.hinge += 0.34 * (1.0 - state.shot_time[panel.end] / 0.26);
        }
    }
    if tool == MultitoolKind::Welder && use_now {
        let phase = PANEL_PHASES[panel.panel];
        fold.hinge += (state.elapsed * 61.0 + phase * 1.7).sin() * 0.013;
        fold.radial += (state.elapsed * 89.0 + phase * 2.3).sin() * 0.0016;
    }
    if tool == MultitoolKind::Connector {
        extra_yaw = state.connector_wind[panel.end];
        fold.hinge *= 1.0 - 0.42 * state.connector_blend[panel.end];
    }
    match panel.node {
        NodeKind::Carrier => {
            transform.rotation = Quat::from_rotation_y(
                PANEL_PHASES[panel.panel] * TAU / 6.0 + (fold.yaw + extra_yaw) * deploy,
            );
        }
        NodeKind::Radial => {
            transform.translation.z =
                HEX_APOTHEM + 0.0055 + 0.008 * unlatch + (fold.radial - 0.008) * deploy;
        }
        NodeKind::Slide => {
            transform.translation.y = -0.100 * run_down + (fold.slide + 0.100) * deploy;
        }
        NodeKind::Hinge => {
            let idle = if tool == MultitoolKind::Connector {
                (state.elapsed * 1.5 + PANEL_PHASES[panel.panel] * 1.05).sin()
                    * 0.05
                    * deploy
                    * (1.0 - state.connector_blend[panel.end])
            } else {
                0.0
            };
            transform.rotation = Quat::from_rotation_x(fold.hinge * deploy + idle);
        }
        NodeKind::Spinner => {
            transform.rotation = Quat::from_rotation_y(fold.spin * deploy);
            transform.scale.z = 1.0 + (fold.thickness - 1.0) * deploy;
        }
    }
}

fn decoration_visibility(state: &MultitoolState, marker: &PanelDecoration) -> Visibility {
    let tool = state.tools[marker.end];
    let role = Fold::for_panel(tool, marker.panel).role;
    let (unlatch, _, deploy) = deployment_beats(state.progress[marker.end]);
    let visible = match marker.decoration {
        Decoration::Tip => {
            (tool == MultitoolKind::Matter || matches!(role, PanelRole::Finger | PanelRole::Rib))
                && deploy > 0.001
        }
        Decoration::Wing => role == PanelRole::Shield && deploy > 0.001,
        Decoration::Fins => role == PanelRole::Fin && deploy > 0.001,
        Decoration::Strike => role == PanelRole::Strike && deploy > 0.001,
        Decoration::InnerFace => !matches!(role, PanelRole::Lamina | PanelRole::Strike),
        Decoration::Accent(accent) => {
            accent == tool
                && unlatch > 0.04
                && !matches!(role, PanelRole::Lamina | PanelRole::Strike)
        }
    };
    if visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    }
}

fn deployment_beats(progress: f32) -> (f32, f32, f32) {
    (
        ease((progress / 0.18).clamp(0.0, 1.0)),
        ease(((progress - 0.18) / 0.32).clamp(0.0, 1.0)),
        ease(((progress - 0.55) / 0.45).clamp(0.0, 1.0)),
    )
}

fn ease(value: f32) -> f32 {
    if value < 0.5 {
        4.0 * value * value * value
    } else {
        1.0 - (-2.0 * value + 2.0).powi(3) / 2.0
    }
}

fn update_accent_lighting(
    model_materials: &MultitoolMaterials,
    materials: &mut Assets<StandardMaterial>,
    state: &MultitoolState,
    interaction: InteractionFrame,
) {
    for tool in MultitoolKind::ALL {
        let color = accent_color(tool);
        let active =
            state.tools[state.active_end] == tool && state.progress[state.active_end] > 0.9;
        let hover = if active { state.highlight } else { 0.0 };
        let use_boost = if active && interaction.using {
            1.0
        } else {
            0.0
        };
        let pulse = match tool {
            MultitoolKind::Matter if active && state.shot_time[state.active_end] < 0.26 => 2.5,
            MultitoolKind::Sledge if active && state.hammer_swing_time < 0.34 => 1.5,
            _ => 0.0,
        };
        if let Some(mut material) = materials.get_mut(&model_materials.accents[tool as usize]) {
            material.emissive = color.to_linear()
                * (accent_intensity(tool) + hover * 4.0 + use_boost * 2.0 + pulse);
        }
    }
}

fn accent_color(tool: MultitoolKind) -> Color {
    match tool {
        MultitoolKind::Matter => Color::srgb_u8(47, 216, 180),
        MultitoolKind::Welder => Color::srgb_u8(226, 86, 90),
        MultitoolKind::Connector => Color::srgb_u8(47, 168, 216),
        MultitoolKind::Sledge => Color::srgb_u8(201, 138, 52),
    }
}

const fn accent_intensity(tool: MultitoolKind) -> f32 {
    match tool {
        MultitoolKind::Matter => 1.10,
        MultitoolKind::Welder => 1.20,
        MultitoolKind::Connector => 1.00,
        MultitoolKind::Sledge => 0.55,
    }
}

fn blend_pose(from: Transform, to: Transform, amount: f32) -> Transform {
    Transform {
        translation: from.translation.lerp(to.translation, amount),
        rotation: from.rotation.slerp(to.rotation, amount),
        scale: from.scale.lerp(to.scale, amount),
    }
}

fn carry_pose() -> Transform {
    Transform::from_translation(CARRY_TRANSLATION)
        .with_rotation(Quat::from_euler(EulerRot::XYZ, -0.10, 0.15, -0.13))
        .with_scale(Vec3::splat(VIEWMODEL_SCALE))
}

fn ready_pose() -> Transform {
    Transform::from_translation(READY_TRANSLATION)
        .with_rotation(Quat::from_euler(EulerRot::XYZ, -0.08, 0.12, -1.31))
        .with_scale(Vec3::splat(VIEWMODEL_SCALE))
}

fn aim_pose() -> Transform {
    let direction = Vec3::new(0.50, 0.12, -0.858).normalize();
    Transform::from_translation(AIM_TRANSLATION)
        .with_rotation(Quat::from_rotation_arc(Vec3::Y, direction) * Quat::from_rotation_y(0.12))
        .with_scale(Vec3::splat(ACTIVE_SCALE))
}

fn hammer_swing_angle(time: f32) -> f32 {
    if time < 0.11 {
        0.30 + (-0.72 - 0.30) * ease((time / 0.11).clamp(0.0, 1.0))
    } else if time < 0.34 {
        -0.72 * (1.0 - ease(((time - 0.11) / 0.23).clamp(0.0, 1.0)))
    } else {
        0.0
    }
}

fn viewmodel_transform(state: &MultitoolState) -> Transform {
    let mut transform = blend_pose(carry_pose(), ready_pose(), ease(state.ready));
    if state.tuck > 0.0 {
        let tucked = Transform::from_translation(TUCKED_TRANSLATION)
            .with_rotation(Quat::from_euler(EulerRot::XYZ, -0.12, 0.15, -0.34))
            .with_scale(Vec3::splat(VIEWMODEL_SCALE));
        transform = blend_pose(transform, tucked, ease(state.tuck));
    }
    transform = blend_pose(transform, aim_pose(), ease(state.aim));

    let active_tool = state.tools[state.active_end];
    if active_tool == MultitoolKind::Matter && state.shot_time[state.active_end] < 0.26 {
        let recoil = (1.0 - state.shot_time[state.active_end] / 0.26).powi(2);
        transform.translation += Vec3::new(0.025, -0.018, 0.10) * recoil;
    }
    if active_tool == MultitoolKind::Sledge {
        let angle = 0.30 * state.hammer_charge + hammer_swing_angle(state.hammer_swing_time);
        transform.rotation *= Quat::from_rotation_x(angle);
        let strike = (-hammer_swing_angle(state.hammer_swing_time)).max(0.0) / 0.72;
        transform.translation += Vec3::NEG_Z * 0.10 * strike;
    }
    transform
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn main_tools_map_to_the_four_authored_folds() {
        assert_eq!(
            MainTool::ALL.map(MultitoolKind::from_main_tool),
            [
                MultitoolKind::Matter,
                MultitoolKind::Welder,
                MultitoolKind::Connector,
                MultitoolKind::Sledge,
            ]
        );
    }

    #[test]
    fn changing_tool_stows_before_flipping() {
        let mut state = MultitoolState::default();
        state.progress[0] = 1.0;
        state.select(MultitoolKind::Welder);
        assert_eq!(state.active_end, 0);
        assert_eq!(state.destination_end, 1);
        assert_eq!(state.tools, [MultitoolKind::Matter, MultitoolKind::Welder]);
        assert_eq!(state.targets, [0.0, 0.0]);
        assert_eq!(state.spin_target, PI);
        assert_eq!(state.flip_phase, FlipPhase::Stowing);
    }

    #[test]
    fn repeated_selection_does_not_restart_the_flip() {
        let mut state = MultitoolState::default();
        state.select(MultitoolKind::Matter);
        assert_eq!(state.active_end, 0);
        assert_eq!(state.spin_target, 0.0);
    }

    #[test]
    fn deployment_uses_three_separate_beats() {
        assert_eq!(deployment_beats(0.0), (0.0, 0.0, 0.0));
        let after_unlatch = deployment_beats(0.18);
        assert_eq!(after_unlatch.0, 1.0);
        assert_eq!(after_unlatch.1, 0.0);
        assert_eq!(after_unlatch.2, 0.0);
        assert_eq!(deployment_beats(1.0), (1.0, 1.0, 1.0));
    }

    #[test]
    fn rapid_tool_changes_replace_the_pending_head_without_restarting() {
        let mut state = MultitoolState::default();
        state.select(MultitoolKind::Welder);
        state.select(MultitoolKind::Connector);
        assert_eq!(state.active_end, 0);
        assert_eq!(state.destination_end, 1);
        assert_eq!(state.tools[1], MultitoolKind::Connector);
        assert_eq!(state.targets, [0.0, 0.0]);
        assert_eq!(state.spin_target, PI);
    }

    #[test]
    fn changing_again_during_deployment_starts_a_new_stow_and_flip() {
        let mut state = MultitoolState {
            active_end: 1,
            destination_end: 1,
            flip_phase: FlipPhase::Deploying,
            tools: [MultitoolKind::Matter, MultitoolKind::Welder],
            progress: [0.0, 0.5],
            targets: [0.0, 1.0],
            spin: PI,
            spin_target: PI,
            ..default()
        };
        state.select(MultitoolKind::Connector);
        assert_eq!(state.destination_end, 0);
        assert_eq!(state.tools[0], MultitoolKind::Connector);
        assert_eq!(state.targets, [0.0, 0.0]);
        assert_eq!(state.spin_target, PI * 2.0);
        assert_eq!(state.flip_phase, FlipPhase::Stowing);
    }

    #[test]
    fn pending_head_stays_stowed_until_rotation_finishes() {
        let mut state = MultitoolState::default();
        state.progress[0] = 1.0;
        state.select(MultitoolKind::Welder);

        while state.flip_phase == FlipPhase::Stowing {
            state.advance(0.05, InteractionFrame::default());
        }
        assert_eq!(state.flip_phase, FlipPhase::Rotating);
        assert_eq!(state.progress[1], 0.0);
        assert_eq!(state.spin, 0.0);

        while state.flip_phase == FlipPhase::Rotating {
            assert_eq!(state.progress[1], 0.0);
            state.advance(0.05, InteractionFrame::default());
        }
        assert_eq!(state.flip_phase, FlipPhase::Deploying);
        assert_eq!(state.spin, PI);
        assert_eq!(state.progress[1], 0.0);

        state.advance(0.05, InteractionFrame::default());
        assert!(state.progress[1] > 0.0);
    }

    #[test]
    fn rotation_happens_in_the_tucked_side_pose() {
        let mut pose = MultitoolState {
            ready: 1.0,
            ..default()
        };
        let held = viewmodel_transform(&pose).translation;
        pose.tuck = 1.0;
        let tucked = viewmodel_transform(&pose).translation;
        assert!(tucked.x > held.x);
        assert!(tucked.y < held.y);
        assert!(tucked.z > held.z);

        let mut state = MultitoolState::default();
        state.progress[0] = 1.0;
        state.select(MultitoolKind::Welder);
        state.advance(0.05, InteractionFrame::default());
        assert_eq!(state.spin, 0.0);
    }

    #[test]
    fn clearing_the_tool_closes_both_heads_before_lowering_to_carry() {
        let mut state = MultitoolState {
            progress: [1.0, 0.0],
            ready: 1.0,
            ..default()
        };
        state.set_selection(None);
        assert_eq!(state.targets, [0.0, 0.0]);
        assert_eq!(state.flip_phase, FlipPhase::Holstering);

        state.advance(0.05, InteractionFrame::default());
        assert_eq!(state.ready, 1.0);
        while state.progress[0] > 0.0 {
            state.advance(0.05, InteractionFrame::default());
        }
        while state.ready > 0.0 {
            state.advance(0.05, InteractionFrame::default());
        }
        assert_eq!(state.ready, 0.0);
        assert_eq!(state.progress, [0.0, 0.0]);
    }

    #[test]
    fn holstering_during_rotation_finishes_on_a_coherent_end() {
        let mut state = MultitoolState {
            active_end: 0,
            destination_end: 1,
            flip_phase: FlipPhase::Rotating,
            spin: PI * 0.4,
            spin_target: PI,
            ..default()
        };
        state.holster();
        assert_eq!(state.active_end, 1);
        assert_eq!(state.spin, PI);
        assert_eq!(state.flip_phase, FlipPhase::Holstering);
    }

    #[test]
    fn carry_is_vertical_and_ready_points_the_active_head_right() {
        let carry_direction = carry_pose().rotation * Vec3::Y;
        let ready_direction = ready_pose().rotation * Vec3::Y;
        assert!(carry_direction.y > 0.9);
        assert!(ready_direction.x > 0.9);
        assert!(ready_direction.y > 0.0);
    }

    #[test]
    fn staff_is_large_and_low_in_both_resting_poses() {
        let carry = carry_pose();
        let ready = ready_pose();
        assert_eq!(carry.scale, Vec3::splat(VIEWMODEL_SCALE));
        assert_eq!(ready.scale, Vec3::splat(VIEWMODEL_SCALE));
        assert!(carry.translation.y < -1.15);
        assert!(ready.translation.y < -0.60);
    }

    #[test]
    fn active_pose_keeps_the_large_head_in_the_lower_right() {
        let pose = aim_pose();
        let direction = pose.rotation * Vec3::Y;
        assert!(direction.x > 0.45);
        assert!(direction.y > 0.0);
        assert!(direction.z < -0.80);
        assert_eq!(pose.scale, Vec3::splat(ACTIVE_SCALE));
        assert!(pose.translation.x > 0.60);
        assert!(pose.translation.y < -0.70);
    }

    #[test]
    fn matter_head_fires_once_on_release_instead_of_looping_while_held() {
        let mut state = MultitoolState {
            progress: [1.0, 0.0],
            ready: 1.0,
            ..default()
        };
        let parity = state.matter_parity[0];
        state.advance(
            0.05,
            InteractionFrame {
                using: true,
                ..default()
            },
        );
        assert_eq!(state.matter_parity[0], parity);
        assert_eq!(state.shot_time[0], 9.0);

        state.advance(
            0.05,
            InteractionFrame {
                activation: Activation::MatterShot,
                ..default()
            },
        );
        assert_ne!(state.matter_parity[0], parity);
        assert!(state.shot_time[0] < 0.1);
    }

    #[test]
    fn connector_head_deploys_more_eagerly_than_other_heads() {
        let mut connector = MultitoolState {
            tools: [MultitoolKind::Connector, MultitoolKind::Matter],
            progress: [0.0; 2],
            targets: [1.0, 0.0],
            ..default()
        };
        let mut matter = MultitoolState::default();
        connector.advance(0.05, InteractionFrame::default());
        matter.advance(0.05, InteractionFrame::default());
        assert!(connector.progress[0] > matter.progress[0]);
    }

    #[test]
    fn hammer_strike_swings_forward_and_returns_to_ready() {
        assert!(hammer_swing_angle(0.11) < -0.7);
        assert!(hammer_swing_angle(0.20) < 0.0);
        assert_eq!(hammer_swing_angle(0.34), 0.0);
    }

    #[test]
    fn hammer_strike_moves_towards_the_target_in_depth() {
        let mut state = MultitoolState {
            active_end: 0,
            tools: [MultitoolKind::Sledge, MultitoolKind::Matter],
            progress: [1.0, 0.0],
            ready: 1.0,
            hammer_swing_time: 9.0,
            ..default()
        };
        let idle = viewmodel_transform(&state).rotation * Vec3::Y;
        state.hammer_swing_time = 0.11;
        let strike = viewmodel_transform(&state).rotation * Vec3::Y;
        let screen_travel = Vec2::new(strike.x - idle.x, strike.y - idle.y).length();
        let depth_travel = (strike.z - idle.z).abs();
        assert!(strike.z < idle.z);
        assert!(depth_travel > screen_travel);
    }
}
