//! The player's visible body.

use crate::camera;
use crate::camera::{MainCamera, PlayerCamera, PlayerState};
use crate::editor::state::EditorGraph;
use crate::seat::seat_world_pose;
use crate::simulation::state::AppSimulation;
use bevy::prelude::{
    Alpha, AlphaMode, Assets, Color, Commands, Component, Cuboid, Handle, Mesh, Mesh3d,
    MeshMaterial3d, Name, Quat, Query, Res, ResMut, Resource, Single, StandardMaterial, Transform,
    Vec3, Visibility, With, Without, default,
};

#[derive(Component)]
pub(crate) struct PlayerAvatar;

#[derive(Component, Clone)]
pub(crate) struct AvatarPart {
    pub(crate) standing: Transform,
    pub(crate) crouched: Transform,
    pub(crate) seated: Transform,
}

#[derive(Resource)]
pub(crate) struct AvatarMaterials {
    pub(crate) clothing: Handle<StandardMaterial>,
    pub(crate) head: Handle<StandardMaterial>,
    pub(crate) boots: Handle<StandardMaterial>,
}

pub(crate) fn avatar_material(color: Color) -> StandardMaterial {
    StandardMaterial {
        base_color: color.with_alpha(0.0),
        perceptual_roughness: 0.92,
        alpha_mode: AlphaMode::Blend,
        ..default()
    }
}

pub(crate) fn sync_avatar_materials(
    materials: &mut Assets<StandardMaterial>,
    handles: &AvatarMaterials,
    alpha: f32,
) {
    // Camera fade alpha is clamped to [0, 1]. Fully visible avatars belong in
    // the opaque pass; intermediate alpha still needs normal blending.
    let alpha_mode = if alpha >= 1.0 {
        AlphaMode::Opaque
    } else {
        AlphaMode::Blend
    };
    for handle in [&handles.clothing, &handles.head, &handles.boots] {
        if let Some(mut material) = materials.get_mut(handle) {
            let base_color = material.base_color.with_alpha(alpha);
            // AssetMut only emits Modified when mutably dereferenced. Reading
            // first avoids rebuilding unchanged materials every frame.
            if material.base_color != base_color || material.alpha_mode != alpha_mode {
                material.base_color = base_color;
                material.alpha_mode = alpha_mode;
            }
        }
    }
}

pub(crate) fn avatar_pose(position: Vec3, scale: Vec3, rotation: Quat) -> Transform {
    Transform::from_translation(position)
        .with_rotation(rotation)
        .with_scale(scale)
}

/// Blends two authored poses so the stance follows the capsule's height instead of
/// snapping between standing and crouched.
pub(crate) fn blend_pose(from: Transform, to: Transform, factor: f32) -> Transform {
    let factor = factor.clamp(0.0, 1.0);
    Transform {
        translation: from.translation.lerp(to.translation, factor),
        rotation: from.rotation.slerp(to.rotation, factor),
        scale: from.scale.lerp(to.scale, factor),
    }
}

#[expect(clippy::too_many_lines)]
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
                Vec3::new(0.00, 0.85, 0.05),
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
                Vec3::new(0.00, 0.60, 0.02),
                Vec3::new(0.42, 0.46, 0.24),
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
                Vec3::new(-0.29, 0.56, 0.08),
                Vec3::new(0.13, 0.46, 0.13),
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
                Vec3::new(0.29, 0.56, 0.08),
                Vec3::new(0.13, 0.46, 0.13),
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
                Vec3::new(-0.12, 0.26, 0.10),
                Vec3::new(0.17, 0.40, 0.18),
                Quat::from_rotation_x(-0.35),
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
                Vec3::new(0.12, 0.26, 0.10),
                Vec3::new(0.17, 0.40, 0.18),
                Quat::from_rotation_x(-0.35),
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
                Vec3::new(-0.12, 0.10, 0.10),
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
                Vec3::new(0.12, 0.10, 0.10),
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
            for (name, material, standing, crouched, seated) in parts {
                avatar.spawn((
                    Name::new(name),
                    Mesh3d(cube.clone()),
                    MeshMaterial3d(material),
                    standing,
                    AvatarPart {
                        standing,
                        crouched,
                        seated,
                    },
                ));
            }
        });
    commands.insert_resource(avatar_materials);
}

#[expect(clippy::too_many_arguments)]
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
    sync_avatar_materials(&mut materials, &handles, alpha);
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
            blend_pose(part.standing, part.crouched, player.crouch)
        };
    }
}
