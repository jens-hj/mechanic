//! Moons drawn as lit spheres on camera-facing quads, far behind the world.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver, light_consts::lux::RAW_SUNLIGHT};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;
use mechanic_world::{MAX_SYSTEM_STARS, SkyMoon, SkyStar};

use super::SkyState;
use crate::camera::MainCamera;

const MOON_SHADER: &str = "shaders/moon.wgsl";

/// Distance at which moons are drawn, in metres: beyond the finite world,
/// inside the atmosphere's aerial-perspective range so daylight veils them.
const DRAW_METRES: f32 = 30_000.0;

/// Share of starlight the planet's day side reflects onto a moon's near side.
const PLANET_ALBEDO: f32 = 0.3;

/// The disk of the moon at this index in the system.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MoonDisk(pub(crate) usize);

/// Shading inputs of one moon, refreshed every frame.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub(crate) struct MoonMaterial {
    #[uniform(0)]
    pub(super) moon: MoonUniform,
}

/// Mirrors `Moon` in `shaders/moon.wgsl`. Directions are in the quad's frame:
/// x right, y up, z towards the camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub(super) struct MoonUniform {
    /// Rows map a quad-frame normal onto the moon's own axes.
    pub(super) surface_x: Vec4,
    pub(super) surface_y: Vec4,
    pub(super) surface_z: Vec4,
    /// Direction to each star; `w` is its illuminance on the moon in lux.
    pub(super) light_direction: [Vec4; MAX_SYSTEM_STARS],
    /// Linear colour of each star's light.
    pub(super) light_colour: [Vec4; MAX_SYSTEM_STARS],
    /// Linear reflectance; `w` selects the surface markings.
    pub(super) reflectance: Vec4,
    /// `x`: illuminance the planet's day side casts on the near side, in lux.
    pub(super) planetshine: Vec4,
}

impl Material for MoonMaterial {
    fn fragment_shader() -> ShaderRef {
        MOON_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::AlphaToCoverage
    }
}

#[derive(Resource)]
pub(super) struct MoonQuad(Handle<Mesh>);

pub(super) fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    commands.insert_resource(MoonQuad(meshes.add(Rectangle::new(2.0, 2.0))));
}

/// Orient a quad at `direction` so its +z faces the viewer.
pub(super) fn facing(direction: Vec3) -> Quat {
    let up = if direction.y.abs() > 0.999 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    Transform::default().looking_to(direction, up).rotation
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the pattern seed only needs to vary"
)]
pub(super) fn uniform(
    moon: &SkyMoon,
    stars: &[SkyStar],
    orbit_radii: f32,
    quad: Quat,
) -> MoonUniform {
    let axes = Mat3::from_quat(quad);
    let surface = moon.body.transpose() * axes;
    let mut light_direction = [Vec4::ZERO; MAX_SYSTEM_STARS];
    let mut light_colour = [Vec4::ZERO; MAX_SYSTEM_STARS];
    for (index, star) in stars.iter().enumerate() {
        light_direction[index] =
            (axes.transpose() * star.direction).extend(moon.lit[index] * RAW_SUNLIGHT);
        light_colour[index] = star.colour.extend(1.0);
    }
    let day_side = 1.0 - moon.illuminated_fraction;
    let planetshine = moon.lit.iter().sum::<f32>() * RAW_SUNLIGHT * PLANET_ALBEDO * day_side
        / (orbit_radii * orbit_radii);
    MoonUniform {
        surface_x: surface.row(0).extend(0.0),
        surface_y: surface.row(1).extend(0.0),
        surface_z: surface.row(2).extend(0.0),
        light_direction,
        light_colour,
        reflectance: (moon.tint * moon.albedo).extend(moon.pattern as f32 / u32::MAX as f32),
        planetshine: Vec4::new(planetshine, 0.0, 0.0, 0.0),
    }
}

/// Keep one disk per moon, placed along its direction from the camera.
pub(super) fn place(
    mut commands: Commands,
    sky: Res<SkyState>,
    quad: Res<MoonQuad>,
    mut materials: ResMut<Assets<MoonMaterial>>,
    camera: Query<&Transform, (With<MainCamera>, Without<MoonDisk>)>,
    mut disks: Query<(
        Entity,
        &MoonDisk,
        &mut Transform,
        &mut Visibility,
        &MeshMaterial3d<MoonMaterial>,
    )>,
) {
    let (Some(current), Some(system)) = (sky.current.as_ref(), sky.system()) else {
        return;
    };
    if !sky.outdoors {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };
    let mut present = vec![false; current.moons.len()];
    for (entity, disk, mut transform, mut visibility, material) in &mut disks {
        let (Some(moon), Some(orbit)) = (current.moons.get(disk.0), system.moons().get(disk.0))
        else {
            commands.entity(entity).despawn();
            continue;
        };
        present[disk.0] = true;
        let rotation = facing(moon.direction);
        *transform = Transform {
            translation: camera.translation + moon.direction * DRAW_METRES,
            rotation,
            scale: Vec3::splat(DRAW_METRES * moon.angular_radius.tan()),
        };
        // Below the horizon the finite world would not hide it.
        *visibility = if moon.direction.y > -moon.angular_radius {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if let Some(mut material) = materials.get_mut(&material.0) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "orbital radius in planet radii"
            )]
            let orbit_radii = orbit.orbit_radii as f32;
            material.moon = uniform(moon, &current.stars, orbit_radii, rotation);
        }
    }
    for (index, _) in present.iter().enumerate().filter(|(_, present)| !**present) {
        commands.spawn((
            Name::new(format!("World moon disk {}", index + 1)),
            MoonDisk(index),
            Mesh3d(quad.0.clone()),
            MeshMaterial3d(materials.add(MoonMaterial::default())),
            Transform::default(),
            Visibility::Hidden,
            NotShadowCaster,
            NotShadowReceiver,
            NoFrustumCulling,
        ));
    }
}
