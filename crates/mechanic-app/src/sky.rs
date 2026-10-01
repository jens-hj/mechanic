//! Outdoor atmosphere, celestial motion, and the per-world solar clock.

mod environment;
mod night;

use bevy::camera::Exposure;
use bevy::core_pipeline::Skybox;
use bevy::light::{
    Atmosphere, AtmosphereEnvironmentMapLight, SunDisk, atmosphere::ScatteringMedium,
};
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
use bevy::prelude::*;
use mechanic_core::SECONDS_PER_DAY;

use crate::camera::MainCamera;
use crate::dev_tools::DevTools;
use crate::pause_menu::PauseMenuState;
use crate::render::environment::StaticEnvironmentMap;
use crate::schedule::FrameSet;
use crate::world::{AppSpace, WorldListState, WorldRuntime};

const CYCLE_SECONDS: f64 = 3600.0;

#[derive(Component)]
pub(crate) struct Sun;

#[derive(Component)]
pub(crate) struct Moon;

#[derive(Resource, Default)]
pub(crate) struct SkyState {
    pub(crate) advancing: bool,
    outdoors: bool,
    garage: CachedEnvironment,
    world: CachedEnvironment,
    night: Handle<Image>,
    /// Render-only fixture override. Never written to the world document.
    fixed_seconds: Option<f64>,
}

impl SkyState {
    pub(crate) fn displayed_seconds(&self, runtime: &WorldRuntime) -> f64 {
        self.fixed_seconds
            .unwrap_or_else(|| runtime.time_of_day_seconds())
    }

    pub(crate) fn status(&self) -> &'static str {
        if self.fixed_seconds.is_some() {
            "fixture"
        } else if self.advancing {
            "running"
        } else {
            "paused"
        }
    }
}

#[derive(Default)]
struct CachedEnvironment {
    filtered: Option<EnvironmentMapLight>,
    generator: Option<GeneratedEnvironmentMapLight>,
}

impl CachedEnvironment {
    fn restore(&self, commands: &mut EntityCommands) {
        commands.remove::<(EnvironmentMapLight, GeneratedEnvironmentMapLight)>();
        if let Some(map) = &self.filtered {
            commands.insert(map.clone());
        }
        if let Some(generator) = &self.generator {
            commands.insert(generator.clone());
        }
    }
}

pub(crate) struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        environment::install(app);
        app.init_resource::<SkyState>()
            .add_systems(Startup, setup)
            .add_systems(Update, advance_clock.in_set(FrameSet::SkyClock))
            .add_systems(
                Update,
                (switch_space, update_sky, environment::update)
                    .chain()
                    .in_set(FrameSet::Sky),
            );
    }
}

fn setup(
    mut commands: Commands,
    mut media: ResMut<Assets<ScatteringMedium>>,
    mut images: ResMut<Assets<Image>>,
    mut sky: ResMut<SkyState>,
) {
    commands.spawn(Atmosphere::earth(
        media.add(ScatteringMedium::earth(256, 256)),
    ));
    sky.night = images.add(night::cubemap());
    sky.fixed_seconds = crate::env::text(crate::env::SKY_TIME).map(|value| {
        let hours = value
            .parse::<f64>()
            .expect("sky fixture time must be hours in [0, 24)");
        assert!(
            (0.0..24.0).contains(&hours),
            "sky fixture time must be hours in [0, 24)"
        );
        hours * 3600.0
    });
}

fn advance_clock(
    time: Res<Time<Real>>,
    space: Res<State<AppSpace>>,
    list: Res<WorldListState>,
    pause: Res<PauseMenuState>,
    dev: Res<DevTools>,
    mut sky: ResMut<SkyState>,
    mut runtime: ResMut<WorldRuntime>,
) {
    sky.advancing = *space.get() == AppSpace::World
        && !list.is_open()
        && !pause.blocks_world_input()
        && !dev.cycle_paused
        && sky.fixed_seconds.is_none();
    if sky.advancing {
        runtime.advance_day(time.delta_secs_f64() * SECONDS_PER_DAY / CYCLE_SECONDS);
    }
}

#[expect(
    clippy::type_complexity,
    reason = "one camera owns its cached environment"
)]
fn switch_space(
    mut commands: Commands,
    space: Res<State<AppSpace>>,
    mut sky: ResMut<SkyState>,
    camera: Query<
        (
            Entity,
            Option<&EnvironmentMapLight>,
            Option<&GeneratedEnvironmentMapLight>,
        ),
        With<MainCamera>,
    >,
    moons: Query<Entity, With<Moon>>,
) {
    let outdoors = *space.get() == AppSpace::World;
    if outdoors == sky.outdoors {
        return;
    }
    let Ok((entity, filtered, generator)) = camera.single() else {
        return;
    };
    let previous = CachedEnvironment {
        filtered: filtered.cloned(),
        generator: generator.cloned(),
    };
    let mut camera = commands.entity(entity);
    if outdoors {
        sky.garage = previous;
        sky.world.restore(&mut camera);
        camera.remove::<StaticEnvironmentMap>().insert((
            AtmosphereSettings {
                rendering_method: AtmosphereMode::LookupTexture,
                ..default()
            },
            AtmosphereEnvironmentMapLight {
                size: UVec2::splat(128),
                ..default()
            },
            Skybox {
                image: Some(sky.night.clone()),
                brightness: 1.0,
                rotation: Quat::IDENTITY,
            },
            AmbientLight {
                color: Color::srgb(0.45, 0.58, 0.8),
                brightness: 0.5,
                ..default()
            },
        ));
        commands.spawn((
            Name::new("World moon"),
            Moon,
            DirectionalLight {
                illuminance: 4.0,
                color: Color::srgb(0.7, 0.8, 1.0),
                shadow_maps_enabled: true,
                ..default()
            },
            SunDisk::OFF,
            Transform::default(),
        ));
    } else {
        sky.world = previous;
        sky.garage.restore(&mut camera);
        camera
            .remove::<(
                AtmosphereSettings,
                AtmosphereEnvironmentMapLight,
                Skybox,
                AmbientLight,
            )>()
            .insert(StaticEnvironmentMap);
        for moon in &moons {
            commands.entity(moon).despawn();
        }
    }
    sky.outdoors = outdoors;
}

/// Fixed obliquity gives a 60 degree noon elevation, sunrise at 06:00.
#[expect(
    clippy::cast_possible_truncation,
    reason = "bounded solar phase rendered in f32"
)]
fn celestial_rotation(seconds: f64) -> Quat {
    let angle = ((seconds / SECONDS_PER_DAY - 0.25) * std::f64::consts::TAU) as f32;
    Quat::from_rotation_x(std::f32::consts::PI / 6.0) * Quat::from_rotation_z(angle)
}

#[expect(
    clippy::type_complexity,
    reason = "disjoint celestial and camera transforms"
)]
fn update_sky(
    sky: Res<SkyState>,
    runtime: Res<WorldRuntime>,
    mut camera: Query<
        (&Transform, &mut Exposure, &mut Skybox),
        (With<MainCamera>, Without<Sun>, Without<Moon>),
    >,
    mut lights: Query<
        (&mut Transform, &mut DirectionalLight, Has<Moon>),
        (Or<(With<Sun>, With<Moon>)>, Without<MainCamera>),
    >,
    mut planets: Query<(&Atmosphere, &mut GlobalTransform)>,
) {
    if !sky.outdoors {
        return;
    }
    let Ok((camera, mut exposure, mut stars)) = camera.single_mut() else {
        return;
    };
    let rotation = celestial_rotation(
        sky.fixed_seconds
            .unwrap_or_else(|| runtime.time_of_day_seconds()),
    );
    let sun = rotation * Vec3::X;
    stars.rotation = rotation;
    // Smooth twilight adaptation; enough fill and moonlight to build at night.
    let twilight = ((sun.y + 0.2) / 0.2).clamp(0.0, 1.0);
    let twilight = twilight * twilight * (3.0 - 2.0 * twilight);
    let daylight = (sun.y / std::f32::consts::FRAC_PI_3.sin()).clamp(0.0, 1.0);
    exposure.ev100 = 1.0 + 9.5 * twilight + 2.5 * daylight.sqrt();
    for (mut transform, mut light, moon) in &mut lights {
        let direction = if moon { -sun } else { sun };
        light.shadow_maps_enabled = direction.y > 0.0;
        *transform = Transform::default().looking_to(-direction, Vec3::Y);
        light.illuminance = if moon {
            4.0
        } else {
            bevy::light::light_consts::lux::RAW_SUNLIGHT
        };
    }
    for (atmosphere, mut planet) in &mut planets {
        *planet = planet_transform(
            atmosphere.inner_radius,
            camera.translation,
            runtime.origin(),
        );
    }
}

/// Flat local tangent plane: preserve global altitude across rebases, while
/// keeping the planet directly below the viewer throughout this finite world.
#[expect(
    clippy::cast_possible_truncation,
    reason = "render coordinates are f32"
)]
fn planet_transform(
    radius: f32,
    camera: Vec3,
    origin: mechanic_world::FloatingOrigin,
) -> GlobalTransform {
    GlobalTransform::from_translation(Vec3::new(camera.x, -radius - origin.0.y as f32, camera.z))
}

#[cfg(test)]
mod tests;
