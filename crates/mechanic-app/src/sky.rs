//! Outdoor atmosphere, celestial motion, and the per-world solar clock.

mod environment;
mod lights;
mod moons;
mod night;

use bevy::camera::Exposure;
use bevy::core_pipeline::Skybox;
use bevy::light::{Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::ScatteringMedium};
use bevy::pbr::{AtmosphereMode, AtmosphereSettings};
use bevy::prelude::*;
use mechanic_core::SECONDS_PER_DAY;
use mechanic_world::{CelestialSky, CelestialSystem, WorldSeed};

use crate::camera::MainCamera;
use crate::dev_tools::DevTools;
use crate::pause_menu::PauseMenuState;
use crate::render::environment::StaticEnvironmentMap;
use crate::schedule::FrameSet;
use crate::world::{AppSpace, WorldListState, WorldRuntime};

pub(crate) use lights::{Companion, Moon, Sun};
pub(crate) use moons::{MoonDisk, MoonMaterial};

const CYCLE_SECONDS: f64 = 3600.0;

#[derive(Resource, Default)]
pub(crate) struct SkyState {
    pub(crate) advancing: bool,
    outdoors: bool,
    garage: CachedEnvironment,
    world: CachedEnvironment,
    night: Handle<Image>,
    /// Render-only fixture override, in seconds since the world's first
    /// midnight. Never written to the world document.
    fixed_seconds: Option<f64>,
    /// The star system of the world last shown, by seed.
    system: Option<(WorldSeed, CelestialSystem)>,
    /// Where the system's bodies stand this frame, while outdoors.
    current: Option<CelestialSky>,
}

impl SkyState {
    /// Displayed time of day, in seconds since midnight.
    pub(crate) fn displayed_seconds(&self, runtime: &WorldRuntime) -> f64 {
        self.fixed_seconds.map_or_else(
            || runtime.time_of_day_seconds(),
            |seconds| seconds.rem_euclid(SECONDS_PER_DAY),
        )
    }

    /// Displayed solar days since the world began.
    pub(crate) fn displayed_days(&self, runtime: &WorldRuntime) -> f64 {
        self.fixed_seconds
            .map_or_else(|| runtime.solar_days(), |seconds| seconds / SECONDS_PER_DAY)
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

    /// The star system of the world on display, if one has been shown.
    pub(crate) fn system(&self) -> Option<&CelestialSystem> {
        self.system.as_ref().map(|(_, system)| system)
    }

    /// Where the system's bodies stand this frame, while outdoors.
    pub(crate) const fn current(&self) -> Option<&CelestialSky> {
        self.current.as_ref()
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
        app.add_plugins(MaterialPlugin::<MoonMaterial>::default())
            .init_resource::<SkyState>()
            .add_systems(Startup, (setup, moons::setup))
            .add_systems(Update, advance_clock.in_set(FrameSet::SkyClock))
            .add_systems(
                Update,
                (
                    switch_space,
                    update_sky,
                    lights::place,
                    moons::place,
                    environment::update,
                )
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
            .expect("sky fixture time must be non-negative hours since day zero");
        assert!(
            hours.is_finite() && hours >= 0.0,
            "sky fixture time must be non-negative hours since day zero"
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
    outdoor_only: Query<Entity, Or<(With<Moon>, With<Companion>, With<MoonDisk>)>>,
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
        lights::spawn_moonlight(&mut commands);
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
        for entity in &outdoor_only {
            commands.entity(entity).despawn();
        }
        sky.current = None;
    }
    sky.outdoors = outdoors;
}

/// Work out where every body stands, then turn the stars and adapt exposure.
fn update_sky(
    mut sky: ResMut<SkyState>,
    runtime: Res<WorldRuntime>,
    mut camera: Query<(&Transform, &mut Exposure, &mut Skybox), With<MainCamera>>,
    mut planets: Query<(&Atmosphere, &mut GlobalTransform)>,
) {
    if !sky.outdoors {
        return;
    }
    let Ok((camera, mut exposure, mut stars)) = camera.single_mut() else {
        return;
    };
    let seed = runtime.seed();
    if sky.system.as_ref().is_none_or(|(shown, _)| *shown != seed) {
        sky.system = Some((seed, CelestialSystem::generate(seed)));
    }
    let days = sky.displayed_days(&runtime);
    let Some(system) = sky.system() else {
        return;
    };
    let celestial = system.sky(days);
    let plan = lights::plan(&celestial);
    stars.rotation = celestial.rotation;
    exposure.ev100 = lights::ev100(&celestial, &plan);
    sky.current = Some(celestial);
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
