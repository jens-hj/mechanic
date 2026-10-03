//! Sunlight from each star of the system, moonlight, and the exposure that
//! adapts to them.

use bevy::light::{SunDisk, light_consts::lux::RAW_SUNLIGHT};
use bevy::prelude::*;
use mechanic_world::CelestialSky;

use super::SkyState;

/// The light of the system's first star: the sun the clock follows.
#[derive(Component)]
pub(crate) struct Sun;

/// Light of the system's star at this index, beyond the first. Each also
/// carries [`Sun`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Companion(pub(crate) usize);

/// Light the moons reflect.
#[derive(Component)]
pub(crate) struct Moon;

/// Boost of moonlight over photometry, so building stays possible at night:
/// a full moon like Earth's gives four lux instead of a third of one.
const MOONLIGHT_GAIN: f32 = 13.0;

/// Smallest disk drawn for a star; farther stars spread their light over it.
const MIN_DISK_RADIANS: f32 = 0.0017;

/// Night light, in lux, the eye is adapted to before any moon or distant
/// star rises.
const NIGHT_ADAPTED_LUX: f32 = 4.0;

/// One celestial light's direction and strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SkyLight {
    /// Unit direction towards the body.
    pub(super) direction: Vec3,
    /// Illuminance at the top of the atmosphere, in lux.
    pub(super) lux: f32,
}

/// Every celestial light this frame.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SkyLights {
    /// One per star, in system order.
    pub(super) stars: Vec<SkyLight>,
    /// Moonlight from the brightest moon above the horizon.
    pub(super) moon: SkyLight,
    /// The one light that casts shadows: a star's index, or the star count
    /// for moonlight. None while nothing is above the horizon.
    pub(super) shadows: Option<usize>,
}

pub(super) fn plan(sky: &CelestialSky) -> SkyLights {
    let stars = sky
        .stars
        .iter()
        .map(|star| SkyLight {
            direction: star.direction,
            lux: star.irradiance * RAW_SUNLIGHT,
        })
        .collect::<Vec<_>>();
    let risen = sky
        .moons
        .iter()
        .filter(|moon| moon.direction.y > 0.0)
        .collect::<Vec<_>>();
    let candidates = if risen.is_empty() {
        sky.moons.iter().collect()
    } else {
        risen
    };
    let moon = candidates
        .iter()
        .max_by(|a, b| a.irradiance.total_cmp(&b.irradiance))
        .map_or(
            SkyLight {
                direction: Vec3::NEG_Y,
                lux: 0.0,
            },
            |brightest| SkyLight {
                direction: brightest.direction,
                // Every risen moon adds its light along the brightest one.
                lux: candidates.iter().map(|moon| moon.irradiance).sum::<f32>()
                    * RAW_SUNLIGHT
                    * MOONLIGHT_GAIN,
            },
        );
    let shadows = stars
        .iter()
        .chain([&moon])
        .enumerate()
        .filter(|(_, light)| light.direction.y > 0.0 && light.lux > 0.0)
        .max_by(|(_, a), (_, b)| a.lux.total_cmp(&b.lux))
        .map(|(index, _)| index);
    SkyLights {
        stars,
        moon,
        shadows,
    }
}

/// Exposure for the sky: dark-adapted at night, brighter for big moons and
/// distant stars, and following the combined suns through twilight.
pub(super) fn ev100(sky: &CelestialSky, lights: &SkyLights) -> f32 {
    let total = sky.stars.iter().map(|star| star.irradiance).sum::<f32>();
    let (mut twilight, mut daylight) = (0.0, 0.0);
    for star in &sky.stars {
        let weight = star.irradiance / total.max(f32::EPSILON);
        let rise = ((star.direction.y + 0.2) / 0.2).clamp(0.0, 1.0);
        twilight += weight * rise * rise * (3.0 - 2.0 * rise);
        daylight += weight * (star.direction.y / std::f32::consts::FRAC_PI_3.sin()).clamp(0.0, 1.0);
    }
    let night_lux = lights
        .stars
        .iter()
        .chain([&lights.moon])
        .filter(|light| light.direction.y > 0.0)
        .map(|light| light.lux)
        .sum::<f32>();
    let night = 1.0 + 0.6 * (night_lux / NIGHT_ADAPTED_LUX).max(1.0).log2();
    let day = 10.5 + total.max(0.25).log2();
    night + (day - night).max(0.0) * twilight + 2.5 * daylight.sqrt()
}

pub(super) fn spawn_moonlight(commands: &mut Commands) {
    commands.spawn((
        Name::new("World moon"),
        Moon,
        DirectionalLight {
            illuminance: 0.0,
            color: Color::srgb(0.7, 0.8, 1.0),
            shadow_maps_enabled: false,
            ..default()
        },
        SunDisk::OFF,
        Transform::default(),
    ));
}

fn aim(direction: Vec3) -> Transform {
    Transform::default().looking_to(-direction, Vec3::Y)
}

/// Point every star's and the moons' light, matching the companion lights to
/// the system's star count.
#[expect(clippy::type_complexity, reason = "disjoint star and moon lights")]
pub(super) fn place(
    mut commands: Commands,
    sky: Res<SkyState>,
    mut suns: Query<
        (
            Entity,
            &mut Transform,
            &mut DirectionalLight,
            Option<&mut SunDisk>,
            Option<&Companion>,
        ),
        (With<Sun>, Without<Moon>),
    >,
    mut moonlight: Query<(&mut Transform, &mut DirectionalLight), (With<Moon>, Without<Sun>)>,
) {
    let Some(current) = sky.current.as_ref().filter(|_| sky.outdoors) else {
        return;
    };
    let lights = plan(current);
    let mut companions = vec![false; current.stars.len()];
    for (entity, mut transform, mut light, disk, companion) in &mut suns {
        let index = companion.map_or(0, |companion| companion.0);
        let Some(star) = current.stars.get(index) else {
            commands.entity(entity).despawn();
            continue;
        };
        if index > 0 {
            companions[index] = true;
        }
        let planned = lights.stars[index];
        *transform = aim(planned.direction);
        light.illuminance = planned.lux;
        light.color = Color::linear_rgb(star.colour.x, star.colour.y, star.colour.z);
        light.shadow_maps_enabled = lights.shadows == Some(index);
        if let Some(mut disk) = disk {
            disk.angular_size = 2.0 * star.angular_radius.max(MIN_DISK_RADIANS);
        }
    }
    for (index, star) in current.stars.iter().enumerate().skip(1) {
        if !companions[index] {
            commands.spawn((
                Name::new(format!("World star {}", index + 1)),
                Sun,
                Companion(index),
                SunDisk {
                    angular_size: 2.0 * star.angular_radius.max(MIN_DISK_RADIANS),
                    intensity: 1.0,
                },
                DirectionalLight {
                    illuminance: lights.stars[index].lux,
                    color: Color::linear_rgb(star.colour.x, star.colour.y, star.colour.z),
                    shadow_maps_enabled: false,
                    ..default()
                },
                aim(star.direction),
            ));
        }
    }
    for (mut transform, mut light) in &mut moonlight {
        *transform = aim(lights.moon.direction);
        light.illuminance = lights.moon.lux;
        light.shadow_maps_enabled = lights.shadows == Some(current.stars.len());
    }
}
