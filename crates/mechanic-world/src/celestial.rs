//! The star system each world belongs to, and where its bodies stand in the
//! sky at any moment.
//!
//! A world's seed picks one, two, or three stars, the planet's year, axial
//! tilt, and the observer's latitude, and up to three moons. Every body keeps
//! its own circular orbit, so the sun drifts along the ecliptic through the
//! year and its noon height follows the seasons, each moon rises at a
//! different hour every night and runs through its phases, and the stars turn
//! once per sidereal day. Moons can eclipse a sun and pass through the
//! planet's shadow. The sky is a pure function of the seed and the elapsed
//! solar time; nothing here is stored.
//!
//! Directions are given in the observer's horizon frame, which is the world
//! frame: `x` east, `y` up, `z` south.

#![expect(
    clippy::cast_possible_truncation,
    reason = "ephemerides are computed in f64 and published to the renderer in f32"
)]

use crate::WorldSeed;
use crate::generation::mix;
use bevy_math::{DMat3, DVec3, Mat3, Quat, Vec3};
use std::f64::consts::{PI, TAU};

/// Most stars a system holds.
pub const MAX_SYSTEM_STARS: usize = 3;

/// Most moons a planet keeps.
pub const MAX_SYSTEM_MOONS: usize = 3;

/// Mean radius of the home planet, in kilometres.
const PLANET_RADIUS_KM: f64 = 6_371.0;

/// Radius of a solar-mass star, in astronomical units.
const SOLAR_RADIUS_AU: f64 = 0.004_65;

/// Photosphere temperature of a solar-mass star, whose light is white.
const SOLAR_TEMPERATURE_K: f64 = 5_772.0;

/// Kepler's third law about the home planet: a moon this many planet radii
/// out takes [`MOON_REFERENCE_DAYS`] to orbit once.
const MOON_REFERENCE_RADII: f64 = 60.27;

/// Sidereal period of a moon at [`MOON_REFERENCE_RADII`], in days.
const MOON_REFERENCE_DAYS: f64 = 27.32;

/// One main-sequence star of the system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CelestialStar {
    /// Mass, in solar masses.
    pub mass: f64,
    /// Luminosity, in solar luminosities.
    pub luminosity: f64,
    /// Radius, in solar radii.
    pub radius: f64,
    /// Photosphere temperature, in kelvin.
    pub temperature: f64,
}

impl CelestialStar {
    fn main_sequence(mass: f64) -> Self {
        let luminosity = mass.powi(4);
        let radius = mass.powf(0.8);
        Self {
            mass,
            luminosity,
            radius,
            temperature: SOLAR_TEMPERATURE_K * (luminosity / (radius * radius)).powf(0.25),
        }
    }

    /// Linear RGB colour of the star's light at unit luminance.
    #[must_use]
    pub fn colour(&self) -> Vec3 {
        blackbody_colour(self.temperature)
    }
}

/// Linear RGB colour of a black body at `kelvin`, at unit luminance; the
/// Sun's photosphere is white.
#[must_use]
pub fn blackbody_colour(kelvin: f64) -> Vec3 {
    // Planck's law relative to the Sun at representative wavelengths.
    let radiance = |nanometres: f64| {
        let exponent = |kelvin: f64| 1.438_8e7 / (nanometres * kelvin);
        (exponent(SOLAR_TEMPERATURE_K).exp() - 1.0) / (exponent(kelvin).exp() - 1.0)
    };
    let rgb = DVec3::new(radiance(610.0), radiance(550.0), radiance(465.0));
    (rgb / rgb.dot(DVec3::new(0.2126, 0.7152, 0.0722))).as_vec3()
}

/// A moon of the home planet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CelestialMoon {
    /// Radius, in kilometres.
    pub radius_km: f64,
    /// Orbital radius, in planet radii.
    pub orbit_radii: f64,
    /// Lambert reflectance of its surface.
    pub albedo: f64,
    /// Linear tint of its surface.
    pub tint: Vec3,
    /// Selects its surface markings.
    pub pattern: u32,
    orbit: Orbit,
}

impl CelestialMoon {
    /// Time for one orbit against the stars, in days.
    #[must_use]
    pub const fn period_days(&self) -> f64 {
        self.orbit.period
    }
}

/// A circular orbit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Orbit {
    /// Tilt of the orbital plane from the ecliptic, in radians.
    inclination: f64,
    /// Longitude of the ascending node, in radians.
    node: f64,
    /// Angle from the ascending node at day zero, in radians.
    phase: f64,
    /// Sidereal period, in days.
    period: f64,
}

impl Orbit {
    fn direction(&self, days: f64) -> DVec3 {
        let angle = self.phase + TAU * (days / self.period).fract();
        let (sin_u, cos_u) = angle.sin_cos();
        let (sin_n, cos_n) = self.node.sin_cos();
        let (sin_i, cos_i) = self.inclination.sin_cos();
        DVec3::new(
            cos_n * cos_u - sin_n * sin_u * cos_i,
            sin_n * cos_u + cos_n * sin_u * cos_i,
            sin_u * sin_i,
        )
    }

    fn normal(&self) -> DVec3 {
        let (sin_n, cos_n) = self.node.sin_cos();
        let (sin_i, cos_i) = self.inclination.sin_cos();
        DVec3::new(sin_n * sin_i, -cos_n * sin_i, cos_i)
    }
}

/// One star, or two in a mutual orbit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StarGroup {
    /// Index of the first star in [`CelestialSystem::stars`].
    first: usize,
    pair: Option<Pair>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pair {
    second: usize,
    orbit: Orbit,
    /// Distance between the two stars, in astronomical units.
    separation: f64,
}

impl StarGroup {
    fn mass(&self, stars: &[CelestialStar]) -> f64 {
        stars[self.first].mass + self.pair.map_or(0.0, |pair| stars[pair.second].mass)
    }

    fn place(&self, stars: &[CelestialStar], centre: DVec3, days: f64, positions: &mut [DVec3]) {
        let Some(pair) = self.pair else {
            positions[self.first] = centre;
            return;
        };
        let (first, second) = (stars[self.first].mass, stars[pair.second].mass);
        let offset = pair.orbit.direction(days) * pair.separation / (first + second);
        positions[self.first] = centre + offset * second;
        positions[pair.second] = centre - offset * first;
    }
}

/// Distant stars the inner system orbits.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Companion {
    group: StarGroup,
    orbit: Orbit,
    /// Distance from the inner stars, in astronomical units.
    distance: f64,
}

/// A world's star system: its stars, the planet's orbit and spin, and moons.
#[derive(Clone, Debug, PartialEq)]
pub struct CelestialSystem {
    stars: Vec<CelestialStar>,
    /// The star or close pair the planet orbits. Its first star is the
    /// brightest of them and is the sun the clock follows.
    inner: StarGroup,
    companion: Option<Companion>,
    /// Planet's orbital radius about the inner stars, in astronomical units.
    distance: f64,
    year_days: f64,
    /// Axial tilt, in radians.
    obliquity: f64,
    /// Observer's latitude, in radians north.
    latitude: f64,
    /// Ecliptic longitude of the inner stars at day zero, in radians; zero
    /// is the spring equinox.
    season: f64,
    moons: Vec<CelestialMoon>,
}

impl CelestialSystem {
    /// The system a world with this seed belongs to.
    #[must_use]
    pub fn generate(seed: WorldSeed) -> Self {
        let mut rng = Rng(mix(seed.0 ^ 0x5ce1_e571_a15e_ed00));
        let (close_pair, distant) = match rng.unit() {
            roll if roll < 0.45 => (false, 0),
            roll if roll < 0.65 => (true, 0),
            roll if roll < 0.8 => (false, 1),
            roll if roll < 0.92 => (true, 1),
            _ => (false, 2),
        };
        let year_days = rng.between(48.0, 96.0).round();
        let primary = rng.between(0.8, 1.2);
        let mut stars = vec![CelestialStar::main_sequence(primary)];
        if close_pair {
            stars.push(CelestialStar::main_sequence(
                primary * rng.between(0.35, 0.9),
            ));
        }
        // The planet sits where it receives the light Earth does.
        let distance = stars.iter().map(|star| star.luminosity).sum::<f64>().sqrt();
        let inner_mass = stars.iter().map(|star| star.mass).sum::<f64>();
        let period = |radius: f64, mass: f64| {
            year_days * ((radius.powi(3) / mass) / (distance.powi(3) / inner_mass)).sqrt()
        };
        let inner = StarGroup {
            first: 0,
            pair: close_pair.then(|| {
                // Circumbinary planets need several binary separations of room.
                let separation = distance / rng.between(3.6, 6.0);
                Pair {
                    second: 1,
                    orbit: rng.orbit(1.5, period(separation, inner_mass)),
                    separation,
                }
            }),
        };
        let companion = (distant > 0).then(|| {
            let first = stars.len();
            stars.push(CelestialStar::main_sequence(rng.between(0.45, 1.05)));
            let pair = (distant > 1).then(|| {
                stars.push(CelestialStar::main_sequence(rng.between(0.4, 1.0)));
                let separation = rng.between(0.6, 4.0);
                Pair {
                    second: first + 1,
                    orbit: rng.orbit(
                        30.0,
                        period(separation, stars[first].mass + stars[first + 1].mass),
                    ),
                    separation,
                }
            });
            let group = StarGroup { first, pair };
            let distance = rng.between(60.0, 400.0);
            Companion {
                group,
                orbit: rng.orbit(50.0, period(distance, inner_mass + group.mass(&stars))),
                distance,
            }
        });
        // Outside the tropics: the noon sun always stands to the south.
        let latitude = rng.between(25.0, 48.0);
        let obliquity = rng.between(8.0, 28.0_f64.min(latitude - 6.0)).to_radians();
        let latitude = latitude.to_radians();
        let season = rng.between(0.0, PI);
        let moons = generate_moons(&mut rng, year_days);
        Self {
            stars,
            inner,
            companion,
            distance,
            year_days,
            obliquity,
            latitude,
            season,
            moons,
        }
    }

    /// The system's stars. The first is the sun the clock follows.
    #[must_use]
    pub fn stars(&self) -> &[CelestialStar] {
        &self.stars
    }

    /// The planet's moons, innermost first.
    #[must_use]
    pub fn moons(&self) -> &[CelestialMoon] {
        &self.moons
    }

    /// Solar days in one orbit of the planet.
    #[must_use]
    pub const fn year_days(&self) -> f64 {
        self.year_days
    }

    /// Whether the planet orbits two stars at once.
    #[must_use]
    pub const fn circumbinary(&self) -> bool {
        self.inner.pair.is_some()
    }

    /// Observer's latitude, in radians north.
    #[must_use]
    pub const fn latitude(&self) -> f64 {
        self.latitude
    }

    /// The planet's axial tilt, in radians.
    #[must_use]
    pub const fn obliquity(&self) -> f64 {
        self.obliquity
    }

    /// Rotation from the system's ecliptic frame into the observer's horizon
    /// frame after `days` solar days. Local mean noon falls at `.5`.
    #[must_use]
    pub fn horizon_rotation(&self, days: f64) -> DMat3 {
        // The planet turns once more per year against the stars than against
        // its sun; the mean sun crosses the meridian at noon.
        let sidereal = self.season - PI + TAU * (days.fract() + (days / self.year_days).fract());
        let (sin_lat, cos_lat) = self.latitude.sin_cos();
        let (sin_t, cos_t) = sidereal.sin_cos();
        let east = DVec3::new(-sin_t, cos_t, 0.0);
        let up = DVec3::new(cos_lat * cos_t, cos_lat * sin_t, sin_lat);
        let south = DVec3::new(sin_lat * cos_t, sin_lat * sin_t, -cos_lat);
        DMat3::from_cols(east, up, south).transpose() * DMat3::from_rotation_x(self.obliquity)
    }

    /// Where every body stands, and how much light it sends, after `days`
    /// solar days.
    #[must_use]
    pub fn sky(&self, days: f64) -> CelestialSky {
        let horizon = self.horizon_rotation(days);
        let zenith = horizon.transpose() * DVec3::Y;
        let mut positions = [DVec3::ZERO; MAX_SYSTEM_STARS];
        let mut inner_centre = DVec3::ZERO;
        if let Some(companion) = &self.companion {
            let inner = self.inner.mass(&self.stars);
            let outer = companion.group.mass(&self.stars);
            let axis = companion.orbit.direction(days) * companion.distance / (inner + outer);
            inner_centre = -axis * outer;
            companion
                .group
                .place(&self.stars, axis * inner, days, &mut positions);
        }
        self.inner
            .place(&self.stars, inner_centre, days, &mut positions);
        let longitude = self.season + TAU * (days / self.year_days).fract();
        let planet =
            inner_centre - DVec3::new(longitude.cos(), longitude.sin(), 0.0) * self.distance;

        let mut stars = self
            .stars
            .iter()
            .zip(positions)
            .map(|(star, position)| {
                let offset = position - planet;
                let distance = offset.length();
                SkyBody {
                    direction: offset / distance,
                    angular_radius: (star.radius * SOLAR_RADIUS_AU / distance).min(1.0).asin(),
                    distance,
                    irradiance: star.luminosity / (distance * distance),
                }
            })
            .collect::<Vec<_>>();
        let moons = self
            .moons
            .iter()
            .map(|moon| moon_view(moon, days, zenith, &stars))
            .collect::<Vec<_>>();

        // Moons and nearer stars hide part of each sun.
        let shade = stars
            .iter()
            .map(|star| {
                let moons = moons.iter().map(|moon| {
                    covered_fraction(
                        star.angular_radius,
                        moon.body.angular_radius,
                        angle_between(star.direction, moon.body.direction),
                    )
                });
                let nearer = stars
                    .iter()
                    .filter(|other| other.distance < star.distance)
                    .map(|other| {
                        covered_fraction(
                            star.angular_radius,
                            other.angular_radius,
                            angle_between(star.direction, other.direction),
                        )
                    });
                1.0 - moons.chain(nearer).sum::<f64>().min(1.0)
            })
            .collect::<Vec<_>>();
        for (star, shade) in stars.iter_mut().zip(shade) {
            star.irradiance *= shade;
        }

        let to_horizon = |direction: DVec3| (horizon * direction).as_vec3();
        CelestialSky {
            rotation: Quat::from_mat3(&horizon.as_mat3()),
            stars: stars
                .iter()
                .zip(&self.stars)
                .map(|(body, star)| SkyStar {
                    direction: to_horizon(body.direction),
                    angular_radius: body.angular_radius as f32,
                    irradiance: body.irradiance as f32,
                    colour: star.colour(),
                })
                .collect(),
            moons: moons
                .into_iter()
                .zip(&self.moons)
                .map(|(view, moon)| SkyMoon {
                    direction: to_horizon(view.body.direction),
                    angular_radius: view.body.angular_radius as f32,
                    irradiance: view.body.irradiance as f32,
                    lit: view.lit.map(|lit| lit as f32),
                    illuminated_fraction: view.illuminated_fraction as f32,
                    body: Mat3::from_cols(
                        to_horizon(view.axes[0]),
                        to_horizon(view.axes[1]),
                        to_horizon(view.axes[2]),
                    ),
                    albedo: moon.albedo as f32,
                    tint: moon.tint,
                    pattern: moon.pattern,
                })
                .collect(),
        }
    }
}

fn moon_view(moon: &CelestialMoon, days: f64, zenith: DVec3, stars: &[SkyBody]) -> MoonView {
    let centre = moon.orbit.direction(days) * moon.orbit_radii * PLANET_RADIUS_KM;
    // Seen from the surface, not the planet's centre: near moons shift.
    let offset = centre - zenith * PLANET_RADIUS_KM;
    let distance = offset.length();
    let direction = offset / distance;
    let mut lit = [0.0; MAX_SYSTEM_STARS];
    let mut irradiance = 0.0;
    for (index, star) in stars.iter().enumerate() {
        lit[index] = star.irradiance
            * planet_shadow(centre, star.direction, star.angular_radius, moon.radius_km);
        let phase = angle_between(star.direction, -direction);
        irradiance += lit[index]
            * (2.0 / 3.0 * moon.albedo)
            * (moon.radius_km / distance).powi(2)
            * lambert_phase(phase);
    }
    // Tidally locked: the near side faces the planet, north along the
    // orbit's pole.
    let towards_planet = -centre.normalize();
    let normal = moon.orbit.normal();
    let north = (normal - towards_planet * normal.dot(towards_planet)).normalize();
    MoonView {
        body: SkyBody {
            direction,
            angular_radius: (moon.radius_km / distance).min(1.0).asin(),
            distance,
            irradiance,
        },
        lit,
        illuminated_fraction: 0.5 * (1.0 + stars[0].direction.dot(-direction)),
        axes: [north.cross(towards_planet), north, towards_planet],
    }
}

/// Where a body stands, in the ecliptic frame.
struct SkyBody {
    direction: DVec3,
    angular_radius: f64,
    /// Kilometres for moons, astronomical units for stars.
    distance: f64,
    irradiance: f64,
}

struct MoonView {
    body: SkyBody,
    lit: [f64; MAX_SYSTEM_STARS],
    illuminated_fraction: f64,
    axes: [DVec3; 3],
}

/// The sky at one moment, in the horizon frame.
#[derive(Clone, Debug, PartialEq)]
pub struct CelestialSky {
    /// Rotation from the system's ecliptic frame into the horizon frame.
    /// Distant stars are fixed in the ecliptic frame.
    pub rotation: Quat,
    /// The system's stars, in [`CelestialSystem::stars`] order.
    pub stars: Vec<SkyStar>,
    /// The planet's moons, in [`CelestialSystem::moons`] order.
    pub moons: Vec<SkyMoon>,
}

/// A star of the system as the observer sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyStar {
    /// Unit direction towards the star.
    pub direction: Vec3,
    /// Apparent radius, in radians.
    pub angular_radius: f32,
    /// Light reaching the top of the atmosphere, relative to Earth's sunlight,
    /// less any part hidden by a moon or nearer star.
    pub irradiance: f32,
    /// Linear colour of its light at unit luminance.
    pub colour: Vec3,
}

/// A moon as the observer sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyMoon {
    /// Unit direction towards the moon.
    pub direction: Vec3,
    /// Apparent radius, in radians.
    pub angular_radius: f32,
    /// Starlight it reflects to the observer, relative to Earth's sunlight.
    pub irradiance: f32,
    /// Light falling on it from each star, relative to Earth's sunlight,
    /// after the planet's shadow. Unused entries are zero.
    pub lit: [f32; MAX_SYSTEM_STARS],
    /// Share of its visible disk the first star lights.
    pub illuminated_fraction: f32,
    /// Its axes in the horizon frame: x, north pole, and the near-side pole
    /// facing the planet.
    pub body: Mat3,
    /// Lambert reflectance of its surface.
    pub albedo: f32,
    /// Linear tint of its surface.
    pub tint: Vec3,
    /// Selects its surface markings.
    pub pattern: u32,
}

fn generate_moons(rng: &mut Rng, year_days: f64) -> Vec<CelestialMoon> {
    let count: u32 = match rng.unit() {
        roll if roll < 0.1 => 0,
        roll if roll < 0.6 => 1,
        roll if roll < 0.88 => 2,
        _ => 3,
    };
    // Moons stay well inside the planet's reach: a third of a year at most.
    let widest =
        (MOON_REFERENCE_RADII * (year_days / 3.0 / MOON_REFERENCE_DAYS).powf(2.0 / 3.0)).min(60.0);
    let nearest = 10.0_f64;
    let span = (widest / nearest).ln() / f64::from(count.max(1));
    let tints = [
        Vec3::ONE,
        Vec3::new(1.0, 0.92, 0.8),
        Vec3::new(0.88, 0.94, 1.0),
        Vec3::new(1.0, 0.8, 0.66),
    ];
    (0..count)
        .map(|index| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "unit draws pick a table entry and a seed"
            )]
            let (tint, pattern) = (
                tints[(rng.unit() * 4.0) as usize % tints.len()],
                (rng.unit() * f64::from(u32::MAX)) as u32,
            );
            let orbit_radii = nearest * (span * (f64::from(index) + rng.between(0.15, 0.85))).exp();
            let apparent = rng.between(0.12, 0.8).to_radians();
            let radius_km =
                ((orbit_radii - 1.0) * PLANET_RADIUS_KM * apparent.tan()).clamp(250.0, 3_000.0);
            let period = MOON_REFERENCE_DAYS * (orbit_radii / MOON_REFERENCE_RADII).powf(1.5);
            CelestialMoon {
                radius_km,
                orbit_radii,
                albedo: rng.between(0.12, 0.5),
                tint,
                pattern,
                orbit: rng.orbit(6.0, period),
            }
        })
        .collect()
}

/// Deterministic stream of unit values.
struct Rng(u64);

impl Rng {
    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits map exactly onto the unit interval"
    )]
    fn unit(&mut self) -> f64 {
        self.0 = mix(self.0.wrapping_add(0x9e37_79b9_7f4a_7c15));
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        (hi - lo).mul_add(self.unit(), lo)
    }

    fn orbit(&mut self, max_inclination_degrees: f64, period: f64) -> Orbit {
        Orbit {
            inclination: self.between(0.0, max_inclination_degrees).to_radians(),
            node: self.between(0.0, TAU),
            phase: self.between(0.0, TAU),
            period,
        }
    }
}

fn angle_between(a: DVec3, b: DVec3) -> f64 {
    a.cross(b).length().atan2(a.dot(b))
}

/// Light a Lambertian sphere returns at `phase` radians between its light
/// and its viewer, relative to full phase.
fn lambert_phase(phase: f64) -> f64 {
    (phase.sin() + (PI - phase) * phase.cos()) / PI
}

/// Share of a disk of `radius` hidden by a disk of `occluder` radius whose
/// centre is `separation` away, all as angles.
fn covered_fraction(radius: f64, occluder: f64, separation: f64) -> f64 {
    if separation >= radius + occluder {
        return 0.0;
    }
    if separation <= occluder - radius {
        return 1.0;
    }
    if separation <= radius - occluder {
        return (occluder / radius).powi(2);
    }
    let lens = |near: f64, far: f64| {
        near * near
            * ((separation * separation + near * near - far * far) / (2.0 * separation * near))
                .clamp(-1.0, 1.0)
                .acos()
    };
    let kite = 0.5
        * ((-separation + radius + occluder)
            * (separation + radius - occluder)
            * (separation - radius + occluder)
            * (separation + radius + occluder))
            .max(0.0)
            .sqrt();
    ((lens(radius, occluder) + lens(occluder, radius) - kite) / (PI * radius * radius)).min(1.0)
}

/// Share of a star's light reaching a moon at `centre` kilometres from the
/// planet past the planet's umbra and penumbra.
fn planet_shadow(centre: DVec3, star: DVec3, star_radius: f64, moon_radius_km: f64) -> f64 {
    let behind = -centre.dot(star);
    if behind <= 0.0 {
        return 1.0;
    }
    let across = (centre + star * behind).length();
    let spread = behind * star_radius.tan();
    let start = PLANET_RADIUS_KM - spread - moon_radius_km;
    let end = PLANET_RADIUS_KM + spread + moon_radius_km;
    let t = ((across - start) / (end - start)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests;
