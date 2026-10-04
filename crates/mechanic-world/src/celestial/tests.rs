use super::*;

fn systems() -> impl Iterator<Item = CelestialSystem> {
    (0..400).map(|seed| CelestialSystem::generate(WorldSeed(seed)))
}

fn elevation_degrees(direction: Vec3) -> f64 {
    f64::from(direction.y.asin().to_degrees())
}

#[test]
fn seeds_choose_one_two_or_three_stars_and_up_to_three_moons() {
    let mut star_counts = [0; MAX_SYSTEM_STARS + 1];
    let mut moon_counts = [0; MAX_SYSTEM_MOONS + 1];
    let mut circumbinary = 0;
    for system in systems() {
        star_counts[system.stars().len()] += 1;
        moon_counts[system.moons().len()] += 1;
        circumbinary += usize::from(system.circumbinary());
        for days in [0.0, 0.37, 17.9, 1234.5] {
            let sky = system.sky(days);
            assert_eq!(sky.stars.len(), system.stars().len());
            assert_eq!(sky.moons.len(), system.moons().len());
            for star in &sky.stars {
                assert!((star.direction.length() - 1.0).abs() < 1e-4);
                assert!(star.irradiance.is_finite() && star.irradiance >= 0.0);
                assert!(star.angular_radius > 0.0 && star.colour.is_finite());
            }
            for moon in &sky.moons {
                assert!((moon.direction.length() - 1.0).abs() < 1e-4);
                assert!(moon.irradiance.is_finite() && moon.irradiance >= 0.0);
                assert!((0.0..=1.0).contains(&moon.illuminated_fraction));
            }
        }
    }
    assert_eq!(star_counts[0], 0);
    assert!(star_counts[1..].iter().all(|&count| count > 10));
    assert!(moon_counts.iter().all(|&count| count > 10));
    assert!(circumbinary > 10);
    assert_eq!(
        CelestialSystem::generate(WorldSeed(7)),
        CelestialSystem::generate(WorldSeed(7))
    );
}

#[test]
fn the_sun_rises_east_crosses_south_at_noon_and_is_down_at_midnight() {
    for system in systems().filter(|system| !system.circumbinary()).take(20) {
        for day in [0.0, 11.0, 30.0, 61.0] {
            let sun = |hour: f64| system.sky(day + hour / 24.0).stars[0].direction;
            let noon = sun(12.0);
            assert!(noon.y > 0.1, "noon sun above the horizon");
            assert!(noon.z > 0.0, "noon sun stands south of a northern observer");
            // The obliquity's equation of time keeps true noon within minutes.
            assert!(noon.x.abs() < 0.07);
            assert!(sun(0.0).y < 0.0, "no midnight sun at these latitudes");
            assert!(sun(9.0).x > 0.0 && sun(15.0).x < 0.0);
        }
    }
}

#[test]
fn noon_height_follows_the_seasons_through_the_year() {
    for system in systems().filter(|system| !system.circumbinary()).take(8) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "whole-day years"
        )]
        let year = system.year_days() as u32;
        let noons = (0..year)
            .map(|day| elevation_degrees(system.sky(f64::from(day) + 0.5).stars[0].direction))
            .collect::<Vec<_>>();
        let latitude = system.latitude().to_degrees();
        let tilt = system.obliquity().to_degrees();
        let highest = noons.iter().copied().fold(f64::MIN, f64::max);
        let lowest = noons.iter().copied().fold(f64::MAX, f64::min);
        assert!((highest - (90.0 - latitude + tilt)).abs() < 1.0);
        assert!((lowest - (90.0 - latitude - tilt)).abs() < 1.0);
    }
}

#[test]
fn background_stars_return_each_sidereal_day_and_drift_a_little_each_solar_day() {
    let system = CelestialSystem::generate(WorldSeed(3));
    let star = DVec3::new(0.3, -0.5, 0.81).normalize();
    let year = system.year_days();
    let at = |days: f64| system.horizon_rotation(days) * star;
    let start = 4.25;
    assert!(at(start).distance(at(start + year / (year + 1.0))) < 1e-9);
    let drift = angle_between(at(start), at(start + 1.0)).to_degrees();
    // A star stands one year's daily share of a turn further west each night.
    assert!(drift > 0.0 && drift < 360.0 / year * 1.01);
    assert!(drift > 360.0 / year * 0.5);
}

#[test]
fn moons_run_through_their_phases_on_their_own_orbits() {
    for system in systems()
        .filter(|system| !system.moons().is_empty())
        .take(10)
    {
        for (index, moon) in system.moons().iter().enumerate() {
            let synodic = 1.0 / (1.0 / moon.period_days() - 1.0 / system.year_days());
            let samples = (0..400)
                .map(|step| system.sky(5.0 + synodic * f64::from(step) / 400.0))
                .collect::<Vec<_>>();
            let fraction = |sky: &CelestialSky| sky.moons[index].illuminated_fraction;
            assert!(samples.iter().any(|sky| fraction(sky) > 0.97));
            assert!(samples.iter().any(|sky| fraction(sky) < 0.03));
            // Unlike a moon pinned opposite the sun, it is sometimes beside it.
            let elongation = |sky: &CelestialSky| {
                sky.moons[index]
                    .direction
                    .angle_between(sky.stars[0].direction)
                    .to_degrees()
            };
            assert!(samples.iter().any(|sky| elongation(sky) < 45.0));
            assert!(samples.iter().any(|sky| elongation(sky) > 135.0));
            let new = samples
                .iter()
                .min_by(|a, b| fraction(a).total_cmp(&fraction(b)))
                .unwrap();
            // The brightest full moon outshines the new one many times over;
            // a full moon may also sit in the planet's shadow.
            let brightest = samples
                .iter()
                .map(|sky| sky.moons[index].irradiance)
                .fold(0.0, f32::max);
            assert!(brightest > 20.0 * new.moons[index].irradiance);
        }
        // The same clock time finds the first moon elsewhere on the next night.
        let midnight = |day: f64| system.sky(day).moons[0].direction;
        assert!(midnight(10.0).distance(midnight(11.0)) > 0.05);
    }
}

#[test]
fn a_moon_crossing_the_planet_shadow_goes_dark() {
    let mut system = CelestialSystem::generate(WorldSeed(1));
    system.moons = vec![CelestialMoon {
        radius_km: 1_700.0,
        orbit_radii: 30.0,
        albedo: 0.3,
        tint: Vec3::ONE,
        pattern: 0,
        orbit: Orbit {
            inclination: 0.0,
            node: 0.0,
            phase: 0.0,
            period: 7.0,
        },
    }];
    let skies = (0..2_000)
        .map(|step| system.sky(f64::from(step) * 0.005))
        .collect::<Vec<_>>();
    let light = |sky: &CelestialSky| sky.moons[0].lit[0] / sky.stars[0].irradiance;
    assert!(skies.iter().any(|sky| light(sky) < 0.01));
    assert!(skies.iter().filter(|sky| light(sky) > 0.99).count() > 1_800);
}

#[test]
fn circumbinary_suns_part_and_meet_and_light_like_one() {
    for system in systems().filter(CelestialSystem::circumbinary).take(10) {
        let pair = system.inner.pair.unwrap();
        let mut light = 0.0;
        let separations = (0..200)
            .map(|step| {
                let sky = system.sky(2.0 + pair.orbit.period * f64::from(step) / 200.0);
                // The planet swings nearer each sun in turn, and one sun
                // can pass in front of the other.
                let total = sky.stars[0].irradiance + sky.stars[1].irradiance;
                assert!((0.2..1.5).contains(&total), "{total}");
                light += total / 200.0;
                sky.stars[0]
                    .direction
                    .angle_between(sky.stars[1].direction)
                    .to_degrees()
            })
            .collect::<Vec<_>>();
        assert!((0.85..1.15).contains(&light), "{light}");
        let widest = separations.iter().copied().fold(0.0, f32::max);
        let closest = separations.iter().copied().fold(f32::MAX, f32::min);
        assert!(widest > 5.0 && widest < 30.0);
        assert!(closest < widest * 0.5);
    }
}

#[test]
fn distant_companions_are_faint_and_nearly_fixed_among_the_stars() {
    for system in systems()
        .filter(|system| system.companion.is_some())
        .take(10)
    {
        let index = system.companion.unwrap().group.first;
        let ecliptic = |days: f64| {
            let sky = system.sky(days);
            (
                sky.rotation.inverse() * sky.stars[index].direction,
                sky.stars[index].irradiance,
            )
        };
        let (start, light) = ecliptic(0.0);
        let (later, _) = ecliptic(system.year_days() / 2.0);
        assert!(light > 1e-6 && light < 0.01);
        assert!(start.angle_between(later).to_degrees() < 3.0);
    }
}

#[test]
fn covered_fraction_matches_disk_geometry() {
    assert!(covered_fraction(1.0, 0.5, 1.5).abs() < 1e-12);
    assert!((covered_fraction(1.0, 2.0, 0.5) - 1.0).abs() < 1e-12);
    assert!((covered_fraction(1.0, 0.5, 0.2) - 0.25).abs() < 1e-12);
    let half = 2.0 * 0.5_f64.acos() - 0.5 * 3.0_f64.sqrt();
    assert!((covered_fraction(1.0, 1.0, 1.0) - half / PI).abs() < 1e-12);
    // Continuous where the partial overlap begins and ends.
    assert!(covered_fraction(1.0, 0.5, 1.499_999).abs() < 1e-3);
    assert!((covered_fraction(1.0, 0.5, 0.500_001) - 0.25).abs() < 1e-3);
}
