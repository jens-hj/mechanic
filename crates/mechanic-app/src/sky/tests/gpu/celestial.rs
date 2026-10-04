//! Opt-in captures of the star system: twin suns, moon phases, a daytime
//! moon, the galaxy, and a distant companion star, with frame cost.

use super::*;
use mechanic_world::{CelestialSky, CelestialSystem, WorldSeed};

fn find(predicate: impl Fn(&CelestialSystem) -> bool) -> CelestialSystem {
    (0..10_000)
        .map(|seed| CelestialSystem::generate(WorldSeed(seed)))
        .find(predicate)
        .unwrap()
}

/// First moment from day one, in steps of `step` days, when `predicate`
/// holds.
fn when(system: &CelestialSystem, step: f64, predicate: impl Fn(&CelestialSky) -> bool) -> f64 {
    (0..200_000)
        .map(|index| 1.0 + step * f64::from(index))
        .find(|&days| predicate(&system.sky(days)))
        .expect("the sky reaches the wanted arrangement")
}

struct Shot {
    name: String,
    system: CelestialSystem,
    days: f64,
    look: fn(&CelestialSky) -> Vec3,
    fov_degrees: f32,
}

fn sun(sky: &CelestialSky) -> Vec3 {
    sky.stars[0].direction
}

fn first_moon(sky: &CelestialSky) -> Vec3 {
    sky.moons[0].direction
}

fn zenith(_: &CelestialSky) -> Vec3 {
    Vec3::new(0.0, 1.0, -0.35).normalize()
}

fn galaxy(sky: &CelestialSky) -> Vec3 {
    sky.rotation * crate::sky::night::galactic_core()
}

fn companion(sky: &CelestialSky) -> Vec3 {
    sky.stars[sky.stars.len() - 1].direction
}

fn shots() -> Vec<Shot> {
    let mut shots = Vec::new();
    let twins = find(CelestialSystem::circumbinary);
    let dusk = when(&twins, 0.001, |sky| {
        let (a, b) = (sky.stars[0].direction, sky.stars[1].direction);
        (0.02..0.12).contains(&a.y) && b.y > 0.02 && a.x < 0.0 && a.angle_between(b) > 0.12
    });
    shots.push(Shot {
        name: "twin-suns-dusk".into(),
        system: twins.clone(),
        days: dusk,
        look: sun,
        fov_degrees: 70.0,
    });
    let noon = when(&twins, 0.001, |sky| {
        sky.stars[0].direction.y > 0.5
            && sky.stars[0].direction.angle_between(sky.stars[1].direction) > 0.12
    });
    shots.push(Shot {
        name: "twin-suns-day".into(),
        system: twins,
        days: noon,
        look: sun,
        fov_degrees: 70.0,
    });
    let moons = find(|system| system.moons().len() >= 2);
    for target in [0.1, 0.5, 0.9] {
        let days = when(&moons, 0.002, |sky| {
            // A crescent never stands high in a dark sky.
            sky.moons[0].direction.y > 0.15
                && sky.stars[0].direction.y < -0.1
                && (sky.moons[0].illuminated_fraction - target).abs() < 0.03
        });
        shots.push(Shot {
            name: format!("moon-{:.0}-percent", target * 100.0),
            system: moons.clone(),
            days,
            look: first_moon,
            fov_degrees: 6.0,
        });
    }
    let day_moon = when(&moons, 0.002, |sky| {
        sky.stars[0].direction.y > 0.4
            && sky.moons[0].direction.y > 0.3
            && sky.moons[0].illuminated_fraction > 0.4
    });
    shots.push(Shot {
        name: "moon-by-day".into(),
        system: moons.clone(),
        days: day_moon,
        look: first_moon,
        fov_degrees: 25.0,
    });
    let moonless = when(&moons, 0.002, |sky| {
        sky.stars[0].direction.y < -0.5 && sky.moons.iter().all(|moon| moon.direction.y < -0.1)
    });
    shots.push(Shot {
        name: "night-sky".into(),
        system: moons.clone(),
        days: moonless,
        look: zenith,
        fov_degrees: 100.0,
    });
    let core_up = when(&moons, 0.002, |sky| {
        sky.stars[0].direction.y < -0.5
            && galaxy(sky).y > 0.4
            && sky.moons.iter().all(|moon| moon.direction.y < -0.1)
    });
    shots.push(Shot {
        name: "galaxy".into(),
        system: moons.clone(),
        days: core_up,
        look: galaxy,
        fov_degrees: 60.0,
    });
    let distant = find(|system| system.stars().len() == 3 && !system.circumbinary());
    let night = when(&distant, 0.002, |sky| {
        sky.stars[0].direction.y < -0.4 && companion(sky).y > 0.3
    });
    shots.push(Shot {
        name: "distant-pair-at-night".into(),
        system: distant,
        days: night,
        look: companion,
        fov_degrees: 50.0,
    });
    shots
}

fn show(app: &mut App, camera: Entity, shot: &Shot) {
    let seed = app.world().resource::<WorldRuntime>().seed();
    {
        let mut sky = app.world_mut().resource_mut::<SkyState>();
        sky.system = Some((seed, shot.system.clone()));
        sky.fixed_seconds = Some(shot.days * SECONDS_PER_DAY);
    }
    let direction = (shot.look)(&shot.system.sky(shot.days));
    let up = if direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    app.world_mut().entity_mut(camera).insert((
        Transform::from_xyz(0.0, 3.0, 10.0).looking_to(direction, up),
        Projection::Perspective(PerspectiveProjection {
            fov: shot.fov_degrees.to_radians(),
            ..default()
        }),
    ));
    for _ in 0..40 {
        frame(app);
    }
}

#[test]
#[ignore = "real GPU capture of suns, moons, and stars"]
fn celestial_captures_show_suns_moons_and_stars() {
    let Fixture {
        mut app,
        camera,
        readback,
        directory,
        adapter,
    } = fixture();
    scene::ground(&mut app);
    app.insert_resource(State::new(AppSpace::World));
    app.world_mut().entity_mut(camera).insert(DistanceFog {
        color: Color::NONE,
        falloff: FogFalloff::Exponential { density: 0.0 },
        ..default()
    });
    let mut report = Vec::new();
    for shot in shots() {
        show(&mut app, camera, &shot);
        save_image(&app, &directory, &shot.name);
        let sky = shot.system.sky(shot.days);
        report.push(serde_json::json!({
            "shot": shot.name,
            "days": shot.days,
            "stars": shot.system.stars().len(),
            "moons": sky.moons.iter().map(|moon| moon.illuminated_fraction).collect::<Vec<_>>(),
        }));
    }
    // Cost of the busiest system against a lone sun, at night and by day.
    let lone = find(|system| system.stars().len() == 1 && system.moons().is_empty());
    let busy = find(|system| system.stars().len() == 3 && system.moons().len() == 3);
    let mut costs = Vec::new();
    for (name, system) in [
        ("one sun, no moons", lone),
        ("three stars, three moons", busy),
    ] {
        for hours in [12.0, 0.0] {
            let shot = Shot {
                name: name.into(),
                system: system.clone(),
                days: 20.0 + hours / 24.0,
                look: zenith,
                fov_degrees: 70.0,
            };
            show(&mut app, camera, &shot);
            let (mean, p95) = timings(&mut app, readback);
            costs.push(
                serde_json::json!({"system": name, "hour": hours, "mean_ms": mean, "p95_ms": p95}),
            );
        }
    }
    let report = serde_json::json!({"adapter": adapter, "resolution": [WIDTH, HEIGHT],
        "measurement": "synchronized offscreen CPU+GPU frame latency, readback disabled",
        "shots": report, "costs": costs});
    std::fs::write(
        directory.join("celestial.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("{report}");
}
