mod gpu;

use super::*;
use mechanic_world::{FloatingOrigin, SkyMoon, SkyStar};

fn star(direction: Vec3, irradiance: f32) -> SkyStar {
    SkyStar {
        direction: direction.normalize(),
        angular_radius: 0.0047,
        irradiance,
        colour: Vec3::ONE,
    }
}

fn moon(direction: Vec3, irradiance: f32) -> SkyMoon {
    SkyMoon {
        direction: direction.normalize(),
        angular_radius: 0.005,
        irradiance,
        lit: [1.0, 0.0, 0.0],
        illuminated_fraction: 0.5,
        body: Mat3::IDENTITY,
        albedo: 0.2,
        tint: Vec3::ONE,
        pattern: 0,
    }
}

fn sky_of(stars: Vec<SkyStar>, moons: Vec<SkyMoon>) -> CelestialSky {
    CelestialSky {
        rotation: Quat::IDENTITY,
        stars,
        moons,
    }
}

#[test]
fn only_the_brightest_light_above_the_horizon_casts_shadows() {
    let up = Vec3::new(0.3, 0.6, 0.4);
    let down = Vec3::new(0.3, -0.6, 0.4);
    let lit = |sun: Vec3, companion: Vec3, moons: Vec<SkyMoon>| {
        lights::plan(&sky_of(vec![star(sun, 1.0), star(companion, 4e-4)], moons)).shadows
    };
    assert_eq!(lit(up, up, vec![moon(up, 1e-6)]), Some(0));
    assert_eq!(lit(down, up, vec![moon(up, 1e-6)]), Some(1));
    assert_eq!(lit(down, down, vec![moon(up, 1e-6)]), Some(2));
    assert_eq!(lit(down, down, vec![moon(down, 1e-6)]), None);
    // Risen moons pool their light along the brightest; set ones add none.
    let plan = lights::plan(&sky_of(
        vec![star(down, 1.0)],
        vec![
            moon(up, 2e-6),
            moon(-up.reflect(Vec3::Y), 1e-6),
            moon(down, 9e-6),
        ],
    ));
    assert!(plan.moon.direction.distance(up.normalize()) < 1e-6);
    assert!(
        (plan.moon.lux / (3e-6 * bevy::light::light_consts::lux::RAW_SUNLIGHT) - 13.0).abs() < 1e-3
    );
}

#[test]
fn exposure_adapts_to_suns_and_to_bright_nights() {
    let noon = Vec3::new(0.0, std::f32::consts::FRAC_PI_3.sin(), 0.5);
    let night = Vec3::new(0.0, -0.8, 0.6);
    let ev = |sky: CelestialSky| lights::ev100(&sky, &lights::plan(&sky));
    assert!((ev(sky_of(vec![star(noon, 1.0)], vec![])) - 13.0).abs() < 1e-3);
    // Two suns together light the day more than one.
    assert!(
        ev(sky_of(vec![star(noon, 0.8), star(noon, 0.6)], vec![]))
            > ev(sky_of(vec![star(noon, 1.0)], vec![]))
    );
    let dark = ev(sky_of(vec![star(night, 1.0)], vec![]));
    assert!((dark - 1.0).abs() < 1e-3);
    let moonlit = ev(sky_of(vec![star(night, 1.0)], vec![moon(noon, 2e-5)]));
    assert!(moonlit > dark + 1.0);
    let companion = ev(sky_of(vec![star(night, 1.0), star(noon, 1e-3)], vec![]));
    assert!(companion > dark + 1.0 && companion < 10.0);
}

#[test]
fn moon_shading_lights_the_side_facing_each_star() {
    let north = moon(Vec3::NEG_Z, 1e-6);
    let stars = [star(Vec3::X, 1.0), star(Vec3::NEG_X, 0.5)];
    let mut lit = north;
    lit.lit = [1.0, 0.5, 0.0];
    let quad = moons::facing(north.direction);
    let uniform = moons::uniform(&lit, &stars, 30.0, quad);
    let east = uniform.light_direction[0];
    let west = uniform.light_direction[1];
    assert!(east.truncate().distance(Vec3::X) < 1e-5);
    assert!(west.truncate().distance(Vec3::NEG_X) < 1e-5);
    let raw = bevy::light::light_consts::lux::RAW_SUNLIGHT;
    assert!((east.w - raw).abs() < 1.0 && (west.w - raw * 0.5).abs() < 1.0);
    assert_eq!(uniform.light_direction[2], Vec4::ZERO);
    // The quad faces the camera: its z axis points back along the view.
    assert!((Mat3::from_quat(quad) * Vec3::Z).distance(Vec3::Z) < 1e-5);
    assert!(uniform.planetshine.x > 0.0);
}

#[test]
fn outdoor_sky_keeps_a_light_per_star_and_a_disk_per_moon() {
    let system = (0..10_000)
        .map(|seed| CelestialSystem::generate(WorldSeed(seed)))
        .find(|system| system.stars().len() == 3 && system.moons().len() >= 2)
        .unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>()
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<MoonMaterial>>()
        .add_systems(Startup, moons::setup)
        .add_systems(Update, (update_sky, lights::place, moons::place).chain());
    let seed = app.world().resource::<WorldRuntime>().seed();
    app.insert_resource(SkyState {
        outdoors: true,
        system: Some((seed, system.clone())),
        fixed_seconds: Some(3.5 * SECONDS_PER_DAY),
        ..default()
    });
    let camera = app
        .world_mut()
        .spawn((
            MainCamera,
            Transform::from_xyz(5.0, 2.0, -3.0),
            Exposure::default(),
            Skybox::default(),
        ))
        .id();
    app.world_mut().spawn((
        Sun,
        DirectionalLight::default(),
        bevy::light::SunDisk::EARTH,
        Transform::default(),
    ));
    let mut commands = app.world_mut().commands();
    lights::spawn_moonlight(&mut commands);
    app.world_mut().flush();
    for _ in 0..3 {
        app.update();
    }
    let world = app.world_mut();
    let suns = world
        .query_filtered::<&DirectionalLight, With<Sun>>()
        .iter(world)
        .count();
    let disks = world.query::<&MoonDisk>().iter(world).count();
    let shadows = world
        .query::<&DirectionalLight>()
        .iter(world)
        .filter(|light| light.shadow_maps_enabled)
        .count();
    assert_eq!(suns, 3);
    assert_eq!(disks, system.moons().len());
    assert!(shadows <= 1);
    let expected = system.sky(3.5).rotation;
    let rotation = app.world().entity(camera).get::<Skybox>().unwrap().rotation;
    assert!(rotation.angle_between(expected) < 1e-4);
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "exact integral clock steps and lossless persistence"
)]
fn clock_steps_carry_whole_days_and_never_run_before_day_zero() {
    let mut runtime = WorldRuntime::from_world(&mut World::new());
    runtime.advance_day(-runtime.time_of_day_seconds());
    runtime.advance_day(30.0 * 3600.0);
    assert_eq!(runtime.solar_days(), 1.25);
    runtime.advance_day(-2.0 * SECONDS_PER_DAY);
    assert_eq!(runtime.solar_days(), 0.25);
    runtime.advance_day(SECONDS_PER_DAY * 3.0 + 18.0 * 3600.0);
    assert_eq!(runtime.solar_days(), 4.0);
    assert_eq!(runtime.time_of_day_seconds(), 0.0);
}

#[test]
fn atmosphere_altitude_is_invariant_under_floating_origin_rebase() {
    let radius = 6_360_000.0;
    let global = Vec3::new(5000.0, 240.0, -3000.0);
    let origin = FloatingOrigin(Vec3::new(4800.0, 128.0, -3200.0).as_dvec3());
    let local = global - origin.0.as_vec3();
    let before = planet_transform(radius, global, FloatingOrigin::default());
    let after = planet_transform(radius, local, origin);
    assert_eq!(global - before.translation(), local - after.translation());
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "exact integral clock steps and lossless persistence"
)]
fn hour_adjustments_wrap_midnight_and_dirty_the_save() {
    let mut runtime = WorldRuntime::from_world(&mut World::new());
    runtime.advance_day(-runtime.time_of_day_seconds());
    runtime.advance_day(-3600.0);
    assert_eq!(runtime.time_of_day_seconds(), 23.0 * 3600.0);
    runtime.advance_day(3600.0);
    assert_eq!(runtime.time_of_day_seconds(), 0.0);
}

#[test]
fn sixty_real_minutes_complete_a_day_independent_of_frame_rate() {
    for frames in [3600_u32, 216_000] {
        let mut runtime = WorldRuntime::from_world(&mut World::new());
        let initial = runtime.time_of_day_seconds();
        for _ in 0..frames {
            runtime
                .advance_day(CYCLE_SECONDS / f64::from(frames) * SECONDS_PER_DAY / CYCLE_SECONDS);
        }
        assert!((runtime.time_of_day_seconds() - initial).abs() < 1e-5);
    }
}

fn clock_app() -> App {
    let mut app = App::new();
    app.init_resource::<WorldRuntime>()
        .insert_resource(WorldListState::empty_capture_garage())
        .insert_resource(State::new(AppSpace::World))
        .init_resource::<PauseMenuState>()
        .init_resource::<DevTools>()
        .init_resource::<SkyState>()
        .init_resource::<Time<Real>>()
        .init_resource::<crate::debug_freeze::DebugFrameFreeze>()
        .add_systems(
            Update,
            advance_clock.run_if(crate::debug_freeze::debug_frame_updates_enabled),
        );
    app.world_mut()
        .resource_mut::<Time<Real>>()
        .advance_by(std::time::Duration::from_secs(1));
    app
}

#[test]
#[expect(
    clippy::float_cmp,
    reason = "exact integral clock steps and lossless persistence"
)]
fn clock_freezes_in_menus_garage_debug_freeze_and_visual_fixtures() {
    for condition in 0..6 {
        let mut app = clock_app();
        let initial = app.world().resource::<WorldRuntime>().time_of_day_seconds();
        match condition {
            0 => app.world_mut().resource_mut::<PauseMenuState>().open(),
            1 => {
                app.insert_resource(State::new(AppSpace::Garage));
            }
            2 => {
                app.world_mut().remove_resource::<WorldListState>();
                app.init_resource::<WorldListState>();
            }
            3 => app.world_mut().resource_mut::<DevTools>().cycle_paused = true,
            4 => app.world_mut().resource_mut::<SkyState>().fixed_seconds = Some(0.0),
            _ => {
                app.world_mut()
                    .resource_mut::<crate::debug_freeze::DebugFrameFreeze>()
                    .active = true;
            }
        }
        app.update();
        assert_eq!(
            app.world().resource::<WorldRuntime>().time_of_day_seconds(),
            initial
        );
    }
    let mut app = clock_app();
    let initial = app.world().resource::<WorldRuntime>().time_of_day_seconds();
    app.update();
    assert_eq!(
        app.world().resource::<WorldRuntime>().time_of_day_seconds(),
        initial + 24.0
    );
}

#[test]
fn repeated_space_transitions_restore_maps_and_keep_outdoor_generation_live() {
    let mut app = App::new();
    app.init_resource::<SkyState>()
        .insert_resource(State::new(AppSpace::World))
        .add_systems(Update, switch_space);
    let mut images = Assets::<Image>::default();
    let garage_image = images.add(Image::default());
    let world_image = images.add(Image::default());
    let garage_map = EnvironmentMapLight {
        diffuse_map: garage_image.clone(),
        ..default()
    };
    let world_map = EnvironmentMapLight {
        diffuse_map: world_image.clone(),
        ..default()
    };
    let camera = app
        .world_mut()
        .spawn((MainCamera, garage_map, StaticEnvironmentMap))
        .id();
    let overlay = app.world_mut().spawn(Camera3d::default()).id();
    for _ in 0..3 {
        app.insert_resource(State::new(AppSpace::World));
        app.update();
        let view = app.world().entity(camera);
        assert!(view.contains::<AtmosphereSettings>());
        assert!(!view.contains::<StaticEnvironmentMap>());
        // Emulate Bevy's first generated outdoor probe; following visits must restore it.
        app.world_mut().entity_mut(camera).insert((
            world_map.clone(),
            GeneratedEnvironmentMapLight {
                environment_map: world_image.clone(),
                ..default()
            },
        ));
        app.insert_resource(State::new(AppSpace::Garage));
        app.update();
        let view = app.world().entity(camera);
        assert!(!view.contains::<AtmosphereSettings>());
        assert!(!view.contains::<Skybox>());
        assert!(!view.contains::<AmbientLight>());
        assert!(view.contains::<StaticEnvironmentMap>());
        assert_eq!(
            view.get::<EnvironmentMapLight>().unwrap().diffuse_map,
            garage_image
        );
        assert!(!view.contains::<GeneratedEnvironmentMapLight>());
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<Moon>>()
                .iter(app.world())
                .count(),
            0
        );
        app.insert_resource(State::new(AppSpace::World));
        app.update();
        let view = app.world().entity(camera);
        assert_eq!(
            view.get::<GeneratedEnvironmentMapLight>()
                .unwrap()
                .environment_map,
            world_image
        );
        assert_eq!(
            view.get::<EnvironmentMapLight>().unwrap().diffuse_map,
            world_image
        );
        assert!(!app.world().entity(overlay).contains::<AtmosphereSettings>());
    }
}

#[test]
fn queued_static_completion_cannot_remove_a_new_outdoor_generator() {
    use crate::render::environment::{
        EnvironmentMapGenerationReady, retain_generated_environment_map,
    };
    fn enter_outdoors(mut commands: Commands, cameras: Query<Entity, With<MainCamera>>) {
        for camera in &cameras {
            commands.entity(camera).remove::<StaticEnvironmentMap>();
        }
    }
    let mut app = App::new();
    let ready = EnvironmentMapGenerationReady::default();
    ready.0.store(true, std::sync::atomic::Ordering::Release);
    app.insert_resource(ready).add_systems(
        Update,
        (enter_outdoors, retain_generated_environment_map).chain_ignore_deferred(),
    );
    let camera = app
        .world_mut()
        .spawn((
            MainCamera,
            StaticEnvironmentMap,
            EnvironmentMapLight::default(),
            GeneratedEnvironmentMapLight::default(),
        ))
        .id();
    app.update();
    assert!(
        app.world()
            .entity(camera)
            .contains::<GeneratedEnvironmentMapLight>()
    );
}
