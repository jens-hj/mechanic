mod gpu;

use super::*;
use mechanic_world::FloatingOrigin;

#[test]
fn sunrise_noon_sunset_and_full_moon_share_one_orbit() {
    for hour in [6.0, 18.0] {
        let sun = celestial_rotation(hour * 3600.0) * Vec3::X;
        assert!(sun.y.abs() < 1e-6);
    }
    let noon = celestial_rotation(12.0 * 3600.0) * Vec3::X;
    assert!((noon.y.asin().to_degrees() - 60.0).abs() < 1e-4);
    let midnight_moon = celestial_rotation(0.0) * Vec3::NEG_X;
    assert!(midnight_moon.distance(noon) < 1e-6);
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
