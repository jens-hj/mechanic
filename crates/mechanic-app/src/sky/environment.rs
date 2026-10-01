//! Refresh slowly changing sky reflections without reallocating their textures.

use super::{MainCamera, SkyState, WorldRuntime};
use bevy::pbr::{
    AtmosphereProbePipeline,
    generate::{EnvironmentMapGeneration, GeneratorBindGroups, GeneratorPipelines},
};
use bevy::prelude::*;
use bevy::render::{Render, RenderApp, render_resource::PipelineCache};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Resource)]
pub(super) struct EnvironmentRefresh {
    pub(super) period_seconds: f64,
    next: f64,
    remaining_frames: u8,
    last_solar: f64,
    last_altitude: f64,
    outdoors: bool,
}

impl Default for EnvironmentRefresh {
    fn default() -> Self {
        Self {
            period_seconds: 0.25,
            next: 0.0,
            remaining_frames: 0,
            last_solar: 0.0,
            last_altitude: 0.0,
            outdoors: false,
        }
    }
}

impl EnvironmentRefresh {
    fn update(&mut self, now: f64, solar: f64, altitude: f64, ready: bool) -> bool {
        let solar_delta = (solar - self.last_solar).abs();
        let solar_delta = solar_delta.min(mechanic_core::SECONDS_PER_DAY - solar_delta);
        if !self.outdoors
            || !ready
            || now >= self.next
            || solar_delta > 120.0
            || (altitude - self.last_altitude).abs() > 2.0
        {
            // Bevy filters before drawing the atmosphere source cubemap. Two
            // consecutive frames publish the new source and then its filtering.
            self.remaining_frames = 2;
            self.next = now + self.period_seconds;
            self.last_solar = solar;
            self.last_altitude = altitude;
        }
        self.outdoors = true;
        let enabled = self.remaining_frames > 0;
        self.remaining_frames = self.remaining_frames.saturating_sub(1);
        enabled
    }
}

#[derive(Resource, Clone, Default)]
pub(super) struct Ready(Arc<AtomicBool>);

pub(super) fn install(app: &mut App) {
    let ready = Ready::default();
    app.init_resource::<EnvironmentRefresh>()
        .insert_resource(ready.clone());
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.insert_resource(ready).add_systems(
            Render,
            mark_ready.after(bevy::pbr::generate::filtering_system),
        );
    }
}

pub(super) fn update(
    mut commands: Commands,
    time: Res<Time<Real>>,
    sky: Res<SkyState>,
    runtime: Res<WorldRuntime>,
    ready: Res<Ready>,
    mut refresh: ResMut<EnvironmentRefresh>,
    camera: Query<(Entity, &Transform), With<MainCamera>>,
) {
    let Ok((entity, camera)) = camera.single() else {
        return;
    };
    if !sky.outdoors {
        refresh.outdoors = false;
        commands.entity(entity).remove::<EnvironmentMapGeneration>();
        return;
    }
    let enabled = refresh.update(
        time.elapsed_secs_f64(),
        sky.displayed_seconds(&runtime),
        f64::from(camera.translation.y) + runtime.origin().0.y,
        ready.0.load(Ordering::Acquire),
    );
    commands
        .entity(entity)
        .insert(EnvironmentMapGeneration { enabled });
}

fn mark_ready(
    ready: Res<Ready>,
    maps: Query<(&GeneratorBindGroups, &EnvironmentMapGeneration)>,
    cache: Res<PipelineCache>,
    filter: Option<Res<GeneratorPipelines>>,
    atmosphere: Option<Res<AtmosphereProbePipeline>>,
) {
    let (Some(filter), Some(atmosphere)) = (filter, atmosphere) else {
        return;
    };
    if maps.iter().any(|(_, update)| update.enabled)
        && [
            filter.copy,
            filter.downsample_first,
            filter.downsample_second,
            filter.radiance,
            filter.irradiance,
            atmosphere.environment,
        ]
        .into_iter()
        .all(|pipeline| cache.get_compute_pipeline(pipeline).is_some())
    {
        ready.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_publish_source_then_filter_and_retain_maps_between_refreshes() {
        let mut refresh = EnvironmentRefresh::default();
        assert!(refresh.update(0.0, 100.0, 5.0, true));
        assert!(refresh.update(0.016, 100.4, 5.0, true));
        assert!(!refresh.update(0.032, 100.8, 5.0, true));
        assert!(!refresh.update(0.249, 105.9, 5.0, true));
        assert!(refresh.update(0.25, 106.0, 5.0, true));
    }

    #[test]
    fn time_jumps_altitude_changes_and_reentry_refresh_immediately() {
        let mut refresh = EnvironmentRefresh::default();
        for time in [0.0, 0.016] {
            assert!(refresh.update(time, 100.0, 5.0, true));
        }
        assert!(refresh.update(0.032, 3700.0, 5.0, true));
        assert!(refresh.update(0.048, 3700.0, 9.0, true));
        assert!(refresh.update(0.064, 3700.0, 9.0, true));
        assert!(!refresh.update(0.08, 3700.0, 9.0, true));
        refresh.outdoors = false;
        assert!(refresh.update(0.096, 3700.0, 9.0, true));
    }

    #[test]
    fn shader_warmup_keeps_generation_running_and_midnight_does_not_force_updates() {
        let mut refresh = EnvironmentRefresh::default();
        for time in [0.0, 0.016, 0.032] {
            assert!(refresh.update(time, 86399.0, 5.0, false));
        }
        assert!(refresh.update(0.048, 0.0, 5.0, true));
        assert!(!refresh.update(0.064, 0.4, 5.0, true));
    }
}
