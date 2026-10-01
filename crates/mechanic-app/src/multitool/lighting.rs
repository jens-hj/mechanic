//! Share scene lighting with the transparent first-person camera.
//!
//! Only the main camera generates atmosphere LUTs and reflections. The tool
//! borrows those textures for PBR sunlight attenuation without a sky pass.

use bevy::camera::Exposure;
use bevy::camera::visibility::RenderLayers;
use bevy::pbr::{
    ExtractedAtmosphere,
    resources::{AtmosphereTextures, GpuAtmosphere, prepare_atmosphere_uniforms},
};
use bevy::prelude::*;
use bevy::render::{
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems, sync_world::RenderEntity,
};

use super::VIEWMODEL_LAYER;
use crate::{camera::MainCamera, schedule::FrameSet, world::AppSpace};

#[derive(Component)]
pub(crate) struct ViewmodelCamera;

#[derive(Component)]
pub(super) struct ViewmodelKeyLight;

#[derive(Resource)]
struct LightingViews {
    scene: Entity,
    tool: Entity,
    outdoors: bool,
}

pub(crate) struct ViewmodelLightingPlugin;

impl Plugin for ViewmodelLightingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, sync_lighting.in_set(FrameSet::ViewmodelLighting));
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .add_systems(ExtractSchedule, extract_views)
                .add_systems(
                    Render,
                    (
                        share_atmosphere
                            .after(RenderSystems::ExtractCommands)
                            .before(RenderSystems::PrepareAssets)
                            .before(RenderSystems::PrepareViews),
                        remove_tool_uniform
                            .after(prepare_atmosphere_uniforms)
                            .before(RenderSystems::PrepareResources)
                            .in_set(RenderSystems::Prepare),
                        share_textures
                            .after(RenderSystems::PrepareResources)
                            .before(RenderSystems::PrepareBindGroups)
                            .in_set(RenderSystems::Prepare),
                    ),
                );
        }
    }
}

#[expect(
    clippy::type_complexity,
    reason = "scene camera lighting is optional during transitions"
)]
fn sync_lighting(
    mut commands: Commands,
    space: Res<State<AppSpace>>,
    scene: Query<
        (
            &Exposure,
            Option<&EnvironmentMapLight>,
            Option<&AmbientLight>,
        ),
        With<MainCamera>,
    >,
    tools: Query<Entity, With<ViewmodelCamera>>,
    mut keys: Query<&mut PointLight, With<ViewmodelKeyLight>>,
    lights: Query<
        (Entity, Option<&RenderLayers>),
        Or<(With<crate::sky::Sun>, With<crate::sky::Moon>)>,
    >,
) {
    let Ok((exposure, environment, ambient)) = scene.single() else {
        return;
    };
    for tool in &tools {
        let mut tool = commands.entity(tool);
        tool.insert(*exposure);
        if let Some(map) = environment {
            tool.insert(map.clone());
        } else {
            tool.remove::<EnvironmentMapLight>();
        }
        if let Some(ambient) = ambient {
            tool.insert(ambient.clone());
        } else {
            tool.remove::<AmbientLight>();
        }
    }
    let outdoors = *space.get() == AppSpace::World;
    for mut key in &mut keys {
        key.intensity = if outdoors { 0.0 } else { 1_800.0 };
    }
    // Include celestial lights in the tool view, preserving their world layer
    // and shadows. Garage lighting keeps its authored local key light.
    if outdoors {
        for (entity, layers) in &lights {
            let layers = layers.cloned().unwrap_or_default();
            if layers.intersects(&RenderLayers::layer(0))
                && !layers.intersects(&RenderLayers::layer(VIEWMODEL_LAYER))
            {
                commands.entity(entity).insert(layers.with(VIEWMODEL_LAYER));
            }
        }
    }
}

fn extract_views(
    mut commands: Commands,
    scene: Extract<Query<RenderEntity, With<MainCamera>>>,
    tool: Extract<Query<RenderEntity, With<ViewmodelCamera>>>,
    space: Extract<Res<State<AppSpace>>>,
) {
    if let (Ok(scene), Ok(tool)) = (scene.single(), tool.single()) {
        commands.insert_resource(LightingViews {
            scene,
            tool,
            outdoors: *space.get() == AppSpace::World,
        });
    } else {
        commands.remove_resource::<LightingViews>();
    }
}

fn share_atmosphere(
    mut commands: Commands,
    views: Option<Res<LightingViews>>,
    atmospheres: Query<&ExtractedAtmosphere>,
) {
    let Some(views) = views else {
        return;
    };
    if views.outdoors
        && let Ok(atmosphere) = atmospheres.get(views.scene)
    {
        commands.entity(views.tool).insert((
            atmosphere.clone(),
            bevy::pbr::DirectionalShadowSource(views.scene),
        ));
    } else {
        commands.entity(views.tool).remove::<(
            ExtractedAtmosphere,
            AtmosphereTextures,
            bevy::pbr::DirectionalShadowSource,
        )>();
    }
}

fn remove_tool_uniform(mut commands: Commands, views: Option<Res<LightingViews>>) {
    let Some(views) = views else {
        return;
    };
    // Bevy's PBR atmosphere buffer is populated from a single camera uniform.
    // This view consumes that shared buffer, never owns a second atmosphere.
    commands.entity(views.tool).remove::<GpuAtmosphere>();
}

fn share_textures(
    mut commands: Commands,
    views: Option<Res<LightingViews>>,
    textures: Query<&AtmosphereTextures>,
) {
    let Some(views) = views else {
        return;
    };
    if views.outdoors
        && let Ok(textures) = textures.get(views.scene)
    {
        commands.entity(views.tool).insert(AtmosphereTextures {
            transmittance_lut: textures.transmittance_lut.clone(),
            multiscattering_lut: textures.multiscattering_lut.clone(),
            sky_view_lut: textures.sky_view_lut.clone(),
            aerial_view_lut: textures.aerial_view_lut.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    mod gpu;

    use super::*;

    #[test]
    fn tool_follows_day_night_and_repeated_garage_transitions() {
        let mut app = App::new();
        app.add_plugins(ViewmodelLightingPlugin)
            .insert_resource(State::new(AppSpace::Garage));
        let scene = app
            .world_mut()
            .spawn((MainCamera, crate::garage::EXPOSURE))
            .id();
        let tool = app
            .world_mut()
            .spawn((ViewmodelCamera, crate::garage::EXPOSURE))
            .id();
        let key = app
            .world_mut()
            .spawn((ViewmodelKeyLight, PointLight::default()))
            .id();
        let sun = app
            .world_mut()
            .spawn((crate::sky::Sun, DirectionalLight::default()))
            .id();
        let garage_light = app.world_mut().spawn(DirectionalLight::default()).id();
        for _ in 0..3 {
            app.insert_resource(State::new(AppSpace::World));
            for ev100 in [13.0, 1.0] {
                app.world_mut().entity_mut(scene).insert((
                    Exposure { ev100 },
                    EnvironmentMapLight {
                        intensity: ev100,
                        ..default()
                    },
                    AmbientLight {
                        brightness: 0.5,
                        ..default()
                    },
                ));
                app.update();
                let view = app.world().entity(tool);
                assert!((view.get::<Exposure>().unwrap().ev100 - ev100).abs() < f32::EPSILON);
                assert!(
                    (view.get::<EnvironmentMapLight>().unwrap().intensity - ev100).abs()
                        < f32::EPSILON
                );
                assert!(
                    (view.get::<AmbientLight>().unwrap().brightness - 0.5).abs() < f32::EPSILON
                );
                assert!(view.get::<GeneratedEnvironmentMapLight>().is_none());
                assert!(view.get::<bevy::pbr::AtmosphereSettings>().is_none());
                assert!(app.world().get::<PointLight>(key).unwrap().intensity.abs() < f32::EPSILON);
                assert!(
                    app.world()
                        .get::<RenderLayers>(sun)
                        .unwrap()
                        .intersects(&RenderLayers::layer(VIEWMODEL_LAYER))
                );
                assert!(app.world().get::<RenderLayers>(garage_light).is_none());
            }
            app.insert_resource(State::new(AppSpace::Garage));
            app.world_mut()
                .entity_mut(scene)
                .insert(crate::garage::EXPOSURE)
                .remove::<(EnvironmentMapLight, AmbientLight)>();
            app.update();
            let view = app.world().entity(tool);
            assert!(
                (view.get::<Exposure>().unwrap().ev100 - crate::garage::EXPOSURE.ev100).abs()
                    < f32::EPSILON
            );
            assert!(view.get::<EnvironmentMapLight>().is_none());
            assert!(view.get::<AmbientLight>().is_none());
            assert!(
                (app.world().get::<PointLight>(key).unwrap().intensity - 1_800.0).abs()
                    < f32::EPSILON
            );
        }
    }

    #[test]
    fn tool_borrows_atmosphere_without_owning_a_second_uniform_or_sky_pass() {
        let mut app = App::new();
        app.add_systems(
            Update,
            (
                share_atmosphere,
                prepare_atmosphere_uniforms,
                remove_tool_uniform,
            )
                .chain(),
        );
        let scene = app
            .world_mut()
            .spawn(ExtractedAtmosphere {
                inner_radius: 6_360_000.0,
                outer_radius: 6_460_000.0,
                ground_albedo: Vec3::splat(0.3),
                medium: default(),
                world_to_atmosphere: Mat4::IDENTITY,
            })
            .id();
        let tool = app.world_mut().spawn_empty().id();
        app.insert_resource(LightingViews {
            scene,
            tool,
            outdoors: true,
        });
        app.update();
        assert!(app.world().get::<ExtractedAtmosphere>(tool).is_some());
        assert!(app.world().get::<GpuAtmosphere>(scene).is_some());
        assert!(app.world().get::<GpuAtmosphere>(tool).is_none());
        assert!(
            app.world()
                .get::<bevy::pbr::GpuAtmosphereSettings>(tool)
                .is_none()
        );
        app.world_mut().resource_mut::<LightingViews>().outdoors = false;
        app.update();
        assert!(app.world().get::<ExtractedAtmosphere>(tool).is_none());
    }
}
