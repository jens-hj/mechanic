//! Paired, settled scene measurements of each optimization independently.
use super::*;
use crate::multitool::lighting::{ViewmodelCamera, ViewmodelLightingPlugin};
use bevy::{
    camera::{CameraOutputMode, visibility::RenderLayers},
    pbr::{DirectionalShadowSource, ViewLightEntities},
    render::{Render, RenderSystems, sync_world::RenderEntity},
};

#[derive(Resource)]
struct ShareShadows(bool);

pub(super) fn install(app: &mut App) {
    app.add_plugins(ViewmodelLightingPlugin);
    app.configure_sets(Update, (FrameSet::Sky, FrameSet::ViewmodelLighting).chain());
    app.sub_app_mut(RenderApp)
        .insert_resource(ShareShadows(true))
        .add_systems(
            Render,
            select_shadows
                .after(RenderSystems::PrepareAssets)
                .before(RenderSystems::CreateViews),
        );
}

fn select_shadows(
    mut commands: Commands,
    mode: Res<ShareShadows>,
    views: Query<Entity, With<DirectionalShadowSource>>,
) {
    if !mode.0 {
        for entity in &views {
            commands.entity(entity).remove::<DirectionalShadowSource>();
        }
    }
}

#[expect(clippy::too_many_lines, reason = "self-contained paired GPU benchmark")]
pub(super) fn compare(
    app: &mut App,
    camera: Entity,
    readback: Entity,
    directory: &std::path::Path,
) {
    app.world_mut().resource_mut::<SkyState>().fixed_seconds = Some(12.0 * 3600.0);
    let (target, transform, projection) = {
        let source = app.world().entity(camera);
        (
            source.get::<RenderTarget>().unwrap().clone(),
            *source.get::<Transform>().unwrap(),
            source.get::<Projection>().unwrap().clone(),
        )
    };
    let tool = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            ViewmodelCamera,
            Camera {
                order: 1,
                clear_color: ClearColorConfig::Custom(Color::NONE),
                output_mode: CameraOutputMode::Write {
                    blend_state: Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING),
                    clear_color: ClearColorConfig::None,
                },
                ..default()
            },
            target,
            transform,
            projection,
            Msaa::Sample4,
            Exposure::OVERCAST,
            bevy::core_pipeline::tonemapping::Tonemapping::SomewhatBoringDisplayTransform,
            RenderLayers::layer(2),
        ))
        .id();
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Sphere::new(0.6));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.5),
            metallic: 0.8,
            perceptual_roughness: 0.25,
            ..default()
        });
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(1.8, 1.3, 5.5),
        RenderLayers::layer(2),
        bevy::light::NotShadowCaster,
    ));
    let mut reports = Vec::new();
    // Reverse alternate rounds to reduce warm-up and thermal order bias.
    for round in 0..3 {
        let mut variants = vec![
            ("before", 0.0, false),
            ("reflections", 0.25, false),
            ("shadows", 0.0, true),
            ("combined", 0.25, true),
        ];
        if round == 1 {
            variants.reverse();
        }
        for (name, period, share) in variants {
            app.world_mut()
                .resource_mut::<super::super::super::environment::EnvironmentRefresh>()
                .period_seconds = period;
            app.sub_app_mut(RenderApp)
                .world_mut()
                .resource_mut::<ShareShadows>()
                .0 = share;
            for _ in 0..24 {
                frame(app);
            }
            let tool_render = app.world().entity(tool).get::<RenderEntity>().unwrap().id();
            let shadow_count = app
                .sub_app(RenderApp)
                .world()
                .get::<ViewLightEntities>(tool_render)
                .unwrap()
                .lights
                .len();
            if share {
                assert_eq!(shadow_count, 0, "tool should reuse the main shadow maps");
            } else {
                assert!(
                    shadow_count > 0,
                    "baseline must render independent tool shadows"
                );
            }
            if round == 0 {
                save_image(app, directory, &format!("optimization-{name}"));
            }
            let (mean, p95) = timings(app, readback);
            reports.push(serde_json::json!({"round": round, "variant": name, "mean_ms": mean, "p95_ms": p95, "tool_shadow_views": shadow_count}));
        }
    }
    let report = serde_json::json!({"resolution": [WIDTH, HEIGHT], "frames_per_sample": 60,
        "measurement": "synchronized CPU+GPU frame latency, no readback, fixed noon, no streaming", "samples": reports});
    std::fs::write(
        directory.join("optimization.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!("Optimization: {report}");
}
