//! Sky cubemap, one-shot environment map generation, and the streaming mesh allocator.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::{
    App, Commands, Entity, EnvironmentMapLight, GeneratedEnvironmentMapLight, Image,
    IntoScheduleConfigs, Plugin, Query, Res, Resource, Update, Vec3, With, default,
};
use bevy::render::mesh::allocator::MeshAllocatorSettings;
use bevy::render::render_resource::{
    Extent3d, PipelineCache, TextureDimension, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::slab_allocator::SlabAllocatorSettings;
use bevy::render::{Render, RenderApp};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared signal from the render world once Bevy has populated both filtered
/// environment maps.
#[derive(Resource, Clone, Default)]
pub(crate) struct EnvironmentMapGenerationReady(pub(crate) Arc<AtomicBool>);

/// Retains Bevy's filtered environment map but turns its generator into a
/// one-shot operation.
pub(crate) struct OneShotEnvironmentMapPlugin;

/// Avoids repeatedly reallocating tiny GPU mesh slabs while the terrain
/// horizon publishes thousands of chunks over successive frames.
pub(crate) struct StreamingMeshAllocatorPlugin;

impl Plugin for StreamingMeshAllocatorPlugin {
    fn build(&self, app: &mut App) {
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("the render app exists after DefaultPlugins");
        render_app.insert_resource(MeshAllocatorSettings {
            slab_allocator_settings: SlabAllocatorSettings {
                min_slab_size: 8 * 1024 * 1024,
                growth_factor: 2.0,
                ..default()
            },
            ..default()
        });
    }
}

impl Plugin for OneShotEnvironmentMapPlugin {
    fn build(&self, app: &mut App) {
        let ready = EnvironmentMapGenerationReady::default();
        app.insert_resource(ready.clone())
            .add_systems(Update, retain_generated_environment_map);

        app.get_sub_app_mut(RenderApp)
            .expect("the render app exists after DefaultPlugins")
            .insert_resource(ready)
            .add_systems(
                Render,
                mark_environment_map_generated.after(bevy::pbr::generate::filtering_system),
            );
    }
}

pub(crate) fn mark_environment_map_generated(
    ready: Res<EnvironmentMapGenerationReady>,
    maps: Query<(), With<bevy::pbr::generate::GeneratorBindGroups>>,
    pipelines: Option<Res<bevy::pbr::generate::GeneratorPipelines>>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Some(pipelines) = pipelines else {
        return;
    };
    if maps.is_empty() {
        return;
    }
    let pipeline_ids = [
        pipelines.downsample_first,
        pipelines.downsample_second,
        pipelines.copy,
        pipelines.radiance,
        pipelines.irradiance,
    ];
    if pipeline_ids
        .into_iter()
        .all(|id| pipeline_cache.get_compute_pipeline(id).is_some())
    {
        ready.0.store(true, Ordering::Release);
    }
}

pub(crate) fn retain_generated_environment_map(
    ready: Res<EnvironmentMapGenerationReady>,
    mut commands: Commands,
    maps: Query<
        Entity,
        (
            With<GeneratedEnvironmentMapLight>,
            With<EnvironmentMapLight>,
        ),
    >,
) {
    if !ready.0.swap(false, Ordering::AcqRel) {
        return;
    }
    for entity in &maps {
        commands
            .entity(entity)
            .remove::<GeneratedEnvironmentMapLight>();
    }
}

/// Edge of each source cubemap face, in texels. Bevy filters this once into a
/// 32x32 diffuse map and a roughness-aware specular mip chain.
pub(crate) const SKY_CUBEMAP_SIZE: u32 = 64;

/// Radiance the sky cubemap is scaled to, in cd/m². A uniform hemisphere of
/// radiance `L` delivers `pi * L` lux, so this is roughly two thousand lux of
/// fill — enough to open the shadows up, far short of flattening them.
pub(crate) const SKY_ENVIRONMENT_INTENSITY: f32 = 700.0;

/// Straight up: cool and bright, the way an overcast sky reads.
pub(crate) const SKY_ZENITH: Vec3 = Vec3::new(0.62, 0.74, 1.0);

/// The band around the horizon, paler than the zenith and near neutral.
pub(crate) const SKY_HORIZON: Vec3 = Vec3::new(0.80, 0.82, 0.88);

/// Straight down: dim and warm, standing in for bounce off the platform.
pub(crate) const SKY_GROUND: Vec3 = Vec3::new(0.26, 0.22, 0.18);

/// Builds the garage's sky-and-ground source cubemap for one-time filtering.
pub(crate) fn sky_cubemap(size: u32) -> Image {
    let edge = f32::from(u16::try_from(size).expect("a cubemap face is a modest number of texels"));
    let mut texels: Vec<u8> =
        Vec::with_capacity(usize::try_from(6 * size * size * 8).expect("the sky map fits memory"));
    for face in 0..6_usize {
        for row in 0..size {
            for column in 0..size {
                let along = |index: u32| {
                    let index =
                        f32::from(u16::try_from(index).expect("a texel index is within its face"));
                    2.0f32.mul_add(index + 0.5, -edge) / edge
                };
                let (u, v) = (along(column), along(row));
                let colour = sky_colour(cubemap_direction(face, u, v));
                for channel in [colour.x, colour.y, colour.z, 1.0] {
                    texels.extend_from_slice(&half_bits(channel).to_le_bytes());
                }
            }
        }
    }
    Image {
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        }),
        ..Image::new(
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            texels,
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}

/// The direction a texel of one cubemap face looks along, in the +X, -X, +Y,
/// -Y, +Z, -Z order the graphics API expects.
pub(crate) fn cubemap_direction(face: usize, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

/// Sky above, ground below, meeting at the horizon.
pub(crate) fn sky_colour(direction: Vec3) -> Vec3 {
    let height = direction.y;
    if height >= 0.0 {
        SKY_HORIZON.lerp(SKY_ZENITH, height.sqrt())
    } else {
        SKY_HORIZON.lerp(SKY_GROUND, (-height).powf(0.7))
    }
}

/// Encodes the sky's finite, non-negative values as IEEE 754 binary16.
pub(crate) fn half_bits(value: f32) -> u16 {
    let bits = value.clamp(0.0, 65_504.0).to_bits();
    let exponent = i32::try_from((bits >> 23) & 0xff).expect("a float exponent fits in i32") - 127;
    if exponent < -14 {
        return 0;
    }
    let exponent = u16::try_from(exponent + 15).expect("a clamped exponent is in range");
    let mantissa = u16::try_from((bits & 0x007f_ffff) >> 13).expect("ten mantissa bits fit in u16");
    (exponent << 10) | mantissa
}
