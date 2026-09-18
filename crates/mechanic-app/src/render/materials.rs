//! Construction, bearing, authored-part, and preview materials with their textures.

use crate::chroma::{ChromaMaterialExtension, ConstructionRenderMaterial};
use crate::{chroma, world};
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageLoaderSettings};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::{
    AlphaMode, AssetServer, Assets, Color, Handle, Image, LinearRgba, ResMut, Resource,
    StandardMaterial, Vec4, default, format, warn,
};
use mechanic_core::ConstructionMaterial;

pub(crate) const BEARING_RENDER_DEPTH_BIAS: f32 = 2.0;

pub(crate) const BEARING_RENDER_RADIAL_SKIN: f32 = 0.001;

pub(crate) const PREVIEW_RENDER_DEPTH_BIAS: f32 = 1.0;

pub(crate) fn bearing_surface_material(asset_server: &AssetServer) -> StandardMaterial {
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                configure_bearing_texture(settings, is_srgb);
            })
            .load(format!("machines/bearing/bearing_{suffix}.png"))
    };
    bearing_pbr_material(
        texture("base_color", true),
        texture("normal", false),
        texture("orm", false),
    )
}

pub(crate) fn bearing_pbr_material(
    base_color: Handle<Image>,
    normal: Handle<Image>,
    orm: Handle<Image>,
) -> StandardMaterial {
    StandardMaterial {
        base_color_texture: Some(base_color),
        metallic: 1.0,
        perceptual_roughness: 1.0,
        metallic_roughness_texture: Some(orm.clone()),
        occlusion_texture: Some(orm),
        normal_map_texture: Some(normal),
        depth_bias: BEARING_RENDER_DEPTH_BIAS,
        ..default()
    }
}

pub(crate) fn configure_bearing_texture(settings: &mut ImageLoaderSettings, is_srgb: bool) {
    settings.is_srgb = is_srgb;
    let sampler = settings.sampler.get_or_init_descriptor();
    sampler.address_mode_u = ImageAddressMode::Repeat;
    sampler.address_mode_v = ImageAddressMode::ClampToEdge;
    sampler.mag_filter = ImageFilterMode::Linear;
    sampler.min_filter = ImageFilterMode::Linear;
    sampler.mipmap_filter = ImageFilterMode::Linear;
    sampler.anisotropy_clamp = 8;
}

#[derive(Resource)]
pub(crate) struct BearingTextureMipsPending(pub(crate) Vec<Handle<Image>>);

pub(crate) fn prepare_bearing_texture_mips(
    mut images: ResMut<Assets<Image>>,
    mut pending: ResMut<BearingTextureMipsPending>,
) {
    let Some(index) = pending
        .0
        .iter()
        .position(|handle| images.contains(handle.id()))
    else {
        return;
    };
    let handle = pending.0.swap_remove(index);
    let Some(mut image) = images.get_mut(&handle) else {
        return;
    };
    if let Err(error) = world::generate_rgba8_mip_chain(&mut image) {
        warn!("failed to generate bearing texture mipmaps: {error}");
        pending.0.clear();
    }
}

pub(crate) fn preview_material(base_color: Color) -> StandardMaterial {
    StandardMaterial {
        base_color,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    }
}

pub(crate) fn authored_part_material(asset_server: &AssetServer, stem: &str) -> StandardMaterial {
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                configure_authored_texture(settings, is_srgb);
            })
            .load(format!("{stem}_{suffix}.png"))
    };
    let orm = texture("orm", false);
    StandardMaterial {
        base_color_texture: Some(texture("base_color", true)),
        metallic: 1.0,
        perceptual_roughness: 1.0,
        metallic_roughness_texture: Some(orm.clone()),
        occlusion_texture: Some(orm),
        normal_map_texture: Some(texture("normal", false)),
        emissive: LinearRgba::WHITE,
        emissive_texture: Some(texture("emissive", true)),
        ..default()
    }
}

pub(crate) fn configure_authored_texture(settings: &mut ImageLoaderSettings, is_srgb: bool) {
    settings.is_srgb = is_srgb;
    let sampler = settings.sampler.get_or_init_descriptor();
    sampler.address_mode_u = ImageAddressMode::ClampToEdge;
    sampler.address_mode_v = ImageAddressMode::ClampToEdge;
    sampler.mag_filter = ImageFilterMode::Linear;
    sampler.min_filter = ImageFilterMode::Linear;
    sampler.mipmap_filter = ImageFilterMode::Linear;
}

pub(crate) const fn material_index(material: ConstructionMaterial) -> usize {
    match material {
        ConstructionMaterial::Aluminium => 0,
        ConstructionMaterial::CarbonFiber => 1,
        ConstructionMaterial::Concrete => 2,
        ConstructionMaterial::Copper => 3,
        ConstructionMaterial::Dirt => 4,
        ConstructionMaterial::Graphite => 5,
        ConstructionMaterial::Iron => 6,
        ConstructionMaterial::Plastic => 7,
        ConstructionMaterial::Rubber => 8,
        ConstructionMaterial::Sand => 9,
        ConstructionMaterial::Steel => 10,
        ConstructionMaterial::Stone => 11,
        ConstructionMaterial::Wood => 12,
    }
}

pub(crate) const fn construction_tint_mask_path(
    material: ConstructionMaterial,
) -> Option<&'static str> {
    match material {
        ConstructionMaterial::Copper => Some("materials/copper/copper_tint.png"),
        ConstructionMaterial::Dirt => Some("materials/dirt/dirt_tint.png"),
        ConstructionMaterial::Aluminium
        | ConstructionMaterial::CarbonFiber
        | ConstructionMaterial::Concrete
        | ConstructionMaterial::Graphite
        | ConstructionMaterial::Iron
        | ConstructionMaterial::Plastic
        | ConstructionMaterial::Rubber
        | ConstructionMaterial::Sand
        | ConstructionMaterial::Steel
        | ConstructionMaterial::Stone
        | ConstructionMaterial::Wood => None,
    }
}

pub(crate) fn construction_material(
    asset_server: &AssetServer,
    material: ConstructionMaterial,
    tint_mask: Handle<Image>,
) -> ConstructionRenderMaterial {
    let stem = match material {
        ConstructionMaterial::Aluminium => "materials/aluminium/aluminium",
        ConstructionMaterial::Graphite => "materials/graphite/graphite",
        ConstructionMaterial::CarbonFiber => "materials/carbon_fiber/carbon_fiber",
        ConstructionMaterial::Concrete => "materials/concrete/concrete",
        ConstructionMaterial::Copper => "materials/copper/copper",
        ConstructionMaterial::Dirt => "materials/dirt/dirt",
        ConstructionMaterial::Iron => "materials/iron/iron",
        ConstructionMaterial::Plastic => "materials/plastic/plastic",
        ConstructionMaterial::Rubber => "materials/rubber/rubber",
        ConstructionMaterial::Sand => "materials/sand/sand",
        ConstructionMaterial::Steel => "materials/steel/steel",
        ConstructionMaterial::Stone => "materials/stone/stone",
        ConstructionMaterial::Wood => "materials/wood/wood",
    };
    let texture = |suffix: &str, is_srgb: bool| {
        asset_server
            .load_builder()
            .with_settings(move |settings: &mut ImageLoaderSettings| {
                configure_repeating_texture(settings, is_srgb);
            })
            .load(format!("{stem}_{suffix}.png"))
    };
    let orm = texture("orm", false);
    ExtendedMaterial {
        base: StandardMaterial {
            base_color_texture: Some(texture("base_color", true)),
            metallic: 1.0,
            perceptual_roughness: 1.0,
            metallic_roughness_texture: Some(orm.clone()),
            occlusion_texture: Some(orm),
            normal_map_texture: Some(texture("normal", false)),
            ..default()
        },
        extension: ChromaMaterialExtension {
            tint_mask,
            base_lightness: Vec4::new(
                chroma::material_profile(material).mean_oklab_lightness,
                0.0,
                0.0,
                0.0,
            ),
        },
    }
}

pub(crate) fn configure_repeating_texture(settings: &mut ImageLoaderSettings, is_srgb: bool) {
    settings.is_srgb = is_srgb;
    let sampler = settings.sampler.get_or_init_descriptor();
    sampler.address_mode_u = ImageAddressMode::Repeat;
    sampler.address_mode_v = ImageAddressMode::Repeat;
    sampler.mag_filter = ImageFilterMode::Linear;
    sampler.min_filter = ImageFilterMode::Linear;
    sampler.mipmap_filter = ImageFilterMode::Linear;
}

pub(crate) fn authored_preview_material(
    mut material: StandardMaterial,
    base_color: Color,
) -> StandardMaterial {
    material.base_color = base_color;
    material.alpha_mode = AlphaMode::Blend;
    material.cull_mode = None;
    material.depth_bias = PREVIEW_RENDER_DEPTH_BIAS;
    material
}
