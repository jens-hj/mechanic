//! Deterministic linear-HDR stars and a full moon, authored in celestial space.

use crate::render::environment::{cubemap_direction, half_bits};
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

pub(super) fn cubemap() -> Image {
    const SIZE: u32 = 1024;
    let edge = f32::from(u16::try_from(SIZE).unwrap());
    let mut texels = Vec::with_capacity((6 * SIZE * SIZE * 8) as usize);
    for face in 0..6 {
        for row in 0..SIZE {
            for column in 0..SIZE {
                let along = |value: u32| {
                    2.0 * f32::from(u16::try_from(value).unwrap()) / edge - 1.0 + 1.0 / edge
                };
                // Bevy's skybox samples a left-handed cubemap.
                let direction =
                    cubemap_direction(face, along(column), along(row)) * Vec3::new(1.0, 1.0, -1.0);
                let colour = colour(
                    direction,
                    u32::try_from(face).unwrap() * SIZE * SIZE + row * SIZE + column,
                );
                for channel in [colour.x, colour.y, colour.z, 1.0] {
                    texels.extend_from_slice(&half_bits(channel).to_le_bytes());
                }
            }
        }
    }
    Image {
        sampler: bevy::image::ImageSampler::linear(),
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        }),
        ..Image::new(
            Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            texels,
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}

fn colour(direction: Vec3, index: u32) -> Vec3 {
    // Slightly enlarged moon (1.4 degrees) resolves its maria at modest map size.
    let radius = 0.012_f32;
    if direction.dot(Vec3::NEG_X) > radius.cos() {
        let uv = Vec2::new(direction.y, direction.z) / radius;
        let maria = (uv.x * 9.0 + (uv.y * 7.0).sin()).sin() * (uv.y * 11.0).cos();
        let limb = (1.0 - uv.length_squared()).max(0.0).sqrt();
        return Vec3::new(1.0, 0.96, 0.88) * (3.2 * (0.75 + 0.12 * maria) * (0.65 + 0.35 * limb));
    }
    let mut hash = index.wrapping_add(0x9e37_79b9);
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x21f0_aaad);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x735a_2d97);
    hash ^= hash >> 15;
    if hash.is_multiple_of(7200) {
        Vec3::new(0.8, 0.88, 1.0) * 0.25 * (1.0 + f32::from((hash >> 24) as u8) / 64.0)
    } else {
        Vec3::splat(0.002)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_moon_is_centred_opposite_the_sun_with_finite_star_radiance() {
        let moon = colour(Vec3::NEG_X, 0);
        assert!(moon.min_element() > 0.5);
        assert!(colour(Vec3::X, 0).max_element() < 5.0);
        for index in 0..10000 {
            let colour = colour(Vec3::Y, index);
            assert!(colour.is_finite());
            assert!(colour.min_element() >= 0.0);
        }
    }
}
