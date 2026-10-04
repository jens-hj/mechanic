//! Deterministic linear-HDR background stars and the galaxy's band, fixed in
//! the star system's ecliptic frame; the skybox turns them with the planet.

use crate::render::environment::{cubemap_direction, half_bits};
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use mechanic_world::blackbody_colour;

const SIZE: u32 = 1024;

/// Band samples along each face edge. The band changes slowly, so the full
/// map interpolates between them.
const BAND_SAMPLES: usize = 65;

/// Pole of the galaxy's plane in the ecliptic frame, steeply tilted to it.
const GALACTIC_POLE: Vec3 = Vec3::new(0.24, -0.48, 0.84);

/// Night-sky glow away from the galaxy, in linear radiance.
const AIRGLOW: f32 = 0.0015;

pub(super) fn cubemap() -> Image {
    let edge = f32::from(u16::try_from(SIZE).unwrap());
    let along =
        |value: u32| 2.0 * f32::from(u16::try_from(value).unwrap()) / edge - 1.0 + 1.0 / edge;
    // Bevy's skybox samples a left-handed cubemap.
    let direction = |face, u, v| cubemap_direction(face, u, v) * Vec3::new(1.0, 1.0, -1.0);
    let mut texels = Vec::with_capacity((6 * SIZE * SIZE * 8) as usize);
    for face in 0..6 {
        let band = band_samples(|u, v| direction(face, u, v));
        for row in 0..SIZE {
            for column in 0..SIZE {
                let (u, v) = (along(column), along(row));
                let colour = colour(
                    sample(&band, u, v),
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

#[expect(clippy::cast_precision_loss, reason = "small sample grid coordinates")]
fn band_samples(direction: impl Fn(f32, f32) -> Vec3) -> Vec<f32> {
    let step = 2.0 / (BAND_SAMPLES - 1) as f32;
    (0..BAND_SAMPLES * BAND_SAMPLES)
        .map(|index| {
            let (row, column) = (index / BAND_SAMPLES, index % BAND_SAMPLES);
            band(direction(
                column as f32 * step - 1.0,
                row as f32 * step - 1.0,
            ))
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "face coordinates map onto the small sample grid"
)]
fn sample(band: &[f32], u: f32, v: f32) -> f32 {
    let scale = (BAND_SAMPLES - 1) as f32;
    let (x, y) = ((u + 1.0) * 0.5 * scale, (v + 1.0) * 0.5 * scale);
    let (column, row) = (
        (x as usize).min(BAND_SAMPLES - 2),
        (y as usize).min(BAND_SAMPLES - 2),
    );
    let (fx, fy) = (x - column as f32, y - row as f32);
    let at = |row: usize, column: usize| band[row * BAND_SAMPLES + column];
    let top = at(row, column) + (at(row, column + 1) - at(row, column)) * fx;
    let bottom = at(row + 1, column) + (at(row + 1, column + 1) - at(row + 1, column)) * fx;
    top + (bottom - top) * fy
}

/// Direction of the galaxy's core in the ecliptic frame.
pub(super) fn galactic_core() -> Vec3 {
    Vec3::X.reject_from(GALACTIC_POLE.normalize()).normalize()
}

/// Glow of the galaxy's disk in `direction`, from zero to about two: thicker
/// and brighter towards its core, mottled by star clouds and split by dust.
fn band(direction: Vec3) -> f32 {
    let pole = GALACTIC_POLE.normalize();
    let core = galactic_core();
    let latitude = direction.dot(pole).clamp(-1.0, 1.0).asin();
    let bulge = (-(direction.angle_between(core) / 0.5).powi(2)).exp();
    let thickness = 0.15 + 0.12 * bulge;
    let disk = (-(latitude / thickness).powi(2)).exp();
    let clouds = fractal(direction * 5.0);
    let lane =
        (-(latitude / 0.07).powi(2)).exp() * smoothstep(0.3, 0.6, fractal(direction * 3.0 + 3.0));
    disk * (0.4 + 0.6 * clouds) * (1.0 - 0.75 * lane) * (1.0 + 1.2 * bulge)
}

fn colour(band: f32, index: u32) -> Vec3 {
    let glow = Vec3::new(0.92, 0.95, 1.0) * (AIRGLOW + 0.05 * band);
    let mut state = u64::from(index) ^ 0x2545_f491_4f6c_dd1d;
    let mut unit = || {
        state = mix(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
        #[expect(
            clippy::cast_precision_loss,
            reason = "24 random bits map exactly onto the unit interval"
        )]
        let value = (state >> 40) as f32 / (1_u32 << 24) as f32;
        value
    };
    // More stars crowd into the galaxy's band.
    if unit() >= (1.0 + 2.5 * band) / 3000.0 {
        return glow;
    }
    // Faint stars far outnumber bright ones: N(>L) ∝ L^-1.5.
    let brightness = (0.12 * unit().max(1e-6).powf(-2.0 / 3.0)).min(10.0);
    let kelvin = 3_000.0 + 9_000.0 * f64::from(unit()).powi(2);
    // Starlight is too dim for the eye to see much colour.
    glow + blackbody_colour(kelvin).lerp(Vec3::ONE, 0.45) * brightness
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

const fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Three octaves of value noise, from zero to one.
fn fractal(point: Vec3) -> f32 {
    (0..3)
        .map(|octave| {
            let scale = f32::from(1_u8 << octave);
            value_noise(point * 2.0 * scale) / scale
        })
        .sum::<f32>()
        / 1.75
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "lattice coordinates are hashed bit for bit"
)]
fn value_noise(point: Vec3) -> f32 {
    let cell = point.floor();
    let f = point - cell;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let corner = |x: f32, y: f32, z: f32| {
        let key = (cell.x + x) as i32 as u64
            ^ ((cell.y + y) as i32 as u64).wrapping_mul(0x9e37_79b9)
            ^ ((cell.z + z) as i32 as u64).wrapping_mul(0x85eb_ca6b_c2b2_ae35);
        #[expect(
            clippy::cast_precision_loss,
            reason = "24 random bits map exactly onto the unit interval"
        )]
        let value = (mix(key) >> 40) as f32 / (1_u32 << 24) as f32;
        value
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let face = |z: f32| {
        lerp(
            lerp(corner(0.0, 0.0, z), corner(1.0, 0.0, z), u.x),
            lerp(corner(0.0, 1.0, z), corner(1.0, 1.0, z), u.x),
            u.y,
        )
    };
    lerp(face(0.0), face(1.0), u.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stars_are_finite_and_crowd_into_a_brighter_galactic_band() {
        let pole = GALACTIC_POLE.normalize();
        let in_plane = Vec3::X.reject_from(pole).normalize();
        assert!(band(in_plane) > 4.0 * band(pole));
        let count = |glow: f32| {
            (0..200_000)
                .filter(|&index| {
                    let colour = colour(glow, index);
                    assert!(colour.is_finite() && colour.min_element() >= 0.0);
                    assert!(colour.max_element() < 12.0);
                    colour.max_element() > 0.1
                })
                .count()
        };
        let (sparse, crowded) = (count(0.0), count(1.0));
        assert!(sparse > 40);
        assert!(crowded > 2 * sparse);
    }
}
