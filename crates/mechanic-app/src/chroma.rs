//! Shared Chroma payload encoding and construction material extension.

use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};
use mechanic_core::ConstructionMaterial;
use mechanic_core::{MaterialAppearance, MaterialColor, MaterialFinish};
use serde::Deserialize;
use std::{collections::HashMap, sync::OnceLock};

pub(crate) type ConstructionRenderMaterial =
    ExtendedMaterial<StandardMaterial, ChromaMaterialExtension>;

/// One session-wide Matter Manipulator appearance brush.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct ChromaBrush {
    pub(crate) appearance: MaterialAppearance,
}

impl Default for ChromaBrush {
    fn default() -> Self {
        Self {
            appearance: MaterialAppearance::BAKED,
        }
    }
}

const PAYLOAD_SCALE: f32 = 36_864.0;
const STRUCTURE_LEVELS: f32 = 4_095.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MaterialColorProfile {
    pub(crate) representative_srgb: [u8; 3],
    pub(crate) mean_oklab_lightness: f32,
}

#[derive(Deserialize)]
struct ProfilePack {
    materials: HashMap<String, ProfileDoc>,
}

#[derive(Deserialize)]
struct ProfileDoc {
    representative_srgb: String,
    mean_oklab_lightness: f32,
}

pub(crate) fn material_profile(material: ConstructionMaterial) -> MaterialColorProfile {
    static PROFILES: OnceLock<HashMap<String, ProfileDoc>> = OnceLock::new();
    let profiles = PROFILES.get_or_init(|| {
        ron::from_str::<ProfilePack>(include_str!("../assets/materials/color-profiles.ron"))
            .expect("generated material color profiles are valid")
            .materials
    });
    let key = match material {
        ConstructionMaterial::Aluminium => "aluminium",
        ConstructionMaterial::CarbonFiber => "carbon_fiber",
        ConstructionMaterial::Concrete => "concrete",
        ConstructionMaterial::Copper => "copper",
        ConstructionMaterial::Dirt => "dirt",
        ConstructionMaterial::Graphite => "graphite",
        ConstructionMaterial::Iron => "iron",
        ConstructionMaterial::Plastic => "plastic",
        ConstructionMaterial::Rubber => "rubber",
        ConstructionMaterial::Sand => "sand",
        ConstructionMaterial::Steel => "steel",
        ConstructionMaterial::Stone => "stone",
        ConstructionMaterial::Wood => "wood",
    };
    let profile = &profiles[key];
    MaterialColorProfile {
        representative_srgb: parse_hex(&profile.representative_srgb),
        mean_oklab_lightness: profile.mean_oklab_lightness,
    }
}

fn parse_hex(value: &str) -> [u8; 3] {
    let value = value
        .strip_prefix('#')
        .expect("generated representative colors start with #");
    let channel = |offset| {
        u8::from_str_radix(&value[offset..offset + 2], 16)
            .expect("generated representative colors contain RGB8 hex")
    };
    [channel(0), channel(2), channel(4)]
}

/// Material-local texture data needed by Chroma's fragment stage.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(crate) struct ChromaMaterialExtension {
    #[texture(100)]
    #[sampler(101)]
    pub(crate) tint_mask: Handle<Image>,
    #[uniform(102)]
    pub(crate) base_lightness: Vec4,
}

impl MaterialExtension for ChromaMaterialExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/chroma_material.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/chroma_material.wgsl".into()
    }
}

/// Encodes one constant appearance into the standard vertex-color channel.
/// Every component remains nonzero so the shader can undo Bevy's ordinary
/// vertex-color multiplication before interpreting the payload.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Validated ranges keep the packed value exactly inside 15 bits.
pub(crate) fn encode_appearance(appearance: MaterialAppearance) -> [f32; 4] {
    let (mode, color) = match appearance.color {
        MaterialColor::Baked => (0_u32, [1.0, 1.0, 1.0]),
        MaterialColor::Shift(shift) => (
            1,
            [
                (shift.hue_degrees() + 180.0) / 360.0 + f32::EPSILON,
                shift.chroma() / 1.8 + f32::EPSILON,
                shift.lightness() / 2.0 + f32::EPSILON,
            ],
        ),
        MaterialColor::Dye(dye) => {
            let [r, g, b] = dye.target_rgb();
            (
                2,
                [
                    (f32::from(r) + 1.0) / 256.0,
                    (f32::from(g) + 1.0) / 256.0,
                    (f32::from(b) + 1.0) / 256.0,
                ],
            )
        }
    };
    let finish = match appearance.finish {
        MaterialFinish::Baked => 0,
        MaterialFinish::Anodised => 1,
        MaterialFinish::Painted => 2,
    };
    let structure = match appearance.color {
        MaterialColor::Dye(dye) => (dye.structure() / 3.0 * STRUCTURE_LEVELS).round() as u32,
        MaterialColor::Baked | MaterialColor::Shift(_) => 0,
    };
    let style = mode * 3 + finish;
    let packed = style * 4_096 + structure + 1;
    [color[0], color[1], color[2], packed as f32 / PAYLOAD_SCALE]
}

pub(crate) fn recolor_reference(
    base_linear: [f32; 3],
    freedom: f32,
    base_lightness: f32,
    appearance: MaterialAppearance,
) -> [f32; 3] {
    let freedom = freedom.clamp(0.0, 1.0);
    let Some((lightness, mut chroma, hue)) = treatment(
        linear_to_oklab(base_linear),
        base_lightness,
        appearance.color,
        freedom,
    ) else {
        return base_linear;
    };
    let mut rgb = oklab_to_linear([lightness, chroma * hue.cos(), chroma * hue.sin()]);
    for _ in 0..6 {
        if rgb
            .iter()
            .all(|channel| (-0.0005..=1.0005).contains(channel))
        {
            break;
        }
        chroma *= 0.85;
        rgb = oklab_to_linear([lightness, chroma * hue.cos(), chroma * hue.sin()]);
    }
    for channel in &mut rgb {
        *channel = channel.clamp(0.0, 1.0);
    }
    for index in 0..3 {
        rgb[index] = base_linear[index] + (rgb[index] - base_linear[index]) * freedom;
    }
    rgb
}

fn treatment(
    lab: [f32; 3],
    base_lightness: f32,
    color: MaterialColor,
    freedom: f32,
) -> Option<(f32, f32, f32)> {
    let baked_chroma = lab[1].hypot(lab[2]);
    let baked_hue = lab[2].atan2(lab[1]);
    match color {
        MaterialColor::Baked => None,
        MaterialColor::Shift(shift) => Some((
            lab[0] * (1.0 + (shift.lightness() - 1.0) * freedom),
            baked_chroma * (1.0 + (shift.chroma() - 1.0) * freedom),
            baked_hue + shift.hue_degrees().to_radians() * freedom,
        )),
        MaterialColor::Dye(dye) => {
            let target = linear_to_oklab(dye.target_rgb().map(srgb8_to_linear));
            Some((
                target[0] + (lab[0] - base_lightness) * dye.structure(),
                target[1].hypot(target[2]),
                target[2].atan2(target[1]),
            ))
        }
    }
}

#[cfg(test)]
pub(crate) fn apply_finish(
    baked_metallic: f32,
    baked_roughness: f32,
    finish: MaterialFinish,
) -> (f32, f32) {
    match finish {
        MaterialFinish::Baked => (baked_metallic, baked_roughness),
        MaterialFinish::Anodised => (1.0, baked_roughness),
        MaterialFinish::Painted => (0.06, baked_roughness + (0.42 - baked_roughness) * 0.85),
    }
}

fn srgb8_to_linear(channel: u8) -> f32 {
    let channel = f32::from(channel) / 255.0;
    if channel <= 0.04045 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let l = (0.412_221_46 * rgb[0] + 0.536_332_55 * rgb[1] + 0.051_445_995 * rgb[2]).cbrt();
    let m = (0.211_903_5 * rgb[0] + 0.680_699_5 * rgb[1] + 0.107_396_96 * rgb[2]).cbrt();
    let s = (0.088_302_46 * rgb[0] + 0.281_718_85 * rgb[1] + 0.629_978_7 * rgb[2]).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn oklab_to_linear(lab: [f32; 3]) -> [f32; 3] {
    let l = (lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2]).powi(3);
    let m = (lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2]).powi(3);
    let s = (lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2]).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// Representative RGB8 used by placement slots and valid placement ghosts.
pub(crate) fn representative_srgb(
    material: ConstructionMaterial,
    appearance: MaterialAppearance,
) -> [u8; 3] {
    let profile = material_profile(material);
    let linear = profile.representative_srgb.map(srgb8_to_linear);
    recolor_reference(linear, 1.0, profile.mean_oklab_lightness, appearance).map(linear_to_srgb8)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
// Clamping before the cast makes the rounded result an exact RGB8 channel.
fn linear_to_srgb8(channel: f32) -> u8 {
    let encoded = if channel <= 0.003_130_8 {
        channel * 12.92
    } else {
        1.055 * channel.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{MaterialDye, MaterialShift};

    #[test]
    fn payload_is_nonzero_and_distinguishes_modes_and_finishes() {
        let baked = encode_appearance(MaterialAppearance::BAKED);
        assert!(baked.into_iter().all(|value| value > 0.0));
        let shifted = encode_appearance(MaterialAppearance::new(
            MaterialColor::Shift(MaterialShift::new(-180.0, 0.0, 0.0).unwrap()),
            MaterialFinish::Anodised,
        ));
        assert!(shifted.into_iter().all(|value| value > 0.0));
        assert!(
            baked
                .into_iter()
                .zip(shifted)
                .any(|(left, right)| (left - right).abs() > f32::EPSILON)
        );
    }

    #[test]
    fn dye_reaches_black_and_white_while_masks_hold_detail() {
        let base = [0.18, 0.11, 0.06];
        for target in [[0, 0, 0], [255, 255, 255]] {
            let appearance = MaterialAppearance::new(
                MaterialColor::Dye(MaterialDye::new(target, 0.0).unwrap()),
                MaterialFinish::Baked,
            );
            let changed = recolor_reference(base, 1.0, 0.5, appearance);
            let held = recolor_reference(base, 0.0, 0.5, appearance);
            assert!(
                held.into_iter()
                    .zip(base)
                    .all(|(left, right)| (left - right).abs() <= f32::EPSILON)
            );
            if target[0] == 0 {
                assert!(changed.iter().all(|channel| *channel < 0.001));
            } else {
                assert!(changed.iter().all(|channel| *channel > 0.999));
            }
        }
    }

    #[test]
    fn out_of_gamut_shift_walks_in_before_clamping() {
        let appearance = MaterialAppearance::new(
            MaterialColor::Shift(MaterialShift::new(150.0, 1.8, 2.0).unwrap()),
            MaterialFinish::Baked,
        );
        let rgb = recolor_reference([0.9, 0.1, 0.05], 1.0, 0.5, appearance);
        assert!(
            rgb.into_iter()
                .all(|channel| (0.0..=1.0).contains(&channel))
        );
    }

    #[test]
    fn finishes_preserve_or_replace_the_documented_channels() {
        assert_eq!(apply_finish(0.7, 0.8, MaterialFinish::Baked), (0.7, 0.8));
        assert_eq!(apply_finish(0.7, 0.8, MaterialFinish::Anodised), (1.0, 0.8));
        let painted = apply_finish(0.7, 0.8, MaterialFinish::Painted);
        assert!((painted.0 - 0.06).abs() < f32::EPSILON);
        assert!((painted.1 - 0.477).abs() < 0.000_01);
    }

    #[test]
    fn every_construction_material_has_a_valid_generated_profile() {
        for material in ConstructionMaterial::ALL {
            let profile = material_profile(material);
            assert!(
                (0.0..=1.0).contains(&profile.mean_oklab_lightness),
                "{material:?} has a valid mean lightness"
            );
            assert_eq!(
                representative_srgb(material, MaterialAppearance::BAKED),
                profile.representative_srgb,
                "Baked leaves the generated representative color unchanged"
            );
        }
    }

    #[test]
    fn shader_keeps_the_cpu_payload_and_finish_contract() {
        let source = include_str!("../assets/shaders/chroma_material.wgsl");
        assert!(!source.contains("let target ="));
        assert!(source.contains("payload.a * 36864.0"));
        assert!(source.contains("i < 6"));
        assert!(source.contains("chroma *= 0.85"));
        assert!(source.contains("pbr_input.material.metallic = 1.0"));
        assert!(source.contains("pbr_input.material.metallic = 0.06"));
        assert!(source.contains("textureSample(tint_mask"));
    }
}
