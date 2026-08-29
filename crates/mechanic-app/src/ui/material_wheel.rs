//! Hold-Tab radial construction-material selector.

#![allow(clippy::wildcard_imports)]

use bevy_mosaic::ui::*;
use mechanic_core::ConstructionMaterial;
use mosaic_core::theme::color;
use mosaic_macros::{component, view};

#[allow(clippy::wildcard_imports)]
use super::theme::*;

const WHEEL_SIZE: f32 = 380.0;
const SECTOR_RADIUS: f32 = 132.0;
const SECTOR_WIDTH: f32 = 72.0;
const MATERIAL_RADIUS: f32 = SECTOR_RADIUS;
const BLOCK_THUMBNAIL_WIDTH: f32 = 54.0;
const BLOCK_THUMBNAIL_HEIGHT: f32 = 60.0;
const LABEL_WIDTH: f32 = 236.0;
const LABEL_HEIGHT: f32 = 54.0;
const LABEL_OFFSET_Y: f32 = 226.0;
const LABEL_FONT_SIZE: f32 = 22.0;
const LABEL_CORNER_RADIUS: f32 = 5.0;
const RATINGS_WIDTH: f32 = LABEL_WIDTH;
const RATINGS_HEIGHT: f32 = 112.0;
const RATINGS_OFFSET_Y: f32 = 318.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Model {
    pub(crate) open: bool,
    pub(crate) highlighted: Option<ConstructionMaterial>,
}

#[component]
pub(crate) fn MaterialWheel(model: State<Model>) -> Element {
    let name = move || {
        model
            .get()
            .highlighted
            .map_or("", ConstructionMaterial::label)
    };
    let shadow_name = move || {
        model
            .get()
            .highlighted
            .map_or("", ConstructionMaterial::label)
    };
    view! {
        stack align:center justify:center nohit {
            stack width:{ Length::px(WHEEL_SIZE) } height:{ Length::px(WHEEL_SIZE) }
                align:center justify:center nohit {
                for (material, index) in { ordered_sectors(model.get().highlighted) } {
                    (sector_arc(model, *material, *index))
                }
                for (material, index) in { ordered_sectors(model.get().highlighted) } {
                    (material_block_thumbnail(*material, *index))
                }
                circle radius:36px fill:shell stroke:(width:3px color:shell-edge)
                stack width:{ Length::px(LABEL_WIDTH) } height:{ Length::px(LABEL_HEIGHT) }
                    align:center justify:center translate:(x:0px y:{ Length::px(LABEL_OFFSET_Y) })
                    nohit {
                    stack width:{ Length::px(LABEL_WIDTH) } height:{ Length::px(LABEL_HEIGHT) }
                        translate:(x:0px y:6px) radius:{ Length::px(LABEL_CORNER_RADIUS) }
                        fill:#090C0F {}
                    stack width:{ Length::px(LABEL_WIDTH) } height:{ Length::px(LABEL_HEIGHT) }
                        align:center justify:center radius:{ Length::px(LABEL_CORNER_RADIUS) } clip {
                        for (material, ()) in {
                            model.get().highlighted.into_iter().map(|material| (material, ()))
                        } {
                            img fit:cover width:{ Length::px(LABEL_WIDTH) }
                                height:{ Length::px(LABEL_HEIGHT) }
                                (ImageSource::encoded(material_base_color_bytes(*material)))
                        }
                        text font-family:typeface.display translate:(x:0px y:2px)
                            font-size:{ Length::px(LABEL_FONT_SIZE) } font-weight:700
                            font-color:#000000CC { shadow_name() }
                        text font-family:typeface.display
                            font-size:{ Length::px(LABEL_FONT_SIZE) } font-weight:700
                            font-color:#F5F8FA { name() }
                    }
                }
                for (material, ()) in {
                    model.get().highlighted.into_iter().map(|material| (material, ()))
                } {
                    stack width:{ Length::px(RATINGS_WIDTH) }
                        height:{ Length::px(RATINGS_HEIGHT) }
                        translate:(x:0px y:{ Length::px(RATINGS_OFFSET_Y) }) nohit {
                        (material_ratings_panel(*material))
                    }
                }
            }
        }
    }
}

fn material_ratings_panel(material: ConstructionMaterial) -> Element {
    view! {
        col width:fill height:fill gap:5px pad:(left:5px right:5px top:5px bottom:5px)
            radius:5px fill:#090C0FDD {
            for (index, rating) in { material_ratings(material).into_iter().enumerate() } {
                (rating_row(rating.0, rating.1, *index))
            }
        }
    }
}

fn rating_row(label: &'static str, rating: u8, index: usize) -> Element {
    let row_tint = if index.is_multiple_of(2) {
        bar.fill
    } else {
        shell
    };
    view! {
        row width:fill height:16px align:center gap:4px pad:(left:4px right:4px)
            radius:2px fill:{ color(row_tint) } {
            text width:104px font-family:typeface.display font-size:10px font-weight:700
                letter-spacing:0.5px font-color:#F5F8FACC (label)
            (rating_segment(rating >= 1))
            (rating_segment(rating >= 2))
            (rating_segment(rating >= 3))
            (rating_segment(rating >= 4))
            (rating_segment(rating >= 5))
        }
    }
}

fn rating_segment(filled: bool) -> Element {
    if filled {
        view! { el width:18px height:8px radius:2px fill:accent.key {} }
    } else {
        view! { el width:18px height:8px radius:2px fill:shell-edge {} }
    }
}

fn material_ratings(material: ConstructionMaterial) -> [(&'static str, u8); 5] {
    let properties = material.properties();
    let youngs_modulus_gpa = properties.youngs_modulus_pa / 1.0e9;
    let softness = ((200.0_f32.ln() - youngs_modulus_gpa.ln()) / (200.0_f32.ln() - 0.01_f32.ln()))
        .clamp(0.0, 1.0);
    [
        ("WEIGHT", rating(properties.density_kg_m3 / 8_000.0)),
        ("GRIP", rating(properties.static_friction / 1.0)),
        ("BOUNCE", rating(properties.restitution / 0.70)),
        (
            "ROLL RESISTANCE",
            rating(properties.rolling_resistance / 0.04),
        ),
        ("SOFTNESS", rating(softness)),
    ]
}

fn rating(normalized: f32) -> u8 {
    let scaled = 4.0 * normalized.clamp(0.0, 1.0);
    let extra: u8 = if scaled < 0.5 {
        0
    } else if scaled < 1.5 {
        1
    } else if scaled < 2.5 {
        2
    } else if scaled < 3.5 {
        3
    } else {
        4
    };
    1 + extra
}

fn ordered_sectors(
    highlighted: Option<ConstructionMaterial>,
) -> [(ConstructionMaterial, usize); ConstructionMaterial::ALL.len()] {
    let mut sectors = std::array::from_fn(|index| (ConstructionMaterial::ALL[index], index));
    sectors.sort_by_key(|(material, _)| Some(*material) == highlighted);
    sectors
}

fn sector_arc(model: State<Model>, material: ConstructionMaterial, index: usize) -> Element {
    let selected = move || model.get().highlighted == Some(material);
    let stroke = move || {
        if selected() {
            color(accent.key)
        } else {
            color(bar.fill)
        }
    };
    match index {
        0 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:-104deg to:-76deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        1 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:-74deg to:-46deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        2 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:-44deg to:-16deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        3 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:-14deg to:14deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        4 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:16deg to:44deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        5 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:46deg to:74deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        6 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:76deg to:104deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        7 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:106deg to:134deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        8 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:136deg to:164deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        9 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:166deg to:194deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        10 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:196deg to:224deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        11 => {
            view! {
                circle radius:{ Length::px(SECTOR_RADIUS) } arc:(from:226deg to:254deg)
                    stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
            }
        }
        _ => unreachable!("material wheel has twelve sectors"),
    }
}

fn material_block_thumbnail(material: ConstructionMaterial, index: usize) -> Element {
    let position = [
        (0.0, -MATERIAL_RADIUS),
        (66.0, -114.3),
        (114.3, -66.0),
        (MATERIAL_RADIUS, 0.0),
        (114.3, 66.0),
        (66.0, 114.3),
        (0.0, MATERIAL_RADIUS),
        (-66.0, 114.3),
        (-114.3, 66.0),
        (-MATERIAL_RADIUS, 0.0),
        (-114.3, -66.0),
        (-66.0, -114.3),
    ][index];
    let source = ImageSource::encoded(block_thumbnail_bytes(material));
    view! {
        stack width:{ Length::px(BLOCK_THUMBNAIL_WIDTH) }
            height:{ Length::px(BLOCK_THUMBNAIL_HEIGHT) } align:center justify:center nohit
            translate:(x:{ Length::px(position.0) } y:{ Length::px(position.1) }) {
            img fit:contain width:{ Length::px(BLOCK_THUMBNAIL_WIDTH) }
                height:{ Length::px(BLOCK_THUMBNAIL_HEIGHT) } (source)
        }
    }
}

const fn block_thumbnail_bytes(material: ConstructionMaterial) -> &'static [u8] {
    match material {
        ConstructionMaterial::Aluminium => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/aluminium/aluminium_block_thumbnail.png"
        )),
        ConstructionMaterial::Graphite => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/graphite/graphite_block_thumbnail.png"
        )),
        ConstructionMaterial::CarbonFiber => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/carbon_fiber/carbon_fiber_block_thumbnail.png"
        )),
        ConstructionMaterial::Concrete => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/concrete/concrete_block_thumbnail.png"
        )),
        ConstructionMaterial::Dirt => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/dirt/dirt_block_thumbnail.png"
        )),
        ConstructionMaterial::Iron => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/iron/iron_block_thumbnail.png"
        )),
        ConstructionMaterial::Plastic => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/plastic/plastic_block_thumbnail.png"
        )),
        ConstructionMaterial::Rubber => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/rubber/rubber_block_thumbnail.png"
        )),
        ConstructionMaterial::Sand => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/sand/sand_block_thumbnail.png"
        )),
        ConstructionMaterial::Steel => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/steel/steel_block_thumbnail.png"
        )),
        ConstructionMaterial::Stone => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/stone/stone_block_thumbnail.png"
        )),
        ConstructionMaterial::Wood => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/wood/wood_block_thumbnail.png"
        )),
    }
}

const fn material_base_color_bytes(material: ConstructionMaterial) -> &'static [u8] {
    match material {
        ConstructionMaterial::Aluminium => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/aluminium/aluminium_base_color.png"
        )),
        ConstructionMaterial::Graphite => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/graphite/graphite_base_color.png"
        )),
        ConstructionMaterial::CarbonFiber => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/carbon_fiber/carbon_fiber_base_color.png"
        )),
        ConstructionMaterial::Concrete => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/concrete/concrete_base_color.png"
        )),
        ConstructionMaterial::Dirt => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/dirt/dirt_base_color.png"
        )),
        ConstructionMaterial::Iron => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/iron/iron_base_color.png"
        )),
        ConstructionMaterial::Plastic => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/plastic/plastic_base_color.png"
        )),
        ConstructionMaterial::Rubber => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/rubber/rubber_base_color.png"
        )),
        ConstructionMaterial::Sand => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/sand/sand_base_color.png"
        )),
        ConstructionMaterial::Steel => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/steel/steel_base_color.png"
        )),
        ConstructionMaterial::Stone => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/stone/stone_base_color.png"
        )),
        ConstructionMaterial::Wood => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/wood/wood_base_color.png"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Model, block_thumbnail_bytes, material_base_color_bytes, material_ratings, ordered_sectors,
    };
    use mechanic_core::ConstructionMaterial;

    #[test]
    fn model_preserves_open_and_highlight_state() {
        let model = Model {
            open: true,
            highlighted: Some(ConstructionMaterial::Wood),
        };
        assert!(model.open);
        assert_eq!(model.highlighted, Some(ConstructionMaterial::Wood));
    }

    #[test]
    fn highlighted_sector_is_painted_last() {
        let ordered = ordered_sectors(Some(ConstructionMaterial::Concrete));
        assert_eq!(
            ordered.last().map(|(material, _)| *material),
            Some(ConstructionMaterial::Concrete)
        );
        assert_eq!(
            ordered
                .iter()
                .map(|(material, _)| *material)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            ConstructionMaterial::ALL.len(),
        );
    }

    #[test]
    fn every_material_has_a_ninety_six_by_one_hundred_six_block_thumbnail() {
        for material in ConstructionMaterial::ALL {
            let png = block_thumbnail_bytes(material);
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "{material:?}");
            assert_eq!(
                u32::from_be_bytes(png[16..20].try_into().expect("PNG width")),
                96,
                "{material:?}",
            );
            assert_eq!(
                u32::from_be_bytes(png[20..24].try_into().expect("PNG height")),
                106,
                "{material:?}",
            );
        }
    }

    #[test]
    fn every_material_has_a_high_resolution_label_texture() {
        for material in ConstructionMaterial::ALL {
            let png = material_base_color_bytes(material);
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "{material:?}");
            assert_eq!(
                u32::from_be_bytes(png[16..20].try_into().expect("PNG width")),
                3072,
                "{material:?}",
            );
            assert_eq!(
                u32::from_be_bytes(png[20..24].try_into().expect("PNG height")),
                3072,
                "{material:?}",
            );
        }
    }

    #[test]
    fn every_material_exposes_five_bounded_ratings() {
        for material in ConstructionMaterial::ALL {
            let ratings = material_ratings(material);
            assert_eq!(ratings.len(), 5);
            assert_eq!(
                ratings.map(|(label, _)| label),
                ["WEIGHT", "GRIP", "BOUNCE", "ROLL RESISTANCE", "SOFTNESS"],
            );
            assert!(
                ratings
                    .into_iter()
                    .all(|(_, rating)| (1..=5).contains(&rating))
            );
        }
    }

    #[test]
    fn ratings_follow_the_underlying_material_values() {
        let steel = material_ratings(ConstructionMaterial::Steel);
        let wood = material_ratings(ConstructionMaterial::Wood);
        let rubber = material_ratings(ConstructionMaterial::Rubber);
        let concrete = material_ratings(ConstructionMaterial::Concrete);
        let graphite = material_ratings(ConstructionMaterial::Graphite);
        assert!(steel[0].1 > wood[0].1);
        assert!(rubber[1].1 > graphite[1].1);
        assert!(rubber[2].1 > concrete[2].1);
        assert!(rubber[3].1 > steel[3].1);
        assert!(rubber[4].1 > steel[4].1);
    }
}
