//! Reusable hold-Tab radial selector for materials, items, and terrain.

#![allow(clippy::wildcard_imports)]

use bevy_mosaic::ui::*;
use mechanic_core::ConstructionMaterial;
use mosaic_core::theme::color;
use mosaic_macros::{component, view};

use crate::hotbar::WheelChoice;

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
    pub(crate) chroma_config: bool,
    pub(crate) highlighted: Option<WheelChoice>,
}

#[component]
pub(crate) fn RadialSelector(model: State<Model>) -> Element {
    let name = move || model.get().highlighted.map_or("", WheelChoice::label);
    let shadow_name = move || model.get().highlighted.map_or("", WheelChoice::label);
    let context_name = move || {
        model
            .get()
            .highlighted
            .map_or("", |choice| choice.context().label())
    };
    view! {
        stack align:center justify:center nohit {
            stack width:{ Length::px(WHEEL_SIZE) } height:{ Length::px(WHEEL_SIZE) }
                align:center justify:center nohit {
                for (choice, index) in { ordered_sectors(model.get().highlighted) } {
                    (sector_arc(model, *choice, *index))
                }
                for (choice, index) in { ordered_sectors(model.get().highlighted) } {
                    (choice_thumbnail(*choice, *index, sector_count(*choice)))
                }
                circle radius:36px exponent:1 fill:shell stroke:(width:3px color:shell-edge)
                text font-family:typeface.display font-size:10px font-weight:700
                    letter-spacing:0.6px font-color:accent.key { context_name() }
                stack width:{ Length::px(LABEL_WIDTH) } height:{ Length::px(LABEL_HEIGHT) }
                    align:center justify:center translate:(x:0px y:{ Length::px(LABEL_OFFSET_Y) })
                    nohit {
                    stack width:{ Length::px(LABEL_WIDTH) } height:{ Length::px(LABEL_HEIGHT) }
                        translate:(x:0px y:6px) radius:{ Length::px(LABEL_CORNER_RADIUS) } exponent:1
                        fill:#090C0F {}
                    stack width:{ Length::px(LABEL_WIDTH) } height:{ Length::px(LABEL_HEIGHT) }
                        align:center justify:center radius:{ Length::px(LABEL_CORNER_RADIUS) } exponent:1 clip {
                        for (texture, ()) in {
                            model.get().highlighted.into_iter().filter_map(|choice| {
                                choice_base_color_bytes(choice).map(|bytes| (bytes, ()))
                            })
                        } {
                            img fit:cover width:{ Length::px(LABEL_WIDTH) }
                                height:{ Length::px(LABEL_HEIGHT) }
                                (ImageSource::encoded(*texture))
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
                    model.get().highlighted.into_iter().filter_map(|choice| match choice {
                        WheelChoice::ConstructionMaterial(material) => Some((material, ())),
                        _ => None,
                    })
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
            radius:5px exponent:1 fill:#090C0FDD {
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
            radius:2px exponent:1 fill:{ color(row_tint) } {
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
        view! { el width:18px height:8px radius:2px exponent:1 fill:accent.key {} }
    } else {
        view! { el width:18px height:8px radius:2px exponent:1 fill:shell-edge {} }
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

fn ordered_sectors(highlighted: Option<WheelChoice>) -> Vec<(WheelChoice, usize)> {
    let mut sectors = choices(highlighted)
        .into_iter()
        .enumerate()
        .map(|(index, choice)| (choice, index))
        .collect::<Vec<_>>();
    sectors.sort_by_key(|(choice, _)| Some(*choice) == highlighted);
    sectors
}

fn choices(highlighted: Option<WheelChoice>) -> Vec<WheelChoice> {
    highlighted.map_or_else(Vec::new, |choice| choice.context().choices().collect())
}

const fn sector_count(choice: WheelChoice) -> usize {
    choice.context().count()
}

fn sector_arc(model: State<Model>, choice: WheelChoice, index: usize) -> Element {
    let selected = move || model.get().highlighted == Some(choice);
    let stroke = move || {
        if selected() {
            color(accent.key)
        } else {
            color(bar.fill)
        }
    };
    let count = small_f32(sector_count(choice));
    let span = 360.0 / count;
    let gap = 2.0_f32.min(span * 0.1);
    let centre = -90.0 + small_f32(index) * span;
    view! {
        circle radius:{ Length::px(SECTOR_RADIUS) } exponent:1
            arc:(from:{ centre - span * 0.5 + gap } to:{ centre + span * 0.5 - gap })
            stroke:(width:{ SECTOR_WIDTH } color:{ stroke() })
    }
}

fn choice_thumbnail(choice: WheelChoice, index: usize, count: usize) -> Element {
    let angle = std::f32::consts::TAU * small_f32(index) / small_f32(count);
    let position = (
        angle.sin() * MATERIAL_RADIUS,
        -angle.cos() * MATERIAL_RADIUS,
    );
    match choice {
        WheelChoice::ConstructionMaterial(material) => {
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
        WheelChoice::Item(item) => view! {
            stack width:{ Length::px(BLOCK_THUMBNAIL_WIDTH) }
                height:{ Length::px(BLOCK_THUMBNAIL_HEIGHT) } align:center justify:center nohit
                translate:(x:{ Length::px(position.0) } y:{ Length::px(position.1) }) {
                (super::hotbar::icon(item.editor_tool()))
            }
        },
        WheelChoice::TerrainMaterial(material) => {
            let source = ImageSource::encoded(terrain_base_color_bytes(material));
            view! {
                stack width:48px height:48px radius:24px exponent:1 clip
                    stroke:(width:3px color:shell-edge) align:center justify:center
                    translate:(x:{ Length::px(position.0) } y:{ Length::px(position.1) }) {
                    img fit:cover width:52px height:52px (source)
                }
            }
        }
        WheelChoice::ShapeMode(edit_mode) => shape_mode_thumbnail(edit_mode, position),
    }
}

fn shape_mode_thumbnail(
    edit_mode: crate::shape_tool::ShapeEditMode,
    position: (f32, f32),
) -> Element {
    match edit_mode {
        crate::shape_tool::ShapeEditMode::Vertex => view! {
            canvas width:54px height:54px
                translate:(x:{ Length::px(position.0) } y:{ Length::px(position.1) }) {
                line from:(x:10px y:40px) to:(x:42px y:12px) stroke:(width:3px color:ink.fg)
                circle at:(x:10px y:40px) radius:5px exponent:1 fill:accent.key
                circle at:(x:42px y:12px) radius:5px exponent:1 fill:accent.key
            }
        },
        crate::shape_tool::ShapeEditMode::Chamfer => view! {
            canvas width:54px height:54px
                translate:(x:{ Length::px(position.0) } y:{ Length::px(position.1) }) {
                line from:(x:8px y:42px) to:(x:28px y:42px) stroke:(width:4px color:ink.fg)
                line from:(x:28px y:42px) to:(x:44px y:26px) stroke:(width:4px color:accent.key)
                line from:(x:44px y:26px) to:(x:44px y:8px) stroke:(width:4px color:ink.fg)
            }
        },
        crate::shape_tool::ShapeEditMode::Fillet => view! {
            canvas width:54px height:54px
                translate:(x:{ Length::px(position.0) } y:{ Length::px(position.1) }) {
                line from:(x:8px y:42px) to:(x:27px y:42px) stroke:(width:4px color:ink.fg)
                line from:(x:27px y:42px) to:(x:35px y:40px)
                    stroke:(width:4px cap:round color:accent.key)
                line from:(x:35px y:40px) to:(x:41px y:34px)
                    stroke:(width:4px cap:round color:accent.key)
                line from:(x:41px y:34px) to:(x:43px y:26px)
                    stroke:(width:4px cap:round color:accent.key)
                line from:(x:43px y:26px) to:(x:43px y:8px) stroke:(width:4px color:ink.fg)
            }
        },
    }
}

fn small_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).expect("selector sector counts fit u16"))
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
        ConstructionMaterial::Copper => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/copper/copper_block_thumbnail.png"
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
        ConstructionMaterial::Copper => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/copper/copper_base_color.png"
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

const fn terrain_base_color_bytes(material: mechanic_world::TerrainMaterial) -> &'static [u8] {
    match material {
        mechanic_world::TerrainMaterial::SurfaceCover => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/terrain/grass/grass_base_color.png"
        )),
        mechanic_world::TerrainMaterial::Soil => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/terrain/dirt/dirt_base_color.png"
        )),
        mechanic_world::TerrainMaterial::Rock => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/terrain/stone/stone_base_color.png"
        )),
        mechanic_world::TerrainMaterial::Sand => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/sand/sand_base_color.png"
        )),
        mechanic_world::TerrainMaterial::Iron => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/iron/iron_base_color.png"
        )),
        mechanic_world::TerrainMaterial::Graphite => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/materials/graphite/graphite_base_color.png"
        )),
    }
}

const fn choice_base_color_bytes(choice: WheelChoice) -> Option<&'static [u8]> {
    match choice {
        WheelChoice::ConstructionMaterial(material) => Some(material_base_color_bytes(material)),
        WheelChoice::TerrainMaterial(material) => Some(terrain_base_color_bytes(material)),
        WheelChoice::Item(_) | WheelChoice::ShapeMode(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Model, block_thumbnail_bytes, material_base_color_bytes, material_ratings, ordered_sectors,
        terrain_base_color_bytes,
    };
    use crate::hotbar::{PlaceableItem, WheelChoice};
    use mechanic_core::ConstructionMaterial;

    #[test]
    fn model_preserves_open_and_highlight_state() {
        let model = Model {
            open: true,
            chroma_config: false,
            highlighted: Some(WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Wood,
            )),
        };
        assert!(model.open);
        assert!(!model.chroma_config);
        assert_eq!(
            model.highlighted,
            Some(WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Wood
            ))
        );
    }

    #[test]
    fn highlighted_sector_is_painted_last() {
        let ordered = ordered_sectors(Some(WheelChoice::ConstructionMaterial(
            ConstructionMaterial::Concrete,
        )));
        assert_eq!(
            ordered.last().map(|(material, _)| *material),
            Some(WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Concrete
            ))
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
    fn contextual_wheels_cover_every_choice() {
        assert_eq!(
            ordered_sectors(Some(WheelChoice::ConstructionMaterial(
                ConstructionMaterial::Steel
            )))
            .len(),
            ConstructionMaterial::ALL.len()
        );
        assert_eq!(
            ordered_sectors(Some(WheelChoice::Item(PlaceableItem::Bearing))).len(),
            PlaceableItem::ALL.len()
        );
        assert_eq!(
            ordered_sectors(Some(WheelChoice::TerrainMaterial(
                mechanic_world::TerrainMaterial::Soil
            )))
            .len(),
            mechanic_world::TerrainMaterial::ALL.len()
        );
        assert_eq!(
            ordered_sectors(Some(WheelChoice::ShapeMode(
                crate::shape_tool::ShapeEditMode::Vertex
            )))
            .len(),
            3
        );
        assert_eq!(
            WheelChoice::ShapeMode(crate::shape_tool::ShapeEditMode::Fillet)
                .context()
                .label(),
            "SHAPE"
        );
        assert_eq!(
            WheelChoice::Item(PlaceableItem::Bearing).context().label(),
            "ITEMS"
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
    fn every_terrain_material_has_a_textured_blob_and_nameplate_source() {
        for material in mechanic_world::TerrainMaterial::ALL {
            let png = terrain_base_color_bytes(material);
            assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "{material:?}");
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
