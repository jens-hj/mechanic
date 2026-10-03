#![expect(clippy::cast_precision_loss, reason = "pixel counts of a few thousand")]

use super::super::{
    BarkTraits, LeafTraits, SpeciesSpec, TREE_TEXTURE_LUMA, TreeSurface, TreeTexture,
    TreeTextureMaps,
};
use super::preset;

const EDGE: u32 = 128;

fn maps(name: &str, surface: TreeSurface) -> TreeTextureMaps {
    TreeTexture {
        species: preset(name),
        surface,
    }
    .maps(EDGE)
}

fn linear(byte: u8) -> f64 {
    let value = f64::from(byte) / 255.0;
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Mean difference between neighbouring columns `a` and `b` of the base colour.
fn column_step(maps: &TreeTextureMaps, a: usize, b: usize) -> f64 {
    let edge = maps.edge as usize;
    let total: f64 = (0..edge)
        .map(|row| {
            let at = |column: usize| linear(maps.base_color[(row * edge + column) * 4]);
            (at(a) - at(b)).abs()
        })
        .sum();
    total / edge as f64
}

/// How far the normals tilt from flat, on average.
fn relief(maps: &TreeTextureMaps) -> f64 {
    let tilts: Vec<f64> = maps
        .normal
        .chunks_exact(4)
        .map(|pixel| 1.0 - (f64::from(pixel[2]) / 127.5 - 1.0))
        .collect();
    tilts.iter().sum::<f64>() / tilts.len() as f64
}

#[test]
fn every_texture_tiles_and_keeps_the_shared_brightness() {
    for species in SpeciesSpec::embedded() {
        for surface in [TreeSurface::Bark, TreeSurface::Foliage] {
            let maps = TreeTexture {
                species: species.clone(),
                surface,
            }
            .maps(EDGE);
            let pixels = (EDGE * EDGE) as usize;
            assert_eq!(maps.base_color.len(), pixels * 4);
            assert_eq!(maps.normal.len(), pixels * 4);
            assert_eq!(maps.orm.len(), pixels * 4);
            let mean = maps
                .base_color
                .chunks_exact(4)
                .map(|pixel| linear(pixel[0]))
                .sum::<f64>()
                / pixels as f64;
            assert!(
                (mean - f64::from(TREE_TEXTURE_LUMA)).abs() < 0.03,
                "{} {surface:?} mean brightness {mean:.3}",
                species.name
            );
            // Across the seam, neighbours differ no more than inside.
            let last = EDGE as usize - 1;
            let inside = (1..last)
                .map(|column| column_step(&maps, column, column + 1))
                .sum::<f64>()
                / (last - 1) as f64;
            let seam = column_step(&maps, last, 0);
            assert!(
                seam < inside * 2.0 + 0.01,
                "{} {surface:?} seam {seam:.3} inside {inside:.3}",
                species.name
            );
        }
    }
}

#[test]
fn the_same_species_draws_the_same_texture() {
    assert_eq!(
        maps("oak", TreeSurface::Bark),
        maps("oak", TreeSurface::Bark)
    );
}

#[test]
fn old_bark_is_furrowed_and_young_bark_smooth() {
    let oak = relief(&maps("oak", TreeSurface::Bark));
    let birch = relief(&maps("birch", TreeSurface::Bark));
    assert!(oak > 3.0 * birch, "oak relief {oak:.4} birch {birch:.4}");
    let (oak, birch, spruce, bamboo) = (
        BarkTraits::of(&preset("oak")),
        BarkTraits::of(&preset("birch")),
        BarkTraits::of(&preset("spruce")),
        BarkTraits::of(&preset("bamboo")),
    );
    assert!(oak.fissure_depth > 5.0 * birch.fissure_depth);
    assert!(spruce.plates > 0.5 && spruce.plate_aspect < birch.plate_aspect);
    assert!(birch.lenticels > oak.lenticels + 0.3);
    assert!(bamboo.rings > 0.5 && oak.rings < 0.1 && birch.rings < 0.3);
}

#[test]
fn leaves_take_their_shape_and_hang_from_the_genome() {
    let traits = |name| LeafTraits::of(&preset(name));
    let (oak, spruce, willow, poplar) = (
        traits("oak"),
        traits("spruce"),
        traits("willow"),
        traits("poplar"),
    );
    assert!(spruce.aspect > 5.0 && willow.aspect > 5.0 && oak.aspect < 2.0);
    assert!(oak.length > spruce.length && oak.lobes > 0.3 && spruce.lobes < 0.1);
    assert!(willow.lift < -0.9 && poplar.lift > 0.8 && willow.spread < oak.spread);
    // Hanging leaves stack in vertical strokes: the colour changes far more
    // from column to column than from row to row.
    let hanging = maps("willow", TreeSurface::Foliage);
    let edge = hanging.edge as usize;
    let at = |row: usize, column: usize| linear(hanging.base_color[(row * edge + column) * 4]);
    let (mut along_rows, mut along_columns) = (0.0, 0.0);
    for row in 0..edge - 1 {
        for column in 0..edge - 1 {
            along_rows += (at(row, column) - at(row, column + 1)).abs();
            along_columns += (at(row, column) - at(row + 1, column)).abs();
        }
    }
    assert!(
        along_rows > 1.5 * along_columns,
        "willow changes {along_rows:.1} across and {along_columns:.1} up"
    );
}
