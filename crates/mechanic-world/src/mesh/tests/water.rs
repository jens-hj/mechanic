//! Water surface sheets.

use super::super::{WaterTile, water_sheet};
use crate::{TerrainField, TerrainOctree, WaterBody, WorldSeed};

/// A column of deep lake water in the seed-42 world.
fn deep_lake(field: &TerrainField) -> (f64, f64) {
    for step in 0..400 {
        let (x, z) = (
            -900.0 + f64::from(step % 20) * 20.0,
            -650.0 + f64::from(step / 20) * 20.0,
        );
        let Some(surface) = field.water_surface(x, z) else {
            continue;
        };
        let deep = field
            .topmost_surface(x, z)
            .is_some_and(|ground| ground < surface.level - 3.0);
        if matches!(surface.body, WaterBody::Lake(_)) && deep {
            return (x, z);
        }
    }
    panic!("no deep lake near the sunken coast");
}

#[test]
fn a_lake_tile_is_a_flat_sheet_at_the_lake_level() {
    let field = TerrainField::new(WorldSeed(42));
    let edits = TerrainOctree::default().snapshot();
    let (x, z) = deep_lake(&field);
    let level = field.water_surface(x, z).expect("lake").level;
    let tile = WaterTile {
        minimum: [x - 8.0, z - 8.0],
        edge: 16.0,
        cells: 16,
    };
    let sheet = water_sheet(&field, &edits, tile).expect("the tile holds water");
    assert!(sheet.indices.len() >= 3 * 2 * 16 * 16 / 2);
    for vertex in &sheet.vertices {
        assert!((f64::from(vertex[1]) - level).abs() < 1.0e-3);
    }
    assert!(sheet.depths.iter().any(|&depth| depth > 2.0));
}

#[test]
fn a_dry_tile_has_no_sheet() {
    let field = TerrainField::new(WorldSeed(42));
    let edits = TerrainOctree::default().snapshot();
    let spawn = field.safe_spawn().0;
    let tile = WaterTile {
        minimum: [spawn.x - 4.0, spawn.z - 4.0],
        edge: 8.0,
        cells: 8,
    };
    let sheet = water_sheet(&field, &edits, tile);
    assert!(sheet.is_none_or(|sheet| {
        sheet
            .vertices
            .iter()
            .all(|vertex| f64::from(vertex[1]) < spawn.y - 0.4)
    }));
}
