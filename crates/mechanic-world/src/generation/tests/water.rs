//! Implicit water: the sea and lakes fill open ground up to their level, and
//! no water stands against open air.

use bevy_math::DVec3;

use super::heart_of;
use crate::generation::rivers::{DRAINAGE_CELL_METRES, drainage_side};
use crate::generation::water::SEALED_ROOF_METRES;
use crate::generation::{TerrainField, WaterBody};
use crate::{WORLD_HALF_EXTENT_METERS, WorldSeed};

/// Drainage-grid positions where `keep` holds for the column's water, one
/// in every `stride` found.
fn water_columns(
    field: &TerrainField,
    stride: usize,
    keep: impl Fn(WaterBody) -> bool,
) -> Vec<(f64, f64)> {
    let side = drainage_side();
    let mut found = Vec::new();
    let mut seen = 0;
    for k in (0..side).step_by(3) {
        for i in (0..side).step_by(3) {
            #[expect(clippy::cast_precision_loss, reason = "small grid")]
            let (x, z) = (
                i as f64 * DRAINAGE_CELL_METRES - WORLD_HALF_EXTENT_METERS,
                k as f64 * DRAINAGE_CELL_METRES - WORLD_HALF_EXTENT_METERS,
            );
            if field
                .water_surface(x, z)
                .is_some_and(|surface| keep(surface.body))
            {
                if seen % stride == 0 {
                    found.push((x, z));
                }
                seen += 1;
            }
        }
    }
    found
}

/// Water points around a centre, and how many of them touch open ground
/// that is not water at the same height.
fn leaks_around(field: &TerrainField, (x0, z0): (f64, f64), reach: f64) -> (usize, usize) {
    let (mut water, mut leaks) = (0, 0);
    #[expect(clippy::cast_possible_truncation, reason = "a few dozen steps")]
    let steps = (reach / 1.5) as i32;
    for i in -steps..=steps {
        for k in -steps..=steps {
            let (x, z) = (x0 + f64::from(i) * 1.5, z0 + f64::from(k) * 1.5);
            let Some(surface) = field.water_surface(x, z) else {
                continue;
            };
            for below in [0.1, 1.5, 4.0] {
                let point = DVec3::new(x, surface.level - below, z);
                if !field.is_water(point) {
                    continue;
                }
                water += 1;
                let open_dry = |offset: DVec3| {
                    let neighbour = point + offset;
                    field.density(neighbour) <= 0.0 && !field.is_water(neighbour)
                };
                if [DVec3::X, DVec3::NEG_X, DVec3::Z, DVec3::NEG_Z]
                    .into_iter()
                    .any(|direction| open_dry(direction * 0.3))
                {
                    leaks += 1;
                }
            }
        }
    }
    (water, leaks)
}

#[test]
fn a_basin_below_sea_level_cut_off_from_the_sea_fills_to_its_rim() {
    let field = TerrainField::new(WorldSeed(42));
    let (x, z) = heart_of(&field, "sunken_coast");
    let surface = field.water_surface(x, z).expect("the basin holds water");
    let ground = field.topmost_surface(x, z).expect("the basin has a floor");
    assert!(matches!(surface.body, WaterBody::Lake(_)));
    assert!(ground < field.sea_level() && surface.level > field.sea_level());
    assert!(field.is_water(DVec3::new(x, ground + 0.5, z)));
}

#[test]
fn the_sea_fills_open_ground_below_sea_level() {
    let field = TerrainField::new(WorldSeed(42));
    let mut filled = 0;
    for (x0, z0) in water_columns(&field, 50, |body| body == WaterBody::Sea) {
        for i in -2..2 {
            let (x, z) = (x0 + f64::from(i) * 8.0, z0 + f64::from(i) * 5.0);
            let Some(surface) = field.water_surface(x, z) else {
                continue;
            };
            let Some(ground) = field.topmost_surface(x, z) else {
                continue;
            };
            if surface.body != WaterBody::Sea || ground > surface.level - 1.0 {
                continue;
            }
            assert!((surface.level - field.sea_level()).abs() < 1.0e-9);
            assert!(field.is_water(DVec3::new(x, (ground + surface.level) * 0.5, z)));
            assert!(!field.is_water(DVec3::new(x, surface.level + 0.1, z)));
            filled += 1;
        }
    }
    assert!(filled > 100, "only {filled} sea columns");
}

#[test]
fn lakes_and_rivers_are_held_by_ground_all_round() {
    let field = TerrainField::new(WorldSeed(42));
    assert!(field.lake_count() > 0, "the world has no lakes");
    for (name, keep) in [
        (
            "lakes",
            (|body| matches!(body, WaterBody::Lake(_))) as fn(WaterBody) -> bool,
        ),
        ("rivers", |body| matches!(body, WaterBody::River(_))),
        ("sea", |body| body == WaterBody::Sea),
    ] {
        let centres = water_columns(&field, 7, keep);
        assert!(!centres.is_empty(), "no {name}");
        let (mut water, mut leaks) = (0, 0);
        for centre in centres.into_iter().take(24) {
            let (w, l) = leaks_around(&field, centre, 40.0);
            water += w;
            leaks += l;
        }
        eprintln!("{name}: {leaks} leaks in {water} water points");
        assert!(water > 500, "{name}: only {water} water points");
        assert!(
            leaks * 1_000 < water,
            "{name}: {leaks} of {water} water points touch dry open ground"
        );
    }
}

#[test]
fn voids_under_lakes_and_rivers_keep_a_sealing_roof_and_stay_dry() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let mut voids = 0;
    for (x0, z0) in water_columns(&field, 12, |body| body != WaterBody::Sea) {
        for i in -3..3 {
            for k in -3..3 {
                let (x, z) = (x0 + f64::from(i) * 7.0, z0 + f64::from(k) * 7.0);
                let column = world.column(x, z);
                let Some(surface) = column.water.surface else {
                    continue;
                };
                for depth in 0..30 {
                    let point = DVec3::new(x, surface.level - f64::from(depth) * 2.0, z);
                    let (density, carved, rock) = world.density_parts(&column, point);
                    if density > 0.0 || carved == 0 || rock <= 0.0 {
                        continue;
                    }
                    voids += 1;
                    assert!(
                        rock > SEALED_ROOF_METRES,
                        "a void {rock:.1} m into rock under water at {point}"
                    );
                    assert!(
                        !field.is_water(point),
                        "a buried void holds water at {point}"
                    );
                }
            }
        }
    }
    assert!(voids > 50, "only {voids} buried void samples");
}
