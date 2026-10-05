//! Work shared between columns, or between boxes over the same columns,
//! gives exactly what computing each alone gives.

use bevy_math::{DVec3, IVec3};

use super::heart_of;
use crate::generation::{GroundColumns, Lattice, TerrainField};
use crate::{WORLD_HALF_EXTENT_METERS, WorldSeed};

#[test]
fn many_columns_match_one_at_a_time_bit_for_bit() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let mut points = Vec::new();
    // Every biome's heart and the borders around it, with its rivers and
    // carve layers.
    for biome in ["titan_crags", "karst_needles", "sunken_coast", "shelf_mire"] {
        let (x, z) = heart_of(&field, biome);
        for step in 0..120_u32 {
            let t = f64::from(step);
            points.push((
                x + (t * 0.618).sin() * 9.0 * t,
                z + (t * 0.618).cos() * 9.0 * t,
            ));
        }
    }
    // Scattered across the world and just either side of its edge.
    for step in 0..200_u32 {
        let t = f64::from(step);
        points.push((
            (t * 97.31) % 30_000.0 - 15_000.0,
            (t * 41.7) % 30_000.0 - 15_000.0,
        ));
    }
    for offset in [-0.5, 0.0, 0.5] {
        let edge = WORLD_HALF_EXTENT_METERS + offset;
        points.extend([(edge, 10.0), (-edge, -3.0), (7.0, edge), (edge, -edge)]);
    }
    let (xs, zs): (Vec<f64>, Vec<f64>) = points.into_iter().unzip();
    let many = world.columns_many(&xs, &zs);
    assert_eq!(many.len(), xs.len());
    for (index, column) in many.iter().enumerate() {
        assert_eq!(
            format!("{column:?}"),
            format!("{:?}", world.column(xs[index], zs[index])),
            "column at ({}, {})",
            xs[index],
            zs[index]
        );
    }
    let (x, z) = heart_of(&field, "titan_crags");
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a heart lies within the world"
    )]
    let lattice = Lattice {
        origin: IVec3::new((x / 0.25) as i32, 0, (z / 0.25) as i32),
        stride: 16,
        dims: [7, 2, 5],
        centred: false,
    };
    let columns = world.lattice_columns(&lattice);
    for k in 0..5 {
        for i in 0..7 {
            assert_eq!(
                format!("{:?}", columns[i + 7 * k]),
                format!(
                    "{:?}",
                    world.column(lattice.coordinate(0, i), lattice.coordinate(2, k))
                )
            );
        }
    }
}

#[test]
fn stacked_boxes_bound_the_ground_as_fresh_ones_do() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let mut columns = GroundColumns::default();
    let mut compared = 0;
    for biome in [
        "titan_crags",
        "verdant_hills",
        "sunken_coast",
        "karst_needles",
    ] {
        let (x, z) = heart_of(&field, biome);
        for (dx, dz, width) in [(0.0, 0.0, 6.0), (40.0, -25.0, 2.0), (-90.0, 60.0, 24.0)] {
            // A column of boxes from deep rock to open sky, then the same
            // extent again after another one.
            for _ in 0..2 {
                for step in -12..24 {
                    let y = f64::from(step) * width;
                    let minimum = DVec3::new(x + dx, y, z + dz);
                    let maximum = minimum + DVec3::splat(width);
                    for enclosed in [false, true] {
                        let shared = world.ground_interval_in(
                            minimum,
                            maximum,
                            enclosed,
                            Some(&mut columns),
                        );
                        let fresh = world.ground_interval(minimum, maximum, enclosed);
                        assert_eq!(
                            [shared.lo.to_bits(), shared.hi.to_bits()],
                            [fresh.lo.to_bits(), fresh.hi.to_bits()],
                            "{biome} box at {minimum}"
                        );
                        compared += 1;
                    }
                }
                let elsewhere = DVec3::new(x - 500.0, 0.0, z + 300.0);
                world.ground_interval_in(
                    elsewhere,
                    elsewhere + DVec3::ONE,
                    false,
                    Some(&mut columns),
                );
            }
        }
    }
    assert!(compared > 800);
}
