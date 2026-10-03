//! Established water paints the same physical bed through every sampling path.

use bevy_math::DVec3;

use super::super::surfaces::SurfaceProbe;
use super::super::{Lattice, LatticeTrees, TerrainField, TreeDetail, TreeSource, WaterBody};
use crate::{TerrainMaterial, WorldPosition, WorldSeed};

#[test]
fn biome_beds_shores_and_steep_faces_use_their_existing_palette() {
    let field = TerrainField::new(WorldSeed(42));
    for (name, bed, shore, rock) in [
        ("verdant_hills", "mire_mud", "loam", "granite"),
        ("shelf_mire", "mire_mud", "mire_mud", "fungal_stone"),
        ("sunken_coast", "seabed_silt", "beach_sand", "sea_cliff"),
        ("dune_sea", "pale_sand", "pale_sand", "buried_sandstone"),
        ("arch_steppe", "ochre_sand", "ochre_sand", "banded_light"),
        (
            "titan_crags",
            "river_gravel",
            "river_gravel",
            "granite_grey",
        ),
        ("karst_needles", "seabed_silt", "seabed_silt", "limestone"),
        ("gyroid_reef", "reef_violet", "reef_violet", "reef_violet"),
        ("drift_isles", "black_rock", "black_rock", "black_rock"),
    ] {
        let rules = &field
            .world
            .biomes
            .iter()
            .find(|b| b.name == name)
            .unwrap()
            .rules;
        let mut probe = SurfaceProbe {
            position: [0.0, 200.0, 0.0],
            depth: 0.4,
            water_depth: Some(5.0),
            up: 0.55,
            river_distance: 100.0,
            carved: 0,
        };
        for (water_depth, up, expected) in
            [(5.0, 0.55, bed), (-0.35, 1.0, shore), (5.0, 0.54, rock)]
        {
            probe.water_depth = Some(water_depth);
            probe.up = up;
            let (material, surface) = rules.paint(&probe);
            assert_eq!(surface, field.palette().id(expected).unwrap(), "{name}");
            assert_eq!(material, field.palette().look(surface).material);
        }
        // Past the shore band, rules must behave exactly as on dry ground.
        probe.depth = 0.05;
        probe.up = 1.0;
        probe.water_depth = Some(-0.36);
        let above = rules.paint(&probe);
        probe.water_depth = None;
        assert_eq!(above, rules.paint(&probe), "{name}");
        // Past the sediment skin, preserve deeper soil and mineral rules.
        for depth in [0.81, 4.0] {
            probe.depth = depth;
            probe.water_depth = None;
            let deep = rules.paint(&probe);
            probe.water_depth = Some(5.0);
            assert_eq!(deep, rules.paint(&probe), "{name} at {depth} m");
        }
    }
}

#[test]
fn established_lake_river_and_sea_beds_are_bare_across_seeds() {
    for seed in [42, 7] {
        let field = TerrainField::new(WorldSeed(seed));
        let mut counts = [0; 3];
        let mut elevated = 0;
        'columns: for k in -210..210 {
            for i in -210..210 {
                if counts.iter().all(|&count| count >= 20) && elevated > 0 {
                    break 'columns;
                }
                let (x, z) = (f64::from(i) * 32.0, f64::from(k) * 32.0);
                let Some(water) = field.water_surface(x, z) else {
                    continue;
                };
                let kind = match water.body {
                    WaterBody::Sea => 0,
                    WaterBody::Lake(_) => 1,
                    WaterBody::River(_) => 2,
                    _ => unreachable!(),
                };
                if counts[kind] >= 20 && elevated > 0 {
                    continue;
                }
                let Some(y) = field.topmost_surface(x, z) else {
                    continue;
                };
                if y > water.level - 0.1 {
                    continue;
                }
                let point = DVec3::new(x, y + 0.01, z);
                if !field.is_water(point) {
                    continue;
                }
                let sample = field.sample_position(WorldPosition(point));
                assert_ne!(
                    sample.material,
                    TerrainMaterial::SurfaceCover,
                    "seed {seed}, {point}"
                );
                assert_eq!(
                    sample.material,
                    field.palette().look(sample.surface).material
                );
                counts[kind] += 1;
                if kind == 1 && water.level > field.sea_level() + 5.0 {
                    elevated += 1;
                }
            }
        }
        assert!(
            counts.iter().all(|&count| count >= 20),
            "seed {seed}: {counts:?}"
        );
        assert!(elevated > 0, "seed {seed}: no elevated lakes checked");
    }
}

#[test]
fn empty_samples_above_water_paint_the_estimated_bed_in_both_mesh_paths() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let (x, z) = super::heart_of(&field, "sunken_coast");
    let water = field.water_surface(x, z).unwrap();
    let column = world.column(x, z);
    let bed = water.level - 1.0;
    let expected = field.palette().id("beach_sand").unwrap();
    // A locally planar bed: its empty samples reach above the waterline at
    // coarse LOD, but the tangent surface remains one metre underwater.
    for above in [0.05, 0.4, 1.6, 6.4] {
        let position = DVec3::new(x, bed + above, z);
        let density = -above;
        let gradient = [0.0, -1.0, 0.0];
        let columns = super::super::LatticeColumns {
            dims: [1; 3],
            columns: vec![column],
            carved: vec![0],
            trees: LatticeTrees::None,
        };
        assert_eq!(field.paint(position, density, gradient, 1).1, expected);
        assert_eq!(
            field
                .paint_lattice(&columns, [0; 3], position, density, gradient)
                .1,
            expected
        );
    }
    // Also compare the two paths on actual sampled lattices at several LODs.
    for stride in [1, 8, 32] {
        let origin = (DVec3::new(x, bed, z) / crate::TERRAIN_CELL_METERS).as_ivec3();
        let lattice = Lattice {
            origin,
            stride,
            dims: [2; 3],
            centred: false,
        };
        let (densities, columns, carved) = world.density_lattice(&lattice, false, true);
        let columns = super::super::LatticeColumns {
            dims: lattice.dims,
            columns,
            carved,
            trees: LatticeTrees::None,
        };
        let point = DVec3::new(
            lattice.coordinate(0, 0),
            lattice.coordinate(1, 0),
            lattice.coordinate(2, 0),
        );
        let gradient = [0.0, -1.0, 0.0];
        assert_eq!(
            field.paint(point, densities[0], gradient, lattice.stride),
            field.paint_lattice(&columns, [0; 3], point, densities[0], gradient)
        );
    }
}

#[test]
fn sealed_cave_materials_stay_dry_under_established_water() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let mut checked = 0;
    for k in -40..40 {
        for i in -40..40 {
            let (x, z) = (f64::from(i) * 128.0, f64::from(k) * 128.0);
            let column = world.column(x, z);
            let Some(water) = column.water.surface else {
                continue;
            };
            for below in 1..30 {
                let position = DVec3::new(x, water.level - f64::from(below) * 2.0, z);
                let (density, carved, blended) = world.density_parts(&column, position);
                if density > 0.0
                    || carved == 0
                    || super::super::CompiledWorld::open_to_water(
                        &column, position.y, carved, blended, water.body,
                    )
                {
                    continue;
                }
                let along = |axis| {
                    (field.density(position + axis * 0.05) - field.density(position - axis * 0.05))
                        / 0.1
                };
                let gradient = [along(DVec3::X), along(DVec3::Y), along(DVec3::Z)];
                let wet = world.sample(
                    &column,
                    position,
                    density,
                    gradient,
                    carved,
                    TreeSource::Lookup(TreeDetail::FINEST),
                );
                let mut dry_column = column;
                dry_column.water.surface = None;
                let dry = world.sample(
                    &dry_column,
                    position,
                    density,
                    gradient,
                    carved,
                    TreeSource::Lookup(TreeDetail::FINEST),
                );
                assert_eq!(
                    (wet.material, wet.surface),
                    (dry.material, dry.surface),
                    "{position}"
                );
                checked += 1;
                if checked >= 40 {
                    return;
                }
            }
        }
    }
    assert!(checked >= 40, "only {checked} buried void samples");
}

#[test]
fn surface_breaking_sea_trenches_receive_bed_materials() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    for k in -120..120 {
        for i in -120..120 {
            let (x, z) = (f64::from(i) * 64.0, f64::from(k) * 64.0);
            let column = world.column(x, z);
            if column
                .water
                .surface
                .is_none_or(|water| water.body != WaterBody::Sea)
            {
                continue;
            }
            let Some(y) = field.topmost_surface(x, z) else {
                continue;
            };
            let point = DVec3::new(x, y + 0.01, z);
            let (_, carved, blended) = world.density_parts(&column, point);
            if carved == 0 || blended < 1.0 || !field.is_water(point) {
                continue;
            }
            let sample = field.sample_position(WorldPosition(point));
            assert_ne!(sample.material, TerrainMaterial::SurfaceCover);
            let expected = match field.biome_at(x, z) {
                "sunken_coast" => ["beach_sand", "seabed_silt", "sea_cliff"],
                "verdant_hills" => ["mire_mud", "river_gravel", "granite"],
                _ => continue,
            };
            assert!(
                expected
                    .iter()
                    .any(|name| field.palette().id(name) == Some(sample.surface)),
                "{point}: {:?}",
                sample.surface
            );
            return;
        }
    }
    panic!("no exposed sea trench checked");
}
