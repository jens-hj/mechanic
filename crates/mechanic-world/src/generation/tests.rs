mod carves;

use bevy_math::{DVec3, IVec3};

use super::{Lattice, TerrainField, TerrainMaterial, WorldgenSpec};
use crate::{TerrainDensityClass, WorldCell, WorldPosition, WorldSeed};

#[test]
fn generation_is_deterministic_and_seeded() {
    let first = TerrainField::new(WorldSeed(42));
    let second = TerrainField::new(WorldSeed(42));
    let other = TerrainField::new(WorldSeed(43));
    let probes = [
        WorldPosition(DVec3::new(900.25, 17.0, -441.75)),
        WorldPosition(DVec3::new(-3_200.0, -20.0, 2_750.0)),
    ];
    for probe in probes {
        assert_eq!(first.sample_position(probe), second.sample_position(probe));
    }
    assert!(
        (first.surface_height(900.25, -441.75) - other.surface_height(900.25, -441.75)).abs()
            > f64::EPSILON
    );
}

#[test]
fn finite_world_is_empty_beyond_horizontal_bounds() {
    let field = TerrainField::new(WorldSeed(1));
    assert!(
        !field
            .sample_position(WorldPosition(DVec3::new(8_000.1, -100.0, 0.0)))
            .is_solid()
    );
}

#[test]
fn spawn_is_open_dry_and_nearly_level() {
    for seed in [WorldSeed(0), WorldSeed(7), WorldSeed(u64::MAX)] {
        let field = TerrainField::new(seed);
        let spawn = field.safe_spawn();
        assert!(!field.sample_position(spawn).is_solid());
        assert!(spawn.0.y > field.sea_level());
        let centre = field.surface_height(spawn.0.x, spawn.0.z);
        for (dx, dz) in [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)] {
            let neighbour = field.surface_height(spawn.0.x + dx, spawn.0.z + dz);
            assert!((neighbour - centre).abs() < 0.7, "seed {seed:?}");
        }
    }
}

#[test]
fn spawn_area_belongs_to_the_spawn_biome() {
    let field = TerrainField::new(WorldSeed(3));
    assert_eq!(field.biome_at(0.0, 0.0), "verdant_hills");
}

#[test]
fn cell_sampling_uses_exact_centres() {
    let field = TerrainField::new(WorldSeed(7));
    let cell = WorldCell::new(20_000, 0, -10_000);
    #[expect(clippy::cast_possible_truncation, reason = "sample densities are f32")]
    let density = field.density(cell.centre().0) as f32;
    assert_eq!(field.sample_cell(cell).density.to_bits(), density.to_bits());
}

#[test]
fn block_sampling_matches_cell_sampling_bit_for_bit() {
    let field = TerrainField::new(WorldSeed(11));
    let surface = field.surface_height(40.0, -25.0);
    #[expect(clippy::cast_possible_truncation, reason = "test cell index")]
    let minimum = WorldCell::new(800, (surface / 0.05) as i32 - 3, -500);
    let dims = [4, 6, 3];
    let block = field.sample_cells(minimum, dims);
    let mut index = 0;
    for z in 0..3 {
        for y in 0..6 {
            for x in 0..4 {
                let cell = WorldCell::new(minimum.x + x, minimum.y + y, minimum.z + z);
                assert_eq!(block[index], field.sample_cell(cell), "{cell:?}");
                index += 1;
            }
        }
    }
    assert!(block.iter().any(|sample| sample.is_solid()));
    assert!(block.iter().any(|sample| !sample.is_solid()));
}

#[test]
fn lattice_densities_match_point_densities() {
    let field = TerrainField::new(WorldSeed(5));
    let lattice = Lattice {
        origin: IVec3::new(-4_000, 200, 9_000),
        stride: 8,
        dims: [3, 4, 5],
        centred: false,
    };
    let values = field.density_lattice(&lattice);
    for k in 0..5 {
        for j in 0..4 {
            for i in 0..3 {
                let position = super::corner_position(WorldCell::new(
                    lattice.origin.x + 8 * i,
                    lattice.origin.y + 8 * j,
                    lattice.origin.z + 8 * k,
                ));
                let index = usize::try_from(i + 3 * (j + 4 * k)).unwrap();
                assert_eq!(values[index].to_bits(), field.density(position).to_bits());
            }
        }
    }
}

#[test]
fn surface_is_painted_with_the_top_rule_and_rock_beneath() {
    let field = TerrainField::new(WorldSeed(99));
    let surface = field.surface_height(30.0, 30.0);
    let material_at_depth = |depth: f64| {
        field
            .sample_position(WorldPosition(DVec3::new(30.0, surface - depth, 30.0)))
            .material
    };
    assert_eq!(material_at_depth(0.03), TerrainMaterial::SurfaceCover);
    assert_eq!(material_at_depth(12.0), TerrainMaterial::Rock);
}

#[test]
fn classification_never_contradicts_sampled_densities() {
    let field = TerrainField::new(WorldSeed(21));
    let mut checked = 0;
    for (x, z) in [
        (0.0, 0.0),
        (1_500.0, -800.0),
        (-3_000.0, 2_200.0),
        (4_100.0, 3_900.0),
        (-6_000.0, -5_500.0),
    ] {
        let ground = field.surface_height(x, z);
        for (edge, offset) in [
            (1.6, 0.0),
            (6.4, -3.0),
            (25.6, -12.0),
            (51.2, 30.0),
            (12.8, -60.0),
        ] {
            let minimum = DVec3::new(x, ground + offset, z);
            let maximum = minimum + DVec3::splat(edge);
            let class = field.classify(minimum, maximum);
            if class == TerrainDensityClass::Mixed {
                continue;
            }
            checked += 1;
            for i in 0..=6 {
                for j in 0..=6 {
                    for k in 0..=6 {
                        let t = DVec3::new(f64::from(i), f64::from(j), f64::from(k)) / 6.0;
                        let density = field.density(minimum + (maximum - minimum) * t);
                        match class {
                            TerrainDensityClass::Empty => assert!(density <= 0.0),
                            TerrainDensityClass::Solid => assert!(density > 0.0),
                            TerrainDensityClass::Mixed => unreachable!(),
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn every_biome_compiles_and_paints_with_known_surfaces() {
    let spec = WorldgenSpec::embedded();
    let field = TerrainField::from_spec(WorldSeed(1), spec).unwrap();
    for look in field.palette().looks() {
        assert!(
            look.tint
                .iter()
                .all(|channel| (0.0..=1.0).contains(channel))
        );
    }
}

/// A column well inside a biome: it and points 150 m around it all belong.
#[test]
fn chunks_judged_clear_hold_no_surface() {
    let field = TerrainField::new(WorldSeed(42));
    let names = field
        .spec()
        .biome_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut clear = 0;
    let mut checked = 0;
    for name in &names {
        let (x, z) = heart_of(&field, name);
        let ground = field.ground_height(x, z);
        for level in 2_u32..=5 {
            let stride = 1_i32 << level;
            let edge = 35_usize;
            let span = f64::from(stride) * crate::TERRAIN_CELL_METERS * 34.0;
            for step in -4_i32..=4 {
                #[expect(clippy::cast_possible_truncation, reason = "a few thousand cells")]
                let cell = |value: f64| (value / crate::TERRAIN_CELL_METERS).floor() as i32;
                let lattice = Lattice {
                    origin: IVec3::new(
                        cell(x + f64::from(step) * 13.0),
                        cell(ground + f64::from(step) * span * 0.4 - span * 0.5),
                        cell(z - f64::from(step) * 7.0),
                    ),
                    stride,
                    dims: [edge; 3],
                    centred: false,
                };
                for enclosed in [false, true] {
                    checked += 1;
                    if !field.lattice_is_clear(&lattice, enclosed) {
                        continue;
                    }
                    clear += 1;
                    let (exact, ..) = field.world.density_lattice(&lattice, false, enclosed);
                    let solid = exact[0] > 0.0;
                    assert!(
                        exact.iter().all(|&density| (density > 0.0) == solid),
                        "{name} L{level} step {step}: a lattice judged clear holds a surface"
                    );
                }
            }
        }
    }
    // Tunnels and ravines keep many lattices below ground from being clear.
    assert!(
        clear * 5 > checked,
        "only {clear} of {checked} lattices were judged clear"
    );
}

fn heart_of(field: &TerrainField, biome: &str) -> (f64, f64) {
    for step in 0..70 {
        for (x, z) in ring(step) {
            let inside = [
                (0.0, 0.0),
                (150.0, 0.0),
                (-150.0, 0.0),
                (0.0, 150.0),
                (0.0, -150.0),
            ]
            .into_iter()
            .all(|(dx, dz)| field.biome_at(x + dx, z + dz) == biome);
            if inside {
                return (x, z);
            }
        }
    }
    panic!("no {biome} in the world");
}

fn ring(step: u32) -> Vec<(f64, f64)> {
    if step == 0 {
        return vec![(0.0, 0.0)];
    }
    let radius = f64::from(step) * 100.0;
    let points = step * 8;
    (0..points)
        .map(|index| {
            let angle = f64::from(index) / f64::from(points) * core::f64::consts::TAU;
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect()
}

/// Columns around a biome's heart where ground stands over open space:
/// arches, overhangs, floating land.
fn overhung_columns(field: &TerrainField, biome: &str) -> usize {
    let (x0, z0) = heart_of(field, biome);
    let mut count = 0;
    for i in -30..=30 {
        for k in -30..=30 {
            let (x, z) = (x0 + f64::from(i) * 5.0, z0 + f64::from(k) * 5.0);
            let Some(top) = field.topmost_surface(x, z) else {
                continue;
            };
            let mut y = top - 1.0;
            let floor = field.ground_height(x, z) + 0.5;
            while y > floor {
                if field.density(DVec3::new(x, y, z)) < -0.5 {
                    count += 1;
                    break;
                }
                y -= 1.0;
            }
        }
    }
    count
}

fn relief(field: &TerrainField, biome: &str) -> (f64, f64) {
    let (x0, z0) = heart_of(field, biome);
    let heights = (-10..=10)
        .flat_map(|i| (-10..=10).map(move |k| (i, k)))
        .filter_map(|(i, k)| {
            field.topmost_surface(x0 + f64::from(i) * 8.0, z0 + f64::from(k) * 8.0)
        })
        .collect::<Vec<_>>();
    let count = f64::from(u32::try_from(heights.len()).expect("a few hundred samples"));
    let mean = heights.iter().sum::<f64>() / count;
    let spread = (heights.iter().map(|h| (h - mean) * (h - mean)).sum::<f64>() / count).sqrt();
    let peak = heights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (spread, peak - mean)
}

#[test]
fn arches_and_floating_isles_leave_open_space_under_ground() {
    let field = TerrainField::new(WorldSeed(42));
    assert!(overhung_columns(&field, "arch_steppe") > 20);
    assert!(overhung_columns(&field, "drift_isles") > 100);
    assert!(overhung_columns(&field, "gyroid_reef") > 50);
    // Familiar ground is a plain heightfield apart from caves and boulders.
    assert!(overhung_columns(&field, "dune_sea") < 5);
}

#[test]
fn dunes_stay_low_while_needles_and_crags_tower() {
    let field = TerrainField::new(WorldSeed(42));
    let (dune_spread, _) = relief(&field, "dune_sea");
    let (_, needle_peak) = relief(&field, "karst_needles");
    let (crag_spread, _) = relief(&field, "titan_crags");
    assert!(dune_spread < 5.0, "dunes spread {dune_spread}");
    assert!(needle_peak > 15.0, "needles rise {needle_peak}");
    assert!(
        crag_spread > dune_spread * 2.0,
        "crags spread {crag_spread}"
    );
}

#[test]
fn every_biome_classifies_soundly_around_its_surface() {
    let field = TerrainField::new(WorldSeed(42));
    let names = field
        .spec()
        .biome_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for name in names {
        let (x0, z0) = heart_of(&field, &name);
        let ground = field.ground_height(x0, z0);
        for (index, edge) in [3.2, 6.4, 12.8, 25.6].into_iter().enumerate() {
            for offset in [-40.0, -12.0, -3.0, 0.0, 6.0, 30.0, 70.0] {
                let shift = f64::from(u32::try_from(index).unwrap()) * 7.3;
                let minimum = DVec3::new(x0 + shift, ground + offset, z0 - shift);
                let maximum = minimum + DVec3::splat(edge);
                for class in [
                    field.classify(minimum, maximum),
                    field.classify_distant(minimum, maximum),
                ] {
                    if class == TerrainDensityClass::Mixed {
                        continue;
                    }
                    for i in 0..=5 {
                        for j in 0..=5 {
                            for k in 0..=5 {
                                let t = DVec3::new(f64::from(i), f64::from(j), f64::from(k)) / 5.0;
                                let density = field.density(minimum + (maximum - minimum) * t);
                                // Distant views close caves, so only their empty
                                // verdicts must hold exactly.
                                if class == TerrainDensityClass::Empty {
                                    assert!(density <= 0.0, "{name} {minimum:?} {edge}");
                                } else if field.classify(minimum, maximum) == class {
                                    assert!(density > 0.0, "{name} {minimum:?} {edge}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn culled_lattices_keep_every_sign() {
    let field = TerrainField::new(WorldSeed(42));
    let names = field
        .spec()
        .biome_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for name in &names {
        let (hx, hz) = heart_of(&field, name);
        for level in [2_u32, 4] {
            let stride = 1_i32 << level;
            for chunk in 0..3 {
                let offset = f64::from(chunk) * 37.0 - 40.0;
                let (x, z) = (hx + offset, hz - offset * 0.7);
                let ground = field.ground_height(x, z);
                #[expect(clippy::cast_possible_truncation, reason = "a few thousand cells")]
                let cell = |value: f64| (value / crate::TERRAIN_CELL_METERS).floor() as i32;
                let lattice = Lattice {
                    origin: IVec3::new(cell(x), cell(ground) - 16 * stride, cell(z)),
                    stride,
                    dims: [35, 35, 35],
                    centred: false,
                };
                for enclosed in [true, false] {
                    let (culled, ..) = field.world.density_lattice(&lattice, true, enclosed);
                    let (exact, ..) = field.world.density_lattice(&lattice, false, enclosed);
                    assert!(
                        culled
                            .iter()
                            .zip(&exact)
                            .all(|(culled, exact)| (*culled > 0.0) == (*exact > 0.0)),
                        "{name} L{level} chunk {chunk}: culling changed a sign"
                    );
                }
            }
        }
    }
}
