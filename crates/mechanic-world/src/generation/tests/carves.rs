//! Carve layers: tunnels that open only where their roof lets them, caves
//! inside mountains, walk-in entrance ramps, and ravines that run open or
//! buried.

use std::collections::BTreeMap;

use bevy_math::{DVec3, IVec3};

use super::heart_of;
use crate::WorldSeed;
use crate::generation::compile::{Scope, compile};
use crate::generation::spec::Expr;
use crate::generation::{Column, CompiledWorld, Lattice, TerrainField};

fn layer(world: &CompiledWorld, name: &str) -> usize {
    world
        .carves
        .iter()
        .position(|carve| carve.name == name)
        .unwrap_or_else(|| panic!("no carve layer `{name}`"))
}

/// Biome density before carving: how deep into rock a point lies.
fn rock(world: &CompiledWorld, column: &Column, point: DVec3) -> f64 {
    column
        .weights
        .iter()
        .map(|(biome, weight)| weight * world.biomes[biome].density.eval(point.to_array(), &[]))
        .sum()
}

#[test]
fn tunnels_breach_where_the_roof_opens_and_stay_sealed_elsewhere() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let tunnels = layer(world, "tunnels");
    let mut breaches = 0;
    let mut tunnel_points = 0;
    for biome in ["verdant_hills", "titan_crags", "karst_needles"] {
        let (x0, z0) = heart_of(&field, biome);
        for i in -50..50 {
            for k in -50..50 {
                let (x, z) = (x0 + f64::from(i) * 6.0, z0 + f64::from(k) * 6.0);
                let column = world.column(x, z);
                let roof = column.carves[tunnels].roof;
                for depth in 0..40 {
                    let point = DVec3::new(x, column.ground + 4.0 - f64::from(depth), z);
                    let (density, carved) = world.density_and_carve(&column, point);
                    if density > 0.0 || usize::from(carved) != tunnels + 1 {
                        continue;
                    }
                    tunnel_points += 1;
                    let depth = rock(world, &column, point);
                    assert!(
                        depth > roof,
                        "{biome}: a tunnel {depth:.1} m into rock under a {roof:.1} m roof"
                    );
                    if depth < 0.5 {
                        breaches += 1;
                    }
                }
            }
        }
    }
    assert!(tunnel_points > 1_000, "only {tunnel_points} tunnel samples");
    assert!(
        breaches > 20,
        "tunnels broke through the surface {breaches} times"
    );
}

#[test]
fn caves_reach_into_mountain_mass_above_the_old_band() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let tunnels = layer(world, "tunnels");
    let (x0, z0) = heart_of(&field, "titan_crags");
    let (mut deep, mut high) = (0, 0);
    for i in -40..40 {
        for k in -40..40 {
            let (x, z) = (x0 + f64::from(i) * 8.0, z0 + f64::from(k) * 8.0);
            let column = world.column(x, z);
            let base = world.fields.base.sample(x, z);
            let mut y = column.ground;
            while y > base - 30.0 {
                let (density, carved) = world.density_and_carve(&column, DVec3::new(x, y, z));
                if density <= 0.0 && usize::from(carved) == tunnels + 1 {
                    // Caves used to stop 48 m under the ground.
                    deep += usize::from(column.ground - y > 48.0);
                    high += usize::from(y - base > 40.0);
                }
                y -= 1.5;
            }
        }
    }
    assert!(deep > 50, "only {deep} tunnel samples more than 48 m down");
    assert!(
        high > 50,
        "only {high} tunnel samples high inside mountains"
    );
}

#[test]
fn entrance_ramps_descend_gradually_from_the_surface() {
    let spec = crate::WorldgenSpec::embedded();
    let empty = BTreeMap::new();
    let scope = Scope {
        local: &empty,
        library: &spec.library.definitions,
        fields: None,
    };
    let inputs = ["hall", "w", "rock"].map(str::to_owned);
    let ramp = compile(
        &Expr::Ref("entrance_ramp".to_owned()),
        scope,
        &inputs,
        1,
        "ramp",
    )
    .expect("the library ramp compiles");
    // Flat ground at y = 0 along the ramp's own axis, x.
    let open = |x: f64, y: f64| ramp.eval([x, y, 0.0], &[9.0, 3.0, -y]) > 0.0;
    // The lowest point of the first open run met coming down from above.
    let floor = |x: f64| {
        let mut y = 2.0;
        while !open(x, y) && y > -60.0 {
            y -= 0.05;
        }
        while open(x, y - 0.05) && y > -60.0 {
            y -= 0.05;
        }
        y
    };
    let mut previous = floor(0.0);
    assert!(previous < -1.0, "the ramp starts {previous} m down");
    for step in 1..=35 {
        let x = f64::from(step) * 2.0;
        let height = floor(x);
        let grade = (previous - height) / 2.0;
        assert!(
            (0.0..0.58).contains(&grade),
            "grade {grade:.2} at {x} m along the ramp"
        );
        previous = height;
    }
    // Open to the sky at its mouth, roofed once it runs deep.
    assert!((floor(2.0)..1.0).step_by_fraction().all(|y| open(2.0, y)));
    let deep = floor(40.0);
    assert!(open(40.0, deep + 1.0) && !open(40.0, 0.0) && !open(40.0, -3.0));
}

trait StepByFraction {
    fn step_by_fraction(self) -> impl Iterator<Item = f64>;
}

impl StepByFraction for core::ops::Range<f64> {
    fn step_by_fraction(self) -> impl Iterator<Item = f64> {
        let (start, end) = (self.start + 0.1, self.end);
        (0..)
            .map(move |index| f64::from(index).mul_add(0.25, start))
            .take_while(move |y| *y < end)
    }
}

#[test]
fn ravines_open_or_buried_follow_the_roof_sign() {
    let field = TerrainField::new(WorldSeed(42));
    let world = &field.world;
    let ravines = layer(world, "ravines");
    let (mut open, mut buried) = (0, 0);
    for biome in ["arch_steppe", "titan_crags"] {
        let (x0, z0) = heart_of(&field, biome);
        for i in -100..100 {
            for k in -100..100 {
                let (x, z) = (x0 + f64::from(i) * 8.0, z0 + f64::from(k) * 8.0);
                let column = world.column(x, z);
                let roof = column.carves[ravines].roof;
                let point = DVec3::new(x, column.ground - 20.0, z);
                let (density, carved) = world.density_and_carve(&column, point);
                if density > 0.0 || usize::from(carved) != ravines + 1 {
                    continue;
                }
                let surface = DVec3::new(x, column.ground - 1.0, z);
                if roof >= 6.0 {
                    // Sealed: every opening the ravine makes in this column
                    // keeps the roof's rock between it and the surface.
                    buried += 1;
                    for step in 0..60 {
                        let point = DVec3::new(x, column.ground + 10.0 - f64::from(step), z);
                        let (density, carved) = world.density_and_carve(&column, point);
                        if density <= 0.0 && usize::from(carved) == ravines + 1 {
                            let depth = rock(world, &column, point);
                            assert!(
                                depth > roof,
                                "{biome}: a buried ravine {depth:.1} m down at {x}, {z}"
                            );
                        }
                    }
                } else if roof <= -6.0 {
                    open += 1;
                    assert!(
                        field.density(surface) <= 0.0,
                        "{biome}: an open ravine is sealed at {x}, {z}"
                    );
                }
            }
        }
    }
    assert!(open > 20, "only {open} open ravine columns");
    assert!(buried > 20, "only {buried} buried ravine columns");
}

#[test]
fn distant_lattices_keep_surface_carves_and_close_enclosed_ones() {
    let field = TerrainField::new(WorldSeed(42));
    let (x, z) = heart_of(&field, "karst_needles");
    let ground = field.ground_height(x, z);
    #[expect(clippy::cast_possible_truncation, reason = "a few thousand cells")]
    let cell = |value: f64| (value / crate::TERRAIN_CELL_METERS).floor() as i32;
    // A metre apart, from 55 m under the ground to 5 m under it.
    let lattice = Lattice {
        origin: IVec3::new(cell(x - 75.0), cell(ground - 55.0), cell(z - 75.0)),
        stride: 20,
        dims: [150, 50, 150],
        centred: false,
    };
    let (near, ..) = field.world.density_lattice(&lattice, false, true);
    let (far, ..) = field.world.density_lattice(&lattice, false, false);
    assert!(
        near.iter().zip(&far).all(|(near, far)| far >= near),
        "closing voids only ever adds ground"
    );
    let index = |i: usize, j: usize, k: usize| i + 150 * (j + 50 * k);
    let (mut enclosed, mut kept) = (0, 0);
    for k in 0..150 {
        for i in 0..150 {
            for j in 0..50 {
                let at = index(i, j, k);
                if near[at] > 0.0 {
                    continue;
                }
                // Air a few samples under ground: a tunnel, shaft, or ramp.
                let y = lattice.coordinate(1, j);
                let column_ground =
                    field.ground_height(lattice.coordinate(0, i), lattice.coordinate(2, k));
                if y > column_ground - 3.0 {
                    continue;
                }
                if far[at] > 0.0 {
                    enclosed += 1;
                } else {
                    kept += 1;
                }
            }
        }
    }
    assert!(
        enclosed > 0,
        "the karst's deep tunnels should close from afar ({kept} kept)"
    );
    assert!(
        kept > 0,
        "shafts and open tunnels should stay open from afar"
    );
}
