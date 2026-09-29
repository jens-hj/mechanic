//! Stored water on generated terrain: a trench dug from a real lake fills.

use bevy_math::{DVec2, DVec3};

use crate::water::{TerrainWater, WaterGround, WaterWorld};
use crate::{
    TerrainField, TerrainOctree, WaterBody, WaterTile, WorldPosition, WorldSeed, joined_water_sheet,
};

/// A lake column near spawn and a dry bank beside it: the lake point, the
/// bank point, and the lake's level.
fn lake_shore(field: &TerrainField) -> (DVec3, DVec3, f64) {
    let spawn = field.safe_spawn().0;
    for ring in 1..80 {
        let radius = f64::from(ring) * 4.0;
        for step in 0..ring * 8 {
            let angle = f64::from(step) / f64::from(ring * 8) * std::f64::consts::TAU;
            let (x, z) = (
                spawn.x + radius * angle.cos(),
                spawn.z + radius * angle.sin(),
            );
            let Some(surface) = field.water_surface(x, z) else {
                continue;
            };
            let deep = field
                .topmost_surface(x, z)
                .is_some_and(|ground| ground < surface.level - 0.6);
            if !matches!(surface.body, WaterBody::Lake(_)) || !deep {
                continue;
            }
            for direction in 0..8 {
                let angle = f64::from(direction) * std::f64::consts::FRAC_PI_4;
                for reach in 2..12 {
                    let (bx, bz) = (
                        x + f64::from(reach) * angle.cos(),
                        z + f64::from(reach) * angle.sin(),
                    );
                    if field
                        .topmost_surface(bx, bz)
                        .is_some_and(|ground| ground > surface.level + 0.3)
                    {
                        let level = surface.level;
                        return (DVec3::new(x, level, z), DVec3::new(bx, level, bz), level);
                    }
                }
            }
        }
    }
    panic!("no lake shore near spawn");
}

#[test]
fn a_trench_dug_from_a_generated_lake_fills_to_its_level() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, bank, level) = lake_shore(&field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    // A trench from well inside the bank out into the lake, its floor half
    // a metre under the water.
    let beyond = bank + (bank - lake).normalize() * 2.0;
    let steps = 40;
    for step in 0..=steps {
        let point = beyond.lerp(lake, f64::from(step) / f64::from(steps));
        let centre = WorldPosition(DVec3::new(point.x, level - 0.2, point.z));
        let outcome = terrain.excavate_sphere(&field, centre, 0.5).unwrap();
        bricks.extend_from_slice(outcome.changed_brick_coordinates());
    }
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    for step in 0..600 {
        let before = water.ledger();
        water.step(&ground, 0.05);
        let after = water.ledger();
        assert!(
            (after.total() - before.total()).abs() < 1.0e-9,
            "step {step} made or lost water: {before:?} -> {after:?}"
        );
    }
    let dug = DVec3::new(beyond.x, level - 0.5, beyond.z);
    let surface = water.surface(&ground, dug).expect("the trench holds water");
    assert!(
        matches!(surface.body, WaterBody::Lake(_)),
        "the dug trench did not join the lake"
    );
    assert!(
        (surface.level - level).abs() < 0.05,
        "the trench stands at {:.3} m under a lake at {level:.3} m",
        surface.level
    );
    let held = water.stored_m3() + water.joined_m3();
    assert!(held > 0.5, "only {held:.2} m³ ran in");
    // The lake's own sheet runs on over the trench, as deep as it was dug.
    let joined = water
        .joined_cells(&ground)
        .into_iter()
        .map(|(cell, _)| cell)
        .collect::<std::collections::HashSet<_>>();
    let tile = WaterTile {
        minimum: [beyond.x - 4.0, beyond.z - 4.0],
        edge: 8.0,
        cells: 32,
    };
    let shifts = water.cycle.shifts(&ground);
    let sheet = joined_water_sheet(
        &field,
        &terrain,
        tile,
        &shifts,
        &joined,
        &std::collections::HashSet::new(),
    )
    .expect("the trench shows water");
    let deepest = sheet
        .vertices
        .iter()
        .zip(&sheet.depths)
        .filter(|(vertex, _)| {
            let point = DVec2::new(
                f64::from(vertex[0]) + tile.minimum[0],
                f64::from(vertex[2]) + tile.minimum[1],
            );
            point.distance(DVec2::new(beyond.x, beyond.z)) < 1.0
        })
        .map(|(_, &depth)| depth)
        .fold(0.0_f32, f32::max);
    assert!(deepest > 0.4, "the trench shows {deepest:.2} m of water");
    // Water fills the dug trench and does not creep off along the shore.
    let flat = |point: DVec3| DVec2::new(point.x, point.z);
    let (from, to) = (flat(beyond), flat(lake));
    for (cell, _) in water.joined_cells(&ground) {
        let point = flat(cell.centre());
        let along = ((point - from).dot(to - from) / from.distance_squared(to)).clamp(0.0, 1.0);
        let off = point.distance(from.lerp(to, along));
        assert!(
            off < 2.5,
            "{cell:?} joined the lake {off:.1} m off the trench"
        );
    }
}

/// Whether a point lies under any triangle of a sheet, seen from above.
fn covers(sheet: &crate::WaterSheet, point: DVec2) -> bool {
    let at = |index: u32| {
        let vertex = sheet.vertices[index as usize];
        DVec2::new(
            f64::from(vertex[0]) + sheet.origin.0.x,
            f64::from(vertex[2]) + sheet.origin.0.z,
        )
    };
    sheet.indices.chunks(3).any(|triangle| {
        let [a, b, c] = [at(triangle[0]), at(triangle[1]), at(triangle[2])];
        let side = |p: DVec2, q: DVec2| (q - p).perp_dot(point - p);
        let (first, second, third) = (side(a, b), side(b, c), side(c, a));
        (first >= 0.0 && second >= 0.0 && third >= 0.0)
            || (first <= 0.0 && second <= 0.0 && third <= 0.0)
    })
}

/// The columns the stored water's surface draws a quad over.
fn drawn_columns(
    water: &mut WaterWorld,
    ground: &impl crate::water::WaterGround,
) -> std::collections::HashSet<(i32, i32)> {
    let mut drawn = std::collections::HashSet::new();
    for tile in water.surface_tiles(ground, &std::collections::HashMap::new()) {
        for quad in tile.indices.chunks(6) {
            let centre = quad
                .iter()
                .map(|&index| DVec3::from(tile.positions[index as usize].map(f64::from)))
                .sum::<DVec3>()
                / 6.0
                + tile.origin;
            drawn.insert((
                crate::water::WaterCell::containing(centre).x,
                crate::water::WaterCell::containing(centre).z,
            ));
        }
    }
    drawn
}

#[test]
fn a_lakes_sheet_meets_the_running_water_in_a_channel_dug_from_it() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, bank, level) = lake_shore(&field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    // A wide channel from the bank out into the lake, deeper than its edge.
    let beyond = bank + (bank - lake).normalize() * 3.0;
    let steps = 60;
    for step in 0..=steps {
        let point = beyond.lerp(lake, f64::from(step) / f64::from(steps));
        for lift in [0.2, 1.0] {
            let centre = WorldPosition(DVec3::new(point.x, level - lift, point.z));
            let outcome = terrain.excavate_sphere(&field, centre, 1.2).unwrap();
            bricks.extend_from_slice(outcome.changed_brick_coordinates());
        }
    }
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    for _ in 0..1_200 {
        water.step(&ground, 0.05);
    }
    let joined = water
        .joined_cells(&ground)
        .into_iter()
        .map(|(cell, _)| cell)
        .collect::<std::collections::HashSet<_>>();
    let meeting = water
        .meeting_columns(&ground)
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    assert!(!meeting.is_empty(), "no running water meets the lake");
    // Water fed by the lake never stands above it.
    let highest = water
        .running_cells()
        .into_iter()
        .map(|view| view.level)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        highest < level + 0.05,
        "running water stands at {highest:.3} m over a lake at {level:.3} m"
    );
    // The lake's finest tiles: 64 m with a vertex every metre.
    let middle = (beyond + lake) * 0.5;
    let tile = WaterTile {
        minimum: [middle.x - 32.0, middle.z - 32.0],
        edge: 64.0,
        cells: 64,
    };
    let shifts = water.cycle.shifts(&ground);
    let sheet = joined_water_sheet(&field, &terrain, tile, &shifts, &joined, &meeting)
        .expect("the lake shows");
    let drawn = drawn_columns(&mut water, &ground);
    // Every lake column beside the running water lies under the lake's
    // sheet or the running water's surface: none is left bare between the
    // two.
    let centre = |x: i32, z: i32| {
        let centre = crate::water::WaterCell::new(x, 0, z).centre();
        DVec2::new(centre.x, centre.z)
    };
    let mut beside = 0;
    for &(x, z) in &meeting {
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x + dx, z + dz);
            let point = centre(nx, nz);
            let probe = DVec3::new(point.x, level - 0.02, point.y);
            let lake_water = field.is_water(probe)
                || joined.contains(&crate::water::WaterCell::containing(probe));
            if meeting.contains(&(nx, nz)) || !lake_water {
                continue;
            }
            beside += 1;
            assert!(
                covers(&sheet, point) || drawn.contains(&(nx, nz)),
                "the lake leaves ({nx}, {nz}) bare"
            );
        }
    }
    assert!(beside > 0, "the running water touches no lake water");
}

/// A lake column near spawn whose bank, walked away from the lake, rises
/// above the lake's level and then falls well below it: the lake point, the
/// crest of the bank, a point down the far side, and the lake's level.
fn lake_over_a_hill(field: &TerrainField) -> Option<(DVec3, DVec3, DVec3, f64)> {
    let spawn = field.safe_spawn().0;
    for ring in 1..80 {
        let radius = f64::from(ring) * 4.0;
        for step in 0..ring * 8 {
            let angle = f64::from(step) / f64::from(ring * 8) * std::f64::consts::TAU;
            let (x, z) = (
                spawn.x + radius * angle.cos(),
                spawn.z + radius * angle.sin(),
            );
            let Some(surface) = field.water_surface(x, z) else {
                continue;
            };
            let level = surface.level;
            let deep = field
                .topmost_surface(x, z)
                .is_some_and(|ground| ground < level - 0.6);
            if !matches!(surface.body, WaterBody::Lake(_)) || !deep {
                continue;
            }
            for direction in 0..16 {
                let angle = f64::from(direction) * std::f64::consts::TAU / 16.0;
                let along = DVec2::new(angle.cos(), angle.sin());
                let mut crest = None;
                for reach in 1..60 {
                    let point = DVec2::new(x, z) + along * (f64::from(reach) * 0.5);
                    let Some(ground) = field.topmost_surface(point.x, point.y) else {
                        break;
                    };
                    match crest {
                        None if ground > level + 0.3 => {
                            // A bank at most a metre high, so a channel through
                            // it stays small.
                            if ground > level + 1.0 {
                                break;
                            }
                            crest = Some(point);
                        }
                        Some(top) if ground < level - 1.0 => {
                            return Some((
                                DVec3::new(x, level, z),
                                DVec3::new(top.x, level, top.y),
                                DVec3::new(point.x, ground, point.y),
                                level,
                            ));
                        }
                        Some(_) if ground > level + 1.0 => break,
                        _ => {}
                    }
                }
            }
        }
    }
    None
}

/// The lake over a hill, with a channel 30 cm deep dug from the lake's
/// edge through the bank and a little way down its far side: the edits,
/// the bricks they changed, the channel's two ends and the lake's level.
fn channel_over_a_hill(
    field: &TerrainField,
) -> (TerrainOctree, Vec<crate::BrickCoord>, DVec3, DVec3, f64) {
    let (lake, crest, below, level) = lake_over_a_hill(field).expect("a lake over a hill");
    let along = (crest - lake).normalize();
    let ground = |point: DVec3| field.topmost_surface(point.x, point.z).unwrap_or(level);
    // From the last of the lake's water to where the far side falls under
    // the channel's floor.
    let mut start = crest;
    while !field.is_water(DVec3::new(start.x, level - 0.05, start.z)) && start.distance(lake) > 0.5
    {
        start -= along * 0.2;
    }
    let mut end = crest;
    while ground(end) > level - 0.3 && end.distance(below) > 0.5 {
        end += along * 0.2;
    }
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    let steps = steps(start.distance(end), 0.2);
    for step in 0..=steps {
        let point = start.lerp(end, f64::from(step) / f64::from(steps));
        let centre = WorldPosition(DVec3::new(point.x, level - 0.3 + 0.5, point.z));
        let outcome = terrain.excavate_sphere(field, centre, 0.5).unwrap();
        bricks.extend_from_slice(outcome.changed_brick_coordinates());
    }
    (terrain, bricks, start, end, level)
}

/// Steps of at most `step` metres along `length` metres.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a walk of a few metres"
)]
fn steps(length: f64, step: f64) -> u32 {
    (length / step).ceil() as u32
}

/// How far a point lies from the segment between two others, over the
/// ground.
fn off_segment(point: DVec3, from: DVec3, to: DVec3) -> f64 {
    let flat = |point: DVec3| DVec2::new(point.x, point.z);
    let (point, from, to) = (flat(point), flat(from), flat(to));
    let along = ((point - from).dot(to - from) / from.distance_squared(to)).clamp(0.0, 1.0);
    point.distance(from.lerp(to, along))
}

/// Whether the generator put water in a cell's column just under `level`.
fn generated_water(field: &TerrainField, cell: crate::water::WaterCell, level: f64) -> bool {
    let centre = cell.centre();
    let half = 0.5 * super::super::WATER_CELL_METRES;
    [
        (0.0, 0.0),
        (-0.8, -0.8),
        (-0.8, 0.8),
        (0.8, -0.8),
        (0.8, 0.8),
    ]
    .into_iter()
    .any(|(dx, dz)| {
        field.is_water(DVec3::new(
            centre.x + dx * half,
            level - 0.02,
            centre.z + dz * half,
        ))
    })
}

#[test]
fn a_lake_holds_water_only_where_the_generator_put_it() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, crest, below, level) = lake_over_a_hill(&field).expect("a lake over a hill");
    let terrain = TerrainOctree::default();
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    // Every cell near the lake's level from the lake over the bank and down
    // its far side, where land lies a little under the lake's level with no
    // water on it.
    let steps = steps(lake.distance(below), 0.1);
    let mut checked = 0;
    for step in 0..=steps {
        let point = lake.lerp(below, f64::from(step) / f64::from(steps));
        for dy in -3..=1 {
            let probe = DVec3::new(point.x, level + f64::from(dy) * 0.2, point.z);
            let cell = crate::water::WaterCell::containing(probe);
            if water.implicit(&ground, cell).is_some() {
                checked += 1;
                assert!(
                    generated_water(&field, cell, level),
                    "{cell:?}, {:.1} m past the crest, holds lake water the generator never put there",
                    (probe - crest).dot((below - lake).normalize())
                );
            }
        }
    }
    assert!(checked > 0, "the walk found no lake water at all");
}

#[test]
fn a_breach_down_a_hill_runs_downhill_at_flowing_speed() {
    let field = TerrainField::new(WorldSeed(42));
    let (terrain, bricks, start, end, level) = channel_over_a_hill(&field);
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    for second in 1..=30_u32 {
        for _ in 0..20 {
            let before = water.ledger().total();
            water.step(&ground, 0.05);
            assert!(
                (water.ledger().total() - before).abs() < 1.0e-9,
                "water was made or lost"
            );
        }
        // Lake water pours in only at the dig.
        for &cell in &water.inlets {
            let off = off_segment(cell.centre(), start, end);
            assert!(
                off < 1.5,
                "after {second} s {cell:?} pours lake water in {off:.1} m off the dig"
            );
        }
        // Only what lies under the lake or in the dig joins it.
        for (cell, _) in water.joined_cells(&ground) {
            let off = off_segment(cell.centre(), start, end);
            assert!(
                off < 1.5 || generated_water(&field, cell, level),
                "after {second} s {cell:?} joined the lake {off:.1} m off the dig"
            );
        }
        // Water beyond the lake spreads from the dig no faster
        // than water flows down a hill.
        let front = water
            .running_cells()
            .into_iter()
            .map(|running| running.cell)
            .chain(water.pools().flat_map(|pool| pool.surface_cells))
            .filter(|&cell| !generated_water(&field, cell, level))
            .map(|cell| off_segment(cell.centre(), start, end))
            .fold(0.0, f64::max);
        let bound = 3.0f64.mul_add(f64::from(second), 1.0);
        assert!(
            front < bound,
            "after {second} s water stands {front:.1} m from the dig"
        );
    }
}

#[test]
fn plugging_a_breach_stops_the_flow_from_the_lake() {
    let field = TerrainField::new(WorldSeed(42));
    let (mut terrain, bricks, start, end, level) = channel_over_a_hill(&field);
    let body = field
        .water_surface(start.x, start.z)
        .map(|surface| surface.body)
        .expect("the dig starts in the lake");
    let mut water = WaterWorld::new();
    water.terrain_changed(
        &TerrainWater {
            field: &field,
            edits: &terrain,
        },
        bricks,
    );
    for _ in 0..400 {
        water.step(
            &TerrainWater {
                field: &field,
                edits: &terrain,
            },
            0.05,
        );
    }
    // Fill the channel back in where it crosses the bank.
    let mut bricks = Vec::new();
    let steps = steps(start.distance(end), 0.2);
    for step in 0..=steps {
        let point = start.lerp(end, f64::from(step) / f64::from(steps));
        let centre = WorldPosition(DVec3::new(point.x, level + 0.2, point.z));
        let outcome = terrain
            .add_sphere(&field, centre, 0.7, crate::TerrainMaterial::Soil)
            .unwrap();
        bricks.extend_from_slice(outcome.changed_brick_coordinates());
    }
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    water.terrain_changed(&ground, bricks);
    water.step(&ground, 0.05);
    let held = |water: &WaterWorld| {
        let ledger = water.ledger();
        ledger.running_m3 + ledger.pools_m3 + ledger.joined_m3 + ledger.falling_m3 + ledger.soil_m3
    };
    let (plugged_held, plugged_surplus) = (held(&water), water.surplus_m3(body));
    for _ in 0..1_200 {
        let before = water.ledger().total();
        water.step(&ground, 0.05);
        assert!(
            (water.ledger().total() - before).abs() < 1.0e-9,
            "water was made or lost"
        );
    }
    assert!(
        water.pools().all(|pool| !pool.surface_cells.is_empty()),
        "water stands as a pool inside the fill"
    );
    assert!(
        held(&water) < plugged_held + 1.0e-3,
        "water beyond the plug grew from {plugged_held:.3} m³ to {:.3} m³",
        held(&water)
    );
    assert!(
        water.surplus_m3(body) > plugged_surplus - 1.0e-3,
        "the lake gave {:.3} m³ more after it was plugged",
        plugged_surplus - water.surplus_m3(body)
    );
}

#[test]
fn water_reads_the_ground_where_the_terrain_mesh_draws_it() {
    let field = TerrainField::new(WorldSeed(42));
    let spawn = field.safe_spawn().0;
    let mut terrain = TerrainOctree::default();
    let dug = {
        let y = field
            .topmost_surface(spawn.x, spawn.z)
            .expect("ground at spawn");
        DVec3::new(spawn.x, y, spawn.z)
    };
    terrain
        .excavate_sphere(&field, WorldPosition(dug), 0.6)
        .unwrap();
    let untouched = {
        let (x, z) = (spawn.x + 20.0, spawn.z + 20.0);
        let y = field.topmost_surface(x, z).expect("ground beside spawn");
        DVec3::new(x, y, z)
    };
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let snapshot = terrain.snapshot();
    for point in [dug, untouched] {
        let brick = WorldPosition(point).cell().unwrap().brick();
        let chunk = crate::mesh_chunk(
            &field,
            &snapshot,
            crate::TerrainMeshRequest {
                node: crate::TerrainNodeId::leaf(brick),
                generation: 0,
                transition_mask: crate::TerrainTransitionMask::default(),
            },
        );
        let mut checked = 0;
        for (vertex, normal) in chunk.vertices.iter().zip(&chunk.normals) {
            let at = chunk.origin.0 + DVec3::from(vertex.map(f64::from));
            let on_column = |value: f64| {
                let cells = value / crate::TERRAIN_CELL_METERS;
                (cells - cells.round()).abs() < 1.0e-4
            };
            // Ground facing up, crossed on a lattice column.
            if normal[1] < 0.7 || !on_column(at.x) || !on_column(at.z) {
                continue;
            }
            let top = ground
                .ground_top(at.x, at.z, at.y)
                .expect("ground under a vertex");
            assert!(
                (top - at.y).abs() < 2.0e-3,
                "the mesh draws the ground at {at:?}, water reads it at {top:.4}"
            );
            checked += 1;
        }
        assert!(
            checked > 10,
            "only {checked} vertices near {point:?} checked"
        );
    }
}
