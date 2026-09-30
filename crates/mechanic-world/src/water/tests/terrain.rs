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
    for tile in water
        .surface_tiles(ground, &std::collections::HashMap::new())
        .tiles
    {
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
    let owned = water
        .surface_tiles(&ground, &std::collections::HashMap::new())
        .owned
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    assert!(
        !owned.is_empty(),
        "the running water draws none of the lake"
    );
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
    let sheet = joined_water_sheet(&field, &terrain, tile, &shifts, &joined, &owned)
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
    for &(x, z) in &owned {
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x + dx, z + dz);
            let point = centre(nx, nz);
            let probe = DVec3::new(point.x, level - 0.02, point.y);
            let lake_water = field.is_water(probe)
                || joined.contains(&crate::water::WaterCell::containing(probe));
            if owned.contains(&(nx, nz)) || !lake_water {
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

/// How opaque water drawn by triangles is over a square of points `step`
/// apart from `minimum`, `side` along each edge, seen from above, as the
/// shader fades it by the depth under it.
fn opacity(
    triangles: &[([DVec2; 3], [f64; 3])],
    minimum: DVec2,
    step: f64,
    side: usize,
) -> Vec<f64> {
    let mut alpha = vec![0.0_f64; side * side];
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "indices of a small square of points"
    )]
    for &([first, second, third], depth) in triangles {
        let area = (second - first).perp_dot(third - first);
        if area.abs() < 1.0e-12 {
            continue;
        }
        let low = (first.min(second).min(third) - minimum) / step;
        let high = (first.max(second).max(third) - minimum) / step;
        let range = |low: f64, high: f64| {
            (low.floor().max(0.0) as usize)..((high.ceil() + 1.0).clamp(0.0, side as f64) as usize)
        };
        for row in range(low.y, high.y) {
            for column in range(low.x, high.x) {
                let point = minimum + DVec2::new(column as f64, row as f64) * step;
                let (along_second, along_third) = (
                    (point - first).perp_dot(third - first) / area,
                    (second - first).perp_dot(point - first) / area,
                );
                if along_second < -1.0e-9
                    || along_third < -1.0e-9
                    || along_second + along_third > 1.0 + 1.0e-9
                {
                    continue;
                }
                let depth = depth[2].mul_add(
                    along_third,
                    depth[1].mul_add(along_second, (1.0 - along_second - along_third) * depth[0]),
                );
                let t = ((depth - 0.003) / 0.009).clamp(0.0, 1.0);
                let slot = &mut alpha[column + row * side];
                *slot = slot.max(t * t * 2.0_f64.mul_add(-t, 3.0));
            }
        }
    }
    alpha
}

/// Triangles seen from above, with the depth of water at each corner.
type Drawn = Vec<([DVec2; 3], [f64; 3])>;

/// The lake's sheet over the square `reach` metres around `middle`, meshed
/// in tiles of 64 vertices `spacing` apart, as the app streams them.
fn lake_triangles(
    field: &TerrainField,
    terrain: &TerrainOctree,
    [middle, reach]: [DVec2; 2],
    spacing: f64,
    joined: &std::collections::HashSet<crate::water::WaterCell>,
    owned: &std::collections::HashSet<(i32, i32)>,
) -> Drawn {
    let edge = 64.0 * spacing;
    #[expect(clippy::cast_possible_truncation, reason = "a few tiles")]
    let tiles = |middle: f64, reach: f64| {
        ((middle - reach) / edge).floor() as i32..=((middle + reach) / edge).floor() as i32
    };
    let mut drawn = Vec::new();
    for z in tiles(middle.y, reach.y) {
        for x in tiles(middle.x, reach.x) {
            let tile = WaterTile {
                minimum: [f64::from(x) * edge, f64::from(z) * edge],
                edge,
                cells: 64,
            };
            let shifts = std::collections::BTreeMap::new();
            let Some(sheet) = joined_water_sheet(field, terrain, tile, &shifts, joined, owned)
            else {
                continue;
            };
            for triangle in sheet.indices.chunks(3) {
                let corner = |index: u32| {
                    let [x, _, z] = sheet.vertices[index as usize].map(f64::from);
                    DVec2::new(x + sheet.origin.0.x, z + sheet.origin.0.z)
                };
                drawn.push((
                    [triangle[0], triangle[1], triangle[2]].map(corner),
                    [0, 1, 2].map(|k| f64::from(sheet.depths[triangle[k] as usize])),
                ));
            }
        }
    }
    drawn
}

/// The stored water's surface seen from above.
fn stored_triangles(tiles: &[crate::SurfaceTile]) -> Drawn {
    tiles
        .iter()
        .flat_map(|tile| {
            tile.indices.chunks(3).map(|triangle| {
                let corner = |index: u32| {
                    let [x, _, z] = tile.positions[index as usize].map(f64::from);
                    DVec2::new(x + tile.origin.x, z + tile.origin.z)
                };
                (
                    [triangle[0], triangle[1], triangle[2]].map(corner),
                    [0, 1, 2].map(|k| f64::from(tile.attributes[triangle[k] as usize][0])),
                )
            })
        })
        .collect()
}

#[test]
fn a_channel_from_a_lake_is_drawn_once_where_it_meets_the_lake() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, bank, level) = lake_shore(&field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
    // A channel from beyond the bank out into the lake.
    let beyond = bank + (bank - lake).normalize() * 3.0;
    for step in 0..=60 {
        let point = beyond.lerp(lake, f64::from(step) / 60.0);
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
    for _ in 0..300 {
        water.step(&ground, 0.05);
    }
    let surface = water.surface_tiles(&ground, &std::collections::HashMap::new());
    let owned = surface.owned.iter().copied().collect();
    let joined = water
        .joined_cells(&ground)
        .into_iter()
        .map(|(cell, _)| cell)
        .collect::<std::collections::HashSet<_>>();
    let running = water
        .running_cells()
        .into_iter()
        .map(|view| ((view.cell.x, view.cell.z), view.depth))
        .collect::<std::collections::HashMap<_, _>>();
    // Points of the lake's water, or of running water a few centimetres
    // deep among other water, at least 3 cm deep.
    let middle = bank.lerp(lake, 0.3);
    let (step, side) = (0.05, 161);
    // Points lie off the columns' edges, which two surfaces meeting there
    // both reach.
    let minimum = DVec2::new(middle.x, middle.z) - DVec2::splat(4.0) + DVec2::splat(0.0123);
    let (mut water_points, mut visible) = (Vec::new(), Vec::new());
    for row in 0..side {
        for along in 0..side {
            #[expect(clippy::cast_precision_loss, reason = "a small square of points")]
            let point = minimum + DVec2::new(along as f64, row as f64) * step;
            let deep = ground
                .ground_top(point.x, point.y, level + 0.25, 1.0)
                .is_none_or(|top| top < level - 0.03);
            if !deep {
                continue;
            }
            visible.push(along + row * side);
            let cell =
                crate::water::WaterCell::containing(DVec3::new(point.x, level - 0.02, point.y));
            let lake_water = field.water_surface(point.x, point.y).is_some()
                && (field.is_water(DVec3::new(point.x, level - 0.02, point.y))
                    || joined.contains(&cell));
            let among = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                .iter()
                .all(|(dx, dz)| running.contains_key(&(cell.x + dx, cell.z + dz)));
            let running_water = among
                && running
                    .get(&(cell.x, cell.z))
                    .is_some_and(|&depth| depth > 0.02);
            if lake_water || running_water {
                water_points.push(along + row * side);
            }
        }
    }
    assert!(water_points.len() > 1_000, "the mouth holds little water");
    let on_stored = opacity(&stored_triangles(&surface.tiles), minimum, step, side);
    // At the lake's finest grids, a vertex every metre and every two, every
    // point of water shows water drawn by one surface, and where the ground
    // lies under the water no point is drawn by both.
    for spacing in [1.0, 2.0] {
        let lake = lake_triangles(
            &field,
            &terrain,
            [DVec2::new(middle.x, middle.z), DVec2::splat(4.0)],
            spacing,
            &joined,
            &owned,
        );
        let on_lake = opacity(&lake, minimum, step, side);
        let bare = water_points
            .iter()
            .filter(|&&point| on_lake[point].max(on_stored[point]) < 0.5)
            .count();
        assert!(
            bare * 500 <= water_points.len(),
            "{bare} of {} points of water are bare at {spacing} m",
            water_points.len()
        );
        let twice = visible
            .iter()
            .filter(|&&point| on_lake[point] > 0.05 && on_stored[point] > 0.05)
            .count();
        assert_eq!(twice, 0, "water is drawn twice at {spacing} m");
    }
}

/// Where a lake's own water runs to the end of its reach near spawn: a
/// point of its water, one beyond its reach, and the lake's level.
fn lake_reach_edge(field: &TerrainField) -> (DVec3, DVec3, f64) {
    let spawn = field.safe_spawn().0;
    for ring in 1..120 {
        let radius = f64::from(ring) * 2.0;
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
            if !matches!(surface.body, WaterBody::Lake(_)) {
                continue;
            }
            for direction in 0..8 {
                let angle = f64::from(direction) * std::f64::consts::FRAC_PI_4;
                let along = DVec3::new(angle.cos(), 0.0, angle.sin());
                let (point, beyond) = (
                    DVec3::new(x, level, z),
                    DVec3::new(x, level, z) + along * 2.0,
                );
                // Open water well inside, and a bank over the water beyond.
                let inside = (1..=4).all(|metres| {
                    let inner = point - along * f64::from(metres);
                    [0.02, 0.3]
                        .iter()
                        .all(|&down| field.is_water(DVec3::new(inner.x, level - down, inner.z)))
                });
                let bank = field.water_surface(beyond.x, beyond.z).is_none()
                    && field
                        .topmost_surface(beyond.x, beyond.z)
                        .is_some_and(|ground| ground > level + 0.2);
                if inside && bank {
                    return (point, beyond, level);
                }
            }
        }
    }
    panic!("no lake water at the end of its reach near spawn");
}

#[test]
fn a_lake_shows_all_its_water_where_its_reach_ends_in_a_dug_channel() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, outside, level) = lake_reach_edge(&field);
    let mut terrain = TerrainOctree::default();
    // A channel from well beyond the lake's reach into its water, below the
    // lake's level all the way, so the reach ends in the open channel.
    let beyond = outside + (outside - lake).normalize() * 6.0;
    let steps = 80;
    for step in 0..=steps {
        let point = beyond.lerp(lake, f64::from(step) / f64::from(steps));
        let centre = WorldPosition(DVec3::new(point.x, level - 0.6, point.z));
        terrain.excavate_sphere(&field, centre, 1.2).unwrap();
    }
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let middle = outside.lerp(lake, 0.5);
    let (step, side) = (0.1, 161);
    // Points lie off the columns' edges, where cut squares meet.
    let minimum = DVec2::new(middle.x, middle.z) - DVec2::splat(8.0) + DVec2::splat(0.0123);
    // Points of the lake's water at least 3 cm deep, and of the open channel
    // beyond its reach.
    let (mut water, mut reach_ends) = (Vec::new(), 0);
    for row in 0..side {
        for along in 0..side {
            #[expect(clippy::cast_precision_loss, reason = "a small square of points")]
            let point = minimum + DVec2::new(along as f64, row as f64) * step;
            let deep = ground
                .ground_top(point.x, point.y, level + 0.25, 1.0)
                .is_none_or(|top| top < level - 0.03);
            if field.water_surface(point.x, point.y).is_none() {
                reach_ends += usize::from(deep);
            } else if deep && field.is_water(DVec3::new(point.x, level - 0.02, point.y)) {
                water.push(along + row * side);
            }
        }
    }
    assert!(
        reach_ends > 100,
        "the lake's reach does not end in the channel"
    );
    assert!(
        water.len() > 1_000,
        "the channel holds little of the lake's water"
    );
    for spacing in [1.0, 2.0] {
        let triangles = lake_triangles(
            &field,
            &terrain,
            [DVec2::new(middle.x, middle.z), DVec2::splat(8.0)],
            spacing,
            &std::collections::HashSet::new(),
            &std::collections::HashSet::new(),
        );
        let drawn = opacity(&triangles, minimum, step, side);
        // Every point of the lake's water at least 3 cm deep shows it, save
        // half columns where the water's own edge crosses one: the sheet
        // cuts a square whose corner lies beyond the reach, or over open
        // ground that is not water, into columns, and keeps those holding
        // water.
        let bare = water.iter().filter(|&&point| drawn[point] < 0.5).count();
        assert!(
            bare * 50 <= water.len(),
            "{bare} of {} points of the lake's water are bare at {spacing} m",
            water.len()
        );
    }
}

#[test]
fn a_trench_filling_from_a_lake_shows_water_over_every_column_that_holds_it() {
    let field = TerrainField::new(WorldSeed(42));
    let (lake, bank, level) = lake_shore(&field);
    let mut terrain = TerrainOctree::default();
    let mut bricks = Vec::new();
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
    let solid = |point: DVec3| {
        WorldPosition(point).cell().is_ok_and(|cell| {
            terrain
                .brick(cell.brick())
                .and_then(|brick| brick.sample(cell.local_in_brick()))
                .map_or_else(|| field.density(point), |sample| f64::from(sample.density))
                > 0.0
        })
    };
    // The lake's finest grid, a vertex every metre, over the trench.
    let (low, high) = (beyond.min(lake), beyond.max(lake));
    let cells = 24;
    assert!((high - low).max_element() < f64::from(cells - 8));
    let tile = WaterTile {
        minimum: [(low.x - 4.0).floor(), (low.z - 4.0).floor()],
        edge: f64::from(cells),
        cells,
    };
    let mut water = WaterWorld::new();
    water.terrain_changed(&ground, bricks);
    let (mut checked, mut run) = (0, 0);
    for seconds in [1, 2, 4, 8, 16, 32] {
        while run < seconds * 20 {
            water.step(&ground, 0.05);
            run += 1;
        }
        let joined = water
            .joined_cells(&ground)
            .into_iter()
            .map(|(cell, _)| cell)
            .collect::<std::collections::HashSet<_>>();
        let owned = water
            .surface_tiles(&ground, &std::collections::HashMap::new())
            .owned
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let shifts = water.cycle.shifts(&ground);
        let sheet = joined_water_sheet(&field, &terrain, tile, &shifts, &joined, &owned)
            .expect("the lake shows");
        let drawn = drawn_columns(&mut water, &ground);
        // Every column of the lake, or dug beside it and joined to it, lies
        // under the lake's sheet or the stored water's surface.
        #[expect(clippy::cast_possible_truncation, reason = "a whole number of columns")]
        let columns = (tile.edge / crate::water::WATER_CELL_METRES).round() as i32;
        let first =
            crate::water::WaterCell::containing(DVec3::new(tile.minimum[0], 0.0, tile.minimum[1]));
        for dx in 0..columns {
            for dz in 0..columns {
                let (x, z) = (first.x + dx, first.z + dz);
                let centre = crate::water::WaterCell::new(x, 0, z).centre();
                let probe = DVec3::new(centre.x, level - 0.02, centre.z);
                let lake_water = field.is_water(probe)
                    || joined.contains(&crate::water::WaterCell::containing(probe));
                if !lake_water || solid(probe) {
                    continue;
                }
                checked += 1;
                assert!(
                    covers(&sheet, DVec2::new(centre.x, centre.z)) || drawn.contains(&(x, z)),
                    "after {seconds} s the water in ({x}, {z}) is not drawn"
                );
            }
        }
    }
    assert!(checked > 100, "only {checked} columns checked");
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
        ledger.running_m3 + ledger.pools_m3 + ledger.joined_m3 + ledger.soil_m3
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
                .ground_top(at.x, at.z, at.y + 0.25, 1.5)
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

#[test]
fn a_lake_standing_over_its_seed_level_shows_over_its_shallows() {
    let field = TerrainField::new(WorldSeed(42));
    let terrain = TerrainOctree::default();
    let ground = TerrainWater {
        field: &field,
        edits: &terrain,
    };
    let (lake, _, level) = lake_shore(&field);
    let body = field.water_surface(lake.x, lake.z).unwrap().body;
    let rise = 0.05;
    let shifts = std::collections::BTreeMap::from([(
        body,
        crate::WaterShift {
            drop: -rise,
            flow_scale: 1.0,
        },
    )]);
    let tile = WaterTile {
        minimum: [
            (lake.x / 64.0).floor() * 64.0,
            (lake.z / 64.0).floor() * 64.0,
        ],
        edge: 64.0,
        cells: 64,
    };
    let sheet = crate::water_sheet(&field, &terrain, tile, &shifts).expect("the lake shows");
    let at = |index: u32| {
        let vertex = sheet.vertices[index as usize];
        DVec2::new(
            f64::from(vertex[0]) + tile.minimum[0],
            f64::from(vertex[2]) + tile.minimum[1],
        )
    };
    // The depth the sheet draws at a point, where it covers it.
    let drawn = |point: DVec2| {
        sheet
            .indices
            .chunks(3)
            .filter_map(|triangle| {
                let [a, b, c] = [at(triangle[0]), at(triangle[1]), at(triangle[2])];
                let area = (b - a).perp_dot(c - a);
                let weights = [
                    (c - b).perp_dot(point - b) / area,
                    (a - c).perp_dot(point - c) / area,
                    (b - a).perp_dot(point - a) / area,
                ];
                weights.iter().all(|&weight| weight >= -1.0e-9).then(|| {
                    (0..3)
                        .map(|k| weights[k] * f64::from(sheet.depths[triangle[k] as usize]))
                        .sum::<f64>()
                })
            })
            .reduce(f64::max)
    };
    // Every point of the lake's shallows, old and new, shows water.
    let (mut shallows, mut bare) = (0, 0);
    for i in 0..128 {
        for j in 0..128 {
            let point = DVec2::new(
                tile.minimum[0] + (f64::from(i) + 0.5) * 0.5,
                tile.minimum[1] + (f64::from(j) + 0.5) * 0.5,
            );
            if field
                .water_surface(point.x, point.y)
                .is_none_or(|surface| surface.body != body)
            {
                continue;
            }
            let Some(top) = ground.ground_top(point.x, point.y, level + 0.5, 3.0) else {
                continue;
            };
            if !(0.015..0.3).contains(&(level + rise - top)) {
                continue;
            }
            shallows += 1;
            if drawn(point).is_none_or(|depth| depth < 0.003) {
                bare += 1;
            }
        }
    }
    assert!(shallows > 50, "only {shallows} shallow points");
    assert!(
        bare * 50 < shallows,
        "{bare} of {shallows} points of a lake's shallows show no water"
    );
}
