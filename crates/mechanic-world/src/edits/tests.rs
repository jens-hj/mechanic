#![expect(
    clippy::float_cmp,
    reason = "exact litres-per-cell accounting is intentional"
)]
use bevy_math::DVec3;

use super::{TerrainEditError, TerrainOctree, decode_brick, encode_brick};
use crate::{BrickCoord, TerrainField, TerrainMaterial, WorldCell, WorldPosition, WorldSeed};

#[test]
fn depth_twenty_seven_root_covers_both_signed_cell_extremes() {
    let minimum = crate::WorldCell::new(i32::MIN, i32::MIN, i32::MIN).brick();
    let maximum = crate::WorldCell::new(i32::MAX, i32::MAX, i32::MAX).brick();
    assert_eq!(
        super::TerrainNodeId::containing(minimum, 27),
        Some(super::TerrainNodeId::ROOT)
    );
    assert_eq!(
        super::TerrainNodeId::containing(maximum, 27),
        Some(super::TerrainNodeId::ROOT)
    );
    assert_eq!(
        super::TerrainNodeId::ROOT.minimum_cell_i64(),
        [i64::from(i32::MIN); 3]
    );
    assert_eq!(
        super::TerrainNodeId::ROOT.maximum_cell_exclusive_i64(),
        [i64::from(i32::MAX) + 1; 3]
    );
}

#[test]
fn untouched_space_allocates_nothing_and_promotion_is_sparse() {
    let field = TerrainField::new(WorldSeed(4));
    let mut edits = TerrainOctree::default();
    assert_eq!(edits.promoted_brick_count(), 0);
    edits.promote(&field, BrickCoord::new(0, 0, 0));
    assert_eq!(edits.promoted_brick_count(), 1);
    assert_eq!(
        edits
            .nodes_between(BrickCoord::new(0, 0, 0), BrickCoord::new(0, 0, 0))
            .count(),
        28
    );
}

#[test]
fn ancestor_bounds_revisions_and_region_traversal_follow_promoted_leaves() {
    let field = TerrainField::new(WorldSeed(44));
    let mut terrain = TerrainOctree::default();
    let surface = field.surface_height(0.0, 0.0);
    let outcome = terrain
        .excavate_sphere(
            &field,
            WorldPosition(DVec3::new(0.0, surface - 0.1, 0.0)),
            0.25,
        )
        .unwrap();
    let leaf = super::TerrainNodeId::leaf(outcome.changed_brick_coordinates()[0]);
    let leaf_summary = terrain.node(leaf).unwrap();
    assert_eq!(leaf_summary.latest_revision, 1);
    assert!(leaf_summary.minimum_density <= leaf_summary.maximum_density);
    let mut ancestor = leaf;
    while let Some(parent) = ancestor.parent() {
        let summary = terrain.node(parent).unwrap();
        assert_eq!(summary.latest_revision, 1);
        assert!(summary.promoted_descendants >= 1);
        assert!(summary.minimum_density <= leaf_summary.minimum_density);
        assert!(summary.maximum_density >= leaf_summary.maximum_density);
        ancestor = parent;
    }
    let traversed = terrain
        .nodes_between(leaf.coordinates, leaf.coordinates)
        .collect::<Vec<_>>();
    assert_eq!(traversed.first().unwrap().id, super::TerrainNodeId::ROOT);
    assert_eq!(traversed.last().unwrap().id, leaf);
}

#[test]
fn octree_range_minimum_matches_promoted_cell_scan() {
    let field = TerrainField::new(WorldSeed(45));
    let mut terrain = TerrainOctree::default();
    terrain
        .excavate_sphere(&field, WorldPosition(DVec3::new(0.8, 3.8, 0.8)), 0.7)
        .unwrap();
    let snapshot = terrain.snapshot();
    let minimum = WorldCell::new(-4, 60, -3);
    let maximum = WorldCell::new(37, 91, 35);
    let brute = snapshot
        .bricks()
        .flat_map(|brick| {
            let brick_minimum = brick.coordinate().minimum_cell();
            (0..crate::BRICK_EDGE_CELLS).flat_map(move |z| {
                (0..crate::BRICK_EDGE_CELLS).flat_map(move |y| {
                    (0..crate::BRICK_EDGE_CELLS).filter_map(move |x| {
                        let cell = WorldCell::new(
                            brick_minimum.x + x,
                            brick_minimum.y + y,
                            brick_minimum.z + z,
                        );
                        (cell.x >= minimum.x
                            && cell.y >= minimum.y
                            && cell.z >= minimum.z
                            && cell.x <= maximum.x
                            && cell.y <= maximum.y
                            && cell.z <= maximum.z)
                            .then(|| brick.sample(cell.local_in_brick()).unwrap().density)
                    })
                })
            })
        })
        .reduce(f32::min);
    assert_eq!(
        snapshot.minimum_promoted_density_between(minimum, maximum),
        brute
    );
}

#[test]
fn snapshot_isolation_is_copy_on_write() {
    let field = TerrainField::new(WorldSeed(55));
    let surface = field.surface_height(0.0, 0.0);
    let centre = WorldPosition(DVec3::new(0.0, surface - 0.1, 0.0));
    let cell = centre.cell().unwrap();
    let mut terrain = TerrainOctree::default();
    let snapshot = terrain.snapshot();
    assert!(snapshot.sample_cell(&field, cell).is_solid());
    terrain.excavate_sphere(&field, centre, 0.25).unwrap();
    assert!(snapshot.sample_cell(&field, cell).is_solid());
    assert!(!terrain.sample_cell(&field, cell).is_solid());
}

#[test]
fn insertion_order_rebuilds_identical_hierarchy() {
    let field = TerrainField::new(WorldSeed(66));
    let coordinates = [
        BrickCoord::new(-12, 3, 7),
        BrickCoord::new(8, -2, 1),
        BrickCoord::new(0, 0, 0),
    ];
    let mut forward = TerrainOctree::default();
    for coordinate in coordinates {
        forward.promote(&field, coordinate);
    }
    let mut reverse = TerrainOctree::default();
    for coordinate in coordinates.into_iter().rev() {
        reverse.promote(&field, coordinate);
    }
    assert_eq!(forward, reverse);
}

#[test]
fn excavating_untouched_air_does_not_promote_empty_bricks() {
    let field = TerrainField::new(WorldSeed(4));
    let mut edits = TerrainOctree::default();
    let centre = WorldPosition(DVec3::new(0.0, field.surface_height(0.0, 0.0) + 10.0, 0.0));
    let outcome = edits
        .excavate_sphere(&field, centre, 0.5)
        .expect("air brush lies inside the world");
    assert_eq!(outcome.total_removed_cells(), 0);
    assert_eq!(edits.promoted_brick_count(), 0);
}

#[test]
fn excavation_is_idempotent_and_accounts_by_material() {
    let field = TerrainField::new(WorldSeed(123));
    let surface = field.surface_height(300.0, 300.0);
    let centre = WorldPosition(DVec3::new(300.0, surface - 1.4, 300.0));
    let mut edits = TerrainOctree::default();
    let first = edits
        .excavate_sphere(&field, centre, 1.5)
        .expect("valid edit");
    assert!(first.total_removed_cells() > 0);
    assert_eq!(
        first.changed_bricks,
        first.changed_brick_coordinates().len()
    );
    assert!(
        first
            .changed_brick_coordinates()
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );
    assert!(first.removed_cells(TerrainMaterial::SurfaceCover) > 0);
    assert!(first.removed_cells(TerrainMaterial::Soil) > 0);
    assert!(first.removed_cells(TerrainMaterial::Rock) > 0);
    assert_eq!(
        first.litres(TerrainMaterial::Soil),
        first.removed_cells(TerrainMaterial::Soil) as f64 * 0.125
    );

    let second = edits
        .excavate_sphere(&field, centre, 1.5)
        .expect("repeat is valid");
    assert_eq!(second.total_removed_cells(), 0);
    assert_eq!(second.changed_bricks, 0);
    assert!(second.changed_brick_coordinates().is_empty());
}

#[test]
fn delta_excavation_matches_two_complete_overlapping_spheres() {
    let field = TerrainField::new(WorldSeed(321));
    let surface = field.surface_height(300.0, 300.0);
    let first_centre = WorldPosition(DVec3::new(300.0, surface - 0.8, 300.0));
    let second_centre = WorldPosition(first_centre.0 + DVec3::new(0.05, 0.0, 0.0));

    let mut complete = TerrainOctree::default();
    complete
        .excavate_sphere(&field, first_centre, 1.0)
        .expect("first complete sphere is valid");
    let complete_outcome = complete
        .excavate_sphere(&field, second_centre, 1.0)
        .expect("second complete sphere is valid");

    let mut delta = TerrainOctree::default();
    delta
        .excavate_sphere(&field, first_centre, 1.0)
        .expect("first delta sphere is valid");
    let delta_outcome = delta
        .excavate_sphere_delta(&field, second_centre, 1.0, Some((first_centre, 1.0)))
        .expect("second delta sphere is valid");

    assert_eq!(delta, complete);
    assert_eq!(delta_outcome.removed_cells, complete_outcome.removed_cells);
}

#[test]
fn addition_assigns_material_without_repainting_existing_solids() {
    let field = TerrainField::new(WorldSeed(456));
    let surface = field.surface_height(300.0, 300.0);
    let centre = WorldPosition(DVec3::new(300.0, surface + 0.3, 300.0));
    let existing_cell = super::cell_containing(DVec3::new(300.0, surface - 0.1, 300.0));
    let mut terrain = TerrainOctree::default();
    let existing = terrain.sample_cell(&field, existing_cell);
    assert!(existing.is_solid());

    let first = terrain
        .add_sphere(&field, centre, 0.5, TerrainMaterial::Rock)
        .expect("addition is valid");
    assert!(first.added_cells(TerrainMaterial::Rock) > 0);
    assert_eq!(first.total_added_cells(), first.total_changed_cells());
    assert_eq!(terrain.sample_cell(&field, existing_cell), existing);

    let added_cell = super::cell_containing(centre.0);
    let added = terrain.sample_cell(&field, added_cell);
    assert!(added.is_solid());
    assert_eq!(added.material, TerrainMaterial::Rock);

    let second = terrain
        .add_sphere(&field, centre, 0.5, TerrainMaterial::Soil)
        .expect("repeat addition is valid");
    assert_eq!(second.total_added_cells(), 0);
    assert_eq!(
        terrain.sample_cell(&field, added_cell).material,
        TerrainMaterial::Rock
    );
}

#[test]
fn delta_addition_matches_two_complete_overlapping_spheres() {
    let field = TerrainField::new(WorldSeed(654));
    let surface = field.surface_height(300.0, 300.0);
    let first_centre = WorldPosition(DVec3::new(300.0, surface + 0.4, 300.0));
    let second_centre = WorldPosition(first_centre.0 + DVec3::new(0.05, 0.0, 0.0));

    let mut complete = TerrainOctree::default();
    complete
        .add_sphere(&field, first_centre, 0.6, TerrainMaterial::Soil)
        .unwrap();
    complete
        .add_sphere(&field, second_centre, 0.6, TerrainMaterial::Soil)
        .unwrap();

    let mut delta = TerrainOctree::default();
    delta
        .add_sphere(&field, first_centre, 0.6, TerrainMaterial::Soil)
        .unwrap();
    delta
        .add_sphere_delta(
            &field,
            second_centre,
            0.6,
            TerrainMaterial::Soil,
            Some((first_centre, 0.6)),
        )
        .unwrap();

    assert_eq!(delta, complete);
}

#[test]
fn boundary_refusal_does_not_promote_bricks() {
    let field = TerrainField::new(WorldSeed(1));
    let mut edits = TerrainOctree::default();
    assert_eq!(
        edits.excavate_sphere(&field, WorldPosition(DVec3::new(7_999.5, 0.0, 0.0)), 1.0),
        Err(TerrainEditError::UnbreakableBoundary)
    );
    assert_eq!(edits.promoted_brick_count(), 0);
}

#[test]
fn edited_brick_rle_round_trips_exactly() {
    let field = TerrainField::new(WorldSeed(8));
    let mut edits = TerrainOctree::default();
    let surface = field.surface_height(0.0, 0.0);
    edits
        .excavate_sphere(&field, WorldPosition(DVec3::new(0.0, surface, 0.0)), 0.25)
        .unwrap();
    let coordinate = edits
        .dirty_leaves()
        .next()
        .expect("the cut changes a brick")
        .coordinates;
    let mut original = edits.brick(coordinate).unwrap().clone();
    original.cells[0].compaction = 173;
    let original = &original;
    let decoded = decode_brick(&encode_brick(original)).expect("payload is valid");
    assert_eq!(&decoded, original);
}

#[test]
fn added_material_rle_round_trips_exactly() {
    let field = TerrainField::new(WorldSeed(9));
    let surface = field.surface_height(0.0, 0.0);
    for material in TerrainMaterial::ALL {
        let mut edits = TerrainOctree::default();
        edits
            .add_sphere(
                &field,
                WorldPosition(DVec3::new(0.0, surface + 0.3, 0.0)),
                0.5,
                material,
            )
            .unwrap();
        assert!(edits.snapshot().bricks().next().is_some(), "{material:?}");
        for original in edits.snapshot().bricks() {
            let decoded = decode_brick(&encode_brick(original)).expect("payload is valid");
            assert_eq!(&decoded, original, "{material:?}");
        }
    }
}
fn soil_fixture(
    material: TerrainMaterial,
) -> (
    TerrainField,
    TerrainOctree,
    crate::SoilPatch,
    crate::WorldCell,
) {
    let field = TerrainField::new(WorldSeed(8));
    let coordinate = crate::BrickCoord::new(0, 100, 0);
    let mut brick = super::TerrainBrick::promote(&field, coordinate);
    for z in 0..32 {
        for y in 0..32 {
            for x in 0..32 {
                brick.cells[super::brick::local_index(bevy_math::IVec3::new(x, y, z)).unwrap()] =
                    crate::TerrainSample {
                        density: if y <= 16 {
                            -super::EMPTY_DENSITY
                        } else {
                            super::EMPTY_DENSITY
                        },
                        material,
                        compaction: 0,
                        looseness: 0,
                    };
            }
        }
    }
    brick.minimum_density = super::EMPTY_DENSITY;
    brick.maximum_density = -super::EMPTY_DENSITY;
    let mut terrain = TerrainOctree::default();
    terrain.insert_brick(brick);
    let cell = crate::WorldCell::new(8, 3216, 8);
    let patch = crate::SoilPatch {
        centre: crate::WorldPosition(cell.centre().0 + DVec3::Y * 0.025),
        normal: DVec3::Y,
        footprint: crate::LoadFootprint::square(DVec3::Y, 0.1),
        pressure_pa: 80_000.0,
        seconds: 0.1,
    };
    (field, terrain, patch, cell)
}

#[test]
fn light_load_on_soil_leaves_the_surface_unchanged() {
    let (field, mut terrain, mut patch, cell) = soil_fixture(TerrainMaterial::Soil);
    let before = terrain.sample_cell(&field, cell);
    patch.pressure_pa = 20_000.0;
    assert_eq!(
        terrain
            .compress_patch(&field, patch)
            .unwrap()
            .total_changed_cells(),
        0
    );
    assert_eq!(terrain.sample_cell(&field, cell), before);
    assert_eq!(terrain.dirty_leaves().count(), 0);
}

#[test]
fn extraction_is_atomic_and_preserves_compressed_material_quantity() {
    let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
    terrain.compress_patch(&field, patch).unwrap();
    let source = crate::ExtractionCell {
        cell,
        sample: terrain.sample_cell(&field, cell),
        throw: DVec3::ZERO,
    };
    let mut clumps = crate::ClumpCollection::default();
    assert!(
        clumps
            .extract(&mut terrain, &field, &[source, source], false)
            .is_none()
    );
    assert_eq!(terrain.sample_cell(&field, cell), source.sample);
    assert!(clumps.bodies.is_empty());
    clumps
        .extract(&mut terrain, &field, &[source], false)
        .unwrap();
    assert!(!terrain.sample_cell(&field, cell).is_solid());
    assert_eq!(
        u64::from(clumps.bodies[&1].quanta),
        source.material_quanta()
    );
    // The cell is gone; what was read of it is stale.
    assert!(
        clumps
            .extract(&mut terrain, &field, &[source], false)
            .is_none()
    );
    assert_eq!(clumps.bodies.len(), 1);
}

// A one-cell load on the fixture's exposed cell that does no work.
fn still_load(centre: WorldPosition) -> crate::BreakagePatch {
    crate::BreakagePatch {
        centre,
        normal: DVec3::Y,
        footprint: crate::LoadFootprint::square(DVec3::Y, 0.025),
        stress_pa: 1e9,
        work_j: 0.0,
        crush_pa: 0.0,
        seconds: 1.0 / 60.0,
        throw: DVec3::ZERO,
    }
}

#[test]
fn resting_weight_never_breaks_ground_however_heavy() {
    let (field, terrain, soil, _) = soil_fixture(TerrainMaterial::Soil);
    let mut damage = crate::BreakageAccumulator::default();
    for _ in 0..600 {
        damage.accumulate(&terrain, &field, still_load(soil.centre));
    }
    assert!(damage.ready(&terrain, &field, 256).is_empty());
}

#[test]
fn soft_ground_driven_sideways_hard_enough_breaks_without_slip() {
    let crushed = |material, crush_pa| {
        let (field, terrain, soil, cell) = soil_fixture(material);
        let mut damage = crate::BreakageAccumulator::default();
        for _ in 0..60 {
            damage.accumulate(
                &terrain,
                &field,
                crate::BreakagePatch {
                    crush_pa,
                    ..still_load(soil.centre)
                },
            );
        }
        damage
            .ready(&terrain, &field, 256)
            .iter()
            .any(|source| source.cell == cell)
    };
    assert!(crushed(TerrainMaterial::Soil, 5.0e6));
    // A machine leaning on a slope pushes sideways well within this.
    assert!(!crushed(TerrainMaterial::Soil, 100_000.0));
    assert!(!crushed(TerrainMaterial::Rock, 1.0e9));
}

#[test]
fn broken_material_leaves_along_the_tool_motion() {
    let (field, mut terrain, soil, _) = soil_fixture(TerrainMaterial::Soil);
    let mut damage = crate::BreakageAccumulator::default();
    damage.accumulate(
        &terrain,
        &field,
        crate::BreakagePatch {
            work_j: 1e6,
            throw: DVec3::X * 3.0,
            ..still_load(soil.centre)
        },
    );
    let ready = damage.ready(&terrain, &field, 256);
    let undisturbed = terrain.clone();
    let mut clumps = crate::ClumpCollection::default();
    clumps.extract(&mut terrain, &field, &ready, false).unwrap();
    let thrown = clumps.bodies[&1].linear_velocity;
    assert!(thrown.x > 1.0, "{thrown:?}");
    assert!(thrown.y > 0.0, "spoil lifts clear of the cut: {thrown:?}");
    assert!(thrown.z.abs() < 1e-9);

    // Past the budget for loose bodies, spoil is laid down where it was cut.
    let mut laid = crate::ClumpCollection::default();
    laid.extract(&mut undisturbed.clone(), &field, &ready, true)
        .unwrap();
    assert_eq!(laid.bodies[&1].linear_velocity, DVec3::ZERO);
    assert!(laid.bodies[&1].can_deposit());

    let terrain = undisturbed;
    let mut damage = crate::BreakageAccumulator::default();
    damage.accumulate(
        &terrain,
        &field,
        crate::BreakagePatch {
            work_j: 1e6,
            throw: DVec3::X * 300.0,
            ..still_load(soil.centre)
        },
    );
    let fast = damage.ready(&terrain, &field, 256)[0].throw;
    assert!(fast.length() < 4.0 + 1e-9, "{fast:?}");
}

#[test]
fn a_knife_edge_compacts_the_cells_along_it_and_not_beside_it() {
    let (field, terrain, soil, cell) = soil_fixture(TerrainMaterial::Soil);
    let edge = crate::SoilPatch {
        footprint: crate::LoadFootprint {
            axis: DVec3::X,
            half_length: 0.125,
            half_width: crate::LoadFootprint::MINIMUM_HALF_EXTENT,
        },
        ..soil
    };
    let cells = terrain.soil_compressions(&field, edge).unwrap();
    assert!(cells.len() >= 4, "{}", cells.len());
    assert!(cells.iter().all(|compression| compression.cell.z == cell.z));
    assert!(cells.iter().any(|compression| compression.cell.x != cell.x));
}

#[test]
fn pressing_ground_packs_it_and_pressing_it_flat_squeezes_it_out_as_spoil() {
    let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
    terrain.compress_patch(&field, patch).unwrap();
    let packed = crate::ExtractionCell {
        cell,
        sample: terrain.sample_cell(&field, cell),
        throw: DVec3::ZERO,
    };
    assert!(packed.sample.is_solid() && packed.sample.compaction > 0);
    // Packed ground is the same material in less room.
    assert_eq!(packed.material_quanta(), u64::from(crate::CELL_QUANTA));

    let mut clumps = crate::ClumpCollection::default();
    let mut pressed_out = Vec::new();
    // A press far beyond what packed soil carries.
    let press = crate::SoilPatch {
        pressure_pa: 1e8,
        ..patch
    };
    for _ in 0..64 {
        let outcome = terrain.compress_patch(&field, press).unwrap();
        pressed_out.extend(outcome.pressed_out);
        if !terrain.sample_cell(&field, cell).is_solid() {
            break;
        }
    }
    assert!(
        !terrain.sample_cell(&field, cell).is_solid(),
        "never pressed flat"
    );
    assert!(pressed_out.contains(&(cell, TerrainMaterial::Soil, crate::CELL_QUANTA)));
    // Pressed on, it sinks further, but it has nothing more to give.
    for _ in 0..16 {
        pressed_out.extend(terrain.compress_patch(&field, press).unwrap().pressed_out);
    }
    let mut cells = pressed_out
        .iter()
        .map(|pressed| pressed.0)
        .collect::<Vec<_>>();
    cells.sort_unstable();
    cells.dedup();
    assert_eq!(
        cells.len(),
        pressed_out.len(),
        "a cell gave up its material twice"
    );
    clumps.heave(&pressed_out);
    let spoil: u64 = clumps
        .bodies
        .values()
        .map(|body| u64::from(body.quanta))
        .sum();
    assert_eq!(
        spoil,
        pressed_out.len() as u64 * u64::from(crate::CELL_QUANTA),
        "every cell pressed flat is owed to the world as spoil"
    );
    assert!(
        clumps
            .bodies
            .values()
            .all(|body| body.is_valid() && body.settled_seconds == 0.0)
    );
}

#[test]
fn scattered_broken_cells_gather_into_one_clod_holding_all_their_material() {
    let field = TerrainField::new(WorldSeed(8));
    let mut terrain = TerrainOctree::default();
    // Cell coordinates divisible by three share a clod block with their neighbours.
    let corner = crate::WorldCell::new(30, 4_200, 30);
    terrain
        .add_sphere(&field, corner.centre(), 0.3, TerrainMaterial::Soil)
        .unwrap();
    // An L of three cells is no cuboid.
    let sources = [(0, 0, 0), (1, 0, 0), (0, 0, 1)].map(|(x, y, z)| {
        let cell = crate::WorldCell::new(corner.x + x, corner.y + y, corner.z + z);
        let sample = terrain.sample_cell(&field, cell);
        assert!(sample.is_solid());
        crate::ExtractionCell {
            cell,
            sample,
            throw: DVec3::ZERO,
        }
    });
    let mut clumps = crate::ClumpCollection::default();
    clumps
        .extract(&mut terrain, &field, &sources, false)
        .unwrap();
    assert_eq!(clumps.bodies.len(), 1);
    let clod = &clumps.bodies[&1];
    assert!(clod.is_valid());
    assert_eq!(
        u64::from(clod.quanta),
        sources
            .iter()
            .map(|source| source.material_quanta())
            .sum::<u64>()
    );
    assert!(
        sources
            .iter()
            .all(|source| !terrain.sample_cell(&field, source.cell).is_solid())
    );
}

#[test]
fn settled_soft_material_deposits_once_while_rock_stays_physical() {
    for material in [
        TerrainMaterial::Sand,
        TerrainMaterial::Soil,
        TerrainMaterial::Rock,
        TerrainMaterial::Iron,
    ] {
        let (field, mut terrain, _, cell) = soil_fixture(material);
        let source = crate::ExtractionCell {
            cell,
            sample: terrain.sample_cell(&field, cell),
            throw: DVec3::ZERO,
        };
        let mut clumps = crate::ClumpCollection::default();
        clumps
            .extract(&mut terrain, &field, &[source], false)
            .unwrap();
        let mut steps = 100;
        assert!(
            clumps
                .settle(&mut terrain, &field, 1, &mut |_| false, &mut steps)
                .is_none()
        );
        clumps
            .bodies
            .get_mut(&1)
            .unwrap()
            .update_settling(true, 1.0);
        let deposited = clumps.settle(&mut terrain, &field, 1, &mut |_| false, &mut steps);
        if crate::BreakageResponse::for_material(material).deposits {
            deposited.unwrap();
            assert!(clumps.bodies.is_empty());
            let sample = terrain.sample_cell(&field, cell);
            assert!(sample.is_solid());
            assert_eq!(sample.material, material);
            assert_eq!(sample.compaction, 0);
        } else {
            assert!(deposited.is_none());
        }
    }
}

#[test]
fn breakage_requires_stress_and_work_at_the_exposed_contact() {
    let (field, terrain, soil, cell) = soil_fixture(TerrainMaterial::Rock);
    let mut damage = crate::BreakageAccumulator::default();
    let mut patch = crate::BreakagePatch {
        centre: soil.centre,
        normal: DVec3::Y,
        footprint: crate::LoadFootprint::square(DVec3::Y, 0.025),
        stress_pa: 1e9,
        work_j: 0.0,
        crush_pa: 0.0,
        seconds: 1.0 / 60.0,
        throw: DVec3::ZERO,
    };
    damage.accumulate(&terrain, &field, patch);
    assert!(damage.ready(&terrain, &field, 256).is_empty());
    patch.stress_pa = 1.0;
    patch.work_j = 1e9;
    damage.accumulate(&terrain, &field, patch);
    assert!(damage.ready(&terrain, &field, 256).is_empty());
    patch.stress_pa = 1e9;
    damage.accumulate(&terrain, &field, patch);
    let ready = damage.ready(&terrain, &field, 256);
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].cell, cell);
    assert!(damage.ready(&terrain, &field, 0).is_empty());
    assert_eq!(damage.ready(&terrain, &field, 1), ready);
    damage.committed(&ready);
    assert!(damage.ready(&terrain, &field, 1).is_empty());
}

#[test]
fn rock_laid_back_as_rubble_breaks_out_far_more_easily_than_bedrock() {
    let (field, mut terrain, soil, cell) = soil_fixture(TerrainMaterial::Rock);
    let mut patch = crate::BreakagePatch {
        centre: soil.centre,
        normal: DVec3::Y,
        footprint: crate::LoadFootprint::square(DVec3::Y, 0.025),
        stress_pa: 200_000.0,
        work_j: 40.0,
        crush_pa: 0.0,
        seconds: 1.0 / 60.0,
        throw: DVec3::ZERO,
    };
    let mut damage = crate::BreakageAccumulator::default();
    damage.accumulate(&terrain, &field, patch);
    assert!(
        damage.ready(&terrain, &field, 256).is_empty(),
        "a spade broke bedrock"
    );
    let above = crate::WorldCell::new(cell.x, cell.y + 1, cell.z);
    let mut steps = 100;
    let (outcome, left) = terrain.lay_spoil(
        &field,
        above,
        TerrainMaterial::Rock,
        crate::CELL_QUANTA,
        &mut |_| false,
        &mut steps,
    );
    assert_eq!(left, 0, "broken rock was not laid");
    // Loose stones take more room than the rock they were.
    assert_eq!(outcome.laid_cells.len(), 2);
    assert_eq!(outcome.laid_cells[0], above);
    patch.centre.0.y += crate::TERRAIN_CELL_METERS;
    damage.accumulate(&terrain, &field, patch);
    let ready = damage.ready(&terrain, &field, 256);
    assert_eq!(ready.len(), 1, "rubble is as hard as bedrock");
    assert_eq!(ready[0].cell, above);
    let held = outcome
        .laid_cells
        .iter()
        .map(|&cell| {
            crate::ExtractionCell {
                cell,
                sample: terrain.sample_cell(&field, cell),
                throw: DVec3::ZERO,
            }
            .material_quanta()
        })
        .sum::<u64>();
    assert_eq!(held, u64::from(crate::CELL_QUANTA), "nothing made or lost");
}

#[test]
fn sustained_overload_sinks_soil_and_then_stops_when_it_compacts() {
    let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
    let before = terrain.sample_cell(&field, cell);
    for _ in 0..100 {
        terrain.compress_patch(&field, patch).unwrap();
    }
    let compacted = terrain.sample_cell(&field, cell);
    assert!(before.density - compacted.density > 0.001);
    for _ in 0..100 {
        terrain.compress_patch(&field, patch).unwrap();
    }
    assert_eq!(terrain.sample_cell(&field, cell), compacted);
}

#[test]
fn rock_does_not_deform_under_any_pressure() {
    for material in [
        TerrainMaterial::Rock,
        TerrainMaterial::Iron,
        TerrainMaterial::Graphite,
    ] {
        let (field, mut terrain, mut patch, cell) = soil_fixture(material);
        let before = terrain.sample_cell(&field, cell);
        patch.pressure_pa = f32::MAX;
        let count = terrain.promoted_brick_count();
        assert_eq!(
            terrain
                .compress_patch(&field, patch)
                .unwrap()
                .changed_bricks,
            0
        );
        assert_eq!(terrain.promoted_brick_count(), count);
        assert_eq!(terrain.sample_cell(&field, cell), before);
        assert_eq!(terrain.dirty_leaves().count(), 0);
    }
}

#[test]
fn a_second_pass_over_compacted_soil_sinks_less_than_the_first() {
    let (field, mut terrain, patch, _) = soil_fixture(TerrainMaterial::Soil);
    let first = terrain.compress_patch(&field, patch).unwrap().sunk_metres;
    let second = terrain.compress_patch(&field, patch).unwrap().sunk_metres;
    assert!(first > second && second > 0.0);
}

#[test]
fn a_fully_collapsed_cell_passes_further_sinking_to_the_cell_below() {
    let (field, mut terrain, mut patch, cell) = soil_fixture(TerrainMaterial::Soil);
    patch.pressure_pa = 1.0e9;
    for _ in 0..5 {
        terrain.compress_patch(&field, patch).unwrap();
    }
    assert!(
        terrain.sample_cell(&field, cell).density
            <= super::EMPTY_DENSITY + super::COMPACTION_STEP_METRES
    );
    terrain.compress_patch(&field, patch).unwrap();
    let below = crate::WorldCell::new(cell.x, cell.y - 1, cell.z);
    assert!(terrain.sample_cell(&field, below).compaction > 0);
}

#[test]
fn compacted_and_loose_brick_rle_round_trips_exactly() {
    let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
    terrain.compress_patch(&field, patch).unwrap();
    let above = WorldCell::new(cell.x + 4, cell.y + 3, cell.z);
    let mut steps = 100;
    let (laid, left) = terrain.lay_spoil(
        &field,
        above,
        TerrainMaterial::Soil,
        crate::CELL_QUANTA * 4,
        &mut |_| false,
        &mut steps,
    );
    assert_eq!(left, 0);
    assert!(
        laid.laid_cells
            .iter()
            .all(|&laid| terrain.sample_cell(&field, laid).looseness > 0)
    );
    let brick = terrain.brick(cell.brick()).unwrap();
    assert_eq!(decode_brick(&encode_brick(brick)).unwrap(), *brick);
    assert_eq!(std::mem::size_of::<crate::TerrainSample>(), 8);
    // Bricks written before cells knew how loose they are do not load.
    let mut old = encode_brick(brick);
    old[4..6].copy_from_slice(&3_u16.to_le_bytes());
    assert_eq!(
        decode_brick(&old),
        Err(super::BrickDecodeError::UnsupportedHeader)
    );
}

#[test]
fn soil_accumulation_defers_edits_and_discards_stale_loads() {
    let (field, mut terrain, mut patch, cell) = soil_fixture(TerrainMaterial::Soil);
    patch.seconds = 1.0 / 60.0;
    let mut pending = crate::SoilAccumulator::default();
    pending.accumulate(&terrain, &field, patch).unwrap();
    assert!(pending.take_ready().is_empty());
    for _ in 0..15 {
        pending.accumulate(&terrain, &field, patch).unwrap();
    }
    let ready = pending.take_ready();
    assert!(!ready.is_empty());
    assert_eq!(terrain.sample_cell(&field, cell).compaction, 0);
    assert!(terrain.compress_cells(&field, &ready).changed_bricks > 0);
    assert_eq!(terrain.compress_cells(&field, &ready).changed_bricks, 0);
}

#[test]
fn invalid_soil_pressure_does_not_promote_bricks() {
    let field = TerrainField::new(WorldSeed(8));
    let mut terrain = TerrainOctree::default();
    let patch = crate::SoilPatch {
        centre: WorldPosition(DVec3::ZERO),
        normal: DVec3::Y,
        footprint: crate::LoadFootprint::square(DVec3::Y, 0.1),
        pressure_pa: f32::NAN,
        seconds: 0.1,
    };
    assert_eq!(
        terrain.compress_patch(&field, patch),
        Err(TerrainEditError::InvalidSoilPatch)
    );
    assert_eq!(terrain.promoted_brick_count(), 0);
}
#[test]
fn compression_lowers_the_meshed_surface_and_survives_reload() {
    let (field, mut terrain, patch, cell) = soil_fixture(TerrainMaterial::Soil);
    let height = |terrain: &TerrainOctree| {
        crate::mesh_chunk(
            &field,
            &terrain.snapshot(),
            crate::TerrainMeshRequest {
                node: super::TerrainNodeId::leaf(cell.brick()),
                generation: 1,
                transition_mask: crate::TerrainTransitionMask::NONE,
            },
        )
        .raycast(
            crate::WorldPosition(cell.centre().0 + DVec3::Y * 0.2),
            -DVec3::Y,
            0.5,
        )
        .unwrap()
        .position
        .0
        .y
    };
    let before = height(&terrain);
    for _ in 0..100 {
        terrain.compress_patch(&field, patch).unwrap();
    }
    let after = height(&terrain);
    assert!(before - after > 0.0005, "{before} -> {after}");
    let mut loaded = TerrainOctree::default();
    loaded.insert_saved_brick(
        decode_brick(&encode_brick(terrain.brick(cell.brick()).unwrap())).unwrap(),
    );
    assert!((height(&loaded) - after).abs() < 1.0e-6);
}
#[test]
fn a_small_soil_load_moves_procedural_surface_continuously_downward() {
    let field = TerrainField::new(WorldSeed(91));
    for x in [0.0, 0.13, 0.27, 0.41] {
        let mut terrain = TerrainOctree::default();
        let centre = WorldPosition(DVec3::new(x, field.surface_height(x, 0.0), 0.0));
        let node = super::TerrainNodeId::leaf(centre.cell().unwrap().brick());
        let height = |terrain: &TerrainOctree| {
            crate::mesh_chunk(
                &field,
                &terrain.snapshot(),
                crate::TerrainMeshRequest {
                    node,
                    generation: 1,
                    transition_mask: crate::TerrainTransitionMask::NONE,
                },
            )
            .raycast(WorldPosition(centre.0 + DVec3::Y * 0.2), -DVec3::Y, 0.5)
            .unwrap()
            .position
            .0
            .y
        };
        let before = height(&terrain);
        let outcome = terrain
            .compress_patch(
                &field,
                crate::SoilPatch {
                    centre,
                    normal: DVec3::Y,
                    footprint: crate::LoadFootprint::square(DVec3::Y, 0.15),
                    pressure_pa: 30_000.0,
                    seconds: 1.0 / 60.0,
                },
            )
            .unwrap();
        assert!(outcome.total_changed_cells() > 0);
        let displacement = before - height(&terrain);
        assert!(
            (0.00001..0.0005).contains(&displacement),
            "x={x}: moved {displacement} m"
        );
    }
}
