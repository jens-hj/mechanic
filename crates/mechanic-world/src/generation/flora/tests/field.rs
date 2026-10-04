//! Trees in the generated world.

use std::collections::VecDeque;

use bevy_math::{DVec3, IVec3};

use super::super::super::interval::Interval;
use super::super::super::{Lattice, TerrainField};
use super::super::forest::TreeInstance;
use crate::{TerrainDensityClass, TerrainMaterial, WorldPosition, WorldSeed};

/// Trees standing within `radius` of the spawn.
fn trees_near_spawn(field: &TerrainField, radius: f64) -> Vec<TreeInstance> {
    let spawn = field.safe_spawn().0;
    let world = &field.world;
    let domain = [
        Interval::new(spawn.x - radius, spawn.x + radius),
        Interval::new(spawn.y - 60.0, spawn.y + 60.0),
        Interval::new(spawn.z - radius, spawn.z + radius),
    ];
    let mut found = Vec::new();
    world.forest.instances_near(
        domain,
        &|layer, x, z| world.place_tree(layer, x, z),
        &mut found,
    );
    found.sort_by(|a, b| {
        let distance = |tree: &TreeInstance| tree.origin().with_y(spawn.y).distance(spawn);
        distance(a).total_cmp(&distance(b))
    });
    found.dedup();
    found
}

fn material(field: &TerrainField, point: DVec3) -> Option<TerrainMaterial> {
    let sample = field.sample_position(WorldPosition(point));
    sample.is_solid().then_some(sample.material)
}

#[test]
fn trees_grow_around_the_spawn() {
    let field = TerrainField::new(WorldSeed(42));
    let trees = trees_near_spawn(&field, 60.0);
    assert!(trees.len() >= 5, "{} trees near the spawn", trees.len());
}

#[test]
fn trees_stand_on_the_ground() {
    let field = TerrainField::new(WorldSeed(42));
    for tree in trees_near_spawn(&field, 60.0).iter().take(8) {
        let base = tree.origin();
        let ground = field
            .topmost_surface(base.x, base.z)
            .expect("ground under a tree");
        assert!(
            (ground - base.y).abs() < 0.5,
            "a tree at {base:?} stands {:.2} m off the ground at {ground:.2}",
            base.y - ground
        );
        // Stems stand on a small disc around the base.
        let trunk = (0..64).any(|step| {
            let angle = f64::from(step) * 0.7;
            let reach = f64::from(step) / 64.0;
            let point = base + DVec3::new(angle.cos() * reach, 1.0, angle.sin() * reach);
            material(&field, point) == Some(TerrainMaterial::Wood)
        });
        assert!(trunk, "no trunk a metre above {base:?}");
        assert!(
            field
                .sample_position(WorldPosition(base - DVec3::Y * 0.3))
                .is_solid(),
            "no ground under {base:?}"
        );
    }
}

#[test]
fn canopies_are_foliage_and_roots_are_wood_inside_the_ground() {
    let field = TerrainField::new(WorldSeed(42));
    let mut leaves = 0;
    let mut roots = 0;
    for tree in trees_near_spawn(&field, 60.0).iter().take(8) {
        let base = tree.origin();
        for step in 0..200 {
            let t = f64::from(step) / 200.0;
            let angle = t * 37.0;
            let crown = base
                + DVec3::new(
                    angle.cos() * 2.0 * t,
                    tree.height() * (0.5 + 0.5 * t),
                    angle.sin() * 2.0 * t,
                );
            if material(&field, crown) == Some(TerrainMaterial::Foliage) {
                leaves += 1;
            }
            let below =
                base + DVec3::new(angle.cos() * 1.5 * t, -0.5 - 0.6 * t, angle.sin() * 1.5 * t);
            if material(&field, below) == Some(TerrainMaterial::Wood) {
                roots += 1;
            }
        }
    }
    assert!(leaves > 20, "{leaves} leaf samples in eight crowns");
    assert!(roots > 5, "{roots} root samples under eight trees");
}

#[test]
fn tree_samples_agree_across_lattices_and_points() {
    let field = TerrainField::new(WorldSeed(42));
    let tree = trees_near_spawn(&field, 60.0)[0];
    let base = tree.origin();
    // A lattice through the crown and trunk, straddling brick borders.
    #[expect(clippy::cast_possible_truncation, reason = "cells near the spawn")]
    let cell = |value: f64| (value / crate::TERRAIN_CELL_METERS).floor() as i32;
    let lattice = Lattice {
        origin: IVec3::new(cell(base.x - 1.6), cell(base.y - 0.8), cell(base.z - 1.6)),
        stride: 2,
        dims: [33, 40, 33],
        centred: false,
    };
    let densities = field.density_lattice(&lattice);
    let mut solid = 0;
    for (index, density) in densities.iter().enumerate() {
        let (i, j, k) = (index % 33, (index / 33) % 40, index / (33 * 40));
        let point = DVec3::new(
            lattice.coordinate(0, i),
            lattice.coordinate(1, j),
            lattice.coordinate(2, k),
        );
        let exact = field.density_at_stride(point, lattice.stride);
        assert!(
            (exact - density).abs() < 1.0e-9,
            "lattice {density} and point {exact} differ at {point:?}"
        );
        solid += usize::from(*density > 0.0);
    }
    assert!(solid > 0, "the lattice missed the tree");
}

#[test]
fn no_trees_stand_in_water_or_on_cliffs() {
    let field = TerrainField::new(WorldSeed(42));
    for tree in trees_near_spawn(&field, 120.0) {
        let base = tree.origin();
        if let Some(water) = field.water_surface(base.x, base.z) {
            assert!(
                water.level < base.y,
                "a tree at {base:?} stands in water at {}",
                water.level
            );
        }
        // The ground's own normal at the base: a ray from the sky can stop on
        // carve skins, so the slope is read from the ground's density.
        let world = &field.world;
        let ground = |offset: DVec3| {
            let point = base + DVec3::Y * 0.1 + offset;
            world
                .density_parts(&world.cached_column(point.x, point.z), point)
                .0
        };
        let h = 0.25;
        let gradient = DVec3::new(
            ground(DVec3::X * h) - ground(-DVec3::X * h),
            ground(DVec3::Y * h) - ground(-DVec3::Y * h),
            ground(DVec3::Z * h) - ground(-DVec3::Z * h),
        );
        let up = -gradient.y / gradient.length();
        assert!(
            up > 0.75,
            "a tree at {base:?} stands on ground facing {up:.2} up"
        );
    }
}

#[test]
fn bounds_never_claim_a_tree_box_is_empty_or_solid_wrongly() {
    let field = TerrainField::new(WorldSeed(42));
    let trees = trees_near_spawn(&field, 60.0);
    let mut state = 0x1234_5678_u64;
    let mut unit = || {
        state = super::super::super::scatter::mix(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
        #[expect(clippy::cast_precision_loss, reason = "53 random bits")]
        let value = (state >> 11) as f64 / (1_u64 << 53) as f64;
        value
    };
    for index in 0..1_000 {
        let tree = trees[index % trees.len().min(6)];
        let centre = tree.origin()
            + DVec3::new(
                (unit() - 0.5) * 12.0,
                unit() * tree.height() * 1.1 - 1.0,
                (unit() - 0.5) * 12.0,
            );
        let half = DVec3::splat(0.2 + unit() * 2.0);
        let (minimum, maximum) = (centre - half, centre + half);
        let class = field.classify(minimum, maximum);
        if class == TerrainDensityClass::Mixed {
            continue;
        }
        for i in 0..=4 {
            for j in 0..=4 {
                for k in 0..=4 {
                    let t = DVec3::new(f64::from(i), f64::from(j), f64::from(k)) / 4.0;
                    let density = field.density(minimum + (maximum - minimum) * t);
                    let solid = density > 0.0;
                    assert_eq!(
                        solid,
                        class == TerrainDensityClass::Solid,
                        "box {minimum:?}..{maximum:?} judged {class:?} holds density {density}"
                    );
                }
            }
        }
    }
}

#[test]
fn no_wood_or_crown_floats_at_coarse_levels_of_detail() {
    #[expect(clippy::cast_possible_truncation, reason = "cells near the spawn")]
    let cell = |value: f64| (value / crate::TERRAIN_CELL_METERS).floor() as i32;
    let fields = [42, 7].map(|seed| TerrainField::new(WorldSeed(seed)));
    for (field, tree) in fields.iter().flat_map(|field| {
        trees_near_spawn(field, 60.0)
            .into_iter()
            .take(6)
            .map(move |tree| (field, tree))
    }) {
        let base = tree.origin();
        let reach = tree.height() * 0.7 + 2.0;
        // Grown at stride 4; drawn from octrees beyond.
        for stride in [4, 8, 16, 32, 64] {
            let span =
                |metres: f64| usize::try_from(cell(metres) / stride + 1).expect("positive span");
            let lattice = Lattice {
                origin: IVec3::new(
                    cell(base.x - reach),
                    cell(base.y - 1.5),
                    cell(base.z - reach),
                ),
                stride,
                dims: [
                    span(2.0 * reach),
                    span(tree.height() * 1.3 + 3.0),
                    span(2.0 * reach),
                ],
                centred: false,
            };
            let [nx, ny, nz] = lattice.dims;
            let densities = field.density_lattice(&lattice);
            let index = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
            // Everything joined to the ground or to the box's sides is held up;
            // whatever is left floats.
            let mut reached = vec![false; densities.len()];
            let mut queue = VecDeque::new();
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let edge = j == 0 || i == 0 || k == 0 || i == nx - 1 || k == nz - 1;
                        let at = index(i, j, k);
                        if edge && densities[at] > 0.0 {
                            reached[at] = true;
                            queue.push_back((i, j, k));
                        }
                    }
                }
            }
            while let Some((i, j, k)) = queue.pop_front() {
                let steps = [
                    (i.wrapping_sub(1), j, k),
                    (i + 1, j, k),
                    (i, j.wrapping_sub(1), k),
                    (i, j + 1, k),
                    (i, j, k.wrapping_sub(1)),
                    (i, j, k + 1),
                ];
                for (a, b, c) in steps {
                    if a < nx && b < ny && c < nz {
                        let at = index(a, b, c);
                        if !reached[at] && densities[at] > 0.0 {
                            reached[at] = true;
                            queue.push_back((a, b, c));
                        }
                    }
                }
            }
            let floating = densities
                .iter()
                .zip(&reached)
                .filter(|(density, reached)| **density > 0.0 && !**reached)
                .count();
            assert_eq!(
                floating, 0,
                "{floating} floating samples around the tree at {base:?} at stride {stride}"
            );
        }
    }
}

#[test]
fn trees_appear_in_terrain_meshes() {
    let field = TerrainField::new(WorldSeed(42));
    let tree = trees_near_spawn(&field, 60.0)[0];
    let snapshot = crate::TerrainOctree::default().snapshot();
    let trunk = WorldPosition(tree.origin() + DVec3::Y * 1.2);
    let brick = trunk.cell().expect("inside the world").brick();
    // Grown at the finest levels, drawn from octrees beyond.
    for level in [0, 1, 2, 3, 4, 5] {
        let node = crate::TerrainNodeId::containing(brick, level).expect("inside the world");
        let chunk = crate::mesh_chunk(
            &field,
            &snapshot,
            crate::TerrainMeshRequest {
                node,
                generation: 0,
                transition_mask: crate::TerrainTransitionMask::default(),
            },
        );
        let tree_part = |weights: &&[f32; TerrainMaterial::COUNT]| {
            [TerrainMaterial::Wood, TerrainMaterial::Foliage]
                .iter()
                .any(|part| weights[usize::from(part.code())] > 0.5)
        };
        let count = chunk.material_weights.iter().filter(tree_part).count();
        assert!(
            count > 10,
            "{count} tree vertices at level {level} around a tree"
        );
    }
}

#[test]
fn crowns_and_their_chunk_caps_are_painted_as_trees() {
    let field = TerrainField::new(WorldSeed(42));
    let tree = trees_near_spawn(&field, 60.0)[0];
    let snapshot = crate::TerrainOctree::default().snapshot();
    let trunk = WorldPosition(tree.origin() + DVec3::Y * 1.2);
    let brick = trunk.cell().expect("inside the world").brick();
    let unpainted = crate::SurfaceId::plain(TerrainMaterial::Rock);
    // Distant crowns are metres dense where a chunk's cap cuts them.
    for level in [0, 1, 2, 3, 4, 5] {
        let node = crate::TerrainNodeId::containing(brick, level).expect("inside the world");
        let chunk = crate::mesh_chunk(
            &field,
            &snapshot,
            crate::TerrainMeshRequest {
                node,
                generation: 0,
                transition_mask: crate::TerrainTransitionMask::default(),
            },
        );
        let bare = chunk
            .surfaces
            .iter()
            .filter(|&&surface| surface == unpainted)
            .count();
        assert_eq!(bare, 0, "{bare} unpainted vertices at level {level}");
        // Whatever stands well clear of the ground is a tree. A coarse
        // lattice places the ground itself up to a sample off.
        let palette = field.palette();
        let clear = (f64::from(1_u32 << level) * crate::TERRAIN_CELL_METERS * 2.0).max(1.0);
        let mut ground = 0;
        for (vertex, surface) in chunk.vertices.iter().zip(&chunk.surfaces) {
            let point = chunk.origin.0
                + DVec3::new(
                    f64::from(vertex[0]),
                    f64::from(vertex[1]),
                    f64::from(vertex[2]),
                );
            let top = field.topmost_surface(point.x, point.z).unwrap_or(point.y);
            let material = palette.look(*surface).material;
            if point.y > top + clear
                && !matches!(material, TerrainMaterial::Wood | TerrainMaterial::Foliage)
            {
                ground += 1;
            }
        }
        assert_eq!(
            ground, 0,
            "{ground} ground vertices in the air at level {level}"
        );
    }
}
