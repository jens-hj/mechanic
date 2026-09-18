#![allow(clippy::cast_possible_truncation, clippy::float_cmp)]
use bevy_math::{DVec3, Vec3};

use super::{PreparedTerrainRegion, TerrainIndexGroups, TerrainMeshRequest, mesh_chunk};
use crate::{
    BRICK_EDGE_CELLS, BrickCoord, TERRAIN_CELL_METERS, TerrainFace, TerrainField, TerrainMaterial,
    TerrainNodeId, TerrainOctree, TerrainSample, TerrainTransitionMask, WorldCell, WorldPosition,
    WorldSeed,
};

fn surface_request(brick_x: i32) -> TerrainMeshRequest {
    TerrainMeshRequest {
        node: TerrainNodeId::leaf(BrickCoord::new(brick_x, 2, -1)),
        generation: 4,
        transition_mask: TerrainTransitionMask::NONE,
    }
}

fn final_indices(chunk: &super::TerrainMeshChunk) -> Vec<u32> {
    chunk.index_groups.final_indices(chunk.transition_mask)
}

#[test]
fn authored_addition_uses_solid_material_without_changing_procedural_cover() {
    let dirt = TerrainSample {
        compaction: 0,
        density: 0.05,
        material: TerrainMaterial::Soil,
    };
    let procedural_air = TerrainSample {
        compaction: 0,
        density: -0.05,
        material: TerrainMaterial::SurfaceCover,
    };
    let lattice = |sample, authored_material| super::LatticePoint {
        sample,
        normal: Vec3::Y,
        authored_material,
    };
    assert_eq!(
        super::crossing_material(lattice(dirt, None), lattice(procedural_air, None)),
        TerrainMaterial::SurfaceCover
    );
    assert_eq!(
        super::crossing_material(
            lattice(dirt, Some(TerrainMaterial::Soil)),
            lattice(procedural_air, None),
        ),
        TerrainMaterial::Soil
    );
    assert_eq!(
        super::crossing_material(
            lattice(procedural_air, None),
            lattice(dirt, Some(TerrainMaterial::Soil)),
        ),
        TerrainMaterial::Soil
    );
}

#[test]
fn isolated_authored_addition_mesh_keeps_its_selected_material() {
    let field = TerrainField::new(WorldSeed(92));
    let surface = field.surface_height(300.0, 300.0);
    let mut terrain = TerrainOctree::default();
    terrain
        .add_sphere(
            &field,
            WorldPosition(DVec3::new(300.0, surface + 4.0, 300.0)),
            0.5,
            TerrainMaterial::Soil,
        )
        .unwrap();
    let coordinates = terrain
        .dirty_leaves()
        .map(|node| node.coordinates)
        .collect::<Vec<_>>();
    let snapshot = terrain.snapshot();
    let mut vertex_count = 0;
    for coordinate in coordinates {
        let chunk = mesh_chunk(
            &field,
            &snapshot,
            TerrainMeshRequest {
                node: TerrainNodeId::leaf(coordinate),
                generation: 1,
                transition_mask: TerrainTransitionMask::NONE,
            },
        );
        vertex_count += chunk.vertices.len();
        for weights in chunk.material_weights {
            let mut expected = [0.0; TerrainMaterial::COUNT];
            expected[TerrainMaterial::Soil.code() as usize] = 1.0;
            assert_eq!(weights, expected);
        }
    }
    assert!(vertex_count > 0);
}

#[test]
fn prepared_region_matches_octree_range_queries_across_signed_bricks() {
    let field = TerrainField::new(WorldSeed(91));
    let mut terrain = TerrainOctree::default();
    for coordinate in [
        BrickCoord::new(-2, 1, -1),
        BrickCoord::new(-1, 1, -1),
        BrickCoord::new(0, 1, 0),
        BrickCoord::new(20, 1, 20),
    ] {
        terrain.promote(&field, coordinate);
    }
    let snapshot = terrain.snapshot();
    let minimum = WorldCell::new(-64, 32, -32);
    let maximum = WorldCell::new(31, 63, 31);
    let prepared = PreparedTerrainRegion::between(&snapshot, minimum, maximum);
    assert_eq!(prepared.promoted_brick_count(), 3);
    for (query_minimum, query_maximum) in [
        (minimum, maximum),
        (WorldCell::new(-33, 40, -2), WorldCell::new(-30, 45, 2)),
        (WorldCell::new(-1, 32, -1), WorldCell::new(1, 34, 1)),
        (WorldCell::new(16, 40, 16), WorldCell::new(20, 44, 20)),
    ] {
        assert_eq!(
            prepared.minimum_promoted_density_between(query_minimum, query_maximum),
            snapshot.minimum_promoted_density_between(query_minimum, query_maximum),
        );
    }
}

#[test]
fn index_counts_match_combined_vectors() {
    let groups = TerrainIndexGroups {
        regular: vec![0, 1, 2],
        transitions: std::array::from_fn(|face| vec![face as u32; face * 3]),
        caps: std::array::from_fn(|face| vec![face as u32; (face + 1) * 3]),
    };
    let transitions = TerrainTransitionMask::from_bits(0b10_0101);
    let ready = TerrainTransitionMask::from_bits(0b11_0010);
    assert_eq!(
        groups.final_index_count(transitions),
        groups.final_indices(transitions).len(),
    );
    assert_eq!(
        groups.sealed_index_count(transitions, ready),
        groups.sealed_indices(transitions, ready).len(),
    );
}

#[test]
fn generated_vertices_stay_in_owning_bounds_and_normals_are_finite() {
    let field = TerrainField::new(WorldSeed(2));
    let terrain = TerrainOctree::default().snapshot();
    let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
    let indices = final_indices(&chunk);
    assert!(!indices.is_empty());
    assert_eq!(chunk.vertex_cache.vertices.capacity(), 0);
    assert_eq!(chunk.vertices.capacity(), chunk.vertices.len());
    assert_eq!(
        chunk.index_groups.regular.capacity(),
        chunk.index_groups.regular.len()
    );
    assert!(chunk.vertices.len() < indices.len());
    let unique = chunk
        .vertices
        .iter()
        .zip(&chunk.normals)
        .zip(&chunk.material_weights)
        .map(|((position, normal), weights)| {
            (
                position.map(f32::to_bits),
                normal.map(f32::to_bits),
                weights.map(f32::to_bits),
            )
        })
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), chunk.vertices.len());
    for (vertex, normal) in chunk.vertices.iter().zip(&chunk.normals) {
        let global = chunk.origin.0 + DVec3::from_array(vertex.map(f64::from));
        assert!(chunk.bounds.contains(WorldPosition(global)));
        assert!(normal.iter().all(|component| component.is_finite()));
    }
}

#[test]
fn promoted_meshing_keeps_the_analytic_surface_height_and_outward_winding() {
    let field = TerrainField::new(WorldSeed(2));
    let terrain = TerrainOctree::default().snapshot();
    let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
    let indices = final_indices(&chunk);
    assert!(!indices.is_empty());
    let regular_vertices = chunk
        .index_groups
        .regular
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    for index in regular_vertices {
        let vertex = chunk.vertices[index as usize];
        assert!((vertex[1] + chunk.origin.0.y as f32 - 4.0).abs() < 1.0e-5);
    }
    for triangle in indices.chunks_exact(3) {
        let first = Vec3::from_array(chunk.vertices[triangle[0] as usize]);
        let second = Vec3::from_array(chunk.vertices[triangle[1] as usize]);
        let third = Vec3::from_array(chunk.vertices[triangle[2] as usize]);
        let geometric = (second - first).cross(third - first);
        let smooth = Vec3::from_array(chunk.normals[triangle[0] as usize])
            + Vec3::from_array(chunk.normals[triangle[1] as usize])
            + Vec3::from_array(chunk.normals[triangle[2] as usize]);
        assert!(
            geometric.dot(smooth) > 0.0,
            "terrain triangle faces inward: {}",
            geometric.dot(smooth)
        );
    }
}

#[test]
fn adjacent_equal_lod_boundaries_are_byte_identical() {
    let field = TerrainField::new(WorldSeed(77));
    let edits = TerrainOctree::default().snapshot();
    let left = mesh_chunk(&field, &edits, surface_request(-1));
    let right = mesh_chunk(&field, &edits, surface_request(0));
    let seam_x = 0.0_f32;
    let mut left_seam = left
        .vertices
        .iter()
        .map(|vertex| {
            [
                vertex[0] + left.origin.0.x as f32,
                vertex[1] + left.origin.0.y as f32,
                vertex[2] + left.origin.0.z as f32,
            ]
        })
        .filter(|vertex| vertex[0] == seam_x)
        .collect::<Vec<_>>();
    let mut right_seam = right
        .vertices
        .iter()
        .map(|vertex| {
            [
                vertex[0] + right.origin.0.x as f32,
                vertex[1] + right.origin.0.y as f32,
                vertex[2] + right.origin.0.z as f32,
            ]
        })
        .filter(|vertex| vertex[0] == seam_x)
        .collect::<Vec<_>>();
    left_seam.sort_by_key(|vertex| (vertex[1].to_bits(), vertex[2].to_bits()));
    right_seam.sort_by_key(|vertex| (vertex[1].to_bits(), vertex[2].to_bits()));
    left_seam.dedup();
    right_seam.dedup();
    assert_eq!(left_seam, right_seam);
}

#[test]
fn untouched_surface_remains_smooth_and_covered_at_every_lod() {
    let field = TerrainField::new(WorldSeed(77));
    let edits = TerrainOctree::default().snapshot();
    let probe = WorldPosition(DVec3::new(500.0, field.surface_height(500.0, 500.0), 500.0));
    let brick = probe.cell().expect("probe is inside cell space").brick();
    for level in 0..=5 {
        let node = TerrainNodeId::containing(brick, level).expect("streamed LOD exists");
        let chunk = mesh_chunk(
            &field,
            &edits,
            TerrainMeshRequest {
                node,
                generation: 1,
                transition_mask: TerrainTransitionMask::NONE,
            },
        );
        let regular_vertices = chunk
            .index_groups
            .regular
            .iter()
            .map(|&index| index as usize)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(!regular_vertices.is_empty(), "LOD {level} has no surface");
        for index in regular_vertices {
            let local = DVec3::from_array(chunk.vertices[index].map(f64::from));
            let global = chunk.origin.0 + local;
            let expected_height = field.surface_height(global.x, global.z);
            assert!(
                (global.y - expected_height).abs() < 0.1,
                "LOD {level} surface error at {global:?}: expected {expected_height}"
            );
            assert_eq!(
                chunk.material_weights[index],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "LOD {level} exposed a subsurface material"
            );
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)] // The fixture assembles both sides of one complete LOD face.
fn transition_surface_has_no_open_interior_edges() {
    type Point = [i64; 3];
    type Edge = (Point, Point);

    fn point(chunk: &super::TerrainMeshChunk, index: u32) -> Point {
        let local = DVec3::from_array(chunk.vertices[index as usize].map(f64::from));
        (chunk.origin.0 + local)
            .to_array()
            .map(|coordinate| (coordinate * 1_000.0).round() as i64)
    }

    fn add_face_edges(
        counts: &mut std::collections::BTreeMap<Edge, usize>,
        chunk: &super::TerrainMeshChunk,
        indices: &[u32],
        minimum_x: i64,
        seam_x: i64,
    ) {
        for triangle in indices.chunks_exact(3) {
            for edge in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                let first = point(chunk, edge[0]);
                let second = point(chunk, edge[1]);
                if first == second {
                    continue;
                }
                if first[0] < minimum_x
                    || second[0] < minimum_x
                    || first[0] > seam_x + 1
                    || second[0] > seam_x + 1
                {
                    continue;
                }
                let edge = if first <= second {
                    (first, second)
                } else {
                    (second, first)
                };
                *counts.entry(edge).or_default() += 1;
            }
        }
    }

    let field = TerrainField::new(WorldSeed(77));
    let edits = TerrainOctree::default().snapshot();
    let seam_brick_x = 320;
    let seam_brick_z = 320;
    let surface_brick = WorldPosition(DVec3::new(512.0, field.surface_height(512.0, 512.0), 512.0))
        .cell()
        .expect("probe is inside cell space")
        .brick();
    let coarse_node = TerrainNodeId::containing(
        BrickCoord::new(seam_brick_x, surface_brick.y, seam_brick_z),
        5,
    )
    .expect("streamed LOD exists");
    let coarse = mesh_chunk(
        &field,
        &edits,
        TerrainMeshRequest {
            node: coarse_node,
            generation: 1,
            transition_mask: TerrainTransitionMask::NONE,
        },
    );
    let transition_mask = TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8);
    let fine = [0, 16]
        .into_iter()
        .flat_map(|y| [0, 16].map(move |z| (y, z)))
        .map(|(y, z)| {
            mesh_chunk(
                &field,
                &edits,
                TerrainMeshRequest {
                    node: TerrainNodeId {
                        coordinates: BrickCoord::new(
                            coarse_node.coordinates.x - 16,
                            coarse_node.coordinates.y + y,
                            coarse_node.coordinates.z + z,
                        ),
                        level: 4,
                    },
                    generation: 1,
                    transition_mask,
                },
            )
        })
        .collect::<Vec<_>>();

    let mut counts = std::collections::BTreeMap::new();
    let seam_x = i64::from(seam_brick_x) * 1_600;
    let transition_band_minimum = seam_x - 201;
    add_face_edges(
        &mut counts,
        &coarse,
        &coarse.index_groups.regular,
        transition_band_minimum,
        seam_x,
    );
    for chunk in &fine {
        add_face_edges(
            &mut counts,
            chunk,
            &chunk.index_groups.regular,
            transition_band_minimum,
            seam_x,
        );
        add_face_edges(
            &mut counts,
            chunk,
            &chunk.index_groups.transitions[TerrainFace::PositiveX.index()],
            transition_band_minimum,
            seam_x,
        );
    }
    let minimum_z = i64::from(coarse_node.coordinates.z) * 1_600;
    let maximum_z = minimum_z + 51_200;
    let unmatched = counts
        .iter()
        .filter(|((first, second), count)| {
            **count % 2 != 0
                && first[2] != minimum_z
                && second[2] != minimum_z
                && first[2] != maximum_z
                && second[2] != maximum_z
        })
        .collect::<Vec<_>>();
    assert!(!counts.is_empty());
    assert!(unmatched.is_empty(), "unmatched seam edges: {unmatched:?}");
}

#[test]
fn transition_surface_occupies_an_inset_band() {
    let field = TerrainField::new(WorldSeed(77));
    let edits = TerrainOctree::default().snapshot();
    let seam_brick_x = 320;
    let seam_brick_z = 320;
    let surface_brick = WorldPosition(DVec3::new(512.0, field.surface_height(512.0, 512.0), 512.0))
        .cell()
        .expect("probe is inside cell space")
        .brick();
    let coarse_node = TerrainNodeId::containing(
        BrickCoord::new(seam_brick_x, surface_brick.y, seam_brick_z),
        5,
    )
    .expect("streamed LOD exists");
    let transition_mask = TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8);
    let mut maximum_inset = 0.0_f64;
    let mut has_coarse_side_vertex = false;

    for (y, z) in [0, 16]
        .into_iter()
        .flat_map(|y| [0, 16].map(move |z| (y, z)))
    {
        let chunk = mesh_chunk(
            &field,
            &edits,
            TerrainMeshRequest {
                node: TerrainNodeId {
                    coordinates: BrickCoord::new(
                        coarse_node.coordinates.x - 16,
                        coarse_node.coordinates.y + y,
                        coarse_node.coordinates.z + z,
                    ),
                    level: 4,
                },
                generation: 1,
                transition_mask,
            },
        );
        let face_x = chunk.origin.0.x + f64::from(BRICK_EDGE_CELLS) * chunk.sample_spacing_metres;
        for &index in &chunk.index_groups.transitions[TerrainFace::PositiveX.index()] {
            let position =
                chunk.origin.0 + DVec3::from_array(chunk.vertices[index as usize].map(f64::from));
            let inset = face_x - position.x;
            assert!(
                inset >= -1.0e-5,
                "transition vertex escaped the fine chunk: {inset}"
            );
            maximum_inset = maximum_inset.max(inset);
            has_coarse_side_vertex |= inset.abs() <= 1.0e-5;
        }
    }

    assert!(
        has_coarse_side_vertex,
        "transition lost its coarse-side edge"
    );
    assert!(
        maximum_inset > 0.1 * f64::from(1_i32 << 4) * TERRAIN_CELL_METERS,
        "transition collapsed onto the chunk face: {maximum_inset}"
    );
}

#[test]
#[allow(clippy::too_many_lines)] // The fixture compares both complete sides of one edited LOD seam.
fn edited_transition_coarse_edge_matches_coarse_regular_surface() {
    type Point = [i64; 3];
    type Edge = (Point, Point);

    fn point(chunk: &super::TerrainMeshChunk, index: u32) -> Point {
        let local = DVec3::from_array(chunk.vertices[index as usize].map(f64::from));
        (chunk.origin.0 + local)
            .to_array()
            .map(|coordinate| (coordinate * 1_000.0).round() as i64)
    }

    fn boundary_edges<'a>(
        groups: impl IntoIterator<Item = (&'a super::TerrainMeshChunk, &'a [u32])>,
        seam_x: i64,
    ) -> std::collections::BTreeSet<Edge> {
        let mut counts = std::collections::BTreeMap::<Edge, usize>::new();
        for (chunk, indices) in groups {
            for triangle in indices.chunks_exact(3) {
                for pair in [
                    [triangle[0], triangle[1]],
                    [triangle[1], triangle[2]],
                    [triangle[2], triangle[0]],
                ] {
                    let first = point(chunk, pair[0]);
                    let second = point(chunk, pair[1]);
                    if first[0] != seam_x || second[0] != seam_x || first == second {
                        continue;
                    }
                    let edge = if first <= second {
                        (first, second)
                    } else {
                        (second, first)
                    };
                    *counts.entry(edge).or_default() += 1;
                }
            }
        }
        counts
            .into_iter()
            .filter_map(|(edge, count)| (count % 2 != 0).then_some(edge))
            .collect()
    }

    let field = TerrainField::new(WorldSeed(2_255_932_754_758_176_049));
    let surface = field.surface_height(0.3, 0.8);
    let mut edits = TerrainOctree::default();
    edits
        .excavate_sphere(
            &field,
            WorldPosition(DVec3::new(0.3, surface - 0.2, 0.8)),
            0.65,
        )
        .expect("edit is inside the world");
    let snapshot = edits.snapshot();
    let surface_brick = WorldPosition(DVec3::new(0.0, surface, 0.8))
        .cell()
        .expect("surface is in cell space")
        .brick();
    let coarse_node =
        TerrainNodeId::containing(BrickCoord::new(0, surface_brick.y, surface_brick.z), 1)
            .expect("coarse node");
    let transition_mask = TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8);
    let coarse = mesh_chunk(
        &field,
        &snapshot,
        TerrainMeshRequest {
            node: coarse_node,
            generation: 1,
            transition_mask: TerrainTransitionMask::NONE,
        },
    );
    let fine = [0, 1]
        .into_iter()
        .flat_map(|y| [0, 1].map(move |z| (y, z)))
        .map(|(y, z)| {
            mesh_chunk(
                &field,
                &snapshot,
                TerrainMeshRequest {
                    node: TerrainNodeId::leaf(BrickCoord::new(
                        coarse_node.coordinates.x - 1,
                        coarse_node.coordinates.y + y,
                        coarse_node.coordinates.z + z,
                    )),
                    generation: 1,
                    transition_mask,
                },
            )
        })
        .collect::<Vec<_>>();
    let seam_x = (coarse.origin.0.x * 1_000.0).round() as i64;
    let mut coarse_edges =
        boundary_edges([(&coarse, coarse.index_groups.regular.as_slice())], seam_x);
    let mut transition_edges = boundary_edges(
        fine.iter().map(|chunk| {
            (
                chunk,
                chunk.index_groups.transitions[TerrainFace::PositiveX.index()].as_slice(),
            )
        }),
        seam_x,
    );
    let minimum = coarse
        .origin
        .0
        .to_array()
        .map(|value| (value * 1_000.0).round() as i64);
    let maximum = coarse
        .bounds
        .maximum
        .0
        .to_array()
        .map(|value| (value * 1_000.0).round() as i64);
    let on_perimeter = |edge: &Edge| {
        [1, 2].into_iter().any(|axis| {
            (edge.0[axis] == minimum[axis] && edge.1[axis] == minimum[axis])
                || (edge.0[axis] == maximum[axis] && edge.1[axis] == maximum[axis])
        })
    };
    coarse_edges.retain(|edge| !on_perimeter(edge));
    transition_edges.retain(|edge| !on_perimeter(edge));

    assert!(
        !coarse_edges.is_empty(),
        "fixture missed the edited surface"
    );
    assert_eq!(transition_edges, coarse_edges);
}

#[test]
fn official_transition_vertices_only_use_crossing_edges() {
    let row_major_point = |point: usize| match point {
        0..=8 => point,
        9 => 0,
        10 => 2,
        11 => 6,
        12 => 8,
        _ => unreachable!("transition point is 0 through C"),
    };
    for row_major_case in 0_u16..512 {
        let table_case = [0, 1, 2, 5, 8, 7, 6, 3, 4]
            .into_iter()
            .enumerate()
            .fold(0_u16, |case, (bit, point)| {
                case | (((row_major_case >> point) & 1) << bit)
            });
        let class = super::TRANSITION_CELL_CLASS[usize::from(table_case)] & 0x7f;
        let vertex_count =
            usize::from(super::TRANSITION_CELL_DATA[usize::from(class)].geometry_counts >> 4);
        for &data in &super::TRANSITION_VERTEX_DATA[usize::from(table_case)][..vertex_count] {
            let edge = data & 0xff;
            let first = row_major_point(usize::from((edge >> 4) as u8));
            let second = row_major_point(usize::from((edge & 0x0f) as u8));
            assert_ne!(
                (row_major_case >> first) & 1,
                (row_major_case >> second) & 1,
                "case {row_major_case:#05x} uses non-crossing edge {edge:#04x}"
            );
        }
    }
}

#[test]
fn terrain_mesh_raycast_hits_the_surface() {
    let field = TerrainField::new(WorldSeed(9));
    let terrain = TerrainOctree::default().snapshot();
    let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
    let hit = chunk
        .raycast(
            WorldPosition(DVec3::new(0.0, 10.0, 0.0)),
            DVec3::NEG_Y,
            20.0,
        )
        .expect("downward ray meets terrain");
    assert!(hit.normal.is_finite());
    assert_eq!(hit.chunk_generation, 4);
}

#[test]
fn excavated_surface_triangles_keep_outward_winding() {
    let field = TerrainField::new(WorldSeed(33));
    let surface = field.surface_height(0.0, 0.0);
    let centre = WorldPosition(DVec3::new(0.0, surface - 0.3, 0.0));
    let mut edits = TerrainOctree::default();
    let outcome = edits
        .excavate_sphere(&field, centre, 0.75)
        .expect("excavation is valid");
    let snapshot = edits.snapshot();
    for coordinate in outcome.changed_brick_coordinates() {
        let chunk = mesh_chunk(
            &field,
            &snapshot,
            TerrainMeshRequest {
                node: TerrainNodeId::leaf(*coordinate),
                generation: 1,
                transition_mask: TerrainTransitionMask::NONE,
            },
        );
        let indices = final_indices(&chunk);
        for triangle in indices.chunks_exact(3) {
            let first = Vec3::from_array(chunk.vertices[triangle[0] as usize]);
            let second = Vec3::from_array(chunk.vertices[triangle[1] as usize]);
            let third = Vec3::from_array(chunk.vertices[triangle[2] as usize]);
            let geometric = (second - first).cross(third - first);
            let smooth = triangle
                .iter()
                .map(|&index| Vec3::from_array(chunk.normals[index as usize]))
                .sum::<Vec3>();
            let alignment = geometric.dot(smooth);
            assert!(
                alignment >= -1.0e-12,
                "excavated triangle faces inward by {alignment}"
            );
        }
    }
}

#[test]
fn official_regular_and_transition_tables_cover_every_case() {
    for case in 0..256 {
        let class = super::REGULAR_CELL_CLASS[case];
        let cell = super::REGULAR_CELL_DATA[usize::from(class)];
        let vertices = usize::from(cell.geometry_counts >> 4);
        let triangles = usize::from(cell.geometry_counts & 0x0f);
        assert!(
            cell.vertex_index[..triangles * 3]
                .iter()
                .all(|&index| usize::from(index) < vertices)
        );
    }
    for case in 0..512 {
        let class = super::TRANSITION_CELL_CLASS[case] & 0x7f;
        let cell = super::TRANSITION_CELL_DATA[usize::from(class)];
        let vertices = usize::from(cell.geometry_counts >> 4);
        let triangles = usize::from(cell.geometry_counts & 0x0f);
        assert!(
            cell.vertex_index[..triangles * 3]
                .iter()
                .all(|&index| usize::from(index) < vertices)
        );
    }
}

#[test]
fn cavity_generates_transitions_and_caps_on_all_six_faces() {
    let field = TerrainField::new(WorldSeed(90));
    let mut terrain = TerrainOctree::default();
    terrain
        .excavate_sphere(&field, WorldPosition(DVec3::new(0.8, 2.4, 0.8)), 1.0)
        .unwrap();
    let snapshot = terrain.snapshot();
    for face in crate::TerrainFace::ALL {
        let chunk = mesh_chunk(
            &field,
            &snapshot,
            TerrainMeshRequest {
                node: TerrainNodeId::leaf(BrickCoord::new(0, 1, 0)),
                generation: 1,
                transition_mask: TerrainTransitionMask::from_bits(1 << face as u8),
            },
        );
        assert!(
            !chunk.index_groups.transitions[face.index()].is_empty(),
            "missing transition triangles on {face:?}"
        );
        assert!(
            !chunk.index_groups.caps[face.index()].is_empty(),
            "missing cap triangles on {face:?}"
        );
        assert!(
            chunk
                .normals
                .iter()
                .flatten()
                .all(|component| component.is_finite())
        );
        let ready_except_transition = TerrainTransitionMask::from_bits(0x3f & !(1 << face as u8));
        let sealed = chunk
            .index_groups
            .sealed_indices(chunk.transition_mask, ready_except_transition);
        assert_eq!(
            sealed.len(),
            chunk.index_groups.regular.len()
                + chunk.index_groups.transitions[face.index()].len()
                + chunk.index_groups.caps[face.index()].len(),
            "pending transition must bridge the inset mesh before its cap on {face:?}"
        );
        let collision = chunk.sealed_collision_chunk(TerrainTransitionMask::NONE);
        assert_eq!(collision.indices, chunk.index_groups.regular);
        assert_eq!(
            collision.active_groups,
            super::TerrainTriangleGroupMask::REGULAR
        );
    }
}

#[test]
fn generated_triangle_bvh_has_real_branches_and_leaf_ranges() {
    let field = TerrainField::new(WorldSeed(91));
    let terrain = TerrainOctree::default().snapshot();
    let chunk = mesh_chunk(&field, &terrain, surface_request(-1));
    assert!(chunk.triangle_bvh.nodes.len() > 1);
    let all_group_triangles = chunk.index_groups.regular.len()
        + chunk
            .index_groups
            .transitions
            .iter()
            .map(Vec::len)
            .sum::<usize>()
        + chunk.index_groups.caps.iter().map(Vec::len).sum::<usize>();
    assert_eq!(chunk.triangle_bvh.triangles.len(), all_group_triangles / 3);
    assert_eq!(
        chunk
            .sealed_collision_chunk(TerrainTransitionMask::NONE)
            .triangle_bvh,
        chunk.triangle_bvh
    );
    assert_eq!(chunk.triangle_bvh.nodes[0].triangle_count, 0);
    assert!(
        chunk
            .triangle_bvh
            .nodes
            .iter()
            .filter(|node| node.triangle_count != 0)
            .all(|node| node.left_child.is_none() && node.right_child.is_none())
    );
}
