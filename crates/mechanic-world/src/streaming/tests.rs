mod coverage;
mod priority;

use std::collections::BTreeSet;

use bevy_math::DVec3;

use super::selection::{classify_node, mesh_dependency_generation};
use super::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainFace, TerrainStreamer, TerrainTransitionMask,
    publication_face_mask, select_active_nodes, select_active_nodes_cached,
};
use crate::{
    BRICK_EDGE_CELLS, BrickCoord, TerrainDensityClass, TerrainField, TerrainNodeId, TerrainOctree,
    WorldCell, WorldPosition, WorldSeed,
};

#[test]
fn selected_cut_is_non_overlapping_and_two_to_one_balanced() {
    let field = TerrainField::new(WorldSeed(7));
    let terrain = TerrainOctree::default().snapshot();
    let cut = select_active_nodes(&field, &terrain, WorldPosition(DVec3::new(0.0, 4.0, 0.0)));
    assert!(!cut.is_empty());
    for (index, first) in cut.iter().enumerate() {
        for second in &cut[index + 1..] {
            assert!(!contains(first.id, second.id) && !contains(second.id, first.id));
        }
    }
    for node in &cut {
        for face in TerrainFace::ALL {
            if let Some(neighbour) = super::adjacent_leaf(node.id, face)
                && let Some(owner) =
                    super::owner_of_leaf(&cut.iter().map(|node| node.id).collect(), neighbour)
            {
                assert!(node.id.level.abs_diff(owner.level) <= 1);
            }
        }
    }
}

fn contains(outer: TerrainNodeId, inner: TerrainNodeId) -> bool {
    if outer.level <= inner.level {
        return false;
    }
    let edge = outer.edge_bricks();
    let offset = |inner: i32, outer: i32| i64::from(inner) - i64::from(outer);
    [
        offset(inner.coordinates.x, outer.coordinates.x),
        offset(inner.coordinates.y, outer.coordinates.y),
        offset(inner.coordinates.z, outer.coordinates.z),
    ]
    .into_iter()
    .all(|value| (0..edge).contains(&value))
}

#[test]
fn stale_active_generation_is_not_ready_for_seam_publication() {
    let id = TerrainNodeId::default();
    let old = ActiveTerrainNode {
        id,
        generation: 1,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let new = ActiveTerrainNode {
        generation: 2,
        ..old
    };
    let mut streamer = TerrainStreamer::default();
    streamer.set_desired([old]);
    streamer.mark_started(old);
    assert!(streamer.stage(old));
    assert_eq!(streamer.activate(id), vec![old]);
    assert_eq!(streamer.current_active().collect::<Vec<_>>(), vec![old]);

    streamer.set_desired([new]);
    assert!(streamer.current_active().next().is_none());
}

#[test]
fn mesh_generation_tracks_edits_in_a_neighbouring_sampling_halo() {
    let field = TerrainField::new(WorldSeed(2));
    let surface = field.surface_height(0.08, -0.025);
    let mut terrain = TerrainOctree::default();
    let outcome = terrain
        .excavate_sphere(
            &field,
            WorldPosition(DVec3::new(0.08, surface - 0.025, -0.025)),
            0.10,
        )
        .expect("fixture edit is inside the world");
    let edited = *outcome
        .changed_brick_coordinates()
        .iter()
        .min_by_key(|coordinate| coordinate.x)
        .expect("fixture removes terrain");
    let adjacent = TerrainNodeId::leaf(BrickCoord::new(edited.x - 1, edited.y, edited.z));
    let snapshot = terrain.snapshot();

    assert_eq!(
        snapshot
            .node(adjacent)
            .map_or(0, |node| node.latest_revision),
        0,
        "the adjacent node itself must remain procedurally untouched"
    );
    assert!(
        mesh_dependency_generation(&snapshot, adjacent, TerrainTransitionMask::NONE) > 0,
        "the adjacent mesh must be regenerated because its halo samples the edit"
    );
}

#[test]
fn cached_reselection_reuses_horizontal_procedural_bounds() {
    let field = TerrainField::new(WorldSeed(17));
    let terrain = TerrainOctree::default().snapshot();
    let focus = WorldPosition(DVec3::new(-37.0, 4.0, 29.0));
    let mut cache = TerrainBoundsCache::default();
    let cold = select_active_nodes_cached(&field, &terrain, focus, &mut cache);
    let warm = select_active_nodes_cached(&field, &terrain, focus, &mut cache);
    assert_eq!(cold.nodes, warm.nodes);
    assert!(warm.stats.cache_hits > 0);
    assert_eq!(warm.stats.cache_misses, 0);
    assert!(warm.stats.cache_memory_bytes <= super::bounds_cache::PROCEDURAL_BOUNDS_CACHE_BYTES);
}

#[test]
fn edited_surface_keeps_coarse_sampling_footprints_outside_the_edit() {
    let field = TerrainField::new(WorldSeed(2_255_932_754_758_176_049));
    let surface = field.surface_height(0.3, 0.8);
    let focus = WorldPosition(DVec3::new(0.3, surface, 0.8));
    let mut terrain = TerrainOctree::default();
    terrain
        .excavate_sphere(
            &field,
            WorldPosition(DVec3::new(0.3, surface - 0.2, 0.8)),
            0.65,
        )
        .expect("edit is inside the world");
    let snapshot = terrain.snapshot();
    let cut = select_active_nodes(&field, &snapshot, focus);
    let offenders = cut
        .iter()
        .filter(|node| node.id.level > 0)
        .filter(|node| {
            let minimum = node.id.minimum_cell_i64();
            let maximum = node.id.maximum_cell_exclusive_i64();
            let stride = 1_i64 << node.id.level;
            snapshot
                .minimum_promoted_density_between(
                    WorldCell::new(
                        i32::try_from(minimum[0] - 1).expect("fixture fits cell space"),
                        i32::try_from(minimum[1] - 1).expect("fixture fits cell space"),
                        i32::try_from(minimum[2] - 1).expect("fixture fits cell space"),
                    ),
                    WorldCell::new(
                        i32::try_from(maximum[0] + stride - 1).expect("fixture fits cell space"),
                        i32::try_from(maximum[1] + stride - 1).expect("fixture fits cell space"),
                        i32::try_from(maximum[2] + stride - 1).expect("fixture fits cell space"),
                    ),
                )
                .is_some()
        })
        .map(|node| node.id)
        .collect::<Vec<_>>();

    assert!(
        offenders.is_empty(),
        "coarse nodes still sample promoted edits: {offenders:?}"
    );
}

#[test]
fn every_selected_and_balance_created_node_is_mixed() {
    let field = TerrainField::new(WorldSeed(23));
    let terrain = TerrainOctree::default().snapshot();
    let mut cache = TerrainBoundsCache::default();
    let selection = select_active_nodes_cached(
        &field,
        &terrain,
        WorldPosition(DVec3::new(61.0, 5.0, -44.0)),
        &mut cache,
    );
    for node in selection.nodes {
        assert_eq!(
            classify_node(&field, &terrain, node.id, false, &mut cache),
            crate::TerrainDensityClass::Mixed
        );
    }
}

#[test]
fn critical_empty_results_count_as_resolved_and_dirty() {
    let id = TerrainNodeId::default();
    let node = ActiveTerrainNode {
        id,
        generation: 9,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let mut streamer = TerrainStreamer::default();
    streamer.set_critical_nodes([id]);
    streamer.set_desired([node]);
    assert_eq!(streamer.local_readiness().resolved, 0);
    streamer.mark_started(node);
    assert!(streamer.stage(node));
    assert_eq!(streamer.activate(id), vec![node]);
    assert!(streamer.local_readiness().is_complete());
    assert!(streamer.take_dirty_publication().contains(&id));
}

#[test]
fn publication_delta_reports_only_changed_upserts_and_removals() {
    let id = TerrainNodeId::default();
    let node = ActiveTerrainNode {
        id,
        generation: 3,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let mut streamer = TerrainStreamer::default();
    streamer.set_desired([node]);
    assert!(streamer.stage(node));
    assert_eq!(streamer.activate(id), vec![node]);

    let published = streamer.take_publication_delta();
    assert_eq!(published.upserts.len(), 1);
    assert_eq!(published.upserts[0].node, node);
    assert_eq!(published.upserts[0].ready_faces.bits(), 0x3f);
    assert!(published.removals.is_empty());
    assert!(streamer.take_publication_delta().is_empty());

    streamer.set_desired([]);
    let removed = streamer.take_publication_delta();
    assert!(removed.upserts.is_empty());
    assert_eq!(removed.removals, vec![id]);
}

#[test]
fn refined_cut_stays_active_until_every_replacement_is_staged() {
    let parent = TerrainNodeId::containing(BrickCoord::new(0, 0, 0), 1).unwrap();
    let old = ActiveTerrainNode {
        id: parent,
        generation: 0,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let children = parent
        .children()
        .unwrap()
        .map(|id| ActiveTerrainNode { id, ..old });
    let mut streamer = TerrainStreamer::default();
    streamer.set_desired([old]);
    assert!(streamer.stage(old));
    assert_eq!(streamer.activate(parent), vec![old]);

    streamer.set_desired(children);
    for child in &children[..7] {
        assert!(streamer.stage(*child));
        assert!(streamer.activate(child.id).is_empty());
        assert_eq!(streamer.active().collect::<Vec<_>>(), vec![old]);
    }

    let last = children[7];
    assert!(streamer.stage(last));
    let activated = streamer.activate(last.id);
    assert_eq!(
        activated.into_iter().collect::<BTreeSet<_>>(),
        children.into()
    );
    assert_eq!(streamer.active().collect::<BTreeSet<_>>(), children.into());
}

#[test]
fn coarsened_cut_replaces_all_fine_nodes_in_one_activation() {
    let parent = TerrainNodeId::containing(BrickCoord::new(0, 0, 0), 1).unwrap();
    let coarse = ActiveTerrainNode {
        id: parent,
        generation: 0,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let children = parent
        .children()
        .unwrap()
        .map(|id| ActiveTerrainNode { id, ..coarse });
    let mut streamer = TerrainStreamer::default();
    streamer.set_desired(children);
    for child in children {
        assert!(streamer.stage(child));
        assert_eq!(streamer.activate(child.id), vec![child]);
    }

    streamer.set_desired([coarse]);
    assert_eq!(streamer.active().collect::<BTreeSet<_>>(), children.into());
    assert!(streamer.stage(coarse));
    assert_eq!(streamer.activate(parent), vec![coarse]);
    assert_eq!(streamer.active().collect::<Vec<_>>(), vec![coarse]);
}

#[test]
fn publication_caps_only_faces_with_pending_desired_neighbors() {
    let node = TerrainNodeId::leaf(BrickCoord::new(0, 0, 0));
    let neighbor = TerrainNodeId::leaf(BrickCoord::new(1, 0, 0));
    let active = BTreeSet::from([node]);
    let isolated = publication_face_mask(node, &active, &active);
    assert!(
        TerrainFace::ALL
            .into_iter()
            .all(|face| isolated.contains(face))
    );

    let desired = BTreeSet::from([node, neighbor]);
    let pending = publication_face_mask(node, &active, &desired);
    assert!(!pending.contains(TerrainFace::PositiveX));
    let ready = publication_face_mask(node, &desired, &desired);
    assert!(ready.contains(TerrainFace::PositiveX));
}

#[test]
fn transition_boundary_samples_propagate_across_fine_edges_and_corners() {
    let node = TerrainNodeId::leaf(BrickCoord::new(0, 0, 0));
    let side = TerrainNodeId::leaf(BrickCoord::new(0, 0, 1));
    let diagonal = TerrainNodeId::leaf(BrickCoord::new(0, 1, 1));
    let selected = BTreeSet::from([node, side, diagonal]);
    let mut masks = std::collections::BTreeMap::from([
        (
            node,
            TerrainTransitionMask::from_bits(1 << TerrainFace::PositiveX as u8),
        ),
        (side, TerrainTransitionMask::NONE),
        (diagonal, TerrainTransitionMask::NONE),
    ]);

    super::selection::propagate_transition_boundary_sync(&selected, &mut masks);

    assert!(masks[&side].synchronizes_boundary_feature(
        (1 << TerrainFace::PositiveX as u8) | (1 << TerrainFace::NegativeZ as u8)
    ));
    assert!(masks[&diagonal].synchronizes_boundary_feature(
        (1 << TerrainFace::PositiveX as u8)
            | (1 << TerrainFace::NegativeY as u8)
            | (1 << TerrainFace::NegativeZ as u8)
    ));
}

#[test]
fn procedural_bounds_are_conservative_on_every_meshing_lattice() {
    for seed in [WorldSeed(0), WorldSeed(97), WorldSeed(u64::MAX)] {
        let field = TerrainField::new(seed);
        let terrain = TerrainOctree::default().snapshot();
        let mut cache = TerrainBoundsCache::default();
        for level in 0..=5 {
            for coordinate in [BrickCoord::new(-37, 100, 29), BrickCoord::new(41, -90, -33)] {
                let id = TerrainNodeId::containing(coordinate, level).unwrap();
                let class = classify_node(&field, &terrain, id, false, &mut cache);
                assert_ne!(class, TerrainDensityClass::Mixed, "{id:?} {seed:?}");
                let minimum = id.minimum_cell_i64();
                let stride = 1_i64 << level;
                let y = if class == TerrainDensityClass::Empty {
                    minimum[1]
                } else {
                    id.maximum_cell_exclusive_i64()[1] - 1
                };
                for z in 0..=BRICK_EDGE_CELLS {
                    for x in 0..=BRICK_EDGE_CELLS {
                        let cell = WorldCell::new(
                            i32::try_from(minimum[0] + i64::from(x) * stride).unwrap(),
                            i32::try_from(y).unwrap(),
                            i32::try_from(minimum[2] + i64::from(z) * stride).unwrap(),
                        );
                        let solid = field.sample_cell(cell).is_solid();
                        assert_eq!(
                            solid,
                            class == TerrainDensityClass::Solid,
                            "false {class:?} bound for {id:?} at {cell:?} with {seed:?}"
                        );
                    }
                }
            }
            let ground = WorldPosition(bevy_math::DVec3::new(
                3.0,
                field.surface_height(3.0, -2.0),
                -2.0,
            ))
            .cell()
            .unwrap();
            let ground_id = TerrainNodeId::containing(ground.brick(), level).unwrap();
            assert_eq!(
                classify_node(&field, &terrain, ground_id, false, &mut cache),
                TerrainDensityClass::Mixed
            );
        }
    }
}
