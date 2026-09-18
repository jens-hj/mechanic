use super::super::*;
use crate::WorldBounds;

fn bounds(minimum: DVec3, maximum: DVec3) -> WorldBounds {
    WorldBounds {
        minimum: WorldPosition(minimum),
        maximum: WorldPosition(maximum),
    }
}

#[test]
fn actual_mesh_bounds_refit_ancestors_across_nominal_seams_and_removal() {
    let node = TerrainNodeId::leaf(BrickCoord::new(0, 0, 0));
    let nominal = node.world_bounds();
    let mut actual = nominal;
    actual.maximum.0.x += 1e-7;
    let probe = bounds(
        DVec3::new(nominal.maximum.0.x + 1e-8, 0.5, 0.5),
        DVec3::new(nominal.maximum.0.x + 5e-8, 1.0, 1.0),
    );
    let mut index = TerrainSpatialIndex::default();
    index.insert(node);
    assert!(index.bounds_candidates(probe).is_empty());
    assert!(index.insert_bounds(node, actual));
    assert_eq!(index.bounds_candidates(probe), vec![node]);
    assert_eq!(
        index.ray_candidates(
            WorldPosition(probe.minimum.0 - DVec3::Y * 2.0),
            DVec3::Y,
            4.0
        ),
        vec![node]
    );
    assert_eq!(
        index.nearest(probe.minimum, |candidate| Some((candidate, 0.0))),
        Some(node)
    );
    assert_eq!(index.descendant_counts[&TerrainNodeId::ROOT], 1);
    index.remove(node);
    assert!(index.bounds_candidates(probe).is_empty());
    assert!(index.aggregate_bounds.is_empty());
}

#[test]
fn mixed_parent_child_updates_keep_every_overlapping_candidate_in_stable_order() {
    let a = TerrainNodeId::leaf(BrickCoord::new(0, 0, 0));
    let b = TerrainNodeId::leaf(BrickCoord::new(1, 0, 0));
    let parent = a.parent().unwrap();
    let region = parent.world_bounds();
    let mut first = TerrainSpatialIndex::default();
    let mut second = TerrainSpatialIndex::default();
    for node in [a, parent, b] {
        first.insert(node);
    }
    for node in [b, parent, a] {
        second.insert(node);
    }
    let mut expected = vec![a, b, parent];
    expected.sort_unstable();
    assert_eq!(first.bounds_candidates(region), expected);
    assert_eq!(first, second);
    first.remove(parent);
    expected.retain(|&node| node != parent);
    assert_eq!(first.bounds_candidates(region), expected);
    assert_eq!(first.descendant_counts[&TerrainNodeId::ROOT], 2);
    let saved = first.clone();
    assert!(!first.insert_bounds(a, bounds(DVec3::splat(f64::NAN), DVec3::ZERO)));
    assert_eq!(first, saved);
}
