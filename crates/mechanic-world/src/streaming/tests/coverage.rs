//! Walking through generated terrain with workers that finish out of order
//! must never leave surface near the player without an active node.

use std::collections::BTreeSet;

use bevy_math::DVec3;

use super::super::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainStreamer, select_active_nodes_cached,
};
use crate::{
    MAX_STREAMED_LEVEL, TerrainField, TerrainNodeId, TerrainOctree, WorldPosition, WorldSeed,
};

/// Jobs a simulated worker pool holds at once.
const IN_FLIGHT: usize = 24;

/// How far around the player the ground must stay covered, and the spacing
/// of the surface points checked within it.
const COVERED_METRES: f64 = 60.0;
const POINT_SPACING_METRES: f64 = 12.0;
const POINT_STEPS: i32 = 5;

fn surface_points(field: &TerrainField, centre: DVec3) -> Vec<DVec3> {
    let mut points = Vec::new();
    for i in -POINT_STEPS..=POINT_STEPS {
        for k in -POINT_STEPS..=POINT_STEPS {
            let (x, z) = (
                f64::from(i).mul_add(POINT_SPACING_METRES, centre.x) + 0.37,
                f64::from(k).mul_add(POINT_SPACING_METRES, centre.z) + 0.71,
            );
            if (x - centre.x).hypot(z - centre.z) > COVERED_METRES {
                continue;
            }
            if let Some(y) = field.topmost_surface(x, z) {
                points.push(DVec3::new(x, y, z));
            }
        }
    }
    points
}

fn covering_node(streamer: &TerrainStreamer, point: DVec3) -> Option<TerrainNodeId> {
    let cell = WorldPosition(point).cell().ok()?;
    let active = streamer
        .active()
        .map(|node| node.id)
        .collect::<BTreeSet<_>>();
    (0..=MAX_STREAMED_LEVEL).find_map(|level| {
        let id = TerrainNodeId::containing(cell.brick(), level)?;
        active.contains(&id).then_some(id)
    })
}

/// Hands out jobs nearest-first, completes them in a scrambled order, and
/// returns once nothing is pending or in flight, checking coverage after
/// every completion when `check` is set.
fn drain(
    field: &TerrainField,
    streamer: &mut TerrainStreamer,
    in_flight: &mut Vec<ActiveTerrainNode>,
    focus: DVec3,
    random: &mut u64,
    check: bool,
) {
    let points = surface_points(field, focus);
    loop {
        while in_flight.len() < IN_FLIGHT {
            let ids = in_flight.iter().map(|node| node.id).collect();
            let Some(node) = streamer.next_request(&ids, WorldPosition(focus)) else {
                break;
            };
            streamer.mark_started(node);
            in_flight.push(node);
        }
        if in_flight.is_empty() {
            break;
        }
        *random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let index = usize::try_from(*random >> 33).unwrap_or(0) % in_flight.len();
        let node = in_flight.swap_remove(index);
        if streamer.stage(node) {
            streamer.activate(node.id);
        }
        if check {
            for &point in &points {
                assert!(
                    covering_node(streamer, point).is_some(),
                    "surface at {point:?} is uncovered while walking at {focus:?}"
                );
            }
        }
    }
}

#[test]
fn walking_never_uncovers_nearby_ground() {
    let field = TerrainField::new(WorldSeed(42));
    let terrain = TerrainOctree::default().snapshot();
    let mut cache = TerrainBoundsCache::default();
    let mut streamer = TerrainStreamer::default();
    let mut in_flight = Vec::new();
    let mut random = 0x5eed;
    let spawn = field.safe_spawn().0;
    for step in 0..4 {
        let focus = spawn + DVec3::new(f64::from(step) * 23.0, 0.0, f64::from(step) * 9.0);
        let selection =
            select_active_nodes_cached(&field, &terrain, WorldPosition(focus), &mut cache);
        streamer.set_desired(selection.nodes);
        // Right after a move, the previous cut must still cover the ground.
        if step > 0 {
            for point in surface_points(&field, focus) {
                assert!(
                    covering_node(&streamer, point).is_some(),
                    "moving to {focus:?} dropped the surface at {point:?}"
                );
            }
        }
        drain(
            &field,
            &mut streamer,
            &mut in_flight,
            focus,
            &mut random,
            step > 0,
        );
    }
}
