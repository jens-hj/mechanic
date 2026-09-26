//! The order streaming work is handed out in, and how the detail scale
//! shapes the cut.

use std::collections::BTreeSet;

use bevy_math::DVec3;

use super::super::{
    ActiveTerrainNode, TerrainBoundsCache, TerrainStreamer, TerrainTransitionMask, TerrainView,
    select_active_nodes_with_interests,
};
use crate::{
    BrickCoord, TerrainFace, TerrainField, TerrainNodeId, TerrainOctree, WorldPosition, WorldSeed,
};

/// A level-two node whose minimum corner is `metres` along x from the origin.
fn node_at(
    x_metres: f64,
    z_metres: f64,
    transition_mask: TerrainTransitionMask,
) -> ActiveTerrainNode {
    #[expect(clippy::cast_possible_truncation, reason = "a few hundred bricks")]
    let brick = |metres: f64| (metres / crate::BRICK_EDGE_METERS).floor() as i32;
    ActiveTerrainNode {
        id: TerrainNodeId::containing(BrickCoord::new(brick(x_metres), 0, brick(z_metres)), 2)
            .expect("level two exists"),
        generation: 0,
        transition_mask,
    }
}

fn first_request(streamer: &mut TerrainStreamer) -> ActiveTerrainNode {
    streamer
        .next_request(&BTreeSet::new(), WorldPosition(DVec3::ZERO))
        .expect("work is pending")
}

#[test]
fn nearby_ground_streams_before_distant_seams() {
    let mut seam = TerrainTransitionMask::NONE;
    seam.insert(TerrainFace::PositiveX);
    let far_seam = node_at(600.0, 0.0, seam);
    let near = node_at(20.0, 0.0, TerrainTransitionMask::NONE);
    let mut streamer = TerrainStreamer::default();
    streamer.set_desired([far_seam, near]);
    assert_eq!(first_request(&mut streamer), near);
}

#[test]
fn ground_in_view_streams_before_ground_behind() {
    let ahead = node_at(90.0, 0.0, TerrainTransitionMask::NONE);
    let behind = node_at(-96.0, 0.0, TerrainTransitionMask::NONE);
    let mut streamer = TerrainStreamer::default();
    streamer.set_desired([ahead, behind]);
    streamer.set_view(Some(TerrainView {
        forward: DVec3::NEG_X,
        half_angle: 0.6,
    }));
    assert_eq!(first_request(&mut streamer), behind);
    streamer.set_view(Some(TerrainView {
        forward: DVec3::X,
        half_angle: 0.6,
    }));
    assert_eq!(first_request(&mut streamer), ahead);
}

#[test]
fn lower_detail_coarsens_the_cut_but_keeps_the_horizon() {
    let field = TerrainField::new(WorldSeed(42));
    let terrain = TerrainOctree::default().snapshot();
    let focus = field.safe_spawn();
    let cut = |scale: f64| {
        select_active_nodes_with_interests(
            &field,
            &terrain,
            focus,
            &[],
            scale,
            &mut TerrainBoundsCache::default(),
        )
    };
    let full = cut(1.0);
    let coarse = cut(0.5);
    let fine = |selection: &super::super::TerrainSelection| {
        selection.stats.selected_by_lod[..=3].iter().sum::<usize>()
    };
    assert!(
        fine(&coarse) * 2 < fine(&full),
        "coarser detail keeps fine nodes"
    );
    let reach = |selection: &super::super::TerrainSelection| {
        selection
            .nodes
            .iter()
            .map(|node| {
                let (minimum, maximum) = super::super::selection::node_bounds(node.id);
                super::super::selection::horizontal_distance_squared_to_bounds(
                    focus.0, minimum, maximum,
                )
                .sqrt()
            })
            .fold(0.0, f64::max)
    };
    assert!((reach(&coarse) - reach(&full)).abs() < 110.0);
}
