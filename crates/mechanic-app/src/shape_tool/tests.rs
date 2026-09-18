use super::{
    FeatureDrag, FeatureEdgeHit, ShapeMirror, ShapeSnap, begin_group_drag, clamp_into_region,
    drag_edits, drag_offset, edge_insertion, hovered_feature_edge, hovered_source_edge,
    hovered_vertex, inflated_bounds_ray_distance, mirrored_edits, most_visible_axis, nudge_edits,
    screen_axis, vertex_position,
};
use bevy::prelude::*;
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ConstructionMaterial, CuboidSpec,
    CylinderDimensions, CylinderSpec, EdgeChainRef, EdgeTreatment, POSITION_TICK_METERS,
    POSITION_TICKS_PER_GRID_UNIT, ShapeRegion, SolidOwner,
};

fn region(size: IVec3) -> ShapeRegion {
    ShapeRegion::new(IVec3::ZERO, size, ConstructionMaterial::Steel).unwrap()
}

fn cell_steps() -> i16 {
    i16::try_from(POSITION_TICKS_PER_GRID_UNIT).unwrap()
}

#[test]
fn a_fresh_region_offers_its_eight_corners() {
    assert_eq!(region(IVec3::new(3, 2, 4)).vertices().count(), 8);
}

#[test]
fn the_pointer_grabs_the_cage_vertex_it_is_aimed_at() {
    let region = region(IVec3::ONE);
    let target = [1, 1, 1];
    let position = vertex_position(&region, target).unwrap();
    let origin = position + Vec3::new(0.0, 0.0, 2.0);
    assert_eq!(
        hovered_vertex(&region, origin, Vec3::new(0.0, 0.0, -1.0)),
        Some(target)
    );
}

#[test]
fn a_vertex_cannot_be_dragged_out_of_the_region() {
    // The whole point of the editable area: a corner is drawn inward, never
    // pushed out past where the blocks were.
    let region = region(IVec3::ONE);
    let cell = i32::from(cell_steps());
    assert_eq!(
        clamp_into_region(&region, [0, 0, 0], [-cell, 0, 0]),
        [0, 0, 0],
        "the minimum corner has nowhere outward to go"
    );
    assert_eq!(
        clamp_into_region(&region, [1, 1, 1], [cell, 0, 0]),
        [0, 0, 0],
        "the maximum corner has nowhere outward to go"
    );
    assert_eq!(
        clamp_into_region(&region, [0, 0, 0], [cell, 0, 0]),
        [cell_steps(), 0, 0],
        "inward as far as the opposite face is allowed"
    );
}

#[test]
fn dragging_lands_only_on_multiples_of_the_increment() {
    let region = region(IVec3::new(4, 4, 4));
    let snap = ShapeSnap { steps: 5 };
    let index = [0, 0, 0];
    let start = vertex_position(&region, index).unwrap();
    let direction = Vec3::new(0.0, 0.0, -1.0);
    let origin = start + Vec3::new(0.0, 0.0, 2.0);
    let drag = begin_group_drag(&region, index, &[], origin, direction);
    for travel in 1_i16..14 {
        let moved = origin
            + Vec3::new(
                mechanic_core::POSITION_TICK_METERS * f32::from(travel),
                0.0,
                0.0,
            );
        let offset = drag_offset(&region, &drag, snap, moved, direction);
        assert_eq!(
            i32::from(offset[0]) % snap.steps,
            0,
            "travel {travel} produced off-grid offset {offset:?}"
        );
    }
}

#[test]
fn dragging_changes_only_the_active_axis() {
    let region = region(IVec3::new(4, 4, 4));
    let snap = ShapeSnap { steps: 1 };
    let index = [0, 0, 0];
    let start = vertex_position(&region, index).unwrap();
    let direction = Vec3::NEG_Z;
    let origin = start + Vec3::Z * 2.0;
    let drag = begin_group_drag(&region, index, &[], origin, direction);
    assert_eq!(drag.axis, 0, "X is most visible from this view");

    let target = start + Vec3::new(POSITION_TICK_METERS * 7.0, POSITION_TICK_METERS * 11.0, 0.0);
    let moved_direction = (target - origin).normalize();
    assert_eq!(
        drag_offset(&region, &drag, snap, origin, moved_direction),
        [7, 0, 0],
        "screen travel on Y must not leak into an X-axis edit"
    );
}

#[test]
fn cycling_the_drag_axis_preserves_prior_travel() {
    let region = region(IVec3::new(4, 4, 4));
    let snap = ShapeSnap { steps: 1 };
    let index = [0, 0, 0];
    let start = vertex_position(&region, index).unwrap();
    let direction = Vec3::NEG_Z;
    let origin = start + Vec3::Z * 2.0;
    let mut drag = begin_group_drag(&region, index, &[], origin, direction);
    let target_x = start + Vec3::X * POSITION_TICK_METERS * 7.0;
    let moved_x = (target_x - origin).normalize();
    drag.offset = drag_offset(&region, &drag, snap, origin, moved_x);

    drag.cycle_axis(origin, moved_x);
    assert_eq!(drag.axis, 1);
    assert_eq!(
        drag_offset(&region, &drag, snap, origin, moved_x),
        [7, 0, 0]
    );
    let target_y = target_x + Vec3::Y * POSITION_TICK_METERS * 5.0;
    let moved_y = (target_y - origin).normalize();
    assert_eq!(
        drag_offset(&region, &drag, snap, origin, moved_y),
        [7, 5, 0]
    );
}

#[test]
fn the_initial_drag_axis_is_the_most_visible_world_axis() {
    assert_eq!(most_visible_axis(Vec3::NEG_Z), 0);
    assert_eq!(most_visible_axis(Vec3::new(0.9, 0.2, 0.3)), 1);
    assert_eq!(most_visible_axis(Vec3::new(0.2, 0.3, 0.9)), 0);
}

#[test]
fn cycling_the_increment_walks_coarse_to_fine_and_wraps() {
    let mut snap = ShapeSnap {
        steps: POSITION_TICKS_PER_GRID_UNIT,
    };
    let mut seen = vec![snap.steps];
    for _ in 0..4 {
        snap.cycle();
        seen.push(snap.steps);
    }
    assert_eq!(seen, vec![100, 50, 25, 20, 5]);
    snap.cycle();
    assert_eq!(snap.steps, 100);
    assert_eq!(ShapeSnap::feature_default().steps, 20);
}

#[test]
fn side_on_feature_drag_reaches_the_first_five_centimetre_increment() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let hit = FeatureEdgeHit {
        target: EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        },
        point: Vec3::ZERO,
        tangent: Vec3::Z,
        bisector: Vec3::X,
        distance: 0.0,
    };
    let ray_origin = Vec3::Y;
    let mut drag = FeatureDrag::begin(
        hit,
        vec![hit.target],
        EdgeTreatment::Fillet,
        None,
        0,
        ray_origin,
        Vec3::NEG_Y,
    );
    let five_centimetres = Vec3::new(0.05, 0.0, 0.0);
    let moved_ray = (five_centimetres - ray_origin).normalize();

    assert_eq!(
        drag.proposed_amount(ShapeSnap::feature_default(), ray_origin, moved_ray),
        20
    );
}

#[test]
fn side_on_feature_drag_keeps_a_coalesced_full_block_motion() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let hit = FeatureEdgeHit {
        target: EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        },
        point: Vec3::ZERO,
        tangent: Vec3::Z,
        bisector: Vec3::X,
        distance: 0.0,
    };
    let ray_origin = Vec3::Y;
    let mut drag = FeatureDrag::begin(
        hit,
        vec![hit.target],
        EdgeTreatment::Fillet,
        None,
        0,
        ray_origin,
        Vec3::NEG_Y,
    );
    let full_block = Vec3::new(0.25, 0.0, 0.0);
    let moved_ray = (full_block - ray_origin).normalize();

    assert_eq!(
        drag.proposed_amount(ShapeSnap::feature_default(), ray_origin, moved_ray),
        100,
        "a slow fillet preview must not discard coalesced pointer motion"
    );

    let beyond_block = Vec3::new(0.30, 0.0, 0.0);
    let beyond_ray = (beyond_block - ray_origin).normalize();
    assert_eq!(
        drag.proposed_amount(ShapeSnap::feature_default(), ray_origin, beyond_ray),
        120
    );
    drag.discard_rejected_excess(100);
    assert_eq!(
        drag.proposed_amount(ShapeSnap::feature_default(), ray_origin, beyond_ray),
        100,
        "holding at the geometry limit must not retry a rejected preview"
    );
}

#[test]
fn grazing_feature_drag_cannot_skip_the_first_five_centimetre_increment() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let hit = FeatureEdgeHit {
        target: EdgeChainRef {
            owner,
            edge: mechanic_core::TopologyKey {
                source: mechanic_core::TopologySource::Base,
                local: 0,
            },
        },
        point: Vec3::ZERO,
        tangent: Vec3::Z,
        bisector: Vec3::X,
        distance: 0.0,
    };
    let ray_origin = Vec3::Y;
    let mut drag = FeatureDrag::begin(
        hit,
        vec![hit.target],
        EdgeTreatment::Fillet,
        None,
        0,
        ray_origin,
        Vec3::NEG_Y,
    );
    let amplified_ray = (Vec3::X * 2.0 - ray_origin).normalize();

    assert_eq!(
        drag.proposed_amount(ShapeSnap::feature_default(), ray_origin, amplified_ray),
        20,
        "one unstable projection sample must not jump to a multi-block radius"
    );
    assert_eq!(
        drag.proposed_amount(ShapeSnap::feature_default(), ray_origin, amplified_ray),
        20,
        "a stationary pointer must not continue increasing the radius"
    );
}

#[test]
fn a_nudge_moves_the_selection_one_increment_along_one_axis() {
    let region = region(IVec3::new(2, 2, 2));
    let edits = nudge_edits(
        &region,
        &[[0, 0, 0], [1, 0, 0]],
        1,
        1,
        ShapeSnap { steps: 5 },
        ShapeMirror::default(),
    );
    assert_eq!(edits.len(), 2);
    assert!(edits.iter().all(|&(_, offset)| offset == [0, 5, 0]));
}

#[test]
fn a_nudge_that_would_leave_the_region_reports_nothing_to_do() {
    let region = region(IVec3::ONE);
    // The minimum corner cannot go further down.
    let edits = nudge_edits(
        &region,
        &[[0, 0, 0]],
        1,
        -1,
        ShapeSnap { steps: 5 },
        ShapeMirror::default(),
    );
    assert!(edits.is_empty(), "clamped moves should not be committed");
}

#[test]
fn both_mirror_planes_together_give_four_way_symmetry() {
    let region = region(IVec3::new(2, 1, 2));
    let edits = mirrored_edits(
        &region,
        [0, 1, 0],
        [2, -3, 4],
        ShapeMirror { x: true, z: true },
    );
    assert_eq!(edits.len(), 4);
    assert!(edits.contains(&([1, 1, 0], [-2, -3, 4])));
    assert!(edits.contains(&([0, 1, 1], [2, -3, -4])));
    assert!(edits.contains(&([1, 1, 1], [-2, -3, -4])));
}

#[test]
fn a_vertex_on_a_mirror_plane_cannot_leave_it() {
    // A three-plane cage has a true middle column on x.
    let mut region = region(IVec3::new(2, 1, 1));
    region.subdivide(0, 1).unwrap();
    let edits = mirrored_edits(
        &region,
        [1, 0, 0],
        [4, -3, 0],
        ShapeMirror { x: true, z: false },
    );
    assert_eq!(
        edits,
        vec![([1, 0, 0], [0, -3, 0])],
        "a vertex on the centre plane must stay on it, or symmetry breaks"
    );
}

#[test]
fn a_group_drag_moves_every_selected_vertex_by_the_same_delta() {
    let mut region = region(IVec3::new(4, 4, 4));
    region.set_offset([1, 0, 0], [0, 3, 0]).unwrap();
    let primary = [0, 0, 0];
    let companion = [1, 0, 0];
    let direction = Vec3::new(0.0, 0.0, -1.0);
    let origin = vertex_position(&region, primary).unwrap() + Vec3::new(0.0, 0.0, 2.0);
    let mut drag = begin_group_drag(&region, primary, &[primary, companion], origin, direction);
    drag.offset = [0, 5, 0];

    let edits = drag_edits(&region, &drag, ShapeMirror::default());
    assert!(edits.contains(&(primary, [0, 5, 0])));
    assert!(
        edits.contains(&(companion, [0, 8, 0])),
        "the companion keeps its own head start: {edits:?}"
    );
}

#[test]
fn nearing_an_edge_offers_a_vertex_at_the_nearest_grid_position() {
    // A two-cell region has one grid position along its long edges where a
    // plane could go, and the pointer has to be near it to be offered one.
    let region = region(IVec3::new(2, 1, 1));
    let along = vertex_position(&region, [0, 0, 0])
        .unwrap()
        .lerp(vertex_position(&region, [1, 0, 0]).unwrap(), 0.5);
    let origin = along + Vec3::new(0.0, 0.0, 2.0);
    let offer = edge_insertion(&region, origin, Vec3::new(0.0, 0.0, -1.0))
        .expect("the midpoint of the long edge is a grid position");
    assert_eq!(offer.axis, 0);
    assert_eq!(offer.position, 1);

    assert!(
        edge_insertion(&region, Vec3::splat(9.0), Vec3::Y).is_none(),
        "a pointer nowhere near an edge is offered nothing"
    );
}

#[test]
fn an_already_subdivided_edge_offers_nothing_more() {
    let mut region = region(IVec3::new(2, 1, 1));
    region.subdivide(0, 1).unwrap();
    let along = vertex_position(&region, [0, 0, 0])
        .unwrap()
        .lerp(vertex_position(&region, [2, 0, 0]).unwrap(), 0.5);
    let origin = along + Vec3::new(0.0, 0.0, 2.0);
    assert!(edge_insertion(&region, origin, Vec3::new(0.0, 0.0, -1.0)).is_none());
}

#[test]
fn screen_directions_resolve_to_the_nearest_grid_axis() {
    assert_eq!(screen_axis(Vec3::new(0.9, 0.3, 0.1)), (0, 1));
    assert_eq!(screen_axis(Vec3::new(-0.9, 0.3, 0.1)), (0, -1));
    assert_eq!(screen_axis(Vec3::new(0.1, -0.8, 0.3)), (1, -1));
    assert_eq!(screen_axis(Vec3::new(0.2, 0.1, 0.95)), (2, 1));
}

#[test]
fn feature_hover_reports_logical_edges_not_internal_tessellation_seams() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1, 1, 1], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let solid = graph.evaluated_solid(owner).unwrap();
    let logical = &solid.logical_edges[0];
    let edge = solid.half_edges[logical.half_edges[0] as usize];
    let next = solid.half_edges[edge.next as usize];
    let start = solid.vertices[edge.origin as usize].position;
    let end = solid.vertices[next.origin as usize].position;
    let midpoint = (start + end) * 0.5;
    let view = (end - start).normalize().any_orthonormal_vector();
    let hit = hovered_feature_edge(&solid, owner, midpoint + view, -view)
        .expect("aiming through a logical edge selects it");
    assert_eq!(hit.target.owner, owner);
    assert!(solid.logical_edge(hit.target.edge).is_some());
}

#[test]
fn concave_logical_edges_are_not_feature_targets() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let mut solid = graph.evaluated_solid(owner).unwrap();
    let logical = &mut solid.logical_edges[0];
    logical.convex = false;
    let target = EdgeChainRef {
        owner,
        edge: logical.key,
    };
    let edge = solid.half_edges[logical.half_edges[0] as usize];
    let next = solid.half_edges[edge.next as usize];
    let midpoint = (solid.vertices[edge.origin as usize].position
        + solid.vertices[next.origin as usize].position)
        * 0.5;

    assert!(hovered_source_edge(&solid, target, midpoint + Vec3::Z, Vec3::NEG_Z).is_none());
}

#[test]
fn silhouette_ray_outside_the_surface_bounds_selects_the_whole_cylinder_rim() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::default(),
            BuildPose::default(),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    let owner = SolidOwner::Part(part);
    let solid = graph.evaluated_solid(owner).unwrap();
    let ray_origin = Vec3::new(0.155, 1.0, 0.0);
    let ray_direction = Vec3::NEG_Y;

    assert!(ray_origin.x > 0.125, "the ray misses the cylinder surface");
    assert!(
        inflated_bounds_ray_distance(&solid, ray_origin, ray_direction).is_some(),
        "the edge-pick-inflated bounds acquire the silhouette"
    );
    let hit = hovered_feature_edge(&solid, owner, ray_origin, ray_direction)
        .expect("the nearby silhouette rim is selectable");
    let logical = solid.logical_edge(hit.target.edge).unwrap();
    assert!(logical.closed);
    assert_eq!(logical.half_edges.len(), 24);
}
