//! Recursive walks over the edit octree.

use super::brick::TerrainBrick;
use super::node::{TerrainDensityClass, TerrainNode, TerrainNodeId, TerrainNodeSummary};
use crate::{BRICK_EDGE_CELLS, BrickCoord, TERRAIN_CELL_METERS, TerrainField, WorldCell};
use bevy_math::{DVec3, IVec3};
use std::sync::Arc;

pub(super) fn find_brick(node: &TerrainNode, coordinate: BrickCoord) -> Option<&TerrainBrick> {
    if node.id.level == 0 {
        return (node.id.coordinates == coordinate)
            .then_some(node.brick.as_deref())
            .flatten();
    }
    let index = node.id.child_index_containing(coordinate);
    node.children[index]
        .as_deref()
        .and_then(|child| find_brick(child, coordinate))
}

pub(super) fn find_node(node: &TerrainNode, id: TerrainNodeId) -> Option<&TerrainNode> {
    if node.id == id {
        return Some(node);
    }
    if id.level >= node.id.level {
        return None;
    }
    let index = node.id.child_index_containing(id.coordinates);
    node.children[index]
        .as_deref()
        .and_then(|child| find_node(child, id))
}

pub(super) fn insert_brick_node(node: &mut Arc<TerrainNode>, brick: Arc<TerrainBrick>) {
    let node = Arc::make_mut(node);
    if node.id.level == 0 {
        debug_assert_eq!(node.id.coordinates, brick.coordinate);
        node.brick = Some(brick);
        node.refresh();
        return;
    }
    let index = node.id.child_index_containing(brick.coordinate);
    let child_id = node.id.children().expect("a branch has children")[index];
    let child = node.children[index].get_or_insert_with(|| Arc::new(TerrainNode::empty(child_id)));
    insert_brick_node(child, brick);
    node.refresh();
}

pub(super) fn collect_bricks(node: &TerrainNode) -> Vec<&TerrainBrick> {
    fn visit<'a>(node: &'a TerrainNode, bricks: &mut Vec<&'a TerrainBrick>) {
        if let Some(brick) = &node.brick {
            bricks.push(brick);
            return;
        }
        for child in node.children.iter().flatten() {
            visit(child, bricks);
        }
    }
    let mut bricks = Vec::with_capacity(usize::try_from(node.promoted_descendants).unwrap_or(0));
    visit(node, &mut bricks);
    bricks.sort_by_key(|brick| brick.coordinate);
    bricks
}

pub(super) fn collect_bricks_between<'a>(
    node: &'a TerrainNode,
    minimum: BrickCoord,
    maximum: BrickCoord,
    bricks: &mut Vec<&'a TerrainBrick>,
) {
    if node.promoted_descendants == 0 || !node_intersects(node.id, minimum, maximum) {
        return;
    }
    if let Some(brick) = &node.brick {
        bricks.push(brick);
        return;
    }
    for child in node.children.iter().flatten() {
        collect_bricks_between(child, minimum, maximum, bricks);
    }
}

pub(super) fn node_intersects(id: TerrainNodeId, minimum: BrickCoord, maximum: BrickCoord) -> bool {
    let edge = id.edge_bricks();
    let maximum_node = [id.coordinates.x, id.coordinates.y, id.coordinates.z]
        .map(|coordinate| i64::from(coordinate) + edge - 1);
    maximum_node[0] >= i64::from(minimum.x)
        && i64::from(id.coordinates.x) <= i64::from(maximum.x)
        && maximum_node[1] >= i64::from(minimum.y)
        && i64::from(id.coordinates.y) <= i64::from(maximum.y)
        && maximum_node[2] >= i64::from(minimum.z)
        && i64::from(id.coordinates.z) <= i64::from(maximum.z)
}

pub(super) fn minimum_promoted_density_between(
    node: &TerrainNode,
    minimum: WorldCell,
    maximum: WorldCell,
) -> Option<f32> {
    if node.promoted_descendants == 0 {
        return None;
    }
    let node_minimum = node.id.minimum_cell_i64();
    let node_maximum = node.id.maximum_cell_exclusive_i64();
    let query_minimum = [minimum.x, minimum.y, minimum.z].map(i64::from);
    let query_maximum = [maximum.x, maximum.y, maximum.z].map(i64::from);
    if (0..3).any(|axis| {
        node_maximum[axis] <= query_minimum[axis] || node_minimum[axis] > query_maximum[axis]
    }) {
        return None;
    }
    if (0..3).all(|axis| {
        query_minimum[axis] <= node_minimum[axis] && query_maximum[axis] >= node_maximum[axis] - 1
    }) {
        return Some(node.minimum_density);
    }
    if let Some(brick) = &node.brick {
        let brick_minimum = brick.coordinate.minimum_cell();
        let first = IVec3::new(
            minimum.x.max(brick_minimum.x) - brick_minimum.x,
            minimum.y.max(brick_minimum.y) - brick_minimum.y,
            minimum.z.max(brick_minimum.z) - brick_minimum.z,
        );
        let last = IVec3::new(
            maximum.x.min(brick_minimum.x + BRICK_EDGE_CELLS - 1) - brick_minimum.x,
            maximum.y.min(brick_minimum.y + BRICK_EDGE_CELLS - 1) - brick_minimum.y,
            maximum.z.min(brick_minimum.z + BRICK_EDGE_CELLS - 1) - brick_minimum.z,
        );
        let mut result = f32::INFINITY;
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    result = result.min(
                        brick
                            .sample(IVec3::new(x, y, z))
                            .expect("clamped promoted coordinate is inside its brick")
                            .density,
                    );
                }
            }
        }
        return result.is_finite().then_some(result);
    }
    node.children
        .iter()
        .flatten()
        .filter_map(|child| minimum_promoted_density_between(child, minimum, maximum))
        .reduce(f32::min)
}

pub(super) fn collect_nodes_between(
    node: &TerrainNode,
    minimum: BrickCoord,
    maximum: BrickCoord,
    nodes: &mut Vec<TerrainNodeSummary>,
) {
    if !node_intersects(node.id, minimum, maximum) {
        return;
    }
    nodes.push(node.summary());
    for child in node.children.iter().flatten() {
        collect_nodes_between(child, minimum, maximum, nodes);
    }
}

pub(super) fn latest_revision_between(
    node: &TerrainNode,
    minimum: BrickCoord,
    maximum: BrickCoord,
) -> u64 {
    if node.promoted_descendants == 0 || !node_intersects(node.id, minimum, maximum) {
        return 0;
    }
    let node_minimum = node.id.coordinates;
    let node_maximum = node.id.edge_bricks() - 1;
    let node_maximum = BrickCoord::new(
        i32::try_from(i64::from(node_minimum.x) + node_maximum)
            .expect("octree node maximum fits brick coordinates"),
        i32::try_from(i64::from(node_minimum.y) + node_maximum)
            .expect("octree node maximum fits brick coordinates"),
        i32::try_from(i64::from(node_minimum.z) + node_maximum)
            .expect("octree node maximum fits brick coordinates"),
    );
    let fully_covered = minimum.x <= node_minimum.x
        && minimum.y <= node_minimum.y
        && minimum.z <= node_minimum.z
        && maximum.x >= node_maximum.x
        && maximum.y >= node_maximum.y
        && maximum.z >= node_maximum.z;
    if fully_covered || node.brick.is_some() {
        return node.latest_revision;
    }
    node.children
        .iter()
        .flatten()
        .map(|child| latest_revision_between(child, minimum, maximum))
        .max()
        .unwrap_or(0)
}

pub(super) fn classify_node(
    root: &TerrainNode,
    field: &TerrainField,
    id: TerrainNodeId,
) -> TerrainDensityClass {
    if let Some(node) = find_node(root, id)
        && id
            .edge_bricks()
            .checked_pow(3)
            .and_then(|count| u64::try_from(count).ok())
            .is_some_and(|count| node.promoted_descendants == count)
    {
        return if node.maximum_density <= 0.0 {
            TerrainDensityClass::Empty
        } else if node.minimum_density > 0.0 {
            TerrainDensityClass::Solid
        } else {
            TerrainDensityClass::Mixed
        };
    }

    let edge_cells = id.edge_bricks() * i64::from(BRICK_EDGE_CELLS);
    let minimum = [id.coordinates.x, id.coordinates.y, id.coordinates.z]
        .map(|coordinate| i64::from(coordinate) * i64::from(BRICK_EDGE_CELLS));
    let lower = DVec3::from_array(minimum.map(|cell| cell as f64 * TERRAIN_CELL_METERS));
    let upper = lower + DVec3::splat(edge_cells as f64 * TERRAIN_CELL_METERS);
    if find_node(root, id).is_some_and(|node| node.promoted_descendants != 0) {
        TerrainDensityClass::Mixed
    } else {
        field.classify(lower, upper)
    }
}
